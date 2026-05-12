"""
RF-DETR segmenter wrapper for DVIS.

This module wraps the RF-DETR model (DINOv2 backbone + projector + deformable
decoder + segmentation head) to produce outputs compatible with the DVIS
temporal modules (ReferringTracker, TemporalRefiner).

Expected output dict keys (matching Mask2Former / DVIS interface):
    - pred_logits: (B, T, Q, C)  classification logits
    - pred_masks:  (B, Q, T, H, W)  mask predictions
    - pred_embds:  (B, C, T, Q)  query embeddings for tracker
    - mask_features: (B*T, C, H, W)  spatial features for tracker mask prediction
    - aux_outputs: list of dicts with pred_logits and pred_masks per decoder layer
"""
import copy
import math
import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import einops

from detectron2.config import configurable

from rfdetr.models.backbone import build_backbone as rfdetr_build_backbone
from rfdetr.models.transformer import Transformer as RFDETRTransformer
from rfdetr.models.heads.segmentation import SegmentationHead
from rfdetr.models.math import MLP
from rfdetr.utilities.tensors import NestedTensor, nested_tensor_from_tensor_list

logger = logging.getLogger(__name__)


class RFDETRSegmenter(nn.Module):
    """
    RF-DETR model adapted for DVIS video instance segmentation.

    Wraps:
      - DINOv2 backbone + multi-scale projector + position encoding (Joiner)
      - Deformable transformer decoder (no encoder)
      - Segmentation head for mask prediction
      - Class embed and mask embed heads

    Produces outputs in the same format as the DVIS video decoder so that
    ReferringTracker / TemporalRefiner can consume them directly.
    """

    @configurable
    def __init__(
        self,
        *,
        # pretrained weights
        pretrain_weights: str = "",
        # backbone
        encoder_name: str,
        out_feature_indexes: List[int],
        projector_scale: List[str],
        patch_size: int,
        num_windows: int,
        positional_encoding_size: int,
        resolution: int,
        freeze_encoder: bool,
        # decoder
        hidden_dim: int,
        num_queries: int,
        dec_layers: int,
        sa_nheads: int,
        ca_nheads: int,
        dec_n_points: int,
        dim_feedforward: int,
        two_stage: bool,
        group_detr: int,
        bbox_reparam: bool,
        lite_refpoint_refine: bool,
        # segmentation
        mask_downsample_ratio: int,
        # classification
        num_classes: int,
        # video
        num_frames: int,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_queries = num_queries
        self.num_classes = num_classes
        self.num_frames = num_frames
        self.dec_layers = dec_layers
        self.two_stage = two_stage
        self.group_detr = group_detr
        self.bbox_reparam = bbox_reparam
        self.lite_refpoint_refine = lite_refpoint_refine
        self.resolution = resolution

        # --- Build backbone (DINOv2 + projector + position encoding) ---
        # Returns a Joiner(backbone, position_embedding).
        # Joiner.forward(nested_tensor) -> (features_list[NestedTensor], pos_list[(B,C,H,W)])
        self.backbone = rfdetr_build_backbone(
            encoder=encoder_name,
            vit_encoder_num_layers=out_feature_indexes[-1] + 1,
            pretrained_encoder=None,
            window_block_indexes=None,
            drop_path=0.0,
            out_channels=hidden_dim,
            out_feature_indexes=out_feature_indexes,
            projector_scale=projector_scale,
            use_cls_token=False,
            hidden_dim=hidden_dim,
            position_embedding="sine",
            freeze_encoder=freeze_encoder,
            layer_norm=True,
            target_shape=(resolution, resolution),
            rms_norm=False,
            backbone_lora=False,
            force_no_pretrain=False,
            gradient_checkpointing=False,
            load_dinov2_weights=True,
            patch_size=patch_size,
            num_windows=num_windows,
            positional_encoding_size=positional_encoding_size,
        )

        # --- Build transformer decoder ---
        num_feature_levels = len(projector_scale)
        self.transformer = RFDETRTransformer(
            d_model=hidden_dim,
            sa_nhead=sa_nheads,
            ca_nhead=ca_nheads,
            num_queries=num_queries,
            num_decoder_layers=dec_layers,
            dim_feedforward=dim_feedforward,
            dropout=0.0,
            activation="relu",
            normalize_before=False,
            return_intermediate_dec=True,
            group_detr=group_detr,
            two_stage=two_stage,
            num_feature_levels=num_feature_levels,
            dec_n_points=dec_n_points,
            lite_refpoint_refine=lite_refpoint_refine,
            bbox_reparam=bbox_reparam,
        )

        # --- Query embeddings ---
        query_dim = 4
        self.refpoint_embed = nn.Embedding(num_queries * group_detr, query_dim)
        self.query_feat = nn.Embedding(num_queries * group_detr, hidden_dim)
        nn.init.constant_(self.refpoint_embed.weight.data, 0)

        # --- Bbox embed (needed for iterative refinement in decoder) ---
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        if not lite_refpoint_refine:
            self.transformer.decoder.bbox_embed = self.bbox_embed
        else:
            self.transformer.decoder.bbox_embed = None

        # --- Two-stage heads ---
        # enc_out_class_embed is only used for internal proposal ranking (scoring
        # encoder outputs to select top-k decoder queries). It does NOT need to
        # match the downstream task's num_classes. We use 91 (COCO) so the
        # pretrained weights load correctly and the segmenter can be frozen.
        enc_out_num_classes = 91
        if two_stage:
            self.transformer.enc_out_bbox_embed = nn.ModuleList(
                [copy.deepcopy(self.bbox_embed) for _ in range(group_detr)]
            )
            enc_class_embed = nn.Linear(hidden_dim, enc_out_num_classes)
            prior_prob = 0.01
            bias_value = -math.log((1 - prior_prob) / prior_prob)
            enc_class_embed.bias.data = torch.ones(enc_out_num_classes) * bias_value
            self.transformer.enc_out_class_embed = nn.ModuleList(
                [copy.deepcopy(enc_class_embed) for _ in range(group_detr)]
            )

        # --- Segmentation head ---
        self.segmentation_head = SegmentationHead(
            hidden_dim, dec_layers, downsample_ratio=mask_downsample_ratio
        )

        # --- DVIS-compatible heads ---
        # Class prediction: use the same 91 classes as the pretrained checkpoint
        # so weights load correctly. The segmenter is frozen and the tracker
        # re-predicts class logits with its own head (matching num_classes).
        self.class_embed = nn.Linear(hidden_dim, enc_out_num_classes)
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(enc_out_num_classes) * bias_value

        # Mask embed: project query to mask_dim for dot-product with mask_features
        # Used by the ReferringTracker for re-generating masks
        self.mask_embed = MLP(hidden_dim, hidden_dim, hidden_dim, 3)

        # Decoder norm
        self.decoder_norm = nn.LayerNorm(hidden_dim)

        # --- Load pretrained RF-DETR weights ---
        if pretrain_weights:
            self._load_pretrain_weights(pretrain_weights)

    def _load_pretrain_weights(self, weights_path):
        """Load RF-DETR pretrained weights with key remapping."""
        logger.info(f"Loading RF-DETR pretrained weights from {weights_path}")
        ckpt = torch.load(weights_path, map_location="cpu")
        state_dict = ckpt["model"] if "model" in ckpt else ckpt

        # Remap checkpoint keys: the checkpoint uses top-level keys like
        # backbone.0.xxx, transformer.xxx, bbox_embed.xxx, etc.
        # Our model wraps them under self.backbone, self.transformer, etc.
        # which already matches, so we can load directly with prefix mapping.
        new_state_dict = {}
        model_state = self.state_dict()
        loaded, skipped = 0, 0
        for k, v in state_dict.items():
            # Skip keys not in our model (e.g. segmentation_head from a seg checkpoint)
            if k not in model_state:
                skipped += 1
                continue
            # Skip shape mismatches (e.g. class_embed with different num_classes)
            if v.shape != model_state[k].shape:
                logger.warning(
                    f"Shape mismatch for {k}: ckpt {v.shape} vs model {model_state[k].shape}, skipping"
                )
                skipped += 1
                continue
            new_state_dict[k] = v
            loaded += 1

        missing, unexpected = self.load_state_dict(new_state_dict, strict=False)
        logger.info(
            f"RF-DETR pretrain: loaded {loaded}, skipped {skipped}, "
            f"missing {len(missing)}, unexpected {len(unexpected)}"
        )

    @classmethod
    def from_config(cls, cfg):
        return {
            "pretrain_weights": cfg.MODEL.RFDETR.PRETRAIN_WEIGHTS,
            "encoder_name": cfg.MODEL.RFDETR.ENCODER,
            "out_feature_indexes": cfg.MODEL.RFDETR.OUT_FEATURE_INDEXES,
            "projector_scale": cfg.MODEL.RFDETR.PROJECTOR_SCALE,
            "patch_size": cfg.MODEL.RFDETR.PATCH_SIZE,
            "num_windows": cfg.MODEL.RFDETR.NUM_WINDOWS,
            "positional_encoding_size": cfg.MODEL.RFDETR.POSITIONAL_ENCODING_SIZE,
            "resolution": cfg.MODEL.RFDETR.RESOLUTION,
            "freeze_encoder": cfg.MODEL.RFDETR.FREEZE_ENCODER,
            "hidden_dim": cfg.MODEL.RFDETR.HIDDEN_DIM,
            "num_queries": cfg.MODEL.RFDETR.NUM_QUERIES,
            "dec_layers": cfg.MODEL.RFDETR.DEC_LAYERS,
            "sa_nheads": cfg.MODEL.RFDETR.SA_NHEADS,
            "ca_nheads": cfg.MODEL.RFDETR.CA_NHEADS,
            "dec_n_points": cfg.MODEL.RFDETR.DEC_N_POINTS,
            "dim_feedforward": cfg.MODEL.RFDETR.DIM_FEEDFORWARD,
            "two_stage": cfg.MODEL.RFDETR.TWO_STAGE,
            "group_detr": cfg.MODEL.RFDETR.GROUP_DETR,
            "bbox_reparam": cfg.MODEL.RFDETR.BBOX_REPARAM,
            "lite_refpoint_refine": cfg.MODEL.RFDETR.LITE_REFPOINT_REFINE,
            "mask_downsample_ratio": cfg.MODEL.RFDETR.MASK_DOWNSAMPLE_RATIO,
            "num_classes": cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
            "num_frames": cfg.INPUT.SAMPLING_FRAME_NUM,
        }

    def forward(self, images_tensor):
        """
        Args:
            images_tensor: (B*T, C, H, W) normalized image tensor

        Returns:
            dict with keys matching DVIS video decoder interface:
                pred_logits: (B, T, Q, num_classes+1)
                pred_masks: (B, Q, T, H, W)
                pred_embds: (B, C, T, Q)
                mask_features: (B*T, C, H, W)  spatial features
                aux_outputs: list of dicts
        """
        bt = images_tensor.shape[0]

        # --- Backbone forward (Joiner: backbone + position encoding) ---
        nested = nested_tensor_from_tensor_list(
            [images_tensor[i] for i in range(bt)]
        )
        # features: list of NestedTensor (one per scale level)
        # poss: list of (B, C, H, W) position encodings
        features, poss = self.backbone(nested)

        srcs = []
        masks = []
        for feat in features:
            src, mask = feat.decompose()
            srcs.append(src)
            masks.append(mask)

        # The first (highest-res) feature map serves as spatial features
        # for the segmentation head and for the tracker's mask generation
        mask_features = srcs[0]  # (BT, C, H, W)

        # --- Transformer decoder forward ---
        if self.training:
            refpoint_embed_weight = self.refpoint_embed.weight
            query_feat_weight = self.query_feat.weight
        else:
            refpoint_embed_weight = self.refpoint_embed.weight[:self.num_queries]
            query_feat_weight = self.query_feat.weight[:self.num_queries]

        # hs: (num_dec_layers, BT, Q, C)
        # ref_unsigmoid: (num_dec_layers, BT, Q, 4)
        hs, ref_unsigmoid, hs_enc, ref_enc = self.transformer(
            srcs, masks, poss, refpoint_embed_weight, query_feat_weight
        )

        # --- Generate masks using segmentation head ---
        # query_features_list: list of (BT, Q, C) for each decoder layer
        if hs is not None:
            query_features_list = [hs[i] for i in range(hs.shape[0])]
        else:
            query_features_list = [hs_enc]

        # mask_logits_list: list of (BT, Q, H_mask, W_mask)
        mask_logits_list = self.segmentation_head(
            features[0].tensors, query_features_list, images_tensor.shape[-2:]
        )

        # --- Classification via DVIS-compatible class head ---
        if hs is not None:
            # hs: (num_layers, BT, Q, C)
            predictions_class = []
            predictions_mask = []
            for layer_idx in range(hs.shape[0]):
                layer_output = self.decoder_norm(hs[layer_idx])  # (BT, Q, C)
                cls_logits = self.class_embed(layer_output)  # (BT, Q, num_cls+1)
                predictions_class.append(cls_logits)
                if layer_idx < len(mask_logits_list):
                    predictions_mask.append(mask_logits_list[layer_idx])
                else:
                    predictions_mask.append(mask_logits_list[-1])

            # Query embeddings from last decoder layer
            pred_embds_flat = self.decoder_norm(hs[-1])  # (BT, Q, C)
        else:
            layer_output = self.decoder_norm(hs_enc)
            predictions_class = [self.class_embed(layer_output)]
            predictions_mask = [mask_logits_list[0]]
            pred_embds_flat = layer_output

        # --- Reshape to video format (B, T, ...) ---
        bs = bt // self.num_frames if self.training else 1
        t = bt // bs

        # pred_logits: (BT, Q, C) -> (B, T, Q, C)
        for i in range(len(predictions_class)):
            predictions_class[i] = einops.rearrange(
                predictions_class[i], '(b t) q c -> b t q c', t=t
            )

        # pred_masks: (BT, Q, H, W) -> (B, Q, T, H, W)
        for i in range(len(predictions_mask)):
            predictions_mask[i] = einops.rearrange(
                predictions_mask[i], '(b t) q h w -> b q t h w', t=t
            )

        # pred_embds: (BT, Q, C) -> (B, C, T, Q)
        pred_embds = einops.rearrange(pred_embds_flat, '(b t) q c -> b c t q', t=t)

        out = {
            'pred_logits': predictions_class[-1],
            'pred_masks': predictions_mask[-1],
            'aux_outputs': self._set_aux_loss(predictions_class, predictions_mask),
            'pred_embds': pred_embds,
            'mask_features': mask_features,
        }

        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_seg_masks):
        return [
            {"pred_logits": a, "pred_masks": b}
            for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])
        ]

    @property
    def num_classes_with_bg(self):
        """For compatibility with DVIS criterion setup."""
        return self.num_classes
