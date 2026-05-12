# RF-DETR + DVIS Integration — Development Log

## Objective

Integrate the RF-DETR model into the DVIS framework as the per-frame segmenter, replacing the Mask2Former-based segmenter. The DVIS temporal modules (ReferringTracker, TemporalRefiner) are retrained on top of the frozen RF-DETR segmenter.

### Key Decisions
- Use RF-DETR's DINOv2 backbone + multi-scale projector for features
- Use RF-DETR's deformable transformer decoder and segmentation head
- Discard RF-DETR bounding box prediction losses
- Retain DVIS's Dice + CE + mask BCE losses for segmentation
- Segmenter is frozen during training; only the tracker/refiner are trained
- Changed `enc_out_class_embed` and `class_embed` inside the segmenter to use 91 classes (COCO) so pretrained weights load correctly

---

## Files Created

### `dvis/rfdetr_segmenter.py`
- `RFDETRSegmenter` class wrapping the full RF-DETR pipeline
- Components: DINOv2 backbone (via `build_backbone` Joiner), deformable transformer decoder, segmentation head, DVIS-compatible class/mask embed heads
- Output format matches DVIS interface: `pred_logits (B,T,Q,C)`, `pred_masks (B,Q,T,H,W)`, `pred_embds (B,C,T,Q)`, `mask_features (BT,C,H,W)`, `aux_outputs`
- Loads RF-DETR pretrained weights via `_load_pretrain_weights()` with shape-mismatch handling
- `enc_out_class_embed` and `class_embed` use 91 classes (COCO) to match pretrained checkpoint; the tracker re-predicts with its own 41-class head

### `configs/youtubevis_2022/rfdetr/DVIS_Online_RFDETR.yaml`
- Config for DVIS Online + RF-DETR Large on YouTubeVIS 2022
- RF-DETR Large params: `dinov2_windowed_small` encoder, `patch_size=16`, `resolution=704`, `num_queries=300`, `dec_layers=4`, `hidden_dim=256`
- `SIZE_DIVISIBILITY=32` (patch_size=16 × num_windows=2 = 32)
- `PRETRAIN_WEIGHTS` points to the RF-DETR Large checkpoint
- `MODEL.WEIGHTS: ""` since detectron2's `DetectionCheckpointer` is not used for RF-DETR weights

---

## Files Modified

### `dvis/meta_architecture.py`
- Added `from .rfdetr_segmenter import RFDETRSegmenter` import
- Added `DVIS_online_rfdetr` meta-architecture class registered with `META_ARCH_REGISTRY`
  - Uses frozen `RFDETRSegmenter` instead of backbone + sem_seg_head
  - Only trains `["labels", "masks"]` losses via `VideoSetCriterion`
  - Includes `run_window_inference`, `prepare_targets`, `frame_decoder_loss_reshape`, inference methods
- Fixed `run_window_inference` bug: extract `mask_features` before deleting segmenter outputs
- Replaced deprecated `torch.range` with `torch.arange` (2 occurrences)

### `dvis/config.py`
- Added `add_rfdetr_config(cfg)` function with all RF-DETR hyperparameters

### `dvis/__init__.py`
- Exported `add_rfdetr_config` and `DVIS_online_rfdetr`

### `train_net_video.py`
- Imported and called `add_rfdetr_config(cfg)` in `setup()`
- Moved `FutureWarning` filters into the try block

### `mask2former_video/modeling/matcher.py`
- Replaced deprecated `from torch.cuda.amp import autocast` with `from torch.amp import autocast`
- Updated `autocast(enabled=False)` to `autocast('cuda', enabled=False)` (2 occurrences)

---

## Issues Encountered & Fixes

### 1. Input size not divisible by backbone block size
- **Error**: `Backbone requires input shape to be divisible by 24, but got torch.Size([40, 3, 448, 768])`
- **Cause**: `SIZE_DIVISIBILITY` was 32 (Mask2Former default), but DINOv2 with `patch_size=12, num_windows=2` needs divisibility by 24
- **Fix**: Set `SIZE_DIVISIBILITY` to match `patch_size × num_windows`. For RF-DETR Large (patch_size=16, num_windows=2) this is 32.

### 2. No checkpoint loaded
- **Error**: `No checkpoint found. Initializing model from scratch`
- **Cause**: `DetectionCheckpointer` only reads `MODEL.WEIGHTS`, not `MODEL.RFDETR.PRETRAIN_WEIGHTS`
- **Fix**: Added `_load_pretrain_weights()` method to `RFDETRSegmenter.__init__` that manually loads the checkpoint with key matching and shape-mismatch handling

### 3. Class embed shape mismatch (91 vs 41)
- **Error**: `Shape mismatch for transformer.enc_out_class_embed.0.weight: ckpt torch.Size([91, 256]) vs model torch.Size([41, 256])`
- **Cause**: Pretrained RF-DETR uses 91 COCO classes, our model had 41 YTVIS classes
- **Fix**: Changed `enc_out_class_embed` and `class_embed` in the segmenter to use 91 classes. These heads are inside the frozen segmenter; the tracker has its own 41-class head for final predictions.

### 4. Deprecated warnings
- `torch.range` → `torch.arange` in `meta_architecture.py`
- `torch.cuda.amp.autocast` → `torch.amp.autocast('cuda')` in `matcher.py`
- Added `FutureWarning` filter for `detectron2` in `train_net_video.py`

---

## Architecture Overview

```
DVIS_online_rfdetr
├── segmenter (RFDETRSegmenter) [FROZEN]
│   ├── backbone (Joiner: DINOv2 + projector + position encoding)
│   ├── transformer (deformable decoder, two-stage)
│   ├── segmentation_head (RF-DETR SegmentationHead)
│   ├── class_embed (91 classes, COCO pretrained)
│   ├── mask_embed (MLP)
│   └── decoder_norm (LayerNorm)
├── tracker (ReferringTracker) [TRAINED]
│   ├── self_attention_layers
│   ├── cross_attention_layers
│   ├── ffn_layers
│   ├── class_embed (41 classes, YTVIS)
│   └── mask_embed
└── criterion (VideoSetCriterion)
    ├── losses: ['labels', 'masks']
    └── matcher: VideoHungarianMatcher_Consistent
```

---

## Run Command

```bash
python train_net_video.py \
  --num-gpus 4 \
  --config-file configs/youtubevis_2022/rfdetr/DVIS_Online_RFDETR.yaml \
  SOLVER.IMS_PER_BATCH 64 \
  OUTPUT_DIR /path/to/output
```

---

## RF-DETR Config Variants Reference

| Parameter | Large (detection) | SegLarge (segmentation) |
|---|---|---|
| encoder | dinov2_windowed_small | dinov2_windowed_small |
| patch_size | 16 | 12 |
| resolution | 704 | 504 |
| num_queries | 300 | 300 (checkpoint actual) |
| dec_layers | 4 | 5 |
| positional_encoding_size | 44 | 42 |
| hidden_dim | 256 | 256 |
| group_detr | 13 | 13 |

Currently using **SegLarge** config with `rf-detr-seg-large.pt` checkpoint.

---

## Changelog

### 2026-04-30 (afternoon)

**Switched from RF-DETR Large (detection) to SegLarge (segmentation) checkpoint**

- **Problem**: RF-DETR Large is a detection-only model with no trained segmentation head. The frozen segmenter produced random masks, causing loss to plateau at ~20 (vs Swin-L dropping to ~10 by iter 600).
- **Fix**: Switched to `rf-detr-seg-large.pt` which includes a trained `SegmentationHead` (41 keys).

**Fixed config mismatch with SegLarge checkpoint**

- The SegLarge checkpoint has `patch_size=12`, `positional_encoding_size=42`, `resolution=504`, `dec_layers=5`, but the config still had Large detection params (`patch_size=16`, etc.). This caused backbone weight shape mismatches (`position_embeddings`, `patch_embeddings.projection`).
- Also discovered the checkpoint uses **300 queries** (not 200 as `RFDETRSegLargeConfig` reports): `refpoint_embed` shape is `(3900, 4) = 300 × 13`.
- Updated `SIZE_DIVISIBILITY` from 32 to 24 (`patch_size=12 × num_windows=2`).
- Updated `MIN_SIZE_TRAIN` to `(360, 480, 504)` and `MAX_SIZE_TRAIN` to 768.

**Fixed class embed dimensions for pretrained weight loading**

- `enc_out_class_embed` and `class_embed` in `RFDETRSegmenter` now use 91 classes (COCO) instead of `num_classes+1` (41), so pretrained weights load without shape mismatch. These are inside the frozen segmenter; the tracker has its own 41-class head.

**Added pretrained weight loading to `RFDETRSegmenter`**

- Added `_load_pretrain_weights()` method that loads checkpoint, matches keys, skips shape mismatches, and logs statistics.
- `MODEL.RFDETR.PRETRAIN_WEIGHTS` config entry specifies the path; loaded during `__init__`, not via `DetectionCheckpointer`.

**Warning fixes**

- `torch.range` → `torch.arange` in `meta_architecture.py` (2 occurrences)
- `torch.cuda.amp.autocast` → `torch.amp.autocast('cuda')` in `matcher.py`
- Added `FutureWarning` filters in `train_net_video.py`

### Training runs

| Run | Checkpoint | Config | Status |
|---|---|---|---|
| `ytvis2022_rfdetr_l_0429` | `rf-detr-large-2026.pth` (detection) | Large (patch_size=16) | Loss plateau ~20, slow convergence |
| `ytvis2022_rfdetr_sl_0430` (1st attempt) | `rf-detr-seg-large.pt` | Large params (wrong!) | Backbone weights failed to load (shape mismatch) |
| `ytvis2022_rfdetr_sl_0430` (2nd attempt) | `rf-detr-seg-large.pt` | SegLarge (patch_size=12, 300 queries) | Running, weights loaded correctly |
