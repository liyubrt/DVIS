#!/usr/bin/env python
"""
Evaluate DVIS predictions on Jupiter humans dataset.
Unlike YTVIS 2022 evaluate.py, this does not require gt_short/gt_long splits.

Usage:
    python evaluate_jupiter_humans.py <gt_dir> <output_dir>

Expects:
    <gt_dir>/gt.json        - ground truth annotations
    <gt_dir>/results.json   - prediction results (symlinked)
"""
import sys
import os
import numpy as np
from pycocotools.ytvos import YTVOS
from pycocotools.ytvoseval import YTVOSeval


def _summarize(self, ap=1, iouThr=None, areaRng='all', maxDets=100):
    p = self.params
    iStr = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'
    titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
    typeStr = '(AP)' if ap == 1 else '(AR)'
    iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
        if iouThr is None else '{:0.2f}'.format(iouThr)

    aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
    mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
    if ap == 1:
        s = self.eval['precision']
        if iouThr is not None:
            t = np.where(iouThr == p.iouThrs)[0]
            s = s[t]
        s = s[:, :, :, aind, mind]
    else:
        s = self.eval['recall']
        if iouThr is not None:
            t = np.where(iouThr == p.iouThrs)[0]
            s = s[t]
        s = s[:, :, aind, mind]
    if len(s[s > -1]) == 0:
        mean_s = -1
    else:
        mean_s = np.mean(s[s > -1])
    print(iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s))
    return mean_s


input_dir = sys.argv[1]
output_dir = sys.argv[2]

submit_file = os.path.join(input_dir, 'results.json')
truth_file = os.path.join(input_dir, 'gt.json')

if not os.path.isfile(submit_file):
    print("%s doesn't exist" % submit_file)
    sys.exit(1)

if not os.path.isfile(truth_file):
    print("%s doesn't exist" % truth_file)
    sys.exit(1)

os.makedirs(output_dir, exist_ok=True)

gts = YTVOS(truth_file)
res = gts.loadRes(submit_file)

ytvosEval = YTVOSeval(gts, res, 'segm')
ytvosEval.evaluate()
ytvosEval.accumulate()
ytvosEval.summarize()

output_filename = os.path.join(output_dir, 'scores.txt')
with open(output_filename, 'w') as f:
    f.write('mAP: {}\n'.format(_summarize(ytvosEval, 1)))
    f.write('AP50: {}\n'.format(
        _summarize(ytvosEval, 1, iouThr=.5, maxDets=ytvosEval.params.maxDets[2])))
    f.write('AP75: {}\n'.format(
        _summarize(ytvosEval, 1, iouThr=.75, maxDets=ytvosEval.params.maxDets[2])))
    f.write('AR1: {}\n'.format(
        _summarize(ytvosEval, 0, maxDets=ytvosEval.params.maxDets[0])))
    f.write('AR10: {}\n'.format(
        _summarize(ytvosEval, 0, maxDets=ytvosEval.params.maxDets[1])))

print("\nScores saved to:", output_filename)
