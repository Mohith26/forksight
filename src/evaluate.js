'use strict';

// Detection scoring. One pallet per frame at most, so this stays simple:
// intersection over union against the rendered ground truth, with a fixed
// acceptance threshold.

function iou(a, b) {
  if (!a || !b) return 0;
  const x0 = Math.max(a.x, b.x);
  const y0 = Math.max(a.y, b.y);
  const x1 = Math.min(a.x + a.w, b.x + b.w);
  const y1 = Math.min(a.y + a.h, b.y + b.h);
  const iw = x1 - x0;
  const ih = y1 - y0;
  if (iw <= 0 || ih <= 0) return 0;
  const inter = iw * ih;
  return inter / (a.w * a.h + b.w * b.h - inter);
}

function score(results, threshold) {
  const thr = threshold === undefined ? 0.5 : threshold;
  let tp = 0, fp = 0, fn = 0, tn = 0;
  const ious = [];
  for (const r of results) {
    const hasTruth = !!r.truth;
    const hasPred = !!r.pred;
    if (hasTruth && hasPred) {
      const v = iou(r.pred, r.truth);
      ious.push(v);
      if (v >= thr) tp += 1; else { fp += 1; fn += 1; }
    } else if (hasTruth && !hasPred) {
      fn += 1;
    } else if (!hasTruth && hasPred) {
      fp += 1;
    } else {
      tn += 1;
    }
  }
  const precision = tp + fp === 0 ? 1 : tp / (tp + fp);
  const recall = tp + fn === 0 ? 1 : tp / (tp + fn);
  const f1 = precision + recall === 0 ? 0 : (2 * precision * recall) / (precision + recall);
  return {
    frames: results.length,
    tp: tp, fp: fp, fn: fn, tn: tn,
    precision: +precision.toFixed(4),
    recall: +recall.toFixed(4),
    f1: +f1.toFixed(4),
    meanIoU: ious.length ? +(ious.reduce((a, b) => a + b, 0) / ious.length).toFixed(4) : 0,
    minIoU: ious.length ? +Math.min.apply(null, ious).toFixed(4) : 0,
  };
}

module.exports = { iou, score };
