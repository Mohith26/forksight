'use strict';

const { test, assert, assertEqual, assertClose } = require('./harness');
const { renderScene } = require('../src/synth');
const { detect } = require('../src/detect');
const { iou, score } = require('../src/evaluate');

// Seeds 1000-1099 were used while tuning the thresholds. Everything reported
// below runs on 5000+, which the detector has never been fitted against.
const HELD_OUT_START = 5000;

function runRange(start, count, opts) {
  const results = [];
  for (let i = 0; i < count; i++) {
    const scene = renderScene(start + i, opts || {});
    const d = detect(scene.image);
    results.push({ pred: d.pallet, truth: scene.truth.pallet, det: d, scene: scene });
  }
  return results;
}

module.exports = function run(out) {
  test('the intersection over union helper behaves at the obvious cases', () => {
    const a = { x: 0, y: 0, w: 10, h: 10 };
    assertClose(iou(a, a), 1, 1e-12, 'identical boxes');
    assertEqual(iou(a, { x: 100, y: 100, w: 10, h: 10 }), 0, 'disjoint boxes');
    assertEqual(iou(a, null), 0, 'missing box');
    assertClose(iou(a, { x: 5, y: 0, w: 10, h: 10 }), 50 / 150, 1e-12, 'half overlap');
  });

  test('an empty floor produces no detection', () => {
    let falsePositives = 0;
    for (let i = 0; i < 40; i++) {
      const scene = renderScene(9000 + i, { withPallet: false });
      if (detect(scene.image).pallet) falsePositives += 1;
    }
    // One empty frame in this range does trip the pairing stage. Two dark
    // distractors can land at the right spacing by chance, and nothing in the
    // pipeline knows about the bright face between them yet.
    assert(falsePositives <= 2, 'at most 2 false positives on 40 empty scenes, got ' + falsePositives);
    if (out) out.emptyScenes = { frames: 40, falsePositives: falsePositives };
  });

  test('the detector finds the pallet on held out scenes', () => {
    const results = runRange(HELD_OUT_START, 200);
    const s = score(results, 0.5);
    assert(s.f1 >= 0.90, 'f1 at least 0.90 on held out data, got ' + s.f1);
    assert(s.precision >= 0.90, 'precision at least 0.90, got ' + s.precision);
    assert(s.meanIoU >= 0.85, 'mean iou at least 0.85, got ' + s.meanIoU);
    if (out) out.heldOut = s;
  });

  test('both fork pockets are localised, not just the outer box', () => {
    const results = runRange(HELD_OUT_START, 60).filter(r => r.truth && r.pred);
    let matched = 0, total = 0;
    for (const r of results) {
      const gt = r.scene.truth.pockets;
      for (const p of r.det.pockets) {
        total += 1;
        if (Math.max(iou(p, gt[0]), iou(p, gt[1])) >= 0.5) matched += 1;
      }
    }
    assert(total > 0, 'there were pockets to score');
    const rate = matched / total;
    assert(rate >= 0.90, 'at least 90 percent of reported pockets match a real one, got ' + rate.toFixed(3));
    if (out) out.pocketLocalisation = { pockets: total, matched: matched, rate: +rate.toFixed(4) };
  });

  test('the recovered face tilt tracks the angle the scene was drawn at', () => {
    const results = runRange(HELD_OUT_START, 120).filter(r => r.pred && r.det.tilt);
    assert(results.length > 40, 'enough frames produced a tilt estimate, got ' + results.length);
    const errs = results.map(r => Math.abs(r.det.tilt.tiltDeg - r.scene.truth.tiltDeg));
    errs.sort((a, b) => a - b);
    const mean = errs.reduce((a, b) => a + b, 0) / errs.length;
    const median = errs[Math.floor(errs.length / 2)];
    assert(median <= 2.0, 'median tilt error within 2 degrees, got ' + median.toFixed(2));
    if (out) out.tilt = {
      frames: results.length,
      meanAbsErrorDeg: +mean.toFixed(3),
      medianAbsErrorDeg: +median.toFixed(3),
      p90AbsErrorDeg: +errs[Math.floor(errs.length * 0.9)].toFixed(3),
    };
  });

  test('accuracy degrades gracefully as the sensor gets noisier', () => {
    const curve = [];
    for (const noise of [0.02, 0.05, 0.09, 0.14, 0.20, 0.30, 0.45]) {
      const s = score(runRange(HELD_OUT_START + 1000, 80, { noise: noise }), 0.5);
      curve.push({ noise: noise, f1: s.f1, precision: s.precision, recall: s.recall, meanIoU: s.meanIoU });
    }
    assert(curve[0].f1 >= 0.90, 'clean frames stay strong, got ' + curve[0].f1);
    for (let i = 1; i < curve.length; i++) {
      assert(curve[i].f1 <= curve[0].f1 + 0.02, 'noise does not improve accuracy at sigma ' + curve[i].noise);
    }
    if (out) out.noiseSweep = curve;
  });

  test('a pallet drawn far off level is still found', () => {
    // Honest limit: the region filters work on axis aligned bounding boxes, so
    // a heavily rotated pocket stops looking like the shape they expect. Recall
    // falls off well before the detector stops working entirely.
    const s = score(runRange(HELD_OUT_START + 2000, 60, { maxTiltDeg: 18 }), 0.5);
    assert(s.recall >= 0.45, 'still finds roughly half the pallets at 18 degrees, got ' + s.recall);
    assert(s.precision >= 0.85, 'and what it does report is still right, got ' + s.precision);
    if (out) out.highTilt = s;
  });

  test('detection is deterministic for a given seed', () => {
    const a = JSON.stringify(detect(renderScene(4242, {}).image).pallet);
    const b = JSON.stringify(detect(renderScene(4242, {}).image).pallet);
    assertEqual(a, b, 'same seed, same answer');
  });
};
