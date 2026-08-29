'use strict';

const { renderScene } = require('../src/synth');
const { detect, pocketCandidates, DEFAULTS } = require('../src/detect');
const { gaussianBlur, canny } = require('../src/filters');
const { adaptiveThreshold, label } = require('../src/regions');
const { score } = require('../src/evaluate');
const { WIDTH, HEIGHT } = require('../src/synth');

function now() {
  if (typeof process !== 'undefined' && process.hrtime && process.hrtime.bigint) return Number(process.hrtime.bigint()) / 1e6;
  if (typeof performance !== 'undefined' && performance.now) return performance.now();
  return Date.now();
}

function timed(fn) {
  const t0 = now();
  const value = fn();
  return { ms: now() - t0, value: value };
}

function run() {
  const out = { frameSize: WIDTH + 'x' + HEIGHT };

  const scenes = [];
  for (let i = 0; i < 120; i++) scenes.push(renderScene(7000 + i, {}));

  const full = timed(() => { let n = 0; for (const s of scenes) if (detect(s.image).pallet) n++; return n; });
  out.endToEnd = {
    frames: scenes.length,
    ms: +full.ms.toFixed(2),
    fps: +(scenes.length / (full.ms / 1000)).toFixed(1),
    msPerFrame: +(full.ms / scenes.length).toFixed(3),
    megapixelsPerSec: +((scenes.length * WIDTH * HEIGHT) / (full.ms / 1000) / 1e6).toFixed(2),
    detected: full.value,
  };

  // Per stage, so the slow part is identified rather than guessed at.
  const one = scenes[0].image;
  const reps = 60;
  const blur = timed(() => { for (let i = 0; i < reps; i++) gaussianBlur(one, DEFAULTS.sigma); });
  const blurred = gaussianBlur(one, DEFAULTS.sigma);
  const thresh = timed(() => { for (let i = 0; i < reps; i++) adaptiveThreshold(blurred, DEFAULTS.threshRadius, DEFAULTS.threshOffset); });
  const mask = adaptiveThreshold(blurred, DEFAULTS.threshRadius, DEFAULTS.threshOffset);
  const cc = timed(() => { for (let i = 0; i < reps; i++) label(mask); });
  const edges = timed(() => { for (let i = 0; i < reps; i++) canny(blurred, { sigma: 0.8 }); });
  out.stages = {
    reps: reps,
    gaussianBlur_ms: +(blur.ms / reps).toFixed(3),
    adaptiveThreshold_ms: +(thresh.ms / reps).toFixed(3),
    connectedComponents_ms: +(cc.ms / reps).toFixed(3),
    canny_ms: +(edges.ms / reps).toFixed(3),
  };

  // Accuracy on the same held out range the tests use, restated here so the
  // results file carries both numbers from one run.
  const results = [];
  for (let i = 0; i < 200; i++) {
    const s = renderScene(5000 + i, {});
    results.push({ pred: detect(s.image).pallet, truth: s.truth.pallet });
  }
  out.accuracy = score(results, 0.5);

  const strict = score(results, 0.75);
  out.accuracyStrictIoU = { threshold: 0.75, precision: strict.precision, recall: strict.recall, f1: strict.f1 };

  return out;
}

module.exports = { run: run };

if (require.main === module) console.log(JSON.stringify(run(), null, 2));
