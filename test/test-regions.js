'use strict';

const { test, assert, assertEqual, assertClose } = require('./harness');
const { Image, Mask } = require('../src/image');
const { UnionFind, label, threshold, adaptiveThreshold } = require('../src/regions');
const { fillRect } = require('../src/synth');

module.exports = function run() {
  test('union find merges and keeps the sets separate', () => {
    const uf = new UnionFind(10);
    assert(uf.union(0, 1), 'first union merges');
    assert(uf.union(1, 2), 'second union merges');
    assert(!uf.union(0, 2), 'already joined returns false');
    assertEqual(uf.find(0), uf.find(2), 'transitively joined');
    assert(uf.find(0) !== uf.find(5), 'untouched element stays separate');
  });

  test('labelling counts separated blobs and merges touching ones', () => {
    const m = new Mask(30, 20);
    for (let y = 2; y < 8; y++) for (let x = 2; x < 9; x++) m.set(x, y, 1);
    for (let y = 12; y < 17; y++) for (let x = 20; x < 26; x++) m.set(x, y, 1);
    const regions = label(m);
    assertEqual(regions.length, 2, 'two separated blobs');
    assertEqual(regions[0].area, 6 * 7, 'largest area');
    assertEqual(regions[1].area, 5 * 6, 'second area');
    // Diagonal touch must merge under eight connectivity.
    m.set(9, 8, 1);
    m.set(10, 9, 1);
    for (let y = 9; y < 12; y++) for (let x = 10; x < 14; x++) m.set(x, y, 1);
    assertEqual(label(m).length, 2, 'diagonal contact merges into one blob');
  });

  test('region statistics match the rectangle they were computed from', () => {
    const m = new Mask(40, 40);
    for (let y = 10; y < 25; y++) for (let x = 5; x < 30; x++) m.set(x, y, 1);
    const r = label(m)[0];
    assertEqual(r.x, 5); assertEqual(r.y, 10);
    assertEqual(r.w, 25); assertEqual(r.h, 15);
    assertEqual(r.area, 25 * 15);
    assertClose(r.aspect, 25 / 15, 1e-12, 'aspect');
    assertClose(r.extent, 1, 1e-12, 'a solid rectangle fills its own box');
    assertClose(r.cx, (5 + 29) / 2, 1e-9, 'centroid x');
    assertClose(r.cy, (10 + 24) / 2, 1e-9, 'centroid y');
  });

  test('a global threshold fails on a lit gradient where the adaptive one works', () => {
    const img = new Image(120, 90);
    for (let y = 0; y < 90; y++) for (let x = 0; x < 120; x++) img.set(x, y, 0.30 + 0.40 * (y / 90));
    fillRect(img, 10, 8, 20, 16, 0.18);   // dark patch on the bright-ish top
    fillRect(img, 80, 66, 20, 16, 0.45);  // dark patch relative to the darker bottom

    // No single global cut can catch both without swallowing half the frame.
    const global = threshold(img, 0.42, true);
    const globalRegions = label(global).filter(r => r.area > 40);
    assert(globalRegions.length < 2 || globalRegions[0].area > 2000,
      'global thresholding either misses one patch or floods, got ' + JSON.stringify(globalRegions.map(r => r.area)));

    const adaptive = adaptiveThreshold(img, 12, 0.05);
    const found = label(adaptive).filter(r => r.area > 100);
    assertEqual(found.length, 2, 'adaptive finds both patches, got ' + JSON.stringify(found.map(r => r.area)));
    for (const r of found) assert(r.area < 600, 'and neither one floods, area ' + r.area);
  });

  test('the integral image behind the adaptive threshold agrees with a direct mean', () => {
    const img = new Image(50, 40);
    for (let i = 0; i < img.length; i++) img.data[i] = ((i * 37) % 101) / 100;
    const radius = 5;
    const mask = adaptiveThreshold(img, radius, 0.02);
    for (const [x, y] of [[10, 10], [0, 0], [49, 39], [25, 3], [7, 36]]) {
      let sum = 0, n = 0;
      for (let yy = Math.max(0, y - radius); yy <= Math.min(39, y + radius); yy++) {
        for (let xx = Math.max(0, x - radius); xx <= Math.min(49, x + radius); xx++) { sum += img.at(xx, yy); n++; }
      }
      const expected = img.at(x, y) < sum / n - 0.02 ? 1 : 0;
      assertEqual(mask.at(x, y), expected, 'adaptive decision at ' + x + ',' + y);
    }
  });
};
