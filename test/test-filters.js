'use strict';

const { test, assert, assertEqual, assertClose } = require('./harness');
const { Image, convolve2d, convolveSeparable } = require('../src/image');
const { gaussianKernel1d, gaussianBlur, sobel, nonMaxSuppression, hysteresis, canny } = require('../src/filters');
const { fillRect } = require('../src/synth');

module.exports = function run(out) {
  test('a gaussian kernel sums to one and is symmetric', () => {
    for (const sigma of [0.6, 1.0, 1.5, 2.4]) {
      const k = gaussianKernel1d(sigma);
      let sum = 0;
      for (const v of k) sum += v;
      assertClose(sum, 1, 1e-12, 'kernel sums to 1 at sigma ' + sigma);
      for (let i = 0; i < k.length; i++) {
        assertClose(k[i], k[k.length - 1 - i], 1e-15, 'symmetric at sigma ' + sigma);
      }
    }
  });

  test('the separable blur equals the full 2d convolution of the outer product', () => {
    const img = new Image(41, 33);
    for (let i = 0; i < img.length; i++) img.data[i] = Math.sin(i * 0.37) * 0.5 + 0.5;
    const k = gaussianKernel1d(1.3);
    const outer = new Float64Array(k.length * k.length);
    for (let y = 0; y < k.length; y++) for (let x = 0; x < k.length; x++) outer[y * k.length + x] = k[y] * k[x];
    const full = convolve2d(img, outer, k.length, k.length);
    const sep = convolveSeparable(img, k, k);
    let worst = 0;
    for (let i = 0; i < img.length; i++) worst = Math.max(worst, Math.abs(full.data[i] - sep.data[i]));
    assert(worst < 1e-6, 'separable matches 2d within 1e-6, worst was ' + worst);
    if (out) out.separableMaxDelta = worst;
  });

  test('blurring a flat image changes nothing', () => {
    const flat = new Image(20, 20).fill(0.42);
    const blurred = gaussianBlur(flat, 2.0);
    let worst = 0;
    for (let i = 0; i < flat.length; i++) worst = Math.max(worst, Math.abs(blurred.data[i] - 0.42));
    // Buffers are Float32, so the bar is float32 epsilon rather than double.
    assert(worst < 1e-6, 'flat stays flat, worst delta ' + worst);
  });

  test('sobel recovers the slope of a linear ramp', () => {
    const img = new Image(30, 30);
    // Intensity rises by 0.01 per column, so the horizontal derivative is known.
    for (let y = 0; y < 30; y++) for (let x = 0; x < 30; x++) img.set(x, y, x * 0.01);
    const g = sobel(img);
    // The 3x3 sobel x kernel sums to 8 times the per pixel slope.
    for (let y = 5; y < 25; y++) {
      for (let x = 5; x < 25; x++) assertClose(g.gx.at(x, y), 0.08, 1e-6, 'gx on ramp');
    }
    for (let y = 5; y < 25; y++) {
      for (let x = 5; x < 25; x++) assertClose(g.gy.at(x, y), 0, 1e-6, 'gy on ramp');
    }
  });

  test('non maximum suppression thins an edge without moving it', () => {
    const img = new Image(40, 40).fill(0.2);
    fillRect(img, 20, 0, 20, 40, 0.9);
    const g = sobel(img);
    const nms = nonMaxSuppression(g.mag, g.dir);
    for (let y = 5; y < 35; y++) {
      let lit = 0;
      let firstX = -1;
      for (let x = 0; x < 40; x++) if (nms.at(x, y) > 1e-6) { lit++; if (firstX < 0) firstX = x; }
      assert(lit <= 2, 'row ' + y + ' thinned to at most 2 pixels, got ' + lit);
      assert(Math.abs(firstX - 19) <= 1, 'edge stays at the step, row ' + y + ' first lit at ' + firstX);
    }
  });

  test('hysteresis keeps a weak pixel only when it touches a strong one', () => {
    const img = new Image(9, 3);
    // A connected weak run reaching a strong pixel, plus an isolated weak pixel.
    img.set(1, 1, 0.9); img.set(2, 1, 0.5); img.set(3, 1, 0.5);
    img.set(7, 1, 0.5);
    const mask = hysteresis(img, 0.4, 0.8);
    assertEqual(mask.at(1, 1), 1, 'strong pixel kept');
    assertEqual(mask.at(2, 1), 1, 'connected weak kept');
    assertEqual(mask.at(3, 1), 1, 'transitively connected weak kept');
    assertEqual(mask.at(7, 1), 0, 'isolated weak dropped');
  });

  test('canny outlines a rectangle and leaves its interior empty', () => {
    const img = new Image(80, 60).fill(0.15);
    fillRect(img, 20, 15, 40, 30, 0.85);
    const res = canny(img, { sigma: 1.0 });
    let border = 0, interior = 0;
    for (let y = 0; y < 60; y++) {
      for (let x = 0; x < 80; x++) {
        if (!res.mask.at(x, y)) continue;
        const nearEdge = (Math.abs(x - 20) <= 2 || Math.abs(x - 59) <= 2 || Math.abs(y - 15) <= 2 || Math.abs(y - 44) <= 2);
        if (nearEdge) border++; else interior++;
      }
    }
    assert(border > 100, 'the rectangle border lights up, got ' + border);
    assertEqual(interior, 0, 'nothing fires inside the solid region');
  });
};
