'use strict';

const { Image, Mask, convolveSeparable } = require('./image');

// Gaussian, Sobel, non maximum suppression and hysteresis, written out rather
// than pulled from a vision library, because the point of the project was to
// understand what those stages actually do.

function gaussianKernel1d(sigma) {
  const radius = Math.max(1, Math.ceil(3 * sigma));
  const size = 2 * radius + 1;
  const k = new Float64Array(size);
  let sum = 0;
  for (let i = 0; i < size; i++) {
    const d = i - radius;
    k[i] = Math.exp(-(d * d) / (2 * sigma * sigma));
    sum += k[i];
  }
  for (let i = 0; i < size; i++) k[i] /= sum;
  return k;
}

function gaussianBlur(img, sigma) {
  const k = gaussianKernel1d(sigma);
  return convolveSeparable(img, k, k);
}

const SOBEL_X = [-1, 0, 1, -2, 0, 2, -1, 0, 1];
const SOBEL_Y = [-1, -2, -1, 0, 0, 0, 1, 2, 1];

function sobel(img) {
  const gx = Image.like(img);
  const gy = Image.like(img);
  const mag = Image.like(img);
  const dir = Image.like(img);
  for (let y = 0; y < img.height; y++) {
    for (let x = 0; x < img.width; x++) {
      let ax = 0, ay = 0, i = 0;
      for (let ky = -1; ky <= 1; ky++) {
        for (let kx = -1; kx <= 1; kx++, i++) {
          const v = img.atClamped(x + kx, y + ky);
          ax += SOBEL_X[i] * v;
          ay += SOBEL_Y[i] * v;
        }
      }
      gx.set(x, y, ax);
      gy.set(x, y, ay);
      mag.set(x, y, Math.hypot(ax, ay));
      dir.set(x, y, Math.atan2(ay, ax));
    }
  }
  return { gx: gx, gy: gy, mag: mag, dir: dir };
}

// Thin the gradient ridge to one pixel by keeping only local maxima along the
// gradient direction, quantized to the four diagonal neighbours.
function nonMaxSuppression(mag, dir) {
  const out = Image.like(mag);
  for (let y = 0; y < mag.height; y++) {
    for (let x = 0; x < mag.width; x++) {
      let angle = (dir.at(x, y) * 180) / Math.PI;
      if (angle < 0) angle += 180;
      let a, b;
      if (angle < 22.5 || angle >= 157.5) {
        a = mag.atClamped(x - 1, y); b = mag.atClamped(x + 1, y);
      } else if (angle < 67.5) {
        a = mag.atClamped(x - 1, y + 1); b = mag.atClamped(x + 1, y - 1);
      } else if (angle < 112.5) {
        a = mag.atClamped(x, y - 1); b = mag.atClamped(x, y + 1);
      } else {
        a = mag.atClamped(x - 1, y - 1); b = mag.atClamped(x + 1, y + 1);
      }
      const v = mag.at(x, y);
      out.set(x, y, v >= a && v >= b ? v : 0);
    }
  }
  return out;
}

// Two thresholds, then keep weak pixels only when they are reachable from a
// strong one. Implemented iteratively so a long thin edge does not blow the
// call stack.
function hysteresis(nms, low, high) {
  const mask = new Mask(nms.width, nms.height);
  const stack = [];
  for (let i = 0; i < nms.length; i++) {
    if (nms.data[i] >= high) { mask.data[i] = 1; stack.push(i); }
  }
  while (stack.length) {
    const i = stack.pop();
    const x = i % nms.width;
    const y = (i / nms.width) | 0;
    for (let dy = -1; dy <= 1; dy++) {
      for (let dx = -1; dx <= 1; dx++) {
        const nx = x + dx, ny = y + dy;
        if (nx < 0 || ny < 0 || nx >= nms.width || ny >= nms.height) continue;
        const j = ny * nms.width + nx;
        if (!mask.data[j] && nms.data[j] >= low) { mask.data[j] = 1; stack.push(j); }
      }
    }
  }
  return mask;
}

function canny(img, opts) {
  opts = opts || {};
  const sigma = opts.sigma === undefined ? 1.2 : opts.sigma;
  const blurred = gaussianBlur(img, sigma);
  const g = sobel(blurred);
  const nms = nonMaxSuppression(g.mag, g.dir);
  const { max } = nms.minMax();
  const high = opts.high === undefined ? max * 0.28 : opts.high;
  const low = opts.low === undefined ? high * 0.4 : opts.low;
  return { mask: hysteresis(nms, low, high), gradient: g, nms: nms, high: high, low: low };
}

module.exports = { gaussianKernel1d, gaussianBlur, sobel, nonMaxSuppression, hysteresis, canny, SOBEL_X, SOBEL_Y };
