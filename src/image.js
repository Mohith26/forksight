'use strict';

// Single channel image buffers. Everything downstream works on Float32 so the
// filter stages do not quantize between steps.

class Image {
  constructor(width, height, data) {
    this.width = width;
    this.height = height;
    this.data = data || new Float32Array(width * height);
  }

  static like(other) { return new Image(other.width, other.height); }

  at(x, y) { return this.data[y * this.width + x]; }
  set(x, y, v) { this.data[y * this.width + x] = v; }

  // Clamp to edge. Every filter in this project uses the same border rule so
  // the differential tests do not trip over border handling.
  atClamped(x, y) {
    const cx = x < 0 ? 0 : (x >= this.width ? this.width - 1 : x);
    const cy = y < 0 ? 0 : (y >= this.height ? this.height - 1 : y);
    return this.data[cy * this.width + cx];
  }

  clone() { return new Image(this.width, this.height, this.data.slice()); }

  fill(v) { this.data.fill(v); return this; }

  minMax() {
    let lo = Infinity, hi = -Infinity;
    for (let i = 0; i < this.data.length; i++) {
      const v = this.data[i];
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
    return { min: lo, max: hi };
  }

  normalized() {
    const { min, max } = this.minMax();
    const out = Image.like(this);
    const range = max - min;
    for (let i = 0; i < this.data.length; i++) {
      out.data[i] = range === 0 ? 0 : (this.data[i] - min) / range;
    }
    return out;
  }

  get length() { return this.data.length; }
}

// Binary mask backed by a byte array, so connected components can walk it fast.
class Mask {
  constructor(width, height, data) {
    this.width = width;
    this.height = height;
    this.data = data || new Uint8Array(width * height);
  }
  at(x, y) { return this.data[y * this.width + x]; }
  set(x, y, v) { this.data[y * this.width + x] = v ? 1 : 0; }
  count() { let n = 0; for (let i = 0; i < this.data.length; i++) if (this.data[i]) n++; return n; }
}

function convolve2d(img, kernel, kw, kh) {
  const out = Image.like(img);
  const hx = (kw - 1) >> 1;
  const hy = (kh - 1) >> 1;
  for (let y = 0; y < img.height; y++) {
    for (let x = 0; x < img.width; x++) {
      let acc = 0;
      for (let ky = 0; ky < kh; ky++) {
        for (let kx = 0; kx < kw; kx++) {
          acc += kernel[ky * kw + kx] * img.atClamped(x + kx - hx, y + ky - hy);
        }
      }
      out.set(x, y, acc);
    }
  }
  return out;
}

// Separable form. Same result as convolve2d on the outer product kernel, which
// is exactly what test/test-filters.js asserts.
function convolveSeparable(img, kx, ky) {
  const hx = (kx.length - 1) >> 1;
  const hy = (ky.length - 1) >> 1;
  const tmp = Image.like(img);
  for (let y = 0; y < img.height; y++) {
    for (let x = 0; x < img.width; x++) {
      let acc = 0;
      for (let i = 0; i < kx.length; i++) acc += kx[i] * img.atClamped(x + i - hx, y);
      tmp.set(x, y, acc);
    }
  }
  const out = Image.like(img);
  for (let y = 0; y < img.height; y++) {
    for (let x = 0; x < img.width; x++) {
      let acc = 0;
      for (let i = 0; i < ky.length; i++) acc += ky[i] * tmp.atClamped(x, y + i - hy);
      out.set(x, y, acc);
    }
  }
  return out;
}

module.exports = { Image, Mask, convolve2d, convolveSeparable };
