'use strict';

const { Image } = require('./image');

// I do not have a warehouse, so the pipeline is developed against a renderer
// that draws pallet faces with known geometry. Every scene ships its own ground
// truth, which is the only reason the precision and recall numbers mean
// anything.
//
// The face is drawn rotated by a random angle. That matters: it is what stops
// the detector from being a trivial axis-aligned template match, and it gives
// the Hough stage a real angle to recover.

function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a = (a + 0x6D2B79F5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function fillRect(img, x0, y0, w, h, value) {
  for (let y = Math.max(0, y0); y < Math.min(img.height, y0 + h); y++) {
    for (let x = Math.max(0, x0); x < Math.min(img.width, x0 + w); x++) {
      img.set(x, y, value);
    }
  }
}

function addNoise(img, rnd, sigma) {
  if (sigma <= 0) return;
  for (let i = 0; i < img.length; i++) {
    // Box-Muller, so the noise is actually gaussian rather than uniform.
    const u1 = Math.max(1e-9, rnd());
    const u2 = rnd();
    const n = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    img.data[i] += n * sigma;
  }
}

// Axis aligned bounding box of a rectangle after rotation about a pivot.
function rotatedAabb(rect, cx, cy, theta) {
  const c = Math.cos(theta), s = Math.sin(theta);
  const pts = [
    [rect.x, rect.y],
    [rect.x + rect.w, rect.y],
    [rect.x, rect.y + rect.h],
    [rect.x + rect.w, rect.y + rect.h],
  ];
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (const [px, py] of pts) {
    const dx = px - cx, dy = py - cy;
    const rx = cx + dx * c - dy * s;
    const ry = cy + dx * s + dy * c;
    if (rx < minX) minX = rx;
    if (rx > maxX) maxX = rx;
    if (ry < minY) minY = ry;
    if (ry > maxY) maxY = ry;
  }
  return { x: Math.round(minX), y: Math.round(minY), w: Math.round(maxX - minX), h: Math.round(maxY - minY) };
}

const WIDTH = 320;
const HEIGHT = 240;

function renderScene(seed, opts) {
  opts = opts || {};
  const rnd = mulberry32(seed);
  const img = new Image(WIDTH, HEIGHT);
  const noise = opts.noise === undefined ? 0.05 : opts.noise;
  const maxTilt = opts.maxTiltDeg === undefined ? 10 : opts.maxTiltDeg;
  const withPallet = opts.withPallet === undefined ? rnd() < 0.8 : opts.withPallet;

  // Floor with a soft vertical lighting gradient, so a single global threshold
  // is not enough.
  for (let y = 0; y < HEIGHT; y++) {
    const base = 0.34 + 0.16 * (y / HEIGHT);
    for (let x = 0; x < WIDTH; x++) img.set(x, y, base);
  }

  const truth = { pockets: [], pallet: null, seed: seed, face: null, tiltDeg: 0 };

  if (withPallet) {
    const pw = Math.round(120 + rnd() * 90);
    const ph = Math.round(pw * (0.34 + rnd() * 0.10));
    const px = Math.round(30 + rnd() * (WIDTH - pw - 60));
    const py = Math.round(60 + rnd() * (HEIGHT - ph - 80));
    const tiltDeg = (rnd() * 2 - 1) * maxTilt;
    const theta = (tiltDeg * Math.PI) / 180;
    const cx = px + pw / 2;
    const cy = py + ph / 2;

    const pocketW = Math.round(pw * (0.24 + rnd() * 0.05));
    const pocketH = Math.round(ph * (0.52 + rnd() * 0.10));
    const pocketY = py + Math.round((ph - pocketH) / 2);
    const inset = Math.round(pw * (0.10 + rnd() * 0.04));
    const leftX = px + inset;
    const rightX = px + pw - inset - pocketW;
    const faceValue = 0.72 + rnd() * 0.12;
    const dark = 0.10 + rnd() * 0.06;

    const face = { x: px, y: py, w: pw, h: ph };
    const pockets = [
      { x: leftX, y: pocketY, w: pocketW, h: pocketH },
      { x: rightX, y: pocketY, w: pocketW, h: pocketH },
    ];

    // Inverse mapping: walk the rotated bounding area and ask, for each output
    // pixel, where it came from in the unrotated face frame.
    const faceAabb = rotatedAabb(face, cx, cy, theta);
    const c = Math.cos(-theta), s = Math.sin(-theta);
    const inside = (rx, ry, r) => rx >= r.x && rx < r.x + r.w && ry >= r.y && ry < r.y + r.h;
    for (let y = Math.max(0, faceAabb.y - 2); y < Math.min(HEIGHT, faceAabb.y + faceAabb.h + 2); y++) {
      for (let x = Math.max(0, faceAabb.x - 2); x < Math.min(WIDTH, faceAabb.x + faceAabb.w + 2); x++) {
        const dx = x - cx, dy = y - cy;
        const sx = cx + dx * c - dy * s;
        const sy = cy + dx * s + dy * c;
        if (!inside(sx, sy, face)) continue;
        let v = faceValue;
        for (const p of pockets) if (inside(sx, sy, p)) { v = dark; break; }
        img.set(x, y, v);
      }
    }

    truth.tiltDeg = tiltDeg;
    truth.face = faceAabb;
    for (const p of pockets) truth.pockets.push(rotatedAabb(p, cx, cy, theta));
    const a = truth.pockets[0], b = truth.pockets[1];
    const minX = Math.min(a.x, b.x), minY = Math.min(a.y, b.y);
    truth.pallet = {
      x: minX,
      y: minY,
      w: Math.max(a.x + a.w, b.x + b.w) - minX,
      h: Math.max(a.y + a.h, b.y + b.h) - minY,
    };
  }

  // Distractors sit on the floor around the pallet, not on top of it. Other
  // stock occluding the face is a different problem than the one this detector
  // is for, so the renderer keeps them clear of the face by a margin.
  const clear = (x, y, w, h) => {
    if (!truth.face) return true;
    const f = truth.face;
    const m = 6;
    return !(x < f.x + f.w + m && x + w > f.x - m && y < f.y + f.h + m && y + h > f.y - m);
  };
  const place = (w, h, value) => {
    for (let attempt = 0; attempt < 40; attempt++) {
      const x = Math.round(rnd() * (WIDTH - w));
      const y = Math.round(rnd() * (HEIGHT - h));
      if (!clear(x, y, w, h)) continue;
      fillRect(img, x, y, w, h, value);
      return true;
    }
    return false;
  };

  const lone = 1 + Math.floor(rnd() * 3);
  for (let i = 0; i < lone; i++) {
    place(Math.round(14 + rnd() * 34), Math.round(12 + rnd() * 30), 0.09 + rnd() * 0.07);
  }
  // A dark pair spaced too far apart to be fork pockets, so the pairing stage
  // has a genuine false positive to reject rather than an easy one.
  if (rnd() < 0.6) {
    const w = Math.round(16 + rnd() * 18);
    const h = Math.round(14 + rnd() * 16);
    const gap = Math.round(90 + rnd() * 60);
    for (let attempt = 0; attempt < 40; attempt++) {
      const x = Math.round(rnd() * Math.max(1, WIDTH - 2 * w - gap));
      const y = Math.round(rnd() * (HEIGHT - h));
      if (!clear(x, y, 2 * w + gap, h)) continue;
      fillRect(img, x, y, w, h, 0.10);
      fillRect(img, x + w + gap, y, w, h, 0.10);
      break;
    }
  }

  addNoise(img, rnd, noise);
  for (let i = 0; i < img.length; i++) {
    img.data[i] = img.data[i] < 0 ? 0 : (img.data[i] > 1 ? 1 : img.data[i]);
  }
  return { image: img, truth: truth };
}

module.exports = { renderScene, mulberry32, fillRect, addNoise, rotatedAabb, WIDTH, HEIGHT };
