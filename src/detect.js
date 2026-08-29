'use strict';

const { gaussianBlur, canny } = require('./filters');
const { adaptiveThreshold, label } = require('./regions');
const { houghLines, dominantHorizontalTilt } = require('./hough');

// Pallet detection. The shape of a pallet face is the thing that makes this
// tractable: two dark pockets of similar size, sitting at the same height, with
// a centre block between them. Everything below is that statement turned into
// filters.

const DEFAULTS = {
  sigma: 1.1,
  threshRadius: 22,
  threshOffset: 0.08,
  minArea: 140,
  maxArea: 9000,
  minAspect: 0.45,
  maxAspect: 3.2,
  minExtent: 0.65,
  heightTol: 0.30,     // how different two pockets may be in height
  areaTol: 0.55,       // and in area
  centreTol: 0.35,     // vertical centre offset, as a fraction of pocket height
  minGapRatio: 0.25,   // gap between pockets, relative to pocket width
  maxGapRatio: 2.60,
};

function pocketCandidates(img, cfg) {
  const blurred = gaussianBlur(img, cfg.sigma);
  const mask = adaptiveThreshold(blurred, cfg.threshRadius, cfg.threshOffset);
  const regions = label(mask);
  const kept = [];
  for (const r of regions) {
    if (r.area < cfg.minArea || r.area > cfg.maxArea) continue;
    if (r.aspect < cfg.minAspect || r.aspect > cfg.maxAspect) continue;
    if (r.extent < cfg.minExtent) continue;
    kept.push(r);
  }
  return { blurred: blurred, mask: mask, regions: regions, candidates: kept };
}

function pairPockets(cands, cfg) {
  const pairs = [];
  for (let i = 0; i < cands.length; i++) {
    for (let j = i + 1; j < cands.length; j++) {
      const a = cands[i].cx <= cands[j].cx ? cands[i] : cands[j];
      const b = cands[i].cx <= cands[j].cx ? cands[j] : cands[i];

      const hRatio = Math.min(a.h, b.h) / Math.max(a.h, b.h);
      if (hRatio < 1 - cfg.heightTol) continue;

      const aRatio = Math.min(a.area, b.area) / Math.max(a.area, b.area);
      if (aRatio < 1 - cfg.areaTol) continue;

      const meanH = (a.h + b.h) / 2;
      if (Math.abs(a.cy - b.cy) > cfg.centreTol * meanH) continue;

      const gap = b.x - (a.x + a.w);
      const meanW = (a.w + b.w) / 2;
      if (gap < cfg.minGapRatio * meanW || gap > cfg.maxGapRatio * meanW) continue;

      const x = a.x;
      const y = Math.min(a.y, b.y);
      const w = b.x + b.w - a.x;
      const h = Math.max(a.y + a.h, b.y + b.h) - y;

      // Confidence is just the agreement between the two pockets. It is used to
      // pick one winner per frame, not presented as a probability.
      const score = hRatio * aRatio * (1 - Math.abs(a.cy - b.cy) / (cfg.centreTol * meanH + 1e-9)) * 0.5 + 0.5 * hRatio * aRatio;
      pairs.push({ x: x, y: y, w: w, h: h, score: score, left: a, right: b, gap: gap });
    }
  }
  pairs.sort((p, q) => q.score - p.score);
  return pairs;
}

function detect(img, options) {
  const cfg = Object.assign({}, DEFAULTS, options || {});
  const stage = pocketCandidates(img, cfg);
  const pairs = pairPockets(stage.candidates, cfg);
  const best = pairs.length ? pairs[0] : null;

  let tilt = null;
  if (best) {
    const edges = canny(stage.blurred, { sigma: 0.8 });
    const h = houghLines(edges.mask, { minVotes: 40 });
    tilt = dominantHorizontalTilt(h.peaks);
  }

  return {
    pallet: best ? { x: best.x, y: best.y, w: best.w, h: best.h, score: best.score } : null,
    pockets: best ? [best.left, best.right].map(r => ({ x: r.x, y: r.y, w: r.w, h: r.h })) : [],
    tilt: tilt,
    candidateCount: stage.candidates.length,
    regionCount: stage.regions.length,
  };
}

module.exports = { detect, pocketCandidates, pairPockets, DEFAULTS };
