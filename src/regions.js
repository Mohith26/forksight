'use strict';

const { Mask } = require('./image');

// Connected components with union-find, plus the region statistics the
// geometry filter needs. Eight connectivity.

class UnionFind {
  constructor(n) {
    this.parent = new Int32Array(n);
    for (let i = 0; i < n; i++) this.parent[i] = i;
    this.rank = new Int32Array(n);
    this.size = n;
  }
  find(a) {
    let root = a;
    while (this.parent[root] !== root) root = this.parent[root];
    while (this.parent[a] !== root) { const next = this.parent[a]; this.parent[a] = root; a = next; }
    return root;
  }
  union(a, b) {
    const ra = this.find(a), rb = this.find(b);
    if (ra === rb) return false;
    if (this.rank[ra] < this.rank[rb]) this.parent[ra] = rb;
    else if (this.rank[ra] > this.rank[rb]) this.parent[rb] = ra;
    else { this.parent[rb] = ra; this.rank[ra] += 1; }
    return true;
  }
}

function label(mask) {
  const { width, height } = mask;
  const uf = new UnionFind(width * height);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      if (!mask.at(x, y)) continue;
      const i = y * width + x;
      // Only look back, so one pass plus union-find is enough.
      const neighbours = [[-1, 0], [-1, -1], [0, -1], [1, -1]];
      for (const [dx, dy] of neighbours) {
        const nx = x + dx, ny = y + dy;
        if (nx < 0 || ny < 0 || nx >= width || ny >= height) continue;
        if (mask.at(nx, ny)) uf.union(i, ny * width + nx);
      }
    }
  }
  const byRoot = new Map();
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      if (!mask.at(x, y)) continue;
      const root = uf.find(y * width + x);
      let r = byRoot.get(root);
      if (!r) { r = { area: 0, minX: x, maxX: x, minY: y, maxY: y, sumX: 0, sumY: 0 }; byRoot.set(root, r); }
      r.area += 1;
      r.sumX += x; r.sumY += y;
      if (x < r.minX) r.minX = x;
      if (x > r.maxX) r.maxX = x;
      if (y < r.minY) r.minY = y;
      if (y > r.maxY) r.maxY = y;
    }
  }
  const regions = [];
  for (const r of byRoot.values()) {
    const w = r.maxX - r.minX + 1;
    const h = r.maxY - r.minY + 1;
    regions.push({
      x: r.minX, y: r.minY, w: w, h: h,
      area: r.area,
      cx: r.sumX / r.area,
      cy: r.sumY / r.area,
      aspect: w / h,
      extent: r.area / (w * h), // how completely the blob fills its own box
    });
  }
  regions.sort((a, b) => b.area - a.area);
  return regions;
}

function threshold(img, value, below) {
  const m = new Mask(img.width, img.height);
  for (let i = 0; i < img.length; i++) {
    m.data[i] = (below ? img.data[i] <= value : img.data[i] >= value) ? 1 : 0;
  }
  return m;
}

// Local mean threshold. A global cut fails here because the synthetic floor has
// a lighting gradient, which is the whole reason it is in the renderer.
function adaptiveThreshold(img, radius, offset) {
  const { width, height } = img;
  const integral = new Float64Array((width + 1) * (height + 1));
  for (let y = 0; y < height; y++) {
    let rowSum = 0;
    for (let x = 0; x < width; x++) {
      rowSum += img.at(x, y);
      integral[(y + 1) * (width + 1) + (x + 1)] = integral[y * (width + 1) + (x + 1)] + rowSum;
    }
  }
  const boxSum = (x0, y0, x1, y1) => {
    const W = width + 1;
    return integral[(y1 + 1) * W + (x1 + 1)] - integral[y0 * W + (x1 + 1)] - integral[(y1 + 1) * W + x0] + integral[y0 * W + x0];
  };
  const m = new Mask(width, height);
  for (let y = 0; y < height; y++) {
    const y0 = Math.max(0, y - radius), y1 = Math.min(height - 1, y + radius);
    for (let x = 0; x < width; x++) {
      const x0 = Math.max(0, x - radius), x1 = Math.min(width - 1, x + radius);
      const n = (x1 - x0 + 1) * (y1 - y0 + 1);
      const mean = boxSum(x0, y0, x1, y1) / n;
      m.set(x, y, img.at(x, y) < mean - offset);
    }
  }
  return m;
}

module.exports = { UnionFind, label, threshold, adaptiveThreshold };
