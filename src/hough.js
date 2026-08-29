'use strict';

// Standard Hough transform over an edge mask. The detector uses it to measure
// how far the pallet face is rotated, which is the number a fork controller
// would actually care about.

function houghLines(mask, opts) {
  opts = opts || {};
  const thetaSteps = opts.thetaSteps || 180;
  const { width, height } = mask;
  const diag = Math.ceil(Math.hypot(width, height));
  const rhoBins = 2 * diag + 1;
  const acc = new Int32Array(thetaSteps * rhoBins);

  const cos = new Float64Array(thetaSteps);
  const sin = new Float64Array(thetaSteps);
  for (let t = 0; t < thetaSteps; t++) {
    const theta = (t * Math.PI) / thetaSteps;
    cos[t] = Math.cos(theta);
    sin[t] = Math.sin(theta);
  }

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      if (!mask.at(x, y)) continue;
      for (let t = 0; t < thetaSteps; t++) {
        const rho = Math.round(x * cos[t] + y * sin[t]) + diag;
        acc[t * rhoBins + rho] += 1;
      }
    }
  }

  const minVotes = opts.minVotes || 40;
  const peaks = [];
  for (let t = 0; t < thetaSteps; t++) {
    for (let r = 0; r < rhoBins; r++) {
      const v = acc[t * rhoBins + r];
      if (v < minVotes) continue;
      // Local maximum in a small neighbourhood, otherwise one real line shows
      // up as a smear of near-identical peaks.
      let isPeak = true;
      for (let dt = -2; dt <= 2 && isPeak; dt++) {
        for (let dr = -2; dr <= 2; dr++) {
          if (dt === 0 && dr === 0) continue;
          const tt = t + dt, rr = r + dr;
          if (tt < 0 || rr < 0 || tt >= thetaSteps || rr >= rhoBins) continue;
          if (acc[tt * rhoBins + rr] > v) { isPeak = false; break; }
        }
      }
      if (isPeak) {
        peaks.push({ theta: (t * Math.PI) / thetaSteps, thetaDeg: (t * 180) / thetaSteps, rho: r - diag, votes: v });
      }
    }
  }
  peaks.sort((a, b) => b.votes - a.votes);
  return { peaks: peaks, accumulator: acc, thetaSteps: thetaSteps, rhoBins: rhoBins, diag: diag };
}

// Angle of the strongest roughly horizontal line, reported in degrees away
// from level. Positive means the face tilts one way, negative the other.
function dominantHorizontalTilt(peaks, tolDeg) {
  const tol = tolDeg === undefined ? 25 : tolDeg;
  for (const p of peaks) {
    const off = p.thetaDeg - 90; // theta of 90 degrees is a horizontal line
    if (Math.abs(off) <= tol) return { tiltDeg: off, votes: p.votes, rho: p.rho };
  }
  return null;
}

module.exports = { houghLines, dominantHorizontalTilt };
