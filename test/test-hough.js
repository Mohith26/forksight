'use strict';

const { test, assert, assertClose } = require('./harness');
const { Image, Mask } = require('../src/image');
const { houghLines, dominantHorizontalTilt } = require('../src/hough');
const { canny } = require('../src/filters');
const { fillRect } = require('../src/synth');

module.exports = function run(out) {
  test('a drawn horizontal line lands in the expected accumulator cell', () => {
    const m = new Mask(120, 90);
    for (let x = 10; x < 110; x++) m.set(x, 45, 1);
    const h = houghLines(m, { minVotes: 60 });
    assert(h.peaks.length > 0, 'found at least one peak');
    const top = h.peaks[0];
    assertClose(top.thetaDeg, 90, 1.5, 'a horizontal line sits at theta 90');
    assertClose(top.rho, 45, 1.5, 'and at rho equal to its row');
  });

  test('a vertical line lands at theta zero', () => {
    const m = new Mask(120, 90);
    for (let y = 5; y < 85; y++) m.set(60, y, 1);
    const top = houghLines(m, { minVotes: 50 }).peaks[0];
    assert(top.thetaDeg < 2 || top.thetaDeg > 178, 'vertical line near theta 0, got ' + top.thetaDeg);
    assertClose(Math.abs(top.rho), 60, 1.5, 'rho equals the column');
  });

  test('a tilted line is recovered to within a degree', () => {
    const errors = [];
    for (const deg of [-12, -7, -3, 3, 7, 12]) {
      const m = new Mask(160, 120);
      const rad = (deg * Math.PI) / 180;
      for (let t = -60; t <= 60; t += 0.25) {
        const x = Math.round(80 + t * Math.cos(rad));
        const y = Math.round(60 + t * Math.sin(rad));
        if (x >= 0 && y >= 0 && x < 160 && y < 120) m.set(x, y, 1);
      }
      const tilt = dominantHorizontalTilt(houghLines(m, { minVotes: 40 }).peaks);
      assert(tilt !== null, 'found a near horizontal line at ' + deg);
      const err = Math.abs(tilt.tiltDeg - deg);
      errors.push(err);
      assert(err <= 1.5, 'angle within 1.5 degrees at ' + deg + ', error was ' + err.toFixed(2));
    }
    if (out) out.houghSyntheticMaxErrorDeg = +Math.max.apply(null, errors).toFixed(3);
  });

  test('the four sides of a rectangle show up as four strong peaks', () => {
    const img = new Image(140, 110).fill(0.15);
    fillRect(img, 30, 25, 70, 55, 0.85);
    const edges = canny(img, { sigma: 1.0 });
    const peaks = houghLines(edges.mask, { minVotes: 45 }).peaks;
    const horizontals = peaks.filter(p => Math.abs(p.thetaDeg - 90) < 6);
    const verticals = peaks.filter(p => p.thetaDeg < 6 || p.thetaDeg > 174);
    assert(horizontals.length >= 2, 'two horizontal sides, got ' + horizontals.length);
    assert(verticals.length >= 2, 'two vertical sides, got ' + verticals.length);
  });
};
