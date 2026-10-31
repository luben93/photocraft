//! DNG 1.6 ProfileGainTableMap (tag 52525): spatially varying gain tables, the local tone
//! mapping phone raws (Apple ProRAW, Google Pixel, Samsung Expert RAW) are rendered with.
//!
//! From the public DNG Specification 1.6.0.0 ("Additional Tags for Version 1.6.0.0"): a
//! `MapPointsV × MapPointsH` grid of tables of `MapPointsN` gains, placed by `MapOrigin` and
//! `MapSpacing` in fractions of the image (the active area), with pixel-centred sampling,
//! bilinear between tables and edge values outside the grid. Each pixel's table input is
//! `clamp((R, G, B, min, max) · MapInputWeights, 0, 1)`; its gain is the table's value there
//! (index = input × `MapPointsN`, linearly interpolated), and it multiplies the pixel's RGB. It
//! is applied in linear RIMM (linear ProPhoto here) after BaselineExposure and before the
//! profile's tone curve. The blob is normally in the file's byte order (an Apple ProRAW file is
//! big-endian, a Pixel file little-endian); the caller retries the other order when it does not
//! parse.

/// Most tables × points accepted (a Pixel 8 Pro file has 48 × 64 × 257 = 789 504).
const MAX_GAINS: usize = 1 << 24;
/// Largest gain accepted (+8 EV); the phone files' gains stay under 4.
const MAX_GAIN: f32 = 256.0;

/// A parsed ProfileGainTableMap.
#[derive(Clone, Debug, PartialEq)]
pub struct GainTableMap {
    points_v: usize,
    points_h: usize,
    spacing_v: f64,
    spacing_h: f64,
    origin_v: f64,
    origin_h: f64,
    points_n: usize,
    weights: [f32; 5],
    /// `points_v × points_h` tables of `points_n` gains, row by row.
    gains: Vec<f32>,
}

impl GainTableMap {
    /// Parses the tag's bytes (`le`: the file's byte order). `None` when they are not a valid map:
    /// a size that does not match the header, an empty or oversized grid, a gain or placement
    /// value that is negative or not finite, a gain above `MAX_GAIN`, or several tables along an
    /// axis with no spacing between them.
    pub(crate) fn parse(b: &[u8], le: bool) -> Option<GainTableMap> {
        let u32_at =
            |o: usize| b.get(o..o + 4).and_then(|s| <[u8; 4]>::try_from(s).ok()).map(|a| if le { u32::from_le_bytes(a) } else { u32::from_be_bytes(a) });
        let f64_at =
            |o: usize| b.get(o..o + 8).and_then(|s| <[u8; 8]>::try_from(s).ok()).map(|a| if le { f64::from_le_bytes(a) } else { f64::from_be_bytes(a) });
        let f32_at = |o: usize| u32_at(o).map(f32::from_bits);
        let (points_v, points_h) = (u32_at(0)? as usize, u32_at(4)? as usize);
        let (spacing_v, spacing_h, origin_v, origin_h) = (f64_at(8)?, f64_at(16)?, f64_at(24)?, f64_at(32)?);
        let points_n = u32_at(40)? as usize;
        let weights = [f32_at(44)?, f32_at(48)?, f32_at(52)?, f32_at(56)?, f32_at(60)?];
        let count = points_v.checked_mul(points_h)?.checked_mul(points_n)?;
        if points_v == 0 || points_h == 0 || points_n == 0 || count > MAX_GAINS || b.len() != count.checked_mul(4)?.checked_add(64)? {
            return None;
        }
        let placement = [spacing_v, spacing_h, origin_v, origin_h];
        // More than one table along an axis needs a positive spacing between them.
        let spaced = |points: usize, spacing: f64| spacing > 0.0 || (points == 1 && spacing == 0.0);
        if !placement.iter().all(|v| v.is_finite()) || !spaced(points_v, spacing_v) || !spaced(points_h, spacing_h) || !weights.iter().all(|w| w.is_finite()) {
            return None;
        }
        let gains: Vec<f32> = (0..count).map(|i| f32_at(64 + 4 * i)).collect::<Option<_>>()?;
        if !gains.iter().all(|g| g.is_finite() && (0.0..=MAX_GAIN).contains(g)) {
            return None;
        }
        Some(GainTableMap { points_v, points_h, spacing_v, spacing_h, origin_v, origin_h, points_n, weights, gains })
    }

    /// The fractional grid coordinate of image fraction `f` on an axis (clamped to the grid).
    fn grid(f: f64, origin: f64, spacing: f64, points: usize) -> (usize, usize, f32) {
        let last = points - 1;
        if last == 0 || spacing <= 0.0 {
            return (0, 0, 0.0);
        }
        let g = ((f - origin) / spacing).clamp(0.0, last as f64);
        let i0 = (g.floor() as usize).min(last);
        let i1 = (i0 + 1).min(last);
        (i0, i1, (g - i0 as f64) as f32)
    }

    /// Gain of table (`row`, `col`) at table input `t` (0–1).
    fn lookup(&self, row: usize, col: usize, t: f32) -> f32 {
        let base = (row * self.points_h + col) * self.points_n;
        let last = self.points_n - 1;
        let x = (t * self.points_n as f32).clamp(0.0, last as f32);
        let i0 = (x.floor() as usize).min(last);
        let i1 = (i0 + 1).min(last);
        let f = x - i0 as f32;
        let at = |i: usize| self.gains.get(base + i).copied().unwrap_or(1.0);
        at(i0) + (at(i1) - at(i0)) * f
    }

    /// The map placed on a `w × h` image area: each column's and row's grid coordinates worked
    /// out once, so a pixel's gain is four table lookups.
    pub(crate) fn placed(&self, w: usize, h: usize) -> Placed<'_> {
        let axis = |n: usize, origin: f64, spacing: f64, points: usize| {
            (0..n).map(|i| Self::grid((i as f64 + 0.5) / n.max(1) as f64, origin, spacing, points)).collect()
        };
        Placed { map: self, cols: axis(w, self.origin_h, self.spacing_h, self.points_h), rows: axis(h, self.origin_v, self.spacing_v, self.points_v) }
    }

    /// The table input of linear RIMM `rgb`: `(R, G, B, min, max) · MapInputWeights`, in 0–1.
    fn input(&self, rgb: [f32; 3]) -> f32 {
        let [r, g, b] = rgb;
        let w5 = self.weights;
        let t = (w5[0] * r + w5[1] * g + w5[2] * b + w5[3] * r.min(g).min(b) + w5[4] * r.max(g).max(b)).clamp(0.0, 1.0);
        if t.is_nan() { 0.0 } else { t }
    }

    /// The gain at grid cell (`r0`/`r1`, `c0`/`c1`) with fractions `fr`, `fc` for table input `t`.
    fn blend(&self, (r0, r1, fr): (usize, usize, f32), (c0, c1, fc): (usize, usize, f32), t: f32) -> f32 {
        let top = self.lookup(r0, c0, t) * (1.0 - fc) + self.lookup(r0, c1, t) * fc;
        let bottom = self.lookup(r1, c0, t) * (1.0 - fc) + self.lookup(r1, c1, t) * fc;
        top * (1.0 - fr) + bottom * fr
    }

    /// `rgb` (linear RIMM) of the pixel at (`x`, `y`) of a `w × h` image area, with its gain.
    #[cfg(test)]
    pub(crate) fn apply(&self, x: usize, y: usize, w: usize, h: usize, rgb: [f32; 3]) -> [f32; 3] {
        let g = self.gain(x, y, w, h, rgb);
        rgb.map(|v| v * g)
    }

    /// The gain of the pixel at (`x`, `y`) of a `w × h` image area whose linear RIMM colour,
    /// after BaselineExposure and before any user adjustment, is `rgb` (the table input).
    #[cfg(test)]
    pub(crate) fn gain(&self, x: usize, y: usize, w: usize, h: usize, rgb: [f32; 3]) -> f32 {
        // Pixel-centred sampling: pixel x covers (x + 0.5) / w of the image width.
        let fx = (x as f64 + 0.5) / w.max(1) as f64;
        let fy = (y as f64 + 0.5) / h.max(1) as f64;
        let rows = Self::grid(fy, self.origin_v, self.spacing_v, self.points_v);
        let cols = Self::grid(fx, self.origin_h, self.spacing_h, self.points_h);
        self.blend(rows, cols, self.input(rgb))
    }
}

/// A [`GainTableMap`] placed on an image area ([`GainTableMap::placed`]).
pub(crate) struct Placed<'a> {
    map: &'a GainTableMap,
    cols: Vec<(usize, usize, f32)>,
    rows: Vec<(usize, usize, f32)>,
}

impl Placed<'_> {
    /// The gain of the pixel at (`x`, `y`) of the area whose colour, after BaselineExposure and
    /// before any user adjustment, is `rgb`; 1 outside the area.
    pub(crate) fn gain(&self, x: usize, y: usize, rgb: [f32; 3]) -> f32 {
        match (self.rows.get(y), self.cols.get(x)) {
            (Some(&r), Some(&c)) => self.map.blend(r, c, self.map.input(rgb)),
            _ => 1.0,
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::GainTableMap;

    #[test]
    fn the_placed_map_gives_the_same_gains() {
        let m = GainTableMap::parse(
            &blob((3, 4, 5), (0.4, 0.3), (0.05, -0.1), [0.2, 0.5, 0.1, 0.1, 0.1], &(0..60).map(|i| 0.25 + (i % 7) as f32 * 0.3).collect::<Vec<_>>(), true),
            true,
        )
        .unwrap();
        let (w, h) = (37, 23);
        let p = m.placed(w, h);
        for y in 0..h {
            for x in 0..w {
                let rgb = [x as f32 / w as f32, y as f32 / h as f32, 0.4];
                assert!((p.gain(x, y, rgb) - m.gain(x, y, w, h, rgb)).abs() < 1e-6, "{x} {y}");
            }
        }
        assert_eq!(p.gain(w, 0, [0.5; 3]), 1.0, "outside the area");
    }

    /// A ProfileGainTableMap blob in the given byte order.
    pub(crate) fn blob(points: (u32, u32, u32), spacing: (f64, f64), origin: (f64, f64), weights: [f32; 5], gains: &[f32], le: bool) -> Vec<u8> {
        let mut b = Vec::new();
        let u = |b: &mut Vec<u8>, v: u32| b.extend(if le { v.to_le_bytes() } else { v.to_be_bytes() });
        let d = |b: &mut Vec<u8>, v: f64| b.extend(if le { v.to_le_bytes() } else { v.to_be_bytes() });
        u(&mut b, points.0);
        u(&mut b, points.1);
        d(&mut b, spacing.0);
        d(&mut b, spacing.1);
        d(&mut b, origin.0);
        d(&mut b, origin.1);
        u(&mut b, points.2);
        for w in weights.iter().chain(gains) {
            u(&mut b, w.to_bits());
        }
        b
    }

    const GREY: [f32; 5] = [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 0.0, 0.0];

    #[test]
    fn a_flat_table_multiplies_every_pixel() {
        for le in [true, false] {
            let m = GainTableMap::parse(&blob((1, 1, 2), (0.0, 0.0), (0.0, 0.0), GREY, &[2.0, 2.0], le), le).unwrap();
            assert_eq!(m.apply(3, 4, 10, 10, [0.1, 0.2, 0.3]), [0.2, 0.4, 0.6]);
        }
    }

    #[test]
    fn the_table_input_indexes_by_map_points_n_and_interpolates() {
        // 5 points: index = input × 5 (the specification's rule), clamped to the last point.
        let m = GainTableMap::parse(&blob((1, 1, 5), (0.0, 0.0), (0.0, 0.0), [0.0, 1.0, 0.0, 0.0, 0.0], &[4.0, 3.0, 2.0, 1.0, 0.5], true), true).unwrap();
        let gain = |g: f32| m.apply(0, 0, 1, 1, [1.0, g, 1.0])[0];
        assert!((gain(0.0) - 4.0).abs() < 1e-6);
        assert!((gain(0.2) - 3.0).abs() < 1e-6, "{}", gain(0.2));
        assert!((gain(0.3) - 2.5).abs() < 1e-6, "{}", gain(0.3));
        assert!((gain(1.0) - 0.5).abs() < 1e-6, "{}", gain(1.0));
        // The max weight: a pixel with one bright channel uses that channel.
        let m = GainTableMap::parse(&blob((1, 1, 2), (0.0, 0.0), (0.0, 0.0), [0.0, 0.0, 0.0, 0.0, 1.0], &[2.0, 1.0], true), true).unwrap();
        assert!((m.apply(0, 0, 1, 1, [0.1, 0.1, 0.4])[2] - 0.4 * 1.2).abs() < 1e-6);
    }

    #[test]
    fn tables_are_bilinear_between_grid_points_and_held_outside() {
        // Two tables across, at x = 0.25 and 0.75 of the width: gain 1 on the left, 3 on the right.
        let m = GainTableMap::parse(&blob((1, 2, 1), (1.0, 0.5), (0.0, 0.25), GREY, &[1.0, 3.0], true), true).unwrap();
        let gain_at = |x: usize| m.apply(x, 0, 100, 1, [0.5, 0.5, 0.5])[0] / 0.5;
        assert!((gain_at(24) - 1.0).abs() < 1e-5, "left of the first table: held");
        assert!((gain_at(49) - 1.98).abs() < 1e-5, "{}", gain_at(49)); // (49.5/100 - 0.25) / 0.5 = 0.49: 1 + 2 * 0.49
        assert!((gain_at(99) - 3.0).abs() < 1e-5, "right of the last table: held");
    }

    #[test]
    fn invalid_maps_are_refused() {
        let ok = blob((1, 1, 2), (0.0, 0.0), (0.0, 0.0), GREY, &[1.0, 1.0], true);
        assert!(GainTableMap::parse(&ok, true).is_some());
        assert!(GainTableMap::parse(&ok, false).is_none(), "the other byte order does not add up");
        assert!(GainTableMap::parse(&ok[..ok.len() - 1], true).is_none());
        assert!(GainTableMap::parse(&[], true).is_none());
        for bad in [
            blob((0, 1, 2), (0.0, 0.0), (0.0, 0.0), GREY, &[], true),
            blob((1, 1, 2), (0.0, 0.0), (0.0, 0.0), GREY, &[1.0, -1.0], true),
            blob((1, 1, 2), (0.0, 0.0), (0.0, 0.0), GREY, &[1.0, f32::NAN], true),
            blob((1, 1, 2), (f64::NAN, 0.0), (0.0, 0.0), GREY, &[1.0, 1.0], true),
            blob((1, 1, 2), (-0.5, 0.0), (0.0, 0.0), GREY, &[1.0, 1.0], true),
            // Two tables down with no spacing between them, and a gain of 1000.
            blob((2, 1, 1), (0.0, 0.0), (0.0, 0.0), GREY, &[1.0, 1.0], true),
            blob((1, 1, 2), (0.0, 0.0), (0.0, 0.0), GREY, &[1.0, 1000.0], true),
            blob((1, 1, 2), (0.0, 0.0), (0.0, 0.0), [f32::INFINITY, 0.0, 0.0, 0.0, 0.0], &[1.0, 1.0], true),
            // A header claiming 2^40 gains, without the bytes.
            blob((1 << 20, 1 << 20, 1), (0.1, 0.1), (0.0, 0.0), GREY, &[1.0], true),
        ] {
            assert!(GainTableMap::parse(&bad, true).is_none());
        }
    }

    #[test]
    fn nan_pixels_take_the_first_gain_and_never_panic() {
        let m = GainTableMap::parse(&blob((2, 2, 3), (1.0, 1.0), (0.0, 0.0), GREY, &[1.0; 12], true), true).unwrap();
        assert!(m.gain(5, 5, 4, 4, [f32::NAN; 3]).is_finite());
        let out = m.apply(5, 5, 4, 4, [f32::NAN, 0.2, 0.2]);
        assert!(out[1].is_finite());
        let _ = m.apply(0, 0, 0, 0, [f32::INFINITY, -1.0, 2.0]);
    }
}
