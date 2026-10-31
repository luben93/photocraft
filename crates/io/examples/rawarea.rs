//! Measure where the camera's full-size JPEG sits in a raw file's sensor data: develops the
//! whole active area (no crop, no orientation), then finds the even (x, y) offset of the
//! embedded JPEG that maximises the luma correlation of several textured patches. Prints the
//! image area as `x, y, width, height` with the per-patch agreement, for `cameras.rs`.
//!
//! ```sh
//! cargo run --release -p photocraft-io --example rawarea -- FILE...
//! ```

use photocraft_codecs::{self as codecs, ChannelLayout, Format};
use photocraft_raw::{DevelopOptions, develop_sensor};

fn luma_u16(rgb: &[u16]) -> Vec<f32> {
    rgb.as_chunks::<3>().0.iter().map(|p| (0.2126 * f32::from(p[0]) + 0.7152 * f32::from(p[1]) + 0.0722 * f32::from(p[2])) / 65535.0).collect()
}

/// Normalised cross-correlation of `n × n` patches at (ax, ay) in `a` and (bx, by) in `b`.
#[allow(clippy::too_many_arguments)]
fn ncc(a: &[f32], aw: usize, ax: usize, ay: usize, b: &[f32], bw: usize, bx: usize, by: usize, n: usize) -> f64 {
    let (mut sa, mut sb, mut saa, mut sbb, mut sab) = (0.0f64, 0.0, 0.0, 0.0, 0.0);
    for j in 0..n {
        for i in 0..n {
            let (u, v) = (f64::from(a[(ay + j) * aw + ax + i]), f64::from(b[(by + j) * bw + bx + i]));
            sa += u;
            sb += v;
            saa += u * u;
            sbb += v * v;
            sab += u * v;
        }
    }
    let m = (n * n) as f64;
    let (va, vb) = (saa - sa * sa / m, sbb - sb * sb / m);
    if va <= 0.0 || vb <= 0.0 { 0.0 } else { (sab - sa * sb / m) / (va * vb).sqrt() }
}

/// Local contrast of an `n × n` patch (to pick textured patches).
fn contrast(a: &[f32], w: usize, x: usize, y: usize, n: usize) -> f32 {
    let mut s = 0.0;
    for j in 0..n {
        for i in 1..n {
            s += (a[(y + j) * w + x + i] - a[(y + j) * w + x + i - 1]).abs();
        }
    }
    s
}

fn measure(path: &str) -> Result<String, String> {
    let bytes = std::fs::read(path).map_err(|e| e.to_string())?;
    let opts = DevelopOptions { orient: false, ..DevelopOptions::default() };
    let mut sensor = photocraft_raw::decode(&bytes, &opts.limits).map_err(|e| e.to_string())?;
    let file_crop = sensor.crop;
    sensor.crop = sensor.active;
    let dev = develop_sensor(&sensor, &opts).map_err(|e| e.to_string())?;
    let p = photocraft_raw::embedded_preview(&bytes).ok_or("no preview")?;
    let jpeg = codecs::decode_as_with(Format::Jpeg, p.jpeg, &codecs::DecodeOptions { keep_orientation: true, ..Default::default() })
        .map_err(|e| e.to_string())?
        .converted(ChannelLayout::Rgb, photocraft_codecs::SampleType::U8)
        .into_owned();
    let (jw, jh) = (jpeg.width() as usize, jpeg.height() as usize);
    let (dw, dh) = (dev.width as usize, dev.height as usize);
    if jw > dw || jh > dh {
        return Err(format!("JPEG {jw}x{jh} larger than the active area {dw}x{dh}"));
    }
    // sRGB-ish luma of the JPEG; the raw's gamma-1.8 ProPhoto luma correlates well enough.
    let jl: Vec<f32> =
        jpeg.data().as_chunks::<3>().0.iter().map(|p| (0.2126 * f32::from(p[0]) + 0.7152 * f32::from(p[1]) + 0.0722 * f32::from(p[2])) / 255.0).collect();
    let dl = luma_u16(&dev.rgb);
    // The 12 most textured of 48 candidate patches.
    let n = 64;
    let mut cands: Vec<(f32, usize, usize)> = (1..8)
        .flat_map(|gy| (1..9).map(move |gx| (gx, gy)))
        .map(|(gx, gy)| {
            let (x, y) = (jw * gx / 9 - n / 2, jh * gy / 8 - n / 2);
            (contrast(&jl, jw, x, y, n), x, y)
        })
        .collect();
    cands.sort_by(|a, b| b.0.total_cmp(&a.0));
    cands.truncate(12);
    let (mx, my) = (dw - jw, dh - jh);
    let mut per_patch = Vec::new();
    let mut total = vec![0.0f64; (mx / 2 + 1) * (my / 2 + 1)];
    for &(_, x, y) in &cands {
        let mut best = (f64::MIN, 0, 0);
        for oy in (0..=my).step_by(2) {
            for ox in (0..=mx).step_by(2) {
                let c = ncc(&jl, jw, x, y, &dl, dw, x + ox, y + oy, n);
                total[(oy / 2) * (mx / 2 + 1) + ox / 2] += c;
                if c > best.0 {
                    best = (c, ox, oy);
                }
            }
        }
        per_patch.push(best);
    }
    let (i, score) = total.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map(|(i, s)| (i, *s)).unwrap_or((0, 0.0));
    let (ox, oy) = ((i % (mx / 2 + 1)) * 2, (i / (mx / 2 + 1)) * 2);
    let agree = per_patch.iter().filter(|p| (p.1, p.2) == (ox, oy)).count();
    let (ax, ay) = (sensor.active.x, sensor.active.y);
    Ok(format!(
        "{path}: make {:?} model {:?} sensor {}x{} active ({ax},{ay},{dw},{dh}) file crop ({},{},{},{}) jpeg {jw}x{jh} -> image area ({}, {}, {jw}, {jh}) mean ncc {:.3}, {agree}/{} patches agree",
        sensor.make,
        sensor.model,
        sensor.width,
        sensor.height,
        file_crop.x,
        file_crop.y,
        file_crop.width,
        file_crop.height,
        ax + ox,
        ay + oy,
        score / cands.len() as f64,
        per_patch.len()
    ))
}

fn main() {
    for f in std::env::args().skip(1) {
        match measure(&f) {
            Ok(s) => println!("{s}"),
            Err(e) => println!("{f}: error: {e}"),
        }
    }
}
