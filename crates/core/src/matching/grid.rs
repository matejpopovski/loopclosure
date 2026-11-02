//! A likelihood field with a pyramid of precomputed upper bounds.

/// Level `d` of the pyramid stores, for each cell, the maximum of level 0 over
/// the `2^d`-wide square whose lowest corner is that cell. Summing a level over
/// a scan therefore bounds the score of every integer translation in that
/// square, which is what lets branch and bound discard whole regions. This is
/// the precomputed-grid construction from Hess et al., "Real-Time Loop Closure
/// in 2D LIDAR SLAM" (ICRA 2016), applied to the likelihood field of Olson,
/// "Real-Time Correlative Scan Matching" (ICRA 2009).
#[derive(Clone, Debug)]
pub struct ScoreGrid {
    pub origin: [f64; 2],
    pub resolution: f64,
    pub width: usize,
    pub height: usize,
    levels: Vec<Vec<f32>>,
}

impl ScoreGrid {
    /// Rasterise `points` as Gaussian blobs. `margin` is the free border kept
    /// around the points so a search window never reads outside the field.
    pub fn build(
        points: &[[f64; 2]],
        resolution: f64,
        sigma: f64,
        margin: f64,
        depth: usize,
    ) -> Self {
        let (mut min_x, mut min_y) = (f64::INFINITY, f64::INFINITY);
        let (mut max_x, mut max_y) = (f64::NEG_INFINITY, f64::NEG_INFINITY);
        for p in points {
            min_x = min_x.min(p[0]);
            max_x = max_x.max(p[0]);
            min_y = min_y.min(p[1]);
            max_y = max_y.max(p[1]);
        }
        if points.is_empty() {
            (min_x, min_y, max_x, max_y) = (0.0, 0.0, 0.0, 0.0);
        }
        let pad = margin + (1usize << depth) as f64 * resolution;
        let origin = [min_x - pad, min_y - pad];
        let width = ((max_x - min_x + 2.0 * pad) / resolution).ceil() as usize + 1;
        let height = ((max_y - min_y + 2.0 * pad) / resolution).ceil() as usize + 1;

        let mut base = vec![0.0f32; width * height];
        let radius = (3.0 * sigma / resolution).ceil() as isize;
        let two_sigma2 = 2.0 * sigma * sigma;
        let side = (2 * radius + 1) as usize;
        let mut kernel = vec![0.0f32; side * side];
        for dy in -radius..=radius {
            for dx in -radius..=radius {
                let d2 = ((dx * dx + dy * dy) as f64) * resolution * resolution;
                kernel[(dy + radius) as usize * side + (dx + radius) as usize] =
                    (-d2 / two_sigma2).exp() as f32;
            }
        }
        for p in points {
            let cx = ((p[0] - origin[0]) / resolution).floor() as isize;
            let cy = ((p[1] - origin[1]) / resolution).floor() as isize;
            for dy in -radius..=radius {
                let y = cy + dy;
                if y < 0 || y >= height as isize {
                    continue;
                }
                let row = y as usize * width;
                let kernel_row = (dy + radius) as usize * side;
                for dx in -radius..=radius {
                    let x = cx + dx;
                    if x < 0 || x >= width as isize {
                        continue;
                    }
                    let cell = &mut base[row + x as usize];
                    let value = kernel[kernel_row + (dx + radius) as usize];
                    if value > *cell {
                        *cell = value;
                    }
                }
            }
        }

        let mut levels = Vec::with_capacity(depth + 1);
        levels.push(base);
        for d in 1..=depth {
            // Doubling the window each level costs one pass per level, whatever
            // the window size.
            let half = 1usize << (d - 1);
            let previous = levels.last().unwrap();
            levels.push(doubled_max(previous, width, height, half));
        }
        Self {
            origin,
            resolution,
            width,
            height,
            levels,
        }
    }

    pub fn depth(&self) -> usize {
        self.levels.len() - 1
    }

    #[inline]
    pub fn level(&self, depth: usize) -> &[f32] {
        &self.levels[depth]
    }

    /// Field value at a world point on level 0, or zero outside the grid.
    pub fn value_at(&self, p: [f64; 2]) -> f32 {
        let x = ((p[0] - self.origin[0]) / self.resolution).floor() as isize;
        let y = ((p[1] - self.origin[1]) / self.resolution).floor() as isize;
        if x < 0 || y < 0 || x >= self.width as isize || y >= self.height as isize {
            return 0.0;
        }
        self.levels[0][y as usize * self.width + x as usize]
    }
}

/// `out[y][x] = max(in[y + a][x + b])` for `a, b` in `{0, half}`.
fn doubled_max(input: &[f32], width: usize, height: usize, half: usize) -> Vec<f32> {
    let mut out = input.to_vec();
    for y in 0..height {
        let row = &mut out[y * width..(y + 1) * width];
        for x in 0..width.saturating_sub(half) {
            if row[x + half] > row[x] {
                row[x] = row[x + half];
            }
        }
    }
    for y in 0..height.saturating_sub(half) {
        let (top, bottom) = out.split_at_mut((y + half) * width);
        let dst = &mut top[y * width..(y + 1) * width];
        for (d, s) in dst.iter_mut().zip(&bottom[..width]) {
            if *s > *d {
                *d = *s;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;

    #[test]
    fn field_peaks_at_points_and_decays_away_from_them() {
        let grid = ScoreGrid::build(&[[1.0, 1.0]], 0.02, 0.05, 0.5, 3);
        assert!(grid.value_at([1.0, 1.0]) > 0.95);
        let near = grid.value_at([1.05, 1.0]);
        assert!(near > 0.5 && near < 0.75, "{near}");
        assert_eq!(grid.value_at([1.5, 1.0]), 0.0);
        assert_eq!(grid.value_at([100.0, 1.0]), 0.0);
    }

    #[test]
    fn each_level_is_the_exact_windowed_maximum_of_level_zero() {
        let mut rng = Rng::new(31);
        let points: Vec<[f64; 2]> = (0..20)
            .map(|_| [rng.uniform(0.0, 3.0), rng.uniform(0.0, 3.0)])
            .collect();
        let grid = ScoreGrid::build(&points, 0.1, 0.1, 0.2, 4);
        let (w, h) = (grid.width, grid.height);
        for d in 1..=4 {
            let size = 1usize << d;
            for y in 0..h {
                for x in 0..w {
                    let mut expected = 0.0f32;
                    for yy in y..(y + size).min(h) {
                        for xx in x..(x + size).min(w) {
                            expected = expected.max(grid.level(0)[yy * w + xx]);
                        }
                    }
                    assert_eq!(grid.level(d)[y * w + x], expected, "d {d} at {x},{y}");
                }
            }
        }
    }
}
