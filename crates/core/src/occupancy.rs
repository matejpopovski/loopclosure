//! Log-odds occupancy mapping, used to render maps. The estimator itself never
//! reads this grid; its map is the set of keyframe scans at their current poses.

use crate::scan::{LidarModel, Scan};
use crate::se2::Pose2;

const LOG_ODDS_HIT: f32 = 1.2;
const LOG_ODDS_MISS: f32 = -0.25;
const LOG_ODDS_CLAMP: f32 = 6.0;

/// The display class of a cell.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Cell {
    Unknown,
    Free,
    Occupied,
}

#[derive(Clone, Debug)]
pub struct OccupancyGrid {
    pub resolution: f64,
    pub origin: [f64; 2],
    pub width: usize,
    pub height: usize,
    log_odds: Vec<f32>,
}

impl OccupancyGrid {
    pub fn new(resolution: f64, min: [f64; 2], max: [f64; 2]) -> Self {
        let width = ((max[0] - min[0]) / resolution).ceil().max(1.0) as usize;
        let height = ((max[1] - min[1]) / resolution).ceil().max(1.0) as usize;
        Self {
            resolution,
            origin: min,
            width,
            height,
            log_odds: vec![0.0; width * height],
        }
    }

    pub fn clear(&mut self) {
        self.log_odds.iter_mut().for_each(|v| *v = 0.0);
    }

    #[inline]
    fn cell_of(&self, p: [f64; 2]) -> (isize, isize) {
        (
            ((p[0] - self.origin[0]) / self.resolution).floor() as isize,
            ((p[1] - self.origin[1]) / self.resolution).floor() as isize,
        )
    }

    /// Grow, in steps, so that `p` is inside. Existing evidence is kept.
    pub fn ensure_contains(&mut self, p: [f64; 2], pad: f64) {
        let (x, y) = self.cell_of(p);
        let pad_cells = (pad / self.resolution).ceil() as isize;
        let left = (pad_cells - x).max(0);
        let bottom = (pad_cells - y).max(0);
        let right = (x + pad_cells + 1 - self.width as isize).max(0);
        let top = (y + pad_cells + 1 - self.height as isize).max(0);
        if left + right + top + bottom == 0 {
            return;
        }
        // Over-allocate so a robot driving outward does not copy the grid per scan.
        let grow = pad_cells * 3;
        let left = if left > 0 { left + grow } else { 0 } as usize;
        let bottom = if bottom > 0 { bottom + grow } else { 0 } as usize;
        let right = if right > 0 { right + grow } else { 0 } as usize;
        let top = if top > 0 { top + grow } else { 0 } as usize;
        let width = self.width + left + right;
        let height = self.height + bottom + top;
        let mut data = vec![0.0; width * height];
        for y in 0..self.height {
            let src = &self.log_odds[y * self.width..(y + 1) * self.width];
            let start = (y + bottom) * width + left;
            data[start..start + self.width].copy_from_slice(src);
        }
        self.log_odds = data;
        self.width = width;
        self.height = height;
        self.origin[0] -= left as f64 * self.resolution;
        self.origin[1] -= bottom as f64 * self.resolution;
    }

    /// Integrate one scan. Beams with no return clear free space up to `free_range`.
    pub fn integrate(&mut self, pose: Pose2, lidar: &LidarModel, scan: &Scan, free_range: f64) {
        let start = self.cell_of([pose.x, pose.y]);
        for (beam, &range) in scan.ranges.iter().enumerate() {
            let hit = lidar.is_valid(range);
            if !hit && !range.is_infinite() {
                continue;
            }
            let length = if hit { range as f64 } else { free_range };
            let (s, c) = (pose.theta + lidar.angle(beam)).sin_cos();
            let end = self.cell_of([pose.x + length * c, pose.y + length * s]);
            self.trace(start, end, hit);
        }
    }

    /// Bresenham's line, clearing every cell before the endpoint.
    fn trace(&mut self, start: (isize, isize), end: (isize, isize), hit: bool) {
        let (mut x, mut y) = start;
        let dx = (end.0 - x).abs();
        let dy = -(end.1 - y).abs();
        let sx = if x < end.0 { 1 } else { -1 };
        let sy = if y < end.1 { 1 } else { -1 };
        let mut error = dx + dy;
        let (w, h) = (self.width as isize, self.height as isize);
        loop {
            let at_end = x == end.0 && y == end.1;
            if x >= 0 && y >= 0 && x < w && y < h {
                let cell = &mut self.log_odds[(y * w + x) as usize];
                let delta = if !at_end {
                    LOG_ODDS_MISS
                } else if hit {
                    LOG_ODDS_HIT
                } else {
                    0.0
                };
                *cell = (*cell + delta).clamp(-LOG_ODDS_CLAMP, LOG_ODDS_CLAMP);
            }
            if at_end {
                return;
            }
            let doubled = 2 * error;
            if doubled >= dy {
                error += dy;
                x += sx;
            }
            if doubled <= dx {
                error += dx;
                y += sy;
            }
        }
    }

    #[inline]
    pub fn classify(&self, x: usize, y: usize) -> Cell {
        let v = self.log_odds[y * self.width + x];
        if v > 0.6 {
            Cell::Occupied
        } else if v < -0.2 {
            Cell::Free
        } else {
            Cell::Unknown
        }
    }

    /// RGBA pixels, top row first, so that +y is up on screen.
    pub fn to_rgba(&self, unknown: [u8; 4], free: [u8; 4], occupied: [u8; 4]) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.width * self.height * 4);
        for y in (0..self.height).rev() {
            for x in 0..self.width {
                out.extend_from_slice(&match self.classify(x, y) {
                    Cell::Unknown => unknown,
                    Cell::Free => free,
                    Cell::Occupied => occupied,
                });
            }
        }
        out
    }

    /// Count of occupied cells. For one world, a sharper map has fewer, because
    /// misaligned walls are drawn twice.
    pub fn occupied_cells(&self) -> usize {
        (0..self.height)
            .flat_map(|y| (0..self.width).map(move |x| (x, y)))
            .filter(|&(x, y)| self.classify(x, y) == Cell::Occupied)
            .count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lidar() -> LidarModel {
        LidarModel {
            beams: 1,
            angle_min: 0.0,
            angle_increment: 0.0,
            range_min: 0.1,
            range_max: 10.0,
        }
    }

    #[test]
    fn a_beam_clears_its_path_and_marks_its_end() {
        let mut grid = OccupancyGrid::new(0.1, [-1.0, -1.0], [5.0, 1.0]);
        let scan = Scan { ranges: vec![2.0] };
        for _ in 0..3 {
            grid.integrate(Pose2::new(0.05, 0.05, 0.0), &lidar(), &scan, 10.0);
        }
        let (x0, y0) = grid.cell_of([0.05, 0.05]);
        let (y, end) = (y0 as usize, grid.cell_of([2.05, 0.05]).0 as usize);
        for x in x0 as usize..end {
            assert_eq!(grid.classify(x, y), Cell::Free, "x {x}");
        }
        assert_eq!(grid.classify(end, y), Cell::Occupied);
        assert_eq!(grid.classify(end + 2, y), Cell::Unknown);
    }

    #[test]
    fn a_no_return_beam_clears_but_marks_nothing() {
        let mut grid = OccupancyGrid::new(0.1, [-1.0, -1.0], [5.0, 1.0]);
        let scan = Scan {
            ranges: vec![f32::INFINITY],
        };
        for _ in 0..3 {
            grid.integrate(Pose2::IDENTITY, &lidar(), &scan, 3.0);
        }
        assert_eq!(grid.occupied_cells(), 0);
        let (x, y) = grid.cell_of([1.0, 0.0]);
        assert_eq!(grid.classify(x as usize, y as usize), Cell::Free);
    }

    #[test]
    fn growing_keeps_existing_evidence_in_place() {
        let mut grid = OccupancyGrid::new(0.1, [0.0, 0.0], [1.0, 1.0]);
        let scan = Scan { ranges: vec![0.55] };
        for _ in 0..3 {
            grid.integrate(Pose2::new(0.05, 0.55, 0.0), &lidar(), &scan, 10.0);
        }
        let probe = [0.65, 0.55];
        let (x, y) = grid.cell_of(probe);
        assert_eq!(grid.classify(x as usize, y as usize), Cell::Occupied);

        grid.ensure_contains([-3.0, 4.0], 0.5);
        let (x, y) = grid.cell_of(probe);
        assert_eq!(grid.classify(x as usize, y as usize), Cell::Occupied);
        let (x, y) = grid.cell_of([-3.0, 4.0]);
        assert!(x >= 0 && y >= 0 && x < grid.width as isize && y < grid.height as isize);
    }

    #[test]
    fn image_is_flipped_so_north_is_up() {
        let mut grid = OccupancyGrid::new(1.0, [0.0, 0.0], [2.0, 2.0]);
        grid.log_odds[2] = 5.0; // cell (0, 1), the top-left on screen
        let image = grid.to_rgba([0; 4], [1; 4], [9; 4]);
        assert_eq!(image[0], 9);
        assert_eq!(image.len(), 16);
    }
}
