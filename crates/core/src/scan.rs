//! Range scans and the per-scan geometry derived from them.

use crate::se2::Pose2;

/// Beam geometry of a planar range finder.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LidarModel {
    pub beams: usize,
    pub angle_min: f64,
    pub angle_increment: f64,
    pub range_min: f64,
    pub range_max: f64,
}

impl LidarModel {
    #[inline]
    pub fn angle(&self, beam: usize) -> f64 {
        self.angle_min + beam as f64 * self.angle_increment
    }

    #[inline]
    pub fn is_valid(&self, range: f32) -> bool {
        let r = range as f64;
        r.is_finite() && r >= self.range_min && r < self.range_max
    }
}

/// One sweep. A beam with no return holds `f32::INFINITY`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Scan {
    pub ranges: Vec<f32>,
}

impl Scan {
    /// Cartesian returns in the sensor frame, with their beam index.
    pub fn points(&self, lidar: &LidarModel) -> Vec<(usize, [f64; 2])> {
        self.ranges
            .iter()
            .enumerate()
            .filter(|(_, &r)| lidar.is_valid(r))
            .map(|(beam, &r)| {
                let (s, c) = lidar.angle(beam).sin_cos();
                (beam, [r as f64 * c, r as f64 * s])
            })
            .collect()
    }
}

/// One input to the SLAM front end, simulated or replayed.
#[derive(Clone, Debug)]
pub struct Frame {
    pub stamp: f64,
    pub odometry: Pose2,
    pub scan: Scan,
    /// Ground truth, when the source has any. Used only for evaluation.
    pub truth: Option<Pose2>,
}

/// Points of a scan with estimated surface normals, in the sensor frame.
#[derive(Clone, Debug, Default)]
pub struct OrientedPoints {
    pub points: Vec<[f64; 2]>,
    pub normals: Vec<[f64; 2]>,
}

/// Fit a line through each return and its neighbouring beams.
///
/// Normals come from beam adjacency rather than a spatial search because a scan
/// is already ordered by angle, which makes this linear and spares a neighbour
/// index per keyframe. Returns with too few nearby beams, or whose neighbourhood
/// is not line-like, are dropped: without a normal they carry no point-to-line
/// constraint, and they are usually the mixed pixels at depth discontinuities.
pub fn estimate_normals(returns: &[(usize, [f64; 2])], radius: f64) -> OrientedPoints {
    const HALF_WINDOW: usize = 4;
    let mut out = OrientedPoints::default();
    let r2 = radius * radius;
    for (index, &(_, p)) in returns.iter().enumerate() {
        let lo = index.saturating_sub(HALF_WINDOW);
        let hi = (index + HALF_WINDOW + 1).min(returns.len());
        let (mut n, mut sx, mut sy) = (0.0, 0.0, 0.0);
        let (mut sxx, mut sxy, mut syy) = (0.0, 0.0, 0.0);
        for &(_, q) in &returns[lo..hi] {
            let (dx, dy) = (q[0] - p[0], q[1] - p[1]);
            if dx * dx + dy * dy <= r2 {
                n += 1.0;
                sx += q[0];
                sy += q[1];
                sxx += q[0] * q[0];
                sxy += q[0] * q[1];
                syy += q[1] * q[1];
            }
        }
        if n < 3.0 {
            continue;
        }
        let cxx = sxx / n - sx * sx / (n * n);
        let cxy = sxy / n - sx * sy / (n * n);
        let cyy = syy / n - sy * sy / (n * n);
        // Eigenvalues of the 2x2 scatter matrix, in closed form.
        let half_trace = 0.5 * (cxx + cyy);
        let spread = (0.25 * (cxx - cyy).powi(2) + cxy * cxy).sqrt();
        let (major, minor) = (half_trace + spread, half_trace - spread);
        if major <= 0.0 || minor > 0.1 * major {
            continue;
        }
        // Eigenvector of the larger eigenvalue, then rotate a quarter turn.
        let (ex, ey) = if cxy.abs() > 1e-12 {
            (major - cyy, cxy)
        } else if cxx >= cyy {
            (1.0, 0.0)
        } else {
            (0.0, 1.0)
        };
        let norm = ex.hypot(ey);
        out.points.push(p);
        out.normals.push([-ey / norm, ex / norm]);
    }
    out
}

/// Keep the first point in each square cell, so dense close-range returns do not
/// dominate a match.
pub fn voxel_filter(points: &[[f64; 2]], cell: f64) -> Vec<[f64; 2]> {
    let mut keys: Vec<(i32, i32, usize)> = points
        .iter()
        .enumerate()
        .map(|(i, p)| {
            (
                (p[0] / cell).floor() as i32,
                (p[1] / cell).floor() as i32,
                i,
            )
        })
        .collect();
    keys.sort_unstable();
    keys.dedup_by_key(|k| (k.0, k.1));
    let mut indices: Vec<usize> = keys.into_iter().map(|k| k.2).collect();
    indices.sort_unstable();
    indices.into_iter().map(|i| points[i]).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lidar() -> LidarModel {
        LidarModel {
            beams: 181,
            angle_min: -std::f64::consts::FRAC_PI_2,
            angle_increment: std::f64::consts::PI / 180.0,
            range_min: 0.1,
            range_max: 10.0,
        }
    }

    #[test]
    fn invalid_beams_produce_no_points() {
        let model = lidar();
        let mut ranges = vec![2.0f32; 181];
        ranges[0] = f32::INFINITY;
        ranges[1] = 0.01;
        ranges[2] = 10.0;
        ranges[3] = f32::NAN;
        let points = Scan { ranges }.points(&model);
        assert_eq!(points.len(), 177);
        assert_eq!(points[0].0, 4);
    }

    #[test]
    fn beam_zero_points_along_angle_min() {
        let model = lidar();
        let scan = Scan {
            ranges: vec![2.0; 181],
        };
        let p = scan.points(&model)[0].1;
        assert!(p[0].abs() < 1e-12 && (p[1] + 2.0).abs() < 1e-12);
    }

    #[test]
    fn normals_are_perpendicular_to_a_wall_at_any_slope() {
        for angle in [0.0f64, 0.3, 1.2, 1.6, 2.5] {
            let (s, c) = angle.sin_cos();
            let returns: Vec<(usize, [f64; 2])> = (0..30)
                .map(|i| {
                    let t = i as f64 * 0.05;
                    (i, [1.0 + c * t, 2.0 + s * t])
                })
                .collect();
            let oriented = estimate_normals(&returns, 0.3);
            assert_eq!(oriented.points.len(), 30);
            for n in &oriented.normals {
                assert!((n[0] * c + n[1] * s).abs() < 1e-6, "angle {angle}");
                assert!((n[0].hypot(n[1]) - 1.0).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn isolated_and_corner_points_are_dropped() {
        let mut returns: Vec<(usize, [f64; 2])> = vec![(0, [50.0, 50.0])];
        for i in 0..10 {
            returns.push((i + 1, [i as f64 * 0.05, 0.0]));
        }
        for i in 1..10 {
            returns.push((i + 10, [0.45, i as f64 * 0.05]));
        }
        let oriented = estimate_normals(&returns, 0.2);
        assert!(oriented.points.iter().all(|p| p[0] < 40.0));
        // The corner neighbourhood is not a line.
        assert!(oriented
            .points
            .iter()
            .all(|p| (p[0] - 0.45).abs() + p[1].abs() > 0.01));
    }

    #[test]
    fn voxel_filter_keeps_one_point_per_cell_in_input_order() {
        let points = [
            [0.01, 0.01],
            [0.02, 0.02],
            [1.0, 1.0],
            [0.03, 0.0],
            [-0.5, 0.0],
        ];
        assert_eq!(
            voxel_filter(&points, 0.1),
            vec![[0.01, 0.01], [1.0, 1.0], [-0.5, 0.0]]
        );
    }
}
