//! Point-to-line ICP against an oriented reference cloud.

use crate::linalg::dense::solve3;
use crate::linalg::Mat3;
use crate::se2::{wrap_angle, Pose2};

/// Reference points and normals in the world frame, bucketed on a uniform grid
/// whose cell equals the largest correspondence distance, so a nearest-neighbour
/// query only touches the surrounding 3x3 cells.
#[derive(Clone, Debug)]
pub struct ReferenceCloud {
    points: Vec<[f64; 2]>,
    normals: Vec<[f64; 2]>,
    origin: [f64; 2],
    cell: f64,
    width: usize,
    height: usize,
    /// Compressed rows: `order[start[c]..start[c + 1]]` are the points of cell `c`.
    start: Vec<u32>,
    order: Vec<u32>,
}

impl ReferenceCloud {
    pub fn new(points: Vec<[f64; 2]>, normals: Vec<[f64; 2]>, cell: f64) -> Self {
        assert_eq!(points.len(), normals.len());
        let (mut min_x, mut min_y) = (f64::INFINITY, f64::INFINITY);
        let (mut max_x, mut max_y) = (f64::NEG_INFINITY, f64::NEG_INFINITY);
        for p in &points {
            min_x = min_x.min(p[0]);
            max_x = max_x.max(p[0]);
            min_y = min_y.min(p[1]);
            max_y = max_y.max(p[1]);
        }
        if points.is_empty() {
            (min_x, min_y, max_x, max_y) = (0.0, 0.0, 0.0, 0.0);
        }
        let origin = [min_x, min_y];
        let width = ((max_x - min_x) / cell).floor() as usize + 1;
        let height = ((max_y - min_y) / cell).floor() as usize + 1;
        let cell_of = |p: &[f64; 2]| {
            let x = ((p[0] - origin[0]) / cell).floor() as usize;
            let y = ((p[1] - origin[1]) / cell).floor() as usize;
            y * width + x
        };
        let mut start = vec![0u32; width * height + 1];
        for p in &points {
            start[cell_of(p) + 1] += 1;
        }
        for c in 0..width * height {
            start[c + 1] += start[c];
        }
        let mut next = start.clone();
        let mut order = vec![0u32; points.len()];
        for (i, p) in points.iter().enumerate() {
            let c = cell_of(p);
            order[next[c] as usize] = i as u32;
            next[c] += 1;
        }
        Self {
            points,
            normals,
            origin,
            cell,
            width,
            height,
            start,
            order,
        }
    }

    pub fn len(&self) -> usize {
        self.points.len()
    }

    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    /// Distance from `q` to the line through its nearest reference point, or
    /// `None` when no reference point is within `max_distance`.
    pub fn line_distance(&self, q: [f64; 2], max_distance: f64) -> Option<f64> {
        let index = self.nearest(q, max_distance)?;
        let (m, n) = (self.points[index], self.normals[index]);
        Some((n[0] * (q[0] - m[0]) + n[1] * (q[1] - m[1])).abs())
    }

    /// Nearest reference point within `max_distance`, which must not exceed the
    /// cell size.
    fn nearest(&self, q: [f64; 2], max_distance: f64) -> Option<usize> {
        let cx = ((q[0] - self.origin[0]) / self.cell).floor() as isize;
        let cy = ((q[1] - self.origin[1]) / self.cell).floor() as isize;
        let mut best = None;
        let mut best_d2 = max_distance * max_distance;
        for y in (cy - 1).max(0)..=(cy + 1).min(self.height as isize - 1) {
            for x in (cx - 1).max(0)..=(cx + 1).min(self.width as isize - 1) {
                let c = y as usize * self.width + x as usize;
                for &i in &self.order[self.start[c] as usize..self.start[c + 1] as usize] {
                    let p = self.points[i as usize];
                    let d2 = (p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2);
                    if d2 < best_d2 {
                        best_d2 = d2;
                        best = Some(i as usize);
                    }
                }
            }
        }
        best
    }
}

#[derive(Clone, Copy, Debug)]
pub struct IcpOptions {
    pub max_iterations: usize,
    /// Correspondence gate in the first half of the iterations.
    pub coarse_distance: f64,
    /// Correspondence gate afterwards, and when counting inliers.
    pub fine_distance: f64,
    /// Residual at which the Huber loss starts down-weighting a point.
    pub huber: f64,
}

impl Default for IcpOptions {
    fn default() -> Self {
        Self {
            max_iterations: 16,
            coarse_distance: 0.30,
            fine_distance: 0.15,
            huber: 0.05,
        }
    }
}

/// A Gaussian prior on the pose, in the world frame.
///
/// Tracking passes the odometry prediction here. Where the walls constrain the
/// pose the scan outvotes it. Where they do not, as along a corridor, the pose
/// stays on the prediction instead of following whichever way the noise happens
/// to pull it.
#[derive(Clone, Copy, Debug)]
pub struct PosePrior {
    pub pose: Pose2,
    pub information: Mat3,
    /// Weight of one scan residual, in inverse square metres.
    pub point_weight: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct IcpResult {
    pub pose: Pose2,
    /// `J^T J` at the solution, for unit residual variance. Its weak directions
    /// are the directions the geometry does not constrain.
    pub hessian: Mat3,
    /// Fraction of the scan with a correspondence at the fine gate.
    pub inlier_ratio: f64,
    /// RMS point-to-line residual of the inliers, in metres.
    pub rms: f64,
    pub iterations: usize,
}

struct Linearised {
    hessian: Mat3,
    gradient: [f64; 3],
    inliers: usize,
    squared_residual: f64,
}

fn linearise(
    reference: &ReferenceCloud,
    points: &[[f64; 2]],
    pose: Pose2,
    gate: f64,
    huber: f64,
) -> Linearised {
    let (s, c) = pose.theta.sin_cos();
    let mut out = Linearised {
        hessian: [[0.0; 3]; 3],
        gradient: [0.0; 3],
        inliers: 0,
        squared_residual: 0.0,
    };
    for p in points {
        // The rotated point, relative to the sensor, drives the angular Jacobian.
        let rx = c * p[0] - s * p[1];
        let ry = s * p[0] + c * p[1];
        let q = [pose.x + rx, pose.y + ry];
        let Some(index) = reference.nearest(q, gate) else {
            continue;
        };
        let m = reference.points[index];
        let n = reference.normals[index];
        let residual = n[0] * (q[0] - m[0]) + n[1] * (q[1] - m[1]);
        let j = [n[0], n[1], n[1] * rx - n[0] * ry];
        let weight = if residual.abs() <= huber {
            1.0
        } else {
            huber / residual.abs()
        };
        for a in 0..3 {
            for b in 0..3 {
                out.hessian[a][b] += weight * j[a] * j[b];
            }
            out.gradient[a] -= weight * j[a] * residual;
        }
        out.inliers += 1;
        out.squared_residual += residual * residual;
    }
    out
}

/// Refine `initial` by point-to-line ICP. `points` are in the sensor frame.
pub fn refine(
    reference: &ReferenceCloud,
    points: &[[f64; 2]],
    initial: Pose2,
    prior: Option<&PosePrior>,
    options: &IcpOptions,
) -> IcpResult {
    let mut pose = initial;
    let mut iterations = 0;
    for iteration in 0..options.max_iterations {
        let gate = if iteration < options.max_iterations / 2 {
            options.coarse_distance
        } else {
            options.fine_distance
        };
        let mut system = linearise(reference, points, pose, gate, options.huber);
        if system.inliers < 3 {
            break;
        }
        if let Some(prior) = prior {
            let error = [
                prior.pose.x - pose.x,
                prior.pose.y - pose.y,
                wrap_angle(prior.pose.theta - pose.theta),
            ];
            for a in 0..3 {
                for b in 0..3 {
                    system.hessian[a][b] =
                        prior.point_weight * system.hessian[a][b] + prior.information[a][b];
                }
                system.gradient[a] = prior.point_weight * system.gradient[a]
                    + (0..3)
                        .map(|b| prior.information[a][b] * error[b])
                        .sum::<f64>();
            }
        } else {
            // Without a prior, a small ridge holds an unconstrained direction
            // where the correlative stage left it.
            let ridge = 1e-3 * (system.hessian[0][0] + system.hessian[1][1]).max(1.0);
            for k in 0..3 {
                system.hessian[k][k] += ridge;
            }
        }
        let Some(step) = solve3(system.hessian, system.gradient) else {
            break;
        };
        pose.x += step[0];
        pose.y += step[1];
        pose.theta = wrap_angle(pose.theta + step[2]);
        iterations += 1;
        if iteration >= options.max_iterations / 2
            && step[0].hypot(step[1]) < 1e-4
            && step[2].abs() < 1e-5
        {
            break;
        }
    }

    let last = linearise(
        reference,
        points,
        pose,
        options.fine_distance,
        options.huber,
    );
    IcpResult {
        pose,
        hessian: last.hessian,
        inlier_ratio: last.inliers as f64 / points.len().max(1) as f64,
        rms: (last.squared_residual / last.inliers.max(1) as f64).sqrt(),
        iterations,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;

    /// A rectangular room, sampled densely, with outward normals.
    fn room() -> ReferenceCloud {
        let (mut points, mut normals) = (Vec::new(), Vec::new());
        for i in 0..=200 {
            let t = i as f64 * 0.05;
            points.push([t, 0.0]);
            normals.push([0.0, 1.0]);
            points.push([t, 6.0]);
            normals.push([0.0, 1.0]);
        }
        for i in 0..=120 {
            let t = i as f64 * 0.05;
            points.push([0.0, t]);
            normals.push([1.0, 0.0]);
            points.push([10.0, t]);
            normals.push([1.0, 0.0]);
        }
        ReferenceCloud::new(points, normals, 0.3)
    }

    fn scan_of_room(pose: Pose2, rng: &mut Rng, noise: f64) -> Vec<[f64; 2]> {
        let inverse = pose.inverse();
        let mut scan = Vec::new();
        for i in 0..100 {
            let t = i as f64 * 0.1 + 0.03;
            let walls = [[t, 0.0], [t, 6.0], [0.0, t * 0.6], [10.0, t * 0.6]];
            for w in walls {
                let wall = [w[0] + noise * rng.normal(), w[1] + noise * rng.normal()];
                scan.push(inverse.transform(wall));
            }
        }
        scan
    }

    #[test]
    fn converges_to_the_true_pose_from_a_small_offset() {
        let mut rng = Rng::new(51);
        let reference = room();
        for _ in 0..20 {
            let truth = Pose2::new(
                rng.uniform(2.0, 8.0),
                rng.uniform(1.5, 4.5),
                rng.uniform(-3.0, 3.0),
            );
            let scan = scan_of_room(truth, &mut rng, 0.0);
            let start = Pose2::new(
                truth.x + rng.uniform(-0.15, 0.15),
                truth.y + rng.uniform(-0.15, 0.15),
                truth.theta + rng.uniform(-0.03, 0.03),
            );
            let result = refine(&reference, &scan, start, None, &IcpOptions::default());
            assert!(result.pose.distance_to(truth) < 0.002, "{:?}", result);
            assert!(wrap_angle(result.pose.theta - truth.theta).abs() < 2e-4);
            assert!(result.inlier_ratio > 0.99);
            // Returns near a corner pair with the other wall, so this is not zero.
            assert!(result.rms < 0.03, "rms {}", result.rms);
        }
    }

    #[test]
    fn a_corridor_reports_a_weak_direction_along_its_axis() {
        // Two long parallel walls constrain y and heading but not x.
        let (mut points, mut normals) = (Vec::new(), Vec::new());
        for i in 0..=600 {
            let t = i as f64 * 0.05;
            for y in [-1.0, 1.0] {
                points.push([t, y]);
                normals.push([0.0, 1.0]);
            }
        }
        let reference = ReferenceCloud::new(points, normals, 0.3);
        let scan: Vec<[f64; 2]> = (-60..60)
            .flat_map(|i| {
                let x = i as f64 * 0.1;
                [[x, 1.0], [x, -1.0]]
            })
            .collect();
        let result = refine(
            &reference,
            &scan,
            Pose2::new(15.0, 0.02, 0.0),
            None,
            &IcpOptions::default(),
        );
        assert!(result.pose.y.abs() < 1e-3);
        assert!((result.pose.x - 15.0).abs() < 1e-3, "x must stay put");
        assert!(result.hessian[0][0] < 1e-6 * result.hessian[1][1]);
    }

    #[test]
    fn a_prior_holds_the_weak_direction_but_not_the_strong_one() {
        let (mut points, mut normals) = (Vec::new(), Vec::new());
        for i in 0..=600 {
            let t = i as f64 * 0.05;
            for y in [-1.0, 1.0] {
                points.push([t, y]);
                normals.push([0.0, 1.0]);
            }
        }
        let reference = ReferenceCloud::new(points, normals, 0.3);
        let scan: Vec<[f64; 2]> = (-60..60)
            .flat_map(|i| {
                let x = i as f64 * 0.1;
                [[x, 1.0], [x, -1.0]]
            })
            .collect();
        // The prediction is 8 cm off across the corridor and 8 cm along it, and
        // the search started somewhere else along the axis.
        let prior = PosePrior {
            pose: Pose2::new(15.08, 0.08, 0.0),
            information: [[400.0, 0.0, 0.0], [0.0, 400.0, 0.0], [0.0, 0.0, 2500.0]],
            point_weight: 1.0 / (0.03 * 0.03 * 6.0),
        };
        let result = refine(
            &reference,
            &scan,
            Pose2::new(15.03, 0.05, 0.0),
            Some(&prior),
            &IcpOptions::default(),
        );
        assert!((result.pose.x - 15.08).abs() < 1e-3, "{:?}", result.pose);
        assert!(result.pose.y.abs() < 2e-3, "{:?}", result.pose);
    }

    #[test]
    fn inlier_ratio_exposes_a_scan_from_somewhere_else() {
        let reference = room();
        let scan: Vec<[f64; 2]> = (0..100).map(|i| [30.0 + i as f64 * 0.1, 40.0]).collect();
        let result = refine(
            &reference,
            &scan,
            Pose2::IDENTITY,
            None,
            &IcpOptions::default(),
        );
        assert_eq!(result.inlier_ratio, 0.0);
    }

    #[test]
    fn nearest_neighbour_matches_a_linear_scan() {
        let mut rng = Rng::new(52);
        let points: Vec<[f64; 2]> = (0..500)
            .map(|_| [rng.uniform(-4.0, 4.0), rng.uniform(-4.0, 4.0)])
            .collect();
        let normals = vec![[1.0, 0.0]; 500];
        let cloud = ReferenceCloud::new(points.clone(), normals, 0.5);
        for _ in 0..500 {
            let q = [rng.uniform(-5.0, 5.0), rng.uniform(-5.0, 5.0)];
            let oracle = points
                .iter()
                .enumerate()
                .map(|(i, p)| (((p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2)).sqrt(), i))
                .filter(|(d, _)| *d < 0.5)
                .min_by(|a, b| a.0.total_cmp(&b.0))
                .map(|(_, i)| i);
            assert_eq!(cloud.nearest(q, 0.5), oracle);
        }
    }
}
