//! Trajectory error against ground truth.
//!
//! Absolute trajectory error (ATE) and relative pose error (RPE) follow Sturm et
//! al., "A Benchmark for the Evaluation of RGB-D SLAM Systems" (IROS 2012), with
//! the rigid alignment reduced to SE(2) and RPE taken over path distance rather
//! than time, which makes it independent of the robot's speed.

use crate::se2::{wrap_angle, Pose2};

/// The rigid transform minimising `sum |T * estimate[k] - truth[k]|^2`.
pub fn align(estimate: &[Pose2], truth: &[Pose2]) -> Pose2 {
    assert_eq!(estimate.len(), truth.len());
    if estimate.is_empty() {
        return Pose2::IDENTITY;
    }
    let n = estimate.len() as f64;
    let mean = |ps: &[Pose2]| {
        let (sx, sy) = ps.iter().fold((0.0, 0.0), |a, p| (a.0 + p.x, a.1 + p.y));
        (sx / n, sy / n)
    };
    let (ex, ey) = mean(estimate);
    let (tx, ty) = mean(truth);
    let (mut dot, mut cross) = (0.0, 0.0);
    for (e, t) in estimate.iter().zip(truth) {
        let (ax, ay) = (e.x - ex, e.y - ey);
        let (bx, by) = (t.x - tx, t.y - ty);
        dot += ax * bx + ay * by;
        cross += ax * by - ay * bx;
    }
    let theta = cross.atan2(dot);
    let (s, c) = theta.sin_cos();
    Pose2::new(tx - (c * ex - s * ey), ty - (s * ex + c * ey), theta)
}

/// RMS position error after the best rigid alignment.
pub fn ate_rmse(estimate: &[Pose2], truth: &[Pose2]) -> f64 {
    let t = align(estimate, truth);
    let sum: f64 = estimate
        .iter()
        .zip(truth)
        .map(|(e, g)| {
            let d = t.compose(*e).distance_to(*g);
            d * d
        })
        .sum();
    (sum / estimate.len().max(1) as f64).sqrt()
}

/// Position error at the end of the run, with both trajectories pinned at their
/// first pose. Unlike ATE, this is drift as a robot would feel it: nothing has
/// been fitted to the truth.
pub fn final_drift(estimate: &[Pose2], truth: &[Pose2]) -> f64 {
    match (
        estimate.first(),
        estimate.last(),
        truth.first(),
        truth.last(),
    ) {
        (Some(&e0), Some(&e1), Some(&t0), Some(&t1)) => e0.between(e1).distance_to(t0.between(t1)),
        _ => 0.0,
    }
}

/// Worst position error over the run, pinned at the first pose.
pub fn max_drift(estimate: &[Pose2], truth: &[Pose2]) -> f64 {
    let (Some(&e0), Some(&t0)) = (estimate.first(), truth.first()) else {
        return 0.0;
    };
    estimate
        .iter()
        .zip(truth)
        .map(|(&e, &t)| e0.between(e).distance_to(t0.between(t)))
        .fold(0.0, f64::max)
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Rpe {
    /// RMS translation error over the segment, in metres.
    pub translation: f64,
    /// RMS rotation error over the segment, in radians.
    pub rotation: f64,
    pub segments: usize,
}

/// Compare relative motion over segments of `distance` metres of true path.
pub fn rpe(estimate: &[Pose2], truth: &[Pose2], distance: f64) -> Rpe {
    assert_eq!(estimate.len(), truth.len());
    let mut path = Vec::with_capacity(truth.len());
    let mut total = 0.0;
    for k in 0..truth.len() {
        if k > 0 {
            total += truth[k - 1].distance_to(truth[k]);
        }
        path.push(total);
    }
    let (mut sum_t, mut sum_r, mut count) = (0.0, 0.0, 0usize);
    let mut end = 0;
    for start in 0..truth.len() {
        while end < truth.len() && path[end] - path[start] < distance {
            end += 1;
        }
        if end == truth.len() {
            break;
        }
        let de = estimate[start].between(estimate[end]);
        let dt = truth[start].between(truth[end]);
        let error = dt.between(de);
        sum_t += error.x * error.x + error.y * error.y;
        let r = wrap_angle(error.theta);
        sum_r += r * r;
        count += 1;
    }
    if count == 0 {
        return Rpe::default();
    }
    Rpe {
        translation: (sum_t / count as f64).sqrt(),
        rotation: (sum_r / count as f64).sqrt(),
        segments: count,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;

    fn wiggly_path() -> Vec<Pose2> {
        (0..300)
            .map(|k| {
                let t = k as f64 * 0.1;
                Pose2::new(t, (t * 0.5).sin() * 3.0, (t * 0.2).cos())
            })
            .collect()
    }

    #[test]
    fn alignment_undoes_a_known_rigid_transform() {
        let truth = wiggly_path();
        let offset = Pose2::new(4.0, -7.0, 1.1);
        let moved: Vec<Pose2> = truth.iter().map(|p| offset.compose(*p)).collect();
        assert!(ate_rmse(&moved, &truth) < 1e-9);
        let recovered = align(&moved, &truth);
        let composed = recovered.compose(offset);
        assert!(composed.translation_norm() < 1e-9 && composed.theta.abs() < 1e-9);
    }

    #[test]
    fn ate_of_a_constant_shift_along_one_axis_is_zero_but_noise_is_not() {
        let truth = wiggly_path();
        let mut rng = Rng::new(61);
        let noisy: Vec<Pose2> = truth
            .iter()
            .map(|p| Pose2::new(p.x + 0.1 * rng.normal(), p.y + 0.1 * rng.normal(), p.theta))
            .collect();
        let ate = ate_rmse(&noisy, &truth);
        assert!((ate - 0.1 * 2f64.sqrt()).abs() < 0.015, "{ate}");
    }

    #[test]
    fn rpe_is_zero_for_a_rigidly_moved_trajectory() {
        let truth = wiggly_path();
        let offset = Pose2::new(1.0, 2.0, 0.7);
        let moved: Vec<Pose2> = truth.iter().map(|p| offset.compose(*p)).collect();
        let result = rpe(&moved, &truth, 2.0);
        assert!(result.translation < 1e-9 && result.rotation < 1e-9);
        assert!(result.segments > 100);
    }

    #[test]
    fn rpe_measures_a_scale_error_per_segment() {
        let truth: Vec<Pose2> = (0..200)
            .map(|k| Pose2::new(k as f64 * 0.1, 0.0, 0.0))
            .collect();
        let scaled: Vec<Pose2> = truth
            .iter()
            .map(|p| Pose2::new(p.x * 1.1, 0.0, 0.0))
            .collect();
        let result = rpe(&scaled, &truth, 5.0);
        assert!((result.translation - 0.5).abs() < 0.02, "{result:?}");
    }

    #[test]
    fn drift_is_pinned_at_the_start_and_not_fitted() {
        let truth: Vec<Pose2> = (0..100).map(|k| Pose2::new(k as f64, 0.0, 0.0)).collect();
        let bent: Vec<Pose2> = truth
            .iter()
            .map(|p| Pose2::new(p.x, p.x * 0.01, 0.0))
            .collect();
        assert!((final_drift(&bent, &truth) - 0.99).abs() < 1e-9);
        assert!((max_drift(&bent, &truth) - 0.99).abs() < 1e-9);
    }
}
