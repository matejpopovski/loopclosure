//! Planar rigid transforms.

use std::f64::consts::PI;

/// Wrap an angle to the half-open interval `(-pi, pi]`.
#[inline]
pub fn wrap_angle(a: f64) -> f64 {
    let wrapped = a.rem_euclid(2.0 * PI);
    if wrapped > PI {
        wrapped - 2.0 * PI
    } else {
        wrapped
    }
}

/// An element of SE(2), mapping points from a child frame into a parent frame.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Pose2 {
    pub x: f64,
    pub y: f64,
    pub theta: f64,
}

impl Pose2 {
    pub const IDENTITY: Pose2 = Pose2 {
        x: 0.0,
        y: 0.0,
        theta: 0.0,
    };

    #[inline]
    pub fn new(x: f64, y: f64, theta: f64) -> Self {
        Self { x, y, theta }
    }

    /// `self * other`: apply `other` first, then `self`.
    #[inline]
    pub fn compose(self, other: Pose2) -> Pose2 {
        let (s, c) = self.theta.sin_cos();
        Pose2 {
            x: self.x + c * other.x - s * other.y,
            y: self.y + s * other.x + c * other.y,
            theta: wrap_angle(self.theta + other.theta),
        }
    }

    #[inline]
    pub fn inverse(self) -> Pose2 {
        let (s, c) = self.theta.sin_cos();
        Pose2 {
            x: -c * self.x - s * self.y,
            y: s * self.x - c * self.y,
            theta: wrap_angle(-self.theta),
        }
    }

    /// The transform taking `self` to `other`, expressed in `self`'s frame.
    #[inline]
    pub fn between(self, other: Pose2) -> Pose2 {
        self.inverse().compose(other)
    }

    #[inline]
    pub fn transform(self, p: [f64; 2]) -> [f64; 2] {
        let (s, c) = self.theta.sin_cos();
        [self.x + c * p[0] - s * p[1], self.y + s * p[0] + c * p[1]]
    }

    #[inline]
    pub fn translation_norm(self) -> f64 {
        self.x.hypot(self.y)
    }

    #[inline]
    pub fn distance_to(self, other: Pose2) -> f64 {
        (self.x - other.x).hypot(self.y - other.y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;

    fn random_pose(rng: &mut Rng) -> Pose2 {
        Pose2::new(
            rng.uniform(-20.0, 20.0),
            rng.uniform(-20.0, 20.0),
            rng.uniform(-PI, PI),
        )
    }

    fn assert_close(a: Pose2, b: Pose2) {
        assert!((a.x - b.x).abs() < 1e-9, "{a:?} vs {b:?}");
        assert!((a.y - b.y).abs() < 1e-9, "{a:?} vs {b:?}");
        assert!(wrap_angle(a.theta - b.theta).abs() < 1e-9, "{a:?} vs {b:?}");
    }

    #[test]
    fn wrap_stays_in_range_and_preserves_the_angle() {
        for k in -1000..1000 {
            let a = k as f64 * 0.37;
            let w = wrap_angle(a);
            assert!(w > -PI - 1e-12 && w <= PI + 1e-12);
            assert!((w.sin() - a.sin()).abs() < 1e-9 && (w.cos() - a.cos()).abs() < 1e-9);
        }
    }

    #[test]
    fn group_laws_hold_for_random_poses() {
        let mut rng = Rng::new(1);
        for _ in 0..1000 {
            let (a, b, c) = (
                random_pose(&mut rng),
                random_pose(&mut rng),
                random_pose(&mut rng),
            );
            assert_close(a.compose(a.inverse()), Pose2::IDENTITY);
            assert_close(a.inverse().compose(a), Pose2::IDENTITY);
            assert_close(a.compose(b).compose(c), a.compose(b.compose(c)));
            assert_close(a.compose(a.between(b)), b);
        }
    }

    #[test]
    fn transform_is_a_group_action() {
        let mut rng = Rng::new(2);
        for _ in 0..1000 {
            let (a, b) = (random_pose(&mut rng), random_pose(&mut rng));
            let p = [rng.uniform(-5.0, 5.0), rng.uniform(-5.0, 5.0)];
            let lhs = a.compose(b).transform(p);
            let rhs = a.transform(b.transform(p));
            assert!((lhs[0] - rhs[0]).abs() < 1e-9 && (lhs[1] - rhs[1]).abs() < 1e-9);
        }
    }
}
