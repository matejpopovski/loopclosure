//! A deterministic simulator: a robot drives a route through a 2D world of wall
//! segments while a range finder and wheel odometry report on it, each with its
//! own noise. The point is ground truth, which no real log has.

mod worlds;

pub use worlds::WorldKind;

use crate::rng::Rng;
use crate::scan::{Frame, LidarModel, Scan};
use crate::se2::{wrap_angle, Pose2};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Segment {
    pub a: [f64; 2],
    pub b: [f64; 2],
}

#[derive(Clone, Debug)]
pub struct World {
    pub name: &'static str,
    pub segments: Vec<Segment>,
    pub route: Vec<[f64; 2]>,
    pub min: [f64; 2],
    pub max: [f64; 2],
}

impl World {
    pub fn route_length(&self) -> f64 {
        self.route
            .windows(2)
            .map(|w| (w[1][0] - w[0][0]).hypot(w[1][1] - w[0][1]))
            .sum()
    }

    /// Distance along the ray to the nearest wall, if any is in range.
    pub fn raycast(&self, segments: &[Segment], origin: [f64; 2], angle: f64, max: f64) -> f64 {
        let (s, c) = angle.sin_cos();
        let mut best = max;
        for segment in segments {
            let (ex, ey) = (segment.b[0] - segment.a[0], segment.b[1] - segment.a[1]);
            let (ox, oy) = (segment.a[0] - origin[0], segment.a[1] - origin[1]);
            let det = c * ey - s * ex;
            if det.abs() < 1e-12 {
                continue;
            }
            let t = (ox * ey - oy * ex) / det;
            let u = (ox * s - oy * c) / det;
            if t > 0.0 && t < best && (0.0..=1.0).contains(&u) {
                best = t;
            }
        }
        best
    }

    /// Walls that can intersect a circle, so a sweep need not test every wall
    /// against every beam.
    pub fn near(&self, centre: [f64; 2], radius: f64) -> Vec<Segment> {
        self.segments
            .iter()
            .filter(|seg| {
                let (lo_x, hi_x) = (seg.a[0].min(seg.b[0]), seg.a[0].max(seg.b[0]));
                let (lo_y, hi_y) = (seg.a[1].min(seg.b[1]), seg.a[1].max(seg.b[1]));
                centre[0] + radius >= lo_x
                    && centre[0] - radius <= hi_x
                    && centre[1] + radius >= lo_y
                    && centre[1] - radius <= hi_y
            })
            .copied()
            .collect()
    }
}

/// Sensor and wheel error model.
///
/// Zero-mean increment noise alone is too kind: its error grows with the square
/// root of distance. Real odometry is dominated by calibration error, which
/// grows linearly and never averages out, so each run draws a fixed scale error
/// and heading bias and keeps them for the whole drive.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NoiseModel {
    pub range_sigma: f64,
    /// Probability that a beam returns nothing.
    pub dropout: f64,
    /// Probability that a beam returns a short spurious reading.
    pub spurious: f64,
    /// Standard deviation of the per-run wheel scale error (fraction).
    pub scale_sigma: f64,
    /// Standard deviation of the per-run heading bias (rad per metre).
    pub heading_bias_sigma: f64,
    /// Increment noise on distance (m per square root of metre).
    pub translation_noise: f64,
    /// Increment noise on heading (rad per square root of metre).
    pub rotation_noise: f64,
}

impl NoiseModel {
    /// A 2D range finder of roughly SICK or Hokuyo quality on uncalibrated
    /// differential-drive wheels. These values are assumptions of this project,
    /// not fitted to a particular robot.
    pub fn typical() -> Self {
        Self {
            range_sigma: 0.015,
            dropout: 0.02,
            spurious: 0.002,
            scale_sigma: 0.012,
            heading_bias_sigma: 0.003,
            translation_noise: 0.012,
            rotation_noise: 0.008,
        }
    }

    /// Multiply every odometry term, leaving the sensor alone.
    pub fn with_odometry_scale(mut self, factor: f64) -> Self {
        self.scale_sigma *= factor;
        self.heading_bias_sigma *= factor;
        self.translation_noise *= factor;
        self.rotation_noise *= factor;
        self
    }
}

/// A 1-degree 270-degree scanner with a 10 m range. The short range is deliberate:
/// it is what keeps a long corridor from showing its far end.
pub fn default_lidar() -> LidarModel {
    LidarModel {
        beams: 271,
        angle_min: -135f64.to_radians(),
        angle_increment: 1f64.to_radians(),
        range_min: 0.05,
        range_max: 10.0,
    }
}

const STEP_SECONDS: f64 = 0.1;
const MAX_SPEED: f64 = 1.0;
const MAX_TURN_RATE: f64 = 1.2;
const ARRIVAL_RADIUS: f64 = 0.35;

pub struct Simulator {
    world: World,
    lidar: LidarModel,
    noise: NoiseModel,
    rng: Rng,
    scale_error: f64,
    heading_bias: f64,
    truth: Pose2,
    odometry: Pose2,
    waypoint: usize,
    step: usize,
}

impl Simulator {
    pub fn new(world: World, lidar: LidarModel, noise: NoiseModel, seed: u64) -> Self {
        let mut rng = Rng::new(seed ^ 0x51AB_1E5E_ED00);
        let scale_error = noise.scale_sigma * rng.normal();
        let heading_bias = noise.heading_bias_sigma * rng.normal();
        let first = world.route[0];
        let second = world.route[1];
        let start = Pose2::new(
            first[0],
            first[1],
            (second[1] - first[1]).atan2(second[0] - first[0]),
        );
        Self {
            world,
            lidar,
            noise,
            rng,
            scale_error,
            heading_bias,
            truth: start,
            odometry: start,
            waypoint: 1,
            step: 0,
        }
    }

    pub fn world(&self) -> &World {
        &self.world
    }

    pub fn lidar(&self) -> &LidarModel {
        &self.lidar
    }

    /// Advance the unicycle towards the next waypoint. Returns the commanded
    /// body-frame motion over one step.
    fn drive(&mut self) -> Option<(f64, f64)> {
        loop {
            let target = *self.world.route.get(self.waypoint)?;
            if (target[0] - self.truth.x).hypot(target[1] - self.truth.y) < ARRIVAL_RADIUS {
                self.waypoint += 1;
                continue;
            }
            let bearing = (target[1] - self.truth.y).atan2(target[0] - self.truth.x);
            let error = wrap_angle(bearing - self.truth.theta);
            let omega = (2.5 * error).clamp(-MAX_TURN_RATE, MAX_TURN_RATE);
            let speed = MAX_SPEED * error.cos().max(0.0).powi(4);
            return Some((speed * STEP_SECONDS, omega * STEP_SECONDS));
        }
    }

    fn scan(&mut self) -> Scan {
        let origin = [self.truth.x, self.truth.y];
        let near = self.world.near(origin, self.lidar.range_max);
        let lidar = self.lidar;
        let mut ranges = Vec::with_capacity(lidar.beams);
        for beam in 0..lidar.beams {
            if self.rng.next_f64() < self.noise.dropout {
                ranges.push(f32::INFINITY);
                continue;
            }
            let angle = self.truth.theta + lidar.angle(beam);
            let true_range = self.world.raycast(&near, origin, angle, lidar.range_max);
            let range = if true_range >= lidar.range_max {
                f32::INFINITY
            } else if self.rng.next_f64() < self.noise.spurious {
                self.rng.uniform(lidar.range_min, true_range) as f32
            } else {
                (true_range + self.noise.range_sigma * self.rng.normal()) as f32
            };
            ranges.push(range);
        }
        Scan { ranges }
    }
}

impl Iterator for Simulator {
    type Item = Frame;

    fn next(&mut self) -> Option<Frame> {
        let (distance, turn) = self.drive()?;
        self.truth = self.truth.compose(Pose2::new(distance, 0.0, turn));

        let root = distance.abs().sqrt();
        let measured_distance = distance * (1.0 + self.scale_error)
            + self.noise.translation_noise * root * self.rng.normal();
        let measured_turn = turn
            + self.heading_bias * distance
            + self.noise.rotation_noise * (root + 0.3 * turn.abs().sqrt()) * self.rng.normal();
        self.odometry = self
            .odometry
            .compose(Pose2::new(measured_distance, 0.0, measured_turn));

        let scan = self.scan();
        let stamp = self.step as f64 * STEP_SECONDS;
        self.step += 1;
        Some(Frame {
            stamp,
            odometry: self.odometry,
            scan,
            truth: Some(self.truth),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn segments_cross(p1: [f64; 2], p2: [f64; 2], seg: &Segment) -> bool {
        let d = |a: [f64; 2], b: [f64; 2], c: [f64; 2]| {
            (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        };
        let (d1, d2) = (d(seg.a, seg.b, p1), d(seg.a, seg.b, p2));
        let (d3, d4) = (d(p1, p2, seg.a), d(p1, p2, seg.b));
        d1 * d2 < 0.0 && d3 * d4 < 0.0
    }

    #[test]
    fn raycast_hits_the_nearest_wall() {
        let world = World {
            name: "test",
            segments: vec![
                Segment {
                    a: [5.0, -1.0],
                    b: [5.0, 1.0],
                },
                Segment {
                    a: [3.0, -1.0],
                    b: [3.0, 1.0],
                },
            ],
            route: vec![],
            min: [0.0, 0.0],
            max: [0.0, 0.0],
        };
        let all = world.segments.clone();
        assert!((world.raycast(&all, [0.0, 0.0], 0.0, 10.0) - 3.0).abs() < 1e-9);
        assert_eq!(
            world.raycast(&all, [0.0, 0.0], std::f64::consts::PI, 10.0),
            10.0
        );
        assert_eq!(world.raycast(&all, [0.0, 0.0], 1.2, 10.0), 10.0);
    }

    #[test]
    fn the_robot_never_drives_through_a_wall() {
        for kind in WorldKind::ALL {
            for seed in 0..4 {
                let world = kind.build(seed);
                let walls = world.segments.clone();
                let sim = Simulator::new(world, default_lidar(), NoiseModel::typical(), seed);
                let mut previous: Option<Pose2> = None;
                let mut frames = 0;
                for frame in sim.step_by(5) {
                    let truth = frame.truth.unwrap();
                    if let Some(last) = previous {
                        for wall in &walls {
                            assert!(
                                !segments_cross([last.x, last.y], [truth.x, truth.y], wall),
                                "{:?} seed {seed} crossed {wall:?} at {truth:?}",
                                kind
                            );
                        }
                    }
                    previous = Some(truth);
                    frames += 1;
                }
                assert!(frames > 300, "{kind:?} only ran {frames} frames");
            }
        }
    }

    #[test]
    fn the_run_visits_the_whole_route() {
        let world = WorldKind::Office.build(0);
        let goal = *world.route.last().unwrap();
        let sim = Simulator::new(world, default_lidar(), NoiseModel::typical(), 1);
        let last = sim.last().unwrap().truth.unwrap();
        assert!((last.x - goal[0]).hypot(last.y - goal[1]) < 2.0 * ARRIVAL_RADIUS);
    }

    #[test]
    fn same_seed_reproduces_the_run_and_a_new_seed_does_not() {
        let build = |seed| {
            let sim = Simulator::new(
                WorldKind::Atrium.build(seed),
                default_lidar(),
                NoiseModel::typical(),
                seed,
            );
            sim.take(20).last().unwrap()
        };
        let (a, b, c) = (build(3), build(3), build(4));
        assert_eq!(a.scan, b.scan);
        assert_eq!(a.odometry, b.odometry);
        assert_ne!(a.scan, c.scan);
    }

    #[test]
    fn odometry_drifts_linearly_because_of_calibration_error() {
        let frames: Vec<Frame> = Simulator::new(
            WorldKind::Office.build(0),
            default_lidar(),
            NoiseModel {
                translation_noise: 0.0,
                rotation_noise: 0.0,
                heading_bias_sigma: 0.0,
                ..NoiseModel::typical()
            },
            0,
        )
        .take(500)
        .collect();
        let at = |k: usize| {
            let f = &frames[k];
            f.odometry.distance_to(f.truth.unwrap())
        };
        let (half, full) = (at(249), at(499));
        assert!(full > 0.0);
        assert!((full / half - 2.0).abs() < 0.1, "{half} then {full}");
    }

    #[test]
    fn ranges_match_the_geometry_when_noise_is_off() {
        let world = WorldKind::Office.build(0);
        let noise = NoiseModel {
            range_sigma: 0.0,
            dropout: 0.0,
            spurious: 0.0,
            ..NoiseModel::typical()
        };
        let lidar = default_lidar();
        let frame = Simulator::new(world.clone(), lidar, noise, 0)
            .next()
            .unwrap();
        let truth = frame.truth.unwrap();
        let walls = world.near([truth.x, truth.y], lidar.range_max);
        for (beam, &r) in frame.scan.ranges.iter().enumerate() {
            let expected = world.raycast(
                &walls,
                [truth.x, truth.y],
                truth.theta + lidar.angle(beam),
                lidar.range_max,
            );
            if r.is_finite() {
                assert!((r as f64 - expected).abs() < 1e-5);
            } else {
                assert_eq!(expected, lidar.range_max);
            }
        }
    }
}
