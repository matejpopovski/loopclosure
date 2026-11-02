//! Browser bindings. A session runs one input through two SLAM instances in
//! lock step, one with loop closure and one without, so the page can show what
//! closure buys on the same data.
//!
//! Geometry crosses the boundary as flat `f32` arrays because the page redraws
//! them every frame and typed arrays need no per-element conversion.

use loopclosure::log;
use loopclosure::occupancy::OccupancyGrid;
use loopclosure::scan::{Frame, LidarModel};
use loopclosure::se2::Pose2;
use loopclosure::sim::{default_lidar, NoiseModel, Simulator, WorldKind};
use loopclosure::slam::{Slam, SlamConfig};
use wasm_bindgen::prelude::*;

const MAP_RESOLUTION: f64 = 0.1;
const FREE_RANGE: f64 = 12.0;
const UNKNOWN: [u8; 4] = [0, 0, 0, 0];
const FREE: [u8; 4] = [24, 50, 78, 255];
const OCCUPIED: [u8; 4] = [190, 238, 255, 255];

enum Source {
    Simulated(Box<Simulator>),
    Recorded { frames: Vec<Frame>, next: usize },
}

impl Source {
    fn next(&mut self) -> Option<Frame> {
        match self {
            Source::Simulated(simulator) => simulator.next(),
            Source::Recorded { frames, next } => {
                let frame = frames.get(*next).cloned();
                *next += 1;
                frame
            }
        }
    }
}

/// One SLAM instance and its map.
struct Track {
    slam: Slam,
    map: OccupancyGrid,
    stale: bool,
    /// How many keyframes of this trajectory have been drawn into the map.
    integrated: usize,
}

impl Track {
    fn new(config: SlamConfig, lidar: LidarModel) -> Self {
        Self {
            slam: Slam::new(config, lidar),
            map: OccupancyGrid::new(MAP_RESOLUTION, [-15.0, -15.0], [15.0, 15.0]),
            stale: false,
            integrated: 0,
        }
    }

    fn step(&mut self, frame: &Frame) -> bool {
        let outcome = self.slam.step(frame);
        self.stale |= outcome.optimized;
        outcome.keyframe
    }

    /// Add new keyframes to the map. After an optimisation every keyframe has
    /// moved, so the map is rebuilt, but only on request: closures arrive in
    /// bursts and each rebuild touches every ray of the run.
    fn extend_map(&mut self) {
        for k in self.integrated..self.slam.keyframes.len() {
            self.draw(k);
        }
        self.integrated = self.slam.keyframes.len();
    }

    fn rebuild_map(&mut self) {
        self.map.clear();
        self.integrated = 0;
        self.stale = false;
        self.extend_map();
    }

    fn draw(&mut self, k: usize) {
        let pose = self.slam.trajectory()[k];
        self.map.ensure_contains([pose.x, pose.y], FREE_RANGE + 1.0);
        let scan = &self.slam.keyframes[k].scan;
        self.map.integrate(pose, &self.slam.lidar, scan, FREE_RANGE);
    }
}

fn flat(poses: impl IntoIterator<Item = Pose2>) -> Vec<f32> {
    poses
        .into_iter()
        .flat_map(|p| [p.x as f32, p.y as f32])
        .collect()
}

#[wasm_bindgen]
pub struct Session {
    source: Source,
    closed: Track,
    open: Track,
    walls: Vec<f32>,
    /// Interleaved `distance, error without, error with`, one row per keyframe.
    errors: Vec<f32>,
    path: f64,
    last_truth: Option<Pose2>,
    finished: bool,
}

#[wasm_bindgen]
impl Session {
    /// Start a simulated run. `odometry_scale` multiplies every odometry error.
    pub fn simulated(world: &str, seed: u32, odometry_scale: f64) -> Result<Session, JsError> {
        let kind = WorldKind::from_name(world).ok_or_else(|| JsError::new("unknown world name"))?;
        let built = kind.build(seed as u64);
        let walls = built
            .segments
            .iter()
            .flat_map(|s| [s.a[0] as f32, s.a[1] as f32, s.b[0] as f32, s.b[1] as f32])
            .collect();
        let noise = NoiseModel::typical().with_odometry_scale(odometry_scale);
        let lidar = default_lidar();
        let simulator = Simulator::new(built, lidar, noise, seed as u64);
        Ok(Self::start(
            Source::Simulated(Box::new(simulator)),
            lidar,
            SlamConfig::default(),
            walls,
        ))
    }

    /// Start a replay of a log, given as decompressed text.
    pub fn recorded(text: &str) -> Result<Session, JsError> {
        let parsed = log::parse(text).map_err(|e| JsError::new(&e.to_string()))?;
        Ok(Self::start(
            Source::Recorded {
                frames: parsed.frames,
                next: 0,
            },
            parsed.lidar,
            SlamConfig::default(),
            Vec::new(),
        ))
    }

    fn start(source: Source, lidar: LidarModel, config: SlamConfig, walls: Vec<f32>) -> Session {
        let mut without = config.clone();
        without.loop_closure.enabled = false;
        Session {
            source,
            closed: Track::new(config, lidar),
            open: Track::new(without, lidar),
            walls,
            errors: Vec::new(),
            path: 0.0,
            last_truth: None,
            finished: false,
        }
    }

    /// Process input until `keyframes` more keyframes exist or the input ends.
    /// Returns the number of frames consumed.
    pub fn advance(&mut self, keyframes: u32) -> u32 {
        let mut made = 0;
        let mut consumed = 0;
        while made < keyframes {
            let Some(frame) = self.source.next() else {
                if !self.finished {
                    self.finished = true;
                    self.closed.slam.finish();
                    self.open.slam.finish();
                    self.closed.stale = true;
                    self.open.stale = true;
                }
                break;
            };
            consumed += 1;
            let keyframe = self.closed.step(&frame);
            self.open.step(&frame);
            if let Some(truth) = frame.truth {
                if let Some(previous) = self.last_truth {
                    self.path += previous.distance_to(truth);
                }
                self.last_truth = Some(truth);
            }
            if keyframe {
                made += 1;
                if let Some(truth) = frame.truth {
                    let error = |track: &Track| {
                        track
                            .slam
                            .trajectory()
                            .last()
                            .map_or(0.0, |p| p.distance_to(truth))
                    };
                    self.errors.extend_from_slice(&[
                        self.path as f32,
                        error(&self.open) as f32,
                        error(&self.closed) as f32,
                    ]);
                }
            }
        }
        consumed
    }

    /// Bring both maps up to date. Rebuilding is costly, so the caller rate-limits
    /// it with `rebuild`, and otherwise new keyframes are simply added.
    pub fn refresh_maps(&mut self, rebuild: bool) {
        for track in [&mut self.closed, &mut self.open] {
            if rebuild && track.stale {
                track.rebuild_map();
            } else {
                track.extend_map();
            }
        }
    }

    pub fn finished(&self) -> bool {
        self.finished
    }

    pub fn maps_stale(&self) -> bool {
        self.closed.stale || self.open.stale
    }

    pub fn has_truth(&self) -> bool {
        self.last_truth.is_some()
    }

    pub fn keyframes(&self) -> u32 {
        self.closed.slam.keyframes.len() as u32
    }

    pub fn closures(&self) -> u32 {
        self.closed.slam.closures().count() as u32
    }

    pub fn loop_attempts(&self) -> u32 {
        self.closed.slam.stats.attempts as u32
    }

    /// True path length when the source has truth, else odometric.
    pub fn path_length(&self) -> f64 {
        if self.last_truth.is_some() {
            self.path
        } else {
            self.closed.slam.keyframes.last().map_or(0.0, |k| k.travel)
        }
    }

    pub fn errors(&self) -> Vec<f32> {
        self.errors.clone()
    }

    /// Wall segments of a simulated world, four values each.
    pub fn walls(&self) -> Vec<f32> {
        self.walls.clone()
    }

    pub fn trajectory(&self, with_closure: bool) -> Vec<f32> {
        flat(self.track(with_closure).slam.trajectory().iter().copied())
    }

    pub fn truth(&self) -> Vec<f32> {
        flat(self.closed.slam.keyframes.iter().filter_map(|k| k.truth))
    }

    pub fn odometry(&self) -> Vec<f32> {
        flat(self.closed.slam.odometry())
    }

    /// Loop closure edges, four values each.
    pub fn closure_segments(&self) -> Vec<f32> {
        let poses = self.closed.slam.trajectory();
        self.closed
            .slam
            .closures()
            .flat_map(|e| {
                let (a, b) = (poses[e.from], poses[e.to]);
                [a.x as f32, a.y as f32, b.x as f32, b.y as f32]
            })
            .collect()
    }

    pub fn map_rgba(&self, with_closure: bool) -> Vec<u8> {
        self.track(with_closure)
            .map
            .to_rgba(UNKNOWN, FREE, OCCUPIED)
    }

    pub fn map_width(&self, with_closure: bool) -> u32 {
        self.track(with_closure).map.width as u32
    }

    pub fn map_height(&self, with_closure: bool) -> u32 {
        self.track(with_closure).map.height as u32
    }

    pub fn map_origin_x(&self, with_closure: bool) -> f64 {
        self.track(with_closure).map.origin[0]
    }

    pub fn map_origin_y(&self, with_closure: bool) -> f64 {
        self.track(with_closure).map.origin[1]
    }

    pub fn map_resolution(&self) -> f64 {
        MAP_RESOLUTION
    }

    fn track(&self, with_closure: bool) -> &Track {
        if with_closure {
            &self.closed
        } else {
            &self.open
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_session_produces_consistent_buffers_for_the_page() {
        let mut session = Session::simulated("atrium", 5, 1.0).unwrap();
        assert!(!session.walls().is_empty());
        session.advance(60);
        assert_eq!(session.keyframes(), 60);
        session.refresh_maps(false);

        for with_closure in [false, true] {
            let (w, h) = (
                session.map_width(with_closure),
                session.map_height(with_closure),
            );
            assert_eq!(session.map_rgba(with_closure).len(), (w * h * 4) as usize);
            assert_eq!(session.trajectory(with_closure).len(), 120);
        }
        assert_eq!(session.errors().len(), 3 * 60);
        assert_eq!(session.truth().len(), 120);
        assert_eq!(session.odometry().len(), 120);
    }

    #[test]
    fn a_session_runs_to_the_end_of_a_short_log() {
        let text = "POS 1 0: 0 0 0\n\
                    LASER-RANGE 1 1 0 3 180.0: 100 100 100\n\
                    POS 2 0: 100 0 0\n\
                    LASER-RANGE 2 1 0 3 180.0: 100 100 100\n";
        let mut session = Session::recorded(text).unwrap();
        session.advance(100);
        assert!(session.finished());
        assert!(!session.has_truth());
        assert!(session.errors().is_empty());
        assert!((session.path_length() - 1.0).abs() < 1e-3);
    }
}
