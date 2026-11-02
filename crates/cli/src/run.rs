//! One simulated run, with everything measured against ground truth.

use crate::render;
use loopclosure::metrics;
use loopclosure::scan::Frame;
use loopclosure::sim::{default_lidar, NoiseModel, Simulator, WorldKind};
use loopclosure::slam::{Slam, SlamConfig};
use serde_json::{json, Value};
use std::path::Path;

/// Closures further than this from the true relative pose count as false.
pub const FALSE_CLOSURE_METRES: f64 = 0.5;

pub struct SimRun {
    pub slam: Slam,
    /// `(true path length, position error at that moment)` per keyframe. This is
    /// what the robot believed while driving, before later closures revised it.
    pub online_error: Vec<(f64, f64)>,
}

pub fn frames(world: WorldKind, seed: u64) -> Vec<Frame> {
    Simulator::new(
        world.build(seed),
        default_lidar(),
        NoiseModel::typical(),
        seed,
    )
    .collect()
}

pub fn run_frames(frames: &[Frame], config: SlamConfig) -> SimRun {
    let mut slam = Slam::new(config, default_lidar());
    let mut online_error = Vec::new();
    let mut path = 0.0;
    let mut last = frames.first().and_then(|f| f.truth);
    for frame in frames {
        let truth = frame.truth.expect("simulated frames have truth");
        if let Some(previous) = last {
            path += previous.distance_to(truth);
        }
        last = Some(truth);
        if slam.step(frame).keyframe {
            let estimate = *slam.trajectory().last().expect("a keyframe was added");
            online_error.push((path, estimate.distance_to(truth)));
        }
    }
    slam.finish();
    SimRun { slam, online_error }
}

/// Trajectory error of one estimate, as JSON.
pub fn trajectory_metrics(
    estimate: &[loopclosure::se2::Pose2],
    truth: &[loopclosure::se2::Pose2],
) -> Value {
    let short = metrics::rpe(estimate, truth, 1.0);
    let long = metrics::rpe(estimate, truth, 10.0);
    json!({
        "ate_rmse_m": metrics::ate_rmse(estimate, truth),
        "final_drift_m": metrics::final_drift(estimate, truth),
        "max_drift_m": metrics::max_drift(estimate, truth),
        "rpe_1m_translation_m": short.translation,
        "rpe_10m_translation_m": long.translation,
        "rpe_10m_rotation_deg": long.rotation.to_degrees(),
    })
}

/// Worst error while the run was in progress.
pub fn worst_online(run: &SimRun) -> f64 {
    run.online_error.iter().map(|e| e.1).fold(0.0, f64::max)
}

pub fn false_closures(slam: &Slam) -> usize {
    let truth = slam.truth().expect("simulated runs have truth");
    slam.closures()
        .filter(|e| {
            let actual = truth[e.from].between(truth[e.to]);
            actual.between(e.measurement).translation_norm() > FALSE_CLOSURE_METRES
        })
        .count()
}

pub fn simulate(
    world: &str,
    seed: u64,
    loop_closure: bool,
    map: Option<&Path>,
) -> Result<(), Box<dyn std::error::Error>> {
    let kind = WorldKind::from_name(world).ok_or("world must be office, warehouse or atrium")?;
    let mut config = SlamConfig::default();
    config.loop_closure.enabled = loop_closure;
    let data = frames(kind, seed);
    let run = run_frames(&data, config);
    let truth = run.slam.truth().ok_or("missing truth")?;
    let report = json!({
        "world": world,
        "seed": seed,
        "loop_closure": loop_closure,
        "keyframes": run.slam.keyframes.len(),
        "closures": run.slam.closures().count(),
        "false_closures": false_closures(&run.slam),
        "worst_online_error_m": worst_online(&run),
        "slam": trajectory_metrics(run.slam.trajectory(), &truth),
        "odometry": trajectory_metrics(&run.slam.odometry(), &truth),
    });
    println!("{}", serde_json::to_string_pretty(&report)?);
    if let Some(path) = map {
        render::write_map(&run.slam, 0.05, path)?;
    }
    Ok(())
}
