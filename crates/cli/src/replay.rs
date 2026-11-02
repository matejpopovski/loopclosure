//! Replay of a recorded log, with and without loop closure.

use crate::input::{read_text, write_json};
use crate::render;
use loopclosure::graph::RobustKernel;
use loopclosure::log;
use loopclosure::scan::Frame;
use loopclosure::slam::{Slam, SlamConfig};
use serde_json::json;
use std::path::Path;
use std::time::Instant;

/// Run a log through the pipeline, timing each step. Returns the finished SLAM
/// state and the wall time of each keyframe step, in milliseconds.
fn run(log: &log::Log, config: SlamConfig) -> (Slam, Vec<f64>, f64) {
    let mut slam = Slam::new(config, log.lidar);
    let mut steps = Vec::new();
    let start = Instant::now();
    for frame in &log.frames {
        let t = Instant::now();
        let outcome = slam.step(frame);
        if outcome.keyframe {
            steps.push(t.elapsed().as_secs_f64() * 1e3);
        }
    }
    slam.finish();
    let total = start.elapsed().as_secs_f64();
    (slam, steps, total)
}

fn percentile(sorted: &[f64], q: f64) -> f64 {
    sorted[((sorted.len() - 1) as f64 * q).round() as usize]
}

fn duration(frames: &[Frame]) -> f64 {
    match (frames.first(), frames.last()) {
        (Some(a), Some(b)) => b.stamp - a.stamp,
        _ => 0.0,
    }
}

pub fn replay(path: &Path, out: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let text = read_text(path)?;
    let start = Instant::now();
    let log = log::parse(&text)?;
    let parse_ms = start.elapsed().as_secs_f64() * 1e3;
    println!("{} scans, {} beams", log.frames.len(), log.lidar.beams);

    let (closed, mut steps, total_closed) = run(&log, SlamConfig::default());
    let mut tracking_only = SlamConfig::default();
    tracking_only.loop_closure.enabled = false;
    let (open, _, total_open) = run(&log, tracking_only);
    steps.sort_by(f64::total_cmp);

    // Both trajectories are scored on the pairs the closed map identifies as
    // revisits. The drifted trajectory cannot be trusted to find its own.
    let pairs = closed.revisit_pairs(5, 1.5);
    const TOLERANCE: f64 = 0.05;
    let agree_closed = closed.revisit_agreement(closed.trajectory(), &pairs, TOLERANCE);
    let agree_open = closed.revisit_agreement(open.trajectory(), &pairs, TOLERANCE);
    let agree_odometry = closed.revisit_agreement(&closed.odometry(), &pairs, TOLERANCE);

    let at_once = SlamConfig {
        graduated: false,
        ..Default::default()
    };
    let mut no_kernel = SlamConfig::default();
    no_kernel.optimize.loop_kernel = RobustKernel::None;
    let mut wide = SlamConfig::default();
    wide.loop_closure.radius_growth = 0.05;
    wide.loop_closure.radius_max = 8.0;
    let mut variants = Vec::new();
    for (name, config) in [
        ("kernel applied at once, no graduation", at_once),
        ("no robust kernel", no_kernel),
        ("loop search radius widened to 8 m", wide),
    ] {
        let (slam, _, _) = run(&log, config);
        variants.push(json!({
            "configuration": name,
            "closures": slam.closures().count(),
            "revisit_agreement": closed.revisit_agreement(slam.trajectory(), &pairs, TOLERANCE),
        }));
    }

    render::write_map(&closed, 0.025, &out.join("maps/intel_with_closure.png"))?;
    render::write_map(&open, 0.025, &out.join("maps/intel_without_closure.png"))?;

    let travel = closed.keyframes.last().map_or(0.0, |k| k.travel);
    write_json(
        &out.join("intel.json"),
        &json!({
            "command": "loopclosure replay data/intel.script.gz",
            "source": "intel_lab record, DSpace@MIT, hdl.handle.net/1721.1/62287, CC0 1.0",
            "hardware": "Apple M3 Max, 36 GB, one thread; other jobs were running on the machine",
            "scans": log.frames.len(),
            "beams": log.lidar.beams,
            "log_duration_s": duration(&log.frames),
            "odometric_path_m": travel,
            "keyframes": closed.keyframes.len(),
            "closures": closed.closures().count(),
            "loop_stats": {
                "attempts": closed.stats.attempts,
                "rejected_score": closed.stats.rejected_score,
                "rejected_ambiguous": closed.stats.rejected_ambiguous,
                "rejected_geometry": closed.stats.rejected_geometry,
                "rejected_inconsistent": closed.stats.rejected_inconsistent,
                "accepted": closed.stats.accepted,
                "optimizations": closed.stats.optimizations,
            },
            "tracking_failures": closed.tracking_failures,
            "ground_truth": "none: this log has no truth, so no ATE or RPE is reported",
            "revisit_agreement": {
                "definition": "share of a later pass's points within 5 cm of a wall seen on an earlier pass",
                "pairs": pairs.len(),
                "with_loop_closure": agree_closed,
                "without_loop_closure": agree_open,
                "raw_odometry": agree_odometry,
            },
            "variants": variants,
            "timing": {
                "parse_ms": parse_ms,
                "total_with_closure_s": total_closed,
                "total_without_closure_s": total_open,
                "real_time_factor": duration(&log.frames) / total_closed,
                "keyframe_step_p50_ms": percentile(&steps, 0.5),
                "keyframe_step_p99_ms": percentile(&steps, 0.99),
                "keyframe_step_max_ms": steps.last(),
            },
        }),
    )
}
