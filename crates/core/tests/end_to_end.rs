//! Whole-pipeline tests on simulated runs, using seeds above 1000: the settings
//! were tuned on seeds 100 to 107 and the published results use seeds 0 to 19.

use loopclosure::metrics;
use loopclosure::scan::Frame;
use loopclosure::sim::{default_lidar, NoiseModel, Simulator, WorldKind};
use loopclosure::slam::{Slam, SlamConfig};

fn frames(world: WorldKind, seed: u64) -> Vec<Frame> {
    Simulator::new(
        world.build(seed),
        default_lidar(),
        NoiseModel::typical(),
        seed,
    )
    .collect()
}

fn run(frames: &[Frame], loop_closure: bool) -> Slam {
    let mut config = SlamConfig::default();
    config.loop_closure.enabled = loop_closure;
    let mut slam = Slam::new(config, default_lidar());
    for frame in frames {
        slam.step(frame);
    }
    slam.finish();
    slam
}

#[test]
fn loop_closure_beats_tracking_alone_and_tracking_beats_odometry() {
    let data = frames(WorldKind::Atrium, 1001);
    let with = run(&data, true);
    let without = run(&data, false);
    let truth = with.truth().unwrap();

    let ate_with = metrics::ate_rmse(with.trajectory(), &truth);
    let ate_without = metrics::ate_rmse(without.trajectory(), &truth);
    let ate_odometry = metrics::ate_rmse(&with.odometry(), &truth);

    assert!(ate_with < 0.15, "with closure {ate_with}");
    assert!(ate_without > 2.0 * ate_with, "{ate_without} vs {ate_with}");
    assert!(
        ate_odometry > 2.0 * ate_without,
        "{ate_odometry} vs {ate_without}"
    );
}

#[test]
fn closures_in_the_graph_are_true_correspondences() {
    let data = frames(WorldKind::Atrium, 1002);
    let slam = run(&data, true);
    let truth = slam.truth().unwrap();
    assert!(slam.closures().count() > 20);
    for edge in slam.closures() {
        let actual = truth[edge.from].between(truth[edge.to]);
        let error = actual.between(edge.measurement).translation_norm();
        assert!(
            error < 0.5,
            "closure {} to {} is off by {error}",
            edge.from,
            edge.to
        );
    }
}

#[test]
fn a_run_is_deterministic() {
    let data = frames(WorldKind::Atrium, 1003);
    let short = &data[..800];
    let a = run(short, true);
    let b = run(short, true);
    assert_eq!(a.trajectory(), b.trajectory());
}

#[test]
fn with_closure_disabled_the_graph_is_a_chain() {
    let data = frames(WorldKind::Warehouse, 1004);
    let slam = run(&data[..600], false);
    assert_eq!(slam.closures().count(), 0);
    assert_eq!(slam.graph.edges.len(), slam.graph.poses.len() - 1);
    assert_eq!(slam.keyframes.len(), slam.graph.poses.len());
}

#[test]
fn keyframes_are_spaced_by_motion_not_by_time() {
    // A robot that does not move produces one keyframe however long it sits.
    let data = frames(WorldKind::Atrium, 1005);
    let parked: Vec<Frame> = (0..50)
        .map(|k| Frame {
            stamp: k as f64,
            ..data[0].clone()
        })
        .collect();
    let slam = run(&parked, true);
    assert_eq!(slam.keyframes.len(), 1);
}

#[test]
fn revisit_agreement_separates_a_closed_map_from_a_drifted_one() {
    let data = frames(WorldKind::Atrium, 1006);
    let with = run(&data, true);
    let without = run(&data, false);
    let pairs = with.revisit_pairs(5, 2.0);
    assert!(pairs.len() > 30, "{} pairs", pairs.len());
    let closed = with.revisit_agreement(with.trajectory(), &pairs, 0.05);
    let drifted = with.revisit_agreement(without.trajectory(), &pairs, 0.05);
    assert!(closed > 0.5, "closed {closed}");
    assert!(closed > 1.5 * drifted, "{closed} vs {drifted}");
}
