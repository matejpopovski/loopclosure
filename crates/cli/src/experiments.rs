//! The simulator experiments. Each writes one JSON file with per-run values and
//! summaries, so a README table can be traced to its source.
//!
//! Settings were tuned on seeds 100 to 107. Everything here runs on seeds counted
//! up from 0, which played no part in the tuning.

use crate::input::write_json;
use crate::run::{false_closures, frames, run_frames, trajectory_metrics, worst_online};
use loopclosure::graph::{Edge, EdgeKind, OptimizeOptions, RobustKernel};
use loopclosure::metrics;
use loopclosure::rng::Rng;
use loopclosure::se2::Pose2;
use loopclosure::sim::{NoiseModel, WorldKind};
use loopclosure::slam::SlamConfig;
use serde_json::{json, Value};
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

type Outcome = Result<(), Box<dyn std::error::Error>>;

/// Run `work` over `items` on up to `jobs` threads, keeping input order.
fn parallel<T: Sync, R: Send>(items: &[T], jobs: usize, work: impl Fn(&T) -> R + Sync) -> Vec<R> {
    let next = AtomicUsize::new(0);
    let results: Mutex<Vec<Option<R>>> = Mutex::new(items.iter().map(|_| None).collect());
    std::thread::scope(|scope| {
        for _ in 0..jobs.max(1) {
            scope.spawn(|| loop {
                let index = next.fetch_add(1, Ordering::Relaxed);
                let Some(item) = items.get(index) else {
                    return;
                };
                let result = work(item);
                results.lock().expect("no worker panicked")[index] = Some(result);
            });
        }
    });
    results
        .into_inner()
        .expect("no worker panicked")
        .into_iter()
        .map(|r| r.expect("every item was processed"))
        .collect()
}

/// Mean, standard deviation, median and maximum of a sample.
fn summary(values: &[f64]) -> Value {
    let n = values.len().max(1) as f64;
    let mean = values.iter().sum::<f64>() / n;
    let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0).max(1.0);
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let median = sorted[sorted.len() / 2];
    json!({
        "mean": mean,
        "std": variance.sqrt(),
        "median": median,
        "max": sorted.last(),
    })
}

fn field(run: &Value, path: &[&str]) -> f64 {
    path.iter()
        .fold(run, |value, key| &value[*key])
        .as_f64()
        .unwrap_or(f64::NAN)
}

fn pick(runs: &[&Value], path: &[&str]) -> Value {
    let values: Vec<f64> = runs.iter().map(|r| field(r, path)).collect();
    summary(&values)
}

pub fn drift(seeds: u64, jobs: usize, out: &Path) -> Outcome {
    let tasks: Vec<(WorldKind, u64)> = WorldKind::ALL
        .into_iter()
        .flat_map(|w| (0..seeds).map(move |s| (w, s)))
        .collect();

    let results = parallel(&tasks, jobs, |&(world, seed)| {
        let data = frames(world, seed);
        let mut without = SlamConfig::default();
        without.loop_closure.enabled = false;
        let closed = run_frames(&data, SlamConfig::default());
        let open = run_frames(&data, without);
        let truth_closed = closed.slam.truth().expect("simulated");
        let truth_open = open.slam.truth().expect("simulated");
        let path_m: f64 = truth_closed
            .windows(2)
            .map(|w| w[0].distance_to(w[1]))
            .sum();
        let run = json!({
            "world": world.name(),
            "seed": seed,
            "path_m": path_m,
            "keyframes": closed.slam.keyframes.len(),
            "closures": closed.slam.closures().count(),
            "false_closures": false_closures(&closed.slam),
            "tracking_failures": closed.slam.tracking_failures,
            "odometry": trajectory_metrics(&closed.slam.odometry(), &truth_closed),
            "without_loop_closure": trajectory_metrics(open.slam.trajectory(), &truth_open),
            "with_loop_closure": trajectory_metrics(closed.slam.trajectory(), &truth_closed),
            "worst_online_without_m": worst_online(&open),
            "worst_online_with_m": worst_online(&closed),
        });
        let curves = (seed == 0).then(|| {
            let thin = |e: &[(f64, f64)]| -> Vec<[f64; 2]> {
                e.iter().step_by(4).map(|&(d, v)| [d, v]).collect()
            };
            json!({
                "world": world.name(),
                "without": thin(&open.online_error),
                "with": thin(&closed.online_error),
            })
        });
        (run, curves)
    });

    let runs: Vec<&Value> = results.iter().map(|r| &r.0).collect();
    let mut summaries = Vec::new();
    let worlds: Vec<&str> = WorldKind::ALL.iter().map(|w| w.name()).collect();
    for scope in worlds.iter().copied().chain(["all"]) {
        let group: Vec<&Value> = runs
            .iter()
            .copied()
            .filter(|r| scope == "all" || r["world"] == scope)
            .collect();
        let mode = |name: &str| {
            json!({
                "ate_rmse_m": pick(&group, &[name, "ate_rmse_m"]),
                "max_drift_m": pick(&group, &[name, "max_drift_m"]),
                "final_drift_m": pick(&group, &[name, "final_drift_m"]),
                "rpe_1m_translation_m": pick(&group, &[name, "rpe_1m_translation_m"]),
                "rpe_10m_translation_m": pick(&group, &[name, "rpe_10m_translation_m"]),
                "rpe_10m_rotation_deg": pick(&group, &[name, "rpe_10m_rotation_deg"]),
            })
        };
        summaries.push(json!({
            "scope": scope,
            "runs": group.len(),
            "path_m": pick(&group, &["path_m"]),
            "closures": pick(&group, &["closures"]),
            "false_closures_total": group.iter().map(|r| field(r, &["false_closures"])).sum::<f64>(),
            "odometry": mode("odometry"),
            "without_loop_closure": mode("without_loop_closure"),
            "with_loop_closure": mode("with_loop_closure"),
            "worst_online_without_m": pick(&group, &["worst_online_without_m"]),
            "worst_online_with_m": pick(&group, &["worst_online_with_m"]),
        }));
    }
    let noise = NoiseModel::typical();
    write_json(
        &out.join("drift.json"),
        &json!({
            "command": format!("loopclosure drift --seeds {seeds}"),
            "source": "simulator; all values measured against simulator ground truth",
            "seeds": format!("0..{seeds}"),
            "noise": {
                "range_sigma_m": noise.range_sigma,
                "dropout": noise.dropout,
                "spurious": noise.spurious,
                "wheel_scale_sigma": noise.scale_sigma,
                "heading_bias_sigma_rad_per_m": noise.heading_bias_sigma,
                "translation_noise_m_per_sqrt_m": noise.translation_noise,
                "rotation_noise_rad_per_sqrt_m": noise.rotation_noise,
            },
            "summary": summaries,
            "runs": runs,
        }),
    )?;
    let curves: Vec<&Value> = results.iter().filter_map(|r| r.1.as_ref()).collect();
    write_json(
        &out.join("drift_curves.json"),
        &json!({ "seed": 0, "curves": curves }),
    )
}

/// Summarise one configuration over a set of runs.
fn config_row(name: &str, tasks: &[(WorldKind, u64)], jobs: usize, config: &SlamConfig) -> Value {
    let results = parallel(tasks, jobs, |&(world, seed)| {
        let data = frames(world, seed);
        let run = run_frames(&data, config.clone());
        let truth = run.slam.truth().expect("simulated");
        (
            world.name(),
            metrics::ate_rmse(run.slam.trajectory(), &truth),
            false_closures(&run.slam),
            run.slam.closures().count(),
        )
    });
    let per_world: Vec<Value> = WorldKind::ALL
        .iter()
        .filter(|w| tasks.iter().any(|t| t.0 == **w))
        .map(|w| {
            let group: Vec<_> = results.iter().filter(|r| r.0 == w.name()).collect();
            let ate: Vec<f64> = group.iter().map(|r| r.1).collect();
            json!({
                "world": w.name(),
                "ate_rmse_m": summary(&ate),
                "closures_per_run": group.iter().map(|r| r.3).sum::<usize>() as f64 / group.len() as f64,
                "false_closures_total": group.iter().map(|r| r.2).sum::<usize>(),
            })
        })
        .collect();
    let row = json!({ "configuration": name, "worlds": per_world });
    println!("{row}");
    row
}

pub fn gates(seeds: u64, jobs: usize, out: &Path) -> Outcome {
    let tasks: Vec<(WorldKind, u64)> = WorldKind::ALL
        .into_iter()
        .flat_map(|w| (0..seeds).map(move |s| (w, s)))
        .collect();
    let mut blocks = Vec::new();
    // At the default radius almost nothing false gets near the graph, so the
    // safeguards are also measured with the search wide open.
    let mut wide = SlamConfig::default();
    wide.loop_closure.radius_growth = 0.05;
    wide.loop_closure.radius_max = 8.0;
    for (label, base) in [
        ("default search radius", SlamConfig::default()),
        ("search radius widened to 8 m", wide),
    ] {
        let variant = |edit: &dyn Fn(&mut SlamConfig)| {
            let mut config = base.clone();
            edit(&mut config);
            config
        };
        let rows = vec![
            config_row("all safeguards", &tasks, jobs, &base),
            config_row(
                "no pairwise consistency check",
                &tasks,
                jobs,
                &variant(&|c| c.loop_closure.require_consistency = false),
            ),
            config_row(
                "no runner-up ambiguity test",
                &tasks,
                jobs,
                &variant(&|c| c.loop_closure.max_ambiguity = f64::INFINITY),
            ),
            config_row(
                "no robust kernel",
                &tasks,
                jobs,
                &variant(&|c| c.optimize.loop_kernel = RobustKernel::None),
            ),
            config_row(
                "kernel applied at once, no graduation",
                &tasks,
                jobs,
                &variant(&|c| c.graduated = false),
            ),
        ];
        blocks.push(json!({ "search": label, "rows": rows }));
    }
    write_json(
        &out.join("gates.json"),
        &json!({
            "command": format!("loopclosure gates --seeds {seeds}"),
            "source": "simulator; ATE against simulator ground truth",
            "seeds": format!("0..{seeds}"),
            "blocks": blocks,
        }),
    )
}

pub fn aliasing(seeds: u64, jobs: usize, out: &Path) -> Outcome {
    let tasks: Vec<(WorldKind, u64)> = (0..seeds).map(|s| (WorldKind::Warehouse, s)).collect();
    let mut rows = Vec::new();
    for radius in [2.0, 3.0, 4.0, 5.0, 6.0, 8.0] {
        let mut config = SlamConfig::default();
        config.loop_closure.radius_max = radius;
        config.loop_closure.radius_growth = radius / 200.0;
        let mut row = config_row(&format!("radius {radius} m"), &tasks, jobs, &config);
        row["radius_max_m"] = json!(radius);
        rows.push(row);
    }
    write_json(
        &out.join("aliasing.json"),
        &json!({
            "command": format!("loopclosure aliasing --seeds {seeds}"),
            "source": "simulator warehouse; aisles repeat every 6 m and bays every 2.7 m",
            "seeds": format!("0..{seeds}"),
            "rows": rows,
        }),
    )
}

/// A closure that is wrong by a few metres and tens of degrees, with the
/// confidence of a good match. Chosen to be far worse than any match that would
/// pass the front-end checks, since the back end is the last line of defence.
fn false_edge(rng: &mut Rng, poses: &[Pose2], information: [[f64; 3]; 3]) -> Edge {
    let from = rng.below(poses.len());
    let to = loop {
        let candidate = rng.below(poses.len());
        if candidate.abs_diff(from) > 40 {
            break candidate;
        }
    };
    let offset = Pose2::new(
        rng.uniform(-5.0, 5.0),
        rng.uniform(-5.0, 5.0),
        rng.uniform(-0.6, 0.6),
    );
    Edge {
        from,
        to,
        measurement: poses[from].between(poses[to]).compose(offset),
        information,
        kind: EdgeKind::LoopClosure,
    }
}

#[derive(Clone, Copy)]
enum Start {
    Tight,
    Wide,
    Both,
}

pub fn outliers(seeds: u64, jobs: usize, out: &Path) -> Outcome {
    let tasks: Vec<(WorldKind, u64)> = WorldKind::ALL
        .into_iter()
        .flat_map(|w| (0..seeds).map(move |s| (w, s)))
        .collect();
    let dcs = OptimizeOptions::default().loop_kernel;
    let kernels: [(&str, RobustKernel, Start); 6] = [
        ("none", RobustKernel::None, Start::Tight),
        ("huber", RobustKernel::Huber { delta: 3.0 }, Start::Both),
        ("cauchy", RobustKernel::Cauchy { c: 3.0 }, Start::Both),
        ("dcs, tight start only", dcs, Start::Tight),
        ("dcs, wide start only", dcs, Start::Wide),
        ("dcs, both starts", dcs, Start::Both),
    ];
    let counts = [0usize, 10, 50, 200];
    let results = parallel(&tasks, jobs, |&(world, seed)| {
        let data = frames(world, seed);
        let run = run_frames(&data, SlamConfig::default());
        let truth = run.slam.truth().expect("simulated");
        let information = run
            .slam
            .closures()
            .next()
            .map_or([[1e3, 0.0, 0.0], [0.0, 1e3, 0.0], [0.0, 0.0, 1e4]], |e| {
                e.information
            });
        let mut rows = Vec::new();
        for &count in &counts {
            for (name, kernel, start) in kernels {
                let mut graph = run.slam.graph.clone();
                let mut rng = Rng::new(seed ^ 0xBAD_C105E);
                for _ in 0..count {
                    let edge = false_edge(&mut rng, &graph.poses, information);
                    graph.edges.push(edge);
                }
                let options = OptimizeOptions {
                    max_iterations: 40,
                    relative_tolerance: 1e-7,
                    loop_kernel: kernel,
                    ..Default::default()
                };
                match start {
                    Start::Tight => graph.optimize(&options),
                    Start::Wide => graph.optimize_wide_first(&options),
                    Start::Both => graph.optimize_graduated(&options),
                };
                rows.push((count, name, metrics::ate_rmse(&graph.poses, &truth)));
            }
        }
        rows
    });
    let flat: Vec<_> = results.into_iter().flatten().collect();
    let mut rows = Vec::new();
    for &count in &counts {
        for (name, _, _) in kernels {
            let values: Vec<f64> = flat
                .iter()
                .filter(|r| r.0 == count && r.1 == name)
                .map(|r| r.2)
                .collect();
            rows.push(json!({
                "false_closures": count,
                "kernel": name,
                "ate_rmse_m": summary(&values),
            }));
        }
    }
    write_json(
        &out.join("outliers.json"),
        &json!({
            "command": format!("loopclosure outliers --seeds {seeds}"),
            "source": "simulator; false edges added to the finished graph, then re-optimised",
            "seeds": format!("0..{seeds}"),
            "worlds": "office, warehouse, atrium",
            "rows": rows,
        }),
    )
}
