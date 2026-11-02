//! Solver timing on synthetic pose graphs.

use crate::input::write_json;
use loopclosure::graph::{Edge, EdgeKind, NormalEquations, PoseGraph, RobustKernel};
use loopclosure::linalg::{dense, Ordering};
use loopclosure::rng::Rng;
use loopclosure::se2::{wrap_angle, Pose2};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::f64::consts::FRAC_PI_2;
use std::path::Path;
use std::time::Instant;

/// A random walk on a unit grid, in the style of the M3500 "Manhattan world"
/// benchmark but generated here: the walk closes a loop whenever it returns to a
/// visited cell. Poses start from the noisy odometry.
pub fn manhattan_graph(nodes: usize, seed: u64) -> PoseGraph {
    let mut rng = Rng::new(seed);
    let information = [[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 400.0]];
    let side = (nodes as f64).sqrt() as i64 / 2 + 2;
    let (mut x, mut y, mut heading) = (0i64, 0i64, 0i64);
    let mut truth = vec![Pose2::IDENTITY];
    let mut visits: HashMap<(i64, i64), Vec<usize>> = HashMap::from([((0, 0), vec![0])]);
    let mut edges = Vec::new();
    let mut estimate = vec![Pose2::IDENTITY];
    for k in 1..nodes {
        if rng.next_f64() < 0.25 {
            heading += if rng.next_f64() < 0.5 { 1 } else { 3 };
        }
        let (mut dx, mut dy) = [(1, 0), (0, 1), (-1, 0), (0, -1)][heading.rem_euclid(4) as usize];
        if (x + dx).abs() > side || (y + dy).abs() > side {
            heading += 2;
            (dx, dy) = (-dx, -dy);
        }
        x += dx;
        y += dy;
        let pose = Pose2::new(x as f64, y as f64, wrap_angle(heading as f64 * FRAC_PI_2));
        let z = truth[k - 1].between(pose);
        let noisy = Pose2::new(
            z.x + 0.01 * rng.normal(),
            z.y + 0.01 * rng.normal(),
            z.theta + 0.002 * rng.normal(),
        );
        edges.push(Edge {
            from: k - 1,
            to: k,
            measurement: noisy,
            information,
            kind: EdgeKind::Sequential,
        });
        estimate.push(estimate[k - 1].compose(noisy));
        truth.push(pose);

        let seen = visits.entry((x, y)).or_default();
        for &i in seen.iter().rev().take(2) {
            if k - i > 3 {
                edges.push(Edge {
                    from: i,
                    to: k,
                    measurement: truth[i].between(pose),
                    information,
                    kind: EdgeKind::LoopClosure,
                });
            }
        }
        seen.push(k);
    }
    PoseGraph {
        poses: estimate,
        edges,
    }
}

fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

fn millis(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1e3
}

fn time_sparse(graph: &PoseGraph, ordering: Ordering, repeats: usize) -> Value {
    let (mut analyse, mut assemble, mut factor) = (Vec::new(), Vec::new(), Vec::new());
    let mut nnz = 0;
    let mut dim = 0;
    for _ in 0..repeats {
        let t = Instant::now();
        let mut system = NormalEquations::analyze(graph, ordering);
        analyse.push(millis(t));
        let t = Instant::now();
        system.assemble(graph, RobustKernel::None);
        assemble.push(millis(t));
        let t = Instant::now();
        system.solve(0.0).expect("the graph is connected");
        factor.push(millis(t));
        nnz = system.nnz_l();
        dim = system.dim();
    }
    json!({
        "unknowns": dim,
        "nnz_l": nnz,
        "analyse_ms": median(analyse),
        "assemble_ms": median(assemble),
        "factor_and_solve_ms": median(factor),
    })
}

fn time_dense(graph: &PoseGraph, repeats: usize) -> Value {
    let mut system = NormalEquations::analyze(graph, Ordering::Natural);
    system.assemble(graph, RobustKernel::None);
    let n = system.dim();
    let hessian = system.hessian_dense();
    let gradient = system.gradient().to_vec();
    let mut times = Vec::new();
    for _ in 0..repeats {
        let mut l = hessian.clone();
        let mut x = gradient.clone();
        let t = Instant::now();
        assert!(dense::cholesky_in_place(&mut l, n));
        dense::cholesky_solve(&l, n, &mut x);
        times.push(millis(t));
    }
    json!({ "factor_and_solve_ms": median(times) })
}

pub fn solver(repeats: usize, out: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let mut rows = Vec::new();
    for nodes in [500, 1000, 3500, 10_000, 35_000] {
        let graph = manhattan_graph(nodes, 7);
        let closures = graph
            .edges
            .iter()
            .filter(|e| e.kind == EdgeKind::LoopClosure)
            .count();
        let amd = time_sparse(&graph, Ordering::ApproximateMinimumDegree, repeats);
        let rcm = time_sparse(&graph, Ordering::ReverseCuthillMcKee, repeats);
        // Chronological order fills in so badly that at 10,000 nodes a single run
        // takes minutes, which adds nothing the smaller sizes do not already show.
        let natural = (nodes <= 3500).then(|| time_sparse(&graph, Ordering::Natural, repeats));
        // The dense baseline is cubic. At 3500 nodes one factorisation already
        // takes minutes, which is the point and does not need measuring twice.
        let baseline = (nodes <= 1000).then(|| time_dense(&graph, repeats.min(3)));
        println!("{nodes} nodes: amd {amd} rcm {rcm} natural {natural:?} dense {baseline:?}");
        rows.push(json!({
            "nodes": nodes,
            "closures": closures,
            "amd": amd,
            "rcm": rcm,
            "natural": natural,
            "dense": baseline,
        }));
    }
    write_json(
        &out.join("solver.json"),
        &json!({
            "command": format!("loopclosure solver --repeats {repeats}"),
            "hardware": "Apple M3 Max, 36 GB, one thread; other jobs were running on the machine",
            "graph": "generated Manhattan-world random walk (seed 7), not the published M3500",
            "statistic": "median over repeats, in milliseconds",
            "rows": rows,
        }),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use loopclosure::graph::OptimizeOptions;

    #[test]
    fn the_generated_graph_solves_back_to_a_small_cost() {
        let mut graph = manhattan_graph(800, 3);
        assert!(graph.edges.iter().any(|e| e.kind == EdgeKind::LoopClosure));
        let report = graph.optimize(&OptimizeOptions {
            loop_kernel: RobustKernel::None,
            max_iterations: 30,
            ..Default::default()
        });
        assert!(report.final_cost < report.initial_cost / 10.0, "{report:?}");
    }
}
