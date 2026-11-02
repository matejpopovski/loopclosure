//! Pose graph optimisation.
//!
//! Minimises the sum of squared Mahalanobis errors over relative-pose
//! constraints, using Gauss-Newton on the sparse normal equations. Damping is
//! zero while the steps reduce the cost and only rises after a rejected step, so
//! a healthy problem runs as plain Gauss-Newton and a badly initialised one
//! degrades into Levenberg-Marquardt instead of diverging.

use crate::linalg::{order, BlockSymmetric, Ldl, Mat3, NotPositiveDefinite, Ordering};
use crate::se2::{wrap_angle, Pose2};

/// Where a constraint came from. Only loop closures can be wrong in a way that
/// breaks the map, so only they are passed through the robust kernel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EdgeKind {
    Sequential,
    LoopClosure,
}

/// A measurement of `poses[to]` in the frame of `poses[from]`.
#[derive(Clone, Copy, Debug)]
pub struct Edge {
    pub from: usize,
    pub to: usize,
    pub measurement: Pose2,
    pub information: Mat3,
    pub kind: EdgeKind,
}

/// Robust loss on the squared Mahalanobis error `s`.
///
/// `weight` is `d cost / d s`, the factor iteratively reweighted least squares
/// multiplies into an edge. Keeping `cost` as the integral of `weight` means the
/// step acceptance test compares the objective the steps are descending.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RobustKernel {
    None,
    Huber {
        delta: f64,
    },
    Cauchy {
        c: f64,
    },
    /// Dynamic Covariance Scaling (Agarwal et al., ICRA 2013).
    Dcs {
        phi: f64,
    },
}

impl RobustKernel {
    pub fn weight(self, s: f64) -> f64 {
        match self {
            RobustKernel::None => 1.0,
            RobustKernel::Huber { delta } => {
                if s <= delta * delta {
                    1.0
                } else {
                    delta / s.sqrt()
                }
            }
            RobustKernel::Cauchy { c } => 1.0 / (1.0 + s / (c * c)),
            RobustKernel::Dcs { phi } => {
                let scale = (2.0 * phi / (phi + s)).min(1.0);
                scale * scale
            }
        }
    }

    /// The same kernel with its transition moved by `factor` in `s`.
    pub fn widened(self, factor: f64) -> Self {
        match self {
            RobustKernel::None => RobustKernel::None,
            RobustKernel::Huber { delta } => RobustKernel::Huber {
                delta: delta * factor.sqrt(),
            },
            RobustKernel::Cauchy { c } => RobustKernel::Cauchy {
                c: c * factor.sqrt(),
            },
            RobustKernel::Dcs { phi } => RobustKernel::Dcs { phi: phi * factor },
        }
    }

    pub fn cost(self, s: f64) -> f64 {
        match self {
            RobustKernel::None => s,
            RobustKernel::Huber { delta } => {
                if s <= delta * delta {
                    s
                } else {
                    2.0 * delta * s.sqrt() - delta * delta
                }
            }
            RobustKernel::Cauchy { c } => c * c * (s / (c * c)).ln_1p(),
            RobustKernel::Dcs { phi } => {
                if s <= phi {
                    s
                } else {
                    3.0 * phi - 4.0 * phi * phi / (phi + s)
                }
            }
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct OptimizeOptions {
    pub max_iterations: usize,
    /// Stop when an accepted step improves the cost by less than this fraction.
    pub relative_tolerance: f64,
    pub ordering: Ordering,
    pub loop_kernel: RobustKernel,
}

impl Default for OptimizeOptions {
    fn default() -> Self {
        Self {
            max_iterations: 15,
            relative_tolerance: 1e-5,
            ordering: Ordering::ApproximateMinimumDegree,
            loop_kernel: RobustKernel::Dcs { phi: 10.0 },
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct OptimizeReport {
    pub iterations: usize,
    pub rejected_steps: usize,
    pub initial_cost: f64,
    pub final_cost: f64,
    pub nnz_l: usize,
}

#[derive(Clone, Debug, Default)]
pub struct PoseGraph {
    pub poses: Vec<Pose2>,
    pub edges: Vec<Edge>,
}

/// Error of one edge and its Jacobians with respect to the two poses.
///
/// The update is additive on `(x, y, theta)` rather than on the group. Since the
/// angle lives in one parameter and is wrapped, the two only differ in the
/// translation chart, and the additive form gives Jacobians that are cheap and
/// easy to audit.
pub fn edge_error(from: Pose2, to: Pose2, z: Pose2) -> ([f64; 3], Mat3, Mat3) {
    let (si, ci) = from.theta.sin_cos();
    let (sz, cz) = z.theta.sin_cos();
    let dx = to.x - from.x;
    let dy = to.y - from.y;
    // Translation of `to` in the frame of `from`.
    let lx = ci * dx + si * dy;
    let ly = -si * dx + ci * dy;
    let ex = cz * (lx - z.x) + sz * (ly - z.y);
    let ey = -sz * (lx - z.x) + cz * (ly - z.y);
    let etheta = wrap_angle(to.theta - from.theta - z.theta);

    // R_z^T R_i^T
    let r00 = cz * ci - sz * si;
    let r01 = cz * si + sz * ci;
    let r10 = -sz * ci - cz * si;
    let r11 = -sz * si + cz * ci;
    // d(l)/d(theta_i), rotated into the measurement frame.
    let dlx = -si * dx + ci * dy;
    let dly = -ci * dx - si * dy;
    let a02 = cz * dlx + sz * dly;
    let a12 = -sz * dlx + cz * dly;

    let a = [[-r00, -r01, a02], [-r10, -r11, a12], [0.0, 0.0, -1.0]];
    let b = [[r00, r01, 0.0], [r10, r11, 0.0], [0.0, 0.0, 1.0]];
    ([ex, ey, etheta], a, b)
}

#[inline]
fn mul(a: &Mat3, b: &Mat3) -> Mat3 {
    let mut out = [[0.0; 3]; 3];
    for r in 0..3 {
        for c in 0..3 {
            out[r][c] = a[r][0] * b[0][c] + a[r][1] * b[1][c] + a[r][2] * b[2][c];
        }
    }
    out
}

#[inline]
fn transpose_mul(a: &Mat3, b: &Mat3) -> Mat3 {
    let mut out = [[0.0; 3]; 3];
    for r in 0..3 {
        for c in 0..3 {
            out[r][c] = a[0][r] * b[0][c] + a[1][r] * b[1][c] + a[2][r] * b[2][c];
        }
    }
    out
}

#[inline]
fn mat_vec(a: &Mat3, v: &[f64; 3]) -> [f64; 3] {
    [
        a[0][0] * v[0] + a[0][1] * v[1] + a[0][2] * v[2],
        a[1][0] * v[0] + a[1][1] * v[1] + a[1][2] * v[2],
        a[2][0] * v[0] + a[2][1] * v[1] + a[2][2] * v[2],
    ]
}

#[inline]
fn transpose_vec(a: &Mat3, v: &[f64; 3]) -> [f64; 3] {
    [
        a[0][0] * v[0] + a[1][0] * v[1] + a[2][0] * v[2],
        a[0][1] * v[0] + a[1][1] * v[1] + a[2][1] * v[2],
        a[0][2] * v[0] + a[1][2] * v[1] + a[2][2] * v[2],
    ]
}

#[inline]
fn scaled(mut m: Mat3, s: f64) -> Mat3 {
    m.iter_mut().flatten().for_each(|v| *v *= s);
    m
}

/// The sparse normal equations of a graph. Pose 0 is held fixed to remove the
/// three gauge degrees of freedom, so pose `k > 0` is block `k - 1` before
/// permutation.
///
/// Exposed so the benchmark can time analysis, assembly and factorisation
/// separately. The core crate takes no clock because `std::time::Instant` panics
/// on `wasm32-unknown-unknown`.
pub struct NormalEquations {
    /// `position[k - 1]` is the permuted block of pose `k`.
    position: Vec<usize>,
    h: BlockSymmetric,
    base: Vec<f64>,
    diagonal: Vec<f64>,
    b: Vec<f64>,
    ldl: Ldl,
}

impl NormalEquations {
    /// Choose an ordering and run the symbolic factorisation.
    pub fn analyze(graph: &PoseGraph, ordering: Ordering) -> Self {
        let n_blocks = graph.poses.len().saturating_sub(1);
        let free_pairs = || {
            graph
                .edges
                .iter()
                .filter(|e| e.from != 0 && e.to != 0)
                .map(|e| (e.from - 1, e.to - 1))
        };
        let perm = order(n_blocks, free_pairs(), ordering);
        let mut position = vec![0; n_blocks];
        for (new, &old) in perm.iter().enumerate() {
            position[old] = new;
        }
        let h = BlockSymmetric::from_pairs(
            n_blocks,
            free_pairs().map(|(a, b)| (position[a], position[b])),
        );
        let ldl = Ldl::analyze(h.dim(), &h.col_ptr, &h.row_idx);
        Self {
            position,
            base: vec![0.0; h.values.len()],
            diagonal: vec![0.0; h.dim()],
            b: vec![0.0; h.dim()],
            h,
            ldl,
        }
    }

    pub fn nnz_l(&self) -> usize {
        self.ldl.nnz_l()
    }

    pub fn dim(&self) -> usize {
        self.h.dim()
    }

    /// Fill `H` and `b` at the current poses. Returns the robust cost.
    pub fn assemble(&mut self, graph: &PoseGraph, kernel: RobustKernel) -> f64 {
        self.h.clear();
        self.b.iter_mut().for_each(|v| *v = 0.0);
        let mut cost = 0.0;
        for edge in &graph.edges {
            let (e, a, b) = edge_error(
                graph.poses[edge.from],
                graph.poses[edge.to],
                edge.measurement,
            );
            let omega_e = mat_vec(&edge.information, &e);
            let s = e[0] * omega_e[0] + e[1] * omega_e[1] + e[2] * omega_e[2];
            let (weight, edge_cost) = match edge.kind {
                EdgeKind::Sequential => (1.0, s),
                EdgeKind::LoopClosure => (kernel.weight(s), kernel.cost(s)),
            };
            cost += edge_cost;

            let omega_a = mul(&edge.information, &a);
            let omega_b = mul(&edge.information, &b);
            let block_i = (edge.from != 0).then(|| self.position[edge.from - 1]);
            let block_j = (edge.to != 0).then(|| self.position[edge.to - 1]);
            if let Some(i) = block_i {
                self.h
                    .add_block(i, i, &scaled(transpose_mul(&a, &omega_a), weight));
                let g = transpose_vec(&a, &omega_e);
                for k in 0..3 {
                    self.b[3 * i + k] -= weight * g[k];
                }
            }
            if let Some(j) = block_j {
                self.h
                    .add_block(j, j, &scaled(transpose_mul(&b, &omega_b), weight));
                let g = transpose_vec(&b, &omega_e);
                for k in 0..3 {
                    self.b[3 * j + k] -= weight * g[k];
                }
            }
            if let (Some(i), Some(j)) = (block_i, block_j) {
                self.h
                    .add_block(i, j, &scaled(transpose_mul(&a, &omega_b), weight));
            }
        }
        self.base.copy_from_slice(&self.h.values);
        self.diagonal = self.h.diagonal();
        cost
    }

    /// Factor `H + lambda diag(H)` and solve for the step, in `poses[1..]` order.
    pub fn solve(&mut self, lambda: f64) -> Result<Vec<f64>, NotPositiveDefinite> {
        self.h.values.copy_from_slice(&self.base);
        if lambda > 0.0 {
            self.h.add_scaled_diagonal(&self.diagonal, lambda);
        }
        self.ldl
            .factor(&self.h.col_ptr, &self.h.row_idx, &self.h.values)?;
        let mut x = self.b.clone();
        self.ldl.solve(&mut x);
        let mut step = vec![0.0; x.len()];
        for (old, &new) in self.position.iter().enumerate() {
            step[3 * old..3 * old + 3].copy_from_slice(&x[3 * new..3 * new + 3]);
        }
        Ok(step)
    }

    /// Dense copy of `H`, for the baseline in the solver benchmark.
    pub fn hessian_dense(&self) -> Vec<f64> {
        self.h.to_dense()
    }

    pub fn gradient(&self) -> &[f64] {
        &self.b
    }
}

impl PoseGraph {
    pub fn new() -> Self {
        Self::default()
    }

    /// Robust cost at the current poses.
    pub fn cost(&self, kernel: RobustKernel) -> f64 {
        self.cost_at(&self.poses, kernel)
    }

    fn cost_at(&self, poses: &[Pose2], kernel: RobustKernel) -> f64 {
        self.edges
            .iter()
            .map(|edge| {
                let (e, _, _) = edge_error(poses[edge.from], poses[edge.to], edge.measurement);
                let w = mat_vec(&edge.information, &e);
                let s = e[0] * w[0] + e[1] * w[1] + e[2] * w[2];
                match edge.kind {
                    EdgeKind::Sequential => s,
                    EdgeKind::LoopClosure => kernel.cost(s),
                }
            })
            .sum()
    }

    /// Optimise while tightening the kernel in stages.
    ///
    /// A correct closure after a long drive has a large error, and a tight
    /// kernel cannot tell that from an outlier: it would switch the edge off at
    /// the moment it is needed. Starting wide lets true closures pull the graph
    /// into place, after which the final tight stage sheds whatever still
    /// disagrees. This is the idea behind graduated non-convexity (Yang et al.,
    /// RA-L 2020), with a short fixed schedule because online optimisation runs
    /// from a good initial guess.
    pub fn optimize_graduated(&mut self, options: &OptimizeOptions) -> OptimizeReport {
        const SCHEDULE: [f64; 3] = [1000.0, 30.0, 1.0];
        let mut report = OptimizeReport::default();
        for (stage, factor) in SCHEDULE.into_iter().enumerate() {
            let stage_report = self.optimize(&OptimizeOptions {
                loop_kernel: options.loop_kernel.widened(factor),
                ..*options
            });
            if stage == 0 {
                report.initial_cost = stage_report.initial_cost;
            }
            report.iterations += stage_report.iterations;
            report.rejected_steps += stage_report.rejected_steps;
            report.final_cost = stage_report.final_cost;
            report.nnz_l = stage_report.nnz_l;
        }
        report
    }

    fn apply(&self, step: &[f64]) -> Vec<Pose2> {
        let mut poses = self.poses.clone();
        for (k, pose) in poses.iter_mut().enumerate().skip(1) {
            let offset = 3 * (k - 1);
            pose.x += step[offset];
            pose.y += step[offset + 1];
            pose.theta = wrap_angle(pose.theta + step[offset + 2]);
        }
        poses
    }

    /// Optimise in place. Pose 0 stays where it is.
    pub fn optimize(&mut self, options: &OptimizeOptions) -> OptimizeReport {
        let mut report = OptimizeReport::default();
        if self.poses.len() < 2 || self.edges.is_empty() {
            return report;
        }
        let mut system = NormalEquations::analyze(self, options.ordering);
        report.nnz_l = system.nnz_l();
        let mut cost = system.assemble(self, options.loop_kernel);
        report.initial_cost = cost;
        let mut lambda = 0.0;

        'outer: for _ in 0..options.max_iterations {
            loop {
                let accepted = match system.solve(lambda) {
                    Ok(step) => {
                        let poses = self.apply(&step);
                        let trial = self.cost_at(&poses, options.loop_kernel);
                        (trial < cost).then_some((poses, trial))
                    }
                    Err(_) => None,
                };
                match accepted {
                    Some((poses, new_cost)) => {
                        self.poses = poses;
                        let improvement = (cost - new_cost) / cost.max(f64::MIN_POSITIVE);
                        cost = system.assemble(self, options.loop_kernel);
                        report.iterations += 1;
                        lambda = if lambda > 1e-7 { lambda / 10.0 } else { 0.0 };
                        if improvement < options.relative_tolerance {
                            break 'outer;
                        }
                        break;
                    }
                    None => {
                        report.rejected_steps += 1;
                        lambda = if lambda == 0.0 { 1e-6 } else { lambda * 10.0 };
                        if lambda > 1e8 {
                            break 'outer;
                        }
                    }
                }
            }
        }
        report.final_cost = cost;
        report
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::dense;
    use crate::rng::Rng;
    use std::f64::consts::PI;

    const UNIT_INFO: Mat3 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

    fn random_pose(rng: &mut Rng) -> Pose2 {
        Pose2::new(
            rng.uniform(-5.0, 5.0),
            rng.uniform(-5.0, 5.0),
            rng.uniform(-PI, PI),
        )
    }

    fn edge(from: usize, to: usize, z: Pose2, kind: EdgeKind) -> Edge {
        Edge {
            from,
            to,
            measurement: z,
            information: UNIT_INFO,
            kind,
        }
    }

    /// A closed ring of `n` poses, with consistent measurements.
    fn ring(n: usize) -> (Vec<Pose2>, Vec<Edge>) {
        let radius = 10.0;
        let truth: Vec<Pose2> = (0..n)
            .map(|k| {
                let a = 2.0 * PI * k as f64 / n as f64;
                Pose2::new(radius * a.cos(), radius * a.sin(), wrap_angle(a + PI / 2.0))
            })
            .collect();
        let mut edges: Vec<Edge> = (1..n)
            .map(|k| {
                edge(
                    k - 1,
                    k,
                    truth[k - 1].between(truth[k]),
                    EdgeKind::Sequential,
                )
            })
            .collect();
        edges.push(edge(
            0,
            n - 1,
            truth[0].between(truth[n - 1]),
            EdgeKind::LoopClosure,
        ));
        (truth, edges)
    }

    #[test]
    fn jacobians_match_finite_differences() {
        let mut rng = Rng::new(21);
        let h = 1e-6;
        for _ in 0..200 {
            let (xi, xj, z) = (
                random_pose(&mut rng),
                random_pose(&mut rng),
                random_pose(&mut rng),
            );
            let (_, a, b) = edge_error(xi, xj, z);
            for c in 0..3 {
                let perturb = |p: Pose2, sign: f64| {
                    let mut v = [p.x, p.y, p.theta];
                    v[c] += sign * h;
                    Pose2::new(v[0], v[1], v[2])
                };
                let (ep, _, _) = edge_error(perturb(xi, 1.0), xj, z);
                let (em, _, _) = edge_error(perturb(xi, -1.0), xj, z);
                let (fp, _, _) = edge_error(xi, perturb(xj, 1.0), z);
                let (fm, _, _) = edge_error(xi, perturb(xj, -1.0), z);
                for r in 0..3 {
                    let numeric_a = wrap_angle(ep[r] - em[r]) / (2.0 * h);
                    let numeric_b = wrap_angle(fp[r] - fm[r]) / (2.0 * h);
                    assert!((numeric_a - a[r][c]).abs() < 1e-6);
                    assert!((numeric_b - b[r][c]).abs() < 1e-6);
                }
            }
        }
    }

    #[test]
    fn robust_cost_is_the_integral_of_its_weight() {
        for kernel in [
            RobustKernel::None,
            RobustKernel::Huber { delta: 2.0 },
            RobustKernel::Cauchy { c: 3.0 },
            RobustKernel::Dcs { phi: 5.0 },
        ] {
            for s in [0.1, 1.0, 3.9, 4.1, 4.99, 5.01, 9.0, 40.0, 1000.0] {
                let h = 1e-5;
                let slope = (kernel.cost(s + h) - kernel.cost(s - h)) / (2.0 * h);
                assert!((slope - kernel.weight(s)).abs() < 1e-5, "{kernel:?} at {s}");
            }
            assert_eq!(kernel.cost(0.0), 0.0);
        }
    }

    #[test]
    fn normal_equations_match_a_dense_assembly() {
        let mut rng = Rng::new(22);
        let n = 8;
        let poses: Vec<Pose2> = (0..n).map(|_| random_pose(&mut rng)).collect();
        let mut edges: Vec<Edge> = (1..n)
            .map(|k| edge(k - 1, k, random_pose(&mut rng), EdgeKind::Sequential))
            .collect();
        edges.push(edge(2, 7, random_pose(&mut rng), EdgeKind::Sequential));
        edges.push(edge(1, 5, random_pose(&mut rng), EdgeKind::Sequential));
        let graph = PoseGraph { poses, edges };

        let dim = 3 * (n - 1);
        let mut h = vec![0.0; dim * dim];
        for e in &graph.edges {
            let (_, a, b) = edge_error(graph.poses[e.from], graph.poses[e.to], e.measurement);
            let mut row =
                |offset_a: Option<usize>, ja: &Mat3, offset_b: Option<usize>, jb: &Mat3| {
                    let m = transpose_mul(ja, jb);
                    if let (Some(i), Some(j)) = (offset_a, offset_b) {
                        for r in 0..3 {
                            for c in 0..3 {
                                h[(i + r) * dim + j + c] += m[r][c];
                            }
                        }
                    }
                };
            let oi = (e.from > 0).then(|| 3 * (e.from - 1));
            let oj = (e.to > 0).then(|| 3 * (e.to - 1));
            row(oi, &a, oi, &a);
            row(oi, &a, oj, &b);
            row(oj, &b, oi, &a);
            row(oj, &b, oj, &b);
        }

        // Natural order, so the dense copy lines up with the hand assembly.
        let mut system = NormalEquations::analyze(&graph, Ordering::Natural);
        system.assemble(&graph, RobustKernel::None);
        let sparse = system.hessian_dense();
        for (x, y) in sparse.iter().zip(&h) {
            assert!((x - y).abs() < 1e-9);
        }
    }

    #[test]
    fn sparse_step_equals_the_dense_step_for_every_ordering() {
        let mut rng = Rng::new(23);
        let (truth, edges) = ring(40);
        let poses = truth
            .iter()
            .map(|p| {
                Pose2::new(
                    p.x + 0.1 * rng.normal(),
                    p.y + 0.1 * rng.normal(),
                    p.theta + 0.02 * rng.normal(),
                )
            })
            .collect();
        let graph = PoseGraph { poses, edges };

        let mut reference = NormalEquations::analyze(&graph, Ordering::Natural);
        reference.assemble(&graph, RobustKernel::None);
        let n = reference.dim();
        let mut l = reference.hessian_dense();
        assert!(dense::cholesky_in_place(&mut l, n));
        let mut oracle = reference.gradient().to_vec();
        dense::cholesky_solve(&l, n, &mut oracle);

        for ordering in [
            Ordering::Natural,
            Ordering::ReverseCuthillMcKee,
            Ordering::ApproximateMinimumDegree,
        ] {
            let mut system = NormalEquations::analyze(&graph, ordering);
            system.assemble(&graph, RobustKernel::None);
            let step = system.solve(0.0).unwrap();
            for k in 0..n {
                assert!((step[k] - oracle[k]).abs() < 1e-7, "{ordering:?}");
            }
        }
    }

    #[test]
    fn recovers_a_consistent_ring_from_a_large_perturbation() {
        let mut rng = Rng::new(24);
        let (truth, edges) = ring(100);
        let poses = truth
            .iter()
            .enumerate()
            .map(|(k, p)| {
                let spread = if k == 0 { 0.0 } else { 1.0 };
                Pose2::new(
                    p.x + spread * rng.normal(),
                    p.y + spread * rng.normal(),
                    wrap_angle(p.theta + spread * 0.3 * rng.normal()),
                )
            })
            .collect();
        let mut graph = PoseGraph { poses, edges };
        let report = graph.optimize(&OptimizeOptions {
            max_iterations: 50,
            relative_tolerance: 1e-14,
            ..Default::default()
        });
        assert!(report.final_cost < 1e-12, "{report:?}");
        for (est, gt) in graph.poses.iter().zip(&truth) {
            assert!(est.distance_to(*gt) < 1e-6);
            assert!(wrap_angle(est.theta - gt.theta).abs() < 1e-6);
        }
    }

    #[test]
    fn the_first_pose_never_moves() {
        let (truth, mut edges) = ring(20);
        edges.push(edge(
            3,
            12,
            Pose2::new(1.0, 1.0, 0.5),
            EdgeKind::LoopClosure,
        ));
        let mut graph = PoseGraph {
            poses: truth.clone(),
            edges,
        };
        graph.optimize(&OptimizeOptions {
            loop_kernel: RobustKernel::None,
            ..Default::default()
        });
        assert_eq!(graph.poses[0], truth[0]);
    }

    #[test]
    fn a_robust_kernel_rejects_an_outlier_that_breaks_plain_least_squares() {
        let (truth, mut edges) = ring(60);
        for e in &mut edges {
            e.information = [[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 400.0]];
        }
        let mut bad = edge(5, 35, Pose2::new(3.0, -1.0, 1.0), EdgeKind::LoopClosure);
        bad.information = edges[0].information;
        edges.push(bad);
        let run = |kernel| {
            let mut graph = PoseGraph {
                poses: truth.clone(),
                edges: edges.clone(),
            };
            graph.optimize(&OptimizeOptions {
                max_iterations: 40,
                loop_kernel: kernel,
                ..Default::default()
            });
            graph
                .poses
                .iter()
                .zip(&truth)
                .map(|(a, b)| a.distance_to(*b))
                .fold(0.0, f64::max)
        };
        let plain = run(RobustKernel::None);
        let robust = run(RobustKernel::Dcs { phi: 10.0 });
        assert!(plain > 2.0, "plain error {plain}");
        assert!(robust < 0.05, "robust error {robust}");
    }

    #[test]
    fn graduation_keeps_a_true_closure_that_a_tight_kernel_would_switch_off() {
        // The chain measures a slightly bent arc, so it opens up, and one correct
        // closure says where its end really is.
        let (truth, mut edges) = ring(60);
        let bend = Pose2::new(0.0, 0.0, 0.012);
        for e in &mut edges {
            if e.kind == EdgeKind::LoopClosure {
                e.information = [[400.0, 0.0, 0.0], [0.0, 400.0, 0.0], [0.0, 0.0, 1600.0]];
            } else {
                e.information = [[25.0, 0.0, 0.0], [0.0, 25.0, 0.0], [0.0, 0.0, 100.0]];
                e.measurement = e.measurement.compose(bend);
            }
        }
        let mut poses = vec![truth[0]];
        for e in edges.iter().filter(|e| e.kind == EdgeKind::Sequential) {
            poses.push(poses.last().unwrap().compose(e.measurement));
        }
        let target = edges.last().unwrap().measurement;
        let gap = |graph: &PoseGraph| {
            let n = graph.poses.len() - 1;
            graph.poses[0].between(graph.poses[n]).distance_to(target)
        };
        let options = OptimizeOptions {
            max_iterations: 40,
            loop_kernel: RobustKernel::Dcs { phi: 10.0 },
            ..Default::default()
        };
        let mut tight = PoseGraph {
            poses: poses.clone(),
            edges: edges.clone(),
        };
        tight.optimize(&options);
        let mut graduated = PoseGraph { poses, edges };
        graduated.optimize_graduated(&options);
        assert!(gap(&graduated) < 0.2, "graduated {}", gap(&graduated));
        assert!(gap(&tight) > 1.0, "tight {}", gap(&tight));
    }

    #[test]
    fn a_step_that_raises_the_cost_is_rejected_and_damped() {
        // Heading far from the solution makes the first undamped step overshoot.
        let z = Pose2::new(1.0, 0.0, 0.0);
        let mut graph = PoseGraph {
            poses: vec![
                Pose2::IDENTITY,
                Pose2::new(1.0, 0.0, 3.0),
                Pose2::new(-1.0, 2.0, -2.5),
            ],
            edges: vec![
                edge(0, 1, z, EdgeKind::Sequential),
                edge(1, 2, z, EdgeKind::Sequential),
            ],
        };
        let start = graph.cost(RobustKernel::None);
        let report = graph.optimize(&OptimizeOptions {
            max_iterations: 200,
            relative_tolerance: 1e-12,
            ..Default::default()
        });
        assert!(graph.cost(RobustKernel::None) < start);
        assert!(report.final_cost < 1e-6, "{report:?}");
    }

    #[test]
    fn an_empty_or_single_node_graph_is_a_no_op() {
        let mut graph = PoseGraph::new();
        assert_eq!(graph.optimize(&OptimizeOptions::default()).iterations, 0);
        graph.poses.push(Pose2::new(1.0, 2.0, 3.0));
        assert_eq!(graph.optimize(&OptimizeOptions::default()).iterations, 0);
    }
}
