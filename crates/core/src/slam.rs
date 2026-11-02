//! The SLAM pipeline: keyframe selection, scan-to-local-map tracking, loop
//! closure search and pose graph optimisation.

use crate::graph::{Edge, EdgeKind, OptimizeOptions, OptimizeReport, PoseGraph};
use crate::linalg::Mat3;
use crate::matching::{
    refine, search, search_excluding, Exclusion, IcpOptions, IcpResult, PosePrior, ReferenceCloud,
    ScoreGrid, SearchWindow,
};
use crate::scan::{estimate_normals, voxel_filter, Frame, LidarModel, OrientedPoints, Scan};
use crate::se2::{wrap_angle, Pose2};

#[derive(Clone, Debug)]
pub struct LoopConfig {
    pub enabled: bool,
    /// Attempt a closure every this many keyframes.
    pub every: usize,
    /// Path length that must separate two poses before they count as a loop,
    /// so tracking overlap is not mistaken for recognition.
    pub min_travel: f64,
    /// Search radius, widening with the distance travelled since the last
    /// closure, as a stand-in for the growth of pose uncertainty. This doubles as
    /// the aliasing limit: at a radius past the spacing of repeating structure,
    /// a place not yet visited can match its neighbour.
    pub radius_base: f64,
    pub radius_growth: f64,
    pub radius_max: f64,
    pub angular_window: f64,
    pub angular_step: f64,
    /// The target is every old keyframe within this distance of the query, in
    /// any visit, so a place driven through three times offers all three views.
    pub target_radius: f64,
    /// Older keyframes are thinned evenly down to this many.
    pub target_keyframes: usize,
    /// Returns beyond this range are left out of a loop search. They rarely land
    /// on the target and they inflate the search grid.
    pub max_point_range: f64,
    /// Recent keyframes merged into the query. One scan sees a single stretch of
    /// corridor and matches any other; a few metres of history see a sequence of
    /// doors, which is what tells two stretches apart.
    pub query_keyframes: usize,
    pub query_voxel: f64,
    pub grid_resolution: f64,
    pub grid_sigma: f64,
    pub min_score: f64,
    pub min_inlier_ratio: f64,
    pub max_rms: f64,
    /// Share of the score given up at the edge of the search window, resolving
    /// plateaus towards the current estimate.
    pub prior_weight: f64,
    /// Reject a match if the best score elsewhere in the window reaches this
    /// fraction of it. Repeating structure produces exactly such a second peak.
    pub max_ambiguity: f64,
    /// Candidates this close to the best match count as the same peak.
    pub peak_separation: f64,
    /// A closure only enters the graph once a second one agrees with it.
    pub require_consistency: bool,
    /// Two closures are compared if separated by at most this much travel.
    pub consistency_gap: f64,
    pub consistency_translation: f64,
    pub consistency_rotation: f64,
}

#[derive(Clone, Debug)]
pub struct SlamConfig {
    pub keyframe_distance: f64,
    pub keyframe_angle: f64,
    pub local_map_keyframes: usize,
    pub voxel: f64,
    pub normal_radius: f64,
    pub grid_resolution: f64,
    pub grid_sigma: f64,
    pub window: SearchWindow,
    pub prior_weight: f64,
    pub min_score: f64,
    pub min_inlier_ratio: f64,
    /// Assumed noise of one independent scan point, metres.
    pub scan_sigma: f64,
    /// How many of a scan's points count as independent. Neighbouring returns
    /// share map and pose error, so taking all of them at face value would make
    /// every edge wildly overconfident.
    pub scan_effective_points: f64,
    /// Odometry standard deviations for the sequential edges.
    pub odom_sigma_per_metre: f64,
    pub odom_sigma_floor: f64,
    pub odom_heading_sigma_per_metre: f64,
    pub odom_heading_sigma_per_radian: f64,
    pub odom_heading_sigma_floor: f64,
    pub icp: IcpOptions,
    pub loop_closure: LoopConfig,
    pub optimize: OptimizeOptions,
    /// Also try a start that tightens the robust kernel in stages, see
    /// [`PoseGraph::optimize_graduated`].
    pub graduated: bool,
}

impl Default for SlamConfig {
    fn default() -> Self {
        Self {
            keyframe_distance: 0.3,
            keyframe_angle: 12f64.to_radians(),
            local_map_keyframes: 30,
            voxel: 0.1,
            normal_radius: 0.35,
            grid_resolution: 0.05,
            grid_sigma: 0.07,
            window: SearchWindow {
                linear: 0.25,
                angular: 6f64.to_radians(),
                angular_step: 1f64.to_radians(),
            },
            prior_weight: 0.05,
            min_score: 0.3,
            min_inlier_ratio: 0.5,
            scan_sigma: 0.03,
            scan_effective_points: 40.0,
            odom_sigma_per_metre: 0.15,
            odom_sigma_floor: 0.01,
            odom_heading_sigma_per_metre: 0.03,
            odom_heading_sigma_per_radian: 0.05,
            odom_heading_sigma_floor: 0.005,
            icp: IcpOptions::default(),
            loop_closure: LoopConfig {
                enabled: true,
                every: 4,
                min_travel: 12.0,
                radius_base: 1.0,
                radius_growth: 0.02,
                radius_max: 4.0,
                angular_window: 45f64.to_radians(),
                angular_step: 1f64.to_radians(),
                target_radius: 10.0,
                target_keyframes: 60,
                max_point_range: 15.0,
                query_keyframes: 30,
                query_voxel: 0.15,
                grid_resolution: 0.08,
                grid_sigma: 0.08,
                min_score: 0.35,
                min_inlier_ratio: 0.45,
                max_rms: 0.09,
                prior_weight: 0.02,
                max_ambiguity: 0.92,
                peak_separation: 0.6,
                require_consistency: true,
                consistency_gap: 8.0,
                consistency_translation: 0.4,
                consistency_rotation: 4f64.to_radians(),
            },
            optimize: OptimizeOptions::default(),
            graduated: true,
        }
    }
}

/// Pairs of keyframes from different visits to the same place, for
/// [`Slam::revisit_agreement`].
pub type RevisitPairs = Vec<(usize, usize)>;

/// A scan kept as a graph node.
#[derive(Clone, Debug)]
pub struct Keyframe {
    pub stamp: f64,
    pub odometry: Pose2,
    pub truth: Option<Pose2>,
    pub scan: Scan,
    /// Odometric path length at this keyframe.
    pub travel: f64,
    oriented: OrientedPoints,
    query: Vec<[f64; 2]>,
}

/// What one input frame led to.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct StepOutcome {
    pub keyframe: bool,
    pub closures_added: usize,
    pub optimized: bool,
}

/// Why loop candidates did or did not make it into the graph.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LoopStats {
    pub attempts: usize,
    pub rejected_score: usize,
    pub rejected_ambiguous: usize,
    pub rejected_geometry: usize,
    pub rejected_inconsistent: usize,
    pub accepted: usize,
    pub optimizations: usize,
}

#[derive(Clone, Copy, Debug)]
struct Closure {
    centre: usize,
    keyframe: usize,
    measurement: Pose2,
    information: Mat3,
    in_graph: bool,
}

pub struct Slam {
    pub config: SlamConfig,
    pub lidar: LidarModel,
    pub graph: PoseGraph,
    pub keyframes: Vec<Keyframe>,
    pub stats: LoopStats,
    /// Scan-to-map tracking failures, where the node fell back on odometry.
    pub tracking_failures: usize,
    pub last_optimization: OptimizeReport,
    last_odometry: Pose2,
    travel: f64,
    last_closure_travel: f64,
    pending: Vec<Closure>,
}

fn diagonal(a: f64, b: f64, c: f64) -> Mat3 {
    [[a, 0.0, 0.0], [0.0, b, 0.0], [0.0, 0.0, c]]
}

impl Slam {
    pub fn new(config: SlamConfig, lidar: LidarModel) -> Self {
        Self {
            config,
            lidar,
            graph: PoseGraph::new(),
            keyframes: Vec::new(),
            stats: LoopStats::default(),
            tracking_failures: 0,
            last_optimization: OptimizeReport::default(),
            last_odometry: Pose2::IDENTITY,
            travel: 0.0,
            last_closure_travel: 0.0,
            pending: Vec::new(),
        }
    }

    pub fn trajectory(&self) -> &[Pose2] {
        &self.graph.poses
    }

    pub fn truth(&self) -> Option<Vec<Pose2>> {
        self.keyframes.iter().map(|k| k.truth).collect()
    }

    pub fn odometry(&self) -> Vec<Pose2> {
        self.keyframes.iter().map(|k| k.odometry).collect()
    }

    pub fn closures(&self) -> impl Iterator<Item = &Edge> {
        self.graph
            .edges
            .iter()
            .filter(|e| e.kind == EdgeKind::LoopClosure)
    }

    /// Pick pairs of keyframes that view the same place on separate passes,
    /// according to the current trajectory.
    pub fn revisit_pairs(&self, sample_every: usize, distance: f64) -> RevisitPairs {
        let min_travel = self.config.loop_closure.min_travel;
        let mut pairs = Vec::new();
        for k in (0..self.keyframes.len()).step_by(sample_every.max(1)) {
            let travel = self.keyframes[k].travel;
            let here = self.graph.poses[k];
            let older = (0..k)
                .take_while(|&i| travel - self.keyframes[i].travel >= min_travel)
                .map(|i| (i, self.graph.poses[i].distance_to(here)))
                .filter(|&(_, d)| d < distance)
                .min_by(|a, b| a.1.total_cmp(&b.1));
            if let Some((i, _)) = older {
                pairs.push((i, k));
            }
        }
        pairs
    }

    /// A ground-truth-free consistency measure: the share of a later visit's
    /// points lying within `tolerance` of a wall seen on the earlier visit, when
    /// both are placed by `poses`.
    ///
    /// With no truth for a real log, ATE is unavailable, but a wall observed on
    /// two passes should land in one place. Points seeing something the earlier
    /// visit did not count as misses, so the value tops out below 1. It only
    /// means something relative to another trajectory scored on the same pairs.
    pub fn revisit_agreement(&self, poses: &[Pose2], pairs: &RevisitPairs, tolerance: f64) -> f64 {
        const NEIGHBOURS: usize = 3;
        let (mut hits, mut total) = (0usize, 0usize);
        for &(i, k) in pairs {
            let (mut world, mut normals) = (Vec::new(), Vec::new());
            let last = (i + NEIGHBOURS).min(self.keyframes.len() - 1);
            for older in i.saturating_sub(NEIGHBOURS)..=last {
                let (s, c) = poses[older].theta.sin_cos();
                let oriented = &self.keyframes[older].oriented;
                world.extend(oriented.points.iter().map(|p| poses[older].transform(*p)));
                normals.extend(
                    oriented
                        .normals
                        .iter()
                        .map(|n| [c * n[0] - s * n[1], s * n[0] + c * n[1]]),
                );
            }
            let cloud = ReferenceCloud::new(world, normals, 0.3);
            for p in &self.keyframes[k].query {
                total += 1;
                let q = poses[k].transform(*p);
                if cloud.line_distance(q, 0.3).is_some_and(|d| d <= tolerance) {
                    hits += 1;
                }
            }
        }
        hits as f64 / total.max(1) as f64
    }

    /// Feed one frame. Frames closer than the keyframe thresholds to the last
    /// keyframe only advance the odometry.
    pub fn step(&mut self, frame: &Frame) -> StepOutcome {
        let mut outcome = StepOutcome::default();
        if self.keyframes.is_empty() {
            self.travel = 0.0;
            self.last_odometry = frame.odometry;
            self.graph.poses.push(frame.odometry);
            let keyframe = self.make_keyframe(frame);
            self.keyframes.push(keyframe);
            outcome.keyframe = true;
            return outcome;
        }

        let delta = self.last_odometry.between(frame.odometry);
        if delta.translation_norm() < self.config.keyframe_distance
            && wrap_angle(delta.theta).abs() < self.config.keyframe_angle
        {
            return outcome;
        }
        self.travel += delta.translation_norm();
        self.last_odometry = frame.odometry;
        outcome.keyframe = true;

        let keyframe = self.make_keyframe(frame);
        let from = self.keyframes.len() - 1;
        let previous = self.graph.poses[from];
        let predicted = previous.compose(delta);
        let tracked = self.track(&keyframe, predicted, delta);

        // Tracking already fused the odometry, so a matched step gets a single
        // edge: adding odometry as an edge of its own would count it twice.
        let odometry_information = self.odometry_information(delta);
        let (pose, information) = match tracked {
            Some(matched) => {
                let scan = self.scan_information(&matched);
                let mut fused = odometry_information;
                for r in 0..3 {
                    for c in 0..3 {
                        fused[r][c] += scan[r][c];
                    }
                }
                (matched.pose, fused)
            }
            None => {
                self.tracking_failures += 1;
                (predicted, odometry_information)
            }
        };
        let to = from + 1;
        self.graph.poses.push(pose);
        self.graph.edges.push(Edge {
            from,
            to,
            measurement: previous.between(pose),
            information,
            kind: EdgeKind::Sequential,
        });
        self.keyframes.push(keyframe);

        if self.config.loop_closure.enabled && to.is_multiple_of(self.config.loop_closure.every) {
            outcome.closures_added = self.search_for_loop(to);
            if outcome.closures_added > 0 {
                let options = self.config.optimize;
                self.last_optimization = self.run_optimizer(&options);
                self.stats.optimizations += 1;
                outcome.optimized = true;
            }
        }
        outcome
    }

    /// Final global optimisation, run once the input ends.
    pub fn finish(&mut self) -> OptimizeReport {
        let options = OptimizeOptions {
            max_iterations: 40,
            relative_tolerance: 1e-7,
            ..self.config.optimize
        };
        self.last_optimization = self.run_optimizer(&options);
        self.last_optimization
    }

    fn run_optimizer(&mut self, options: &OptimizeOptions) -> OptimizeReport {
        if self.config.graduated {
            self.graph.optimize_graduated(options)
        } else {
            self.graph.optimize(options)
        }
    }

    fn make_keyframe(&self, frame: &Frame) -> Keyframe {
        let returns = frame.scan.points(&self.lidar);
        let oriented = estimate_normals(&returns, self.config.normal_radius);
        let raw: Vec<[f64; 2]> = returns.iter().map(|r| r.1).collect();
        Keyframe {
            stamp: frame.stamp,
            odometry: frame.odometry,
            truth: frame.truth,
            scan: frame.scan.clone(),
            travel: self.travel,
            query: voxel_filter(&raw, self.config.voxel),
            oriented,
        }
    }

    /// World-frame points and normals of some keyframes, out to `max_range` of
    /// each keyframe.
    fn gather(
        &self,
        indices: impl IntoIterator<Item = usize>,
        max_range: f64,
    ) -> (Vec<[f64; 2]>, Vec<[f64; 2]>) {
        let (mut points, mut normals) = (Vec::new(), Vec::new());
        let limit = max_range * max_range;
        for k in indices {
            let pose = self.graph.poses[k];
            let (s, c) = pose.theta.sin_cos();
            let oriented = &self.keyframes[k].oriented;
            for (p, n) in oriented.points.iter().zip(&oriented.normals) {
                if p[0] * p[0] + p[1] * p[1] <= limit {
                    points.push(pose.transform(*p));
                    normals.push([c * n[0] - s * n[1], s * n[0] + c * n[1]]);
                }
            }
        }
        (points, normals)
    }

    /// Match a new scan against the most recent keyframes.
    fn track(&self, keyframe: &Keyframe, predicted: Pose2, delta: Pose2) -> Option<IcpResult> {
        let cfg = &self.config;
        let n = self.keyframes.len();
        let (points, normals) =
            self.gather(n.saturating_sub(cfg.local_map_keyframes)..n, f64::INFINITY);
        if points.len() < 10 || keyframe.query.len() < 10 {
            return None;
        }
        let grid = ScoreGrid::build(
            &points,
            cfg.grid_resolution,
            cfg.grid_sigma,
            cfg.window.linear + 0.1,
            3,
        );
        let found = search(
            &grid,
            &keyframe.query,
            predicted,
            cfg.window,
            cfg.prior_weight,
        );
        if found.score < cfg.min_score {
            return None;
        }
        let cloud = ReferenceCloud::new(points, normals, cfg.icp.coarse_distance);
        let prior = PosePrior {
            pose: predicted,
            information: self.odometry_information(delta),
            point_weight: cfg.scan_effective_points
                / (keyframe.query.len() as f64 * cfg.scan_sigma * cfg.scan_sigma),
        };
        let matched = refine(&cloud, &keyframe.query, found.pose, Some(&prior), &cfg.icp);
        (matched.inlier_ratio >= cfg.min_inlier_ratio).then_some(matched)
    }

    /// Information of a match, as seen by a graph edge ending at the matched pose.
    ///
    /// ICP linearises in the world frame, while [`crate::graph::edge_error`]
    /// reports error in the frame of the edge's end pose, so the translation
    /// block has to be rotated. Skipping that rotation would point a corridor's
    /// weak direction the wrong way on every corridor not aligned with the
    /// x-axis.
    fn scan_information(&self, matched: &IcpResult) -> Mat3 {
        let cfg = &self.config;
        let trace = matched.hessian[0][0] + matched.hessian[1][1];
        // `trace` is close to the number of matched points, whatever the walls'
        // orientation, which makes it a convenient normaliser.
        let scale = cfg.scan_effective_points / trace.max(1.0) / (cfg.scan_sigma * cfg.scan_sigma);
        let (s, c) = matched.pose.theta.sin_cos();
        let t = [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]];
        let mut out = [[0.0; 3]; 3];
        for r in 0..3 {
            for k in 0..3 {
                for a in 0..3 {
                    for b in 0..3 {
                        out[r][k] += scale * t[a][r] * matched.hessian[a][b] * t[b][k];
                    }
                }
            }
        }
        out
    }

    fn odometry_information(&self, delta: Pose2) -> Mat3 {
        let cfg = &self.config;
        let distance = delta.translation_norm();
        let sigma_t = cfg.odom_sigma_floor + cfg.odom_sigma_per_metre * distance;
        let sigma_r = cfg.odom_heading_sigma_floor
            + cfg.odom_heading_sigma_per_metre * distance
            + cfg.odom_heading_sigma_per_radian * wrap_angle(delta.theta).abs();
        diagonal(
            1.0 / (sigma_t * sigma_t),
            1.0 / (sigma_t * sigma_t),
            1.0 / (sigma_r * sigma_r),
        )
    }

    /// Search older keyframes for a match to keyframe `k`. Returns how many
    /// closures entered the graph.
    fn search_for_loop(&mut self, k: usize) -> usize {
        let cfg = self.config.loop_closure.clone();
        let here = self.graph.poses[k];
        let travel = self.keyframes[k].travel;
        let radius = (cfg.radius_base + cfg.radius_growth * (travel - self.last_closure_travel))
            .min(cfg.radius_max);

        let mut nearest: Option<(usize, f64)> = None;
        for i in 0..k {
            if travel - self.keyframes[i].travel < cfg.min_travel {
                break;
            }
            let d = self.graph.poses[i].distance_to(here);
            if d < radius && nearest.is_none_or(|(_, best)| d < best) {
                nearest = Some((i, d));
            }
        }
        let Some((centre, _)) = nearest else {
            return 0;
        };
        self.stats.attempts += 1;

        let old: Vec<usize> = (0..k)
            .take_while(|&i| travel - self.keyframes[i].travel >= cfg.min_travel)
            .filter(|&i| self.graph.poses[i].distance_to(here) < cfg.target_radius)
            .collect();
        let stride = old.len().div_ceil(cfg.target_keyframes).max(1);
        let (points, normals) =
            self.gather(old.iter().copied().step_by(stride), cfg.max_point_range);
        let window = SearchWindow {
            linear: radius,
            angular: cfg.angular_window,
            angular_step: cfg.angular_step,
        };
        let query = self.loop_query(k);
        let depth = ((2.0 * radius / cfg.grid_resolution).log2().floor() as usize).clamp(2, 5);
        let grid = ScoreGrid::build(&points, cfg.grid_resolution, cfg.grid_sigma, radius, depth);
        let found = search(&grid, &query, here, window, cfg.prior_weight);
        if found.score < cfg.min_score {
            self.stats.rejected_score += 1;
            return 0;
        }
        let runner_up = search_excluding(
            &grid,
            &query,
            here,
            window,
            0.0,
            Some(Exclusion {
                centre: found.pose,
                linear: cfg.peak_separation,
                angular: 3.0 * cfg.angular_step,
            }),
        );
        if runner_up.score > cfg.max_ambiguity * found.score {
            self.stats.rejected_ambiguous += 1;
            return 0;
        }
        let cloud = ReferenceCloud::new(points, normals, self.config.icp.coarse_distance);
        let matched = refine(&cloud, &query, found.pose, None, &self.config.icp);
        if matched.inlier_ratio < cfg.min_inlier_ratio || matched.rms > cfg.max_rms {
            self.stats.rejected_geometry += 1;
            return 0;
        }

        let closure = Closure {
            centre,
            keyframe: k,
            measurement: self.graph.poses[centre].between(matched.pose),
            information: self.scan_information(&matched),
            in_graph: false,
        };
        self.admit(closure)
    }

    /// The last few keyframes as one point set in the frame of keyframe `k`.
    fn loop_query(&self, k: usize) -> Vec<[f64; 2]> {
        let cfg = &self.config.loop_closure;
        let here = self.graph.poses[k].inverse();
        let lo = (k + 1).saturating_sub(cfg.query_keyframes);
        let limit = cfg.max_point_range * cfg.max_point_range;
        let mut points = Vec::new();
        for i in lo..=k {
            let to_here = here.compose(self.graph.poses[i]);
            points.extend(
                self.keyframes[i]
                    .query
                    .iter()
                    .filter(|p| p[0] * p[0] + p[1] * p[1] <= limit)
                    .map(|&p| to_here.transform(p)),
            );
        }
        voxel_filter(&points, cfg.query_voxel)
    }

    /// Two closures agree when going around the small cycle they form gives the
    /// identity. A false match has no reason to agree with the next one.
    fn agrees(&self, a: &Closure, b: &Closure) -> bool {
        let cfg = &self.config.loop_closure;
        let poses = &self.graph.poses;
        let via_old = poses[a.centre]
            .between(poses[b.centre])
            .compose(b.measurement);
        let via_new = a
            .measurement
            .compose(poses[a.keyframe].between(poses[b.keyframe]));
        let error = via_old.between(via_new);
        error.translation_norm() < cfg.consistency_translation
            && wrap_angle(error.theta).abs() < cfg.consistency_rotation
    }

    fn admit(&mut self, mut closure: Closure) -> usize {
        let cfg = self.config.loop_closure.clone();
        if !cfg.require_consistency {
            self.add_closure(&closure);
            return 1;
        }
        let travel = self.keyframes[closure.keyframe].travel;
        let keyframes = &self.keyframes;
        self.pending
            .retain(|c| travel - keyframes[c.keyframe].travel <= cfg.consistency_gap);

        let partners: Vec<usize> = (0..self.pending.len())
            .filter(|&i| self.agrees(&self.pending[i], &closure))
            .collect();
        if partners.is_empty() {
            self.stats.rejected_inconsistent += 1;
            self.pending.push(closure);
            return 0;
        }
        let mut added = 0;
        for i in partners {
            if !self.pending[i].in_graph {
                self.pending[i].in_graph = true;
                let partner = self.pending[i];
                self.add_closure(&partner);
                added += 1;
            }
        }
        closure.in_graph = true;
        self.add_closure(&closure);
        self.pending.push(closure);
        added + 1
    }

    fn add_closure(&mut self, closure: &Closure) {
        self.graph.edges.push(Edge {
            from: closure.centre,
            to: closure.keyframe,
            measurement: closure.measurement,
            information: closure.information,
            kind: EdgeKind::LoopClosure,
        });
        self.stats.accepted += 1;
        self.last_closure_travel = self.keyframes[closure.keyframe].travel;
    }
}
