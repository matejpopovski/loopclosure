//! Exhaustive scan matching made cheap by branch and bound.
//!
//! A brute-force search of a translation window of `n` cells at `m` rotations
//! costs `n^2 m` scan evaluations. Each search node here covers a square of
//! translations and is scored on a pyramid level that upper-bounds all of them,
//! so a region whose bound is below the best full-resolution score so far is
//! dropped without visiting it. The result is still the true optimum.

use super::grid::ScoreGrid;
use crate::se2::{wrap_angle, Pose2};

#[derive(Clone, Copy, Debug)]
pub struct SearchWindow {
    /// Half-width of the translation square, in metres.
    pub linear: f64,
    /// Half-width of the rotation interval, in radians.
    pub angular: f64,
    pub angular_step: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct SearchResult {
    pub pose: Pose2,
    /// Mean likelihood of the scan, in `[0, 1]`.
    pub score: f64,
    /// Full-resolution evaluations performed, to measure the pruning.
    pub leaves: usize,
    /// All evaluations, on any pyramid level.
    pub evaluations: usize,
}

/// A region of the search space to skip, used to find the runner-up to a match.
#[derive(Clone, Copy, Debug)]
pub struct Exclusion {
    pub centre: Pose2,
    pub linear: f64,
    pub angular: f64,
}

struct Node {
    rotation: usize,
    x: i32,
    y: i32,
    score: f32,
}

struct Search<'a> {
    grid: &'a ScoreGrid,
    /// Per rotation, the integer cell of every point at zero offset.
    cells: Vec<Vec<[i32; 2]>>,
    /// Per rotation, the leaf penalty for leaving the prior heading.
    rotation_penalty: Vec<f32>,
    linear_penalty: f32,
    /// Exclusion in search coordinates: cell offset, rotation index, radius in
    /// cells, half-width in rotation steps.
    excluded: Option<(f32, f32, i32, f32, i32)>,
    best: Option<(usize, i32, i32, f32, f32)>,
    leaves: usize,
    evaluations: usize,
}

impl Search<'_> {
    #[inline]
    fn sum(&mut self, depth: usize, rotation: usize, x: i32, y: i32) -> f32 {
        self.evaluations += 1;
        let width = self.grid.width as i32;
        let height = self.grid.height as i32;
        let level = self.grid.level(depth);
        let mut total = 0.0f32;
        for &[cx, cy] in &self.cells[rotation] {
            let (gx, gy) = (cx + x, cy + y);
            if gx >= 0 && gy >= 0 && gx < width && gy < height {
                total += level[(gy * width + gx) as usize];
            }
        }
        total
    }

    fn descend(&mut self, depth: usize, nodes: &mut [Node]) {
        nodes.sort_unstable_by(|a, b| b.score.total_cmp(&a.score));
        for i in 0..nodes.len() {
            let (rotation, x, y, score) =
                (nodes[i].rotation, nodes[i].x, nodes[i].y, nodes[i].score);
            let incumbent = self.best.map_or(f32::NEG_INFINITY, |b| b.3);
            // Nodes are sorted, so every later sibling is bounded by this one.
            if score <= incumbent {
                return;
            }
            if depth == 0 {
                if let Some((ex, ey, er, radius, span)) = self.excluded {
                    let d = (x as f32 - ex).hypot(y as f32 - ey);
                    if d <= radius && (rotation as i32 - er).abs() <= span {
                        continue;
                    }
                }
                self.leaves += 1;
                let offset = ((x * x + y * y) as f32).sqrt();
                let penalised =
                    score - self.linear_penalty * offset - self.rotation_penalty[rotation];
                if penalised > incumbent {
                    self.best = Some((rotation, x, y, penalised, score));
                }
                continue;
            }
            let half = 1i32 << (depth - 1);
            let mut children = Vec::with_capacity(4);
            for (cx, cy) in [(x, y), (x + half, y), (x, y + half), (x + half, y + half)] {
                let child_score = self.sum(depth - 1, rotation, cx, cy);
                children.push(Node {
                    rotation,
                    x: cx,
                    y: cy,
                    score: child_score,
                });
            }
            self.descend(depth - 1, &mut children);
        }
    }
}

/// Find the pose in `window` around `seed` that maximises the likelihood of
/// `points` (in the sensor frame).
///
/// `prior_weight` is the score, as a fraction of a perfect match, given up for
/// reaching the edge of the window. It breaks the ties a corridor produces in
/// favour of the prediction rather than an arbitrary spot on the plateau.
/// Because it is only subtracted at the leaves, the unpenalised bounds stay valid.
pub fn search(
    grid: &ScoreGrid,
    points: &[[f64; 2]],
    seed: Pose2,
    window: SearchWindow,
    prior_weight: f64,
) -> SearchResult {
    search_excluding(grid, points, seed, window, prior_weight, None)
}

/// [`search`], ignoring every candidate inside `exclusion`. Comparing this with
/// the unrestricted result measures how distinctive a match is: in a building
/// with repeating bays, the runner-up one bay along scores nearly the same.
pub fn search_excluding(
    grid: &ScoreGrid,
    points: &[[f64; 2]],
    seed: Pose2,
    window: SearchWindow,
    prior_weight: f64,
    exclusion: Option<Exclusion>,
) -> SearchResult {
    if points.is_empty() {
        return SearchResult {
            pose: seed,
            score: 0.0,
            leaves: 0,
            evaluations: 0,
        };
    }
    let steps = (window.angular / window.angular_step).ceil() as i32;
    let reach = (window.linear / grid.resolution).ceil() as i32;
    let n = points.len() as f64;

    let mut cells = Vec::with_capacity((2 * steps + 1) as usize);
    let mut rotation_penalty = Vec::with_capacity(cells.capacity());
    for k in -steps..=steps {
        let dtheta = k as f64 * window.angular_step;
        let (s, c) = (seed.theta + dtheta).sin_cos();
        cells.push(
            points
                .iter()
                .map(|p| {
                    let x = seed.x + c * p[0] - s * p[1];
                    let y = seed.y + s * p[0] + c * p[1];
                    [
                        ((x - grid.origin[0]) / grid.resolution).floor() as i32,
                        ((y - grid.origin[1]) / grid.resolution).floor() as i32,
                    ]
                })
                .collect(),
        );
        let frac = if steps == 0 {
            0.0
        } else {
            (k as f64 / steps as f64).abs()
        };
        rotation_penalty.push((prior_weight * n * frac) as f32);
    }

    let depth = grid.depth();
    let stride = 1i32 << depth;
    let mut state = Search {
        grid,
        cells,
        rotation_penalty,
        linear_penalty: if reach > 0 {
            (prior_weight * n / reach as f64) as f32
        } else {
            0.0
        },
        excluded: exclusion.map(|e| {
            (
                ((e.centre.x - seed.x) / grid.resolution) as f32,
                ((e.centre.y - seed.y) / grid.resolution) as f32,
                (wrap_angle(e.centre.theta - seed.theta) / window.angular_step).round() as i32
                    + steps,
                (e.linear / grid.resolution) as f32,
                (e.angular / window.angular_step).ceil() as i32,
            )
        }),
        best: None,
        leaves: 0,
        evaluations: 0,
    };

    let mut roots = Vec::new();
    for rotation in 0..(2 * steps + 1) as usize {
        let mut y = -reach;
        while y <= reach {
            let mut x = -reach;
            while x <= reach {
                let score = state.sum(depth, rotation, x, y);
                roots.push(Node {
                    rotation,
                    x,
                    y,
                    score,
                });
                x += stride;
            }
            y += stride;
        }
    }
    state.descend(depth, &mut roots);

    let (rotation, x, y, _, raw) = state.best.unwrap_or((steps as usize, 0, 0, 0.0, 0.0));
    let dtheta = (rotation as i32 - steps) as f64 * window.angular_step;
    SearchResult {
        pose: Pose2::new(
            seed.x + x as f64 * grid.resolution,
            seed.y + y as f64 * grid.resolution,
            wrap_angle(seed.theta + dtheta),
        ),
        score: raw as f64 / n,
        leaves: state.leaves,
        evaluations: state.evaluations,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;

    /// An L-shaped room seen from a pose.
    fn walls() -> Vec<[f64; 2]> {
        let mut points = Vec::new();
        for i in 0..200 {
            let t = i as f64 * 0.05;
            points.push([t, 0.0]);
            points.push([0.0, t]);
            points.push([t * 0.5, 6.0 - t * 0.2]);
        }
        points
    }

    fn in_frame(pose: Pose2, world: &[[f64; 2]]) -> Vec<[f64; 2]> {
        let inverse = pose.inverse();
        world.iter().map(|&p| inverse.transform(p)).collect()
    }

    /// The pruned search must equal plain enumeration, including the penalty.
    fn brute_force(
        grid: &ScoreGrid,
        points: &[[f64; 2]],
        seed: Pose2,
        window: SearchWindow,
    ) -> f64 {
        let steps = (window.angular / window.angular_step).ceil() as i32;
        let reach = (window.linear / grid.resolution).ceil() as i32;
        let mut best = 0.0f64;
        for k in -steps..=steps {
            let (s, c) = (seed.theta + k as f64 * window.angular_step).sin_cos();
            for dy in -reach..=reach {
                for dx in -reach..=reach {
                    let mut total = 0.0;
                    for p in points {
                        let x = seed.x + c * p[0] - s * p[1];
                        let y = seed.y + s * p[0] + c * p[1];
                        let gx =
                            ((x - grid.origin[0]) / grid.resolution).floor() as isize + dx as isize;
                        let gy =
                            ((y - grid.origin[1]) / grid.resolution).floor() as isize + dy as isize;
                        if gx >= 0
                            && gy >= 0
                            && gx < grid.width as isize
                            && gy < grid.height as isize
                        {
                            total += grid.level(0)[gy as usize * grid.width + gx as usize] as f64;
                        }
                    }
                    best = best.max(total);
                }
            }
        }
        best / points.len() as f64
    }

    #[test]
    fn branch_and_bound_finds_the_same_optimum_as_enumeration() {
        let world = walls();
        let mut rng = Rng::new(41);
        let grid = ScoreGrid::build(&world, 0.05, 0.05, 1.0, 3);
        let window = SearchWindow {
            linear: 0.4,
            angular: 0.1,
            angular_step: 0.05,
        };
        for _ in 0..10 {
            let truth = Pose2::new(
                rng.uniform(0.5, 4.0),
                rng.uniform(0.5, 4.0),
                rng.uniform(-0.5, 0.5),
            );
            let scan: Vec<_> = in_frame(truth, &world).into_iter().step_by(7).collect();
            let seed = Pose2::new(
                truth.x + rng.uniform(-0.3, 0.3),
                truth.y + rng.uniform(-0.3, 0.3),
                truth.theta + rng.uniform(-0.08, 0.08),
            );
            let pruned = search(&grid, &scan, seed, window, 0.0);
            let exact = brute_force(&grid, &scan, seed, window);
            assert!(
                (pruned.score - exact).abs() < 1e-4,
                "{} {exact}",
                pruned.score
            );
        }
    }

    #[test]
    fn recovers_a_large_offset_and_prunes_most_of_the_window() {
        let world = walls();
        let grid = ScoreGrid::build(&world, 0.05, 0.05, 2.0, 5);
        let truth = Pose2::new(2.0, 2.5, 0.2);
        let scan: Vec<_> = in_frame(truth, &world).into_iter().step_by(5).collect();
        let seed = Pose2::new(3.1, 1.5, 0.0);
        let window = SearchWindow {
            linear: 1.5,
            angular: 0.4,
            angular_step: 0.01,
        };
        let result = search(&grid, &scan, seed, window, 0.0);
        assert!(result.pose.distance_to(truth) < 0.06, "{:?}", result.pose);
        assert!(wrap_angle(result.pose.theta - truth.theta).abs() < 0.011);
        assert!(result.score > 0.7, "score {}", result.score);
        let full = (61 * 61 * 81) as usize;
        assert!(
            result.evaluations * 20 < full,
            "{} of {full}",
            result.evaluations
        );
    }

    #[test]
    fn exclusion_finds_the_second_of_two_identical_places() {
        // Two identical corners, 3 m apart.
        let corner = |ox: f64| {
            (0..80).flat_map(move |i| {
                let t = i as f64 * 0.025;
                [[ox + t, 0.0], [ox, t]]
            })
        };
        let world: Vec<[f64; 2]> = corner(0.0).chain(corner(3.0)).collect();
        let scan: Vec<[f64; 2]> = corner(0.0).step_by(4).collect();
        let grid = ScoreGrid::build(&world, 0.05, 0.05, 2.0, 4);
        let window = SearchWindow {
            linear: 3.5,
            angular: 0.0,
            angular_step: 0.05,
        };
        let seed = Pose2::new(1.5, 0.0, 0.0);
        let best = search(&grid, &scan, seed, window, 0.0);
        let runner_up = search_excluding(
            &grid,
            &scan,
            seed,
            window,
            0.0,
            Some(Exclusion {
                centre: best.pose,
                linear: 0.5,
                angular: 0.0,
            }),
        );
        assert!(best.pose.x.abs() < 0.06 || (best.pose.x - 3.0).abs() < 0.06);
        assert!(
            (best.pose.x - runner_up.pose.x).abs() > 2.5,
            "{best:?} {runner_up:?}"
        );
        assert!(runner_up.score > 0.9 * best.score);
    }

    #[test]
    fn a_tied_plateau_resolves_to_the_prediction() {
        // A single long wall: sliding along it scores the same everywhere.
        let wall: Vec<[f64; 2]> = (0..600).map(|i| [i as f64 * 0.02, 0.0]).collect();
        let grid = ScoreGrid::build(&wall, 0.05, 0.05, 1.0, 3);
        let seed = Pose2::new(6.0, 1.0, 0.0);
        let scan: Vec<_> = (-40..40).map(|i| [i as f64 * 0.05, -1.0]).collect();
        let window = SearchWindow {
            linear: 0.5,
            angular: 0.0,
            angular_step: 0.01,
        };
        let result = search(&grid, &scan, seed, window, 0.02);
        assert!((result.pose.x - seed.x).abs() < 0.11, "{:?}", result.pose);
    }

    #[test]
    fn an_empty_scan_returns_the_seed() {
        let grid = ScoreGrid::build(&walls(), 0.1, 0.1, 0.5, 2);
        let seed = Pose2::new(1.0, 2.0, 0.3);
        let window = SearchWindow {
            linear: 0.3,
            angular: 0.1,
            angular_step: 0.05,
        };
        let result = search(&grid, &[], seed, window, 0.0);
        assert_eq!(result.score, 0.0);
        assert_eq!(result.pose, seed);
    }
}
