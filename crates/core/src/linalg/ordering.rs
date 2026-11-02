//! Fill-reducing orderings of a symmetric sparsity graph.
//!
//! A pose graph is a chain plus loop edges. Chronological order is fine for the
//! chain, but every loop edge then fills in a long stretch of the factor. The
//! solver benchmark measures that cost directly by comparing the orderings here.

use std::collections::VecDeque;

/// How to permute the unknowns before factorising.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ordering {
    /// Keep chronological order.
    Natural,
    /// Reverse Cuthill-McKee: minimises bandwidth, not fill.
    ReverseCuthillMcKee,
    /// Approximate minimum degree.
    ApproximateMinimumDegree,
}

/// Returns `perm` with `perm[new] = old`. `pairs` are undirected; self-loops and
/// duplicates are ignored.
pub fn order(
    n: usize,
    pairs: impl IntoIterator<Item = (usize, usize)>,
    kind: Ordering,
) -> Vec<usize> {
    match kind {
        Ordering::Natural => (0..n).collect(),
        Ordering::ReverseCuthillMcKee => reverse_cuthill_mckee(&build_adjacency(n, pairs)),
        Ordering::ApproximateMinimumDegree => approximate_minimum_degree(build_adjacency(n, pairs)),
    }
}

fn build_adjacency(n: usize, pairs: impl IntoIterator<Item = (usize, usize)>) -> Vec<Vec<u32>> {
    let mut adjacency = vec![Vec::new(); n];
    for (a, b) in pairs {
        if a != b {
            adjacency[a].push(b as u32);
            adjacency[b].push(a as u32);
        }
    }
    for list in &mut adjacency {
        list.sort_unstable();
        list.dedup();
    }
    adjacency
}

fn reverse_cuthill_mckee(adjacency: &[Vec<u32>]) -> Vec<usize> {
    let n = adjacency.len();
    let mut visited = vec![false; n];
    let mut result = Vec::with_capacity(n);
    let mut by_degree: Vec<usize> = (0..n).collect();
    by_degree.sort_by_key(|&i| adjacency[i].len());

    for &seed in &by_degree {
        if visited[seed] {
            continue;
        }
        let start = pseudo_peripheral(adjacency, seed);
        visited[start] = true;
        let mut queue = VecDeque::from([start]);
        while let Some(v) = queue.pop_front() {
            result.push(v);
            let mut next: Vec<usize> = adjacency[v]
                .iter()
                .map(|&u| u as usize)
                .filter(|&u| !visited[u])
                .collect();
            next.sort_by_key(|&u| (adjacency[u].len(), u));
            for u in next {
                visited[u] = true;
                queue.push_back(u);
            }
        }
    }
    result.reverse();
    result
}

/// George-Liu heuristic: walk to the far end of a BFS until eccentricity stops
/// growing.
fn pseudo_peripheral(adjacency: &[Vec<u32>], start: usize) -> usize {
    let mut current = start;
    let mut best_depth = 0;
    loop {
        let (far, depth) = farthest(adjacency, current);
        if depth <= best_depth {
            return current;
        }
        best_depth = depth;
        current = far;
    }
}

fn farthest(adjacency: &[Vec<u32>], start: usize) -> (usize, usize) {
    let mut depth = vec![usize::MAX; adjacency.len()];
    depth[start] = 0;
    let mut queue = VecDeque::from([start]);
    let mut far = (start, 0);
    while let Some(v) = queue.pop_front() {
        let candidate = (depth[v], std::cmp::Reverse(adjacency[v].len()));
        if candidate > (far.1, std::cmp::Reverse(adjacency[far.0].len())) {
            far = (v, depth[v]);
        }
        for &u in &adjacency[v] {
            let u = u as usize;
            if depth[u] == usize::MAX {
                depth[u] = depth[v] + 1;
                queue.push_back(u);
            }
        }
    }
    far
}

/// Doubly linked buckets of variables keyed by degree, so the minimum-degree
/// pivot and each degree update are O(1) apart from the scan for the new minimum.
struct DegreeBuckets {
    head: Vec<u32>,
    next: Vec<u32>,
    prev: Vec<u32>,
    min: usize,
}

const NIL: u32 = u32::MAX;

impl DegreeBuckets {
    fn new(n: usize) -> Self {
        Self {
            head: vec![NIL; n + 1],
            next: vec![NIL; n],
            prev: vec![NIL; n],
            min: n,
        }
    }

    fn insert(&mut self, v: usize, degree: usize) {
        let h = self.head[degree];
        self.next[v] = h;
        self.prev[v] = NIL;
        if h != NIL {
            self.prev[h as usize] = v as u32;
        }
        self.head[degree] = v as u32;
        self.min = self.min.min(degree);
    }

    fn remove(&mut self, v: usize, degree: usize) {
        let (p, n) = (self.prev[v], self.next[v]);
        if p != NIL {
            self.next[p as usize] = n;
        } else {
            self.head[degree] = n;
        }
        if n != NIL {
            self.prev[n as usize] = p;
        }
    }

    fn pop_min(&mut self) -> usize {
        while self.head[self.min] == NIL {
            self.min += 1;
        }
        let v = self.head[self.min] as usize;
        self.remove(v, self.min);
        v
    }
}

/// Approximate minimum degree on a quotient graph (Amestoy, Davis and Duff,
/// SIAM J. Matrix Anal. Appl., 1996), with element absorption and aggressive
/// absorption.
///
/// Supervariable detection is left out. It matters when many variables become
/// indistinguishable, which is rare on sparse chain-like graphs at the block
/// level, and the ordering here runs on blocks.
fn approximate_minimum_degree(mut vars: Vec<Vec<u32>>) -> Vec<usize> {
    let n = vars.len();
    let mut elems: Vec<Vec<u32>> = vec![Vec::new(); n];
    let mut element_members: Vec<Vec<u32>> = vec![Vec::new(); n];
    let mut element_alive = vec![false; n];
    let mut eliminated = vec![false; n];
    let mut degree: Vec<usize> = vars.iter().map(Vec::len).collect();

    let mut buckets = DegreeBuckets::new(n);
    for v in (0..n).rev() {
        buckets.insert(v, degree[v]);
    }

    let mut in_pivot = vec![usize::MAX; n];
    let mut external = vec![0usize; n];
    let mut external_stamp = vec![usize::MAX; n];
    let mut perm = Vec::with_capacity(n);

    for k in 0..n {
        let p = buckets.pop_min();
        perm.push(p);
        eliminated[p] = true;

        let mut members = Vec::new();
        in_pivot[p] = k;
        for &v in &vars[p] {
            let v_idx = v as usize;
            if in_pivot[v_idx] != k {
                in_pivot[v_idx] = k;
                members.push(v);
            }
        }
        for &e in &elems[p] {
            let e = e as usize;
            if !element_alive[e] {
                continue;
            }
            for &v in &element_members[e] {
                let v_idx = v as usize;
                if in_pivot[v_idx] != k {
                    in_pivot[v_idx] = k;
                    members.push(v);
                }
            }
            element_alive[e] = false;
            element_members[e] = Vec::new();
        }
        vars[p] = Vec::new();
        elems[p] = Vec::new();

        for &i in &members {
            let i = i as usize;
            buckets.remove(i, degree[i]);
            vars[i].retain(|&v| in_pivot[v as usize] != k);
            elems[i].retain(|&e| element_alive[e as usize]);
        }

        // For each element e touching the pivot, |L_e \ L_p|. Computed in one
        // pass because every shared member decrements it once.
        for &i in &members {
            for &e in &elems[i as usize] {
                let e = e as usize;
                if external_stamp[e] != k {
                    external_stamp[e] = k;
                    external[e] = element_members[e].len();
                }
                external[e] -= 1;
            }
        }

        let remaining = n - k - 1;
        let lp_minus_one = members.len().saturating_sub(1);
        for &i in &members {
            let i_idx = i as usize;
            elems[i_idx].retain(|&e| {
                // Aggressive absorption: L_e is a subset of L_p.
                if external[e as usize] == 0 {
                    element_alive[e as usize] = false;
                    false
                } else {
                    true
                }
            });
            let bound: usize = vars[i_idx].len()
                + lp_minus_one
                + elems[i_idx]
                    .iter()
                    .map(|&e| external[e as usize])
                    .sum::<usize>();
            let d = bound
                .min(degree[i_idx] + lp_minus_one)
                .min(remaining.saturating_sub(1));
            degree[i_idx] = d;
            elems[i_idx].push(p as u32);
        }
        for &i in &members {
            let i = i as usize;
            buckets.insert(i, degree[i]);
        }
        element_members[p] = members;
        element_alive[p] = true;
    }
    perm
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::ldl::Ldl;
    use crate::linalg::BlockSymmetric;
    use crate::rng::Rng;

    fn fill(n: usize, pairs: &[(usize, usize)], kind: Ordering) -> usize {
        let perm = order(n, pairs.iter().copied(), kind);
        let mut inverse = vec![0; n];
        for (new, &old) in perm.iter().enumerate() {
            inverse[old] = new;
        }
        let m = BlockSymmetric::from_pairs(n, pairs.iter().map(|&(a, b)| (inverse[a], inverse[b])));
        Ldl::analyze(m.dim(), &m.col_ptr, &m.row_idx).nnz_l()
    }

    fn assert_permutation(perm: &[usize], n: usize) {
        let mut seen = vec![false; n];
        for &p in perm {
            assert!(!seen[p], "duplicate {p}");
            seen[p] = true;
        }
        assert_eq!(perm.len(), n);
    }

    fn grid(w: usize, h: usize) -> Vec<(usize, usize)> {
        let mut pairs = Vec::new();
        for y in 0..h {
            for x in 0..w {
                if x + 1 < w {
                    pairs.push((y * w + x, y * w + x + 1));
                }
                if y + 1 < h {
                    pairs.push((y * w + x, (y + 1) * w + x));
                }
            }
        }
        pairs
    }

    #[test]
    fn every_ordering_is_a_permutation_on_random_graphs() {
        let mut rng = Rng::new(5);
        for _ in 0..200 {
            let n = 1 + rng.below(80);
            let m = rng.below(3 * n + 1);
            let pairs: Vec<_> = (0..m).map(|_| (rng.below(n), rng.below(n))).collect();
            for kind in [
                Ordering::Natural,
                Ordering::ReverseCuthillMcKee,
                Ordering::ApproximateMinimumDegree,
            ] {
                assert_permutation(&order(n, pairs.iter().copied(), kind), n);
            }
        }
    }

    #[test]
    fn amd_eliminates_an_arrow_from_the_tips() {
        // Star centred on 0. Eliminating the centre first fills the whole matrix;
        // AMD must leave it for last.
        let pairs: Vec<_> = (1..50).map(|i| (0, i)).collect();
        let perm = order(
            50,
            pairs.iter().copied(),
            Ordering::ApproximateMinimumDegree,
        );
        assert!(perm[..48].iter().all(|&v| v != 0));
        assert!(
            fill(50, &pairs, Ordering::ApproximateMinimumDegree)
                < fill(50, &pairs, Ordering::Natural) / 4
        );
    }

    #[test]
    fn amd_beats_natural_order_on_a_grid() {
        let pairs = grid(30, 30);
        let natural = fill(900, &pairs, Ordering::Natural);
        let amd = fill(900, &pairs, Ordering::ApproximateMinimumDegree);
        assert!(
            (amd as f64) < 0.6 * natural as f64,
            "amd {amd} natural {natural}"
        );
    }

    #[test]
    fn rcm_recovers_a_shuffled_chain() {
        let mut labels: Vec<usize> = (0..200).collect();
        let mut rng = Rng::new(9);
        for i in (1..200).rev() {
            labels.swap(i, rng.below(i + 1));
        }
        let pairs: Vec<_> = (0..199).map(|i| (labels[i], labels[i + 1])).collect();
        let tri = BlockSymmetric::from_pairs(200, (0..199).map(|i| (i, i + 1)));
        let optimal = Ldl::analyze(tri.dim(), &tri.col_ptr, &tri.row_idx).nnz_l();
        assert_eq!(fill(200, &pairs, Ordering::ReverseCuthillMcKee), optimal);
    }

    #[test]
    fn handles_a_disconnected_graph_and_isolated_nodes() {
        let pairs = [(0, 1), (3, 4)];
        for kind in [
            Ordering::ReverseCuthillMcKee,
            Ordering::ApproximateMinimumDegree,
        ] {
            assert_permutation(&order(6, pairs, kind), 6);
        }
    }
}
