//! Up-looking sparse LDL^T, following the elimination-tree algorithm in
//! Davis, "Algorithm 849: A Concise Sparse Cholesky Factorization Package"
//! (ACM TOMS, 2005).
//!
//! The symbolic and numeric phases are split because the sparsity pattern of a
//! pose graph is fixed within one optimisation, so every Gauss-Newton iteration
//! after the first only pays for the numeric phase.
//!
//! LDL^T rather than LL^T: it needs no square roots, and a non-positive `d`
//! entry is a direct signal for the Levenberg fallback in the optimiser.

use std::fmt;

const NONE: usize = usize::MAX;

/// A pivot was zero, negative or non-finite.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NotPositiveDefinite {
    pub column: usize,
}

impl fmt::Display for NotPositiveDefinite {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "matrix is not positive definite at column {}",
            self.column
        )
    }
}

impl std::error::Error for NotPositiveDefinite {}

/// Factors of `A = L D L^T` for a symmetric matrix given by its upper triangle
/// in compressed-column form with sorted row indices.
#[derive(Clone, Debug, Default)]
pub struct Ldl {
    n: usize,
    parent: Vec<usize>,
    l_ptr: Vec<usize>,
    l_idx: Vec<u32>,
    l_val: Vec<f64>,
    d: Vec<f64>,
    // Workspace, kept between factorisations so Gauss-Newton does not reallocate.
    y: Vec<f64>,
    pattern: Vec<usize>,
    flag: Vec<usize>,
    l_nz: Vec<usize>,
}

impl Ldl {
    /// Symbolic analysis: elimination tree and exact column counts of `L`.
    pub fn analyze(n: usize, col_ptr: &[usize], row_idx: &[u32]) -> Self {
        let mut parent = vec![NONE; n];
        let mut flag = vec![NONE; n];
        let mut l_nz = vec![0usize; n];
        for k in 0..n {
            flag[k] = k;
            for &i in &row_idx[col_ptr[k]..col_ptr[k + 1]] {
                let mut i = i as usize;
                debug_assert!(i <= k, "only the upper triangle is expected");
                while flag[i] != k {
                    if parent[i] == NONE {
                        parent[i] = k;
                    }
                    l_nz[i] += 1;
                    flag[i] = k;
                    i = parent[i];
                }
            }
        }
        let mut l_ptr = vec![0usize; n + 1];
        for k in 0..n {
            l_ptr[k + 1] = l_ptr[k] + l_nz[k];
        }
        let nnz = l_ptr[n];
        Self {
            n,
            parent,
            l_ptr,
            l_idx: vec![0; nnz],
            l_val: vec![0.0; nnz],
            d: vec![0.0; n],
            y: vec![0.0; n],
            pattern: vec![0; n],
            flag,
            l_nz,
        }
    }

    /// Number of stored off-diagonal entries of `L`.
    pub fn nnz_l(&self) -> usize {
        self.l_ptr[self.n]
    }

    /// Numeric factorisation. The pattern must match the one given to
    /// [`Ldl::analyze`].
    pub fn factor(
        &mut self,
        col_ptr: &[usize],
        row_idx: &[u32],
        values: &[f64],
    ) -> Result<(), NotPositiveDefinite> {
        let n = self.n;
        self.l_nz.iter_mut().for_each(|c| *c = 0);
        for k in 0..n {
            self.y[k] = 0.0;
            let mut top = n;
            self.flag[k] = k;
            // Row k of L has the pattern of the reach of column k of A in the
            // elimination tree; collect it in topological order.
            for p in col_ptr[k]..col_ptr[k + 1] {
                let mut i = row_idx[p] as usize;
                self.y[i] += values[p];
                let mut len = 0;
                while self.flag[i] != k {
                    self.pattern[len] = i;
                    len += 1;
                    self.flag[i] = k;
                    i = self.parent[i];
                }
                while len > 0 {
                    len -= 1;
                    top -= 1;
                    self.pattern[top] = self.pattern[len];
                }
            }
            self.d[k] = self.y[k];
            self.y[k] = 0.0;
            for t in top..n {
                let i = self.pattern[t];
                let yi = self.y[i];
                self.y[i] = 0.0;
                let start = self.l_ptr[i];
                let end = start + self.l_nz[i];
                for p in start..end {
                    self.y[self.l_idx[p] as usize] -= self.l_val[p] * yi;
                }
                let l_ki = yi / self.d[i];
                self.d[k] -= l_ki * yi;
                self.l_idx[end] = k as u32;
                self.l_val[end] = l_ki;
                self.l_nz[i] += 1;
            }
            if self.d[k] <= 0.0 || !self.d[k].is_finite() {
                return Err(NotPositiveDefinite { column: k });
            }
        }
        Ok(())
    }

    /// Solve `A x = b` in place.
    pub fn solve(&self, b: &mut [f64]) {
        debug_assert_eq!(b.len(), self.n);
        for j in 0..self.n {
            let xj = b[j];
            for p in self.l_ptr[j]..self.l_ptr[j + 1] {
                b[self.l_idx[p] as usize] -= self.l_val[p] * xj;
            }
        }
        for (x, d) in b.iter_mut().zip(&self.d) {
            *x /= d;
        }
        for j in (0..self.n).rev() {
            let mut xj = b[j];
            for p in self.l_ptr[j]..self.l_ptr[j + 1] {
                xj -= self.l_val[p] * b[self.l_idx[p] as usize];
            }
            b[j] = xj;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::dense;
    use crate::rng::Rng;

    /// Random sparse SPD matrix: `B^T B + I`, with `B` sparse.
    fn random_spd(n: usize, density: f64, rng: &mut Rng) -> Vec<f64> {
        let mut b = vec![0.0; n * n];
        for v in b.iter_mut() {
            if rng.next_f64() < density {
                *v = rng.normal();
            }
        }
        let mut a = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                a[i * n + j] = (0..n).map(|k| b[k * n + i] * b[k * n + j]).sum();
            }
            a[i * n + i] += 1.0;
        }
        a
    }

    fn upper_csc(a: &[f64], n: usize) -> (Vec<usize>, Vec<u32>, Vec<f64>) {
        let (mut ptr, mut idx, mut val) = (vec![0], vec![], vec![]);
        for j in 0..n {
            for i in 0..=j {
                if a[i * n + j] != 0.0 {
                    idx.push(i as u32);
                    val.push(a[i * n + j]);
                }
            }
            ptr.push(idx.len());
        }
        (ptr, idx, val)
    }

    #[test]
    fn agrees_with_dense_cholesky_on_random_matrices() {
        let mut rng = Rng::new(11);
        for trial in 0..200 {
            let n = 1 + rng.below(60);
            let density = [0.02, 0.1, 0.3, 1.0][trial % 4];
            let a = random_spd(n, density, &mut rng);
            let (ptr, idx, val) = upper_csc(&a, n);
            let mut ldl = Ldl::analyze(n, &ptr, &idx);
            ldl.factor(&ptr, &idx, &val).unwrap();
            let b: Vec<f64> = (0..n).map(|_| rng.normal()).collect();
            let mut sparse = b.clone();
            ldl.solve(&mut sparse);
            let mut chol = a.clone();
            assert!(dense::cholesky_in_place(&mut chol, n));
            let mut oracle = b.clone();
            dense::cholesky_solve(&chol, n, &mut oracle);
            for i in 0..n {
                assert!((sparse[i] - oracle[i]).abs() < 1e-8, "trial {trial}");
            }
        }
    }

    #[test]
    fn refactoring_with_new_values_reuses_the_symbolic_phase() {
        let mut rng = Rng::new(12);
        let n = 30;
        let a = random_spd(n, 0.1, &mut rng);
        let (ptr, idx, val) = upper_csc(&a, n);
        let mut ldl = Ldl::analyze(n, &ptr, &idx);
        ldl.factor(&ptr, &idx, &val).unwrap();
        // Doubling A halves the solution.
        let doubled: Vec<f64> = val.iter().map(|v| v * 2.0).collect();
        let b: Vec<f64> = (0..n).map(|_| rng.normal()).collect();
        let mut x1 = b.clone();
        ldl.solve(&mut x1);
        ldl.factor(&ptr, &idx, &doubled).unwrap();
        let mut x2 = b;
        ldl.solve(&mut x2);
        for i in 0..n {
            assert!((x1[i] - 2.0 * x2[i]).abs() < 1e-9);
        }
    }

    #[test]
    fn reports_the_failing_pivot() {
        // [[1, 2], [2, 1]] has a negative second pivot.
        let (ptr, idx, val) = (vec![0, 1, 3], vec![0, 0, 1], vec![1.0, 2.0, 1.0]);
        let mut ldl = Ldl::analyze(2, &ptr, &idx);
        assert_eq!(
            ldl.factor(&ptr, &idx, &val),
            Err(NotPositiveDefinite { column: 1 })
        );
    }

    #[test]
    fn fill_of_a_tridiagonal_matrix_is_exactly_its_subdiagonal() {
        let n = 50;
        let mut a = vec![0.0; n * n];
        for i in 0..n {
            a[i * n + i] = 4.0;
            if i + 1 < n {
                a[i * n + i + 1] = -1.0;
                a[(i + 1) * n + i] = -1.0;
            }
        }
        let (ptr, idx, _) = upper_csc(&a, n);
        assert_eq!(Ldl::analyze(n, &ptr, &idx).nnz_l(), n - 1);
    }
}
