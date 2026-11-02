//! Dense Cholesky, kept as a correctness oracle for the sparse factorisation
//! and as the baseline in the solver benchmark.

/// Overwrite the lower triangle of the row-major `n * n` matrix `a` with its
/// Cholesky factor. Returns `false` if `a` is not positive definite.
pub fn cholesky_in_place(a: &mut [f64], n: usize) -> bool {
    debug_assert_eq!(a.len(), n * n);
    for j in 0..n {
        let (head, tail) = a.split_at_mut(j * n);
        let row_j = &mut tail[..n];
        for k in 0..j {
            let row_k = &head[k * n..k * n + n];
            let lkk = row_k[k];
            let dot: f64 = row_j[..k].iter().zip(&row_k[..k]).map(|(x, y)| x * y).sum();
            row_j[k] = (row_j[k] - dot) / lkk;
        }
        let dot: f64 = row_j[..j].iter().map(|x| x * x).sum();
        let pivot = row_j[j] - dot;
        if pivot <= 0.0 || !pivot.is_finite() {
            return false;
        }
        row_j[j] = pivot.sqrt();
    }
    true
}

/// Solve `L L^T x = b` in place, given the factor from [`cholesky_in_place`].
pub fn cholesky_solve(l: &[f64], n: usize, b: &mut [f64]) {
    for i in 0..n {
        let row = &l[i * n..i * n + n];
        let dot: f64 = row[..i].iter().zip(&b[..i]).map(|(x, y)| x * y).sum();
        b[i] = (b[i] - dot) / row[i];
    }
    for i in (0..n).rev() {
        let xi = b[i] / l[i * n + i];
        b[i] = xi;
        for k in 0..i {
            b[k] -= l[i * n + k] * xi;
        }
    }
}

/// Solve a 3x3 symmetric positive definite system.
pub fn solve3(a: [[f64; 3]; 3], b: [f64; 3]) -> Option<[f64; 3]> {
    let mut flat = [
        a[0][0], a[0][1], a[0][2], a[1][0], a[1][1], a[1][2], a[2][0], a[2][1], a[2][2],
    ];
    if !cholesky_in_place(&mut flat, 3) {
        return None;
    }
    let mut x = b;
    cholesky_solve(&flat, 3, &mut x);
    Some(x)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;

    #[test]
    fn solves_a_random_spd_system() {
        let mut rng = Rng::new(3);
        let n = 25;
        let g: Vec<f64> = (0..n * n).map(|_| rng.normal()).collect();
        let mut a = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                a[i * n + j] = (0..n).map(|k| g[i * n + k] * g[j * n + k]).sum::<f64>();
            }
            a[i * n + i] += 1.0;
        }
        let x_true: Vec<f64> = (0..n).map(|_| rng.normal()).collect();
        let mut b: Vec<f64> = (0..n)
            .map(|i| (0..n).map(|j| a[i * n + j] * x_true[j]).sum())
            .collect();
        let mut l = a.clone();
        assert!(cholesky_in_place(&mut l, n));
        cholesky_solve(&l, n, &mut b);
        for i in 0..n {
            assert!((b[i] - x_true[i]).abs() < 1e-9);
        }
    }

    #[test]
    fn rejects_an_indefinite_matrix() {
        let mut a = [1.0, 2.0, 2.0, 1.0];
        assert!(!cholesky_in_place(&mut a, 2));
    }
}
