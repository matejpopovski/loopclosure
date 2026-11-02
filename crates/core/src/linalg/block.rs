//! Symmetric matrix of 3x3 blocks, stored as a scalar upper-triangular CSC.
//!
//! Pose graphs have a natural block structure, and fill-reducing orderings work
//! best on the block graph, so the permutation is applied to blocks and expanded
//! afterwards. Keeping each block's three rows contiguous also makes block lookup
//! a binary search over one block column.

use super::Mat3;

#[derive(Clone, Debug)]
pub struct BlockSymmetric {
    pub n_blocks: usize,
    /// For block column `j`, the sorted block rows `< j` that hold a block.
    block_ptr: Vec<usize>,
    block_rows: Vec<u32>,
    pub col_ptr: Vec<usize>,
    pub row_idx: Vec<u32>,
    pub values: Vec<f64>,
}

impl BlockSymmetric {
    /// Build the pattern from block coordinates `(i, j)` already expressed in the
    /// permuted order. Duplicates are merged and the diagonal is always present.
    pub fn from_pairs(n_blocks: usize, pairs: impl IntoIterator<Item = (usize, usize)>) -> Self {
        let mut columns: Vec<Vec<u32>> = vec![Vec::new(); n_blocks];
        for (a, b) in pairs {
            if a != b {
                let (lo, hi) = if a < b { (a, b) } else { (b, a) };
                columns[hi].push(lo as u32);
            }
        }
        let mut block_ptr = Vec::with_capacity(n_blocks + 1);
        let mut block_rows = Vec::new();
        block_ptr.push(0);
        for col in &mut columns {
            col.sort_unstable();
            col.dedup();
            block_rows.extend_from_slice(col);
            block_ptr.push(block_rows.len());
        }

        let mut col_ptr = Vec::with_capacity(3 * n_blocks + 1);
        let mut row_idx = Vec::new();
        col_ptr.push(0);
        for j in 0..n_blocks {
            let rows = &block_rows[block_ptr[j]..block_ptr[j + 1]];
            for c in 0..3 {
                for &b in rows {
                    let b = b as usize * 3;
                    row_idx.extend_from_slice(&[b as u32, b as u32 + 1, b as u32 + 2]);
                }
                for r in 0..=c {
                    row_idx.push((3 * j + r) as u32);
                }
                col_ptr.push(row_idx.len());
            }
        }
        let values = vec![0.0; row_idx.len()];
        Self {
            n_blocks,
            block_ptr,
            block_rows,
            col_ptr,
            row_idx,
            values,
        }
    }

    pub fn dim(&self) -> usize {
        self.n_blocks * 3
    }

    pub fn clear(&mut self) {
        self.values.iter_mut().for_each(|v| *v = 0.0);
    }

    /// Add `m` at block `(i, j)` and implicitly `m^T` at `(j, i)`.
    pub fn add_block(&mut self, i: usize, j: usize, m: &Mat3) {
        if i < j {
            let slot = self.block_slot(i, j);
            for c in 0..3 {
                let start = self.col_ptr[3 * j + c] + 3 * slot;
                for r in 0..3 {
                    self.values[start + r] += m[r][c];
                }
            }
        } else if i > j {
            let slot = self.block_slot(j, i);
            for c in 0..3 {
                let start = self.col_ptr[3 * i + c] + 3 * slot;
                for r in 0..3 {
                    self.values[start + r] += m[c][r];
                }
            }
        } else {
            let rows_above = self.block_ptr[j + 1] - self.block_ptr[j];
            for c in 0..3 {
                let start = self.col_ptr[3 * j + c] + 3 * rows_above;
                for r in 0..=c {
                    self.values[start + r] += m[r][c];
                }
            }
        }
    }

    fn block_slot(&self, row: usize, col: usize) -> usize {
        let rows = &self.block_rows[self.block_ptr[col]..self.block_ptr[col + 1]];
        rows.binary_search(&(row as u32))
            .expect("block outside the sparsity pattern")
    }

    /// Add `lambda * d[k]` to each scalar diagonal entry.
    pub fn add_scaled_diagonal(&mut self, d: &[f64], lambda: f64) {
        for (k, dk) in d.iter().enumerate() {
            // The diagonal is the last entry of each upper-triangular column.
            let p = self.col_ptr[k + 1] - 1;
            self.values[p] += lambda * dk;
        }
    }

    pub fn diagonal(&self) -> Vec<f64> {
        (0..self.dim())
            .map(|k| self.values[self.col_ptr[k + 1] - 1])
            .collect()
    }

    /// Expand to a dense row-major matrix, for tests and the dense baseline.
    pub fn to_dense(&self) -> Vec<f64> {
        let n = self.dim();
        let mut a = vec![0.0; n * n];
        for j in 0..n {
            for p in self.col_ptr[j]..self.col_ptr[j + 1] {
                let i = self.row_idx[p] as usize;
                a[i * n + j] = self.values[p];
                a[j * n + i] = self.values[p];
            }
        }
        a
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stores_blocks_symmetrically_regardless_of_argument_order() {
        let mut m = BlockSymmetric::from_pairs(3, [(0, 2), (2, 1), (0, 2)]);
        let block = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
        m.add_block(0, 2, &block);
        let a = m.to_dense();
        for r in 0..3 {
            for c in 0..3 {
                assert_eq!(a[r * 9 + 6 + c], block[r][c]);
                assert_eq!(a[(6 + c) * 9 + r], block[r][c]);
            }
        }

        let mut flipped = BlockSymmetric::from_pairs(3, [(0, 2), (2, 1)]);
        let mut transposed = [[0.0; 3]; 3];
        for r in 0..3 {
            for c in 0..3 {
                transposed[c][r] = block[r][c];
            }
        }
        flipped.add_block(2, 0, &transposed);
        assert_eq!(flipped.to_dense(), a);
    }

    #[test]
    fn diagonal_block_only_stores_the_upper_triangle() {
        let mut m = BlockSymmetric::from_pairs(1, []);
        m.add_block(0, 0, &[[1.0, 2.0, 3.0], [2.0, 4.0, 5.0], [3.0, 5.0, 6.0]]);
        assert_eq!(m.values, vec![1.0, 2.0, 4.0, 3.0, 5.0, 6.0]);
        assert_eq!(m.diagonal(), vec![1.0, 4.0, 6.0]);
    }

    #[test]
    fn row_indices_are_sorted_within_columns() {
        let m = BlockSymmetric::from_pairs(5, [(0, 4), (2, 4), (1, 3), (3, 4)]);
        for j in 0..m.dim() {
            let col = &m.row_idx[m.col_ptr[j]..m.col_ptr[j + 1]];
            assert!(col.windows(2).all(|w| w[0] < w[1]));
            assert_eq!(*col.last().unwrap(), j as u32);
        }
    }
}
