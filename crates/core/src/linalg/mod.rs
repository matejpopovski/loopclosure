//! The linear algebra the back end needs, written out in full: a block-sparse
//! symmetric matrix, fill-reducing orderings, and an up-looking sparse LDL^T.

pub mod block;
pub mod dense;
pub mod ldl;
pub mod ordering;

pub use block::BlockSymmetric;
pub use ldl::{Ldl, NotPositiveDefinite};
pub use ordering::{order, Ordering};

/// A dense 3x3 block, row major.
pub type Mat3 = [[f64; 3]; 3];
