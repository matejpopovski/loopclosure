//! Scan matching in two stages.
//!
//! [`correlative`] searches a whole window of poses, so it cannot get trapped in
//! a local minimum, but its answer is quantised to the grid and rotation step.
//! [`icp`] converges to sub-grid precision but only from nearby. Matching runs
//! the first and then the second, and the Hessian from ICP supplies the edge
//! information, which is how a corridor's along-axis ambiguity reaches the graph.

pub mod correlative;
pub mod grid;
pub mod icp;

pub use correlative::{search, search_excluding, Exclusion, SearchResult, SearchWindow};
pub use grid::ScoreGrid;
pub use icp::{refine, IcpOptions, IcpResult, PosePrior, ReferenceCloud};
