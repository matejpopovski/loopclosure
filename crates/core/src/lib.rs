//! 2D LiDAR graph SLAM from first principles, with no dependencies.
//!
//! Feed [`scan::Frame`]s from the simulator or a recorded log into
//! [`slam::Slam`], which keeps a pose graph of keyframes, tracks each new scan
//! against recent ones and closes loops against older ones.
//!
//! ```
//! use loopclosure::sim::{default_lidar, NoiseModel, Simulator, WorldKind};
//! use loopclosure::slam::{Slam, SlamConfig};
//!
//! let lidar = default_lidar();
//! let world = WorldKind::Atrium.build(0);
//! let frames = Simulator::new(world, lidar, NoiseModel::typical(), 0);
//!
//! let mut slam = Slam::new(SlamConfig::default(), lidar);
//! for frame in frames.take(300) {
//!     slam.step(&frame);
//! }
//! slam.finish();
//! assert!(slam.trajectory().len() > 20);
//! ```

pub mod graph;
pub mod linalg;
pub mod log;
pub mod matching;
pub mod metrics;
pub mod occupancy;
pub mod rng;
pub mod scan;
pub mod se2;
pub mod sim;
pub mod slam;
