//! Command line front end: single runs, log replay and the experiments behind
//! every number in the README.

mod experiments;
mod figures;
mod input;
mod render;
mod replay;
mod run;
mod solver;

use clap::{Parser, Subcommand};
use std::path::PathBuf;

#[derive(Parser)]
#[command(
    name = "loopclosure",
    about = "2D LiDAR graph SLAM from first principles"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Run the pipeline once on a simulated world and print its error.
    Simulate {
        /// office, warehouse or atrium.
        #[arg(long, default_value = "office")]
        world: String,
        #[arg(long, default_value_t = 0)]
        seed: u64,
        /// Turn loop closure off.
        #[arg(long)]
        no_loop_closure: bool,
        /// Write the occupancy map to this PNG.
        #[arg(long)]
        map: Option<PathBuf>,
    },
    /// Replay a recorded log (script or CARMEN format, optionally gzipped).
    Replay {
        log: PathBuf,
        /// Directory for the JSON report and map images.
        #[arg(long, default_value = "results")]
        out: PathBuf,
    },
    /// Drift with and without loop closure, over every world and seed.
    Drift {
        #[arg(long, default_value_t = 20)]
        seeds: u64,
        #[arg(long, default_value_t = 8)]
        jobs: usize,
        #[arg(long, default_value = "results")]
        out: PathBuf,
    },
    /// Remove each loop closure safeguard in turn and measure the damage.
    Gates {
        #[arg(long, default_value_t = 10)]
        seeds: u64,
        #[arg(long, default_value_t = 8)]
        jobs: usize,
        #[arg(long, default_value = "results")]
        out: PathBuf,
    },
    /// Sweep the loop search radius in the warehouse, past the aisle spacing.
    Aliasing {
        #[arg(long, default_value_t = 10)]
        seeds: u64,
        #[arg(long, default_value_t = 8)]
        jobs: usize,
        #[arg(long, default_value = "results")]
        out: PathBuf,
    },
    /// Inject false loop closures and compare robust kernels.
    Outliers {
        #[arg(long, default_value_t = 10)]
        seeds: u64,
        #[arg(long, default_value_t = 8)]
        jobs: usize,
        #[arg(long, default_value = "results")]
        out: PathBuf,
    },
    /// Time the sparse solver against orderings and a dense baseline.
    Solver {
        #[arg(long, default_value_t = 7)]
        repeats: usize,
        #[arg(long, default_value = "results")]
        out: PathBuf,
    },
    /// Draw the SVG figures from the JSON results.
    Figures {
        #[arg(long, default_value = "results")]
        results: PathBuf,
    },
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    match Cli::parse().command {
        Command::Simulate {
            world,
            seed,
            no_loop_closure,
            map,
        } => run::simulate(&world, seed, !no_loop_closure, map.as_deref()),
        Command::Replay { log, out } => replay::replay(&log, &out),
        Command::Drift { seeds, jobs, out } => experiments::drift(seeds, jobs, &out),
        Command::Gates { seeds, jobs, out } => experiments::gates(seeds, jobs, &out),
        Command::Aliasing { seeds, jobs, out } => experiments::aliasing(seeds, jobs, &out),
        Command::Outliers { seeds, jobs, out } => experiments::outliers(seeds, jobs, &out),
        Command::Solver { repeats, out } => solver::solver(repeats, &out),
        Command::Figures { results } => figures::figures(&results),
    }
}
