//! The full pipeline on the real Intel Research Lab log in `data/`. There is no
//! ground truth, so this checks that walls seen on separate passes line up.

use flate2::read::GzDecoder;
use loopclosure::log;
use loopclosure::slam::{Slam, SlamConfig};
use std::io::Read;
use std::path::Path;

fn load() -> log::Log {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data/intel.script.gz");
    let bytes = std::fs::read(path).expect("data/intel.script.gz is committed");
    let mut text = String::new();
    GzDecoder::new(&bytes[..])
        .read_to_string(&mut text)
        .expect("valid gzip");
    log::parse(&text).expect("valid log")
}

fn replay(data: &log::Log, loop_closure: bool) -> Slam {
    let mut config = SlamConfig::default();
    config.loop_closure.enabled = loop_closure;
    let mut slam = Slam::new(config, data.lidar);
    for frame in &data.frames {
        slam.step(frame);
    }
    slam.finish();
    slam
}

#[test]
fn the_log_has_the_expected_shape() {
    let data = load();
    assert_eq!(data.frames.len(), 13_631);
    assert_eq!(data.lidar.beams, 180);
    assert!(data.frames.iter().all(|f| f.truth.is_none()));
}

#[test]
fn loop_closure_aligns_separate_passes_of_the_real_building() {
    let data = load();
    let closed = replay(&data, true);
    let open = replay(&data, false);
    assert!(closed.closures().count() > 150);

    let pairs = closed.revisit_pairs(5, 1.5);
    assert!(pairs.len() > 150, "{} pairs", pairs.len());
    let with = closed.revisit_agreement(closed.trajectory(), &pairs, 0.05);
    let without = closed.revisit_agreement(open.trajectory(), &pairs, 0.05);
    assert!(with > 0.4, "agreement with closure {with}");
    assert!(with > 5.0 * without, "{with} vs {without}");
}
