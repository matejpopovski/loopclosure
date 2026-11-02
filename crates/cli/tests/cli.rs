//! The binary, run the way the README runs it.

use std::process::Command;

fn cli() -> Command {
    Command::new(env!("CARGO_BIN_EXE_loopclosure"))
}

#[test]
fn simulate_prints_a_json_report() {
    let output = cli()
        .args(["simulate", "--world", "atrium", "--seed", "2000"])
        .output()
        .expect("the binary runs");
    assert!(output.status.success());
    let report: serde_json::Value = serde_json::from_slice(&output.stdout).expect("json");
    assert_eq!(report["world"], "atrium");
    assert!(report["slam"]["ate_rmse_m"].as_f64().unwrap() < 0.3);
    assert!(report["closures"].as_u64().unwrap() > 10);
}

#[test]
fn an_unknown_world_is_a_clear_error() {
    let output = cli()
        .args(["simulate", "--world", "moon"])
        .output()
        .expect("the binary runs");
    assert!(!output.status.success());
    let message = String::from_utf8_lossy(&output.stderr);
    assert!(message.contains("office, warehouse or atrium"), "{message}");
}

#[test]
fn a_missing_log_fails_without_a_panic() {
    let output = cli()
        .args(["replay", "no/such/file"])
        .output()
        .expect("the binary runs");
    assert!(!output.status.success());
    assert!(!String::from_utf8_lossy(&output.stderr).contains("panicked"));
}
