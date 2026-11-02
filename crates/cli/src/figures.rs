//! SVG figures, drawn from the result files so that a figure cannot disagree
//! with the numbers beside it.

use serde_json::Value;
use std::fmt::Write as _;
use std::fs;
use std::path::Path;

const SURFACE: &str = "#0b1a2b";
const GRID: &str = "#1d3550";
const AXIS: &str = "#35536f";
const TEXT: &str = "#c9dbe9";
const MUTED: &str = "#88a3b8";
const WITHOUT: &str = "#cf7d14";
const WITH: &str = "#2a9fd0";

fn polyline(points: &[(f64, f64)], colour: &str) -> String {
    let mut path = String::new();
    for (i, (x, y)) in points.iter().enumerate() {
        let _ = write!(path, "{}{x:.1} {y:.1}", if i == 0 { "M" } else { "L" });
    }
    format!(
        r#"<path d="{path}" fill="none" stroke="{colour}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>"#
    )
}

fn series(curve: &Value, key: &str) -> Vec<(f64, f64)> {
    curve[key]
        .as_array()
        .map(|points| {
            points
                .iter()
                .map(|p| (p[0].as_f64().unwrap_or(0.0), p[1].as_f64().unwrap_or(0.0)))
                .collect()
        })
        .unwrap_or_default()
}

/// One error-versus-distance panel per world, all on one y scale so the panels
/// compare directly.
fn drift_curves(curves: &[Value]) -> String {
    const PANEL_W: f64 = 300.0;
    const PANEL_H: f64 = 210.0;
    const LEFT: f64 = 58.0;
    const GAP: f64 = 26.0;
    const TOP: f64 = 70.0;
    const BOTTOM: f64 = 46.0;
    let n = curves.len() as f64;
    let width = LEFT + n * PANEL_W + (n - 1.0) * GAP + 20.0;
    let height = TOP + PANEL_H + BOTTOM;

    let peak = curves
        .iter()
        .flat_map(|c| series(c, "without").into_iter().chain(series(c, "with")))
        .map(|p| p.1)
        .fold(0.0, f64::max);
    let y_max = (peak.ceil()).max(1.0);

    let mut s = format!(
        r#"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" font-family="ui-monospace, SFMono-Regular, Menlo, monospace"><rect width="100%" height="100%" fill="{SURFACE}"/>"#
    );
    let _ = write!(
        s,
        r#"<text x="{LEFT}" y="22" fill="{TEXT}" font-size="14">Position error while driving, simulated runs, seed 0</text>"#
    );
    let _ = write!(
        s,
        r#"<rect x="{LEFT}" y="35" width="14" height="3" fill="{WITHOUT}"/><text x="{}" y="41" fill="{MUTED}" font-size="12">without loop closure</text>"#,
        LEFT + 20.0
    );
    let _ = write!(
        s,
        r#"<rect x="{}" y="35" width="14" height="3" fill="{WITH}"/><text x="{}" y="41" fill="{MUTED}" font-size="12">with loop closure</text>"#,
        LEFT + 190.0,
        LEFT + 210.0
    );

    for (index, curve) in curves.iter().enumerate() {
        let x0 = LEFT + index as f64 * (PANEL_W + GAP);
        let without = series(curve, "without");
        let with = series(curve, "with");
        let x_max = without.last().map_or(1.0, |p| p.0);
        let sx = |d: f64| x0 + d / x_max * PANEL_W;
        let sy = |e: f64| TOP + PANEL_H - e / y_max * PANEL_H;

        let steps = y_max.max(1.0) as usize;
        let stride = (steps / 4).max(1);
        for tick in (0..=steps).step_by(stride) {
            let y = sy(tick as f64);
            let _ = write!(
                s,
                r#"<line x1="{x0}" x2="{}" y1="{y:.1}" y2="{y:.1}" stroke="{GRID}"/>"#,
                x0 + PANEL_W
            );
            if index == 0 {
                let _ = write!(
                    s,
                    r#"<text x="{}" y="{:.1}" fill="{MUTED}" font-size="11" text-anchor="end">{tick} m</text>"#,
                    x0 - 8.0,
                    y + 4.0
                );
            }
        }
        let x_tick = if x_max > 400.0 { 200.0 } else { 100.0 };
        let mut d = 0.0;
        while d <= x_max {
            let _ = write!(
                s,
                r#"<text x="{:.1}" y="{}" fill="{MUTED}" font-size="11" text-anchor="middle">{d:.0}</text>"#,
                sx(d),
                TOP + PANEL_H + 16.0
            );
            d += x_tick;
        }
        let _ = write!(
            s,
            r#"<line x1="{x0}" x2="{}" y1="{}" y2="{}" stroke="{AXIS}"/>"#,
            x0 + PANEL_W,
            sy(0.0),
            sy(0.0)
        );
        let _ = write!(
            s,
            r#"<text x="{x0}" y="{}" fill="{TEXT}" font-size="12">{}</text>"#,
            TOP - 6.0,
            curve["world"].as_str().unwrap_or("")
        );
        let project = |points: &[(f64, f64)]| -> Vec<(f64, f64)> {
            points.iter().map(|&(d, e)| (sx(d), sy(e))).collect()
        };
        s += &polyline(&project(&without), WITHOUT);
        s += &polyline(&project(&with), WITH);
    }
    let _ = write!(
        s,
        r#"<text x="{}" y="{}" fill="{MUTED}" font-size="11" text-anchor="middle">distance driven (m)</text></svg>"#,
        width / 2.0,
        height - 8.0
    );
    s
}

pub fn figures(results: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let curves: Value =
        serde_json::from_str(&fs::read_to_string(results.join("drift_curves.json"))?)?;
    let list = curves["curves"].as_array().ok_or("no curves in file")?;
    let dir = results.join("figures");
    fs::create_dir_all(&dir)?;
    let path = dir.join("drift_curves.svg");
    fs::write(&path, drift_curves(list))?;
    println!("wrote {}", path.display());
    Ok(())
}
