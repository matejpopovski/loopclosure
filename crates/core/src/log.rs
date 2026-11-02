//! Readers for recorded range-finder logs.
//!
//! Two formats, because the same robot data circulates in both:
//!
//! - The `POS` / `LASER-RANGE` script format of the `intel_lab` record on
//!   DSpace@MIT: centimetres and degrees, with a `POS` line roughly twice per scan.
//! - CARMEN `FLASER` lines: metres and radians, with the raw odometry pose on the
//!   same line as the scan.

use crate::scan::{Frame, LidarModel, Scan};
use crate::se2::{wrap_angle, Pose2};
use std::fmt;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParseError {
    pub line: usize,
    pub message: String,
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "line {}: {}", self.line, self.message)
    }
}

impl std::error::Error for ParseError {}

#[derive(Clone, Debug)]
pub struct Log {
    pub lidar: LidarModel,
    pub frames: Vec<Frame>,
}

/// Readings at or beyond this are the sensor's "no return" code. The SICK LMS
/// encodes it as 8183 cm in the Intel log, and no reading in that log falls
/// between 30 m and that code.
const MAX_VALID_RANGE: f64 = 50.0;

fn half_circle_lidar(beams: usize) -> LidarModel {
    let increment = std::f64::consts::PI / beams as f64;
    LidarModel {
        beams,
        // Beams are centred in their angular bins, symmetric about straight ahead.
        angle_min: -std::f64::consts::FRAC_PI_2 + 0.5 * increment,
        angle_increment: increment,
        range_min: 0.05,
        range_max: MAX_VALID_RANGE,
    }
}

fn error(line: usize, message: impl Into<String>) -> ParseError {
    ParseError {
        line,
        message: message.into(),
    }
}

/// Read either format, chosen by the first non-empty line that identifies one.
pub fn parse(text: &str) -> Result<Log, ParseError> {
    for line in text.lines() {
        match line.split_ascii_whitespace().next() {
            Some("POS") | Some("LASER-RANGE") => return parse_script(text),
            Some("FLASER") => return parse_carmen(text),
            _ => {}
        }
    }
    Err(error(1, "no POS, LASER-RANGE or FLASER line found"))
}

fn stamp(seconds: &str, micros: &str, line: usize) -> Result<f64, ParseError> {
    let s: f64 = seconds
        .parse()
        .map_err(|_| error(line, "bad timestamp seconds"))?;
    let u: f64 = micros
        .parse()
        .map_err(|_| error(line, "bad timestamp microseconds"))?;
    Ok(s + u * 1e-6)
}

/// Parse the script format. Odometry is interpolated to each scan's timestamp,
/// since the two streams were logged independently.
pub fn parse_script(text: &str) -> Result<Log, ParseError> {
    let mut poses: Vec<(f64, Pose2)> = Vec::new();
    let mut scans: Vec<(f64, Scan)> = Vec::new();
    for (index, line) in text.lines().enumerate() {
        let number = index + 1;
        let Some((head, body)) = line.split_once(':') else {
            if line.trim().is_empty() {
                continue;
            }
            return Err(error(number, "missing ':'"));
        };
        let fields: Vec<&str> = head.split_ascii_whitespace().collect();
        match fields.first().copied() {
            Some("POS") if fields.len() == 3 => {
                let values: Vec<f64> = body
                    .split_ascii_whitespace()
                    .map(str::parse)
                    .collect::<Result<_, _>>()
                    .map_err(|_| error(number, "bad POS value"))?;
                let [x, y, theta] = values[..] else {
                    return Err(error(number, "POS needs three values"));
                };
                poses.push((
                    stamp(fields[1], fields[2], number)?,
                    Pose2::new(x / 100.0, y / 100.0, wrap_angle(theta.to_radians())),
                ));
            }
            Some("LASER-RANGE") if fields.len() == 6 => {
                let count: usize = fields[4]
                    .parse()
                    .map_err(|_| error(number, "bad beam count"))?;
                let ranges: Vec<f32> = body
                    .split_ascii_whitespace()
                    .map(|v| {
                        v.parse::<f64>().map(|cm| {
                            let metres = cm / 100.0;
                            if metres >= MAX_VALID_RANGE {
                                f32::INFINITY
                            } else {
                                metres as f32
                            }
                        })
                    })
                    .collect::<Result<_, _>>()
                    .map_err(|_| error(number, "bad range value"))?;
                if ranges.len() != count {
                    return Err(error(number, "beam count does not match the data"));
                }
                scans.push((stamp(fields[1], fields[2], number)?, Scan { ranges }));
            }
            _ => return Err(error(number, "unrecognised record")),
        }
    }
    let Some((_, first)) = scans.first() else {
        return Err(error(1, "no scans in log"));
    };
    if poses.is_empty() {
        return Err(error(1, "no odometry in log"));
    }
    let lidar = half_circle_lidar(first.ranges.len());
    if scans.iter().any(|(_, s)| s.ranges.len() != lidar.beams) {
        return Err(error(1, "scans have differing beam counts"));
    }

    let mut cursor = 0;
    let frames = scans
        .into_iter()
        .map(|(time, scan)| {
            while cursor + 1 < poses.len() && poses[cursor + 1].0 <= time {
                cursor += 1;
            }
            Frame {
                stamp: time,
                odometry: interpolate(&poses, cursor, time),
                scan,
                truth: None,
            }
        })
        .collect();
    Ok(Log { lidar, frames })
}

fn interpolate(poses: &[(f64, Pose2)], cursor: usize, time: f64) -> Pose2 {
    let (t0, p0) = poses[cursor];
    let Some(&(t1, p1)) = poses.get(cursor + 1) else {
        return p0;
    };
    if time <= t0 || t1 <= t0 {
        return p0;
    }
    let a = (time - t0) / (t1 - t0);
    Pose2::new(
        p0.x + a * (p1.x - p0.x),
        p0.y + a * (p1.y - p0.y),
        wrap_angle(p0.theta + a * wrap_angle(p1.theta - p0.theta)),
    )
}

/// Parse CARMEN `FLASER num r_1 .. r_num x y theta odom_x odom_y odom_theta
/// ipc_timestamp hostname logger_timestamp`. The field of view is taken to be
/// 180 degrees, which the format does not record. Other record types are skipped.
pub fn parse_carmen(text: &str) -> Result<Log, ParseError> {
    let mut lidar = None;
    let mut frames = Vec::new();
    for (index, line) in text.lines().enumerate() {
        let number = index + 1;
        let mut fields = line.split_ascii_whitespace();
        if fields.next() != Some("FLASER") {
            continue;
        }
        let fields: Vec<&str> = fields.collect();
        let count: usize = fields
            .first()
            .and_then(|v| v.parse().ok())
            .ok_or_else(|| error(number, "bad beam count"))?;
        if fields.len() < count + 8 {
            return Err(error(number, "FLASER line is too short"));
        }
        let float = |i: usize| -> Result<f64, ParseError> {
            fields[i]
                .parse()
                .map_err(|_| error(number, "bad FLASER value"))
        };
        let mut ranges = Vec::with_capacity(count);
        for i in 0..count {
            let r = float(1 + i)?;
            ranges.push(if r >= MAX_VALID_RANGE {
                f32::INFINITY
            } else {
                r as f32
            });
        }
        let base = 1 + count;
        let model = *lidar.get_or_insert_with(|| half_circle_lidar(count));
        if model.beams != count {
            return Err(error(number, "scans have differing beam counts"));
        }
        frames.push(Frame {
            stamp: float(base + 6)?,
            odometry: Pose2::new(
                float(base + 3)?,
                float(base + 4)?,
                wrap_angle(float(base + 5)?),
            ),
            scan: Scan { ranges },
            truth: None,
        });
    }
    let lidar = lidar.ok_or_else(|| error(1, "no FLASER lines in log"))?;
    Ok(Log { lidar, frames })
}

#[cfg(test)]
mod tests {
    use super::*;

    const SCRIPT: &str = "\
POS 100 000000: 0.0 0.0 0.0
LASER-RANGE 100 500000 0 3 180.0: 100.0 200.0 8183.0
POS 101 000000: 100.0 0.0 90.0
POS 102 000000: 100.0 100.0 90.0
LASER-RANGE 102 000000 0 3 180.0: 150.0 150.0 150.0
";

    #[test]
    fn script_units_are_converted_and_the_no_return_code_is_mapped() {
        let log = parse(SCRIPT).unwrap();
        assert_eq!(log.frames.len(), 2);
        assert_eq!(log.lidar.beams, 3);
        let ranges = &log.frames[0].scan.ranges;
        assert_eq!(ranges[0], 1.0);
        assert_eq!(ranges[1], 2.0);
        assert!(ranges[2].is_infinite());
    }

    #[test]
    fn odometry_is_interpolated_to_the_scan_time() {
        let log = parse(SCRIPT).unwrap();
        let first = log.frames[0].odometry;
        assert!((first.x - 0.5).abs() < 1e-9 && first.y.abs() < 1e-9);
        assert!((first.theta - 45f64.to_radians()).abs() < 1e-9);
        let last = log.frames[1].odometry;
        assert!((last.x - 1.0).abs() < 1e-9 && (last.y - 1.0).abs() < 1e-9);
    }

    #[test]
    fn interpolation_takes_the_short_way_across_the_wrap() {
        let a = Pose2::new(0.0, 0.0, 179f64.to_radians());
        let b = Pose2::new(0.0, 0.0, -179f64.to_radians());
        let mid = interpolate(&[(0.0, a), (1.0, b)], 0, 0.5);
        assert!(wrap_angle(mid.theta - std::f64::consts::PI).abs() < 1e-9);
    }

    #[test]
    fn carmen_lines_use_the_raw_odometry_fields() {
        let text = "# a comment\nODOM 0 0 0 0 0 0 1.0 host 1.0\n\
                    FLASER 3 1.5 2.5 81.83 9 9 9 1.0 2.0 0.5 1234.5 host 1234.6\n";
        let log = parse(text).unwrap();
        assert_eq!(log.frames.len(), 1);
        let frame = &log.frames[0];
        assert_eq!(frame.scan.ranges[1], 2.5);
        assert!(frame.scan.ranges[2].is_infinite());
        assert_eq!(frame.odometry, Pose2::new(1.0, 2.0, 0.5));
        assert_eq!(frame.stamp, 1234.5);
    }

    #[test]
    fn malformed_input_reports_the_line_number() {
        let bad = "POS 100 000000: 0.0 0.0\n";
        assert_eq!(parse(bad).unwrap_err().line, 1);
        let bad = "POS 100 000000: 0 0 0\nLASER-RANGE 1 2 0 5 180.0: 1 2 3\n";
        let err = parse(bad).unwrap_err();
        assert_eq!(err.line, 2);
        assert!(err.message.contains("beam count"));
        assert!(parse("nonsense\n").is_err());
        assert!(parse("").is_err());
    }

    #[test]
    fn a_short_flaser_line_is_an_error_not_a_panic() {
        assert!(parse("FLASER 3 1.0 2.0\n").is_err());
    }
}
