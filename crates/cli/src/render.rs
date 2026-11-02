//! Map images in the project's blueprint style.

use loopclosure::occupancy::OccupancyGrid;
use loopclosure::slam::Slam;
use std::fs::File;
use std::io::BufWriter;
use std::path::Path;

const UNKNOWN: [u8; 4] = [11, 26, 43, 255];
const FREE: [u8; 4] = [20, 42, 66, 255];
const OCCUPIED: [u8; 4] = [190, 238, 255, 255];
const GRID_LINE: [u8; 3] = [34, 66, 97];
const GRID_SPACING_M: f64 = 5.0;
/// Beams with no return clear space only this far, so the map's free space
/// stops short of the sensor's full reach.
const FREE_RANGE_M: f64 = 12.0;

/// Replay every keyframe into a grid at the graph's final poses.
pub fn build_map(slam: &Slam, resolution: f64) -> OccupancyGrid {
    let mut grid = OccupancyGrid::new(resolution, [-10.0, -10.0], [10.0, 10.0]);
    for (keyframe, pose) in slam.keyframes.iter().zip(slam.trajectory()) {
        grid.ensure_contains([pose.x, pose.y], FREE_RANGE_M + 2.0);
        grid.integrate(*pose, &slam.lidar, &keyframe.scan, FREE_RANGE_M);
    }
    grid
}

pub fn write_map(
    slam: &Slam,
    resolution: f64,
    path: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let grid = build_map(slam, resolution);
    let (w, h) = (grid.width, grid.height);
    let mut rgb = vec![0u8; w * h * 3];
    let rgba = grid.to_rgba(UNKNOWN, FREE, OCCUPIED);

    // Blend grid lines into empty cells only, so walls stay legible.
    let line_cells = (GRID_SPACING_M / resolution).round() as usize;
    let first = |origin: f64| {
        let offset = (-origin).rem_euclid(GRID_SPACING_M);
        (offset / resolution).round() as usize
    };
    let (x0, y0) = (first(grid.origin[0]), first(grid.origin[1]));
    for row in 0..h {
        let world_row = h - 1 - row;
        for col in 0..w {
            let px = &rgba[(row * w + col) * 4..(row * w + col) * 4 + 4];
            let on_line = (col + line_cells - x0 % line_cells).is_multiple_of(line_cells)
                || (world_row + line_cells - y0 % line_cells).is_multiple_of(line_cells);
            let colour = if on_line && px != OCCUPIED {
                GRID_LINE
            } else {
                [px[0], px[1], px[2]]
            };
            rgb[(row * w + col) * 3..(row * w + col) * 3 + 3].copy_from_slice(&colour);
        }
    }

    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let (crop_rgb, crop_w, crop_h) = crop(&rgb, w, h, &rgba);
    let mut encoder = png::Encoder::new(
        BufWriter::new(File::create(path)?),
        crop_w as u32,
        crop_h as u32,
    );
    encoder.set_color(png::ColorType::Rgb);
    encoder.set_depth(png::BitDepth::Eight);
    encoder.write_header()?.write_image_data(&crop_rgb)?;
    eprintln!("wrote {}", path.display());
    Ok(())
}

/// Trim the empty border, keeping a small margin.
fn crop(rgb: &[u8], w: usize, h: usize, rgba: &[u8]) -> (Vec<u8>, usize, usize) {
    let (mut x0, mut y0, mut x1, mut y1) = (w, h, 0, 0);
    for y in 0..h {
        for x in 0..w {
            if rgba[(y * w + x) * 4..(y * w + x) * 4 + 4] == OCCUPIED {
                x0 = x0.min(x);
                x1 = x1.max(x);
                y0 = y0.min(y);
                y1 = y1.max(y);
            }
        }
    }
    if x0 > x1 {
        return (rgb.to_vec(), w, h);
    }
    let margin = 30;
    let (x0, y0) = (x0.saturating_sub(margin), y0.saturating_sub(margin));
    let (x1, y1) = ((x1 + margin).min(w - 1), (y1 + margin).min(h - 1));
    let (cw, ch) = (x1 - x0 + 1, y1 - y0 + 1);
    let mut out = Vec::with_capacity(cw * ch * 3);
    for y in y0..=y1 {
        out.extend_from_slice(&rgb[(y * w + x0) * 3..(y * w + x1 + 1) * 3]);
    }
    (out, cw, ch)
}
