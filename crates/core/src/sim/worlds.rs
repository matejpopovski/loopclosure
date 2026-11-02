//! Procedural worlds. Each one tests something different:
//!
//! - `Office`: long corridors with few features, where scan matching cannot see
//!   motion along the corridor axis and must lean on odometry.
//! - `Warehouse`: racks on a regular pitch, so many places look alike.
//! - `Atrium`: open space with pillars, often no wall in range.
//!
//! The seed changes the clutter and door positions, not the route.

use super::{Segment, World};
use crate::rng::Rng;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WorldKind {
    Office,
    Warehouse,
    Atrium,
}

impl WorldKind {
    pub const ALL: [WorldKind; 3] = [WorldKind::Office, WorldKind::Warehouse, WorldKind::Atrium];

    pub fn name(self) -> &'static str {
        match self {
            WorldKind::Office => "office",
            WorldKind::Warehouse => "warehouse",
            WorldKind::Atrium => "atrium",
        }
    }

    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|k| k.name() == name)
    }

    pub fn build(self, seed: u64) -> World {
        let mut b = Builder {
            segments: Vec::new(),
            rng: Rng::new(seed ^ 0xB10C_C0DE),
        };
        match self {
            WorldKind::Office => office(&mut b),
            WorldKind::Warehouse => warehouse(&mut b),
            WorldKind::Atrium => atrium(&mut b),
        }
    }
}

struct Builder {
    segments: Vec<Segment>,
    rng: Rng,
}

impl Builder {
    fn wall(&mut self, a: [f64; 2], b: [f64; 2]) {
        self.segments.push(Segment { a, b });
    }

    fn polyline(&mut self, points: &[[f64; 2]]) {
        for pair in points.windows(2) {
            self.wall(pair[0], pair[1]);
        }
    }

    fn rect(&mut self, x0: f64, y0: f64, x1: f64, y1: f64) {
        self.polyline(&[[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]);
    }

    /// A wall from `a` to `b` with openings given as `(start, end)` distances
    /// from `a`, sorted.
    fn wall_with_gaps(&mut self, a: [f64; 2], b: [f64; 2], gaps: &[(f64, f64)]) {
        let length = (b[0] - a[0]).hypot(b[1] - a[1]);
        let (ux, uy) = ((b[0] - a[0]) / length, (b[1] - a[1]) / length);
        let at = |s: f64| [a[0] + ux * s, a[1] + uy * s];
        let mut cursor = 0.0;
        for &(start, end) in gaps {
            if start > cursor {
                self.wall(at(cursor), at(start));
            }
            cursor = end;
        }
        if cursor < length {
            self.wall(at(cursor), b);
        }
    }

    /// A block of rooms between two corridors, opening onto the south or north
    /// side. Doors are placed irregularly, which is what gives a corridor any
    /// features along its axis at all.
    fn room_block(&mut self, x0: f64, y0: f64, x1: f64, y1: f64, door_side_south: bool) {
        let width = x1 - x0;
        let rooms = (width / 6.5).round().max(2.0) as usize;
        let room_width = width / rooms as f64;
        let mut gaps = Vec::new();
        for k in 0..rooms {
            let start = k as f64 * room_width;
            if self.rng.next_f64() < 0.7 {
                let door = start + self.rng.uniform(1.0, room_width - 2.2);
                gaps.push((door, door + 1.2));
            }
            if k > 0 {
                let x = x0 + start;
                self.wall([x, y0], [x, y1]);
            }
            // Furniture, so a room seen through a door is not an empty box.
            let cx = x0 + start + self.rng.uniform(1.2, room_width - 1.2);
            let cy = self.rng.uniform(y0 + 1.5, y1 - 1.5);
            let (hw, hh) = (self.rng.uniform(0.3, 0.9), self.rng.uniform(0.3, 0.7));
            self.rect(cx - hw, cy - hh, cx + hw, cy + hh);
        }
        if door_side_south {
            self.wall_with_gaps([x0, y0], [x1, y0], &gaps);
            self.wall([x0, y1], [x1, y1]);
        } else {
            self.wall_with_gaps([x0, y1], [x1, y1], &gaps);
            self.wall([x0, y0], [x1, y0]);
        }
        self.wall([x0, y0], [x0, y1]);
        self.wall([x1, y0], [x1, y1]);
    }

    fn finish(self, name: &'static str, route: Vec<[f64; 2]>) -> World {
        let (mut min, mut max) = ([f64::INFINITY; 2], [f64::NEG_INFINITY; 2]);
        for s in &self.segments {
            for p in [s.a, s.b] {
                for k in 0..2 {
                    min[k] = min[k].min(p[k]);
                    max[k] = max[k].max(p[k]);
                }
            }
        }
        World {
            name,
            segments: self.segments,
            route,
            min,
            max,
        }
    }
}

fn office(b: &mut Builder) -> World {
    let (w, h) = (64.0, 44.0);
    b.rect(0.0, 0.0, w, h);
    // Four blocks of rooms, separated by a ring corridor and a central cross.
    b.room_block(3.0, 3.0, 30.5, 20.5, false);
    b.room_block(33.5, 3.0, 61.0, 20.5, false);
    b.room_block(3.0, 23.5, 30.5, 41.0, true);
    b.room_block(33.5, 23.5, 61.0, 41.0, true);

    let (w0, w1, cx) = (1.5, 62.5, 32.0);
    let (s, n, cy) = (1.5, 42.5, 22.0);
    let route = vec![
        [w0, s],
        [w1, s],
        [w1, n],
        [w0, n],
        [w0, s],
        [cx, s],
        [cx, cy],
        [w0, cy],
        [w0, s],
        [w1, s],
        [w1, cy],
        [cx, cy],
        [cx, n],
        [w1, n],
        [w1, cy],
        [cx, cy],
    ];
    std::mem::take(b).finish("office", route)
}

fn warehouse(b: &mut Builder) -> World {
    let (w, h) = (48.0, 33.0);
    b.rect(0.0, 0.0, w, h);
    const PITCH: f64 = 2.7;
    const FIRST_X: f64 = 6.0;
    const BAYS: usize = 13;
    let rows = [6.0, 12.0, 18.0, 24.0, 30.0];
    for &y in &rows[..4] {
        for bay in 0..=BAYS {
            let x = FIRST_X + bay as f64 * PITCH;
            b.rect(x, y - 1.1, x + 0.12, y + 1.1);
            if bay == BAYS {
                break;
            }
            for side in [-1.0, 1.0] {
                // Most bays hold a pallet. Occasional empty bays are all that
                // tell one part of an aisle from another.
                if b.rng.next_f64() < 0.85 {
                    let depth = b.rng.uniform(0.6, 1.0);
                    let (y0, y1) = (y + side * 0.1, y + side * depth);
                    b.rect(x + 0.3, y0.min(y1), x + PITCH - 0.2, y0.max(y1));
                }
            }
        }
    }
    let (w0, w1) = (3.0, 45.0);
    let aisles = [3.0, 9.0, 15.0, 21.0, 27.0];
    let route = vec![
        [w0, aisles[0]],
        [w1, aisles[0]],
        [w1, aisles[1]],
        [w0, aisles[1]],
        [w0, aisles[2]],
        [w1, aisles[2]],
        [w1, aisles[3]],
        [w0, aisles[3]],
        [w0, aisles[4]],
        [w1, aisles[4]],
        [w1, aisles[0]],
        [w0, aisles[0]],
        [w0, aisles[4]],
        [w1, aisles[4]],
    ];
    std::mem::take(b).finish("warehouse", route)
}

fn atrium(b: &mut Builder) -> World {
    b.polyline(&[
        [0.0, 6.0],
        [8.0, 0.0],
        [44.0, 0.0],
        [54.0, 9.0],
        [54.0, 34.0],
        [46.0, 42.0],
        [10.0, 42.0],
        [0.0, 33.0],
        [0.0, 6.0],
    ]);
    // Central island, off-axis so opposite sides are distinguishable.
    b.polyline(&[
        [18.0, 14.0],
        [34.0, 12.0],
        [38.0, 22.0],
        [31.0, 29.0],
        [20.0, 27.0],
        [18.0, 14.0],
    ]);
    let mut pillars = Vec::new();
    for gx in 0..9 {
        for gy in 0..7 {
            let x = 5.0 + gx as f64 * 5.8 + b.rng.uniform(-0.7, 0.7);
            let y = 4.0 + gy as f64 * 5.6 + b.rng.uniform(-0.7, 0.7);
            let on_route = ROUTE_CLEARANCE.iter().any(|r| r.contains(x, y));
            let in_island = (16.5..39.5).contains(&x) && (10.5..30.5).contains(&y);
            let outside = x + y < 9.0
                || (54.0 - x) + y < 12.0
                || (54.0 - x) + (42.0 - y) < 11.0
                || x + (42.0 - y) < 12.0;
            if !on_route && !in_island && !outside && b.rng.next_f64() < 0.75 {
                pillars.push((x, y));
            }
        }
    }
    for (x, y) in pillars {
        let half = 0.3;
        b.rect(x - half, y - half, x + half, y + half);
    }
    let route = vec![
        [10.0, 7.0],
        [44.0, 7.0],
        [47.0, 21.0],
        [44.0, 35.0],
        [10.0, 35.0],
        [10.0, 7.0],
        [44.0, 7.0],
        [47.0, 21.0],
        [44.0, 35.0],
        [10.0, 35.0],
        [7.0, 21.0],
        [10.0, 7.0],
    ];
    std::mem::take(b).finish("atrium", route)
}

struct Keepout {
    x0: f64,
    y0: f64,
    x1: f64,
    y1: f64,
}

impl Keepout {
    fn contains(&self, x: f64, y: f64) -> bool {
        x >= self.x0 && x <= self.x1 && y >= self.y0 && y <= self.y1
    }
}

/// Bands along the atrium route that pillars must stay out of.
const ROUTE_CLEARANCE: [Keepout; 4] = [
    Keepout {
        x0: 6.0,
        y0: 5.0,
        x1: 48.0,
        y1: 9.5,
    },
    Keepout {
        x0: 6.0,
        y0: 32.5,
        x1: 48.0,
        y1: 37.0,
    },
    Keepout {
        x0: 4.0,
        y0: 5.0,
        x1: 12.0,
        y1: 37.0,
    },
    Keepout {
        x0: 42.0,
        y0: 5.0,
        x1: 50.0,
        y1: 37.0,
    },
];

impl Default for Builder {
    fn default() -> Self {
        Self {
            segments: Vec::new(),
            rng: Rng::new(0),
        }
    }
}
