//! `RangeLoader` — streaming policy that keeps a chunk-window around a
//! moving centre resident in the [`World`] chunk store.
//!
//! Owns:
//! - The chunk-coord centre + `render_distance` (in chunks).
//! - Event-driven load / unload queues. When the centre moves or the
//!   render distance changes, the window cube is diffed against the
//!   previous one: `new − old` feeds the load queue, `old − new` the
//!   unload queue. [`Self::tick_chunk_loading`] drains up to
//!   `MAX_CHUNK_LOADS` / `MAX_CHUNK_UNLOADS` installs / evictions per
//!   call from the queue tails (nearest-first for loads, farthest-first
//!   for unloads), skipping entries that have already become resident /
//!   been evicted in the meantime. Steady state (empty queues) is O(1)
//!   per tick — no per-tick window or resident-set scans.
//! - A `HashMap<Vec3i, Lease>` keeping every chunk in the load
//!   window pinned. A `Lease` is the canonical pin in the new
//!   chunk-store protocol: while it's alive, an `unload_chunk` for
//!   that coord blocks in its drain step, so the world cannot
//!   evict a chunk we still need.
//! - `pending_tile_loads` (tile → waiting coords): chunks popped from
//!   the load queue whose terrain tile is still computing. Once
//!   [`TerrainGenerator::drain_results`] reports the tile ready, the
//!   parked coords re-enter the load queue.

use std::collections::{HashMap, HashSet};

use crate::core::game::worldgen::{TerrainGenerator, TileKey};
use crate::core::math::Vec3i;
use crate::core::world::{Lease, World, chunk_coord};

// ----------------------------------------------------------------------
//   Tuning constants
// ----------------------------------------------------------------------

/// Maximum chunks dispatched on a single load tick.
pub const MAX_CHUNK_LOADS: usize = 64;
/// Maximum chunks evicted on a single load tick.
pub const MAX_CHUNK_UNLOADS: usize = 64;
/// Extra chunks kept resident outside the render distance — the
/// renderer needs the 3×3×3 neighbourhood loaded before it can mesh
/// a chunk, so the streaming window has to reach one further.
pub const LOAD_RADIUS_BUFFER: i32 = 1;

// ----------------------------------------------------------------------
//   Cube helpers — window diffing
// ----------------------------------------------------------------------

/// Squared euclidean distance between chunk coords.
fn dist2(c: Vec3i, center: Vec3i) -> i32 {
    let d = c - center;
    d.x * d.x + d.y * d.y + d.z * d.z
}

/// True if `c` lies inside the Chebyshev cube `cube(center, radius)`.
fn in_cube(c: Vec3i, center: Vec3i, radius: i32) -> bool {
    let d = c - center;
    d.x.abs() <= radius && d.y.abs() <= radius && d.z.abs() <= radius
}

/// Every coord of the Chebyshev cube `cube(center, radius)`.
fn cube_coords(center: Vec3i, radius: i32) -> Vec<Vec3i> {
    if radius < 0 {
        return Vec::new();
    }
    let side = (2 * radius + 1) as usize;
    let mut out = Vec::with_capacity(side * side * side);
    for dx in -radius..=radius {
        for dy in -radius..=radius {
            for dz in -radius..=radius {
                out.push(center + Vec3i::new(dx, dy, dz));
            }
        }
    }
    out
}

/// Set difference `cube(a_center, a_radius) \ cube(b_center, b_radius)`.
fn cube_minus_cube(a_center: Vec3i, a_radius: i32, b_center: Vec3i, b_radius: i32) -> Vec<Vec3i> {
    cube_coords(a_center, a_radius)
        .into_iter()
        .filter(|c| !in_cube(*c, b_center, b_radius))
        .collect()
}

// ----------------------------------------------------------------------
//   RangeLoader
// ----------------------------------------------------------------------

pub struct RangeLoader {
    /// Chunk-coord centre.
    center: Vec3i,
    /// Render distance in chunks. The load window is
    /// `render_distance + LOAD_RADIUS_BUFFER`.
    render_distance: i32,
    pins: HashMap<Vec3i, Lease>,
    /// HUD counter — wraps on overflow; only ever read for display.
    unloaded_chunks: u32,
    /// Chunks waiting to be installed, sorted far→near: `pop` takes
    /// the nearest coord first.
    load_queue: Vec<Vec3i>,
    /// Chunks waiting to be evicted, sorted near→far: `pop` takes
    /// the farthest coord first.
    unload_queue: Vec<Vec3i>,
    /// Chunks popped for loading whose terrain tile was still
    /// computing, grouped by tile. Re-queued once the tile is ready.
    pending_tile_loads: HashMap<TileKey, Vec<Vec3i>>,
    /// Whether [`Self::set_center`] has run at least once; the very
    /// first call seeds the whole window (there is no previous cube).
    initialized: bool,
}

impl RangeLoader {
    pub fn new(render_distance: i32) -> Self {
        Self {
            center: Vec3i::new(0, 0, 0),
            render_distance,
            pins: HashMap::new(),
            unloaded_chunks: 0,
            load_queue: Vec::new(),
            unload_queue: Vec::new(),
            pending_tile_loads: HashMap::new(),
            initialized: false,
        }
    }

    pub fn center_ccoord(&self) -> Vec3i {
        self.center
    }

    pub fn render_distance(&self) -> i32 {
        self.render_distance
    }

    /// Total chunks evicted by this loader since construction (wraps).
    pub fn unloaded_chunks(&self) -> u32 {
        self.unloaded_chunks
    }

    /// Update the load-window centre. Diffs the new window cube
    /// against the previous one and feeds the difference into the
    /// load / unload queues. Block-coord input — floored to the
    /// containing chunk.
    pub fn set_center(&mut self, world: &World, center_block: Vec3i) {
        let new_center = chunk_coord(center_block);
        let radius = self.render_distance + LOAD_RADIUS_BUFFER;
        let (load_diff, unload_diff) = if self.initialized {
            (
                cube_minus_cube(new_center, radius, self.center, radius),
                cube_minus_cube(self.center, radius, new_center, radius),
            )
        } else {
            // Cold start — nothing is queued yet, so the whole window
            // is the load region and there is nothing to unload.
            (cube_coords(new_center, radius), Vec::new())
        };
        self.apply_window_diff(world, new_center, radius, load_diff, unload_diff);
        self.initialized = true;
        self.center = new_center;
    }

    /// Change the render distance at runtime. Diffs the resized window
    /// against the current one (same centre) and feeds the difference
    /// into the queues.
    pub fn set_render_distance(&mut self, world: &World, distance: i32) {
        if distance == self.render_distance {
            return;
        }
        let old_radius = self.render_distance + LOAD_RADIUS_BUFFER;
        let radius = distance + LOAD_RADIUS_BUFFER;
        let load_diff = cube_minus_cube(self.center, radius, self.center, old_radius);
        let unload_diff = cube_minus_cube(self.center, old_radius, self.center, radius);
        self.apply_window_diff(world, self.center, radius, load_diff, unload_diff);
        self.render_distance = distance;
    }

    /// Drain the load / unload queues: up to `MAX_CHUNK_LOADS` installs
    /// and `MAX_CHUNK_UNLOADS` evictions per call.
    pub fn tick_chunk_loading(&mut self, world: &World, terrain_gen: &mut TerrainGenerator) {
        let ready_tiles = terrain_gen.drain_results();
        if !ready_tiles.is_empty() {
            for tile in ready_tiles {
                // Terrain for `tile` finished computing — re-queue the
                // coords parked on it. Stale entries (window moved on,
                // already resident, duplicated) are filtered when popped.
                if let Some(coords) = self.pending_tile_loads.remove(&tile) {
                    self.load_queue.extend(coords);
                }
            }
            self.sort_queues(self.center);
        }

        let radius = self.render_distance + LOAD_RADIUS_BUFFER;
        let mut loaded = 0usize;
        while loaded < MAX_CHUNK_LOADS {
            let Some(cc) = self.load_queue.pop() else {
                break; // queue drained
            };
            if !in_cube(cc, self.center, radius) || world.is_loaded(cc) {
                continue; // drifted out of the window / already resident
            }
            let tile = TerrainGenerator::tile_for_chunk(cc);
            if !terrain_gen.has_tile(tile) {
                let waiting = self.pending_tile_loads.entry(tile).or_default();
                let is_new_request = waiting.is_empty();
                waiting.push(cc);
                if is_new_request {
                    terrain_gen.request_tile(tile);
                    std::thread::yield_now();
                }
                continue; // parked on the tile — does not consume the budget
            }
            world.load_chunk(cc, || terrain_gen.build_blocks(cc));
            world.mark_neighbour_chunks_updated(cc);
            if let Some(lease) = world.try_acquire_lease(cc) {
                self.pins.insert(cc, lease);
            }
            loaded += 1;
        }

        let mut unloaded = 0usize;
        while unloaded < MAX_CHUNK_UNLOADS {
            let Some(cc) = self.unload_queue.pop() else {
                break; // queue drained
            };
            if !world.is_loaded(cc) {
                continue; // already evicted
            }
            self.unload_one(world, cc);
            unloaded += 1;
        }
    }

    // ---- internal ----

    /// Merge a window diff into the queues: prune stale entries,
    /// append the new load / unload regions (deduped), then re-sort
    /// both queues around the new centre.
    fn apply_window_diff(
        &mut self,
        world: &World,
        new_center: Vec3i,
        radius: i32,
        load_diff: Vec<Vec3i>,
        unload_diff: Vec<Vec3i>,
    ) {
        // Entries that drifted back inside the window must not be
        // unloaded; entries that drifted out must not be loaded.
        self.unload_queue
            .retain(|c| !in_cube(*c, new_center, radius));
        self.load_queue.retain(|c| in_cube(*c, new_center, radius));

        let already_queued: HashSet<Vec3i> = self.load_queue.iter().copied().collect();
        for cc in load_diff {
            if world.is_loaded(cc) || already_queued.contains(&cc) {
                continue;
            }
            self.load_queue.push(cc);
        }
        let already_queued: HashSet<Vec3i> = self.unload_queue.iter().copied().collect();
        for cc in unload_diff {
            // Only chunks we pinned (i.e. loaded into the window) are
            // ours to evict.
            if !self.pins.contains_key(&cc) || already_queued.contains(&cc) {
                continue;
            }
            self.unload_queue.push(cc);
        }

        self.sort_queues(new_center);
    }

    /// Load queue: far→near, so `pop` yields the nearest coord first.
    /// Unload queue: near→far, so `pop` yields the farthest coord first.
    fn sort_queues(&mut self, center: Vec3i) {
        self.load_queue
            .sort_by_key(|c| std::cmp::Reverse(dist2(*c, center)));
        self.unload_queue.sort_by_key(|c| dist2(*c, center));
    }

    fn unload_one(&mut self, world: &World, cc: Vec3i) {
        // Drop our lease first so World::unload_chunk's drain step
        // doesn't deadlock against our own pin. The eviction
        // state-machine then runs to completion (CAS → wait_drain
        // → flush → remove).
        self.pins.remove(&cc);
        world.unload_chunk(cc);
        self.unloaded_chunks = self.unloaded_chunks.wrapping_add(1);
    }
}

// ----------------------------------------------------------------------
//   Tests — pure geometry
// ----------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cube_coords_side_length() {
        assert_eq!(cube_coords(Vec3i::new(0, 0, 0), 0).len(), 1);
        assert_eq!(cube_coords(Vec3i::new(5, -3, 2), 2).len(), 125);
        assert!(cube_coords(Vec3i::new(0, 0, 0), -1).is_empty());
    }

    #[test]
    fn cube_minus_itself_is_empty() {
        let c = Vec3i::new(7, -7, 7);
        assert!(cube_minus_cube(c, 3, c, 3).is_empty());
    }

    #[test]
    fn cube_minus_disjoint_is_whole_cube() {
        let whole = cube_minus_cube(Vec3i::new(0, 0, 0), 2, Vec3i::new(100, 100, 100), 2);
        assert_eq!(whole.len(), 125);
    }

    #[test]
    fn cube_minus_shifted_is_one_face() {
        // Sliding +1 in x: `new \ old` is exactly the +x face of the
        // new cube — a 5×5 slab at x = 0 + 2 + 1 = 3.
        let diff = cube_minus_cube(Vec3i::new(1, 0, 0), 2, Vec3i::new(0, 0, 0), 2);
        assert_eq!(diff.len(), 25);
        assert!(diff.iter().all(|c| c.x == 3));
        // The reverse difference is the −x face of the old cube.
        let rev = cube_minus_cube(Vec3i::new(0, 0, 0), 2, Vec3i::new(1, 0, 0), 2);
        assert_eq!(rev.len(), 25);
        assert!(rev.iter().all(|c| c.x == -2));
    }

    #[test]
    fn in_cube_bounds() {
        let c = Vec3i::new(2, -2, 2);
        assert!(in_cube(c, Vec3i::new(0, 0, 0), 2));
        assert!(!in_cube(Vec3i::new(3, 0, 0), Vec3i::new(0, 0, 0), 2));
    }
}
