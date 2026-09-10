//! Integration tests for `crate::worlds::World`.
//!
//! Moved out of `src/worlds/world/tests.rs` so test cases run against the
//! published API only and don't need to chdir into a scratch directory.
//! Each test gets its own absolute scratch path via [`common::ScratchDir`]
//! and constructs the world with [`World::new_at`].

mod common;

use std::sync::Arc;
use std::thread;
use std::time::Duration;

use neworld::core::blocks::BlockRegistry;
use neworld::core::game::base_blocks::{BaseBlocks, register_base_blocks};
use neworld::core::game::range_loader::RangeLoader;
use neworld::core::game::worldgen::{TerrainGenerator, world_tables_for};
use neworld::core::math::{Vec3i, Vec3u};
use neworld::core::world::{Chunk, World, block_coord, chunk_coord};

use common::ScratchDir;

fn make_registry() -> (Arc<BlockRegistry>, BaseBlocks) {
    let mut r = BlockRegistry::new();
    let base = register_base_blocks(&mut r);
    (Arc::new(r), base)
}

/// Build a world + terrain generator + range loader for a test.
fn build_world(
    scratch: &ScratchDir,
    name: &str,
    render_distance: i32,
) -> (
    World,
    TerrainGenerator,
    RangeLoader,
    BaseBlocks,
    Arc<BlockRegistry>,
) {
    let (registry, base) = make_registry();
    let tables = world_tables_for(scratch.path(), name, &registry).expect("world_tables_for");
    let world = World::new_at(scratch.path(), name.to_owned(), tables).expect("World::new_at");
    let terrain = TerrainGenerator::new(Arc::clone(&registry), base, 0);
    let loader = RangeLoader::new(render_distance);
    (world, terrain, loader, base, registry)
}

fn pump_until_loaded(
    world: &mut World,
    terrain: &mut TerrainGenerator,
    loader: &mut RangeLoader,
    target_count: usize,
) {
    for _ in 0..60_000 {
        if world.loaded_count() >= target_count {
            return;
        }
        loader.tick_chunk_loading(world, terrain);
        thread::sleep(Duration::from_millis(1));
    }
    panic!(
        "timed out loading {target_count} chunks; got {}",
        world.loaded_count()
    );
}

// ---------- coord helpers ----------

#[test]
fn chunk_coord_negative_arithmetic_shift() {
    // SIZE_LOG = 4, SIZE = 16. -1 >> 4 == -1.
    assert_eq!(chunk_coord(Vec3i::new(0, 0, 0)), Vec3i::new(0, 0, 0));
    assert_eq!(chunk_coord(Vec3i::new(15, 15, 15)), Vec3i::new(0, 0, 0));
    assert_eq!(chunk_coord(Vec3i::new(16, 16, 16)), Vec3i::new(1, 1, 1));
    assert_eq!(chunk_coord(Vec3i::new(-1, -1, -1)), Vec3i::new(-1, -1, -1));
    assert_eq!(
        chunk_coord(Vec3i::new(-16, -16, -16)),
        Vec3i::new(-1, -1, -1)
    );
    assert_eq!(
        chunk_coord(Vec3i::new(-17, -17, -17)),
        Vec3i::new(-2, -2, -2)
    );
}

#[test]
fn block_coord_modulo_bitmask() {
    assert_eq!(block_coord(Vec3i::new(0, 0, 0)), Vec3u::new(0, 0, 0));
    assert_eq!(block_coord(Vec3i::new(15, 7, 3)), Vec3u::new(15, 7, 3));
    assert_eq!(block_coord(Vec3i::new(16, 16, 16)), Vec3u::new(0, 0, 0));
    // -1 in two's-complement is ...11111111 → low 4 bits = 15.
    assert_eq!(block_coord(Vec3i::new(-1, -1, -1)), Vec3u::new(15, 15, 15));
    assert_eq!(block_coord(Vec3i::new(-16, -16, -16)), Vec3u::new(0, 0, 0));
}

// ---------- World ----------

#[test]
fn world_set_block_then_block_round_trips() {
    let scratch = ScratchDir::new("set-block");
    let (mut w, mut terrain, mut loader, base, registry) = build_world(&scratch, "set-block", 1);
    let mut q = neworld::core::game::block_update::BlockUpdateQueue::new();
    loader.set_center(&w, Vec3i::new(0, 0, 0));
    pump_until_loaded(&mut w, &mut terrain, &mut loader, 125);
    let coord = Vec3i::new(1, 2, 3);
    neworld::core::game::block_update::set_block(
        &w, &mut q, &base, &registry, coord, base.stone, false,
    );
    let got = w.block(coord).expect("loaded");
    assert_eq!(got.id, base.stone);
    let cc = chunk_coord(coord);
    assert!(w.is_loaded(cc));
}

#[test]
fn world_block_or_air_returns_air_for_unloaded_coord() {
    let scratch = ScratchDir::new("air");
    let (w, _terrain, _loader, _base, _registry) = build_world(&scratch, "air", 1);
    let far_off = Vec3i::new(100_000, 100_000, 100_000);
    let b = w.block(far_off);
    assert_eq!(b, None);
    assert!(w.block(far_off).is_none());
}

#[test]
fn world_loads_chunk_visible_via_read_txn() {
    use neworld::core::world::WorkingSet;
    let scratch = ScratchDir::new("chunk-lookup");
    let (mut w, mut terrain, mut loader, _base, _registry) =
        build_world(&scratch, "chunk-lookup", 1);
    loader.set_center(&w, Vec3i::new(0, 0, 0));
    pump_until_loaded(&mut w, &mut terrain, &mut loader, 125);
    let cc = Vec3i::new(0, 0, 0);
    assert!(w.is_loaded(cc));
    // The chunk is reachable through a 1-coord ReadTxn.
    let txn = w
        .begin_read_txn_sync(WorkingSet::Single(cc))
        .expect("loaded chunk");
    assert!(txn.chunk_at(cc).is_some());
}

#[test]
fn world_set_center_does_not_unload_existing_chunks() {
    // `set_center` only diffs the window and queues work; load /
    // unload execution is `tick_chunk_loading`'s job. After a slide
    // far past the old chunk, the chunk must still be reachable until
    // a `tick_chunk_loading` reaps it.
    let scratch = ScratchDir::new("set-center");
    let (mut w, mut terrain, mut loader, _base, _registry) = build_world(&scratch, "set-center", 1);
    loader.set_center(&w, Vec3i::new(0, 0, 0));
    pump_until_loaded(&mut w, &mut terrain, &mut loader, 125);
    let cc = Vec3i::new(0, 0, 0);
    assert!(w.is_loaded(cc));

    // Slide far away — origin chunk is now outside the window, but no
    // unload tick has happened yet.
    let far = Vec3i::new(10_000, 0, 10_000);
    loader.set_center(&w, far * Chunk::SIZE as i32);
    assert!(w.is_loaded(cc), "set_center alone must not unload");

    // After ticking, the now-distant chunk gets reaped.
    for _ in 0..8 {
        if !w.is_loaded(cc) {
            break;
        }
        loader.tick_chunk_loading(&w, &mut terrain);
        thread::sleep(Duration::from_millis(1));
    }
    assert!(!w.is_loaded(cc), "tick_chunk_loading should unload it");
}

#[test]
fn world_update_block_skips_when_neighbours_unloaded() {
    let scratch = ScratchDir::new("update-skip");
    // `render_distance = 0` + the load buffer (1) loads chunks at chunk
    // coords `-1..=1` on every axis. Block coord (31, 5, 5) sits in chunk
    // (1, 0, 0) — its +x neighbour at (32, 5, 5) is in chunk (2, 0, 0)
    // which stays unloaded, so `update_block` must bail.
    let (w, mut terrain, mut loader, base, registry) = build_world(&scratch, "update-skip", 0);
    let mut q = neworld::core::game::block_update::BlockUpdateQueue::new();
    loader.set_center(&w, Vec3i::new(0, 0, 0));
    loader.tick_chunk_loading(&w, &mut terrain);
    let coord = Vec3i::new(31, 5, 5);
    assert!(!neworld::core::game::block_update::update_block(
        &w, &mut q, &base, &registry, coord, true,
    ));
    assert!(q.is_empty());
}

#[test]
fn world_update_block_queues_neighbour_updates_when_all_loaded() {
    let scratch = ScratchDir::new("update-queue");
    let (mut w, mut terrain, mut loader, base, registry) = build_world(&scratch, "update-queue", 1);
    let mut q = neworld::core::game::block_update::BlockUpdateQueue::new();
    loader.set_center(&w, Vec3i::new(0, 0, 0));
    pump_until_loaded(&mut w, &mut terrain, &mut loader, 125);
    let coord = Vec3i::new(2, 3, 4);
    let drained_before = q.len();
    let ok =
        neworld::core::game::block_update::update_block(&w, &mut q, &base, &registry, coord, true);
    assert!(ok, "neighbours should be loaded");
    assert_eq!(
        q.len() - drained_before,
        6,
        "expected 6 neighbour updates queued"
    );
}

#[test]
fn world_tick_chunk_loading_is_idempotent() {
    // `render_distance + LOAD_RADIUS_BUFFER` = 2 chunks → 5³=125 candidates,
    // beyond `MAX_CHUNK_LOADS = 64` per tick. Drain to a steady state, then
    // assert one more tick adds nothing.
    let scratch = ScratchDir::new("idempotent");
    let (w, mut terrain, mut loader, _base, _registry) = build_world(&scratch, "idempotent", 1);
    loader.set_center(&w, Vec3i::new(0, 0, 0));
    let mut prev = 0;
    for _ in 0..16 {
        loader.tick_chunk_loading(&w, &mut terrain);
        if w.loaded_count() == prev {
            break;
        }
        prev = w.loaded_count();
    }
    let n1 = w.loaded_count();
    loader.tick_chunk_loading(&w, &mut terrain);
    let n2 = w.loaded_count();
    assert_eq!(n1, n2, "post-stable tick should not double-load");
}

#[test]
fn world_move_back_cancels_pending_unload() {
    // Walk away one chunk: the origin chunk is queued for unload (but
    // set_center never unloads synchronously). Walk back before any
    // tick: the queued unload must be pruned, so a tick leaves the
    // chunk resident.
    let scratch = ScratchDir::new("move-back");
    let (mut w, mut terrain, mut loader, _base, _registry) = build_world(&scratch, "move-back", 1);
    loader.set_center(&w, Vec3i::new(0, 0, 0));
    pump_until_loaded(&mut w, &mut terrain, &mut loader, 125);
    let cc = Vec3i::new(0, 0, 0);
    assert!(w.is_loaded(cc));

    let one_chunk = Chunk::SIZE as i32;
    loader.set_center(&w, Vec3i::new(one_chunk, 0, 0));
    assert!(w.is_loaded(cc), "set_center alone must not unload");

    loader.set_center(&w, Vec3i::new(0, 0, 0));
    loader.tick_chunk_loading(&w, &mut terrain);
    assert!(
        w.is_loaded(cc),
        "moving back must cancel the pending unload"
    );
}

#[test]
fn world_render_distance_change_loads_and_unloads() {
    let scratch = ScratchDir::new("rd-change");
    let (mut w, mut terrain, mut loader, _base, _registry) = build_world(&scratch, "rd-change", 1);
    loader.set_center(&w, Vec3i::new(0, 0, 0));
    pump_until_loaded(&mut w, &mut terrain, &mut loader, 125);

    // Shrink to 0: only the 3×3×3 = 27 core chunks remain.
    loader.set_render_distance(&w, 0);
    for _ in 0..16 {
        if w.loaded_count() <= 27 {
            break;
        }
        loader.tick_chunk_loading(&w, &mut terrain);
        thread::sleep(Duration::from_millis(1));
    }
    assert!(w.is_loaded(Vec3i::new(0, 0, 0)), "core must stay resident");
    assert!(
        !w.is_loaded(Vec3i::new(2, 0, 0)),
        "chunk outside the shrunk window must unload"
    );

    // Grow back: the ring re-loads (tiles are already cached).
    loader.set_render_distance(&w, 1);
    pump_until_loaded(&mut w, &mut terrain, &mut loader, 125);
}

#[test]
fn tiles_store_round_trips_raw_bytes() {
    use neworld::core::world::Store;
    let scratch = ScratchDir::new("tiles");
    let store = Store::open_at(&scratch.path().join("chunks.db")).expect("TilesStore::open_at");
    let coord = Vec3i::new(-2, 3, 17);
    let payload = b"hello-chunk-bytes-1234567890".to_vec();
    store.save(coord, &payload).expect("save");
    store.flush().expect("flush");
    let loaded = store.load(coord).expect("load");
    assert_eq!(loaded.as_deref(), Some(payload.as_slice()));
    let missing = store.load(Vec3i::new(99, 99, 99)).expect("load missing");
    assert!(missing.is_none());
}
