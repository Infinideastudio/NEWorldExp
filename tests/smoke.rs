//! End-to-end smoke test ([F7] in `docs/rust_migration.md` §5).
//!
//! Exercises the world create → mutate → save → drop → reopen → verify flow
//! using only public crate APIs. The wgpu / window layer is not involved —
//! the goal is to validate that the simulation + persistence layer survives
//! a process-style restart. Uses `World::new_at` with an absolute scratch
//! path so the test doesn't depend on cwd.

mod common;

use std::sync::Arc;
use std::thread;
use std::time::Duration;

use neworld::core::blocks::BlockRegistry;
use neworld::core::game::base_blocks::register_base_blocks;
use neworld::core::game::block_update::BlockUpdateQueue;
use neworld::core::game::commands::{CommandContext, CommandRegistry, register_base_commands};
use neworld::core::game::daylight_cycle::DaylightCycle;
use neworld::core::game::player::Player;
use neworld::core::game::range_loader::RangeLoader;
use neworld::core::game::worldgen::{TerrainGenerator, world_tables_for};
use neworld::core::math::Vec3i;
use neworld::core::world::World;

use common::ScratchDir;

fn pump_until_loaded(
    world: &mut World,
    terrain: &mut TerrainGenerator,
    loader: &mut RangeLoader,
    target_count: usize,
) {
    let mut idle = 0u32;
    let mut last_count = 0usize;
    while world.loaded_count() < target_count && idle < 60_000 {
        loader.tick_chunk_loading(world, terrain);
        let now = world.loaded_count();
        if now == last_count {
            idle += 1;
        } else {
            idle = 0;
            last_count = now;
        }
        thread::sleep(Duration::from_millis(1));
    }
}

#[test]
fn round_trip_world_through_set_block_save_reopen() {
    let scratch = ScratchDir::new("rt");

    let world_name = "smoke-rt".to_owned();
    let render_distance = 1;
    let seed = 0x00C0_FFEE;
    // World loads `render_distance + LOAD_RADIUS_BUFFER` (= 2 here) chunks
    // out from the centre so every meshable chunk has all 6 axis neighbours
    // loaded — see `worlds::world::LOAD_RADIUS_BUFFER`.
    let load_radius = render_distance + 1;
    let target_chunks = ((2 * load_radius + 1) as usize).pow(3);

    let coord = Vec3i::new(3, 5, 7);
    let stone_id;

    {
        let mut registry = BlockRegistry::new();
        let base = register_base_blocks(&mut registry);
        stone_id = base.stone;
        let registry = Arc::new(registry);

        let tables = world_tables_for(scratch.path(), &world_name, &registry).expect("tables");
        let mut world =
            World::new_at(scratch.path(), world_name.clone(), tables).expect("World::new_at");
        let mut terrain = TerrainGenerator::new(Arc::clone(&registry), base, seed);
        let mut loader = RangeLoader::new(render_distance);
        loader.set_center(&world, Vec3i::new(0, 0, 0));
        pump_until_loaded(&mut world, &mut terrain, &mut loader, target_chunks);
        assert_eq!(
            world.loaded_count(),
            target_chunks,
            "expected all chunks to load via async pipeline"
        );

        // Verify pre-mutation state — should not already be stone.
        let before = world.block(coord).expect("block loaded").id;
        assert_ne!(
            before, stone_id,
            "test coord must not already be stone in generated terrain"
        );

        let mut q = neworld::core::game::block_update::BlockUpdateQueue::new();
        neworld::core::game::block_update::set_block(
            &world, &mut q, &base, &registry, coord, stone_id, false,
        );
        assert_eq!(
            world.block(coord).expect("block still loaded").id,
            stone_id,
            "set_block did not take"
        );

        world.save_to_disk().expect("save_to_disk");
        // Drop here — the World's pipeline worker shuts down via Drop.
    }

    // Reopen the same world; the chunk that was modified must come back from
    // sled with its mutation intact (rather than being re-generated).
    {
        let mut registry = BlockRegistry::new();
        let base = register_base_blocks(&mut registry);
        let registry = Arc::new(registry);

        let tables = world_tables_for(scratch.path(), &world_name, &registry).expect("tables");
        let mut world =
            World::new_at(scratch.path(), world_name.clone(), tables).expect("World::new_at");
        let mut terrain = TerrainGenerator::new(Arc::clone(&registry), base, seed);
        let mut loader = RangeLoader::new(render_distance);
        loader.set_center(&world, Vec3i::new(0, 0, 0));
        pump_until_loaded(&mut world, &mut terrain, &mut loader, target_chunks);

        let after = world.block(coord).expect("block loaded after reopen").id;
        assert_eq!(
            after, stone_id,
            "set_block did not survive save/reopen round trip"
        );
    }
}

#[test]
fn slash_command_dispatch_through_full_stack() {
    let scratch = ScratchDir::new("cmd");

    let mut registry = BlockRegistry::new();
    let base = register_base_blocks(&mut registry);
    let registry = Arc::new(registry);

    let tables = world_tables_for(scratch.path(), "smoke-cmd", &registry).expect("tables");
    let mut world =
        World::new_at(scratch.path(), "smoke-cmd".to_owned(), tables).expect("World::new_at");
    let mut terrain = TerrainGenerator::new(Arc::clone(&registry), base, 0xDEAD_BEEF);
    let mut loader = RangeLoader::new(1);
    loader.set_center(&world, Vec3i::new(0, 0, 0));
    pump_until_loaded(&mut world, &mut terrain, &mut loader, 27);

    let mut commands = CommandRegistry::new();
    register_base_commands(&mut commands);

    // Build the gameplay-state bundle the new commands consume. The
    // smoke test owns its own player / queue / clock since it runs
    // outside `Game`.
    let mut player = Player::default();
    let mut queue = BlockUpdateQueue::new();
    let mut daylight = DaylightCycle::new();

    // `/setblock <x> <y> <z> <id>` — exercises argument parsing, world
    // mutation, and the registry dispatch path. The C++ command takes a
    // numeric block id (matching the Rust port).
    let stone_id_int: u16 = base.stone.get();
    let line = format!("/setblock 4 6 8 {stone_id_int}");
    let mut messages = Vec::<String>::new();
    {
        let mut ctx = CommandContext {
            world: &world,
            player: &mut player,
            queue: &mut queue,
            daylight: &mut daylight,
            base: &base,
            registry: &registry,
        };
        let ok = commands.execute_on(&line, &mut ctx, &mut messages);
        assert!(ok, "/setblock should succeed: {messages:?}");
    }
    assert_eq!(
        world.block(Vec3i::new(4, 6, 8)).expect("loaded").id,
        base.stone
    );

    // `/help` — read-only, should always succeed and push at least one line.
    messages.clear();
    {
        let mut ctx = CommandContext {
            world: &world,
            player: &mut player,
            queue: &mut queue,
            daylight: &mut daylight,
            base: &base,
            registry: &registry,
        };
        let ok = commands.execute_on("/help", &mut ctx, &mut messages);
        assert!(ok, "/help should succeed: {messages:?}");
    }
    assert!(!messages.is_empty(), "/help should produce output");
}
