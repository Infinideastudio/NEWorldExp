//! `core::game::worldgen` — terrain generation: rules + state.
//!
//! Two layers, both registry-aware:
//!
//! - **Rules** (Perlin/erosion + the private `init_generate`): pure
//!   functions that map coords + seed + `BaseBlocks` to chunk
//!   contents. The shaping rules are not exposed; callers reach them
//!   through [`TerrainGenerator::build_blocks`] or via the async
//!   pipeline.
//! - **Orchestrator** ([`TerrainGenerator`]): owns the chunk pipeline
//!   worker thread + the [`ChunkBuilder`] that encapsulates the
//!   sled-backed store, the height-map worker, the resolved `BaseBlocks`,
//!   and the per-world id translation table.
//!
//! Lives outside `core::world` because everything here knows about
//! `BaseBlocks` and the `BlockRegistry` — neither of which the
//! database half should depend on.

mod erosion;
mod perlin;
mod terrain_generator;

pub use terrain_generator::{TerrainGenerator, TileKey, world_tables_for, world_tables_for_seed};
