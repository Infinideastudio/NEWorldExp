//! Game state — the orchestrator that owns the world, the camera, and the
//! GPU mesh cache, and drives the per-frame simulation.
//!
//! Implements `[F2]`–`[F6]` from `docs/rust_migration.md` §5:
//!
//! * `[F2]` — block selection raycast, breaking, and placing.
//! * `[F3]` — chat dispatch via [`crate::commands::CommandRegistry`].
//! * `[F5]` — async chunk loading: [`World::tick_chunk_loading_async`] +
//!   [`World::poll_load_results`] are pumped per frame; chunks stream into
//!   the slab over the first few frames.
//! * `[F6]` — async chunk meshing: dirty coords ship to a worker via
//!   [`MeshPipeline`] and the resulting `MeshOutput`s are uploaded to GPU as
//!   they arrive (see [`Self::pump_meshing`]).
//!
//! Player / camera split:
//! * The [`crate::core::game::player::Player`] owns the simulation state — position,
//!   velocity, orientation, gravity, hitbox collision, jump bookkeeping. Its
//!   `update(world)` runs in [`Self::tick_sim`] at 30 Hz.
//! * The [`Camera`] is a passive view: every frame, [`Self::tick_render`]
//!   mirrors `player.orientation()` into `camera.yaw/pitch`, and
//!   [`Self::write_frame_uniforms`] reads the interpolated eye position
//!   (`player.look_coord() - velocity * (1 - interp)`) into `camera.position`.
//!
//! Owns:
//!
//! * a [`World`] whose chunk window is sized by the configured render distance;
//! * a coord-keyed map of [`ChunkMesh`]es so async-mesh delivery can find
//!   the right slot in O(1);
//! * a [`Camera`] whose state is pushed in from the player every frame;
//! * the [`ChunkPipeline`] / [`ParticlePipeline`] / [`DepthTarget`] from `[D]`;
//! * a [`ParticleSystem`] (block-break particles spawn into it);
//! * a [`CommandRegistry`] populated by [`register_base_commands`];
//! * the current selection [`Hit`] and the current view-projection matrix
//!   (used by the HUD to draw a selection outline overlay).

pub mod base_blocks;
pub mod camera;
pub mod hud;
pub mod inventory;
pub mod raycast;

use std::collections::{BinaryHeap, HashMap, HashSet};
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use cgmath::{Matrix4, Point3, SquareMatrix, Vector3};

use crate::client::blocks::{BlockRenderRegistry, BlockTextureRegistry};
use crate::client::game::base_blocks::register_base_block_visuals;
use crate::client::render::chunk_rendering::{ChunkMesh, ChunkPipeline};
use crate::client::render::{
    CompositionFeatures, CompositionPipeline, DebugShadowPipeline, FrameUniforms, GBuffer,
    MeshInput, MeshOptions, MeshPipeline, PADDED_SIZE, PADDED_VOLUME, ParticleMesh,
    ParticlePipeline, SelectionPipeline, ShadowMap, ShadowPipeline, UnderwaterPipeline,
    UniformBuffer, mat4_to_array, padded_index,
};
use crate::core::blocks::{BlockData, BlockFaceMapping, BlockId, BlockRegistry, BlockState};
use crate::core::game::base_blocks::{BaseBlocks, register_base_blocks};
use crate::core::game::commands::{CommandRegistry, register_base_commands};
use crate::core::game::player::{BlockView, GameMode};
use crate::core::math::{Aabbd, Aabbf, Frustumf, Vec3d, Vec3i};
use crate::core::world::{Chunk, World, WorldError, chunk_coord};
use crate::input::{InputState, Key, MouseButton};
use crate::items::ItemStack;
use crate::particles::{Particle, ParticleSystem};
use crate::textures::Atlases;

/// Bounded-heap entry for the mesh-dispatch ordering. Local to game
/// because the mesher's heap is ordering on a different distance metric
/// than the streaming policy's load shells.
#[derive(Clone, Copy, PartialEq, Eq)]
struct ByDist {
    dist: i32,
    coord: Vec3i,
}
impl Ord for ByDist {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.dist.cmp(&other.dist)
    }
}
impl PartialOrd for ByDist {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

pub use camera::Camera;
pub use raycast::{Hit, RAYCAST_MAX};

/// Half-extent of the player-collision rejection box used by `try_place`.
/// Matches the C++ `Player::aabb` x/z half-width — placements that would
/// drop a solid block into the player's hitbox are silently rejected.
const PLAYER_HALF_EXTENT_HORIZ: f64 = 0.3;

/// Lifetime of a freshly-spawned chat message before it fades from the
/// auto-decay panel. `chat_open` overrides this — open chat always shows
/// recent history.
const CHAT_MESSAGE_LIFETIME_SECS: f32 = 5.0;

/// Number of debris particles spawned per block break.
const PARTICLES_PER_BREAK: u32 = 10;

/// Maximum delay between two W-presses that still counts as a sprint
/// double-tap. Matches the C++ `0.5` second window in `neworld.ixx::game_update`.
const SPRINT_DOUBLE_TAP_SECS: f64 = 0.5;

/// Per-pixel mouse-look gain in radians. The TOML `mouse_speed` value (≈0.1)
/// times this constant gives the effective rad/pixel mapping. Matches the
/// C++ formula `MouseSpeed * π / 180 * Δpx` evaluated for a 1° step.
const MOUSE_LOOK_RAD_PER_PIXEL: f64 = std::f64::consts::PI / 180.0;

/// Tilt of the sun's daily great circle, in radians, away from the
/// vertical XY plane. Positive values tilt the noon position toward
/// `+Z` — a mid-latitude site where the sun rides the southern sky
/// rather than passing directly overhead. ~20° (`0.35` rad) is a
/// modest amount that gives noticeable directional shadows without
/// looking like winter sun.
const SUN_TILT_RAD: f32 = 0.35;

/// Compute the current sun direction (TO the sun, world space) from
/// the integer game-tick clock. The sun's daily great circle is the
/// unit XY circle rotated around the X axis by [`SUN_TILT_RAD`], so:
///
/// * tick 0 → sun at `+X` horizon (sunrise),
/// * mid-day → sun near `+Y` zenith but offset by the tilt toward `+Z`
///   (mid-latitude "southern sky" position),
/// * sunset → `-X` horizon,
/// * midnight → mirrored beneath.
///
/// All time-of-day shading scalars (sun radiance ramp, ambient sky
/// colour, sky-light multiplier for basic mode) are derived from the
/// `y` component of the returned direction directly in the shaders —
/// no separate scalars need to ride in `FrameUniforms`.
///
/// `game_time` is interpreted modulo [`World::DAY_TICKS`].
fn time_of_day(game_time: u32) -> Vector3<f32> {
    let day = crate::core::game::daylight_cycle::DAY_TICKS as f32;
    let angle = std::f32::consts::TAU * (game_time as f32) / day;
    // Tilt the great circle around the X (sunrise/sunset) axis so the
    // sun's noon position sits a bit toward +Z instead of straight
    // overhead — gives directional shadows that aren't perfectly
    // axis-aligned with the world grid.
    let s = angle.sin();
    Vector3::new(angle.cos(), s * SUN_TILT_RAD.cos(), s * SUN_TILT_RAD.sin())
}

/// Maximum new mesh jobs to issue per frame. Caps the per-frame CPU spike
/// when many chunks land at once (e.g. on the first frame, or after a fast
/// teleport that invalidates everything in the load window).
const MAX_MESH_DISPATCHES_PER_FRAME: usize = 8;

/// Game-clock step per sim tick while F8 is held. 100× normal speed scrubs
/// a 20-minute day in ~12 seconds.
const FAST_FORWARD_TICKS_PER_SIM_TICK: u32 = 100;

/// `sqrt(3)` to f32 precision. Used by `write_frame_uniforms` to scale the
/// fog distance to the diagonal of the loaded chunk cube. `f32::sqrt` is
/// `const` in newer toolchains but not in our stable target, so we keep
/// this as a literal to avoid a runtime call per frame.
const SQRT_3_F32: f32 = 1.732_050_8;

/// All the game-side state, owned by `App`.
pub struct Game {
    pub world: World,
    /// Owns the chunk pipeline + terrain-generation state. World is
    /// registry-agnostic; this is the registry-aware counterpart that
    /// turns coords into chunks.
    pub terrain_generator: crate::core::game::worldgen::TerrainGenerator,
    /// Streaming policy: maintains pin guards on every chunk in a
    /// window around the player, drives loads/unloads through
    /// `World` + `TerrainGenerator`'s pipeline.
    pub range_loader: crate::core::game::range_loader::RangeLoader,
    /// Gameplay write/tick queue. Stub-driven this round.
    pub block_update_queue: crate::core::game::block_update::BlockUpdateQueue,
    /// In-game clock (formerly `World::game_time`).
    pub daylight_cycle: crate::core::game::daylight_cycle::DaylightCycle,
    /// Player state (formerly `World::player`).
    pub player: crate::core::game::player::Player,
    pub camera: Camera,
    /// Per-coord GPU mesh. Keyed by chunk coord so dirty-chunk rebuilds
    /// (`[F2]`) and async-mesh delivery (`[F6]`) can swap one entry in O(1),
    /// and chunk unload can drop the entry by coord directly.
    pub chunk_meshes: HashMap<Vec3i, ChunkMesh>,
    pub chunk_pipeline: ChunkPipeline,
    /// Deferred G-buffer (diffuse / normal / material / depth). Owns the
    /// world-pass depth target — the chunk MRT pass writes here, the
    /// composition pass samples from here, and forward passes (particles
    /// / selection / underwater) attach `gbuffer.depth_view()` so they
    /// depth-test against the world.
    pub gbuffer: GBuffer,
    /// Composition pipeline — reads the G-buffer + shadow map + noise and
    /// writes the lit color to the surface.
    pub composition_pipeline: CompositionPipeline,
    /// Shadow map — sized to `Config::shadow_res` once
    /// [`Self::apply_shadow_config`] runs. The composition shader gates
    /// real sampling on `FrameUniforms::shadow_params.x > 0`, which mirrors
    /// the [`advanced_render`] flag the host writes to `shadow_params`.
    pub shadow_map: ShadowMap,
    /// Depth-only chunk pipeline that fills [`Self::shadow_map`] from the
    /// sun's POV every frame when [`Self::advanced_render`] is true. Reuses
    /// `ChunkMesh` opaque buffers — separate pipeline because the chunk
    /// pipeline's bind-group layouts can't be shared across pipelines that
    /// expect different fragment outputs (depth-only here, MRT there).
    pub shadow_pipeline: ShadowPipeline,
    /// F3+M debug overlay — draws [`Self::shadow_map`]'s depth atlas as
    /// a square in the top-right of the screen, decoded via the C++
    /// 8-step binary search of `textureSampleCompare`. Only visible when
    /// [`Self::advanced_render`] AND [`Self::show_shadow_map`].
    pub debug_shadow_pipeline: DebugShadowPipeline,
    /// Toggled by F3+M while advanced rendering is on (mirrors C++
    /// `showShadowMap` from `neworld.ixx::62`). Force-cleared whenever
    /// `advanced_render` flips off, so the overlay can never end up on
    /// screen with the placeholder 1×1 texture behind it.
    pub show_shadow_map: bool,
    /// Master switch between advanced (deferred + shadow + composition)
    /// and basic (forward) rendering. Mirrors C++ `AdvancedRender` from
    /// `globals.ixx`. When `false`:
    /// * `record_world_pass` runs the basic forward path (opaque chunks
    ///   straight to surface, no G-buffer, no composition, no shadow).
    /// * The F3+M shadow-debug overlay is force-disabled because the
    ///   shadow map is the placeholder 1×1.
    /// * `FrameUniforms::shadow_params.x` is zeroed (the composition
    ///   shader reads it as a "shadow off" flag, but composition itself
    ///   doesn't run in basic mode anyway).
    ///
    /// Toggled live by [`Self::apply_shadow_config`] from
    /// `Config::advanced_render`.
    advanced_render: bool,
    /// Currently-applied shadow side length in pixels (`shadow_map.resolution`
    /// at the last `apply_shadow_config` call). Tracked separately so we can
    /// detect a real change without poking ShadowMap directly.
    shadow_res: u32,
    /// Currently-applied shadow distance cap, in chunks. Mirrors C++
    /// `min(MaxShadowDistance, RenderDistance)` — the shadow ortho box's
    /// half-extent in world blocks is `shadow_distance_chunks * CHUNK_SIZE`.
    shadow_distance_chunks: i32,
    pub particles: ParticleSystem,
    pub particle_mesh: ParticleMesh,
    pub particle_pipeline: ParticlePipeline,
    /// Wireframe outline for the currently-selected block. Drawn as a real
    /// 3-D pass against the world depth buffer, so solid geometry occludes
    /// it correctly and UI elements (egui) sit on top.
    /// Block-selection wireframe **and** the center-screen crosshair —
    /// the two were originally separate pipelines but share enough state
    /// (line-list topology, color-inversion blend, frame-uniform bind
    /// group, depth-write off) that they live in one
    /// [`SelectionPipeline`] now. Crosshair vertices use a screen-space
    /// kind that emits clip-space `z = 1.0`, beating the cube's reverse-Z
    /// `Greater` test against any geometry the world pass wrote.
    pub selection_pipeline: SelectionPipeline,
    /// Full-screen water-textured tint, drawn after the world pass when the
    /// player's eye is inside a water block (mirrors C++ behaviour). Toggle
    /// is pushed to the GPU only on transitions; a no-op cost the rest of
    /// the time.
    pub underwater_pipeline: UnderwaterPipeline,
    pub sun_dir: Vector3<f32>,
    /// Currently-selected block (raycast hit). Updated each tick.
    pub selected: Option<Hit>,
    /// Off-thread mesher ([F6]). Drained per frame by [`Self::pump_meshing`].
    mesh_worker: MeshPipeline,
    /// Coords currently in flight on the mesh worker.
    meshing_in_flight: HashSet<Vec3i>,
    /// View frustum for the current camera, refreshed in
    /// [`Self::write_frame_uniforms`]. Cached so [`Self::record_world_pass`]
    /// can cull `chunk_meshes` before walking the per-pass draw loops —
    /// big win at high render distance, where the encoder's `finish` cost
    /// is dominated by the recorded draw count.
    camera_frustum: Frustumf,
    /// Number of chunk meshes that survived camera frustum + distance
    /// culling on the last frame. Surfaced in the F3 debug overlay as
    /// "rendered" — distinct from `chunk_meshes.len()` which counts every
    /// uploaded mesh, visible or not.
    pub last_rendered_chunks: usize,
    /// Chat lines + the time they were posted.
    pub chat_messages: Vec<(String, Instant)>,
    /// `[F3]` — slash-command dispatch table.
    pub commands: CommandRegistry,
    /// Block registry shared with the mesher.
    registry: Arc<BlockRegistry>,
    /// Per-block face-texture lookup; client-only. Used to resolve atlas
    /// layers for the inventory icons, water/ice overlays, and the chunk
    /// mesher.
    render_registry: Arc<BlockRenderRegistry>,
    /// `BaseBlocks` ids needed by the raycast predicate and break/place logic.
    pub base_blocks: BaseBlocks,
    /// Tiny LCG state used to jitter break-particle positions / velocities.
    /// We deliberately avoid pulling in the `rand` crate — the spec calls for
    /// "a tiny LCG; do NOT add a `rand` dep".
    rng: u64,
    /// Tracks the last time the W key was pressed (`Instant`-monotonic, seconds
    /// since `Game::new`). Mirrors `wPressTimer` in `neworld.ixx::game_update`
    /// so a W-press within `SPRINT_DOUBLE_TAP_SECS` of the previous one
    /// triggers sprinting.
    last_w_press: Option<Instant>,
    /// Mouse sensitivity multiplier sourced from `Config.mouse_speed`. Pushed
    /// in by the app's `apply_config` each frame.
    pub mouse_speed: f64,
    /// Latest accumulator fraction of the simulation tick, in `[0, 1)`.
    /// Set by [`Self::write_frame_uniforms`] and consumed by
    /// [`Self::record_world_pass`] so particles can lerp between their
    /// pre- and post-tick positions for smooth motion at render rates
    /// faster than the 30 Hz simulation.
    tick_alpha: f32,
    /// Live meshing toggles (smooth lighting / merge-face / nice grass)
    /// captured into each `MeshInput` at submit time. Updated by
    /// [`Self::apply_mesh_config`] when the menu flips one of the flags;
    /// the change handler also drops every cached `ChunkMesh` and re-marks
    /// the loaded set dirty so the next pump rebuilds with the new options.
    mesh_options: MeshOptions,
}

impl Game {
    /// Build the world (chunks stream in asynchronously over the next few
    /// frames via [F5]/[F6]) and stand up the [D] pipelines.
    #[allow(clippy::too_many_arguments)] // app-level wiring is naturally wide
    pub fn new(
        device: &wgpu::Device,
        _queue: &wgpu::Queue,
        surface_format: wgpu::TextureFormat,
        surface_size: (u32, u32),
        registry: &Arc<BlockRegistry>,
        render_registry: &Arc<BlockRenderRegistry>,
        base_blocks: BaseBlocks,
        frame_uniforms: &UniformBuffer<FrameUniforms>,
        atlases: &Atlases,
        render_distance: i32,
        worlds_root: &Path,
        world_name: String,
        world_seed: u32,
    ) -> Result<Self, WorldError> {
        tracing::info!(
            ?worlds_root,
            world_name,
            world_seed,
            render_distance,
            "creating world"
        );

        // Build the per-world id translation tables (registry-aware,
        // outside the database).
        let tables = crate::core::game::worldgen::world_tables_for_seed(
            worlds_root,
            &world_name,
            registry,
            world_seed,
        )?;
        let world_seed = tables.metadata.seed;

        let world = World::new_at(worlds_root, world_name, tables)?;

        // Try to restore the player from disk; fall back to default.
        let mut player = crate::core::game::player::Player::default();
        let player_path = world.dir().join("player.bin");
        if player_path.exists() {
            match crate::core::game::player::Player::load_from(&player_path) {
                Ok(p) => {
                    player = p;
                    tracing::info!(?player_path, "player restored");
                }
                Err(err) => {
                    tracing::warn!(error = %err, ?player_path, "player load failed, using defaults")
                }
            }
        }

        let mut terrain_generator = crate::core::game::worldgen::TerrainGenerator::new(
            Arc::clone(registry),
            base_blocks,
            world_seed,
        );
        let daylight_cycle = crate::core::game::daylight_cycle::DaylightCycle::new();

        // Streaming policy + the gameplay block-update queue live
        // outside `World` (in `core::game`); the game owns instances
        // and threads them into the world for each tick.
        let mut range_loader = crate::core::game::range_loader::RangeLoader::new(render_distance);
        let block_update_queue = crate::core::game::block_update::BlockUpdateQueue::new();

        // Pin the load center to the player chunk so the first set of load
        // requests covers the player's actual surroundings. `tick_sim` keeps
        // the center following the player from then on.
        let player_world = player.coord();
        range_loader.set_center(
            &world,
            Vec3i::new(
                player_world.x.floor() as i32,
                player_world.y.floor() as i32,
                player_world.z.floor() as i32,
            ),
        );
        range_loader.tick_chunk_loading(&world, &mut terrain_generator);

        // Start in basic G-buffer shape (diffuse + depth only). The
        // first `apply_shadow_config` call after construction reads the
        // saved `Config::advanced_render` and switches to the full MRT
        // shape if needed.
        let gbuffer = GBuffer::new(device, surface_size.0.max(1), surface_size.1.max(1), false);
        let shadow_map = ShadowMap::new(device);
        let shadow_pipeline = ShadowPipeline::new(device, frame_uniforms, atlases);
        let debug_shadow_pipeline = DebugShadowPipeline::new(device, surface_format, &shadow_map);

        // Deferred chunk pipelines for both modes — basic + advanced
        // share the vertex format / bind groups, only fragment entry +
        // color targets differ. Always built; the right pair is bound
        // each frame based on `advanced_render`. Takes the gbuffer
        // because translucent pipelines bind opaque depth as a sampled
        // texture for shader-side discard.
        let chunk_pipeline = ChunkPipeline::new(device, frame_uniforms, atlases, &gbuffer);
        // Composition features start fully off — `apply_shadow_config`
        // pushes the live `Config` values on the first frame and rebuilds
        // the pipeline if any flag is on.
        let composition_pipeline = CompositionPipeline::new(
            device,
            surface_format,
            frame_uniforms,
            &gbuffer,
            &shadow_map,
            atlases,
            CompositionFeatures::default(),
        );
        // Forward overlays (particles / selection / underwater) draw onto
        // the surface AFTER composition with the G-buffer depth attached so
        // world geometry occludes them correctly.
        let particle_pipeline = ParticlePipeline::new(
            device,
            surface_format,
            GBuffer::DEPTH_FORMAT,
            frame_uniforms,
            atlases,
        );
        let selection_pipeline = SelectionPipeline::new(
            device,
            surface_format,
            GBuffer::DEPTH_FORMAT,
            frame_uniforms,
        );
        // Resolve the water atlas layer once at world creation. `face(0)` is
        // the top-face texture index — same as the C++ reference. Falls back
        // to layer 0 if `base_blocks.water` somehow isn't registered (only
        // a stripped-down test setup hits that path).
        let water_layer = u32::from(render_registry.face(base_blocks.water, 0).0);
        let underwater_pipeline = UnderwaterPipeline::new(
            device,
            surface_format,
            GBuffer::DEPTH_FORMAT,
            atlases,
            water_layer,
        );

        let particles = ParticleSystem::new();
        let mut particle_mesh = ParticleMesh::new();
        // Empty particle list, alpha is unused — pass 0.0.
        particle_mesh.rebuild(device, particles.particles(), 0.0);

        // Camera starts at the eye position; tick_render keeps it synced.
        let look = player.look_coord();
        let mut camera = Camera::new(look);
        camera.set_orientation(player.orientation());

        let mesh_worker = MeshPipeline::spawn(Arc::clone(registry), Arc::clone(render_registry));

        let mut commands = CommandRegistry::new();
        register_base_commands(&mut commands);

        Ok(Self {
            world,
            terrain_generator,
            range_loader,
            block_update_queue,
            daylight_cycle,
            player,
            camera,
            chunk_meshes: HashMap::new(),
            chunk_pipeline,
            gbuffer,
            composition_pipeline,
            shadow_map,
            shadow_pipeline,
            debug_shadow_pipeline,
            show_shadow_map: false,
            // Disabled until `apply_shadow_config` flips it. Default
            // matches C++ — `AdvancedRender` is off until the user opts in
            // via the shader-options menu.
            advanced_render: false,
            shadow_res: 1,
            // Default mirrors `Config::max_shadow_distance` (16); reset
            // each tick by `apply_shadow_config` to `min(max, render_dist)`.
            shadow_distance_chunks: 16,
            particles,
            particle_mesh,
            particle_pipeline,
            selection_pipeline,
            underwater_pipeline,
            sun_dir: time_of_day(0),
            selected: None,
            mesh_worker,
            meshing_in_flight: HashSet::new(),
            // Identity placeholder — the first `write_frame_uniforms`
            // overwrites this with the real frustum before any pass uses
            // it for culling. `Frustum::from_mvp(identity)` is the unit
            // cube and would falsely accept everything in the meantime.
            camera_frustum: Frustumf::from_mvp(&cgmath::Matrix4::identity()),
            last_rendered_chunks: 0,
            chat_messages: Vec::new(),
            commands,
            registry: Arc::clone(registry),
            render_registry: Arc::clone(render_registry),
            base_blocks,
            rng: 0x9E37_79B9_7F4A_7C15,
            last_w_press: None,
            mouse_speed: 0.1,
            tick_alpha: 0.0,
            // `apply_mesh_config` rewrites this on the first frame from the
            // live `Config`; the default keeps things sensible if the world
            // were ever ticked before the app's `apply_config` ran.
            mesh_options: MeshOptions {
                grass_id: base_blocks.grass,
                ..MeshOptions::default()
            },
        })
    }

    /// Reconcile the live mesh-options snapshot with `desired`. When any of
    /// the toggles flip, every cached `ChunkMesh` is dropped, every loaded
    /// chunk is re-marked dirty, and any in-flight mesh job is forgotten —
    /// the next [`Self::pump_meshing`] re-issues the loaded set against
    /// the new options. Cheap on the no-op path (single struct compare).
    pub fn apply_mesh_config(&mut self, desired: MeshOptions) {
        if self.mesh_options.smooth_lighting == desired.smooth_lighting
            && self.mesh_options.merge_face == desired.merge_face
            && self.mesh_options.nice_grass == desired.nice_grass
            && self.mesh_options.grass_id == desired.grass_id
        {
            return;
        }
        tracing::info!(
            smooth_lighting = desired.smooth_lighting,
            merge_face = desired.merge_face,
            nice_grass = desired.nice_grass,
            "mesh options changed; rebuilding all chunk meshes"
        );
        self.mesh_options = desired;
        self.chunk_meshes.clear();
        self.meshing_in_flight.clear();
        // Re-insert every loaded chunk into World's mesh-dirty set.
        // The next `pump_meshing` will see them through
        // `dirty_snapshot` and rebuild against the new rules.
        self.world.mark_all_loaded_for_remesh();
    }

    /// Render-rate per-frame update. `dt` is real elapsed seconds since the
    /// last render frame, **not** the simulation step length. Runs every
    /// frame so mouse-look + selection raycast respond immediately rather
    /// than at the 30 Hz simulation rate (which makes both feel laggy at
    /// high render FPS).
    ///
    /// `chat_open` and `inventory_open` and `paused` come from the UI layer;
    /// while any is true we suppress mouse-look + break/place + hotbar
    /// scrolling so menu / chat clicks don't double up as world edits.
    pub fn tick_render(
        &mut self,
        _dt: f32,
        input: &InputState,
        chat_open: bool,
        inventory_open: bool,
        paused: bool,
    ) {
        let ui_modal = chat_open || inventory_open || paused;
        if !ui_modal {
            // Mouse-look — apply directly to the player's orientation. The
            // C++ uses `MouseSpeed * π / 180 * Δpx`; we mirror that scaling
            // exactly so the TOML `mouse_speed` value carries the same
            // meaning across builds.
            let dx = f64::from(input.mouse_motion.x);
            let dy = f64::from(input.mouse_motion.y);
            if dx != 0.0 || dy != 0.0 {
                let gain = self.mouse_speed * MOUSE_LOOK_RAD_PER_PIXEL;
                let mut o = self.player.orientation();
                o.heading -= dx * gain;
                o.pitch -= dy * gain;
                self.player.set_orientation(o);
            }
        }

        // Camera mirrors the player's orientation each frame, regardless of
        // pause / chat — UI overlays still want a coherent view of the world
        // behind them. Camera position is set in `write_frame_uniforms` so it
        // can include the simulation interpolation factor.
        self.camera.set_orientation(self.player.orientation());
        self.camera.position = self.player.look_coord();

        // Selection raycast is cheap and feels nicer when it tracks the
        // camera at full frame rate.
        self.update_selection();

        // Mouse → break/place + hotbar scroll. Press edges only fire on the
        // frame they happened, so we have to consume them at frame rate too —
        // running this in `tick_sim` would drop clicks that happened mid-frame.
        if !ui_modal {
            if input.is_mouse_button_pressed(MouseButton::Left) {
                self.try_break();
            }
            if input.is_mouse_button_pressed(MouseButton::Right) {
                self.try_place();
            }
            // Hotbar cycling. Z/X step through the 10 slots; mouse wheel
            // scrolls (positive = up = previous slot, mirroring most voxel
            // games' convention).
            if input.is_key_pressed(Key::Z) {
                let cur = self.player.held_item_stack_index();
                self.player.set_held_item_stack_index((cur + 9) % 10);
            }
            if input.is_key_pressed(Key::X) {
                let cur = self.player.held_item_stack_index();
                self.player.set_held_item_stack_index((cur + 1) % 10);
            }
            let wheel = input.mouse_wheel_delta;
            if wheel.abs() >= 0.5 {
                let cur = self.player.held_item_stack_index();
                let steps = wheel.round() as i32;
                let next = ((cur as i32 - steps).rem_euclid(10)) as usize;
                self.player.set_held_item_stack_index(next);
            }
            // F3+M → toggle the shadow-map debug overlay. Only honored
            // while advanced rendering is on (shadow pass actually has
            // data); mirrors `neworld.ixx::545–551`. F3 is also the
            // debug-panel toggle in the C++ build, so we require F3 to
            // be HELD (not just pressed) so a debug-panel keypress
            // doesn't accidentally flip the overlay.
            if self.advanced_render && input.is_key_down(Key::F3) && input.is_key_pressed(Key::M) {
                self.show_shadow_map = !self.show_shadow_map;
            }
            // Mode toggles. F1 → game mode, F4 → cross-wall (creative only).
            if input.is_key_pressed(Key::F1) {
                let next = match self.player.game_mode() {
                    GameMode::Survival => GameMode::Creative,
                    GameMode::Creative => GameMode::Survival,
                };
                self.player.set_game_mode(next);
            }
            if input.is_key_pressed(Key::F4) {
                let cw = self.player.cross_wall();
                self.player.set_cross_wall(!cw);
            }
        }
    }

    /// Fixed-step simulation. `dt` is the simulation step length (1/30 s);
    /// callers run this in an accumulator loop in [`crate::app::App::frame`].
    /// Particle physics, chunk pipeline polling, the world's load center, and
    /// the player's physics tick all run here — anything where rate stability
    /// matters more than input latency.
    ///
    /// `first_tick_of_frame` is `true` for the first slice the accumulator
    /// drains in a given frame; subsequent ticks see `false`. Press-edge keys
    /// (Space, Enter…) are only processed on the first tick so a slow frame
    /// that drains multiple ticks doesn't fire a single keypress repeatedly.
    pub fn tick_sim(
        &mut self,
        dt: f32,
        input: &InputState,
        first_tick_of_frame: bool,
        chat_open: bool,
        inventory_open: bool,
        paused: bool,
    ) {
        let ui_modal = chat_open || inventory_open || paused;
        // Particle physics — gravity / drag / lifetime. Field-disjoint
        // borrow so the simulation can read world blocks via BlockView.
        {
            let view = WorldHitView::new(&self.world, &self.registry, &self.base_blocks);
            self.particles.tick(dt, &view);
        }

        // Player input + physics. Mirrors the C++ `game_update` flow:
        //   1. WSAD adds horizontal velocity in the player heading frame.
        //   2. Space / Shift drive jump / crouch.
        //   3. `player.update(world)` damps, applies gravity, and clips
        //      against block hitboxes.
        if !ui_modal {
            self.process_movement_input(input, first_tick_of_frame);
        } else {
            // Stop sprint if input is gated this tick; matches C++
            // `set_running(false)` when W is not held.
            self.player.set_running(false);
            self.last_w_press = None;
        }
        // Physics runs unconditionally — gravity should keep applying while
        // a menu is open, just like the C++ build.
        {
            let view = WorldHitView::new(&self.world, &self.registry, &self.base_blocks);
            self.player.update(&view);
        }

        // Drain the block-update queue. Mirrors the C++ `update_thread` loop
        // calling `process_block_updates()` once per tick. Drives the BFS
        // light-removal flushes and per-cell relaxation (e.g. after break /
        // place). Suppressed while paused so a paused game stays perfectly
        // frozen.
        if !paused {
            crate::core::game::block_update::process_block_updates(
                &self.world,
                &mut self.block_update_queue,
                &self.base_blocks,
                &self.registry,
            );
            // Random tick drives slow world dynamics: grass spread / smother
            // and future tickable blocks. Cheap (per-chunk fixed sample
            // count) but skipped while paused so a frozen world stays
            // visually frozen.
            crate::core::game::block_update::random_update(
                &self.world,
                &mut self.block_update_queue,
                &self.base_blocks,
                &self.registry,
            );
            // Advance the in-game clock. F8 (held) multiplies the
            // per-tick step so the user can scrub day/night quickly.
            let step = if input.is_key_down(Key::F8) {
                FAST_FORWARD_TICKS_PER_SIM_TICK
            } else {
                1
            };
            self.daylight_cycle.advance(step);
            // No drain into a separate dirty queue — `pump_meshing`
            // snapshots World's mesh-dirty set each frame and evicts only
            // the coords it actually meshed.
        }

        // Slide the chunk grid + height map to follow the player. Mirrors
        // C++ `update_chunk_lists`: only re-pivot when the player crosses a
        // chunk boundary so we don't churn the height-map cache every tick.
        let player_world = self.player.coord();
        let player_block = Vec3i::new(
            player_world.x.floor() as i32,
            player_world.y.floor() as i32,
            player_world.z.floor() as i32,
        );
        let player_chunk = chunk_coord(player_block);
        if self.range_loader.center_ccoord() != player_chunk {
            self.range_loader.set_center(&self.world, player_block);
        }

        // Drive the async chunk pipeline (F5): issue load/unload requests.
        // RangeLoader inserts every freshly-arrived chunk coord and those
        // of its 26 neighbours into World's mesh-dirty set via
        // `World::mark_neighbour_chunks_updated`, so the next
        // `dirty_snapshot` covers the load-completion
        // case automatically — no Game-side dirty-marking call needed.
        self.range_loader
            .tick_chunk_loading(&self.world, &mut self.terrain_generator);

        // Drop cached meshes for any coord that's no longer loaded — happens
        // every tick (not just on insert) so unload-on-walk reaps GPU
        // buffers as the player moves out of range. Borrow-split: collect
        // stale coords first, then mutate.
        let stale: Vec<Vec3i> = self
            .chunk_meshes
            .keys()
            .copied()
            .filter(|c| !self.world.is_loaded(*c))
            .collect();
        for c in stale {
            self.chunk_meshes.remove(&c);
        }
    }

    /// Process WSAD / Space / Shift / sprint detection and feed the player.
    /// Press-edge events only fire when `first_tick_of_frame` is true.
    fn process_movement_input(&mut self, input: &InputState, first_tick_of_frame: bool) {
        let player = &self.player;
        let heading = player.orientation().heading;
        let speed = player.speed();
        let flying = player.flying();
        let cross_wall = player.cross_wall();

        // Direction unit vectors derived purely from heading (the C++
        // walking model: WSAD is horizontal regardless of pitch).
        let (sin_h, cos_h) = heading.sin_cos();
        let forward = Vec3d::new(-sin_h, 0.0, -cos_h);
        let right = Vec3d::new(cos_h, 0.0, -sin_h);

        let mut delta_v = Vec3d::new(0.0, 0.0, 0.0);
        if input.is_key_down(Key::W) {
            delta_v += forward * speed;
        }
        if input.is_key_down(Key::S) {
            delta_v -= forward * speed;
        }
        if input.is_key_down(Key::D) {
            delta_v += right * speed;
        }
        if input.is_key_down(Key::A) {
            delta_v -= right * speed;
        }

        if delta_v.x != 0.0 || delta_v.z != 0.0 {
            let mut velocity = self.player.velocity();
            velocity += delta_v;
            // Walking speed cap — only when grounded / non-flying. Matches
            // C++: horizontal velocity magnitude clipped to `speed`.
            if !flying && !cross_wall {
                let xz_mag2 = velocity.x * velocity.x + velocity.z * velocity.z;
                if xz_mag2 > speed * speed {
                    let inv = speed / xz_mag2.sqrt();
                    velocity.x *= inv;
                    velocity.z *= inv;
                }
            }
            self.player.set_velocity(velocity);
        }

        // Sprint detection — double-tap W within `SPRINT_DOUBLE_TAP_SECS`.
        if first_tick_of_frame {
            let w_pressed = input.is_key_pressed(Key::W);
            if w_pressed {
                let now = Instant::now();
                let sprinted = match self.last_w_press {
                    Some(prev)
                        if now.duration_since(prev).as_secs_f64() <= SPRINT_DOUBLE_TAP_SECS =>
                    {
                        self.player.set_running(true);
                        true
                    }
                    _ => false,
                };
                self.last_w_press = if sprinted { None } else { Some(now) };
            }
        }
        // Releasing W stops the sprint (mirrors C++ `set_running(false)`).
        if !input.is_key_down(Key::W) {
            self.player.set_running(false);
            self.last_w_press = None;
        }

        // Jump — Space. on_jump consumes a mid-air jump only on press edge.
        if input.is_key_down(Key::Space) {
            let just_pressed = first_tick_of_frame && input.is_key_pressed(Key::Space);
            self.player.on_jump(just_pressed);
        }
        // Crouch — Shift held.
        if input.is_key_down(Key::LeftShift) || input.is_key_down(Key::RightShift) {
            self.player.on_crouch();
        }
    }

    /// Drive the async meshing pipeline (F6): submit up to N dirty coords to
    /// the mesh worker, then drain finished meshes back into `chunk_meshes`.
    /// Call once per frame from `App::frame`, after [`Self::tick_sim`]. Splits
    /// from the simulation tick because the upload step needs `&wgpu::Device`.
    ///
    /// Source of truth for "needs re-mesh" is World's `mesh_dirty` coord set.
    /// [`World::dirty_snapshot`] copies it out without modifying it (no
    /// hashing); we heap-pick the [`MAX_MESH_DISPATCHES_PER_FRAME`] closest
    /// dirty chunks (filtered on not-in-flight), submit them, and evict
    /// exactly the submitted coords via [`World::remove_dirty_chunks`] —
    /// no drain/requeue round-trip, so a dirty coord costs one SipHash at
    /// mark time and one at eviction, not one per frame. In-flight and
    /// neighbour-not-ready coords simply stay in the set. Unloaded chunks
    /// prune their coords in [`World::unload_chunk`], so the set never
    /// accumulates stale entries and no `is_loaded` guard is needed here.
    pub fn pump_meshing(&mut self, device: &wgpu::Device) {
        let player_world = self.player.coord();
        let player_chunk = chunk_coord(Vec3i::new(
            player_world.x.floor() as i32,
            player_world.y.floor() as i32,
            player_world.z.floor() as i32,
        ));

        let snapshot = self.world.dirty_snapshot();

        // Bounded max-heap keeps the closest `MAX_MESH_DISPATCHES_PER_FRAME`
        // dirty chunks. O(N log K) where N = dirty count, K = dispatch cap.
        let mut heap: BinaryHeap<ByDist> =
            BinaryHeap::with_capacity(MAX_MESH_DISPATCHES_PER_FRAME + 1);
        for cc in &snapshot {
            if self.meshing_in_flight.contains(cc) {
                continue;
            }
            let d = *cc - player_chunk;
            let dist = d.x * d.x + d.y * d.y + d.z * d.z;
            heap.push(ByDist { dist, coord: *cc });
            if heap.len() > MAX_MESH_DISPATCHES_PER_FRAME {
                heap.pop();
            }
        }
        // Closest-first iteration order so visible neighbourhood meshes
        // before distant chunks under back-pressure.
        let picked: Vec<Vec3i> = heap
            .into_sorted_vec()
            .into_iter()
            .map(|e| e.coord)
            .collect();

        // Submit each picked coord. `build_mesh_input` opens a 27-coord
        // ReadTxn over the chunk + its 26 neighbours; the txn errors
        // with `BeginError::NotLoaded` if any neighbour is missing,
        // which is the gate that prevents cracks at the render
        // boundary. Only successfully submitted coords are evicted from
        // World's set; everything else stays for a later frame. A coord
        // whose chunk was unloaded between snapshot and here was already
        // pruned by `World::unload_chunk`, so no stale-entry handling.
        let mut submitted: Vec<Vec3i> = Vec::with_capacity(picked.len());
        for &coord in &picked {
            let Some(input) = build_mesh_input(&self.world, coord, self.mesh_options) else {
                continue;
            };
            if self.mesh_worker.submit(input) {
                self.meshing_in_flight.insert(coord);
                submitted.push(coord);
            }
        }
        if !submitted.is_empty() {
            self.world.remove_dirty_chunks(&submitted);
        }

        // ---- drain finished meshes ----
        for done in self.mesh_worker.drain() {
            let coord = done.output.coord;
            self.meshing_in_flight.remove(&coord);
            // Re-resolve by coord — the chunk may have been unloaded while
            // its mesh was in flight on the worker.
            if !self.world.is_loaded(coord) {
                self.chunk_meshes.remove(&coord);
                continue;
            }
            if done.output.opaque.is_empty() && done.output.translucent.is_empty() {
                self.chunk_meshes.remove(&coord);
                continue;
            }
            let mesh = ChunkMesh::upload(device, &done.output);
            self.chunk_meshes.insert(coord, mesh);
        }
    }

    /// React to a window resize: recreate every G-buffer attachment
    /// and rebuild the composition + chunk-pipeline bind groups that
    /// reference the fresh views (composition reads both layers; the
    /// translucent chunk pipeline samples opaque depth).
    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        self.gbuffer.resize(device, width.max(1), height.max(1));
        self.composition_pipeline
            .rebuild_gbuffer_bind_groups(device, &self.gbuffer);
        self.chunk_pipeline
            .rebuild_opaque_depth_bind_group(device, &self.gbuffer);
    }

    /// Reconcile the shadow-map resolution + enable flag with `Config`.
    /// Called every frame from the app's `apply_config`.
    ///
    /// * Resizes [`Self::shadow_map`] when `shadow_res` changes — also
    ///   rebuilds the composition pass's aux bind group so its shadow
    ///   texture binding points at the fresh view (the old view dies
    ///   with the old texture).
    /// * Caches `min(max_shadow_distance, render_distance)` in
    ///   [`Self::shadow_distance_chunks`] so the shadow ortho box can
    ///   recompute its half-extent each frame without re-reading the
    ///   config lock.
    /// * Toggles [`Self::advanced_render`] from `advanced_render`. While
    ///   off, `record_world_pass` skips the shadow pass entirely and
    ///   `write_frame_uniforms` zeroes `shadow_params.x` so composition
    ///   masks PCF sampling.
    #[allow(clippy::too_many_arguments)] // mirrors `Config` shader-options
    pub fn apply_shadow_config(
        &mut self,
        device: &wgpu::Device,
        atlases: &Atlases,
        advanced_render: bool,
        shadow_res: i32,
        max_shadow_distance: i32,
        soft_shadow: bool,
        volumetric_clouds: bool,
        ambient_occlusion: bool,
    ) {
        // Composition feature flags. When advanced rendering is off, the
        // composition pipeline isn't run at all, but rebuilding it with
        // `default()` (all-off) is a cheap consistency move so the
        // pipeline reflects the live config.
        let comp_features = if advanced_render {
            CompositionFeatures {
                soft_shadow,
                volumetric_clouds,
                ambient_occlusion,
            }
        } else {
            CompositionFeatures::default()
        };
        self.composition_pipeline
            .rebuild_with_features(device, comp_features);
        // Toggle the G-buffer between basic (diffuse + depth) and
        // advanced (full MRT) shapes. Idempotent when the mode hasn't
        // changed; reallocates the optional normal/material attachments
        // and rebuilds the composition's gbuffer bind groups otherwise.
        let mode_changed = self.gbuffer.is_advanced() != advanced_render;
        if mode_changed {
            self.gbuffer.set_advanced(device, advanced_render);
            self.composition_pipeline
                .rebuild_gbuffer_bind_groups(device, &self.gbuffer);
            // Translucent chunk pipeline samples opaque depth; the
            // gbuffer reallocation replaced the underlying texture.
            self.chunk_pipeline
                .rebuild_opaque_depth_bind_group(device, &self.gbuffer);
        }
        self.advanced_render = advanced_render;
        // Force-clear the debug overlay when shadows are off so a stale
        // toggle doesn't paint the placeholder 1×1 texture into the
        // top-right of the screen (mirrors C++ `else { showShadowMap =
        // false; }` at `neworld.ixx::551`).
        if !advanced_render {
            self.show_shadow_map = false;
        }
        let render_distance = self.range_loader.render_distance();
        // C++ `shadow_distance() = min(MaxShadowDistance, RenderDistance)`.
        self.shadow_distance_chunks = max_shadow_distance.min(render_distance).max(1);

        // Treat anything < 256 as "off" — keeps the placeholder 1×1
        // texture in place for users who flip advanced rendering off.
        // Clamp to a sane upper bound so a typo in the TOML can't try to
        // allocate a 32k² texture.
        let want_res = if advanced_render {
            shadow_res.clamp(256, 4096) as u32
        } else {
            1
        };
        if want_res != self.shadow_res {
            self.shadow_map.resize(device, want_res);
            self.shadow_res = want_res;
            // The shadow texture view changed — rebind both consumers
            // (composition + the F3+M debug overlay) so they sample the
            // fresh view, not a dropped one.
            self.composition_pipeline.rebuild_advanced_aux_bind_group(
                device,
                &self.shadow_map,
                atlases,
            );
            self.debug_shadow_pipeline
                .rebuild_shadow_bind_group(device, &self.shadow_map);
            tracing::info!(resolution = want_res, advanced_render, "shadow map resized");
        }
    }

    /// Update `FrameUniforms` with the latest camera + sun + screen-size
    /// snapshot. Call once per frame, before [`Self::record_world_pass`].
    /// `tick_alpha` is the `[0, 1)` accumulator fraction since the last
    /// simulation tick — used to interpolate the eye position so render-rate
    /// motion is smooth even though physics ticks at 30 Hz. C++ flow: see
    /// `neworld.ixx::render_scene` (`view_coord = look - velocity * (1-α)`).
    pub fn write_frame_uniforms(
        &mut self,
        queue: &wgpu::Queue,
        frame_uniforms: &UniformBuffer<FrameUniforms>,
        surface_size: (u32, u32),
        elapsed: f32,
        tick_alpha: f32,
    ) {
        // Cache `tick_alpha` so `record_world_pass` can lerp particle
        // positions on the same fraction the camera uses.
        self.tick_alpha = tick_alpha.clamp(0.0, 1.0);
        // Interpolated eye position. Mirrors the C++ formula from
        // `render_scene`: at α=0 (tick just happened) we show the pre-tick
        // position so motion is continuous; at α=1 (about to tick again) we
        // show the post-tick position.
        let player = &self.player;
        let look = player.look_coord();
        let velocity = player.velocity();
        let alpha = f64::from(tick_alpha.clamp(0.0, 1.0));
        let eye = look - velocity * (1.0 - alpha);
        self.camera.position = eye;
        self.camera.set_orientation(player.orientation());

        let (w, h) = surface_size;
        let aspect = w.max(1) as f32 / h.max(1) as f32;
        let view = self.camera.view_matrix();
        let proj = self.camera.proj_matrix(aspect);
        let view_proj = proj * view;
        // Refresh the cached frustum — `record_world_pass` reads it to
        // cull `chunk_meshes` before each per-pass draw loop.
        self.camera_frustum = Frustumf::from_mvp(&view_proj);

        // Push the current selection to the wireframe pipeline. Doing it
        // here keeps the upload colocated with the per-frame uniform writes
        // so the data is consistent within a single frame's render.
        match self.selected {
            Some(hit) => self.selection_pipeline.set_block(queue, hit.coord),
            None => self.selection_pipeline.clear(),
        }
        // Underwater check — match C++ `block_or_air(int_view_coord).id ==
        // water` from `neworld.ixx:784`. The eye coord is the same value
        // we just lerped above, so `floor` agrees with what the camera
        // actually sees.
        let eye_block = Vec3i::new(
            eye.x.floor() as i32,
            eye.y.floor() as i32,
            eye.z.floor() as i32,
        );
        let underwater =
            self.world.block(eye_block).unwrap_or_default().id == self.base_blocks.water;
        self.underwater_pipeline.set_enabled(queue, underwater);

        // Inverse view-projection — composition unprojects screen-space +
        // depth to world position. `cgmath::Matrix4::invert` returns
        // `Option`; the perspective × translation we just composed is
        // always invertible, but fall back to the identity for safety so a
        // numerical edge case doesn't blow up the frame.
        let inv_view_proj = view_proj.invert().unwrap_or_else(Matrix4::identity);

        // Repeat trick (mirrors C++ `repeat = 25600`): wrap world coords
        // into a manageable range so SSR / volumetric clouds don't lose
        // precision far from the world origin. `25600 = 1600 * 16`.
        const REPEAT: i32 = 25600;
        let coord = self.camera.position;
        let coord_int = [
            coord.x.floor() as i32,
            coord.y.floor() as i32,
            coord.z.floor() as i32,
            0,
        ];
        let coord_mod = [
            coord_int[0].rem_euclid(REPEAT),
            coord_int[1].rem_euclid(REPEAT),
            coord_int[2].rem_euclid(REPEAT),
            0,
        ];
        let coord_frac = [
            (coord.x - f64::from(coord_int[0])) as f32,
            (coord.y - f64::from(coord_int[1])) as f32,
            (coord.z - f64::from(coord_int[2])) as f32,
            0.0,
        ];

        // Shadow view-projection — places the player at origin looking
        // along `-sun_dir`, then maps a 4L-side ortho box (half-extent
        // `length` in xy, ±2L in z) into wgpu reversed-Z `[0, 1]` clip
        // space. Vertices entering the shadow shader are world-space (the
        // chunk pipeline pre-bakes `coord * CHUNK_SIZE` at upload time),
        // so the matrix has to do its own world→camera translation —
        // unlike the C++ build where `u_translation = chunk - camera` is
        // pushed per-chunk and the shadow MVP is just the rotation +
        // ortho. Mirrors `Renderer::getShadowMatrix` from
        // `rendering.ixx::229` modulo that delta. Skipped (identity) when
        // shadows are off so the value is at least invertible if anything
        // tries to inverse-project it.
        // Refresh the cached sun direction + sky-light multiplier from
        // the in-game clock. Stored on `Game` so the shadow matrix and
        // the uniform write below see the same vector.
        self.sun_dir = time_of_day(self.daylight_cycle.game_time());

        let shadow_view_proj = if self.advanced_render {
            let length = (self.shadow_distance_chunks as f32) * (Chunk::SIZE as f32);
            shadow_matrix(self.camera.position, self.sun_dir, length)
        } else {
            Matrix4::identity()
        };

        let mut u = FrameUniforms::default();
        u.view = mat4_to_array(view);
        u.proj = mat4_to_array(proj);
        u.view_proj = mat4_to_array(view_proj);
        u.inv_view_proj = mat4_to_array(inv_view_proj);
        u.shadow_view_proj = mat4_to_array(shadow_view_proj);
        u.camera_pos = [
            self.camera.position.x as f32,
            self.camera.position.y as f32,
            self.camera.position.z as f32,
            1.0,
        ];
        u.sun_dir = [self.sun_dir.x, self.sun_dir.y, self.sun_dir.z, 0.0];
        u.screen_size = [w as f32, h as f32];
        u.time = elapsed;
        // Fog scales with the live render distance: `fog_end` lands just past
        // the diagonal of the loaded cube (so corner chunks fade smoothly into
        // sky), `fog_start` 65% of the way there for a generous fade band.
        let r = self.range_loader.render_distance() as f32;
        let chunk = Chunk::SIZE as f32;
        u.fog_end = (r + 0.5) * SQRT_3_F32 * chunk;
        u.fog_start = u.fog_end * 0.65;
        u.render_distance = r * chunk;
        // `shadow_params = (resolution, distance, fisheye_factor, _)`.
        // Resolution doubles as the composition shader's "shadows on"
        // gate (`if (shadow_params.x > 0)`); zeroed while disabled so
        // composition skips PCF sampling. Fisheye factor matches C++
        // `SetUniforms` (`u_shadow_fisheye_factor = 0.8f`).
        // `shadow_params.w` doubles as the "camera is inside water"
        // flag for the composition shader's SSR path. When set, water
        // surfaces skip the Schlick mirror so the user can see through
        // the surface to the world beyond — otherwise the back-facing
        // water boundaries (where `cos_theta < 0`) get fresnel = 1
        // and the player's underwater view turns into a solid water
        // tint.
        let inside_water_flag = if underwater { 1.0 } else { 0.0 };
        u.shadow_params = if self.advanced_render {
            [
                self.shadow_res as f32,
                (self.shadow_distance_chunks as f32) * (Chunk::SIZE as f32),
                0.8,
                inside_water_flag,
            ]
        } else {
            [0.0, 0.0, 0.8, inside_water_flag]
        };
        // Atlas-layer indices for materials composition needs to detect
        // per-pixel — currently water (slot 0) and ice (slot 1). The
        // chunk shader writes the per-face atlas layer to the
        // G-buffer's material attachment, so composition can compare
        // these against `material` to apply screen-space reflection
        // on water / ice without per-block-id constants in the
        // shader. `face(0)` is the surface (top) face's texture index
        // — water/ice use the same texture across all faces.
        let water_layer = u32::from(self.render_registry.face(self.base_blocks.water, 0).0);
        let ice_layer = u32::from(self.render_registry.face(self.base_blocks.ice, 0).0);
        u.material_layers = [water_layer, ice_layer, 0, 0];
        u.player_coord_int = coord_int;
        u.player_coord_mod = coord_mod;
        u.player_coord_frac = coord_frac;
        frame_uniforms.write(queue, &u);
    }

    /// Unified deferred pre-overlay path for both basic and advanced
    /// modes:
    ///
    /// 1. (advanced only) shadow pass populates the sun-POV depth atlas.
    /// 2. Opaque G-buffer pass — clears the gbuffer (1 color target in
    ///    basic mode, 3 in advanced) + depth, then dispatches the
    ///    mode-specific opaque chunk pipeline.
    /// 3. Translucent G-buffer pass — loads the gbuffer and alpha-blends
    ///    water / ice / leaves into the diffuse target via the
    ///    mode-specific translucent chunk pipeline; depth-writes the
    ///    water surface so SSR raymarches in advanced mode find it.
    /// 4. Composition pass — manually blends the two layers (front-most
    ///    translucent over opaque, sky behind any missing layer).
    ///    Basic copies linearly + tonemaps the sky; advanced runs full
    ///    lambert + shadow PCF + ambient + emissive + ACES on each
    ///    layer.
    ///
    /// All four steps run in both modes; the only branches are the
    /// shadow skip in basic mode and the per-pipeline mode flag.
    fn record_pre_overlay(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        color_view: &wgpu::TextureView,
        camera_visible: &[&ChunkMesh],
        shadow_visible: &[&ChunkMesh],
    ) {
        let advanced = self.advanced_render;
        let clear_color = wgpu::Color::TRANSPARENT;

        // ---- 0. Shadow pass (advanced only) ----
        if advanced {
            self.shadow_pipeline
                .record(encoder, &self.shadow_map, shadow_visible.iter().copied());
        }

        // ---- 1. Opaque G-buffer pass — opaque layer attachments + opaque depth.
        {
            let opaque_color = self
                .gbuffer
                .opaque
                .color_attachments(wgpu::LoadOp::Clear(clear_color), wgpu::StoreOp::Store);
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("game.gbuffer_opaque_pass"),
                color_attachments: &opaque_color,
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: self.gbuffer.opaque_depth_view(),
                    depth_ops: Some(wgpu::Operations {
                        // Reversed-Z far-plane clear.
                        load: wgpu::LoadOp::Clear(0.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            self.chunk_pipeline.begin_opaque(&mut pass, advanced);
            for cm in camera_visible {
                cm.draw_opaque(&mut pass);
            }
        }

        // ---- 2. Translucent G-buffer pass — translucent layer
        // attachments + translucent depth (its own buffer, cleared
        // each frame so the front-most translucent fragment wins).
        // The fragment shader additionally samples the opaque depth
        // (bound as group 2 in the chunk pipeline) and discards
        // fragments behind opaque.
        {
            let translucent_color = self
                .gbuffer
                .translucent
                .color_attachments(wgpu::LoadOp::Clear(clear_color), wgpu::StoreOp::Store);
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("game.gbuffer_translucent_pass"),
                color_attachments: &translucent_color,
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: self.gbuffer.translucent_depth_view(),
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(0.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            self.chunk_pipeline.begin_translucent(&mut pass, advanced);
            for cm in camera_visible {
                cm.draw_translucent(&mut pass);
            }
        }

        // ---- 3. Composition pass (G-buffer → surface) ----
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("game.composition_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: color_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        // Composition writes every pixel; clear to
                        // transparent black as a safety net for
                        // fullscreen-resize races.
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            self.composition_pipeline.record(&mut pass, advanced);
        }
    }

    /// Record the world render into `encoder`, writing into `color_view`.
    /// Both modes follow the same shape: optional shadow pass + opaque
    /// G-buffer + translucent G-buffer + composition + forward
    /// overlays. Mode differences live entirely in the mode-aware
    /// pipelines and the gbuffer's attachment count — see
    /// [`Self::record_pre_overlay`].
    pub fn record_world_pass(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        color_view: &wgpu::TextureView,
        surface_size: (u32, u32),
    ) {
        // Refresh the debug-overlay quad to the live aspect ratio. The
        // shader needs `xi = 1 - h/w` re-evaluated whenever the surface
        // resizes; the write is a single u32×4 buffer copy so paying it
        // every frame is cheap.
        if self.advanced_render && self.show_shadow_map {
            self.debug_shadow_pipeline
                .update_layout(queue, surface_size);
        }
        // Particle vertex buffer — rebuild per-frame from the current
        // particle list (cheap when empty; a one-buffer write otherwise).
        // `tick_alpha` lerps each particle between its pre- and post-tick
        // position so motion stays smooth at sub-tick frame rates.
        self.particle_mesh
            .rebuild(device, self.particles.particles(), self.tick_alpha);

        // Pre-cull `chunk_meshes` once per frame — every per-pass loop
        // below iterates the filtered slices instead of the full map.
        // Frustum culling alone typically drops 70-80% of recorded draws
        // (a ~70° FOV camera covers ~1/4 of the sphere); the chebyshev
        // distance filter trims the small ring of formerly-loaded meshes
        // that briefly outlives the load window after a player slide.
        // At rd=32 with all chunks meshed this directly cuts the
        // per-frame `CommandEncoder::finish` cost, which scales with
        // recorded command count.
        let player_world = self.player.coord();
        let player_chunk = chunk_coord(Vec3i::new(
            player_world.x.floor() as i32,
            player_world.y.floor() as i32,
            player_world.z.floor() as i32,
        ));
        let render_distance = self.range_loader.render_distance();
        let shadow_distance = self.shadow_distance_chunks;
        let chunk_size = Chunk::SIZE as f32;
        let camera_visible: Vec<&ChunkMesh> = self
            .chunk_meshes
            .values()
            .filter(|cm| {
                let d = cm.coord - player_chunk;
                d.x.abs() <= render_distance
                    && d.y.abs() <= render_distance
                    && d.z.abs() <= render_distance
            })
            .filter(|cm| {
                let lo = cgmath::Vector3::new(
                    cm.coord.x as f32 * chunk_size,
                    cm.coord.y as f32 * chunk_size,
                    cm.coord.z as f32 * chunk_size,
                );
                let hi = lo + cgmath::Vector3::new(chunk_size, chunk_size, chunk_size);
                self.camera_frustum.test(&Aabbf::new(lo, hi))
            })
            .collect();
        self.last_rendered_chunks = camera_visible.len();
        // Shadow pass uses a sun-POV ortho whose world-space footprint
        // is roughly `shadow_distance` chunks per side. A chunk visible
        // to the sun isn't necessarily visible to the camera (it can be
        // behind the camera and still cast a shadow into view), so we
        // can't reuse `camera_visible` here. A chebyshev cube around
        // the player is conservative enough — the sun ortho is rotated
        // but the world-space AABB of its footprint fits inside.
        let shadow_visible: Vec<&ChunkMesh> = if self.advanced_render {
            self.chunk_meshes
                .values()
                .filter(|cm| {
                    let d = cm.coord - player_chunk;
                    d.x.abs() <= shadow_distance
                        && d.y.abs() <= shadow_distance
                        && d.z.abs() <= shadow_distance
                })
                .collect()
        } else {
            Vec::new()
        };

        // The pre-overlay sub-passes — shadow + G-buffer + composition
        // for advanced rendering, a single forward opaque pass for
        // basic rendering — both end with the surface color attachment
        // populated and the G-buffer depth attachment populated. The
        // forward overlay pass below loads from both regardless of which
        // path ran, so the rest of the function is shared.
        self.record_pre_overlay(encoder, color_view, &camera_visible, &shadow_visible);

        // ---- Forward overlays (particles / selection / underwater) ----
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("game.forward_overlay_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: color_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                // Re-attach the opaque-layer G-buffer depth so
                // particles + selection depth-test against world
                // geometry. Depth is loaded (not cleared) — same
                // buffer the opaque chunk pass just populated. No
                // depth-write here; everything reads. The translucent
                // depth attachment isn't used by overlays — water
                // shouldn't occlude particles / selection.
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: self.gbuffer.opaque_depth_view(),
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });

            // Translucent chunks now go through the deferred path in
            // both modes — they were drawn by the chunk-translucent
            // pipeline into the G-buffer's diffuse, and the composition
            // pass already presented the blended result to the surface.
            // No forward translucent overlay required.

            self.particle_pipeline
                .render(&mut pass, &self.particle_mesh);
            // Selection wireframe rides the same depth buffer so terrain
            // occludes it correctly. Pre-underwater so the water tint
            // layers over it the same way the standalone selection
            // pipeline did before the merge.
            self.selection_pipeline.draw_cube(&mut pass);
            // Underwater tint sits beneath the crosshair so the `+`
            // stays readable through water. No depth read; the draw is
            // a flat full-screen quad.
            self.underwater_pipeline.draw(&mut pass);
            // Crosshair — drawn last (after all world overlays). The
            // selection-pipeline merge keeps it on the same pipeline
            // as the cube; the screen-space vertex kind emits clip
            // `z = 1.0` so it beats the reverse-Z `Greater` test
            // against any geometry the world pass wrote.
            self.selection_pipeline.draw_crosshair(&mut pass);

            // F3+M shadow debug overlay sits on top of every world
            // overlay so the user can read its grayscale even when
            // they're underwater / aiming at a selected block. The
            // egui pass that follows in `App::frame` lays UI on top.
            if self.advanced_render && self.show_shadow_map {
                self.debug_shadow_pipeline.draw(&mut pass);
            }
        }
    }

    /// Run a chat / slash command line. If it begins with `/`, dispatches
    /// through the [`CommandRegistry`]; otherwise echoes the raw text into
    /// chat history.
    pub fn submit_chat_line(&mut self, line: String) {
        if line.is_empty() {
            return;
        }
        if line.starts_with('/') {
            let mut out = Vec::<String>::new();
            let mut ctx = crate::core::game::commands::CommandContext {
                world: &self.world,
                player: &mut self.player,
                queue: &mut self.block_update_queue,
                daylight: &mut self.daylight_cycle,
                base: &self.base_blocks,
                registry: &self.registry,
            };
            self.commands.execute_on(&line, &mut ctx, &mut out);
            for s in out {
                self.push_chat(s);
            }
        } else {
            self.push_chat(line);
        }
    }

    /// Push a chat line stamped with `Instant::now()`. Used both by
    /// `submit_chat_line` and by the registry callbacks (commands push chat
    /// output via the `messages` argument).
    pub fn push_chat(&mut self, line: String) {
        self.chat_messages.push((line, Instant::now()));
        // Bound history to a reasonable cap so we don't grow forever.
        if self.chat_messages.len() > 64 {
            let extra = self.chat_messages.len() - 64;
            self.chat_messages.drain(0..extra);
        }
    }

    /// Recompute the selected block from the camera. The "non-empty"
    /// predicate is `id != air && id != water` — water is treated as empty
    /// for break/place purposes.
    fn update_selection(&mut self) {
        let origin = self.camera.position;
        let dir = self.camera.forward();
        let view = WorldHitView::new(&self.world, &self.registry, &self.base_blocks);
        self.selected = raycast::raycast(&view, origin, dir, RAYCAST_MAX, |w, c| {
            let id = w.block(c).unwrap_or(BlockData::default()).id;
            let info = self.registry.get(id);
            info.solid
        });
    }

    /// Break the currently-selected block. The broken block is added to the
    /// player's inventory as a single-item stack (mirrors the C++
    /// `player.add_item({selb, 1})` in `neworld.ixx::game_update`).
    fn try_break(&mut self) {
        let Some(hit) = self.selected else {
            return;
        };
        let Some(block) = self.world.block(hit.coord) else {
            return;
        };

        // Record texture layer before mutation so the particles inherit the
        // broken block's face art.
        let tex_layer = u32::from(self.render_registry.face(block.id, 0).0);

        // `set_block` inserts the cell's chunk and the parent chunks of
        // its 26 block-neighbours into World's mesh-dirty set;
        // `pump_meshing` picks them up via `World::dirty_snapshot`.
        crate::core::game::block_update::set_block(
            &self.world,
            &mut self.block_update_queue,
            &self.base_blocks,
            &self.registry,
            hit.coord,
            BlockId::default(),
            true,
        );

        // Drop the broken block into the player's inventory. `add_item`
        // returns false if the inventory is completely full of incompatible
        // stacks; in that case the item is lost (matching the C++ behaviour
        // — there is no in-world dropped-item entity to fall back to yet).
        let _ = self.player.add_item(ItemStack::new(block.id, 1));

        // Spawn debris. Each fleck samples a small random fragment of the
        // broken block's face art (mirrors C++ `throw_particle` choosing a
        // `psize × psize` sub-rect at a random origin) so a single break
        // shows visually distinct flecks instead of 10 identical thumbnails.
        for _ in 0..PARTICLES_PER_BREAK {
            let px = f64::from(hit.coord.x) + self.rand_unit();
            let py = f64::from(hit.coord.y) + self.rand_unit();
            let pz = f64::from(hit.coord.z) + self.rand_unit();
            let vx = (self.rand_unit() - 0.5) * 0.4;
            let vy = (self.rand_unit() - 0.5) * 0.4;
            let vz = (self.rand_unit() - 0.5) * 0.4;
            let rng_u = self.rand_unit() as f32;
            let rng_v = self.rand_unit() as f32;
            self.particles.spawn(
                Particle::new(
                    Vec3d::new(px, py, pz),
                    Vec3d::new(vx, vy, vz),
                    tex_layer,
                    1.0,
                )
                .with_random_tex_uv(rng_u, rng_v),
            );
        }
    }

    /// Place the currently-held hotbar block onto the face the ray entered
    /// through. Decrements the held stack by one. No-op when the held slot
    /// is empty.
    fn try_place(&mut self) {
        let Some(hit) = self.selected else {
            return;
        };
        let target = hit.coord + hit.normal;
        // Hotbar slot's id determines what gets placed. Empty hotbar = no
        // placement (mirrors C++: `if (!holdingItem.empty())`).
        let hotbar = *self.player.held_item_stack();
        if hotbar.empty() {
            return;
        }
        let placed_id = hotbar.id;
        if placed_id == BlockId::default() {
            return;
        }
        // Reject placement that would overlap the player's hitbox. C++ does
        // this via `Player::put_block`, which calls `aabb().intersects(...)`;
        // we replicate the same shape here.
        if player_overlaps_block(&self.player, target) {
            return;
        }
        // Don't replace existing solid blocks (lets `+normal` placement work
        // even if the normal happens to be zero — the ray-start-inside-solid
        // case).
        let existing = self.world.block(target).unwrap_or_default().id;
        if existing != BlockId::default() && existing != self.base_blocks.water {
            return;
        }
        // For orientation-bearing blocks (logs etc.), pick the placement
        // state from the clicked face's normal — Minecraft-style: the
        // log's cap axis is whichever axis the normal lies along, and the
        // sign distinguishes the two ends of that axis (the texture isn't
        // necessarily symmetric, so we keep all 6 orientations rather
        // than collapsing to 3).
        let placed_info = self.registry.get(placed_id);
        let placed_state = match placed_info.face_mapping {
            BlockFaceMapping::AxisAligned => state_from_face_normal(hit.normal),
            BlockFaceMapping::Static => BlockState::default(),
        };
        // `set_block_with_state` handles all dirty-mesh marking — see
        // `try_break`.
        crate::core::game::block_update::set_block_with_state(
            &self.world,
            &mut self.block_update_queue,
            &self.base_blocks,
            &self.registry,
            target,
            placed_id,
            placed_state,
            true,
        );

        // Decrement the hotbar stack. In creative we could keep the stack
        // full, but the C++ build always decrements — so we mirror that.
        let idx = self.player.held_item_stack_index();
        let slot = self.player.inventory_item_stack_mut(3, idx);
        if slot.count > 0 {
            slot.count -= 1;
            if slot.count == 0 {
                *slot = ItemStack::default();
            }
        }
    }

    /// Tiny linear-congruential PRNG returning a value in `[0, 1)`. Avoids
    /// pulling in the `rand` crate — the break-particle jitter doesn't need
    /// statistical quality.
    fn rand_unit(&mut self) -> f64 {
        // Numerical Recipes constants.
        self.rng = self
            .rng
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        // Take the high 53 bits and divide by 2^53.
        ((self.rng >> 11) as f64) / 9_007_199_254_740_992.0
    }
}

/// Map a clicked-face normal to a state byte for `FaceMapping::AxisAligned`
/// blocks (logs etc.): each unit axis direction picks one of six discrete
/// states. Mirrors Minecraft log placement — the cap axis is whichever
/// axis the normal lies along, and the sign distinguishes the two ends.
///
/// | normal       | state |
/// |--------------|-------|
/// | `(0,  1, 0)` | 0     |
/// | `(0, -1, 0)` | 1     |
/// | `(1,  0, 0)` | 2     |
/// | `(-1, 0, 0)` | 3     |
/// | `(0,  0, 1)` | 4     |
/// | `(0,  0,-1)` | 5     |
///
/// Falls back to `State(0)` (vertical / +Y) for any non-axis-aligned
/// input, including the zero vector (a ray-start-inside-solid hit).
fn state_from_face_normal(normal: Vec3i) -> BlockState {
    if normal.y > 0 {
        BlockState::inline(0)
    } else if normal.y < 0 {
        BlockState::inline(1)
    } else if normal.x > 0 {
        BlockState::inline(2)
    } else if normal.x < 0 {
        BlockState::inline(3)
    } else if normal.z > 0 {
        BlockState::inline(4)
    } else if normal.z < 0 {
        BlockState::inline(5)
    } else {
        BlockState::inline(0)
    }
}

/// World→shadow-clip matrix for a directional sun.
///
/// Mirrors C++ `Renderer::getShadowMatrix` (`rendering.ixx::229`) in
/// purpose, but built from the live `sun_dir` rather than the
/// `sunlightHeading` / `sunlightPitch` parameterization. The result is
/// equivalent up to choice of "up" vector — since the ortho box is
/// square in xy, any rotation around the sun axis just spins the
/// shadow texels in place without affecting which fragments occlude
/// which.
///
/// Steps:
///   1. `look_to_rh(player_pos, -sun_dir, up)` — places the player at
///      shadow-view origin looking back along the sun ray.
///   2. wgpu reversed-Z orthographic projection — maps a `±length` xy
///      square and `±2*length` z slab into clip-space `[0, 1]` with
///      near = 1, far = 0. Pairs with the shadow pipeline's
///      `CompareFunction::Greater` depth test and the shadow map's
///      `GreaterEqual` comparison sampler.
fn shadow_matrix(player_pos: Vec3d, sun_dir: Vector3<f32>, length: f32) -> Matrix4<f32> {
    let eye = Point3::new(
        player_pos.x as f32,
        player_pos.y as f32,
        player_pos.z as f32,
    );
    // Pick an "up" axis that isn't parallel to the sun direction. World
    // +Y is ideal except when the sun is pointing straight up or down,
    // in which case +Z works.
    let up = if sun_dir.y.abs() > 0.999 {
        Vector3::unit_z()
    } else {
        Vector3::unit_y()
    };
    let view = Matrix4::look_to_rh(eye, -sun_dir, up);
    let proj = ortho_wgpu_rev(
        -length,
        length,
        -length,
        length,
        -length * 2.0,
        length * 2.0,
    );
    proj * view
}

/// Reversed-Z orthographic projection in wgpu's clip-space convention
/// (`Z in [0, 1]`, near = 1, far = 0). Differs from `cgmath::ortho` in
/// the Z mapping: cgmath produces GL `[-1, 1]` standard-Z; this version
/// maps view-space `z = -near → 1`, `z = -far → 0` directly.
///
/// `near` and `far` are conventional cgmath ortho parameters — positive
/// distances along the view direction. They may be negative if the box
/// straddles the camera (the C++ shadow ortho passes
/// `near = -length * 2`, `far = length * 2` to do exactly that).
fn ortho_wgpu_rev(
    left: f32,
    right: f32,
    bottom: f32,
    top: f32,
    near: f32,
    far: f32,
) -> Matrix4<f32> {
    let rl = right - left;
    let tb = top - bottom;
    let fn_ = far - near;
    Matrix4::new(
        2.0 / rl,
        0.0,
        0.0,
        0.0, // column 0
        0.0,
        2.0 / tb,
        0.0,
        0.0, // column 1
        0.0,
        0.0,
        1.0 / fn_,
        0.0, // column 2
        -(right + left) / rl,
        -(top + bottom) / tb,
        far / fn_,
        1.0, // column 3
    )
}

/// Predicate: would placing a unit-cube block at `block` overlap the player's
/// hitbox right now? Mirrors the player-collision check in C++
/// `Player::put_block` (`player_aabb.intersects(block_aabb)`).
fn player_overlaps_block(player: &crate::core::game::player::Player, block: Vec3i) -> bool {
    if player.cross_wall() {
        return false;
    }
    let coord = player.coord();
    let lo = Vec3d::new(f64::from(block.x), f64::from(block.y), f64::from(block.z));
    let hi = Vec3d::new(
        f64::from(block.x + 1),
        f64::from(block.y + 1),
        f64::from(block.z + 1),
    );
    let p_lo = Vec3d::new(
        coord.x - PLAYER_HALF_EXTENT_HORIZ,
        coord.y,
        coord.z - PLAYER_HALF_EXTENT_HORIZ,
    );
    let p_hi = Vec3d::new(
        coord.x + PLAYER_HALF_EXTENT_HORIZ,
        coord.y + 1.7,
        coord.z + PLAYER_HALF_EXTENT_HORIZ,
    );
    p_lo.x < hi.x
        && p_hi.x > lo.x
        && p_lo.y < hi.y
        && p_hi.y > lo.y
        && p_lo.z < hi.z
        && p_hi.z > lo.z
}

/// `BlockView` adapter pairing a [`World`] with the [`BlockRegistry`]
/// and [`BaseBlocks`] it lacks awareness of. Player physics, particle
/// collision, and the raycaster all read the world through this
/// wrapper — the bare `World` doesn't know which ids count as solid
/// or which one is water/lava.
pub struct WorldHitView<'a> {
    world: &'a World,
    registry: &'a BlockRegistry,
    base: &'a BaseBlocks,
}

impl<'a> WorldHitView<'a> {
    pub fn new(world: &'a World, registry: &'a BlockRegistry, base: &'a BaseBlocks) -> Self {
        Self {
            world,
            registry,
            base,
        }
    }
}

impl BlockView for WorldHitView<'_> {
    fn block(&self, coord: Vec3i) -> Option<BlockData> {
        self.world.block(coord)
    }
    fn hitboxes(&self, box_: Aabbd) -> Option<Vec<Aabbd>> {
        crate::core::game::hit_test::hitboxes(self.world, self.registry, box_)
    }
    fn in_water(&self, box_: Aabbd) -> Option<bool> {
        crate::core::game::hit_test::in_water(self.world, self.base, box_)
    }
}

/// Build a [`MeshInput`] for `ccoord` by sampling the world's blocks at every
/// padded cell. Out-of-range neighbors return `block_or_air`'s air default.
/// `options` is captured at submit time so the worker meshes against a stable
/// view; menu toggles take effect by re-issuing every dirty chunk
/// (see [`Game::apply_mesh_config`]).
fn build_mesh_input(world: &World, ccoord: Vec3i, options: MeshOptions) -> Option<MeshInput> {
    use crate::core::world::WorkingSet;

    let air = BlockData::default();
    // Heap-allocate without putting the array on the stack first.
    let v = vec![air; PADDED_VOLUME];
    let boxed: Box<[BlockData]> = v.into_boxed_slice();
    let mut padded: Box<[BlockData; PADDED_VOLUME]> = match boxed.try_into() {
        Ok(arr) => arr,
        Err(_) => unreachable!("PADDED_VOLUME constant is correct"),
    };

    // Open a 27-coord ReadTxn over the chunk + its 26 neighbours.
    // Errors with `NotLoaded` if any of the 27 isn't currently in
    // memory — this *is* the "all neighbours loaded" gate that
    // prevents seam cracks at the render boundary; we don't need a
    // separate `has_neighbours_loaded` check upstream. Each padded
    // cell read goes through the held guards via array index, no
    // per-cell DashMap or RwLock acquisition.
    let mut coords = Vec::with_capacity(27);
    for dx in -1..=1_i32 {
        for dy in -1..=1_i32 {
            for dz in -1..=1_i32 {
                coords.push(ccoord + Vec3i::new(dx, dy, dz));
            }
        }
    }
    let txn = world.begin_read_txn_sync(WorkingSet::List(coords)).ok()?;

    let chunk_origin = Vec3i::new(
        ccoord.x * Chunk::SIZE as i32,
        ccoord.y * Chunk::SIZE as i32,
        ccoord.z * Chunk::SIZE as i32,
    );
    let p_size = i32::try_from(PADDED_SIZE).unwrap_or(18);
    for pz in 0..p_size {
        for py in 0..p_size {
            for px in 0..p_size {
                let global = chunk_origin + Vec3i::new(px - 1, py - 1, pz - 1);
                let block = txn.read(global).expect("inside the 3x3x3 cube");
                let idx = padded_index(px as usize, py as usize, pz as usize);
                padded[idx] = block;
            }
        }
    }
    drop(txn);

    Some(MeshInput {
        coord: ccoord,
        padded,
        options,
    })
}

/// Build a fresh block registry + render registry + texture registry +
/// base block ids — everything the client needs to spin up a world.
///
/// The block registry is server-safe (only `core::blocks` types); the
/// render and texture registries are client-only and would be skipped in
/// a future headless server. All three are wrapped in `Arc` so the
/// renderer, mesher, and async chunk worker can hold cheap clones.
pub fn build_block_registry() -> (
    Arc<BlockRegistry>,
    Arc<BlockRenderRegistry>,
    Arc<BlockTextureRegistry>,
    BaseBlocks,
) {
    let mut registry = BlockRegistry::new();
    let base = register_base_blocks(&mut registry);
    let mut render = BlockRenderRegistry::new();
    let mut textures = BlockTextureRegistry::new();
    register_base_block_visuals(&base, &mut render, &mut textures);
    (
        Arc::new(registry),
        Arc::new(render),
        Arc::new(textures),
        base,
    )
}

impl Game {
    /// Filter chat history to lines whose timestamp is within the recent
    /// decay window, plus everything when chat is open. Public for the HUD.
    pub fn visible_chat_lines(&self, chat_open: bool) -> Vec<&str> {
        let now = Instant::now();
        let cutoff = std::time::Duration::from_secs_f32(CHAT_MESSAGE_LIFETIME_SECS);
        // Show at most the last 8 lines.
        self.chat_messages
            .iter()
            .rev()
            .take(8)
            .filter(|(_, t)| chat_open || now.saturating_duration_since(*t) < cutoff)
            .map(|(s, _)| s.as_str())
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect()
    }
}
