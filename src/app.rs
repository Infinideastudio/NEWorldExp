//! Application bring-up — winit `ApplicationHandler` plus the per-frame
//! render entry point.
//!
//! Owns the window, the wgpu context (`Gfx`), the shared atlases and frame
//! uniform buffer, the HUD text renderer, the egui renderer, the screen
//! stack, and (when a world is loaded) the [`Game`] instance that holds the
//! live world + camera + chunk meshes.
//!
//! Lifecycle:
//! * App boots into the title screen with `game = None`. The world is not
//!   touched until the user picks one.
//! * The title / world-select / pause screens push `WorldAction`s into a
//!   shared mailbox; [`App::process_world_actions`] drains it at the start
//!   of each frame and reacts (load a world, save + drop the current world,
//!   delete a world's directory).
//! * In-game flow: `screen_stack` is empty, `game.is_some()`, the
//!   `GameScreen` runs the HUD; pause / chat overlays sit on the stack.
//! * Out-of-game flow: `game = None`, `screen_stack` has at least the title
//!   on the bottom.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::error::EventLoopError;
use winit::event::{
    DeviceEvent, DeviceId, ElementState, MouseButton as WinitMouseButton, WindowEvent,
};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{CursorGrabMode, Fullscreen, Window, WindowAttributes, WindowId};

use crate::client::game::{Game, build_block_registry};
use crate::client::menus::{
    GameScreen, ScreenStack, Transition, WorldAction, WorldActionQueue, default_worlds_root,
    initial_screen_stack,
};
use crate::client::render::{
    EguiRenderer, FrameUniforms, Gfx, MenuBackground, Screenshot, UniformBuffer,
};
use crate::config::Config;
use crate::core::math::Vec2f;
use crate::globalization::I18n;
use crate::input::{InputState, Key, MouseButton};
use crate::text_rendering::TextRenderer;
use crate::textures::Atlases;

/// Per-window runtime state. Created on the first `resumed` event.
struct AppState {
    window: Arc<Window>,
    gfx: Gfx,
    /// Held to keep the wgpu textures alive — `Game`'s bind groups reference
    /// these texture views.
    #[allow(dead_code)]
    atlases: Atlases,
    frame_uniforms: UniformBuffer<FrameUniforms>,
    /// Held for the GPU resources; egui superseded the debug-text HUD, but
    /// `TextRenderer` may be re-used for chat rendering.
    #[allow(dead_code)]
    text: TextRenderer,
    /// Live world + simulation. `None` while sitting at the title screen.
    game: Option<Game>,
    /// In-game HUD / pause menu. Created with the world; dropped when the
    /// world is. Mirrors `game.is_some()` exactly.
    game_screen: Option<GameScreen>,
    input: InputState,
    last_tick: Instant,
    start_time: Instant,
    /// Start of the current FPS sampling window. The displayed FPS is
    /// `fps_accum_frames / (now - fps_accum_start).as_secs_f32()`,
    /// computed once per second so the readout doesn't jitter on
    /// individual frame-time spikes (which `1 / dt` would amplify).
    fps_accum_start: Instant,
    fps_accum_frames: u32,
    /// Last computed FPS — sticky between window rollovers.
    fps: f32,
    /// Sim-tick counter window (mirrors fps machinery, but counts
    /// `tick_sim` invocations instead of `frame` invocations). With the
    /// fixed-step accumulator clamped at `MAX_TICKS_PER_FRAME`, ups
    /// stays at `1 / TICK_DT = 30` in the steady state and dips when the
    /// frame rate falls so far that the simulation can't keep up.
    ups_accum_start: Instant,
    ups_accum_ticks: u32,
    ups: f32,
    /// Fixed-step accumulator. Drained in [`App::TICK_DT`] slices on every
    /// frame so the simulation runs at a fixed rate independent of render
    /// FPS — see `docs/rust_migration.md` §4.16.
    tick_accumulator: f32,
    cursor_grabbed: bool,
    egui_renderer: EguiRenderer,
    /// Menu / overlay screens. Empty when in-game; non-empty for title,
    /// world-select, options, etc. The top screen receives input.
    screen_stack: ScreenStack,
    /// Surface readback for F2 screenshots ([F4] in the migration plan).
    screenshot: Screenshot,
    /// Live game configuration. Loaded from disk on `resumed`, saved on
    /// `exiting`, edited live by `OptionsScreen`. App applies the live-
    /// applicable values (FOV, mouse, vsync, font scale) each frame.
    config: Arc<Mutex<Config>>,
    /// `VSync` state from the most recent surface configuration. Tracked so
    /// `apply_config` only triggers a `Surface::configure` on actual change.
    last_vsync: bool,
    /// Set by `Transition::Exit` from a screen; causes the event loop to exit.
    exit_requested: bool,
    /// Mailbox for cross-screen world-lifecycle requests. Drained at the
    /// start of each frame by [`App::process_world_actions`].
    world_actions: Arc<WorldActionQueue>,
    /// Mirrors `game.is_some()` so the title screen can show "Back to Game"
    /// only when there's actually a game to return to.
    game_loaded: Arc<AtomicBool>,
    /// Where on disk worlds live. Threaded into the world-select / create
    /// screens so they don't have to re-derive it.
    worlds_root: PathBuf,
    /// Block registry (shared with `Game`). Built once at startup; reused
    /// across world loads.
    registry: Arc<crate::core::blocks::BlockRegistry>,
    /// Per-block face-texture lookup (client-only). Built once at startup
    /// alongside `registry`. Threaded into `Game` and the in-game inventory
    /// for atlas-layer resolution.
    render_registry: Arc<crate::client::blocks::BlockRenderRegistry>,
    /// Base block ids resolved from `registry`. `Copy`, so cheap to clone
    /// when constructing a new `Game`.
    base_blocks: crate::core::game::base_blocks::BaseBlocks,
    /// Egui texture id for each layer of the block-diffuse atlas. Indexed by
    /// `BlockInfo::face(0).0` so the inventory can paint the front-face art
    /// of each block. Built once at startup, after both `egui_renderer` and
    /// `atlases` exist.
    block_icons: Vec<egui::TextureId>,
    /// Active language table. Mirrors the C++ `Globalization::LoadLang` /
    /// `GetStrbyKey` singleton, but locally owned so menus take it via a
    /// shared `Arc<Mutex<…>>` rather than a global. Reloaded by
    /// [`Self::apply_config`] when `Config::language` changes.
    i18n: Arc<Mutex<I18n>>,
    /// `<assets>/lang` — directory the language picker enumerates and that
    /// `i18n.reload` reads from. Cached so we don't recompute the path
    /// every frame.
    lang_dir: PathBuf,
    /// Out-of-game menu background — rotating sky cube + Gaussian blur.
    /// Drawn under the screen stack whenever `game.is_none()`.
    menu_background: MenuBackground,
    /// Egui texture id for `assets/textures/ui/title.png` and its source
    /// pixel size. Threaded into the title screen so it can paint the
    /// rendered logo in place of a "NEWorld" text label.
    title_icon: egui::TextureId,
    title_icon_size: (u32, u32),
}

/// Application root. Implements [`winit::application::ApplicationHandler`].
#[derive(Default)]
pub struct App {
    state: Option<AppState>,
}

impl App {
    /// Fixed simulation tick length (30 Hz). Mirrors the C++ `update_thread`'s
    /// per-tick budget so particle drag/gravity, async load polling, and
    /// (future) block-update queue draining run at a frame-rate-independent
    /// rate (`docs/rust_migration.md` §4.16, [F1]). `Game::tick_sim` is called
    /// with this `dt` once per accumulator slice.
    const TICK_DT: f32 = 1.0 / 30.0;

    /// Maximum number of fixed-step ticks per render frame. Bounds the
    /// "spiral of death" if a slow frame accumulates too much time — extra
    /// time is dropped on the floor rather than chasing forever.
    const MAX_TICKS_PER_FRAME: u32 = 5;

    /// Build an event loop and run the application until exit.
    pub fn run() -> Result<(), EventLoopError> {
        let event_loop = EventLoop::new()?;
        event_loop.set_control_flow(ControlFlow::Poll);
        let mut app = Self::default();
        event_loop.run_app(&mut app)
    }

    /// Resolve the assets directory at compile time so the binary can locate
    /// PNGs no matter where it's launched from.
    fn assets_root() -> PathBuf {
        if let Some(dir) = option_env!("CARGO_MANIFEST_DIR") {
            return PathBuf::from(dir).join("assets");
        }
        PathBuf::from("rs").join("assets")
    }

    /// Resolve the options.toml location. Lives next to the binary's
    /// `assets/` directory at build time; in dev runs that's
    /// `<crate>/configs/options.toml`. We don't try to be clever about
    /// per-user XDG paths — the migration plan calls for a single TOML file.
    fn config_path() -> PathBuf {
        if let Some(dir) = option_env!("CARGO_MANIFEST_DIR") {
            return PathBuf::from(dir).join("configs").join("options.toml");
        }
        PathBuf::from(crate::config::DEFAULT_PATH)
    }

    /// Read the live `Config` and push every applicable setting into the
    /// pieces of state that need it: camera FOV / mouse sensitivity, egui
    /// scale factor, the surface present mode (only reconfigured on vsync
    /// transitions), and the world's render distance (resizes the height-map
    /// cache and re-issues load/unload over the next few ticks).
    fn apply_config(state: &mut AppState) {
        let cfg = state.config.lock().expect("config poisoned");

        if let Some(game) = state.game.as_mut() {
            // Field of view (degrees in TOML, radians on the camera).
            game.camera.fov_y = cfg.fov_y_normal.to_radians();
            // Mouse sensitivity. The TOML field is `mouse_speed`; the player
            // mouse-look code multiplies it by `π/180` (a 1° per pixel base).
            game.mouse_speed = f64::from(cfg.mouse_speed);
            // Live render-distance update. Resizes the load window by
            // diffing the new cube against the current one, feeding the
            // difference into the loader's load / unload queues so
            // subsequent `tick_chunk_loading` calls issue loads / unloads
            // for the new window.
            game.range_loader
                .set_render_distance(&game.world, cfg.render_distance);
            // Mesh-options live-update. Drops every cached `ChunkMesh` and
            // re-marks the loaded set dirty when any of the three flags
            // changes; no-op otherwise.
            game.apply_mesh_config(crate::client::render::MeshOptions {
                smooth_lighting: cfg.smooth_lighting,
                merge_face: cfg.merge_face,
                nice_grass: cfg.nice_grass,
                grass_id: state.base_blocks.grass,
            });
            // Shadow toggle + resolution + composition feature flags.
            // Resizes the shadow map (and rebuilds composition's aux
            // bind group) when the user changes the shader-options menu
            // picker; rebuilds the composition pipeline only when the
            // feature flags change (override-constants force naga to
            // re-DCE the disabled branches).
            game.apply_shadow_config(
                state.gfx.device(),
                &state.atlases,
                cfg.advanced_render,
                cfg.shadow_res,
                cfg.max_shadow_distance,
                cfg.soft_shadow,
                cfg.volumetric_clouds,
                cfg.ambient_occlusion,
            );
        }

        // Live language switch. If the user picked a different language in
        // the language menu, reload the strings so subsequent screen frames
        // pull from the new table. Cheap when the language hasn't changed —
        // a string compare against `i18n.current()`.
        let want_lang = cfg.language.clone();
        if state.i18n.lock().expect("i18n poisoned").current() != want_lang {
            let mut i18n = state.i18n.lock().expect("i18n poisoned");
            if let Err(err) = i18n.reload(&want_lang, &state.lang_dir) {
                tracing::warn!(error = %err, lang = want_lang, "language reload failed");
            }
        }

        // Effective egui pixels-per-point: OS DPI × user UI scale.
        // Drives both layout (widget sizes, hit tests) and rendering
        // (font rasterisation), so a 1.5× ui_scale grows the whole UI
        // uniformly — `egui::Context::set_pixels_per_point` is the
        // single knob that controls both.
        let effective_ppp = state.window.scale_factor() as f32 * cfg.ui_scale.max(0.1);
        state.egui_renderer.set_pixels_per_point(effective_ppp);

        // Live theme switch — re-applying every frame is cheap (egui
        // diffs internally) and keeps us from caching a separate
        // "applied theme" flag.
        //
        // `override_text_color` is forced to the button-text colour
        // (`widgets.inactive.fg_stroke.color`) so every text widget —
        // labels, slider value displays, window content, our own
        // `Label` element — renders at the same brightness as button
        // labels. Without this, egui's default `text_color()` falls
        // back to `widgets.noninteractive.fg_stroke.color`, which is
        // visibly dimmer under the dark theme and causes slider /
        // banner labels to read as washed-out gray next to the
        // brighter button text.
        let mut visuals = if cfg.dark_theme {
            egui::Visuals::dark()
        } else {
            egui::Visuals::light()
        };
        visuals.override_text_color = Some(visuals.widgets.inactive.fg_stroke.color);
        state.egui_renderer.context().set_visuals(visuals);

        // VSync — only reconfigure the surface on actual change, since
        // `Surface::configure` is not free.
        if cfg.vertical_sync != state.last_vsync {
            state.gfx.set_vsync(cfg.vertical_sync);
            state.last_vsync = cfg.vertical_sync;
            tracing::info!(vsync = cfg.vertical_sync, "surface vsync updated");
        }
    }

    /// Drain every pending `WorldAction` from the mailbox and react. Called
    /// at the start of [`Self::frame`] so a click on "Singleplayer → Enter"
    /// in the previous frame results in a live world by the time we tick.
    fn process_world_actions(state: &mut AppState) {
        while let Some(action) = state.world_actions.take() {
            match action {
                WorldAction::Enter { name, seed } => {
                    Self::enter_world(state, name, seed);
                }
                WorldAction::LeaveToTitle => {
                    Self::leave_world_to_title(state);
                }
                WorldAction::Delete { name } => {
                    if let Err(err) =
                        crate::core::world::World::delete_at(&state.worlds_root, &name)
                    {
                        tracing::warn!(error = %err, name, "failed to delete world directory");
                    } else {
                        tracing::info!(name, "world deleted");
                    }
                }
            }
        }
    }

    /// Build a fresh `Game` for `name` / `seed` and switch into the in-game
    /// view. Empties the screen stack so the next frame the HUD overlay
    /// shows on top of the live world. If construction fails, we surface
    /// the error and stay at the title.
    fn enter_world(state: &mut AppState, name: String, seed: u32) {
        // Tear down any previous game before constructing the new one. This
        // shouldn't happen normally (we'd see LeaveToTitle first) but
        // protects against the user's clicks racing the actions queue.
        if state.game.is_some() {
            Self::leave_world_to_title(state);
        }

        let render_distance = state.config.lock().map(|c| c.render_distance).unwrap_or(8);

        match Game::new(
            state.gfx.device(),
            state.gfx.queue(),
            state.gfx.surface_format(),
            state.gfx.surface_size(),
            &state.registry,
            &state.render_registry,
            state.base_blocks,
            &state.frame_uniforms,
            // SAFETY of construction order: `Game` borrows the bind-group
            // layout from `frame_uniforms` + `atlases` while building its
            // pipelines, so they have to outlive the Game. They're held in
            // `AppState` for the whole app lifetime, so that's fine.
            &state.atlases,
            render_distance,
            &state.worlds_root,
            name.clone(),
            seed,
        ) {
            Ok(g) => {
                state.game = Some(g);
                let mut gs = GameScreen::new(
                    Arc::clone(&state.config),
                    Arc::clone(&state.i18n),
                    Arc::clone(&state.world_actions),
                );
                // Capture renderer backend once; the wgpu adapter info is
                // immutable for the session, and the F3 panel surfaces it.
                gs.backend = backend_label(state.gfx.adapter().get_info().backend);
                state.game_screen = Some(gs);
                state.screen_stack = ScreenStack::new();
                state.game_loaded.store(true, Ordering::Relaxed);
                state.tick_accumulator = 0.0;
                state.last_tick = Instant::now();
                tracing::info!(name, seed, "world entered");
            }
            Err(err) => {
                tracing::error!(error = %err, name, "failed to load world");
            }
        }
    }

    /// Save the current world (if any) and drop both `game` and
    /// `game_screen`. Push a fresh title screen so the user lands somewhere.
    fn leave_world_to_title(state: &mut AppState) {
        if let Some(game) = state.game.take() {
            if let Err(err) = game.world.save_to_disk() {
                tracing::error!(error = %err, "save_to_disk failed on leave-to-title");
            }
            tracing::info!("world saved and closed");
            // game (and its ChunkPipeline / MeshPipeline workers) drop here.
            drop(game);
        }
        state.game_screen = None;
        state.game_loaded.store(false, Ordering::Relaxed);
        // Reset the screen stack to a clean title. Drops any pause / options
        // screens that were on top before the leave-to-title click.
        state.screen_stack = initial_screen_stack(
            Arc::clone(&state.config),
            Arc::clone(&state.i18n),
            state.worlds_root.clone(),
            Arc::clone(&state.world_actions),
            state.title_icon,
            state.title_icon_size,
        );
        state.tick_accumulator = 0.0;
    }

    /// Reconcile the wgpu surface size with the live window size. After an
    /// F11 fullscreen toggle (or any OS-driven resize) `window.inner_size()`
    /// updates immediately but the corresponding `Resized` event may not
    /// arrive until a later frame — egui reads the window size at
    /// `begin_frame`/`end_frame` to compute its scissor, so if the surface
    /// is still the old size wgpu rejects the draw with
    /// `Scissor Rect ... is not contained in the render target`.
    /// Calling this whenever a resize might have just happened keeps both
    /// sources of truth in lockstep.
    fn reconcile_surface_size(state: &mut AppState) {
        let win = state.window.inner_size();
        if win.width == 0 || win.height == 0 {
            return;
        }
        let (surf_w, surf_h) = state.gfx.surface_size();
        if win.width == surf_w && win.height == surf_h {
            return;
        }
        state.gfx.resize(win.width, win.height);
        if let Some(game) = state.game.as_mut() {
            game.resize(state.gfx.device(), win.width, win.height);
        }
        state
            .menu_background
            .resize(state.gfx.device(), state.gfx.queue(), win.width, win.height);
    }

    /// Tick + render one frame. Returns `true` if the app should exit.
    fn frame(state: &mut AppState) -> bool {
        // ---------- timing ----------
        let now = Instant::now();
        let dt = (now - state.last_tick).as_secs_f32().min(0.1);
        state.last_tick = now;
        let elapsed = (now - state.start_time).as_secs_f32();

        // Pre-frame size sync. Cheap when nothing has changed.
        Self::reconcile_surface_size(state);

        // Drain any world-lifecycle requests submitted by screens last frame.
        // Doing this first means the rest of `frame` already sees the new
        // game / screen state.
        Self::process_world_actions(state);

        // Apply any live config changes (FOV / mouse / vsync / font scale)
        // before reading from `state.game.camera` etc.
        Self::apply_config(state);

        let chat_open = state
            .game_screen
            .as_ref()
            .is_some_and(|gs| gs.hud.chat_open);
        let inventory_open = state
            .game_screen
            .as_ref()
            .is_some_and(|gs| gs.inventory.open);
        let game_paused = state.game_screen.as_ref().is_some_and(|gs| gs.paused)
            || !state.screen_stack.is_empty();

        // ---------- per-frame render-rate tick ----------
        if let Some(game) = state.game.as_mut() {
            game.tick_render(dt, &state.input, chat_open, inventory_open, game_paused);
        }

        // ---------- fixed-step simulation ([F1]) ----------
        // Particle physics, chunk pipeline polling, and the load-center
        // follow run at exactly `TICK_DT` regardless of render FPS so they
        // stay rate-stable. Bounded by `MAX_TICKS_PER_FRAME` to avoid the
        // spiral-of-death case after a long pause.
        if state.game.is_some() {
            state.tick_accumulator += dt;
            let mut ticks = 0u32;
            while state.tick_accumulator >= Self::TICK_DT && ticks < Self::MAX_TICKS_PER_FRAME {
                if let Some(game) = state.game.as_mut() {
                    game.tick_sim(
                        Self::TICK_DT,
                        &state.input,
                        ticks == 0,
                        chat_open,
                        inventory_open,
                        game_paused,
                    );
                }
                state.tick_accumulator -= Self::TICK_DT;
                ticks += 1;
                state.ups_accum_ticks += 1;
            }
            if state.tick_accumulator > Self::TICK_DT * Self::MAX_TICKS_PER_FRAME as f32 {
                state.tick_accumulator = 0.0;
            }
        } else {
            // No game; never let the accumulator run away.
            state.tick_accumulator = 0.0;
        }

        // ---------- async chunk meshing ([F6]) ----------
        if let Some(game) = state.game.as_mut() {
            game.pump_meshing(state.gfx.device());
        }

        // ---------- frame uniforms ----------
        let surface_size = state.gfx.surface_size();
        if let Some(game) = state.game.as_mut() {
            // `tick_alpha` ∈ [0, 1) — fractional time since the last sim
            // tick, used by `Game::write_frame_uniforms` to interpolate the
            // eye position so render-rate motion is smooth.
            let tick_alpha = (state.tick_accumulator / Self::TICK_DT).clamp(0.0, 1.0);
            game.write_frame_uniforms(
                state.gfx.queue(),
                &state.frame_uniforms,
                surface_size,
                elapsed,
                tick_alpha,
            );
        }

        // F2 → screenshot. Sample before `begin_frame` clears the press
        // edges; the actual capture is enqueued after the world+egui
        // passes are encoded but before submit (see below).
        let screenshot_requested = state.game.is_some() && state.input.is_key_pressed(Key::F2);

        // F11 → fullscreen toggle. Mirrors the C++ `setup.ixx` toggle bound
        // to F11. Borderless fullscreen on the current monitor; pressing F11
        // again restores the windowed size. Windows updates `inner_size()`
        // synchronously inside `set_fullscreen`, but the `Resized` winit
        // event lags by a frame — call the reconciliation helper here so
        // the egui pass below sees the post-toggle surface size.
        if state.input.is_key_pressed(Key::F11) {
            let next = if state.window.fullscreen().is_some() {
                None
            } else {
                Some(Fullscreen::Borderless(None))
            };
            state.window.set_fullscreen(next);
            Self::reconcile_surface_size(state);
        }

        // ---------- consume per-frame input transients ----------
        state.input.begin_frame();

        // ---------- FPS / UPS over a 1 s window ----------
        // Accumulate frame and tick counts instead of computing 1/dt:
        // averaging over a fixed window is steadier than per-frame
        // reciprocals, which spike on any single slow frame. UPS rides
        // the same window so the two readouts stay in lockstep.
        state.fps_accum_frames += 1;
        let elapsed = now.duration_since(state.fps_accum_start).as_secs_f32();
        if elapsed >= 1.0 {
            state.fps = state.fps_accum_frames as f32 / elapsed;
            state.fps_accum_frames = 0;
            state.fps_accum_start = now;
        }
        let ups_elapsed = now.duration_since(state.ups_accum_start).as_secs_f32();
        if ups_elapsed >= 1.0 {
            state.ups = state.ups_accum_ticks as f32 / ups_elapsed;
            state.ups_accum_ticks = 0;
            state.ups_accum_start = now;
        }

        // ---------- update game screen with latest frame data ----------
        if let (Some(game), Some(game_screen)) = (state.game.as_ref(), state.game_screen.as_mut()) {
            let player = &game.player;
            game_screen.fps = state.fps;
            game_screen.ups = state.ups;
            game_screen.camera_pos = [
                game.camera.position.x,
                game.camera.position.y,
                game.camera.position.z,
            ];
            game_screen.yaw = game.camera.yaw;
            game_screen.pitch = game.camera.pitch;
            game_screen.chunk_count = game.world.loaded_count();
            game_screen.rendered_chunks = game.last_rendered_chunks;
            game_screen.unloaded_chunks = game.range_loader.unloaded_chunks();
            game_screen.meshed_chunks = game.chunk_meshes.len();
            game_screen.creative = matches!(
                player.game_mode(),
                crate::core::game::player::GameMode::Creative
            );
            game_screen.cross_wall = player.cross_wall();
            game_screen.grounded = player.grounded();
            game_screen.near_wall = player.near_wall();
            game_screen.in_water = player.in_water();
            game_screen.game_time = game.daylight_cycle.game_time();
            // `updated_blocks` is gameplay state on `BlockUpdateQueue` — stub
            // 0 while block-update writes are stubbed during the migration.
            game_screen.updated_blocks = 0;
            game_screen.pending_block_updates = game.block_update_queue.len();
            // Read advanced_render from Config — Game keeps a private
            // mirror but the live truth is the config lock.
            game_screen.advanced_render = state
                .config
                .lock()
                .map(|c| c.advanced_render)
                .unwrap_or(false);
            game_screen.show_shadow_map = game.show_shadow_map;
            game_screen.chat_history = game
                .visible_chat_lines(chat_open)
                .into_iter()
                .map(str::to_owned)
                .collect();
        }

        // Belt-and-suspenders: any path that mutated `state.window`'s size
        // since the start of the frame must land before egui reads it.
        Self::reconcile_surface_size(state);

        // ---------- begin egui frame ----------
        state.egui_renderer.begin_frame(&state.window);

        // ---------- tick UI (game screen or menu stack) ----------
        let ctx = state.egui_renderer.context();
        if state.screen_stack.is_empty() {
            // In-game (or no game and no menu — but the latter shouldn't
            // happen). Tick the game screen if it exists. The screen needs
            // mutable access to the player so the inventory can shuffle
            // stacks on click.
            if let (Some(game), Some(game_screen)) =
                (state.game.as_mut(), state.game_screen.as_mut())
            {
                match game_screen.tick(
                    ctx,
                    &mut game.player,
                    &state.registry,
                    &state.render_registry,
                    &state.block_icons,
                ) {
                    Transition::Push(s) => state.screen_stack.push(s),
                    Transition::Exit => {
                        state.exit_requested = true;
                        return true;
                    }
                    _ => {}
                }
            }
        } else {
            // Menus overlaying the game (or sitting on top of nothing). Tick
            // the screen stack.
            let action = state.screen_stack.tick(ctx);
            if action {
                // A screen requested exit.
                state.exit_requested = true;
                return true;
            }
        }

        // ---------- chat dispatch (F3) ----------
        if let (Some(game), Some(game_screen)) = (state.game.as_mut(), state.game_screen.as_mut()) {
            let submitted = game_screen.hud.drain_submitted();
            for line in submitted {
                game.submit_chat_line(line);
            }
            if game_screen.hud.chat_open {
                let tab_pressed = state
                    .egui_renderer
                    .context()
                    .input(|i| i.key_pressed(::egui::Key::Tab));
                if tab_pressed
                    && let Some(completed) =
                        game.commands.try_auto_complete(&game_screen.hud.chat_input)
                {
                    game_screen.hud.set_chat_input(completed);
                }
            }
        }

        // ---------- end egui frame (tessellates) ----------
        state.egui_renderer.end_frame(&state.window);

        // ---------- upload egui textures (glyph atlas etc.) ----------
        state
            .egui_renderer
            .update_textures(state.gfx.device(), state.gfx.queue());

        // ---------- cursor grab logic ----------
        // Grab when in-game and no UI modal is up; release as soon as any
        // modal (pause menu, screen stack, chat bar, inventory) needs the
        // pointer. Inventory in particular needs the OS cursor for slot
        // clicks; chat / menus need it for buttons + text fields.
        let want_grab = state.game.is_some()
            && state.screen_stack.is_empty()
            && !game_paused
            && !chat_open
            && !inventory_open;
        if want_grab {
            Self::grab_cursor(&state.window, &mut state.cursor_grabbed);
        } else {
            Self::release_cursor(&state.window, &mut state.cursor_grabbed);
        }

        // ---------- acquire surface ----------
        let frame = match state.gfx.acquire() {
            wgpu::CurrentSurfaceTexture::Success(tex)
            | wgpu::CurrentSurfaceTexture::Suboptimal(tex) => tex,
            wgpu::CurrentSurfaceTexture::Lost | wgpu::CurrentSurfaceTexture::Outdated => {
                let (w, h) = state.gfx.surface_size();
                tracing::warn!(w, h, "surface lost/outdated; reconfiguring");
                state.gfx.reconfigure();
                state.window.request_redraw();
                return false;
            }
            wgpu::CurrentSurfaceTexture::Timeout => {
                tracing::warn!("surface acquire timed out; skipping frame");
                state.window.request_redraw();
                return false;
            }
            wgpu::CurrentSurfaceTexture::Occluded => {
                return false;
            }
            wgpu::CurrentSurfaceTexture::Validation => {
                tracing::error!("surface acquire raised validation error");
                state.window.request_redraw();
                return false;
            }
        };

        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        // Linear-format alias of the same surface texture. Used by the
        // egui pass so its gamma-encoded colors land verbatim instead
        // of being silently double-corrected by the sRGB attachment's
        // store-side encode. When the surface isn't sRGB, this format
        // equals the default and the view degenerates to an alias.
        let egui_view = frame.texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("neworld.egui_view"),
            format: Some(state.gfx.egui_view_format()),
            ..wgpu::TextureViewDescriptor::default()
        });

        let mut encoder =
            state
                .gfx
                .device()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("neworld.frame_encoder"),
                });

        // ---------- upload egui buffers into encoder ----------
        let extra_cbs =
            state
                .egui_renderer
                .update_buffers(state.gfx.device(), state.gfx.queue(), &mut encoder);

        // ---------- world render pass (clears to sky) ----------
        if let Some(game) = state.game.as_mut() {
            game.record_world_pass(
                state.gfx.device(),
                state.gfx.queue(),
                &mut encoder,
                &view,
                surface_size,
            );
        } else {
            // No game — draw the rotating sky cube background, blurred,
            // under the menu screens.
            let (sw, sh) = surface_size;
            // `elapsed` is shadowed by the FPS-window timer further up in
            // this function; recompute the wall-clock elapsed seconds for
            // the cube rotation so the bg motion is independent of the
            // FPS sample window.
            let bg_time = (now - state.start_time).as_secs_f32();
            state
                .menu_background
                .render(state.gfx.queue(), &mut encoder, &view, sw, sh, bg_time);
        }

        // ---------- egui render pass (on top of the world) ----------
        {
            let mut egui_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("neworld.egui_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    // Linear-format view of the surface — see comment
                    // above where `egui_view` is created.
                    view: &egui_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            state.egui_renderer.render(&mut egui_pass);
        }

        // ---------- screenshot copy (F2) ----------
        if screenshot_requested {
            let path = screenshot_path();
            match state.screenshot.capture(
                state.gfx.device(),
                &mut encoder,
                &frame.texture,
                state.gfx.surface_format(),
                state.gfx.surface_size(),
                path,
            ) {
                Ok(()) => {}
                Err(err) => tracing::warn!(error = %err, "screenshot capture skipped"),
            }
        }

        // ---------- submit & present ----------
        state
            .gfx
            .queue()
            .submit(std::iter::once(encoder.finish()).chain(extra_cbs));
        frame.present();
        state.window.request_redraw();

        false
    }

    /// Try to lock the cursor to the window center for FPS-style mouse-look.
    fn grab_cursor(window: &Window, state: &mut bool) {
        if *state {
            return;
        }
        let result = window
            .set_cursor_grab(CursorGrabMode::Locked)
            .or_else(|_| window.set_cursor_grab(CursorGrabMode::Confined));
        if let Err(err) = result {
            tracing::warn!(error = %err, "failed to grab cursor");
            return;
        }
        window.set_cursor_visible(false);
        *state = true;
    }

    fn release_cursor(window: &Window, state: &mut bool) {
        if !*state {
            return;
        }
        let _ = window.set_cursor_grab(CursorGrabMode::None);
        window.set_cursor_visible(true);
        *state = false;
    }
}

/// Short human label for a wgpu backend, surfaced in the F3 debug
/// panel. The C++ build prints `[GL {Major}.{Minor}]`; the Rust port
/// prints `[Vulkan]` / `[DX12]` / etc. depending on what the adapter
/// landed on.
fn backend_label(backend: wgpu::Backend) -> &'static str {
    match backend {
        wgpu::Backend::Vulkan => "Vulkan",
        wgpu::Backend::Metal => "Metal",
        wgpu::Backend::Dx12 => "DX12",
        wgpu::Backend::Gl => "OpenGL",
        wgpu::Backend::BrowserWebGpu => "WebGPU",
        wgpu::Backend::Noop => "noop",
    }
}

/// Build the output path for a new F2 screenshot:
/// `screenshots/screenshot_<unix_seconds>.png`. Avoids the `chrono`
/// dependency by using `SystemTime`.
fn screenshot_path() -> PathBuf {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());
    PathBuf::from("screenshots").join(format!("screenshot_{secs}.png"))
}

/// Translate a winit `KeyCode` into our crate-local [`Key`] enum, or `None`
/// if we don't track the key.
fn translate_key(code: KeyCode) -> Option<Key> {
    Some(match code {
        KeyCode::KeyA => Key::A,
        KeyCode::KeyD => Key::D,
        KeyCode::KeyE => Key::E,
        KeyCode::KeyF => Key::F,
        KeyCode::KeyG => Key::G,
        KeyCode::KeyH => Key::H,
        KeyCode::KeyL => Key::L,
        KeyCode::KeyM => Key::M,
        KeyCode::KeyR => Key::R,
        KeyCode::KeyS => Key::S,
        KeyCode::KeyW => Key::W,
        KeyCode::KeyX => Key::X,
        KeyCode::KeyZ => Key::Z,
        KeyCode::Slash => Key::Slash,
        KeyCode::Space => Key::Space,
        KeyCode::Tab => Key::Tab,
        KeyCode::Enter => Key::Enter,
        KeyCode::Backspace => Key::Backspace,
        KeyCode::Escape => Key::Escape,
        KeyCode::ShiftLeft => Key::LeftShift,
        KeyCode::ShiftRight => Key::RightShift,
        KeyCode::ControlLeft => Key::LeftControl,
        KeyCode::ControlRight => Key::RightControl,
        KeyCode::AltLeft => Key::LeftAlt,
        KeyCode::AltRight => Key::RightAlt,
        KeyCode::F1 => Key::F1,
        KeyCode::F2 => Key::F2,
        KeyCode::F3 => Key::F3,
        KeyCode::F4 => Key::F4,
        KeyCode::F5 => Key::F5,
        KeyCode::F6 => Key::F6,
        KeyCode::F7 => Key::F7,
        KeyCode::F8 => Key::F8,
        KeyCode::F11 => Key::F11,
        KeyCode::ArrowLeft => Key::ArrowLeft,
        KeyCode::ArrowRight => Key::ArrowRight,
        KeyCode::ArrowUp => Key::ArrowUp,
        KeyCode::ArrowDown => Key::ArrowDown,
        _ => return None,
    })
}

fn translate_mouse_button(b: WinitMouseButton) -> Option<MouseButton> {
    Some(match b {
        WinitMouseButton::Left => MouseButton::Left,
        WinitMouseButton::Right => MouseButton::Right,
        WinitMouseButton::Middle => MouseButton::Middle,
        WinitMouseButton::Back => MouseButton::X1,
        WinitMouseButton::Forward => MouseButton::X2,
        WinitMouseButton::Other(_) => return None,
    })
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_some() {
            return;
        }

        // Load options.toml from disk; fall back to defaults if missing.
        let config_path = Self::config_path();
        let initial_config = match Config::load_from(&config_path) {
            Ok(cfg) => cfg,
            Err(err) => {
                tracing::warn!(error = %err, ?config_path, "config load failed, using defaults");
                Config::default()
            }
        };
        let initial_vsync = initial_config.vertical_sync;
        let config = Arc::new(Mutex::new(initial_config));

        let attributes = WindowAttributes::default()
            .with_title("NEWorld")
            .with_inner_size(LogicalSize::new(1280, 720))
            .with_min_inner_size(LogicalSize::new(256, 144));

        let window = match event_loop.create_window(attributes) {
            Ok(w) => Arc::new(w),
            Err(err) => {
                tracing::error!(error = %err, "failed to create window");
                event_loop.exit();
                return;
            }
        };

        let mut gfx = Gfx::new(window.clone());
        if !initial_vsync {
            // Default Gfx surface picks Fifo (vsync); honour the user's
            // saved preference for Immediate up front.
            gfx.set_vsync(false);
        }

        let assets = Self::assets_root();

        // Build the block, render, and texture registries before loading
        // atlases: `Atlases::load` walks the texture registry's names in
        // registration order to build the D2Array layers, so the registries
        // must be populated first. Reused across world loads so each
        // `Game::new` doesn't have to rebuild them.
        let (registry, render_registry, block_textures, base) = build_block_registry();

        let atlases = match Atlases::load(gfx.device(), gfx.queue(), &assets, &block_textures) {
            Ok(a) => {
                tracing::info!(
                    diffuse_layers = a.block_diffuse.layers,
                    normal_layers = a.block_normal.layers,
                    "atlases loaded"
                );
                a
            }
            Err(err) => {
                tracing::error!(error = %err, "fatal: atlas load failed");
                event_loop.exit();
                return;
            }
        };

        let frame_uniforms =
            UniformBuffer::<FrameUniforms>::new(gfx.device(), "neworld.frame_uniforms");

        let (surf_w, surf_h) = gfx.surface_size();
        let menu_background =
            MenuBackground::new(gfx.device(), &atlases, gfx.surface_format(), surf_w, surf_h);

        let text = TextRenderer::new(gfx.device(), gfx.queue(), gfx.surface_format());

        // Egui uses the linear-format sibling view of the same surface
        // texture. With an sRGB surface format the GPU's automatic
        // store-side encode would gamma-correct egui's already-gamma
        // pixels twice; the unorm view bypasses that.
        let mut egui_renderer = EguiRenderer::new(
            gfx.device(),
            gfx.egui_view_format(),
            &window,
            window.scale_factor() as f32,
        );

        // Register each layer of the block-diffuse atlas as an egui texture
        // so the inventory can paint real block art per slot. Built once at
        // startup; the ids stay valid for the renderer's lifetime.
        let block_icons = egui_renderer
            .register_native_textures(gfx.device(), &atlases.block_diffuse.layer_views);
        tracing::info!(count = block_icons.len(), "registered block icons");

        // The title screen paints `title.png` in place of a "NEWorld" label.
        // Linear filtering — the title PNG is a hand-drawn high-res asset,
        // not voxel pixel art, so bilinear sampling keeps it smooth when
        // scaled to fit the banner row.
        let title_icon = egui_renderer.register_native_texture(
            gfx.device(),
            &atlases.title.view,
            wgpu::FilterMode::Linear,
        );
        let (title_icon_w, title_icon_h) = atlases.title.size;

        // `block_textures` (the texture-name registry) is no longer needed
        // past atlas load — drop it here. The block registry stays alive
        // for the world simulator and the render registry stays alive for
        // the mesher and inventory.
        let _ = block_textures;

        let world_actions = WorldActionQueue::new();
        let game_loaded = Arc::new(AtomicBool::new(false));
        let worlds_root = default_worlds_root();

        // Resolve the language directory and load the configured language.
        // Falls back to defaults (empty table) on missing / malformed file
        // so a misconfigured options.toml doesn't fail the whole boot.
        let lang_dir = assets.join("lang");
        let initial_lang = config
            .lock()
            .map(|c| c.language.clone())
            .unwrap_or_else(|_| "en_US".to_owned());
        let i18n = match I18n::load(&initial_lang, &lang_dir) {
            Ok(i) => i,
            Err(err) => {
                tracing::warn!(error = %err, lang = initial_lang, "i18n load failed; using empty table");
                I18n::empty(&initial_lang)
            }
        };
        let i18n = Arc::new(Mutex::new(i18n));

        let screen_stack = initial_screen_stack(
            Arc::clone(&config),
            Arc::clone(&i18n),
            worlds_root.clone(),
            Arc::clone(&world_actions),
            title_icon,
            (title_icon_w, title_icon_h),
        );

        window.request_redraw();

        let now = Instant::now();
        self.state = Some(AppState {
            window,
            gfx,
            atlases,
            frame_uniforms,
            text,
            game: None,
            game_screen: None,
            input: InputState::new(),
            last_tick: now,
            start_time: now,
            fps_accum_start: now,
            fps_accum_frames: 0,
            fps: 0.0,
            ups_accum_start: now,
            ups_accum_ticks: 0,
            ups: 0.0,
            tick_accumulator: 0.0,
            cursor_grabbed: false,
            egui_renderer,
            screen_stack,
            screenshot: Screenshot::new(),
            config,
            last_vsync: initial_vsync,
            exit_requested: false,
            world_actions,
            game_loaded,
            worlds_root,
            registry,
            render_registry,
            base_blocks: base,
            block_icons,
            i18n,
            lang_dir,
            menu_background,
            title_icon,
            title_icon_size: (title_icon_w, title_icon_h),
        });
        tracing::info!("app ready");
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        let Some(state) = self.state.as_mut() else {
            return;
        };

        // Check exit flag (set by screen stack or close request).
        if state.exit_requested {
            event_loop.exit();
            return;
        }

        // Pass all events to egui for modifier / focus tracking.
        // We check egui_wants_focus before processing game keyboard input
        // so typing in an egui text field doesn't also move the player.
        let egui_resp = state.egui_renderer.handle_event(&state.window, &event);
        let egui_wants_focus = egui_resp.consumed;

        match event {
            WindowEvent::CloseRequested => {
                tracing::info!("close requested; exiting");
                event_loop.exit();
            }
            WindowEvent::Resized(size) => {
                state.gfx.resize(size.width, size.height);
                if let Some(game) = state.game.as_mut() {
                    game.resize(state.gfx.device(), size.width, size.height);
                }
                state.menu_background.resize(
                    state.gfx.device(),
                    state.gfx.queue(),
                    size.width,
                    size.height,
                );
                // Re-apply the combined OS DPI × ui_scale so the
                // post-resize layout uses the user's preferred scale.
                let ui_scale = state.config.lock().map_or(1.0, |c| c.ui_scale).max(0.1);
                let effective_ppp = state.window.scale_factor() as f32 * ui_scale;
                state.egui_renderer.set_pixels_per_point(effective_ppp);
                state.window.request_redraw();
            }
            WindowEvent::RedrawRequested => {
                let should_exit = Self::frame(state);
                if should_exit {
                    event_loop.exit();
                }
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if egui_wants_focus {
                    return;
                }
                let PhysicalKey::Code(code) = event.physical_key else {
                    return;
                };
                let Some(key) = translate_key(code) else {
                    return;
                };
                match event.state {
                    ElementState::Pressed => {
                        // Release cursor on Escape (handled by game screen now,
                        // but keep as safety for when menus are not open).
                        if key == Key::Escape && state.cursor_grabbed {
                            Self::release_cursor(&state.window, &mut state.cursor_grabbed);
                        }
                        if !event.repeat && state.input.keys_down.insert(key) {
                            state.input.keys_pressed.insert(key);
                        }
                    }
                    ElementState::Released => {
                        if state.input.keys_down.remove(key) {
                            state.input.keys_released.insert(key);
                        }
                    }
                }
            }
            WindowEvent::MouseInput {
                state: btn_state,
                button,
                ..
            } => {
                if egui_wants_focus {
                    return;
                }
                let Some(button) = translate_mouse_button(button) else {
                    return;
                };
                match btn_state {
                    ElementState::Pressed => {
                        if !state.cursor_grabbed
                            && state.screen_stack.is_empty()
                            && state.game.is_some()
                        {
                            Self::grab_cursor(&state.window, &mut state.cursor_grabbed);
                        }
                        if !state.input.mouse_buttons.contains(button) {
                            state.input.mouse_buttons.insert(button);
                            state.input.mouse_buttons_pressed.insert(button);
                        }
                    }
                    ElementState::Released => {
                        if state.input.mouse_buttons.contains(button) {
                            state.input.mouse_buttons.remove(button);
                            state.input.mouse_buttons_released.insert(button);
                        }
                    }
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                state.input.mouse_pos = Vec2f::new(position.x as f32, position.y as f32);
            }
            WindowEvent::MouseWheel { delta, .. } => {
                if egui_wants_focus {
                    return;
                }
                let dy = match delta {
                    winit::event::MouseScrollDelta::LineDelta(_, y) => y,
                    winit::event::MouseScrollDelta::PixelDelta(p) => p.y as f32 / 32.0,
                };
                state.input.mouse_wheel_delta += dy;
            }
            WindowEvent::Focused(false) => {
                Self::release_cursor(&state.window, &mut state.cursor_grabbed);
            }
            _ => {}
        }
    }

    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _device_id: DeviceId,
        event: DeviceEvent,
    ) {
        let Some(state) = self.state.as_mut() else {
            return;
        };
        if let DeviceEvent::MouseMotion { delta: (dx, dy) } = event
            && state.cursor_grabbed
        {
            state.input.mouse_motion += Vec2f::new(dx as f32, dy as f32);
        }
    }

    /// Synchronous save-on-exit: persist the world (if any) AND the live
    /// config back to disk. The pipeline worker is still alive at this
    /// point — `World::drop` (running shortly after) closes its request
    /// channel and joins the thread, so any pending background save flushes
    /// before the process exits.
    fn exiting(&mut self, _event_loop: &ActiveEventLoop) {
        let Some(state) = self.state.as_mut() else {
            return;
        };
        if let Some(game) = state.game.as_mut()
            && let Err(err) = game.world.save_to_disk()
        {
            tracing::error!(error = %err, "save_to_disk failed on exit");
        }
        // Snapshot the config under the lock so we can write without
        // holding the lock across IO.
        let cfg_snapshot = match state.config.lock() {
            Ok(g) => g.clone(),
            Err(p) => p.into_inner().clone(),
        };
        let path = Self::config_path();
        if let Err(err) = cfg_snapshot.save_to(&path) {
            tracing::error!(error = %err, ?path, "config save failed on exit");
        }
    }
}
