//! Inventory overlay (E5) — full grid + always-visible hotbar.
//!
//! The full grid (4 rows × 10 cols) shows when [`Inventory::open`] is true and
//! supports mouse drag-and-drop:
//!   * Left click — pick up the slot's stack into the cursor (or merge if the
//!     cursor already holds the same id, or swap if a different id).
//!   * Right click — split: pick up half of the slot's stack, or drop one
//!     item into the slot if the cursor already holds something.
//!
//! The hotbar (row 3) is always drawn at the bottom of the screen, with the
//! currently-held slot highlighted. When the full inventory is closed,
//! mouse-wheel and Z/X cycle the held slot — that's wired in
//! `crate::game::Game::tick_render`, not here.
//!
//! Each non-empty slot paints the front-face block art via per-layer
//! `egui::TextureId`s sourced from the block-diffuse atlas (built once at
//! startup by `EguiRenderer::register_native_textures`).

use egui::{Align2, Color32, Context, FontId, Pos2, Rect, Sense, Stroke, TextureId, Ui, vec2};

use crate::client::blocks::BlockRenderRegistry;
use crate::core::blocks::{BlockId, BlockRegistry};
use crate::core::game::player::Player;
use crate::items::ItemStack;

/// Slot size in pixels (matches the C++ 32×32).
const SLOT_SIZE: f32 = 32.0;

/// Inventory state — `open` flag plus the cursor's currently-held stack.
#[derive(Default)]
pub struct Inventory {
    pub open: bool,
    /// What the cursor is currently dragging. Empty when no transfer is in
    /// progress. Drawn under the cursor while non-empty.
    held: ItemStack,
}

impl Inventory {
    /// Draw the always-visible hotbar; if `open`, also draw the full inventory
    /// window. `player` is borrowed mutably because clicks transfer items
    /// into / out of `player.inventory_item_stack_mut(...)`.
    pub fn render(
        &mut self,
        ctx: &Context,
        player: &mut Player,
        registry: &BlockRegistry,
        render_registry: &BlockRenderRegistry,
        block_icons: &[TextureId],
    ) {
        self.render_hotbar(ctx, player, registry, render_registry, block_icons);

        if self.open {
            self.render_full(ctx, player, registry, render_registry, block_icons);
        }
        // If the inventory is closed while `self.held` is non-empty, the
        // stack stays cached on the cursor and reappears the next time the
        // user opens the inventory. Items don't get lost, and we don't have
        // to deal with a "drop on the world" path right now.
    }

    /// Bottom-of-screen hotbar — 10 slots, the held slot highlighted. Always
    /// rendered, regardless of `open`.
    fn render_hotbar(
        &mut self,
        ctx: &Context,
        player: &mut Player,
        registry: &BlockRegistry,
        render_registry: &BlockRenderRegistry,
        block_icons: &[TextureId],
    ) {
        let row = 3;
        let held_idx = player.held_item_stack_index();
        let total_w = 10.0 * SLOT_SIZE + 9.0 * 2.0;
        egui::Area::new("hotbar".into())
            .anchor(Align2::CENTER_BOTTOM, vec2(0.0, -8.0))
            .interactable(false)
            .show(ctx, |ui| {
                ui.set_min_size(vec2(total_w, SLOT_SIZE + 8.0));
                ui.horizontal(|ui| {
                    ui.spacing_mut().item_spacing.x = 2.0;
                    for col in 0..10 {
                        let slot = *player.inventory_item_stack(row, col);
                        let highlighted = col == held_idx;
                        Self::draw_slot(
                            ui,
                            slot,
                            registry,
                            render_registry,
                            block_icons,
                            highlighted,
                            false,
                        );
                    }
                });
            });
    }

    /// Centered window with the upper 3 rows of the inventory plus the hotbar
    /// at the bottom. Each slot is interactable with the mouse.
    #[allow(clippy::too_many_lines)] // single-screen UI; splitting hurts more than it helps
    fn render_full(
        &mut self,
        ctx: &Context,
        player: &mut Player,
        registry: &BlockRegistry,
        render_registry: &BlockRenderRegistry,
        block_icons: &[TextureId],
    ) {
        // Background dimmer behind the window so the world is muted.
        egui::Area::new("inv_dim".into())
            .anchor(Align2::LEFT_TOP, vec2(0.0, 0.0))
            .interactable(false)
            .order(egui::Order::Background)
            .show(ctx, |ui| {
                let rect = ctx.content_rect();
                ui.painter()
                    .rect_filled(rect, 0.0, Color32::from_black_alpha(120));
            });

        egui::Window::new("Inventory")
            .collapsible(false)
            .resizable(false)
            .anchor(Align2::CENTER_CENTER, egui::vec2(0.0, 0.0))
            .show(ctx, |ui| {
                // Rows 0..3 in inventory order (top to bottom). The hotbar
                // is row 3 in the player's storage; we render it last with a
                // visual separator.
                for row in 0..3 {
                    ui.horizontal(|ui| {
                        ui.spacing_mut().item_spacing.x = 2.0;
                        for col in 0..10 {
                            self.handle_slot(
                                ui,
                                player,
                                registry,
                                render_registry,
                                block_icons,
                                row,
                                col,
                            );
                        }
                    });
                    ui.add_space(2.0);
                }
                ui.separator();
                ui.horizontal(|ui| {
                    ui.spacing_mut().item_spacing.x = 2.0;
                    let held_idx = player.held_item_stack_index();
                    for col in 0..10 {
                        let highlighted = col == held_idx;
                        self.handle_slot_with_highlight(
                            ui,
                            player,
                            registry,
                            render_registry,
                            block_icons,
                            3,
                            col,
                            highlighted,
                        );
                    }
                });
            });

        // Held stack follows the cursor. Drawn on top of everything via the
        // `Foreground` layer.
        if !self.held.empty()
            && let Some(pos) = ctx.pointer_latest_pos()
        {
            let painter = ctx.layer_painter(egui::LayerId::new(
                egui::Order::Foreground,
                egui::Id::new("inv_cursor"),
            ));
            let rect = Rect::from_min_size(
                Pos2::new(pos.x - SLOT_SIZE * 0.5, pos.y - SLOT_SIZE * 0.5),
                vec2(SLOT_SIZE, SLOT_SIZE),
            );
            Self::paint_slot_into(
                &painter,
                rect,
                self.held,
                registry,
                render_registry,
                block_icons,
                false,
                true,
            );
        }
    }

    /// Render one slot with click handling. Pure dispatch wrapper around
    /// [`Self::handle_slot_with_highlight`] with `highlighted=false`.
    #[allow(clippy::too_many_arguments)]
    fn handle_slot(
        &mut self,
        ui: &mut Ui,
        player: &mut Player,
        registry: &BlockRegistry,
        render_registry: &BlockRenderRegistry,
        block_icons: &[TextureId],
        row: usize,
        col: usize,
    ) {
        self.handle_slot_with_highlight(
            ui,
            player,
            registry,
            render_registry,
            block_icons,
            row,
            col,
            false,
        );
    }

    /// Render one slot and react to clicks. Mutates the player's slot and
    /// `self.held` to model the C++ `draw_inventory` interactions:
    ///   * left click + cursor empty + slot non-empty → cursor takes slot.
    ///   * left click + cursor non-empty, same id, slot non-full → cursor
    ///     deposits into slot (capped at MAX_COUNT).
    ///   * left click + ids differ (or one side empty) → swap.
    ///   * right click + cursor non-empty + slot empty/same id → drop one.
    ///   * right click + cursor empty + slot non-empty → take half (rounded up).
    #[allow(clippy::too_many_arguments)] // grouping the args into a struct hurts more than it helps
    fn handle_slot_with_highlight(
        &mut self,
        ui: &mut Ui,
        player: &mut Player,
        registry: &BlockRegistry,
        render_registry: &BlockRenderRegistry,
        block_icons: &[TextureId],
        row: usize,
        col: usize,
        highlighted: bool,
    ) {
        let slot = *player.inventory_item_stack(row, col);
        let response = Self::draw_slot(
            ui,
            slot,
            registry,
            render_registry,
            block_icons,
            highlighted,
            true,
        );

        if response.clicked() {
            self.left_click_slot(player, row, col);
        } else if response.secondary_clicked() {
            self.right_click_slot(player, row, col);
        }
    }

    fn left_click_slot(&mut self, player: &mut Player, row: usize, col: usize) {
        let slot = player.inventory_item_stack_mut(row, col);
        if self.held.empty() {
            // Cursor empty + slot has stuff → cursor takes the whole slot.
            self.held = *slot;
            *slot = ItemStack::default();
        } else if slot.empty() {
            // Cursor non-empty + slot empty → drop everything into slot.
            *slot = self.held;
            self.held = ItemStack::default();
        } else if slot.id == self.held.id {
            // Same id, both non-empty → top up the slot, leftover stays held.
            slot.merge_into(&mut self.held);
        } else {
            // Different ids → swap.
            std::mem::swap(slot, &mut self.held);
        }
    }

    fn right_click_slot(&mut self, player: &mut Player, row: usize, col: usize) {
        let slot = player.inventory_item_stack_mut(row, col);
        if self.held.empty() {
            // Cursor empty + slot non-empty → take half (rounded up so a
            // single item picks itself up rather than nothing).
            if slot.empty() {
                return;
            }
            let take = slot.count / 2 + slot.count % 2;
            self.held = ItemStack::new(slot.id, take);
            slot.count -= take;
            if slot.count == 0 {
                *slot = ItemStack::default();
            }
        } else if slot.empty() {
            // Cursor has items + slot empty → place one.
            *slot = ItemStack::new(self.held.id, 1);
            self.held.count -= 1;
            if self.held.count == 0 {
                self.held = ItemStack::default();
            }
        } else if slot.id == self.held.id && !slot.is_full() {
            // Same id, room left → drop one in.
            slot.count += 1;
            self.held.count -= 1;
            if self.held.count == 0 {
                self.held = ItemStack::default();
            }
        }
    }

    /// Allocate a slot-sized rectangle and paint it. Returns the egui
    /// response so the caller can react to clicks.
    #[allow(clippy::too_many_arguments)] // grouping into a struct hurts more than it helps here
    fn draw_slot(
        ui: &mut Ui,
        stack: ItemStack,
        registry: &BlockRegistry,
        render_registry: &BlockRenderRegistry,
        block_icons: &[TextureId],
        highlighted: bool,
        interactive: bool,
    ) -> egui::Response {
        let sense = if interactive {
            Sense::click()
        } else {
            Sense::hover()
        };
        let (rect, response) = ui.allocate_exact_size(vec2(SLOT_SIZE, SLOT_SIZE), sense);
        let hovered = response.hovered();
        Self::paint_slot_into(
            ui.painter(),
            rect,
            stack,
            registry,
            render_registry,
            block_icons,
            highlighted,
            hovered,
        );
        response
    }

    #[allow(clippy::too_many_arguments)]
    fn paint_slot_into(
        painter: &egui::Painter,
        rect: Rect,
        stack: ItemStack,
        registry: &BlockRegistry,
        render_registry: &BlockRenderRegistry,
        block_icons: &[TextureId],
        highlighted: bool,
        hovered: bool,
    ) {
        // Source the theme from the painter's context so the inventory
        // tracks the live `Config::dark_theme` setting without each call
        // site having to thread the bool through.
        let dark = painter.ctx().global_style().visuals.dark_mode;
        let (bg, border, fallback_text, count_text, count_shadow) =
            slot_palette(dark, highlighted, hovered);
        painter.rect_filled(rect, 2.0, bg);
        painter.rect_stroke(
            rect,
            2.0,
            Stroke::new(1.0_f32, border),
            egui::StrokeKind::Middle,
        );

        if stack.empty() || stack.id == BlockId::default() {
            return;
        }

        let info = registry.get(stack.id);
        // Block icon — face(0) is the front-facing texture index in the
        // block-diffuse atlas. Fall back to a centred letter abbreviation
        // when the icon would be out of range (registries with more block
        // ids than texture layers, or block_icons not yet populated in the
        // unit tests).
        let layer = render_registry.face(stack.id, 0).0 as usize;
        if let Some(tex_id) = block_icons.get(layer) {
            // Inset by 2 px so the slot border stays visible around the icon.
            let icon_rect = rect.shrink(2.0);
            painter.image(
                *tex_id,
                icon_rect,
                Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)),
                Color32::WHITE,
            );
        } else {
            let label = abbreviate(&info.display_name);
            painter.text(
                rect.center() + vec2(0.0, -4.0),
                Align2::CENTER_CENTER,
                label,
                FontId::proportional(13.0),
                fallback_text,
            );
        }
        // Count in the bottom-right corner. Drawn over the icon with a
        // small contrasting drop-shadow so it stays readable against any
        // block art on either theme.
        if stack.count > 1 {
            let text = format!("{}", stack.count);
            let pos = rect.right_bottom() + vec2(-3.0, -2.0);
            painter.text(
                pos + vec2(1.0, 1.0),
                Align2::RIGHT_BOTTOM,
                &text,
                FontId::proportional(11.0),
                count_shadow,
            );
            painter.text(
                pos,
                Align2::RIGHT_BOTTOM,
                text,
                FontId::proportional(11.0),
                count_text,
            );
        }
    }
}

/// Pick the per-slot palette for a given theme + state combo. Returns
/// `(background, border, fallback_text, count_text, count_shadow)`.
fn slot_palette(
    dark: bool,
    highlighted: bool,
    hovered: bool,
) -> (Color32, Color32, Color32, Color32, Color32) {
    if dark {
        let bg = if highlighted {
            Color32::from_rgb(80, 80, 110)
        } else {
            Color32::from_gray(50)
        };
        let border = if hovered {
            Color32::from_gray(220)
        } else if highlighted {
            Color32::from_rgb(180, 180, 220)
        } else {
            Color32::from_gray(110)
        };
        (
            bg,
            border,
            Color32::from_rgb(240, 240, 240),
            Color32::WHITE,
            Color32::from_black_alpha(180),
        )
    } else {
        let bg = if highlighted {
            Color32::from_rgb(200, 215, 245)
        } else {
            Color32::from_gray(225)
        };
        let border = if hovered {
            Color32::from_gray(60)
        } else if highlighted {
            Color32::from_rgb(80, 110, 180)
        } else {
            Color32::from_gray(150)
        };
        (
            bg,
            border,
            Color32::from_rgb(40, 40, 40),
            Color32::BLACK,
            Color32::from_white_alpha(180),
        )
    }
}

/// Pick a short, readable label for a block name. Used as the fallback when
/// no atlas icon is registered for the block id (e.g. unit tests). Strips the
/// leading capital of multi-word names ("Stone Bricks" → "SB") and truncates
/// otherwise.
fn abbreviate(name: &str) -> String {
    let words: Vec<&str> = name.split_whitespace().collect();
    if words.len() >= 2 {
        words
            .iter()
            .filter_map(|w| w.chars().next())
            .map(|c| c.to_ascii_uppercase())
            .take(3)
            .collect()
    } else {
        let mut chars = name.chars();
        let first = chars.next().map_or('?', |c| c.to_ascii_uppercase());
        let rest: String = chars.take(2).collect();
        format!("{first}{rest}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn abbreviate_single_word_returns_first_three_chars() {
        assert_eq!(abbreviate("stone"), "Sto");
        assert_eq!(abbreviate("dirt"), "Dir");
        assert_eq!(abbreviate("a"), "A");
    }

    #[test]
    fn abbreviate_multi_word_uses_initials() {
        assert_eq!(abbreviate("stone bricks"), "SB");
        assert_eq!(abbreviate("nether stone bricks"), "NSB");
    }
}
