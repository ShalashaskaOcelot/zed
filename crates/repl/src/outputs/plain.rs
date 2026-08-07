//! # Plain Text Output
//!
//! This module provides functionality for rendering plain text output in a terminal-like format.
//! It uses Zed's terminal emulator to process and display text, supporting ANSI escape
//! sequences for formatting, colors, and other terminal features.
//!
//! The main component of this module is the `TerminalOutput` struct, which handles the parsing
//! and rendering of text input, simulating a basic terminal environment within REPL output.
//!
//! This module is used for displaying:
//!
//! - Standard output (stdout)
//! - Standard error (stderr)
//! - Plain text content
//! - Error tracebacks
//!

use editor::{HighlightedRange, HighlightedRangeLine};
use gpui::{
    Bounds, ClipboardItem, CursorStyle, DispatchPhase, Entity, FontStyle, HitboxBehavior,
    MouseButton, MouseDownEvent, MouseMoveEvent, MouseUpEvent, Pixels, TextStyle, WhiteSpace,
    canvas, size,
};
use language::Buffer;
use settings::Settings as _;
use terminal::{Terminal, TerminalBuilder, terminal_settings::TerminalSettings};
use terminal_view::terminal_element::TerminalElement;
use theme_settings::ThemeSettings;
use ui::{IntoElement, prelude::*};
use util::paths::PathStyle;

use crate::outputs::OutputContent;
use crate::repl_settings::ReplSettings;

/// The `TerminalOutput` struct handles the parsing and rendering of text input,
/// simulating a basic terminal environment within REPL output.
///
/// `TerminalOutput` is designed to handle various types of text-based output, including:
///
/// * stdout (standard output)
/// * stderr (standard error)
/// * text/plain content
/// * error tracebacks
///
/// It uses Zed's terminal emulator backend to process and render text,
/// supporting ANSI escape sequences for text formatting and colors.
///
pub struct TerminalOutput {
    full_buffer: Option<Entity<Buffer>>,
    terminal: Entity<Terminal>,
    /// A left-button drag that began inside this output is in progress; mouse
    /// moves keep extending the terminal selection until the button releases.
    selecting: bool,
    /// Window-space bounds adopted by the terminal at the last sync, so idle
    /// outputs (no selection activity, unmoved) skip the per-frame sync.
    last_synced_bounds: Option<Bounds<Pixels>>,
    /// Show the START of the content rather than following the tail. Opt-in
    /// (notebook cells) because the inline REPL wants the console behaviour of
    /// keeping the newest output in view.
    pin_to_top: bool,
    /// A `scroll_to_top` is queued on the terminal and needs a `sync` to be
    /// applied. Scroll events are only drained by `Terminal::sync`, which the
    /// canvas otherwise skips for an idle output.
    pending_scroll_pin: bool,
}

/// Returns the default text style for the terminal output.
pub fn text_style(window: &mut Window, cx: &App) -> TextStyle {
    let settings = ThemeSettings::get_global(cx).clone();

    let font_size = settings.buffer_font_size(cx).into();
    let font_family = settings.buffer_font.family;
    let font_features = settings.buffer_font.features;
    let font_weight = settings.buffer_font.weight;
    let font_fallbacks = settings.buffer_font.fallbacks;

    let theme = cx.theme();

    TextStyle {
        font_family,
        font_features,
        font_weight,
        font_fallbacks,
        font_size,
        font_style: FontStyle::Normal,
        line_height: window.line_height().into(),
        background_color: Some(theme.colors().terminal_ansi_background),
        white_space: WhiteSpace::Normal,
        // These are going to be overridden per-cell
        color: theme.colors().terminal_foreground,
        ..Default::default()
    }
}

/// Returns the default terminal size for the terminal output.
pub fn terminal_size(window: &mut Window, cx: &mut App) -> terminal::TerminalBounds {
    let text_style = text_style(window, cx);
    let text_system = window.text_system();

    let line_height = window.line_height();

    let font_pixels = text_style.font_size.to_pixels(window.rem_size());
    let font_id = text_system.resolve_font(&text_style.font());

    let cell_width = text_system
        .advance(font_id, font_pixels, 'w')
        .map(|advance| advance.width)
        .unwrap_or(Pixels::ZERO);

    let num_lines = ReplSettings::get_global(cx).max_lines;
    let columns = ReplSettings::get_global(cx).max_columns;

    // Reversed math from terminal::TerminalSize to get pixel width according to terminal width
    let width = columns as f32 * cell_width;
    let height = num_lines as f32 * window.line_height();

    terminal::TerminalBounds {
        cell_width,
        line_height,
        bounds: Bounds {
            origin: gpui::Point::default(),
            size: size(width, height),
        },
    }
}

pub fn max_width_for_columns(
    columns: usize,
    window: &mut Window,
    cx: &App,
) -> Option<gpui::Pixels> {
    if columns == 0 {
        return None;
    }

    let text_style = text_style(window, cx);
    let text_system = window.text_system();
    let font_pixels = text_style.font_size.to_pixels(window.rem_size());
    let font_id = text_system.resolve_font(&text_style.font());
    let cell_width = text_system
        .advance(font_id, font_pixels, 'w')
        .map(|advance| advance.width)
        .unwrap_or(Pixels::ZERO);

    Some(cell_width * columns as f32)
}

impl TerminalOutput {
    /// Creates a new `TerminalOutput` instance.
    ///
    /// This method initializes a new terminal emulator with default configuration
    /// and sets up the necessary components for handling terminal events and rendering.
    ///
    pub fn new(window: &mut Window, cx: &mut Context<Self>) -> Self {
        let terminal_bounds = terminal_size(window, cx);
        let background_executor = cx.background_executor().clone();
        let terminal_builder = TerminalBuilder::new_display_only_with_bounds(
            TerminalSettings::get_global(cx).cursor_shape,
            TerminalSettings::get_global(cx).alternate_scroll,
            None,
            0,
            &background_executor,
            PathStyle::local(),
            terminal_bounds,
        );

        Self {
            terminal: cx.new(|cx| terminal_builder.subscribe(cx)),
            full_buffer: None,
            selecting: false,
            last_synced_bounds: None,
            pin_to_top: false,
            pending_scroll_pin: false,
        }
    }

    /// Keep the viewport at the START of the content as more is appended, so a
    /// long output shows its beginning instead of its last `max_lines` lines.
    /// Everything is still fed to the terminal, so the scrollback — and hence
    /// `full_text` / "open in buffer" — remains complete.
    pub fn set_pin_to_top(&mut self, pin_to_top: bool) {
        self.pin_to_top = pin_to_top;
        if pin_to_top {
            self.pending_scroll_pin = true;
        }
    }

    /// How many lines have scrolled out of the viewport into scrollback, i.e.
    /// how much of this output is not currently visible.
    pub fn hidden_line_count(&self, cx: &App) -> usize {
        let terminal = self.terminal.read(cx);
        terminal
            .total_lines()
            .saturating_sub(terminal.viewport_lines())
    }

    /// Creates a new `TerminalOutput` instance with initial content.
    ///
    /// Initializes a new terminal output and populates it with the provided text.
    ///
    /// # Arguments
    ///
    /// * `text` - A string slice containing the initial text for the terminal output.
    /// * `cx` - A mutable reference to the `WindowContext` for initialization.
    ///
    /// # Returns
    ///
    /// A new instance of `TerminalOutput` containing the provided text.
    pub fn from(text: &str, window: &mut Window, cx: &mut Context<Self>) -> Self {
        let mut output = Self::new(window, cx);
        output.append_text(text, cx);
        output
    }

    /// Appends text to the terminal output.
    ///
    /// Processes each byte of the input text, handling newline characters specially
    /// to ensure proper cursor movement. Uses the ANSI parser to process the input
    /// and update the terminal state.
    ///
    /// As an example, if the user runs the following Python code in this REPL:
    ///
    /// ```python
    /// import time
    /// print("Hello,", end="")
    /// time.sleep(1)
    /// print(" world!")
    /// ```
    ///
    /// Then append_text will be called twice, with the following arguments:
    ///
    /// ```ignore
    /// terminal_output.append_text("Hello,");
    /// terminal_output.append_text(" world!");
    /// ```
    /// Resulting in a single output of "Hello, world!".
    ///
    /// # Arguments
    ///
    /// * `text` - A string slice containing the text to be appended.
    pub fn append_text(&mut self, text: &str, cx: &mut Context<Self>) {
        let pin_to_top = self.pin_to_top;
        self.terminal.update(cx, |terminal, cx| {
            terminal.write_output(text.as_bytes(), cx);
            // Re-pin after every append: the first overflow pushes the viewport
            // off the top, and a scroll is only a queued event, so it has to be
            // requeued rather than set once.
            if pin_to_top {
                terminal.scroll_to_top();
            }
        });
        if pin_to_top {
            self.pending_scroll_pin = true;
        }

        // This will keep the buffer up to date, though with some terminal codes it won't be perfect
        if let Some(buffer) = self.full_buffer.as_ref() {
            buffer.update(cx, |buffer, cx| {
                buffer.edit([(buffer.len()..buffer.len(), text)], None, cx);
            });
        }
    }

    pub fn full_text(&self, cx: &App) -> String {
        Self::sanitize_terminal_text(self.terminal.read(cx).get_content())
    }

    /// Text of the active in-place mouse selection, if any. Reflects the last
    /// frame's terminal sync, which is fresh by the time a copy shortcut
    /// dispatched after the selection gesture can run.
    pub fn selection_text(&self, cx: &App) -> Option<String> {
        let text = self.terminal.read(cx).last_content.selection_text.clone()?;
        if text.is_empty() { None } else { Some(text) }
    }

    fn sanitize_terminal_text(text: String) -> String {
        fn sanitize(mut line: String) -> Option<String> {
            line.retain(|ch| ch != '\u{0}' && ch != '\r');
            if line.trim().is_empty() {
                return None;
            }
            let trimmed = line.trim_end_matches([' ', '\t']);
            Some(trimmed.to_owned())
        }

        let lines = text
            .lines()
            .filter_map(|line| sanitize(line.to_string()))
            .collect::<Vec<_>>();

        if lines.is_empty() {
            String::new()
        } else {
            let mut full_text = lines.join("\n");
            full_text.push('\n');
            full_text
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gpui::{TestAppContext, VisualTestContext};
    use settings::SettingsStore;

    fn init_test(cx: &mut TestAppContext) -> &mut VisualTestContext {
        cx.update(|cx| {
            let settings_store = SettingsStore::test(cx);
            cx.set_global(settings_store);
            theme_settings::init(theme::LoadThemes::JustBase, cx);
        });
        cx.add_empty_window()
    }

    #[test]
    fn test_selection_highlight_lines_geometry() {
        use gpui::{Point as GpuiPoint, px};
        use terminal::{Point, Range};

        let cell_width = px(10.);
        let line_height = px(20.);
        let origin = GpuiPoint::new(px(0.), px(0.));

        // Single-line selection: line 0, columns 2..=4 → one span [2cw, 5cw).
        let (start_y, lines) = selection_highlight_lines(
            &Range::new(Point::new(0, 2), Point::new(0, 4)),
            0,
            3,
            80,
            cell_width,
            line_height,
            origin,
        )
        .expect("selection is within the viewport");
        assert_eq!(start_y, px(0.));
        assert_eq!(lines.len(), 1);
        assert_eq!(lines[0].start_x, px(20.));
        assert_eq!(lines[0].end_x, px(50.));

        // Multi-line: (0,3)..(2,1) → first line from col 3 to the right edge,
        // middle line full width, last line up to col 1 inclusive.
        let (start_y, lines) = selection_highlight_lines(
            &Range::new(Point::new(0, 3), Point::new(2, 1)),
            0,
            3,
            80,
            cell_width,
            line_height,
            origin,
        )
        .expect("selection is within the viewport");
        assert_eq!(start_y, px(0.));
        assert_eq!(lines.len(), 3);
        assert_eq!(lines[0].start_x, px(30.));
        assert_eq!(lines[0].end_x, px(800.));
        assert_eq!(lines[1].start_x, px(0.));
        assert_eq!(lines[1].end_x, px(800.));
        assert_eq!(lines[2].start_x, px(0.));
        assert_eq!(lines[2].end_x, px(20.));

        // Entirely below the rendered lines → no highlight.
        assert!(
            selection_highlight_lines(
                &Range::new(Point::new(5, 0), Point::new(6, 0)),
                0,
                3,
                80,
                cell_width,
                line_height,
                origin,
            )
            .is_none()
        );

        // A scrollback selection (negative line) maps into the viewport via
        // the display offset.
        let (start_y, lines) = selection_highlight_lines(
            &Range::new(Point::new(-1, 0), Point::new(-1, 1)),
            1,
            3,
            80,
            cell_width,
            line_height,
            origin,
        )
        .expect("offset selection is within the viewport");
        assert_eq!(start_y, px(0.));
        assert_eq!(lines.len(), 1);
        assert_eq!(lines[0].end_x, px(20.));
    }

    #[gpui::test]
    fn test_max_width_for_columns_zero(cx: &mut TestAppContext) {
        let cx = init_test(cx);
        let result = cx.update(|window, cx| max_width_for_columns(0, window, cx));
        assert!(result.is_none());
    }

    #[gpui::test]
    fn test_max_width_for_columns_matches_cell_width(cx: &mut TestAppContext) {
        let cx = init_test(cx);
        let columns = 5;
        let (result, expected) = cx.update(|window, cx| {
            let text_style = text_style(window, cx);
            let text_system = window.text_system();
            let font_pixels = text_style.font_size.to_pixels(window.rem_size());
            let font_id = text_system.resolve_font(&text_style.font());
            let cell_width = text_system
                .advance(font_id, font_pixels, 'w')
                .map(|advance| advance.width)
                .unwrap_or(gpui::Pixels::ZERO);
            let result = max_width_for_columns(columns, window, cx);
            (result, cell_width * columns as f32)
        });

        let Some(result) = result else {
            panic!("expected max width for columns {columns}");
        };
        let result_f32: f32 = result.into();
        let expected_f32: f32 = expected.into();
        assert!((result_f32 - expected_f32).abs() < 0.01);
    }

    #[gpui::test]
    fn test_append_text_preserves_split_ansi_sequence(cx: &mut TestAppContext) {
        let cx = init_test(cx);
        let text = cx.update(|window, cx| {
            let output = cx.new(|cx| TerminalOutput::new(window, cx));
            output.update(cx, |output, cx| {
                output.append_text("\x1b[", cx);
                output.append_text("31mred\x1b[0m", cx);
                output.full_text(cx)
            })
        });

        assert_eq!(text, "red\n");
    }

    #[gpui::test]
    fn test_full_text_reads_terminal_output(cx: &mut TestAppContext) {
        let cx = init_test(cx);
        cx.update(|window, cx| {
            let output = cx.new(|cx| TerminalOutput::new(window, cx));
            output.update(cx, |output, cx| {
                output.append_text("hello\n", cx);
                assert_eq!(output.full_text(cx), "hello\n");
            });
        });
    }

    #[gpui::test]
    fn test_initial_text_uses_repl_terminal_size(cx: &mut TestAppContext) {
        let cx = init_test(cx);
        let (text, expected) = cx.update(|window, cx| {
            let columns = ReplSettings::get_global(cx).max_columns;
            let input = format!("\x1b[{columns}Gx");
            let output = cx.new(|cx| TerminalOutput::from(&input, window, cx));
            (
                output.read(cx).full_text(cx),
                format!("{}x\n", " ".repeat(columns - 1)),
            )
        });

        assert_eq!(text, expected);
    }

    #[gpui::test]
    fn test_hidden_line_count_zero_when_output_fits(cx: &mut TestAppContext) {
        let cx = init_test(cx);
        let hidden = cx.update(|window, cx| {
            let output = cx.new(|cx| TerminalOutput::from("one\ntwo\nthree\n", window, cx));
            output.read(cx).hidden_line_count(cx)
        });

        assert_eq!(hidden, 0);
    }

    #[gpui::test]
    fn test_hidden_line_count_reports_overflow(cx: &mut TestAppContext) {
        let cx = init_test(cx);
        let (hidden, max_lines) = cx.update(|window, cx| {
            let max_lines = ReplSettings::get_global(cx).max_lines;
            let input = (0..max_lines + 10)
                .map(|line| format!("line-{line}\n"))
                .collect::<String>();
            let output = cx.new(|cx| TerminalOutput::from(&input, window, cx));
            (output.read(cx).hidden_line_count(cx), max_lines)
        });

        assert!(
            hidden >= 10,
            "expected at least the 10 lines past the {max_lines}-line viewport to be hidden, got {hidden}"
        );
    }

    /// Pinning changes only the VIEWPORT — everything written must still reach
    /// the terminal, since `full_text` (and so "open in buffer") reads it.
    #[gpui::test]
    fn test_pin_to_top_keeps_full_text(cx: &mut TestAppContext) {
        let cx = init_test(cx);
        let (text, expected) = cx.update(|window, cx| {
            let max_lines = ReplSettings::get_global(cx).max_lines;
            let input = (0..max_lines + 10)
                .map(|line| format!("line-{line}\n"))
                .collect::<String>();
            let output = cx.new(|cx| {
                let mut output = TerminalOutput::new(window, cx);
                output.set_pin_to_top(true);
                output.append_text(&input, cx);
                output
            });
            (output.read(cx).full_text(cx), input)
        });

        assert_eq!(text, expected);
    }

    #[gpui::test]
    fn test_repl_history_ignores_terminal_scrollback_setting(cx: &mut TestAppContext) {
        let cx = init_test(cx);
        let (text, expected) = cx.update(|window, cx| {
            cx.update_global::<SettingsStore, _>(|settings_store, cx| {
                settings_store.update_user_settings(cx, |settings| {
                    settings
                        .terminal
                        .get_or_insert_default()
                        .max_scroll_history_lines = Some(0);
                });
            });

            let input = (0..40)
                .map(|line| format!("line-{line}\n"))
                .collect::<String>();
            let output = cx.new(|cx| TerminalOutput::from(&input, window, cx));
            (output.read(cx).full_text(cx), input)
        });

        assert_eq!(text, expected);
    }
}

impl Render for TerminalOutput {
    /// Renders the terminal output as a GPUI element.
    ///
    /// Converts the current terminal state into a renderable GPUI element. It handles
    /// the layout of the terminal grid, calculates the dimensions of the output, and
    /// creates a canvas element that paints the terminal cells and background rectangles.
    /// Mouse events are routed to the terminal's selection machinery so output
    /// text can be selected in place and copied.
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let terminal = self.terminal.clone();
        let this = cx.entity();

        let text_style = text_style(window, cx);
        let minimum_contrast = TerminalSettings::get_global(cx).minimum_contrast;
        let (rects, batched_text_runs) = terminal.read(cx).with_renderable_cells(|cells| {
            TerminalElement::layout_grid(cells, 0, &text_style, None, minimum_contrast, cx)
        });

        // lines are 0-indexed, so we must add 1 to get the number of lines
        let text_line_height = text_style.line_height_in_pixels(window.rem_size());
        let num_lines = batched_text_runs
            .iter()
            .map(|b| b.start_point.line())
            .max()
            .unwrap_or(0)
            + 1;
        let height = num_lines as f32 * text_line_height;

        let text_system = window.text_system();
        let font_pixels = text_style.font_size.to_pixels(window.rem_size());
        let font_id = text_system.resolve_font(&text_style.font());

        let cell_width = text_system
            .advance(font_id, font_pixels, 'w')
            .map(|advance| advance.width)
            .unwrap_or(Pixels::ZERO);

        let num_columns = ReplSettings::get_global(cx).max_columns;
        let selection_color = cx.theme().players().local().selection;
        let corner_radius = 0.15 * text_line_height;

        canvas(
            // prepaint: adopt the element's window-space bounds (the terminal's
            // mouse math subtracts its recorded origin from event positions),
            // pump queued selection events through sync, and lay out the
            // selection highlight for paint.
            {
                let terminal = terminal.clone();
                let this = this.clone();
                move |bounds, window, cx| {
                    // Syncing locks the terminal and rebuilds its content
                    // snapshot, so idle outputs (nothing selected, bounds
                    // unchanged) skip it and stay as cheap as before selection
                    // support existed.
                    let needs_sync = this.read(cx).selecting
                        || this.read(cx).last_synced_bounds != Some(bounds)
                        || this.read(cx).pending_scroll_pin
                        || terminal.read(cx).last_content.selection.is_some();
                    if needs_sync {
                        let mut terminal_bounds = terminal_size(window, cx);
                        terminal_bounds.bounds.origin = bounds.origin;
                        // Wrap at the width this output is ACTUALLY laid out
                        // at. `terminal_size` sizes to `max_columns`, so
                        // adopting only the origin left the terminal wrapping
                        // at a fixed 128 columns however wide the block was —
                        // notebook outputs (which span their cell's full width
                        // since phase 40) wrapped early and left a gap. The
                        // inline REPL is unaffected: its container is already
                        // capped to `max_columns` wide, so the laid-out width
                        // it reports here is that same cap.
                        if bounds.size.width > Pixels::ZERO {
                            terminal_bounds.bounds.size.width = bounds.size.width;
                        }
                        let pin_to_top = this.read(cx).pin_to_top;
                        terminal.update(cx, |terminal, cx| {
                            terminal.set_size(terminal_bounds);
                            // After the resize, not before: a resize reflows the
                            // grid and drops the display offset, so a scroll
                            // queued earlier would be undone by it.
                            if pin_to_top {
                                terminal.scroll_to_top();
                            }
                            terminal.sync(window, cx);
                        });
                        this.update(cx, |this, _| {
                            this.last_synced_bounds = Some(bounds);
                            this.pending_scroll_pin = false;
                        });
                    }

                    let hitbox = window.insert_hitbox(bounds, HitboxBehavior::Normal);
                    let content = &terminal.read(cx).last_content;
                    let highlight = content.selection.as_ref().and_then(|selection| {
                        selection_highlight_lines(
                            &selection.point_range(),
                            content.display_offset,
                            num_lines as usize,
                            num_columns,
                            cell_width,
                            text_line_height,
                            bounds.origin,
                        )
                    });
                    (hitbox, highlight)
                }
            },
            // paint
            move |bounds, (hitbox, highlight), window, cx| {
                window.set_cursor_style(CursorStyle::IBeam, &hitbox);

                if let Some((start_y, lines)) = highlight {
                    HighlightedRange {
                        start_y,
                        line_height: text_line_height,
                        lines,
                        color: selection_color,
                        corner_radius,
                    }
                    .paint(true, bounds, window);
                }

                for rect in rects {
                    rect.paint(
                        bounds.origin,
                        &terminal::TerminalBounds {
                            cell_width,
                            line_height: text_line_height,
                            bounds,
                        },
                        window,
                    );
                }

                for batch in batched_text_runs {
                    batch.paint(
                        bounds.origin,
                        &terminal::TerminalBounds {
                            cell_width,
                            line_height: text_line_height,
                            bounds,
                        },
                        window,
                        cx,
                    );
                }

                // Window-level handlers so a drag keeps tracking after the
                // pointer leaves this output's bounds; they are re-registered
                // each frame.
                window.on_mouse_event({
                    let terminal = terminal.clone();
                    let this = this.clone();
                    move |event: &MouseDownEvent, phase, _window, cx| {
                        if phase != DispatchPhase::Bubble || event.button != MouseButton::Left {
                            return;
                        }
                        if bounds.contains(&event.position) {
                            terminal.update(cx, |terminal, cx| terminal.mouse_down(event, cx));
                            this.update(cx, |this, cx| {
                                this.selecting = true;
                                cx.notify();
                            });
                        } else if terminal.read(cx).last_content.selection.is_some() {
                            // Click-away deselects, like an editor selection.
                            terminal.update(cx, |terminal, _| terminal.clear_selection());
                            this.update(cx, |_, cx| cx.notify());
                        }
                    }
                });
                window.on_mouse_event({
                    let terminal = terminal.clone();
                    let this = this.clone();
                    move |event: &MouseMoveEvent, phase, _window, cx| {
                        if phase != DispatchPhase::Bubble
                            || event.pressed_button != Some(MouseButton::Left)
                            || !this.read(cx).selecting
                        {
                            return;
                        }
                        terminal.update(cx, |terminal, cx| terminal.mouse_drag(event, bounds, cx));
                        this.update(cx, |_, cx| cx.notify());
                    }
                });
                window.on_mouse_event({
                    let terminal = terminal.clone();
                    move |event: &MouseUpEvent, phase, _window, cx| {
                        if phase != DispatchPhase::Bubble
                            || event.button != MouseButton::Left
                            || !this.read(cx).selecting
                        {
                            return;
                        }
                        terminal.update(cx, |terminal, cx| terminal.mouse_up(event, cx));
                        this.update(cx, |this, cx| {
                            this.selecting = false;
                            cx.notify();
                        });
                    }
                });
            },
        )
        // We must set the height explicitly for the editor block to size itself correctly
        .h(height)
        .into_any_element()
    }
}

/// Converts the terminal's selection range into per-line highlight spans,
/// clamped to the rendered viewport. Adapted from terminal_element's
/// `to_highlighted_range_lines` for the display-only output canvas.
fn selection_highlight_lines(
    range: &terminal::Range,
    display_offset: usize,
    num_lines: usize,
    num_columns: usize,
    cell_width: Pixels,
    line_height: Pixels,
    origin: gpui::Point<Pixels>,
) -> Option<(Pixels, Vec<HighlightedRangeLine>)> {
    let display_offset = i32::try_from(display_offset).unwrap_or(i32::MAX);
    let unclamped_start_line = range.start().line.saturating_add(display_offset);
    let unclamped_end_line = range.end().line.saturating_add(display_offset);

    if unclamped_end_line < 0 || unclamped_start_line >= num_lines as i32 {
        return None;
    }

    let clamped_start_line = unclamped_start_line.max(0) as usize;
    let clamped_end_line = (unclamped_end_line as usize).min(num_lines.saturating_sub(1));
    let start_y = origin.y + clamped_start_line as f32 * line_height;

    let mut lines = Vec::new();
    for line in clamped_start_line..=clamped_end_line {
        let mut line_start = 0;
        let mut line_end = num_columns;
        if line == clamped_start_line && unclamped_start_line >= 0 {
            line_start = range.start().column;
        }
        if line == clamped_end_line && unclamped_end_line < num_lines as i32 {
            line_end = range.end().column + 1; // +1 for inclusive
        }
        lines.push(HighlightedRangeLine {
            start_x: origin.x + line_start as f32 * cell_width,
            end_x: origin.x + line_end as f32 * cell_width,
        });
    }

    Some((start_y, lines))
}

impl OutputContent for TerminalOutput {
    fn clipboard_content(&self, _window: &Window, _cx: &App) -> Option<ClipboardItem> {
        Some(ClipboardItem::new_string(self.full_text(_cx)))
    }

    fn has_clipboard_content(&self, _window: &Window, _cx: &App) -> bool {
        true
    }

    fn has_buffer_content(&self, _window: &Window, _cx: &App) -> bool {
        true
    }

    fn buffer_content(&mut self, _: &mut Window, cx: &mut App) -> Option<Entity<Buffer>> {
        if self.full_buffer.as_ref().is_some() {
            return self.full_buffer.clone();
        }

        let buffer = cx.new(|cx| {
            let mut buffer = Buffer::local(self.full_text(cx), cx)
                .with_language(language::PLAIN_TEXT.clone(), cx);
            buffer.set_capability(language::Capability::ReadOnly, cx);
            buffer
        });

        self.full_buffer = Some(buffer.clone());
        Some(buffer)
    }
}
