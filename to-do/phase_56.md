# Phase 56 — Notebook cell-list scrollbar

⚠️ AWAITING USER TESTING

Kind: **new feature** (notebook-primary QoL). Requested by the user 2026-07-23
(promoted from the backlog item "Notebook scrollbar missing / hidden behind the
right bar", reported 2026-07-16). Also unblocks verifying far jumps for bug #45
(no scrollbar today means no way to gauge scroll position / jump distance).

Goal: give the notebook cell list a visible vertical scrollbar, positioned to
the LEFT of the right-hand control sidebar (not occluded by it).

## Key facts (code inspection 2026-07-23)

- The cell list renders via the raw gpui `list()` element with `.size_full()`
  (`notebook_ui.rs:4405` `cell_list()`), which draws NO scrollbar of its own —
  so there was never one (confirmed; the user's suspicion was right).
- Layout (`render`, `notebook_ui.rs:4679-4685`): an `h_flex().gap_2()` row with
  two SEPARATE siblings — `div().flex_1().h_full().child(cell_list)` and
  `render_notebook_controls(...)`. The control bar is its own column (not an
  overlay), so the list column's right edge already sits left of the gap and
  the control bar. Attaching the scrollbar to the list column therefore lands
  it exactly where wanted, with no occlusion and no manual insets.
- `ListState` implements `ui::ScrollableHandle` (`scrollbar.rs:949`: offset /
  set_offset / max_offset / drag start+end / viewport), so a `Scrollbars` can
  track the existing `cell_list` handle directly — no new scroll plumbing.
- Idiomatic wiring (matches `code_context_menus.rs:1158`, `markdown.rs:2262`):
  build `Scrollbars::for_settings::<editor::EditorSettingsScrollbarProxy>()
  .show_along(ScrollAxes::Vertical).tracked_scroll_handle(&self.cell_list)` and
  apply it to the list `div` with `.custom_scrollbars(scrollbars, window, cx)`.
  Using `for_settings::<EditorSettingsScrollbarProxy>` makes visibility/autohide
  follow the user's existing editor scrollbar setting, consistent with the rest
  of Zed. Imports needed: `ui::{ScrollAxes, Scrollbars, WithScrollbar}`.

## Tasks

- [x] Import `ScrollAxes, Scrollbars, WithScrollbar` from `ui` in `notebook_ui.rs`.
- [x] Wrap the cell-list column div with `.custom_scrollbars(...)` tracking the
      `cell_list` `ListState`, vertical axis, editor-settings visibility.
- [x] Confirm placement: scrollbar sits at the right edge of the list column,
      left of the `gap_2` and the control sidebar (no occlusion), by
      construction (separate flex siblings) — verified by inspection; user to
      confirm visually.
- [x] Polish (user feedback 2026-07-23 → 2026-07-30): first attempt added
      `.thumb_color`/`.thumb_padding` knobs to `ui::Scrollbars` and set a lighter
      shade + `thumb_padding(px(2.))` — but that read as TOO THIN. Final: use
      `.style(ScrollbarStyle::Editor)` so the notebook scrollbar matches the
      editor's (wider, full-width thumb, no side gaps), fixing both the original
      "space either side" and the "too thin" feedback. (The `.thumb_color` /
      `.thumb_padding` knobs remain in `ui::Scrollbars` — backward-compatible,
      unused by the notebook now; harmless, could be removed in a future tidy.)
- [ ] **User test:** open a notebook — a vertical scrollbar is visible on the
      cell list (per the user's scrollbar visibility setting), sits to the LEFT
      of the right control bar (not hidden behind it), tracks scroll position,
      and can be dragged to scroll.
- [ ] **User test (polish):** the thumb is easier to spot (lighter) and the
      gutter is tighter (less empty space either side). Values are first-cut —
      tune `thumb_color` / `thumb_padding` at the call site if desired.

## Known limitation (note, not a blocker)

- The list is virtualized and unmeasured cells contribute 0 px to its height
  (same root as bug #45), so the thumb size/position is APPROXIMATE until cells
  are measured by scrolling, then converges. Acceptable for a first cut; a
  future enhancement could improve the estimate (or `measure_all`, weighed
  against open-time cost on huge notebooks — see #45 discussion).

## Verification

- `./script/clippy` clean; `cargo build -p repl` succeeds.
- User confirms the scrollbar is visible, correctly placed, and draggable.
