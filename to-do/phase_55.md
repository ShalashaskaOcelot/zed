# Phase 55 — Smooth scrolling for the notebook cell list

Kind: **new feature** (notebook-primary; sibling to phase 54). Requested by the
user 2026-07-23. Depends conceptually on phase 54 (shares the
`editor.smooth_scrolling` setting) but is a SEPARATE mechanism — do phase 54
first so the setting and the easing approach already exist to reuse.

Goal: animate the notebook cell list's scroll changes (mouse wheel + cell-to-cell
navigation reveals) with the same short eased transition as the editor, instead
of the current instant jumps. The most visible win is **cell navigation**:
pressing up/down to move the selected cell currently *snaps* the viewport to
reveal the cell; smooth scrolling makes that glide.

## Key facts (code inspection 2026-07-23)

- The notebook renders cells through gpui's `ListState`
  (`crates/repl/src/notebook/notebook_ui.rs:191` `cell_list: ListState`, built at
  `:410` / rebuilt at `:1448`). This is a DIFFERENT scroll model from the
  editor's anchor+offset.
- `ListState` (`crates/gpui/src/elements/list.rs:54`) stores scroll position as
  `logical_scroll_top: Option<ListOffset>` — a `{ item_ix, offset_in_item:
  Pixels }` pair (index of the top item + pixel offset into it). Every public
  mover sets it INSTANTLY: `scroll_by(distance)` (`:527`), `scroll_to(offset)`
  (`:609`), `scroll_to_reveal_item(ix)` (`:626`),
  `scroll_to_reveal_item_top_aligned(ix)` (`:671`), `scroll_to_end()` (`:565`).
- Mouse-wheel scrolling of the list is handled inside `ListState` (routes through
  `scroll_by`). Cell navigation in `notebook_ui.rs` calls the
  `scroll_to_reveal_item*` family (many sites, e.g. `:2588`, `:2980`, `:3142`,
  `:3491`, `:3692`, `:3896`).
- `ListState` exposes item geometry: `bounds_for_item(ix)`
  (`notebook_ui.rs:3924` uses it) and `viewport_bounds()` (`:2216`) — usable to
  derive pixel distances for interpolation.
- `ListState` is a **SHARED gpui primitive** (used by chat, etc.), so any
  animation added there must be OPT-IN and default to today's instant behavior
  for other callers.
- Driver + timing: `window.request_animation_frame()`
  (`crates/gpui/src/window.rs:2202`) and `Instant::now()`, same as phase 54.

## Approach decision (resolve first)

Two ways to add the animation; the phase should pick one up front:

1. **In `ListState` (recommended):** add an opt-in smooth mode — a target
   `ListOffset`, animation bookkeeping, and a per-frame step that eases
   `logical_scroll_top` toward the target and calls `request_animation_frame`
   until converged. The scroll offset and item geometry both live here, so
   interpolation across variable-height items is cleanest. The notebook opts in;
   all other `ListState` users keep instant behavior. Cost: touches shared gpui.
2. **Notebook-side driver:** keep `ListState` unchanged and animate from
   `notebook_ui.rs` by repeatedly calling `scroll_to`/`scroll_by` with
   interpolated offsets each frame. Avoids changing shared gpui but is awkward:
   interpolating a `ListOffset` across variable item heights needs the notebook
   to measure per-item pixel distances (only laid-out items expose bounds), and
   an off-screen reveal target may not be measurable yet.

Recommendation: **option 1** — the geometry lives in `ListState`, and an opt-in
flag contains the blast radius.

## Tasks

- [ ] Decide + record the approach (recommend option 1) at the top of this phase
      before coding.
- [ ] (Option 1) Add opt-in smooth-scroll support to `ListState`: a way to mark
      the state as smooth, a target `ListOffset`, last-frame `Instant`, and a
      per-frame step easing current→target (reuse phase 54's exponential-decay
      model + `tau`). Default OFF so existing `ListState` users are unaffected.
- [ ] Route the wheel path and the `scroll_to*` / `scroll_to_reveal_item*` /
      `scroll_to_end` movers through the target-and-animate path when smooth mode
      is on; snap instantly when off. A new scroll while animating re-targets.
- [ ] Wire the notebook to opt in based on the `editor.smooth_scrolling` setting
      (gpui can't read editor settings, so pass the bool in from
      `notebook_ui.rs`; update it if the setting changes). Reuse the phase-54
      setting — one toggle governs both editor and notebook.
- [ ] Confirm cell navigation (up/down cell selection → `scroll_to_reveal_item*`)
      glides, and that `splice`-driven scrolls on add/remove cell
      (`notebook_ui.rs:2712`, `:3371`, `:3421`) behave sensibly (probably snap,
      not animate — a structural edit shouldn't glide).

## Risks / gaps

- **Shared primitive:** the `ListState` change must not alter behavior for any
  non-opted-in caller. Verify chat/other `ListState` users still scroll
  instantly.
- Variable item heights + `splice`: item indices and offsets shift when cells
  are added/removed/resized mid-animation — the animation must cancel or
  re-resolve its target so it can't chase a stale `ListOffset` (outputs
  streaming in resize cells live). Structural edits should snap.
- Off-screen reveal targets: revealing a far-away cell has a large pixel
  distance — cap the duration or distance so a jump across a huge notebook
  doesn't feel sluggish (or snap beyond a threshold, animate within it).
- Convergence + perf: terminate on epsilon (+ optional max duration); stop
  requesting frames once converged or when smooth mode is off.
- Interaction with existing helpers `scroll_to_item_near_top` (`:2218`) and the
  top-aligned reveals — keep their final resting position identical, only
  animate the approach.

## Verification

- `./script/clippy` clean; `cargo test -p gpui` (list tests) and
  `cargo test -p repl` pass.
- User test: with `smooth_scrolling` on, notebook wheel scrolling and up/down
  cell navigation glide; adding/removing a cell snaps; a non-notebook
  `ListState` view (e.g. chat) is unchanged; turning the setting off restores
  instant notebook scrolling.
