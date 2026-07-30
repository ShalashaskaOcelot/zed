# Phase 55 — Smooth scrolling for the notebook cell list

⚠️ AWAITING USER TESTING (implementation complete, clippy-clean).

Kind: **new feature** (notebook-primary; sibling to phase 54). Requested by the
user 2026-07-23. Shares the `editor.smooth_scrolling` setting from phase 54.

**Approach decided during implementation (2026-07-30) — NEITHER of the two
options originally written below.** Both assumed easing toward an absolute
`ListOffset` target, which is exactly the fragile math behind bug #45: item
heights are estimates until laid out, so an absolute target across unmeasured
items is unreliable. Instead the list keeps a **pending pixel delta that decays
to zero**: a wheel tick adds its distance to `pending_smooth_delta`, and each
frame applies a fraction of what remains through the list's existing
`scroll()`. This needs no cumulative-height math at all, so it cannot regress
landing accuracy, and repeated ticks simply add to the pending distance.
Cell-navigation reveals are deliberately NOT animated — they stay exact (see
the note under Tasks).

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

- [x] Decided + recorded the approach (pending-delta decay — see the note at the
      top; neither original option).
- [x] Add opt-in smooth-scroll support to `ListState`: `smooth_scroll` flag,
      `pending_smooth_delta`, `smooth_scroll_last_step`, plus
      `set_smooth_scroll()`, `cancel_smooth_scroll()` and
      `take_smooth_scroll_step()`. Default OFF, so every other `ListState` user
      (chat, pickers, csv preview, settings UI) is unaffected.
- [x] Route the WHEEL path through it: `ScrollDelta::Lines` adds to the pending
      delta and refreshes; the element's `paint` applies one eased step per
      frame and calls `request_animation_frame` until it is spent. Trackpad
      (`ScrollDelta::Pixels`) still applies immediately.
- [x] Wire the notebook to opt in from `editor.smooth_scrolling` (re-applied
      each render in `cell_list()`, so toggling the setting takes effect live).
- [x] Absolute scrolls cancel any in-flight glide (`scroll_to`,
      `scroll_to_reveal_item`, `..._top_aligned`, `scroll_to_item_near_top`,
      `scroll_to_end`, `reset`, scrollbar drag). This is what keeps cell
      navigation landing EXACTLY on target — i.e. it preserves the bug #45 fix
      rather than risking it.

**Deliberately not animated: cell-navigation reveals.** The original phase
called these "the most visible win", but bug #45 was *just* fixed by making
those jumps index-anchored and exact, and easing them would reintroduce the
landing inaccuracy that fix removed. Structural edits (add/remove cell) snap for
the same reason. If gliding navigation is still wanted, it needs its own phase
that keeps the final resting position exact.

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

- [x] `./script/clippy -p gpui -p repl` clean; `cargo test -p gpui --lib list`
      passes (27 passed, 0 failed — covers scroll, reveal, scrollbar drag and
      follow-tail, confirming the default-off path is unchanged).
- [ ] **User test:** with `smooth_scrolling` on (the default), notebook WHEEL
      scrolling glides and repeated ticks accumulate; up/down cell navigation
      and Home/End still land EXACTLY on target (no drift — this is the bug #45
      behaviour); adding/removing a cell snaps; a non-notebook `ListState` view
      (e.g. chat, the kernel picker) is completely unchanged; setting
      `"smooth_scrolling": false` restores instant notebook scrolling.
