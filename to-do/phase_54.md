# Phase 54 — Smooth scrolling (general/core editor)

⚠️ AWAITING USER TESTING (implementation complete, clippy-clean).

Kind: **new feature** (general editor QoL, Zed-wide — NOT notebook-specific;
fits the fork's secondary "core editor improvements" scope). Requested by the
user 2026-07-23. Added as a 6th phase (the "5 in rotation" is a floor).

**Scope change during implementation (2026-07-30):** keyboard scroll actions
were split out into **phase 58**. `Editor::scroll_screen` applies its position
synchronously and callers depend on that — vim
(`crates/vim/src/normal/scroll.rs:130`) calls it and then IMMEDIATELY reads
`scroll_top_display_point` (`:144`) to place the cursor, so animating it would
put vim's cursor in the wrong place (a correctness bug, not just a test
failure); `editor_tests.rs:2932-2955` asserts the position synchronously too.
Doing it properly means animating at the RENDER layer (VS Code's model) rather
than the logical position — a different, larger change. The wheel path, which
this phase's own notes call "the primary target", is delivered.

Goal: animate editor scroll-position changes over a short eased transition
instead of instant jumps — the equivalent of VS Code's `editor.smoothScrolling`.
Applies to user-driven scrolls (mouse wheel + keyboard scroll actions).
Trackpad pixel scrolling is LEFT AS-IS (it already produces smooth, momentum
deltas — animating it again would add lag).

## Key facts (code inspection 2026-07-23)

- Scroll state lives in `ScrollManager` (`crates/editor/src/scroll.rs`) as a
  `ScrollAnchor` (a buffer `Anchor` + a line/glyph `offset`).
  `set_scroll_position` → `set_anchor` (`scroll.rs:400-458`) apply the new
  position INSTANTLY; there is no animation anywhere today.
- Mouse-wheel handler: `EditorElement`'s scroll `on_mouse_event`
  (`crates/editor/src/element/mouse.rs:504-586`). It already distinguishes
  input kinds: `ScrollDelta::Pixels` = TRACKPAD (smoothed via
  `ongoing_scroll.filter`, momentum) vs `ScrollDelta::Lines` = MOUSE WHEEL
  (converted lines→pixels, applied instantly via `editor.scroll(...)`). The
  wheel (`Lines`) path is the primary target.
- Keyboard scroll actions use `ScrollAmount::{Line,Page}`
  (`crates/editor/src/scroll/scroll_amount.rs`, handlers in
  `crates/editor/src/scroll/actions.rs`).
- Cursor-reveal scrolling is `autoscroll.rs` (OUT OF SCOPE here — see gaps).
- Frame driver: `window.request_animation_frame()` schedules a redraw + notify
  on the next frame (`crates/gpui/src/window.rs:2202`); `on_next_frame` runs a
  callback next frame. gpui's own element animations (`elements/animation.rs`)
  use `Instant::now()` deltas — the same approach works here (`Instant::now()`
  is fine in Rust; the workflow-script restriction does not apply).
- Settings: `EditorSettings` (`crates/editor/src/editor_settings.rs:19`, e.g.
  `scroll_sensitivity`, `fast_scroll_sensitivity`, `mouse_wheel_zoom`), sourced
  from `crates/settings_content/src/editor.rs` and defaulted in
  `assets/settings/default.json` (~line 700). Add the new setting in all three
  (+ `.unwrap()` wire-up at `editor_settings.rs:264` pattern) plus docs.

## Tasks

- [x] Add an `editor.smooth_scrolling` setting: `EditorSettings.smooth_scrolling:
      bool`, the `Option<bool>` source in `settings_content/src/editor.rs`, the
      default in `assets/settings/default.json`, and a docs entry. **Default:
      `true`** (decided by the user 2026-07-23). Also mapped VS Code's
      `editor.smoothScrolling` in `settings/src/vscode_import.rs`.
- [x] Add smooth-scroll animation state to `ScrollManager`
      (`smooth_scroll_target`, `smooth_scroll_last_step`,
      `smooth_scroll_running`, `applying_smooth_scroll`, `smooth_scroll_task`)
      plus `pending_smooth_scroll_target()` and `cancel_smooth_scroll()`.
- [x] Easing: frame-rate-independent exponential decay
      (`current += (target - current) * (1 - exp(-dt/tau))`) with
      `SMOOTH_SCROLL_TAU = 0.07s`, stepped every 8ms. Snaps within
      `SMOOTH_SCROLL_EPSILON`, and also stops when a step makes no progress
      (`SMOOTH_SCROLL_STALL_EPSILON`) so an unreachable/clamped target can't
      spin forever.
- [x] Route the mouse-wheel `Lines` path (`element/mouse.rs`) through
      `Editor::scroll_smoothly`. The `Pixels` (trackpad) path is untouched.
      Successive wheel ticks extend `pending_smooth_scroll_target()` so input
      accumulates instead of restarting the glide.
- [~] Keyboard scroll actions — **moved to phase 58** (see the scope note
      above: vim reads the scroll position back synchronously).
- [x] Drive the animation: a task on the editor steps it every 8ms via
      `set_scroll_position` (local, not an autoscroll request), so existing
      `forbid_vertical_scroll` / `scroll_beyond_last_line` clamping applies
      unchanged. Terminates on convergence, on stall, or when superseded.
- [x] Cancel/snap on inputs that must be immediate — handled centrally in
      `ScrollManager::set_anchor`: any scroll that isn't the animation's own
      frame (guarded by `applying_smooth_scroll`) cancels the glide, so
      scrollbar drags, go-to-line, and cursor autoscroll all land immediately.

## Risks / gaps

- Do NOT double-animate the trackpad `Pixels` path (adds perceptible lag);
  gate the smooth path on `ScrollDelta::Lines` + keyboard actions only.
- Anchor model: the animation interpolates in scroll-position (line) space and
  re-derives the anchor via `set_scroll_position` each frame — verify this
  stays stable near buffer edges and under `scroll_max` clamping, and that
  overscroll/top-overscroll notifications aren't disrupted.
- Convergence + perf: a smooth scroll requests a redraw every frame while
  active — guarantee termination (epsilon + optionally a max duration) so it
  can't spin. Stops the moment it converges or the setting is off.
- Interruptibility must feel natural: mid-animation wheel input re-targets;
  a jump (go-to-line) snaps.
- **Out of scope (note as follow-ups, do NOT implement here):** (1) animating
  cursor-reveal `autoscroll` — risks laggy cursor tracking on fast movement,
  deserves its own decision; (2) the NOTEBOOK cell list, which scrolls via a
  gpui `ListState` (`crates/gpui/src/elements/list.rs`), a DIFFERENT mechanism
  from the editor — that is **phase 55**.
- Upstream divergence: this touches core editor scroll; keep the change
  localized (ideally most logic inside `ScrollManager`) and record it in the
  upstream-merge playbook (phase 46) as a fork behavior addition.

## Verification

- [x] `./script/clippy -p editor` clean; `cargo test -p editor scroll` passes
      (21 passed, 0 failed).
- [ ] **User test:** with `smooth_scrolling` on (the default), the MOUSE WHEEL
      glides instead of jumping and stays interruptible (a new tick mid-glide
      extends it rather than restarting); the trackpad still feels native with
      no added lag; scrollbar drag and go-to-line land immediately; setting
      `"smooth_scrolling": false` restores instant wheel scrolling.
      (PageUp/PageDown are NOT animated — that is phase 58.)
