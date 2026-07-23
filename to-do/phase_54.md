# Phase 54 — Smooth scrolling (general/core editor)

Kind: **new feature** (general editor QoL, Zed-wide — NOT notebook-specific;
fits the fork's secondary "core editor improvements" scope). Requested by the
user 2026-07-23. Added as a 6th phase (the "5 in rotation" is a floor).

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

- [ ] Add an `editor.smooth_scrolling` setting: `EditorSettings.smooth_scrolling:
      bool`, the `Option<bool>` source in `settings_content/src/editor.rs`, the
      default in `assets/settings/default.json`, and a docs entry. **Default:**
      recommend `true` (the user asked for the feature; it's their fork) — but
      call it out; VS Code defaults its equivalent to `false`, so a conservative
      opt-in default is also defensible.
- [ ] Add smooth-scroll animation state to `ScrollManager`: a target scroll
      position, the current animated position, the last-frame `Instant`, and an
      "animating" flag. Add a method to (re)target the animation to a new scroll
      position and one to advance it one frame (ease current→target).
- [ ] Choose the easing model + tuning constant: exponential decay toward the
      target (frame-rate-independent: `current += (target - current) * (1 -
      exp(-dt / tau))`) reads well and naturally handles a moving target; pick
      `tau` (~60-100ms) as a tunable constant. Snap to target and stop once
      within an epsilon.
- [ ] Route the mouse-wheel `Lines` path (`element/mouse.rs`) through the
      smooth target instead of `editor.scroll(...)` instantly. Keep the
      `Pixels` (trackpad) path exactly as-is. A new wheel delta while animating
      RE-TARGETS (adds to the pending target) rather than restarting.
- [ ] Route keyboard scroll actions (`ScrollAmount::Line`/`Page`, `actions.rs`)
      through the smooth target under the same setting.
- [ ] Drive the animation each frame: while animating, advance the eased step
      and write it via the existing `set_scroll_position` (local scroll, NOT an
      autoscroll request), respecting `forbid_vertical_scroll`,
      `scroll_beyond_last_line` clamping, and `scroll_max`; call
      `window.request_animation_frame()` until converged, then snap + clear the
      flag. When the setting is off, keep today's instant behavior.
- [ ] Cancel/snap the animation on inputs that must be immediate: scrollbar
      drag, `scroll_to`/go-to-line/programmatic jumps, and selection-follow —
      these should not lag behind an in-flight smooth animation.

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
  from the editor — smooth scrolling there would be a separate phase.
- Upstream divergence: this touches core editor scroll; keep the change
  localized (ideally most logic inside `ScrollManager`) and record it in the
  upstream-merge playbook (phase 46) as a fork behavior addition.

## Verification

- `./script/clippy` clean; `cargo test -p editor` passes.
- User test: with `smooth_scrolling` on, mouse wheel and PageUp/PageDown animate
  smoothly and remain interruptible; the trackpad still feels native (no added
  lag); go-to-line jumps immediately; turning the setting off restores instant
  scrolling.
