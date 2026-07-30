# Phase 58 — Smooth scrolling for keyboard scroll actions

Kind: **change to existing behaviour**. Split out of phase 54 on 2026-07-30
because it is NOT a safe drop-in: it changes when `scroll_position()` reflects a
keyboard scroll, and several callers read that back synchronously.

Phase 54 delivered the animation infrastructure (`ScrollManager`'s
smooth-scroll target/step, `Editor::scroll_smoothly`, the
`editor.smooth_scrolling` setting) and routed the MOUSE WHEEL through it. This
phase extends it to `Editor::scroll_screen` (PageUp/PageDown, `ctrl-d`/`ctrl-f`,
vim `z` commands, etc.).

## Why this was split out (do not skip — it is the whole problem)

`scroll_screen` currently applies its new position SYNCHRONOUSLY, and callers
depend on that:

- **Vim** (`crates/vim/src/normal/scroll.rs:130`) calls
  `editor.scroll_screen(&amount, window, cx)` and then IMMEDIATELY reads
  `editor.scroll_top_display_point(...)` (`:144`) to decide where to put the
  cursor. If the scroll only starts a glide, vim reads the PRE-scroll position
  and places the cursor wrong — a real correctness bug, not just a test failure.
- **Tests** assert the position synchronously, e.g.
  `editor_tests.rs:2932-2955` (`scroll_screen(Page(1.))` → asserts
  `scroll_position() == (0., 3.)`). Vim has its own scroll tests too.

So making `scroll_screen` animate requires deciding what `scroll_position()`
means during a glide, and updating every synchronous read-back accordingly.

## Approach options (decide before coding)

1. **Visual-only animation (recommended, matches VS Code).** The LOGICAL scroll
   position updates instantly — so every read-back, vim, and existing test stays
   correct — and only the RENDERED offset eases toward it. Keyboard scrolling
   then animates with zero semantic change. Cost: the editor element must paint
   at an interpolated offset while computing its visible line range from the
   target position (the range must be widened by the in-flight delta, or rows at
   the leading edge will be missing). Touches `EditorElement`
   layout/prepaint/paint. Higher implementation risk, but it is the correct
   long-term model — and if adopted, the phase-54 wheel path should move onto it
   too, so the two paths share one mechanism.
2. **Logical animation + fix the read-backs.** Route `scroll_screen` through
   `Editor::scroll_smoothly` (already exists) and give vim (and anything else
   that reads back) a way to ask for the SETTLED position — e.g. have
   `pending_smooth_scroll_target()` be the value those callers use, or have vim
   snap the animation before reading. Smaller diff, but every current and future
   synchronous reader of `scroll_position()` becomes a potential bug, and it
   diverges from upstream vim behaviour.

Recommendation: option 1 if this is worth doing properly; otherwise leave
keyboard scrolling instant (it is defensible — the jump is expected there).

## Tasks

- [ ] Decide and record the approach (1 vs 2) at the top of this file.
- [ ] Implement smooth keyboard scrolling for `Editor::scroll_screen` under the
      existing `editor.smooth_scrolling` setting.
- [ ] Keep vim's scroll+cursor-reposition correct (`normal/scroll.rs`) — verify
      `ctrl-d`/`ctrl-u`/`ctrl-f`/`ctrl-b` land the cursor exactly where they do
      today.
- [ ] Keep `editor_tests.rs` scroll_screen assertions and the vim scroll tests
      passing (adjust them ONLY if the semantic change is deliberate and
      documented here).
- [ ] `./script/clippy` clean; `cargo test -p editor -p vim` passes.
- [ ] **User test:** PageUp/PageDown and vim scroll commands glide; the cursor
      still lands correctly; turning `smooth_scrolling` off restores instant
      behaviour.

## Risks / gaps

- Option 1 must widen the painted row range by the animation delta or the
  leading edge will show blank rows mid-glide.
- Autoscroll (cursor reveal) interacts: a keyboard scroll that also moves the
  cursor triggers autoscroll, which currently cancels the glide (phase 54 makes
  ANY external scroll supersede it). Decide whether cursor-follow should ride
  the same animation or keep snapping.
- Do not animate `scroll_to`/go-to-line style jumps — those should stay instant.
