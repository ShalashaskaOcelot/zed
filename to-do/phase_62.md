# Phase 62 — Global kernel busy/idle indicator

Kind: **change to existing behaviour** (the indicator exists; this makes it
readable). Promoted from the backlog at the user's request (2026-08-06);
originally raised 2026-07-16.

Goal: one indicator, visible regardless of scroll position, that says at a
glance whether the kernel is idle or actively working. NOT the per-cell spinner
(that already exists) — this is about the notebook as a whole.

## Where it goes

The top kernel strip (`render_kernel_strip`, `notebook_ui.rs:4438`) is already
pinned above the cell list, so it is on screen at any scroll offset — the right
home for this. It already maps `KernelStatus` to an icon + colour
(`notebook_ui.rs:4461`: Idle = `Circle`/Success, Busy = `ArrowCircle`/Warning,
Starting = `ArrowCircle`/Muted, Error = `XCircle`/Error, …) but the icon is
STATIC, small, and sits next to the kernel name, so "busy" and "idle" look
nearly identical in peripheral vision — which is the complaint.

## Approach

- **Animate the busy state.** `ui`'s `CommonAnimationExt::with_rotate_animation`
  (`crates/ui/src/traits/animation_ext.rs`) spins any `Transformable` — apply it
  to the status icon for `Busy`, `Starting`, `Restarting` and `ShuttingDown`
  (every state that means "something is happening"), leaving `Idle`, `Error` and
  `Shutdown` static. Motion is what makes it readable from across the screen.
- **Label the state.** Add a short muted label next to the icon (`Busy` /
  `Idle` / `Starting…`), from the existing `KernelStatus` display string
  (`kernels/mod.rs:778`) rather than a second mapping. The strip currently shows
  the kernel NAME only, so status is icon-only today.
- **Colour weight.** Keep Success/Warning/Error semantics; make sure the busy
  colour reads as active rather than as a warning at small size.
- Only the notebook strip is in scope. The inline-REPL status
  (`notebook_ui.rs:4394`, `IconName::ReplNeutral`) is a different surface and is
  not part of this phase.

## Watch out for

- The spinner must not animate forever after the kernel dies — drive it purely
  off the CURRENT `KernelStatus`, never a latched flag, so any terminal state
  stops it.
- `creating_kernel_name` already forces `KernelStatus::Starting` while an env
  builds (phase 48); the animation must follow that too, since that is exactly a
  "something is happening" state.
- Keep the strip compact: it is right-aligned next to the kernel selector and
  the failure count from phase 60, so the label must be small
  (`LabelSize::Small`, `Color::Muted`) and must not wrap.

## Tasks

- [ ] Rotate the status icon while the kernel is Busy / Starting / Restarting /
      ShuttingDown; static otherwise. Derived from `KernelStatus` each render.
- [ ] Add the small muted status label next to the icon, sourced from
      `KernelStatus`'s existing display string.
- [ ] Verify the env-creation `Starting` override animates too.
- [ ] Check the strip's layout at a narrow pane width — icon + label + failure
      count + kernel name must not wrap or clip.
- [ ] `./script/clippy` clean and `cargo test -p repl` passes.

## User tests (runtime)

- [ ] Run a long cell: the strip's status is obviously animated/labelled Busy
      from anywhere in the notebook, and returns to Idle when it finishes.
- [ ] Scroll far down mid-run — the indicator is still visible and still moving.
- [ ] Stop/Restart the kernel and let a kernel error occur: the animation stops
      in every terminal state (no perpetual spinner).
- [ ] Creating a new env shows the animated Starting state.
