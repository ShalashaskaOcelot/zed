# Phase 60 — Cell error detection and "go to error" navigation

Kind: **new feature** (notebook-primary). Promoted from the backlog at the
user's request 2026-07-31.

Goal: when a cell fails, let the user jump straight to it instead of scrolling
to hunt for it — VS Code's "go to error". Most valuable after a Run All on a
long notebook, where the failing cell can be far off-screen and stop-on-error
means everything below it is now Cancelled.

## Key facts (code inspection 2026-07-31)

The DETECTION half already exists. Do not re-implement it:

- `CellExecutionStatus::Failed` is already set on a cell that raises and is
  already drawn as a red ✕ (`notebook/cell.rs`).
- Errors are already a distinct `Output::ErrorOutput(ErrorView)` (`outputs.rs`),
  separate from stream output.
- Stop-on-error already halts a batch: the `ReplyStatus::Error | Aborted` arm in
  `notebook_ui.rs` (~line 5705) calls `cancel_run_queue`.

So the notebook already knows exactly which cell failed; nothing surfaces it as
a navigation target. This phase is navigation and affordance only.

## Two questions already settled (user 2026-07-31) — do not re-litigate

- **Most-recent vs first-in-document-order is a non-question.** Stop-on-error
  means nothing after the failure runs, so an unhandled error is ALWAYS the most
  recent. Handled errors never reach `Failed` unless the handler re-raises.
- **No run-scoping is needed — `Failed` is accurate by construction.** A cell
  re-run this session either raised again (a NEW error) or succeeded (no longer
  `Failed`); a queued-but-not-yet-reached cell is `Pending`, not `Failed`, even
  while still displaying last run's output. The only cell keeping a `Failed`
  status from a previous run is one neither re-run nor queued — and that status
  is still truthful. So: just navigate `Failed` cells, no session accumulator.

## Tasks

- [x] Add a `GoToError` action that selects the failed cell and reveals it, plus
      `NextError` / `PreviousError` cycling `Failed` cells in document order
      (wrapping at the ends) for when cells were run individually and more than
      one failed. No-op quietly when there are no failed cells.
- [x] Reveal the ERROR, not the cell top. Land so the bottom of the cell's
      source and its status footer are visible with as much of the error output
      below as fits — the traceback is the informational part. This is the same
      requirement as the deferred follow-running-cell backlog item; if a shared
      helper falls out naturally, put it where both can use it, but do not
      implement follow mode here.
- [x] Make it reachable without knowing the shortcut: a failure indicator in the
      top kernel strip (`render_kernel_strip`) that is clickable → jumps to the
      error. The strip is pinned, so this doubles as the "did something fail?"
      signal at any scroll position.
- [x] Keybinding + command-palette entries, consistent with the existing cell
      navigation actions.
- [x] `./script/clippy` clean; `cargo test -p repl` passes. Add a test covering
      the `Failed`-cell selection order (next/previous/wrap) at the state level.
- [ ] **User test:** Run All a notebook whose middle cell raises → the strip
      shows a failure indicator; the action (and clicking the indicator) jumps
      to that cell with the traceback in view, not the cell's first line. With
      several individually-run failures, next/previous cycles them and wraps.
      With no failures, the action does nothing and no indicator shows.

## Implementation notes (2026-07-31)

- `Cell::has_failed()` exposes the existing `CellExecutionStatus::Failed`.
- The reveal uses a new `ListState::scroll_to_item_bottom_aligned` (gpui): it
  puts the item's BOTTOM at the viewport bottom, so a cell taller than the
  viewport lands on its output rather than its source. It degrades to the plain
  minimal reveal when the cell already fits (or is unmeasured), since the whole
  cell is visible either way and moving it would be gratuitous.
- Next/previous wrap logic is factored into pure `next_failed_index` /
  `previous_failed_index` helpers and unit-tested directly, which avoids
  standing up a whole notebook entity to test the arithmetic.
- Keybindings: `f8` / `shift-f8` (next/previous), mirroring the editor's
  go-to-diagnostic pair. `GoToError` deliberately has NO keybinding —
  `ctrl-shift-e` / `cmd-shift-e` is already project-panel focus, and shadowing
  it inside a notebook would repeat the phase 24 `ctrl-shift-v` conflict. It is
  reachable from the palette and the kernel-strip button.

## Explicitly NOT in this phase

- **Caught exceptions are out of scope.** An exception the user's code catches
  and prints with `traceback.print_exc()` is stderr stream output and the cell
  legitimately succeeds (green tick — confirmed with the user 2026-07-31). This
  keys on execution status, so it will not — and should not — find those.
- Follow-running-cell changes (deferred backlog item), though the reveal helper
  is shared ground.

## Risks / gaps

- The reveal must not fight the phase 59 top-pinned output viewport: once long
  output is truncated from the top, "as much error output as fits" is the
  truncated view. Land this after or alongside 59 and check they compose.
- Clearing the indicator: decide when the strip stops showing a failure —
  simplest is "whenever no cell is `Failed`", which follows from re-running the
  cell, and needs no extra state.
