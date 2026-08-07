# Phase 67 — Page-wise follow mode

Kind: **new feature** — a SECOND follow mode alongside the existing one, not a
replacement (user 2026-08-06). Promoted from the "Alternative follow running
cell modes" backlog item, which stays there for the still-unbuilt
status-footer-anchor variant.

Goal: a follow mode for WATCHING a long run. The command-mode selection carries
"where execution is up to", and the viewport moves rarely and in whole pages
instead of nudging on every cell.

## Behaviour (settled with the user 2026-08-06, three screenshots)

1. **The selection follows execution.** As each cell starts running, the
   command-mode selection highlight moves to it, so you watch the highlight
   fall down the notebook as cells complete. This is the primary progress
   signal — not the viewport.
2. **The viewport does not move while the running cell is visible.** If the
   whole notebook fits on screen it never moves at all. Today's mode re-pins on
   every cell even when nothing needed to change; that churn is the complaint.
3. **When the running cell is NOT visible, jump a whole page.** The first cell
   that was below the fold becomes the TOP of the viewport, and the viewport
   then stays put while execution walks down the newly revealed page.
4. **At the end of the notebook, clamp.** When less than a page remains, shift
   only far enough to show the remainder including the last cell — no blank
   space past the end.
5. **A cell taller than the viewport** pins its top and is allowed to overflow.
   There is no other sensible option.
6. **Entering edit mode disables follow mode** (user's answer to the
   focus-stealing problem). Follow is for watching; the moment you start
   editing, you are not watching. This is what makes selection-following safe —
   without it, moving the selection during a run would fight a user editing
   elsewhere. Turning follow back on is manual (the existing toggle).

The user's rationale for the big jumps, which deliberately contradicts the
smallest-possible-movement rule used everywhere else in the notebook: in follow
mode you are watching, not interacting, so a large deliberate jump is not
disorienting the way it would be mid-edit.

**Consequence worth checking at review time (user's own observation):** except
for the last cell on each page — and cells taller than the viewport — this
framing naturally leaves each running cell's status footer and the start of its
output on screen, because the cell sits above the fold with room below it. If
that holds in practice, it also covers most of what the deferred
status-footer-anchor item was for.

## Where this lives in the code

- The existing mode is `follow_running_cell: bool` +
  `follow_scroll_to(index)` (`notebook_ui.rs`), called from
  `advance_run_queue` as each cell is submitted. That call site is where the
  new mode hooks in too.
- `follow_scroll_to` uses `ListState::scroll_to_item_near_top`. Page-wise needs
  different primitives: "is item `ix` currently visible?" and "put item `ix` at
  the top". `ListState` exposes `viewport_bounds()`, and gpui's list already
  has index-anchored scrolling (`ListOffset { item_ix, offset_in_item }`) —
  putting an item at the top is exactly `offset_in_item: 0` with no margin,
  which is what `scroll_to_item_near_top(ix, px(0.))` already does. The missing
  piece is the visibility query; check what `ListState` can answer before
  adding anything to gpui (a new gpui method is a merge-conflict surface —
  prefer composing existing ones).
- Selection movement is `set_selected_index(index, false, window, cx)` — the
  same call the existing reveal uses, minus its own scroll.
- Bug #65 (classic follow landing mid-cell on a very long cell) lives in the
  CURRENT mode's reveal path. It is not inherited here, and it is not fixed by
  this phase either — the classic mode keeps its own bug.

## Setting

`repl.notebook_follow_mode`, an enum, replacing nothing:
`minimal` (today's near-top pin, the default) | `page` (this). The existing
`ToggleFollowRunningCell` action stays as the on/off switch; this setting only
chooses WHICH follow behaviour runs when it is on. Wire it up the same way as
`notebook_run_landing_mode` (settings_content → repl_settings → page_data →
`docs/src/repl.md`), which is the existing enum-setting precedent.

## Tasks

- [ ] Add the `repl.notebook_follow_mode` enum setting end to end
      (settings_content, repl_settings, settings UI page item, docs), default
      `minimal` so nothing changes for anyone who doesn't opt in.
- [ ] Add a visibility query for a list item (composed from `ListState`'s
      existing viewport/offset APIs if at all possible; only touch gpui if it
      genuinely cannot be answered from outside).
- [ ] Implement the page-wise reveal: no-op while the running cell is visible;
      otherwise put the first previously-below-the-fold cell at the top,
      clamped at the end of the notebook so the last cell is never cut off.
- [ ] Move the SELECTION to each cell as it starts running, in page mode only,
      leaving edit/command mode otherwise untouched.
- [ ] Turn follow mode OFF when the user enters edit mode, in both modes (it is
      the same focus-stealing risk in each). Make sure the toggle's UI state
      reflects that it switched off.
- [ ] Unit-test the page arithmetic where it is pure: given item heights, a
      viewport height and a running index, which index becomes the new top —
      including the clamp at the end and the taller-than-viewport case.
- [ ] `./script/clippy` clean and `cargo test -p repl` passes.

## User tests (runtime)

- [ ] With `notebook_follow_mode: "minimal"` (default) nothing changes.
- [ ] In `page` mode with a notebook that fits on screen: Run All never moves
      the viewport, and the selection highlight walks down the cells.
- [ ] With a longer notebook: the viewport holds still while execution walks
      down the visible cells, then jumps a full page when execution passes the
      fold, putting the previously-hidden cell at the top.
- [ ] At the end of the notebook the last jump stops with the final cell
      visible — no scrolling past the end into blank space.
- [ ] A cell taller than the viewport pins its top and overflows, and execution
      continues normally afterwards.
- [ ] Clicking into a cell to edit during a run turns follow mode off and
      leaves the cursor alone from then on.
