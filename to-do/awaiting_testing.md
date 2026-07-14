# Awaiting user testing

Outstanding MANUAL TEST items from phases whose implementation is complete
(the phase files themselves are archived). Nothing here needs implementation —
these are tasks for the user to verify at runtime.

Workflow: tick an item the moment the user confirms it. If a test FAILS, file
it in `bugs.md` (or as a backlog/phase item per the kind rules), annotate the
line with the bug number, and tick it here (the follow-up is tracked
elsewhere). Delete a section once all its boxes are ticked.

## Phase 5 — Create Python environments from the kernel picker

- [ ] No Python on PATH: clicking "Create Python Environment" shows a clear
      error toast (instead of failing silently). (Needs a machine/session
      where `python3`/`python` isn't on PATH.)

## Phase 17 — Execution status & queue correctness

- [x] Interrupt mid-batch: the interrupted (running) cell now shows the muted
      ✕ Cancelled (traceback still visible) instead of ✓ + time.
      (Change made 2026-07-12, commit 051c3c6.) CONFIRMED 2026-07-12 (screenshot).
- [x] Rerun during a run no longer hangs: pressing Run All while cells are
      running interrupts the old run and the new batch actually starts
      (deadlock fix 051c3c6 — previously everything could sit stuck if the
      kernel wasn't busy at that instant). CONFIRMED 2026-07-12.
- [x] Re-run after a run finishes: cells can be re-queued/run again without a
      kernel restart (regression fix 93f8a96). CONFIRMED 2026-07-12.
- [ ] Ongoing observation: no cells get stuck showing "Running" (channel-race
      fix 5fcfca1 — user monitoring, nothing seen since).

## Phase 21 — Live elapsed-time counter while a cell runs

- [x] Run a multi-second cell → the time ticks up live next to "Running…",
      then settles to the final ✓ + time. CONFIRMED 2026-07-12.
- [ ] No stray ticking/refreshing when nothing is running (ongoing
      observation): the per-cell 100ms refresh timer exists only while a cell
      is Running and self-terminates on finish/cancel, so an idle notebook
      should not be repainting on a timer. Verify nothing keeps refreshing
      after all cells finish.

## Phase 22 — Multi-select cells

- [x] shift+down/up in command mode grows/shrinks a contiguous selection;
      plain up/down collapses it. CONFIRMED 2026-07-12.
- [x] shift+click selects the range to the clicked cell. CONFIRMED 2026-07-12.
- [ ] ctrl/cmd+click toggles individual cells into a discontiguous selection;
      ctrl/cmd+up/down do nothing.
- [x] Delete/cut with a multi-selection removes all selected (one undo restores
      them all). CONFIRMED 2026-07-12. NOTE: restored cells lose their OUTPUT →
      backlog item (retain output through cut/paste and delete/undo).
- [x] copy+paste reproduces all selected cells. CONFIRMED 2026-07-12.
- [ ] Run with a multi-selection executes the selected cells in order.
- [ ] Move up/down shifts a contiguous selected block (and does nothing for a
      discontiguous selection); convert converts all selected.
- [x] Add cell / paste act on the primary cell. CONFIRMED 2026-07-12
      (duplicate / Enter still to verify).

## Phase 23 — Collapse / expand cell input & output

- [x] Toolbar chevron collapses the input to a one-line summary; clicking the
      summary (or the chevron) expands it; the cell still runs while collapsed.
      CONFIRMED 2026-07-12.
- [x] Output "…" menu collapses/expands the output; clicking the collapsed row
      expands it. CONFIRMED 2026-07-12.
- [x] Input (code) collapse state survives save + reopen. CONFIRMED 2026-07-12.
- [ ] Output collapse state survives save + reopen. BLOCKED by bug #24 (outputs
      themselves are dropped on save, so there is no output to re-collapse on
      reopen) — retest once bug #24 is fixed.
- [x] Toggling collapse marks the notebook dirty (save persists it). CONFIRMED
      2026-07-12.

