# Awaiting user testing

Outstanding MANUAL TEST items from phases whose implementation is complete
(the phase files themselves have been recorded in `CHANGELOG.md` and removed).
Nothing here needs implementation — these are tasks for the user to verify at
runtime.

Workflow: tick an item the moment the user confirms it. If a test FAILS, file
it in `bugs.md` (or as a backlog/phase item per the kind rules), annotate the
line with the bug number, and tick it here (the follow-up is tracked
elsewhere). Delete a section once all its boxes are ticked.

## Bug fixes awaiting confirmation

Pointer list so there's ONE place to see everything needing a test. The full
detail (symptom, analysis, fix) lives in `bugs.md`; these stay at
`fix attempted - untested` there until confirmed. When the user confirms: tick
here, add a one-line entry to `CHANGELOG.md`, and delete the bug's `bugs.md`
entry. If a fix failed, leave the bug open with the new finding and keep it
listed here.

- [ ] Bug #26 — run a cell that raises (e.g. a bad import) → red ✕ + time
      (traceback below); interrupt a running cell → still the muted ✕
      "Cancelled"; successful cells still show ✓.
- [ ] Bug #28 — Run All with no kernel → Escape the picker → cells show NO
      status marker (not Cancelled). Run All → pick a kernel → cells stay
      Pending through kernel startup (no Cancelled flash), then run. Restart
      Kernel mid-batch still cancels the queue (no frozen queue).

- [ ] Bug #25 — (2nd fix: the reveal now re-runs after the new cell is
      measured) scroll to the very bottom, add a cell below the last cell → the
      new cell scrolls fully into view.

## Phase 24 — Cell operations polish

- [x] Paste Cell Above: the COMMAND works (confirmed via the palette,
      2026-07-14). The `ctrl-shift-v` / `cmd-shift-v` keybind conflicted with an
      existing panel binding and was REMOVED at the user's request — command +
      menu entry remain, no replacement keybind for now.
- [x] Delete the only cell → replaced with a fresh empty cell; undo restores
      the original. CONFIRMED 2026-07-14.
- [x] Smart arrows: edit-mode up/down cross cell boundaries at the first/last
      line. CONFIRMED 2026-07-14.
- [ ] Completion-popup check (not yet tested): with a completions dropdown open
      in a cell, up/down should navigate the dropdown's entries — not jump
      between cells. (That's what the popup caveat means: the `!menu` gate is
      supposed to hand up/down to the popup while it's open.)

## Phase 5 — Create Python environments from the kernel picker

- [ ] No Python on PATH: clicking "Create Python Environment" shows a clear
      error toast (instead of failing silently). (Needs a machine/session
      where `python3`/`python` isn't on PATH.)

## Phase 21 — Live elapsed-time counter while a cell runs

- [x] Run a multi-second cell → the time ticks up live next to "Running…",
      then settles to the final ✓ + time. CONFIRMED 2026-07-12.
- [ ] No stray ticking/refreshing when nothing is running (ongoing
      observation): the per-cell 100ms refresh timer exists only while a cell
      is Running and self-terminates on finish/cancel, so an idle notebook
      should not be repainting on a timer. Verify nothing keeps refreshing
      after all cells finish.


