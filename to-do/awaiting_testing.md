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

- [ ] Bug #22 — (2nd fix: clear_outputs no longer wipes the duration) run
      several near-instant cells → each shows a small ms duration next to the ✓,
      none bare, none inconsistently blank vs 0ms.
- [ ] Bug #23 — (2nd fix: gutter-only select) click a code cell's gutter/accent
      strip → selects in command mode; click the cell body/text → ENTERS EDIT
      mode (regression check); shift/ctrl-click ranges still work.
- [ ] Bug #25 — scroll to the bottom, add a cell below the last cell → it
      scrolls into view above the kernel status bar (not hidden behind it).

## Phase 24 — Cell operations polish

- [ ] Paste Cell Above: copy/cut a cell, then `ctrl-shift-v` / `cmd-shift-v`
      (or the "Paste Cell Above" menu entry) inserts it ABOVE the current cell;
      multi-cell paste lands the whole block above.
- [ ] Delete the only cell (or cut/delete a selection covering every cell) →
      the notebook is left with one fresh empty code cell; undo restores the
      original cell(s) and removes the fresh one.
- [ ] Smart arrows: in edit mode, up/down move line-by-line within a cell and
      cross into the previous/next cell at the first/last line. With a
      completion popup open, up/down navigate the popup (not cells).

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


