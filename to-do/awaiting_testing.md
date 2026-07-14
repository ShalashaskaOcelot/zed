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

- [ ] Bug #21 — collapse a cell + save → NO "changed on disk" toast; then let
      VS Code edit the file under unsaved Zed changes → the toast still appears
      for a real external change.
- [ ] Bug #22 — run several near-instant cells → each shows a small ms duration
      next to the ✓ (not a bare ✓).
- [ ] Bug #23 — click a cell's gutter/margin → selects it in command mode
      without entering edit; clicking the editor text still enters edit mode.
- [ ] Bug #24 — run a cell that produces a DataFrame table and/or a matplotlib
      plot, save, close, reopen → the rich output is still there. Then re-test
      output-collapse persistence (Phase 23), which was blocked on this.
- [ ] Bug #25 — scroll to the bottom, add a cell below the last cell → it
      scrolls into view above the kernel status bar (not hidden behind it).

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

