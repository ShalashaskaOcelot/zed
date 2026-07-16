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

- [ ] Bug #40 — command-mode up/down with the viewport mid-notebook: the
      viewport must only move when the target cell is not already fully
      visible, in BOTH directions (upward moves used to bottom-pin an
      already-visible cell; root cause was in the list widget, so re-test a
      few heights/positions including tall cells at the top of the view).
- [ ] Bug #34 — soak test (no direct repro known): create/save/reopen
      notebooks normally over a few sessions; the same file should never
      end up open in two tabs again. (Cause found by inspection: stale
      entry id after save defeated the already-open dedup.)

## Phase 42 — Kernel environment validation (stale-env handling)

Kind: change to existing behaviour — if a check misses, say so and it gets
fixed in place.

- [ ] Delete the selected venv while Zed runs (notebook closed so the kernel
      is dead): reopening + running does NOT attempt a launch/error — the
      stale selection is dropped (top-right shows "Select Kernel") and the
      kernel picker opens.
- [ ] The picker shows NO ghost entry for the deleted env (registered
      kernelspecs whose interpreter vanished are pruned on open; deleted
      workspace .venvs disappear after the re-discovery lands).
- [ ] Recreate the same `.venv` (same name/place) → select/run works on the
      FIRST try (discovery hands back the new interpreter, not the cached
      dead one).
- [ ] Clean bug #30-persistence retest (the 2026-07-16 attempt was confounded
      by the deleted env): two notebooks, different kernels, RUN + SAVE both,
      restart Zed → each notebook's first run uses its own saved kernel
      without prompting.

## Phase 40 — UI polish from the 2026-07-16 testing round

Kind: change to existing behaviour — if either change didn't take effect,
say so and it gets fixed in place (not archived-and-refiled).

- [ ] Output blocks span the CELL'S FULL WIDTH — round 2 (first attempt
      FAILED for tables, 2026-07-16: the table's wrapper claimed no width,
      so the table shrank to its natural size and the copy buttons hugged
      it). Now: a DataFrame's output box reaches the right edge (controls at
      the far right, like text outputs), wide tables scroll inside it, and
      small tables still don't stretch.
- [ ] The top kernel strip is slimmer (reduced padding); the kernel
      cluster still sits top-right and everything still works.

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
