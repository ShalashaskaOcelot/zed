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

- [ ] Bug #32 — open a notebook → NO dirty dot / no save prompt on close.
      Then: Restart Kernel on a never-started notebook → still not dirty.
      (Restarting a RUNNING kernel still dirties by design — it resets the
      cells' [N] execution numbers, which are savable state.)
- [ ] Bug #33 — clear a ran cell's output (single, selection, and Clear All) →
      the [N] number and ✓/✕ status disappear along with the output.

- [ ] Bug #31 — delete the kernel env while Zed stays running (notebook
      closed); reopen + run → the launch errors once, and the NEXT run opens
      the kernel picker instead of repeating the error forever.

(Bug #30's fix is in progress and will get a pointer line when it lands.)

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

## Phase 26 — More multi-select gestures

- [x] In command mode: `ctrl-a`/`cmd-a` selects ALL cells. CONFIRMED 2026-07-14.
- [x] In command mode: `shift-home`/`shift-end` select to the first/last cell.
      CONFIRMED 2026-07-14.
- [ ] In EDIT mode all three keep their text meanings inside the cell editor:
      ctrl/cmd-a selects the cell's text, shift-home/end select to line
      start/end.

## Phase 5 — Create Python environments from the kernel picker

- [ ] No Python on PATH: clicking "Create Python Environment" shows a clear
      error toast (instead of failing silently). (Needs a machine/session
      where `python3`/`python` isn't on PATH.)


