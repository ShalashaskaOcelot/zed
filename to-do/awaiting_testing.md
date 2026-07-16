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

- [ ] Bug #34 — soak test (no direct repro known): create/save/reopen
      notebooks normally over a few sessions; the same file should never
      end up open in two tabs again. (Cause found by inspection: stale
      entry id after save defeated the already-open dedup.)
- [ ] Bug #35 — run a notebook cell, CLOSE the notebook tab → the kernel
      process (python.exe) disappears from Task Manager and the venv can be
      deleted without "file in use". (Also check the standalone `.py` REPL:
      close the editor → its kernel dies too.)
- [ ] Bug #36 — create a venv at a custom location (outside the project) via
      the kernel picker, restart Zed → it now shows in the picker as
      "Python (<name>)" and runs cells. (Your existing `test_venv` was
      created before the fix — recreate it via the picker, or run
      `<env>\Scripts\python.exe -m ipykernel install --user --name test_venv`
      once by hand.)
- [ ] Bug #38 — remove ipykernel from the env, run a cell → the cell status
      is the red ✕ failed state (not "Cancelled"), with the kernel stderr in
      the error output.
- [ ] Bug #37 — a DataFrame output now shows BOTH hover controls: Copy Output
      and Open in Buffer (read-only buffer with the markdown table); and the
      output "…" menu → Copy Output now copies table output (as markdown)
      instead of nothing.
- [ ] Bug #31 — BLOCKED by bug #35 (closing a notebook leaves the kernel
      running, so the env can't be deleted while Zed is up). Once #35 is
      fixed: delete the kernel env while Zed stays running (notebook closed);
      reopen + run → the launch errors once, and the NEXT run opens the
      kernel picker instead of repeating the error forever.

- [ ] Bug #30 — retest in two parts (2026-07-16 restart finding recorded in
      bugs.md — the blank indicator on launch is likely just lazy start):
      (1) SAME SESSION: two notebooks open, pick kernel A in one and kernel B
      in the other → each picker checkmark shows its own; running each uses
      its own kernel. (2) PERSISTENCE: pick a kernel, RUN a cell (launch
      writes the metadata), SAVE the notebook, restart Zed, reopen → the
      indicator may stay grey until you run, but the first run should use the
      saved kernel WITHOUT prompting.

## Bug #39 — one investigation check (not a fix confirmation)

- [ ] Run a cell in VS Code, SAVE there, then open the .ipynb in a TEXT
      editor: does that cell's `"metadata": { "execution": { … } }` contain
      the new run's timestamps ("shell.execute_reply" etc.), the old ones, or
      nothing? Zed's format already matches VS Code's exactly, so this
      answer determines whether there's anything to fix on Zed's side (see
      bugs.md #39).

## Phase 33 — Truly unsaved "New Jupyter Notebook"

Kind: change to existing behaviour — if the new flow misses the mark, say so
and it gets fixed in place (not archived-and-refiled).

- [ ] Command palette → "New Jupyter Notebook": tab opens as "Untitled", NO
      `Untitled-N.ipynb` appears on disk. Edit a cell → dirty dot appears.
- [ ] Ctrl-S (or closing and choosing Save) on the untitled notebook opens
      the save-path prompt (suggested name `Untitled.ipynb`); after saving,
      the tab shows the chosen name, the file exists where chosen, and
      further saves go straight to it (no prompt).
- [ ] After that first save the notebook behaves like any opened one:
      external-change reload/conflict toast works, the kernel picker
      remembers the pick for that file.
- [ ] File browser → New File `some.ipynb` still creates and opens the real
      file immediately (unchanged behaviour).
- [ ] Untitled notebooks are NOT restored after restarting Zed (accepted v1
      behaviour — unsaved means gone; confirm nothing crashes on restart).

## Phase 40 — UI polish from the 2026-07-16 testing round

Kind: change to existing behaviour — if either change didn't take effect,
say so and it gets fixed in place (not archived-and-refiled).

- [ ] Output blocks now span the CELL'S FULL WIDTH (the `max_columns`
      setting no longer caps the notebook output box — it only sizes the
      inline `.py` REPL). A wide DataFrame gets the whole editor width
      before falling back to horizontal scroll inside the block.
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
