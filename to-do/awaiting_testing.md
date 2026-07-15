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

- [ ] Bug #30 — two notebooks with DIFFERENT saved kernels: each shows and
      runs its own; picking a different kernel in one leaves the other's
      kernel unchanged (check the picker checkmark shows the current
      notebook's kernel too).

## Phase 34 — Notebook & kernel configuration

- [ ] Autostart is OFF by default: open a notebook with a remembered kernel →
      no kernel starts until the first run. Set
      `"repl": { "notebook_autostart_kernel": true }` → reopening that
      notebook starts its kernel immediately; a notebook with NO remembered
      kernel still doesn't start anything.
- [ ] Kernel picker → "Create Python Environment" now asks where: "Create
      .venv" (fast path, same as before → workspace `.venv`) or "Choose
      Location…" (directory picker; the chosen folder becomes the env, named
      after the folder). Both produce a working kernel and continue any queued
      run.
- [ ] Settings UI (`zed: open settings`) has a new "REPL & Notebooks" page:
      Notebook section (Run Landing Mode dropdown, Show Last Executed Time,
      Autostart Kernel) + REPL Output section (max lines/columns, inline
      output, inline max length, output max height). Changing a value writes
      it to settings.json and takes effect.

## Phase 36 — Notebook code health & stubs

- [ ] "notebook: open notebook" from the command palette now opens the
      workspace file-open dialog (was a no-op that printed to stdout).
- [ ] General smoke test: no notebook feature regressed after the dead-code
      sweep (nothing user-visible should have changed besides the above).

## Phase 29 — Output interaction polish

- [ ] Each notebook output now shows small controls on hover/right edge:
      Copy Output and "Open in Buffer" (a read-only editor tab — this is also
      how to SELECT part of a long output for now).
- [ ] The `docs_df` DataFrame from the screenshot: path-style long columns no
      longer wrap/truncate mid-cell; the table fills the output box width,
      giving the long columns the extra room; horizontal scroll only when the
      natural width truly exceeds the box.
- [ ] A small table (short headers, small numbers) does NOT stretch — it
      keeps its compact natural width.

## Phase 31 — Kernel launch robustness

- [ ] Kernels start reliably from a fresh app start (no os error 10054); fast
      machines launch with no artificial delay, slow ones no longer get cut
      off at 500ms. (Windows is the machine that mattered for bug #6.)
- [ ] Break a kernel deliberately (e.g. remove ipykernel from its env) → the
      cell error shows the kernel's own stderr, both when it dies at startup
      and when it dies after connecting.
- [ ] A kernel that hangs at startup gives "did not answer its heartbeat
      within 30s" instead of waiting forever.

## Phase 32 — Notebook UX niceties

- [ ] Multi-select some cells (shift-down or ctrl-a), press Esc → selection
      collapses to just the primary cell; a second Esc does nothing more.
      Esc from EDIT mode still only returns to command mode (selection kept).
- [ ] Use the output "…" menu (Copy/Collapse/Clear Output) → afterwards the
      cell is selected in command mode and single-key shortcuts (a/b/dd/…)
      work immediately, without clicking a cell first.
- [ ] Edit a markdown cell, press Esc (or click away) → the rendered preview
      comes back (this was already wired via editor blur — confirm it holds).

## Phase 30 — Notebook chrome rework (drop the bottom kernel bar)

Kind: change to existing behaviour — if the new layout misses the mark, say
so and it gets fixed in place (not archived-and-refiled).

- [ ] The bottom kernel bar is GONE; cells start slightly lower, below a slim
      strip whose top-right corner holds the kernel status icon + name.
- [ ] Clicking the top-right kernel cluster opens the kernel picker (and the
      run-with-no-kernel prompt flow still opens it too).
- [ ] Restart Kernel and Interrupt Kernel now live in the right sidebar
      (bottom cluster, above the kernel indicator); interrupt is disabled when
      no kernel is connected.
- [ ] Nothing that lived on the bottom bar is unreachable; re-check bug #25's
      add-cell-at-bottom now that the bar is gone.

## Phase 28 — Per-cell "last executed" timestamp

- [ ] Run a cell → a timestamp (e.g. `· 14:32:05`) appears after the ✓ and
      duration; failed cells show it after the red ✕ too.
- [ ] Save, close, reopen → previously-run cells show ✓ + duration + time
      restored from the file (no kernel needed).
- [ ] Open a notebook last run in VS CODE → its cells show VS Code's recorded
      times; run a cell in Zed, save, open in VS Code → VS Code shows the time.
- [ ] Clear outputs → the timestamp clears with the rest of the run record.
- [ ] Set `"repl": { "notebook_show_last_executed": false }` → timestamps hide
      (✓ + duration remain).

## Phase 27 — Retain cell output through clipboard & undo

- [ ] Run a cell (text output) and one with a rich output (DataFrame/plot);
      cut → paste: outputs come back with the cells.
- [ ] Delete → undo and cut → undo: outputs restored too.

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


