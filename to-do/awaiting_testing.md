# Awaiting user testing

Outstanding MANUAL TEST items from phases whose implementation is complete
(the phase files themselves have been recorded in `CHANGELOG.md` and removed).
Nothing here needs implementation — these are tasks for the user to verify at
runtime.

Workflow: tick an item the moment the user confirms it. If a test FAILS, file
it in `bugs.md` (or as a backlog/phase item per the kind rules), annotate the
line with the bug number, and tick it here (the follow-up is tracked
elsewhere). Delete a section once all its boxes are ticked.

## Phases awaiting confirmation (detail lives in the phase file)

These are implementation-complete and marked `⚠️ AWAITING USER TESTING` in
`to-do/`; their test steps are in the phase file itself. Listed here so this
stays the one place to see everything needing a look.

- [ ] Phase 56 — notebook cell-list scrollbar (user says still needs work)

## Bug fixes awaiting confirmation

Pointer list so there's ONE place to see everything needing a test. The full
detail (symptom, analysis, fix) lives in `bugs.md`; these stay at
`fix attempted - untested` there until confirmed. When the user confirms: tick
here, add a one-line entry to `CHANGELOG.md`, and delete the bug's `bugs.md`
entry. If a fix failed, leave the bug open with the new finding and keep it
listed here.

- [ ] Bug #15 — Soak test: use command-mode shortcuts across focus changes
      (click away to another pane/panel and back, close and reopen a notebook).
      Shortcuts should never stop responding. The on_focus part was already
      reported working 2026-07-09; this is the remaining desync watch.
- [ ] Bug #64 — FIX FAILED 2026-08-11 (`f8`/`shift-f8` still don't move between
      failed cells even though the strip counts them). No new fix yet: the next
      step is establishing whether the action dispatches at all. Low priority —
      two failed cells only happen if you make them happen.

- [ ] Bug #69 — SECOND fix (the first was wrong). Display a DataFrame with
      column titles long enough to have wrapped before (`Column_number_10` and
      up): no title wraps its last character, at any table width. The table's
      text may look very slightly different in size — it now renders at the
      buffer font size, which is what its column widths were always measured
      against.
- [ ] Bug #70 — A table with a long text column (e.g. paths): every column's
      left edge is a straight vertical line down the whole table, regardless of
      how long individual values are.

- [ ] Bug #74 — Dirty a notebook, quit Zed, delete the file from disk, reopen:
      the notebook comes back (with the unsaved changes it was holding) instead
      of the tab silently disappearing. It will be blank-plus-your-changes,
      since the file itself is gone.
- [ ] Bug #75 — Create a new notebook, type something, quit Zed, reopen: it
      comes back WITH the dirty marker, not looking saved. Editing and undoing
      back to the restored state should leave it dirty throughout.
- [ ] Bug #34 — soak test (no direct repro known): create/save/reopen
      notebooks normally over a few sessions; the same file should never
      end up open in two tabs again. (Cause found by inspection: stale
      entry id after save defeated the already-open dedup.)
- [ ] Bug #50 — Open a workspace folder, then (a) also have another root that
      you delete, or (b) create an UNSAVED notebook/buffer and then delete the
      whole workspace folder. Quit and relaunch → the session should restore:
      surviving roots come back, and unsaved items are recovered even if their
      folder is gone (no more whole-session loss over a deleted path).

## Phase 44 — Build the user-level Windows installer locally

Kind: new feature (fork release engineering). Implementation (the three
`script/bundle-windows.ps1` fixes + `docs/fork/windows-installer-build.md`)
is done; these tests need the user's Windows machine. If a build step fails,
file it in `bugs.md`, annotate the failing line, and tick it here.

- [ ] Follow `docs/fork/windows-installer-build.md` from scratch: the bundle
      completes and produces `target/Zed-x86_64.exe`. Note wall-clock time and
      disk used (feeds phase 47's runner sizing).
- [ ] Run the installer WITHOUT admin rights: no UAC prompt; installs under
      `%LOCALAPPDATA%\Programs\Zed Dev`; Start-menu entry, optional desktop
      icon, and `zed` on the user PATH all work; SmartScreen's unsigned-installer
      warning is the expected cost (note what it looks like).
- [ ] The installed fork build opens and runs a Jupyter notebook end-to-end;
      uninstall from per-user Apps & Features cleans up.

## Phase 48 — Select a newly-created kernel immediately

Kind: change to existing behaviour — if the new behaviour didn't take effect,
say so and it gets fixed in place (not archived-and-refiled).

- [x] With kernel A already selected and running, Create Env (venv or conda):
      the NEW env shows as the notebook's selected kernel immediately (top
      strip, "Starting"), NOT kernel A. Run a cell during the build → it goes
      Pending and kernel A does NOT start/run it; once the env is ready the
      new kernel launches and the held cell(s) run on it. (This is the
      behaviour that was missing — previously it fell back to A.)
- [ ] Build FAILURE (e.g. a conda name that can't solve) → the selection
      reverts (no longer shows the half-made env) and any cell held during the
      build returns to Idle (not stuck Pending); the failure toast still shows.
- [x] Not regressed: creating from the "Select Kernel" (no-kernel) state still
      works as before; and picking a DIFFERENT kernel from the picker WHILE an
      env is building switches to that kernel (the building env no longer
      auto-steals the selection when it finishes).

## Phase 39 — Conda environment creation from the kernel picker

Kind: new feature — once confirmed present and basically working, refinements
and defects become new backlog/bug items.

- [ ] On a machine WITH conda (or mamba/micromamba) on PATH: the kernel
      picker's "Create Python Environment" prompt now shows a "Create Conda
      Env…" button. Choosing it opens a name modal; entering a name creates
      the env (`conda create -y -n <name> python ipykernel`), shows a
      progress toast, then selects it and runs the queued cell. The kernel
      persists across a restart (registered kernelspec).
- [ ] The venv fast path is unchanged: Enter still creates the workspace
      `.venv`; "Choose Location…" still works.
- [ ] On a machine WITHOUT conda on PATH: no "Create Conda Env…" button
      appears (only the venv options).
- [ ] Error surfacing: give the name of an env that can't solve, or a bad
      name — the failure toast shows conda's error rather than failing
      silently. (Also: no project folder open + conda present → the conda
      option still appears and works.)

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

## Phase 66 — Notebook file surfaces: save-as and global search

Kind: **mixed** — item 2 is a defect fix (say so if a search hit still opens
JSON, it gets fixed in place), item 1 is a change to save-as behaviour.

- [ ] Save-as an untitled notebook and type a name with NO extension: the file
      is written as `<name>.ipynb`, the tab shows it, and reopening it from the
      file tree gives a notebook (not JSON).
- [ ] Save-as OUTSIDE any project folder (somewhere in your home directory)
      with a bare name — this is the path that needed the fix to happen before
      the worktree is created, so it is the one most worth trying.
- [ ] Type a name that already ends in `.ipynb`: it is used exactly as typed,
      no second extension.
- [ ] Ctrl-Shift-F for text that lives in a notebook cell, click the result:
      the notebook opens in the notebook editor. (The jump to the matching line
      is deliberately not carried over — it opens the notebook, not the cell.)
- [ ] Ctrl-Shift-F for text in a normal file: unchanged — opens the editor at
      the matching line, including the split (`ctrl-enter`) variant.
- [ ] A notebook already open in a tab doesn't get a second tab from a search
      hit; the existing tab activates.

## Phase 69 — Notebooks behave like text files on disk

Kind: **mixed**. Tested 2026-08-11 — most of it works; the failures became bugs
#74, #75 and #76 and are tracked there.

- [ ] Loose notebook restores after a restart. NOT yet tested — the restore
      checks below were all done on a notebook inside a project folder, which
      always worked. This one needs a notebook opened from OUTSIDE every project
      root (no folder open at all, or a notebook somewhere unrelated): quit with
      it open, reopen, the tab should come back.
- [x] Deleting the file in Explorer strikes the tab title through. CONFIRMED
      2026-08-11.
- [ ] With `"close_on_file_delete": true`, the same deletion closes the notebook
      tab when it has no unsaved changes; with unsaved changes it stays open and
      closing it prompts. (Not reported on — the setting is off by default, so
      it needs turning on first.)
- [x] Unsaved changes to a saved notebook survive a quit. CONFIRMED 2026-08-11.
- [x] Saving then quitting gives back the SAVED version, not the older unsaved
      copy. CONFIRMED 2026-08-11 — this was the stale-content trap.
- [x] Conflict when the file was edited elsewhere. FAILED 2026-08-11 — the
      cached version opened with no notification at all. Filed as bug #76; the
      flag may well be set and simply invisible until you save, which is the
      first thing to check there.
- [ ] A large notebook with image outputs stays responsive while typing. Still
      untested — the DataFrames used so far don't exercise it. What matters is a
      notebook whose OUTPUTS are heavy (`df.plot()` or any matplotlib figure,
      several of them, so the file is megabytes of base64), then typing in a
      cell with unsaved changes pending. The encode now runs for every dirty
      notebook, not just untitled ones.
- [x] Not regressed: untitled notebooks still restore with their cells.
      CONFIRMED 2026-08-11 — though they came back looking clean, which is bug
      #75.
- [x] A blank untitled notebook does NOT restore. Confirmed 2026-08-11 and
      explicitly accepted by the user as not an issue: nothing is stored for it
      because there is nothing worth storing.

Known and deliberate (user decision 2026-08-11): a notebook deleted BETWEEN
sessions restores blank with no strikethrough, because generic session restore
never marks a missing file as deleted. `.md` behaves the same way. Note this is
DIFFERENT from bug #74, where the tab didn't come back at all.

## Phase 67 — Page-wise follow mode

Kind: **new feature** (a SECOND follow mode; the existing one is untouched and
still the default). Once it is confirmed present and basically working, any
refinement becomes a new backlog/bug item rather than reopening this.

Turn it on with `"repl": { "notebook_follow_mode": "page" }` in settings (or
Settings → REPL & Notebooks → Notebook → Follow Mode), then switch
"Follow Running Cell" on in the notebook's control sidebar. Every test below
needs BOTH.

- [ ] With the setting left at its default (`minimal`) follow mode behaves
      exactly as before — each running cell re-pinned near the top.
- [ ] In `page` mode with a notebook that fits on screen: Run All never moves
      the viewport at all, and the selection highlight walks down the cells as
      each one runs.
- [ ] With a longer notebook: the viewport holds still while execution walks
      down the cells that are on screen, then jumps a whole page when execution
      passes the fold — the newly-reached cell becomes the TOP of the viewport,
      and it holds still again for that page.
- [ ] The last jump stops with the final cell visible instead of scrolling the
      tail of the notebook up into blank space.
- [ ] A cell taller than the viewport pins its top and overflows (no attempt to
      fit it), and execution carries on normally afterwards.
- [ ] Clicking into a cell to edit during a run turns Follow Running Cell OFF —
      the sidebar toggle visibly flips — and the selection stops moving, so
      your cursor is left alone. Turning it back on jumps to the running cell.
- [ ] Worth a look while testing (the user's own prediction, phase 67 design):
      because each running cell sits at or above the fold with room below it,
      its status footer and the start of its output should usually be on
      screen — except for the last cell on a page and cells taller than the
      viewport. If that holds, the deferred "anchor on the status footer"
      backlog item is largely covered; if it doesn't, say so.

## Phase 64 — Kernel picker and notebook control polish

Kind: mixed — mostly changes to existing behaviour, so if one of these still
behaves as before, say so and it gets fixed in place rather than refiled.

- [x] Launch the app and open the kernel picker immediately: it says
      "Searching for kernels…" and then fills in, instead of "No matches".
      CONFIRMED 2026-08-06.
- [x] Registered Jupyter kernelspec entries now show the INTERPRETER they
      launch (`argv[0]`) as their second line. CONFIRMED 2026-08-06.
- [x] The right sidebar no longer has a kernel selector at the bottom.
      CONFIRMED 2026-08-06.
- [ ] Open a notebook from the project panel WITHOUT clicking into it, press
      Run All in the sidebar, then use a keyboard shortcut (`escape`, arrows,
      `shift-enter`) — it now acts on the notebook, because the button moved
      focus there. Clicking a control WHILE editing a cell should NOT throw you
      out of the cell.
- [x] Creating an env still works and the "Creating…" row looks tidier.
      CONFIRMED 2026-08-06 — and the user re-confirmed it does NOT refresh when
      the build finishes (still bug #58), and asked for a "creating" status
      outside the picker too, since the strip shows a misleading "Starting"
      (backlogged, not a defect in this phase).

## Phase 63 — Wide notebook outputs and output-body interaction

Kind: mixed — the wide-output part is a change to existing behaviour (say so if
it still clips), the click-to-select part is a new affordance.

- [x] Display a pandas DataFrame far wider than the pane: it scrolls sideways
      within its output block. PARTIALLY CONFIRMED 2026-08-06 — the sideways
      scrolling itself works, but the scrollbar was unusable (bug #68) and the
      table's own rendering had two defects the wider view exposed: column
      titles wrapping (bug #69) and columns not lining up row to row (bug #70).
- [x] Scrolling the wheel over a wide table still scrolls the NOTEBOOK
      vertically. FAILED 2026-08-06 — it scrolled the notebook AND dragged the
      table sideways at the same time. Filed as bug #67.
- [x] Clicking anywhere on an output selects that cell, and dragging across
      output text still selects the text. CONFIRMED 2026-08-06. (The wording
      here was ambiguous: selecting the cell as well as the text on a drag is
      the intended behaviour, and that is what happens.)
- [x] Narrow tables and ordinary text output are unchanged. CONFIRMED
      2026-08-06 — except that a table small enough to fit still wrapped its
      column titles, which is bug #69, not a width regression.

