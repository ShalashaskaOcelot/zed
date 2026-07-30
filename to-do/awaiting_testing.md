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

- [ ] Bug #34 — soak test (no direct repro known): create/save/reopen
      notebooks normally over a few sessions; the same file should never
      end up open in two tabs again. (Cause found by inspection: stale
      entry id after save defeated the already-open dedup.)
- [ ] Bug #56 — Open/save a notebook OUTSIDE the workspace (e.g. on the
      Desktop) and start a kernel → it should launch instead of failing with
      "The directory name is invalid. (os error 267)". The kernel's working
      directory is now the file's folder. In-project notebooks unchanged.
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
