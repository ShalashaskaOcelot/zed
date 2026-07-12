# Phase 14 — Notebook data safety: save-conflict guard + reload affordance

> ⚠️ STATUS: IMPLEMENTED, AWAITING USER TESTING. Compiles, clippy-clean, tests
> pass.
> Kind: **mixed** — a bug fix (bug #14: keep OPEN until the user confirms the
> save prompt actually appears) plus a new feature (Reload command/button:
> archive once confirmed present and working).

Goal: never silently lose on-disk changes, and give the user a one-click way to
pull external changes in. Phase 9 already handles the READ side (a toast when
the file changes on disk under unsaved changes); this closes the SAVE side and
adds the reload affordance the toast currently only describes in words.

Primary files: `crates/repl/src/notebook/notebook_ui.rs` (`Item::save`,
`Item::reload`, the conflict toast, the retained `buffer`).

## Implemented

- [x] Save-time conflict check (bug #14): a `disk_changed_externally` flag is
      set when the file changes on disk under unsaved changes (the phase-9
      conflict path). While set, `Item::save` prompts (Overwrite / Cancel) via
      `window.prompt` and only writes on explicit Overwrite; Cancel leaves the
      notebook dirty and the disk untouched.
- [x] "Reload Notebook" command (`notebook::ReloadNotebook`, in the command
      palette and the More options menu) + a "Reload (discard my changes)"
      button on the conflict toast, both calling the existing `Item::reload`.
      The command prompts first when there are unsaved changes.
- [x] Conflict state clears on reload (`reload_cells_from_notebook`) and on a
      confirmed overwrite save.

## Risks / gaps

- Data-loss risk is the whole point — default the confirm dialog to the
  non-destructive choice (Cancel), never auto-overwrite.
- Avoid a spurious conflict on our OWN save (phase 9 already skips reload when
  disk content matches ours — reuse that comparison here).
- Requires a confirm dialog; check for an existing workspace prompt helper
  before building one.

## Manual test checklist (for the user)

- [ ] Make a change in Zed (don't save), edit the .ipynb externally → conflict
      toast appears with a Reload button; clicking it loads the disk version.
- [ ] Same setup, then Ctrl-S in Zed → an Overwrite/Cancel prompt appears;
      Cancel leaves the disk file untouched (still dirty in Zed); Overwrite
      writes your version.
- [ ] "Reload Notebook" from the command palette / More options: with unsaved
      changes it prompts first; without, it reloads immediately.
- [ ] Normal saves (no external change) do NOT prompt.

## Verification (automated)

- `cargo check -p repl` + `./script/clippy -p repl` clean; tests pass.
