# Phase 14 — Notebook data safety: save-conflict guard + reload affordance

Kind: **mixed** — a bug fix (bug #14, save overwrites external changes) plus a
new feature (a Reload command/button). Not yet started — this is a plan.

Goal: never silently lose on-disk changes, and give the user a one-click way to
pull external changes in. Phase 9 already handles the READ side (a toast when
the file changes on disk under unsaved changes); this closes the SAVE side and
adds the reload affordance the toast currently only describes in words.

Primary files: `crates/repl/src/notebook/notebook_ui.rs` (`Item::save`,
`Item::reload`, the conflict toast, the retained `buffer`).

## Tasks

- [ ] Save-time conflict check (bug #14): before `fs.atomic_write`, compare the
      on-disk content (or the retained buffer's `has_conflict` / mtime) against
      what we loaded. If it changed under us, prompt: Overwrite / Cancel
      (/ optionally Reload-and-lose-local). Only write on explicit confirm.
- [ ] "Reload Notebook" command-palette action + a Reload button on the
      conflict toast, both calling the existing `Item::reload`
      (`reload_cells_from_notebook`, wired in phase 9).
- [ ] After a save that the user confirmed as an overwrite, clear the conflict
      state so the toast doesn't linger.

## Risks / gaps

- Data-loss risk is the whole point — default the confirm dialog to the
  non-destructive choice (Cancel), never auto-overwrite.
- Avoid a spurious conflict on our OWN save (phase 9 already skips reload when
  disk content matches ours — reuse that comparison here).
- Requires a confirm dialog; check for an existing workspace prompt helper
  before building one.

## Verification

- `cargo check -p repl` + `./script/clippy -p repl` clean; unit-test the
  conflict-detection comparison.
- User test: edit the file externally, then save from Zed → prompted, not
  silently overwritten; Reload command pulls the external version.
