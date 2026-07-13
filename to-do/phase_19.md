# Phase 19 — Configurable post-run landing mode (shift-enter / ctrl-enter)

Kind: **new feature** (a setting) built on top of the phase-11 behaviour. Not
yet started — this is a plan. Requested by the user 2026-07-11.

Goal: make the mode you land in after running a cell configurable, instead of
always dropping to command mode (phase 11). A new Zed `repl` setting with three
values:

- `command` (always): after run, the next/new cell is in COMMAND mode.
- `edit` (always): after run, the next/new cell is in EDIT mode.
- `remember` (last): land in whatever mode you were in when you pressed run —
  editing the cell before running → next cell is in edit mode; command mode →
  stays command mode.

Primary files: `crates/settings_content/src/settings_content.rs`
(`ReplSettingsContent`), `crates/repl/src/repl_settings.rs` (`ReplSettings`),
`crates/repl/src/notebook/notebook_ui.rs` (`run_current_cell` /
`run_and_advance` / `advance_in_command_mode`).

## Tasks

- [ ] Add a `repl` setting, e.g. `notebook_run_landing_mode` with values
      `command` | `edit` | `remember` (default `command`, matching current
      phase-11 behaviour). Wire through `ReplSettingsContent` → `ReplSettings`.
- [ ] Capture the mode at the moment run is triggered (before focus changes),
      so `remember` can restore it.
- [ ] Apply the setting in `run_current_cell` and `run_and_advance`: `command`
      → `enter_command_mode` (current behaviour); `edit` → focus the
      next/new cell's editor; `remember` → reuse the captured mode.
- [ ] Document the setting in `docs/src/repl.md`.

## Risks / gaps

- `remember`/`edit` re-introduce landing in edit mode, which historically
  tangled with the focus/mode desync (bug #15). Make sure the edit-mode path
  focuses the correct cell editor and doesn't desync.
- Keep `command` as the default so existing behaviour is unchanged unless the
  user opts in.

## Verification

- `cargo check -p repl` + `./script/clippy -p repl` clean.
- User test: each of the three settings behaves as described for both
  shift-enter (advance) and ctrl-enter (stay).
