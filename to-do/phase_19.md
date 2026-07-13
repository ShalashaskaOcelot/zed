# Phase 19 — Configurable post-run landing mode (shift-enter / ctrl-enter)

> ⚠️ STATUS: IMPLEMENTED, AWAITING USER TESTING. Compiles, clippy-clean, repl +
> settings tests pass.
> Kind: **new feature** (a setting) — on confirmation the three values behave
> as described, archive; tweaks become new items.

Requested by the user 2026-07-11.

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

## Implemented

- [x] New `repl.notebook_run_landing_mode` setting with values `command` |
      `edit` | `remember` (default `command` — current behaviour unchanged).
      Wired `ReplSettingsContent` → `ReplSettings`, listed in
      `assets/settings/default.json`, documented in `docs/src/repl.md`.
- [x] The mode is captured at the moment run is triggered (before any focus
      change), so `remember` restores it faithfully.
- [x] Applied in both `run_current_cell` (ctrl-enter, stays on the cell) and
      `run_and_advance` (shift-enter, next/new cell) via a shared
      `apply_post_run_landing`: `command` → command mode; `edit` → the
      selected cell's editor gets focus (markdown cells open their editor);
      `remember` → the captured mode.
- [x] Guard: if the kernel picker is open (running with no kernel selected),
      the edit landing falls back to command mode instead of focusing an
      editor — focusing would dismiss the picker and drop the queued cells
      (same guard as phase 17).

## Manual test checklist (for the user)

- [ ] Default (`command`): unchanged — ctrl-enter and shift-enter land in
      command mode.
- [ ] `edit`: both land with the cursor in the (next/new) cell's editor.
- [ ] `remember`: run from edit mode → land editing; run from command mode →
      land in command mode. Shift-enter from edit mode moves to the NEXT cell
      in edit mode.
- [ ] With no kernel selected, shift-enter still opens the picker and all
      queued cells run after choosing (no regression from the edit landing).

## Verification (automated)

- `cargo check` + `./script/clippy` clean for repl and settings_content;
  repl (42) and settings (30) tests pass.
