# Phase 26 — More multi-select gestures

Kind: **new feature**. Not yet started — this is a plan. Promoted from the
backlog (user 2026-07-14), extends the phase-22 multi-selection model.

Primary files: `crates/repl/src/notebook/notebook_ui.rs` (selection handlers),
`crates/zed_actions/src/lib.rs` (new actions), `assets/keymaps/*`.

## Tasks

- [ ] `ctrl/cmd-a` — select ALL cells as one contiguous selection (anchor at
      the first cell, primary at the last). New action `SelectAllCells`.
- [ ] `shift-home` — extend the contiguous selection from the current cell up
      to the FIRST cell; `shift-end` — from the current cell down to the LAST
      cell. New actions `ExtendSelectionToStart` / `ExtendSelectionToEnd`,
      reusing `select_range`.
- [ ] Register the actions and bind them in all three keymaps
      (`default-{linux,macos,windows}.json`) ONLY in the command-mode notebook
      context.

## Risks / gaps

- **Command-mode only (critical).** In edit mode these keys have text meaning
  inside the cell editor — `ctrl-a` = select-all-text, `shift-home`/`shift-end`
  = select-to-line-start/end. They must NOT be bound in the editor context;
  bind only where the notebook is in command mode so the editor keeps them.
- Mouse border-clicks already drop to command mode, so mouse range-select is
  unaffected.

## Verification

- `cargo check -p repl` + clippy clean; tests pass.
- User test: in command mode, ctrl-a selects all; shift-home/shift-end select
  to the first/last cell; in edit mode all three still do their text-selection
  jobs inside the cell.
