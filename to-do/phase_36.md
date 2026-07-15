# Phase 36 — Notebook code health & stubs

Kind: **mixed** (cleanup + two small missing implementations). Not yet
started — this is a plan. Promoted from the backlog (cleanup + low-priority
items) to keep 5 phases in rotation after phase 31 completed.

Primary files: `crates/repl/src/notebook/notebook_ui.rs`.

## Tasks

- [ ] Remove `#![allow(unused, dead_code)]` from `notebook_ui.rs`, delete the
      large commented-out `NotebookControls` block, and fix/remove whatever
      dead code the lints then surface.
- [ ] Implement `open_notebook` (currently a `println!` stub) or remove the
      action if it duplicates the normal open path.
- [ ] Implement `Item::pixel_position_of_cursor` so the workspace can track
      the notebook cursor position.

## Risks / gaps

- Removing the allow may surface a pile of dead code — delete or wire up each
  finding deliberately; no blanket re-allow.

## Verification

- `cargo check -p repl` + clippy clean with the allow gone; tests pass.
