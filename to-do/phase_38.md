# Phase 38 — Cell structure operations

Kind: **new feature**. Not yet started — this is a plan. Promoted from the
backlog (low-priority items) to keep 5 phases in rotation after phase 36
completed.

Primary files: `crates/repl/src/notebook/notebook_ui.rs`,
`crates/zed_actions/src/lib.rs`, `assets/keymaps/*`.

## Tasks

- [ ] Split cell: in edit mode, split the current code/markdown cell at the
      cursor into two cells (Jupyter's ctrl-shift-minus), as one undo group.
- [ ] Join cells: merge the selected cell with the one below (or the whole
      contiguous multi-selection) into one cell, sources concatenated,
      outputs cleared, as one undo group.
- [ ] Dedicated "Run all above" / "Run cell and below" buttons in the right
      sidebar (the actions already exist in the More-options menu).

## Risks / gaps

- Split must preserve cell type and metadata sensibly (collapse state stays
  on the top half; execution record cleared on both halves).
- Join across cell TYPES: restrict to same-type neighbors (no-op otherwise).

## Verification

- `cargo check -p repl` + clippy clean; tests pass.
- User test: split at cursor produces two cells (undo re-joins); join merges
  selected cells; the new sidebar buttons run above / cell-and-below.
