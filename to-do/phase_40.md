# Phase 40 — UI polish from the 2026-07-16 testing round

Kind: **change to existing behaviour** (both items are user-requested tweaks
to shipped UI — keep each open until the user confirms the change took
effect). Added as a 6th phase at the user's explicit request ("start on any
actionable items from the above").

Primary files: `crates/repl/src/notebook/notebook_ui.rs` (top strip),
`crates/repl/src/notebook/cell.rs` / `crates/repl/src/outputs.rs` (output
block width).

## Tasks

- [ ] Output block spans the full editor width (user 2026-07-16): the output
      box currently takes ~50% of the usable space even when its content
      (e.g. a wide DataFrame) is being truncated. Make the output container
      stretch to the cell/editor content width so tables get the room before
      falling back to horizontal scroll. (Phase 29 made the TABLE fill the
      BOX — this widens the box itself.)
- [ ] Slim down the notebook top strip (user 2026-07-16): reduce the
      padding/margin around the top-right kernel indicator bar from phase 30 —
      it takes more vertical space than it needs.

## Verification

- clippy clean; `cargo test -p repl` passes.
- User test: wide DataFrame output box reaches the editor's full content
  width; top strip is visibly slimmer with the kernel cluster intact.
