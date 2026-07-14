# Phase 27 — Retain cell output through clipboard & undo

Kind: **change to existing behaviour** (the deliverable is that output is
retained; keep OPEN until confirmed it actually is). Not yet started — this is
a plan. Promoted from the backlog (user 2026-07-12).

Cutting/deleting a cell and then pasting or undoing restores the cell's SOURCE
and metadata but not its rendered OUTPUT — the output area comes back empty.

Primary files: `crates/repl/src/notebook/notebook_ui.rs` (clipboard payload,
`CellEdit` undo snapshots), `crates/repl/src/notebook/cell.rs` (cell → nbformat
snapshot incl. outputs).

## Tasks

- [ ] Include the cell's outputs (as nbformat `Output`s) in the clipboard cell
      snapshot for copy/cut, and in the undo `CellEdit` for delete/cut. The
      snapshot already carries source + metadata; add outputs.
- [ ] On paste / undo, rebuild the cell WITH its outputs (via the existing
      `convert_outputs` load path) so the restored cell shows its previous
      output — including rich outputs, which now round-trip via the phase-24
      source-media retention (bug #24).
- [ ] Cover all three flows: cut→paste, delete→undo, cut→undo.

## Risks / gaps

- Clipboard payload size grows with rich outputs (base64 images etc.) — fine
  for in-app clipboard; if serialized to the system clipboard, keep it JSON.
- Output round-trip must reuse the same nbformat path as save/load so the
  behaviour matches a reopened notebook (no second serialization format).

## Verification

- `cargo check -p repl` + clippy clean; tests pass.
- User test: run a cell (with output), cut it, paste → output restored; delete
  it, undo → output restored; cut, undo → output restored. Works for text and
  rich (table/plot) outputs.
