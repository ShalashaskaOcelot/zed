# Phase 22 — Multi-select cells

> ⚠️ STATUS: IMPLEMENTED, AWAITING USER TESTING. Compiles, clippy-clean, tests
> pass. Kind: **new feature** — on confirmation the gestures and multi-cell
> actions work, archive; tweaks become new items.
> User decisions (2026-07-12): single-target actions act on the PRIMARY cell;
> Move works on CONTIGUOUS selections only (no-op for discontiguous).

Requested by the user 2026-07-11. VS Code / file-explorer / Excel-style
multi-selection.

Goal: select multiple cells and act on them together.

Desired behaviour:
- shift + down/up (command mode): keep the anchor cell selected and extend the
  selection to the adjacent cell (contiguous range grows/shrinks).
- shift + click a cell: select the whole contiguous range from the anchor to
  the clicked cell, inclusive.
- ctrl/cmd + click: toggle an individual cell into a discontiguous selection.
- ctrl/cmd + arrows: do nothing (no selection change).
- Cell actions (delete, copy/cut, run, move, convert) operate on the FULL
  selection.

Primary files: `crates/repl/src/notebook/notebook_ui.rs` (selection state,
navigation handlers, every cell action), `cell.rs` (selected rendering),
`assets/keymaps/*` (shift+arrow bindings).

## Implemented

- [x] Selection state: `selected_indices: BTreeSet<usize>` (the full selection
      incl. the primary when multi) + `selection_anchor`, alongside the
      existing `selected_cell_index` primary. Index-based, so every structural
      change (insert/remove/move/replace/reload) collapses the selection.
- [x] shift+down/up (`notebook::ExtendSelectionDown/Up`, bound in all three
      keymaps) extends the contiguous range from the anchor; plain arrows
      collapse to a single selection.
- [x] shift+click selects anchor→clicked range; ctrl/cmd+click toggles a cell
      in a discontiguous selection; ctrl+down/up are explicit no-ops (bound to
      null). Clicks are intercepted in the CAPTURE phase on the cell root
      (code, markdown edit+preview, raw) so a modified click selects without
      focusing the cell's editor (`CellEvent::ModifiedClick`).
- [x] All selected cells render with the selection treatment (accent bar).
- [x] Actions over the full selection: DELETE (bottom-up, guard keeps ≥1
      cell), COPY/CUT (multi copies serialize as a JSON array; paste accepts
      both the single-cell and array formats), RUN (batch over the selection,
      in order), MOVE up/down (contiguous block only, selection follows the
      block), CONVERT to code/markdown (selection preserved), CLEAR CELL
      OUTPUTS. Single-target actions (add above/below, paste, duplicate,
      Enter/edit) act on the PRIMARY cell per the user's decision.
- [x] Undo/redo: new `CellEdit::Group` — a multi-cell delete/move/convert/
      paste is one logical operation (undo replays members in reverse).

## Scope notes

- Raw cells get the modified-click wiring on initial load but not after an
  external reload (they're rare; the reload path has no raw-cell wiring hook).
- Multi-selection is session-visual state; it does not persist.

## Manual test checklist (for the user)

- [ ] shift+down/up in command mode grows/shrinks a contiguous selection;
      plain up/down collapses it.
- [ ] shift+click selects the range to the clicked cell; ctrl+click toggles
      individual cells; ctrl+up/down do nothing.
- [ ] Delete/cut with a multi-selection removes all selected (one undo
      restores them all); copy+paste reproduces all selected cells.
- [ ] Run with a multi-selection executes the selected cells in order.
- [ ] Move up/down shifts a contiguous selected block (and does nothing for a
      discontiguous selection); convert converts all selected.
- [ ] Add cell / paste / duplicate / Enter still act on the primary cell.

## Verification (automated)

- `cargo check -p repl` + clippy clean; 42 repl tests + settings tests pass.
