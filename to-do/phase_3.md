# Phase 3 — Navigation and scrolling

Goal: nav-mode cell navigation and edit-mode cursor movement keep the right
thing in view.

Primary files: `crates/repl/src/notebook/notebook_ui.rs`,
`crates/gpui/src/elements/list.rs` (scroll primitive),
`assets/keymaps/default-{linux,macos,windows}.json`.

## Tasks

- [ ] Fit-aware cell reveal: when selecting a cell via arrow keys (or any
      `jump_to_cell`), if the cell does NOT fully fit in the viewport, align
      its TOP edge to the top of the view regardless of travel direction; if
      it fits, minimally scroll to fully reveal it (current nearest-edge
      behaviour is fine for that case). Implement as a new/extended
      `ListState` reveal method — the current
      `scroll_to_reveal_item` (`list.rs:626-656`) reveals the nearest edge
      with no height-vs-viewport comparison. Primitives available:
      `bounds_for_item`, `logical_scroll_top`, `last_layout_bounds`.
- [ ] View follows cursor in edit mode: when the cursor moves within a cell
      editor, scroll the notebook `ListState` just enough to keep the cursor
      line in view. Only scroll when the cursor would leave the viewport —
      do NOT keep it centered. Requires wiring cell-editor cursor/selection
      events to a notebook scroll computation (cell editors are
      `SizeByContent` with no internal scroll, so the outer list must do it).
- [ ] Bind Home/End in nav mode to first/last cell — handlers
      `select_first` / `select_last` already exist (`notebook_ui.rs:954,
      967`), they are just unbound.
- [ ] Review `NotebookMoveUp` / `NotebookMoveDown` (implemented, unbound
      "smart arrows" that cross cell boundaries from first/last line,
      `notebook_ui.rs:1395, 1431`): decide whether to bind them for edit-mode
      up/down so cursor travel flows between cells, and ensure they respect
      the two scroll behaviours above.
- [ ] Manual test checklist: tall cell (taller than viewport) reached from
      above and below; last cell in notebook (fits → sits naturally at
      bottom; doesn't fit → top-aligned); edit-mode cursor travel through a
      tall cell in both directions.

## Notes

- Changes to `crates/gpui` are shared infrastructure — keep the new reveal
  behaviour opt-in (new method or parameter) so other `list()` users are
  unaffected.
