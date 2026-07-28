# Phase 57 — In-notebook search (Ctrl-F), part 1: find / highlight / navigate

Kind: **new feature** (notebook-primary). Requested by the user 2026-07-23
(promoted from the backlog item "In-notebook search (Ctrl-F)", reported
2026-07-16). Priority: user's main ask this round.

Goal: make Ctrl-F work inside a notebook — type a query, matches highlight
across all cells, Enter / Shift-Enter (and the search bar's next/prev) jump
between them, and activating a match selects+scrolls to that cell. Replace and
advanced options are deferred to a part-2 phase.

## Key facts (code inspection 2026-07-23)

- The Ctrl-F pipeline is generic: the `BufferSearchBar` drives whatever the
  active pane item returns from `Item::as_searchable`. Today
  `NotebookEditor::as_searchable` returns `None` (`notebook_ui.rs:4952`), so
  Ctrl-F does nothing. Returning `Some(Box::new(handle.clone()))` (as `Editor`
  does at `editor/src/items.rs:1040`) turns it on — once `NotebookEditor`
  implements `SearchableItem`.
- `SearchableItem` (`crates/workspace/src/searchable.rs:73`) requires:
  `type Match: Any + Sync + Send + Clone`, and methods `clear_matches`,
  `update_matches`, `query_suggestion`, `activate_match`, `select_matches`,
  `replace`, `find_matches`, `active_match_index` (others have defaults:
  `supported_options`, `replace_all`, `match_index_for_direction`,
  `find_matches_with_token`, `get_matches`). Also requires
  `EventEmitter<SearchEvent>` (`SearchEvent::{MatchesInvalidated,
  ActiveMatchChanged}`) — NotebookEditor currently only emits `()`
  (`notebook_ui.rs:4870`), so add the emitter.
- `Editor` is the reference impl (`editor/src/items.rs:1628`), `type Match =
  Range<Anchor>`, operating on its single buffer:
  - `find_matches`: background search over the buffer → `Vec<Range<Anchor>>`.
  - `update_matches`: sets `HighlightKey::BufferSearchHighlights` background
    highlights, active match a different color.
  - `activate_match`: `change_selections` to the match range with autoscroll.
  - `active_match_index`: the free fn `editor::active_match_index(direction,
    ranges, &cursor_anchor, &buffer_snapshot)` (`items.rs:2051`), a binary
    search of ranges vs the cursor.
  - `query_suggestion`: seeds from the current selection/word.
- A notebook is N independent cell editors, NOT a multibuffer. Each `Cell`
  exposes its source editor via `cell.editor(cx) -> Option<&Entity<Editor>>`
  (`cell.rs:338`); `cell_order: Vec<CellId>` + `cell_map` give ordered access.
  So delegate every primitive to the per-cell `Entity<Editor>`, tagging matches
  with their cell.
- Reveal/scroll for `activate_match` should reuse the index-anchored
  `follow_scroll_to` / `scroll_to_item_near_top` from bug #45 (immune to
  unmeasured/stale heights) after selecting the cell, so jumping to a match in a
  far cell actually lands on it.

## Design

- `type Match = NotebookSearchMatch { cell_id: CellId, range: Range<Anchor> }`
  (or `(CellId, Range<Anchor>)`), ordered globally by (index in `cell_order`,
  then `range.start` within the cell). All fields are `Send + Sync + Clone`.
- Matches are produced and consumed per cell:
  - `find_matches`: for each cell in `cell_order` with an editor, call that
    editor's `find_matches(query)` (each returns a `Task`), collect the tasks,
    then `cx.spawn` to await them in order and concatenate, tagging each range
    with its `cell_id`. Skip cells without an editor.
  - `update_matches` / `clear_matches`: group the matches by `cell_id` and
    delegate to each cell editor's `update_matches` / `clear_matches` (so the
    existing per-editor highlight machinery draws them); clear cells that have
    no matches. Map the global active index to the owning cell's local index so
    the active match gets the active-match color.
  - `activate_match(index)`: look up `matches[index].cell_id`, select that cell
    (`set_selected_index(.., false)`), focus its editor, delegate
    `activate_match(0, &[range])` to the cell editor for in-cell selection, then
    `follow_scroll_to(cell_index)` to reveal the cell.
  - `active_match_index`: compute at notebook scope from the selected cell +
    that cell editor's cursor: find the match at/after (Next) or at/before
    (Prev) the (selected_cell_index, cursor) position in the global ordering.
  - `query_suggestion`: delegate to the selected cell's editor.
  - `select_matches`: group by cell, delegate (select-all within each cell).
- `supported_options`: part 1 → `{ case, word, regex: true, replacement:
  false, selection: false, select_all: true, find_in_results: false }`.
  `replace`: no-op in part 1 (never called while `replacement: false`).

## Tasks

- [ ] Add `impl EventEmitter<SearchEvent> for NotebookEditor` and emit
      `MatchesInvalidated` / `ActiveMatchChanged` where the per-editor impl does.
- [ ] Define the `Match` type and implement `SearchableItem for NotebookEditor`
      (find / update / clear / activate / active_index / query_suggestion /
      select / supported_options / no-op replace) delegating to per-cell editors.
- [ ] Flip `as_searchable` to `Some(Box::new(handle.clone()))`.
- [ ] Cross-cell active-match index helper (ordered position vs selected cell +
      cursor), reusing `editor::active_match_index` per cell where possible.
- [ ] `activate_match` selects + focuses the cell and reveals it via the
      index-anchored `follow_scroll_to` (bug-#45-safe), then selects the range
      in the cell editor.
- [ ] `./script/clippy` clean; `cargo build -p repl` succeeds.
- [ ] **User test:** Ctrl-F in a notebook highlights matches across cells;
      next/prev cycles through them in document order; activating a match
      selects the cell, scrolls to it, and selects the text; Esc closes search.

## Out of scope (separate future phases / backlog)

- Replace / replace-all + full options (part 2).
- Global-search (Ctrl-Shift-F) result opening raw JSON instead of the notebook
  — separate backlog DEFECT, different root cause (type vs path opener
  registry).

## Risks / gaps

- Borrow/reentrancy: iterating `cell_map` while calling `cell_editor.update`;
  collect the ordered `Entity<Editor>` handles first, then operate. Capture the
  outer `window` into the sync `editor.update` closure (editor methods take a
  `&mut Window` they mostly ignore for find/highlight).
- Anchors are per-cell-buffer; a `Match`'s range is only valid in its own cell
  editor — never mix. Edits invalidate matches (emit `MatchesInvalidated`).
- Markdown/raw cells: search their source editor if `cell.editor` yields one;
  otherwise skip. Confirm which cell kinds expose an editor.
- Far/unmeasured cells: `activate_match` must reveal via the index-anchored
  path (bug #45), not a cumulative-height reveal.
