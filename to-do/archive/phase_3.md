# Phase 3 — Navigation and scrolling (COMPLETE, archived 2026-07-08)

> STATUS: CONFIRMED by the user 2026-07-08. Fit-aware reveal, natural fit,
> cursor-follow, and Home/End (via dedicated `notebook::SelectFirstCell`/
> `SelectLastCell` actions) all confirmed working. Kind: mixed — all items
> resolved.

Goal: nav-mode cell navigation and edit-mode cursor movement keep the right
thing in view.

Primary files: `crates/repl/src/notebook/notebook_ui.rs`,
`crates/gpui/src/elements/list.rs` (scroll primitive),
`assets/keymaps/default-{linux,macos,windows}.json`.

## Tasks

- [x] Fit-aware cell reveal (change): new
      `ListState::scroll_to_reveal_item_top_aligned` (`list.rs`) — if the item
      is taller than the viewport it pins the item's TOP to the top of the
      view regardless of travel direction; if it fits it falls back to the
      existing minimal reveal. `jump_to_cell` now uses it. ✅ CONFIRMED (user
      2026-07-08).
- [x] View follows cursor in edit mode (change): `follow_cursor_in_cell`,
      driven by a shared `on_cell_editor_event` handler on `SelectionsChanged`
      for the selected cell. Scrolls the notebook list only when the cursor
      would fall outside the viewport (with a small margin) — does NOT keep it
      centered. NOTE: the cursor's vertical position is estimated from its
      fractional display-row within the cell's laid-out height (the cell
      includes non-editor chrome), so it is approximate; may need tuning after
      testing. ✅ CONFIRMED working (user 2026-07-08): "view correctly follows
      the cursor, no jumping around, no leaving viewport". (The fractional-row
      approximation held up in practice.)
- [x] Home/End → first/last cell. FIRST ATTEMPT (bind to `menu::SelectFirst`/
      `SelectLast`) did NOT work — those actions were swallowed even though
      `menu::SelectNext`/`Previous` (arrows) reach the notebook. FIX: added
      dedicated `notebook::SelectFirstCell` / `SelectLastCell` actions and
      bound `home`/`end` to them in the command-mode context (all three
      keymaps). ✅ CONFIRMED working (user 2026-07-10).
- [~] Smart arrows (`NotebookMoveUp`/`NotebookMoveDown`) in edit mode:
      DECIDED to defer — moved to `backlog.md`. Binding up/down in the
      `NotebookEditor > Editor` context risks overriding completion-menu
      navigation and needs runtime testing to get context precedence right.

## Manual test checklist (for the user)

- [ ] Arrow down into a cell TALLER than the viewport (coming from above):
      the cell's top aligns to the top of the view (not its bottom).
- [ ] Arrow up into a tall cell (coming from below): still top-aligned.
- [ ] A cell that fits: revealed naturally (VS Code-like), last cell sits at
      the bottom.
- [ ] Home/End jump to the first/last cell.
- [ ] Edit a tall cell and move the cursor down/up with arrows: the view
      follows only when the cursor reaches the edge (not centered every line).

## Notes

- `crates/gpui/src/elements/list.rs` change is additive (a new method); the
  existing `scroll_to_reveal_item` is unchanged, so other `list()` users are
  unaffected.
- If cursor-follow scrolls too much / too little or feels off, that's the
  approximation above — it's a change item, so it stays open and gets tuned in
  place rather than re-filed.

## Verification (automated)

- `cargo check -p repl` clean, `./script/clippy -p repl -p gpui` clean.
- `cargo test -p repl`: 37 passed, 0 failed.
