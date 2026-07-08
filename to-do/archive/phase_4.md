# Phase 4 — Cell actions, Jupyter shortcuts, and the two dead menus (COMPLETE, archived 2026-07-08)

Kind: new feature. Confirmed by the user 2026-07-08: delete (`x`/`d d`),
convert (`m`/`y`), add (`a`/`b`), the "More options" menu, run-above /
run-cell-and-below, and the output "..." menu all work. Per the new-feature
rule, the phase is delivered; the tweaks the user raised are filed as new
items rather than reopening this phase.

## Delivered

- `DeleteCell` (`x`, `d d`), refuses to delete the only cell.
- `AddCellAbove` / `AddCellBelow` (`a` / `b`).
- `RunCellsAbove` / `RunCellAndBelow` (via the More options menu).
- `ConvertToCode` / `ConvertToMarkdown` (`y` / `m`).
- Right-toolbar "More options" popover menu.
- Output "..." per-cell menu (Copy Output / Clear Output).
- Shared `wire_*` / `build_*` cell helpers.

## Follow-ups spun off from user feedback (NOT reopened here)

- Deleting the last cell: currently refused. User asked whether it should
  instead clear/replace with an empty cell. → `backlog.md` (functionality
  tweak).
- `a`/`b` (and the + buttons) currently enter EDIT mode immediately; VS Code
  focuses the new cell but stays in command mode until Enter. → `backlog.md`.
- Run-above / run-cell-and-below live only in the "More options" menu; user
  prefers VS Code-style per-cell buttons in the focused cell's top-right. →
  covered by the "cell hover controls" backlog item (updated with this
  preference).

## Verification

- Confirmed working at runtime by the user (2026-07-08).
- `cargo test -p repl`: 37 passed; clippy clean.
