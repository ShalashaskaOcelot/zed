# Phase 37 — In-place selectable output text

Kind: **new feature**. Not yet started — this is a plan. Re-phased out of
phase 29 (2026-07-14): it needs phase-sized work of its own, so it fills the
rotation slot phase 29's completion opened.

Goal: select a PORTION of a cell's output text with the mouse and copy it,
without leaving the notebook.

Findings from phase 29's investigation:
- Text outputs render via `TerminalOutput` (`outputs/plain.rs`), a custom
  ANSI-aware element with NO text-selection support; markdown/table outputs
  are custom elements too. Nothing in the current element tree can host a
  selection.
- Interim affordance shipped in phase 29: every notebook output now has the
  inline REPL's controls, including "Open in Buffer" — a read-only editor
  where text is fully selectable/searchable.

Approach options (decide at implementation):
1. Render text-bearing outputs (stream/plain/error) through a read-only,
   borderless Editor instead of `TerminalOutput` — selection, copy, and
   search come free; ANSI colors need mapping to text highlights.
2. Add mouse-selection support to `TerminalOutput`'s element (selection
   ranges + copy handling) — keeps rendering as-is, more custom code.

## Tasks

- [ ] Pick the approach (spike both if needed) and implement selection for
      stream/plain/error outputs.
- [ ] Copy selected text with ctrl/cmd-c without breaking cell-level
      copy in command mode.
- [ ] Ensure selection doesn't fight cell selection/focus (bug #23 area:
      output clicks must not steal cell edit-mode semantics).

## Verification

- `cargo check -p repl` + clippy clean; tests pass.
- User test: drag-select part of a long output, copy it; cell shortcuts and
  gutter selection still behave.
