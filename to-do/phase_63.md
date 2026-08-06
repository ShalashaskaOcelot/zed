# Phase 63 — Wide notebook outputs and output-body interaction

Kind: **mixed** — a change to existing behaviour (wide outputs must be
reachable) plus a small new behaviour (clicking an output selects its cell).
Promoted from the backlog 2026-08-06 to replace the parked release-engineering
phases.

Goal: an output that is WIDER than the cell area can be read in full, and
clicking on an output body behaves like clicking the cell.

## Key facts (code inspection 2026-08-06 — read before assuming the fix)

- The output block ALREADY opts into horizontal scrolling: the outputs
  container in `cell.rs` (the `"output-scroll"` div) is `.w_full()` +
  `.overflow_x_scroll()`, with `max_h`/`overflow_y_scroll` layered on when
  `output_max_height_lines` is set. So the missing piece is NOT "add
  overflow_x_scroll" — the first task is to find out why wide content still
  clips instead of scrolling.
- Two plausible causes, both worth checking before writing code:
  1. the inner output element is itself `w_full`, so it shrinks to the
     container instead of laying out at its natural width (nothing to scroll);
  2. plain/stream output goes through `TerminalOutput`, which is a terminal
     emulator sized to `ReplSettings::max_columns` (default 128) — it WRAPS at
     that width rather than producing an over-wide line, so text output can
     never overflow horizontally no matter what the container does. Rich
     outputs (pandas `DataFrame` HTML/table) are the ones that actually exceed
     the width — see `outputs/table.rs`.
- There is no visible horizontal scrollbar on the output block; `overflow_x_scroll`
  alone doesn't draw one. The notebook's cell list got its scrollbar via
  `.custom_scrollbars(...)` (phase 56) — the same `ui::Scrollbars` machinery is
  available here, tracked on the output div's scroll handle.
- Cell selection today comes from the gutter/border click handlers; the output
  body has no click handler, so clicking an output leaves the selection where
  it was. Any handler added must NOT break text selection by dragging inside
  the output (a drag is not a click).

## Tasks

- [ ] Reproduce with a wide rich output (a pandas DataFrame with many columns)
      and determine which of the two causes above is in play; write the finding
      into this file before fixing.
- [ ] Make wide output content actually scrollable horizontally within the
      output block (size the inner element to its content rather than the
      container, whatever that takes for the table/rich-output path).
- [ ] Add a visible horizontal scrollbar for the output block when its content
      overflows, using the same `ui::Scrollbars`/`custom_scrollbars` approach as
      the cell list, following the user's editor scrollbar visibility setting.
- [ ] Do NOT let the output capture the vertical scrollwheel (the bug-#63
      decision stands): horizontal scroll must not swallow the notebook's own
      scrolling.
- [ ] Clicking an output body selects its owning cell, while a click-and-drag
      inside the output still selects TEXT (no selection stolen mid-drag).
- [ ] `./script/clippy` clean and `cargo test -p repl` passes.

## User tests (runtime)

- [ ] A DataFrame far wider than the pane can be scrolled sideways within its
      output block, with a visible scrollbar, and the notebook itself does not
      scroll sideways.
- [ ] Vertical scrolling over an output still scrolls the NOTEBOOK, not the
      output.
- [ ] Clicking on an output selects that cell (the gutter highlights) and
      keyboard shortcuts then act on it; dragging across output text still
      selects the text.
- [ ] Narrow outputs are unchanged — no scrollbar, no layout shift.
