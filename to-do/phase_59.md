# Phase 59 — Truncate long cell output from the TOP, VS Code style

Kind: **change to existing behaviour** (notebook-primary). Decided by the user
2026-07-30 after investigating bug #63. Fixes that bug.

Goal: a long text output shows its BEGINNING plus a truncation notice, instead
of silently showing only its last ~32 lines. `max_lines` stays at 32 — the user
confirmed that's a fine default; the problem is *which* 32 lines are shown and
the lack of any indication that there are more.

## Key facts (code inspection 2026-07-30)

- Notebook plain/stream output renders through `TerminalOutput`
  (`crates/repl/src/outputs/plain.rs`) — a real terminal emulator sized to
  `ReplSettings::max_lines` rows (`terminal_size`), default 32, clamped [4, 256].
- Appending more than that scrolls earlier lines off exactly as a console does,
  so the viewport ends up on the TAIL. That is the whole of bug #63.
- Nothing is lost: the emulator keeps `DEFAULT_SCROLL_HISTORY_LINES` = 10,000
  lines of scrollback, and `full_text()` reads
  `terminal.get_content()` — scrollback included — which is what
  `buffer_content()` ("open in buffer") serialises. **So the fix must keep
  feeding the terminal everything; only the VIEWPORT should change.**
- `Terminal::scroll_to_top()` exists (`crates/terminal/src/terminal.rs`), as do
  `scroll_to_bottom`, `scroll_up_by`, `scroll_page_up`/`down`.
- `TerminalOutput` is SHARED with the inline REPL, where following the tail is
  the RIGHT behaviour (live output, newest matters most). So top-pinning must be
  opt-in per instance, not a global change — same pattern as
  `Editor::set_minimal_container_autoscroll` (bug #60) and
  `ListState::set_smooth_scroll` (phase 55).

## Tasks

- [x] Add an opt-in "pin to top" mode to `TerminalOutput` (default OFF, so the
      inline REPL is untouched): after `append_text`, keep the viewport at the
      start of the content via `Terminal::scroll_to_top()`.
- [x] Track whether the content exceeds the viewport (a line counter in
      `append_text` compared against the configured `max_lines`, or the
      terminal's own total-lines vs rows) so the notebook can tell when output
      has been truncated, and by how much.
- [x] Notebook cell outputs opt in (`notebook/cell.rs`, where the cell's
      `Output::Stream` / plain outputs are built).
- [x] Render a truncation notice under a truncated output — VS Code's wording is
      "Output is truncated. View as a scrollable element or open in a text
      editor." Ours should say the output is truncated and offer the EXISTING
      open-in-buffer action (already implemented; it yields the full text
      including scrollback). Keep it unobtrusive — one muted line.
- [x] `./script/clippy` clean; `cargo test -p repl` passes.
- [ ] **User test:** a cell printing far more than 32 lines shows the FIRST ~32
      with a truncation notice; open-in-buffer still yields the whole output;
      short outputs are unchanged with no notice; the INLINE REPL still follows
      the tail as before.

## Explicitly NOT in this phase

- **Do not make outputs capture the scrollwheel.** The user was clear they do
  not want every long output grabbing the wheel as the pointer passes over it.
  A "view as a scrollable element" affordance (opt-in per output, like VS Code's)
  is a possible FOLLOW-UP, not part of this.
- The existing selection-drag scrolling inside an output is janky (user
  2026-07-30: "you can 'scroll' outputs but it doesn't work properly" —
  highlighting and dragging up selects everything and moves the view). Left
  as-is here; if it still grates once truncation lands, file it separately.

## Risks / gaps

- A LIVE cell that is still streaming will show its first lines rather than the
  newest. That matches VS Code and is what was asked for, but is a behaviour
  change worth confirming for long-running cells that print progress.
- `scroll_to_top` is queued as an internal terminal event; confirm it survives
  subsequent `write_output` calls (each append may reset the viewport to the
  bottom, so the pin probably has to be re-applied after every append).
- Don't regress in-place output text selection (phase 37) or the copy button —
  both read from the terminal, whose content is unchanged by this.
