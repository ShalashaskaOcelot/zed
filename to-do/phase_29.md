# Phase 29 — Output interaction polish

Kind: **mixed** (a behaviour change + two new features). Not yet started — this
is a plan. Promoted from the backlog (user 2026-07-11) to keep 5 phases in
rotation after phase 24 completed.

Theme: make cell OUTPUT easier to read, select, and copy.

Primary files: `crates/repl/src/outputs.rs` (output rendering/selection),
`crates/repl/src/outputs/plain.rs` + `crates/repl/src/outputs/table.rs`
(text / table views), `crates/repl/src/notebook/cell.rs` (output area layout).

## Tasks

- [ ] Selectable output text: allow selecting a PORTION of a cell's output to
      copy, instead of only the whole output via the "…" menu's Copy Output.
      Today outputs render as `TerminalOutput` / markdown / table elements that
      don't support text selection. Make at least the text-bearing outputs
      (stream / plain / error traceback) selectable with the mouse.
- [ ] "Open output in new editor": for long / scrolling outputs, an affordance
      (like VS Code's "Open in text editor") on the output "…" menu that opens
      the full output text in a regular read-only editor buffer/tab for
      searching, selecting, and comfortable scrolling.
- [ ] Further DataFrame/table polish (phase-16 follow-up): spacing, column
      sizing/eliding, header styling, very wide frames, and dark/light contrast.

## Risks / gaps

- Text selection inside a `TerminalOutput` may need the element to opt into
  selection; verify it doesn't break the notebook's own click-to-select /
  command-mode handling (bug #23 lives in the same area).
- "Open in editor" needs a path-less / scratch buffer and an editor item;
  reuse the inline-REPL output-in-editor plumbing if it exists.

## Verification

- `cargo check -p repl` + clippy clean; tests pass.
- User test: select part of a stream/error output and copy it; open a long
  output in an editor tab; a DataFrame renders cleanly (incl. wide frames and
  in both themes).
