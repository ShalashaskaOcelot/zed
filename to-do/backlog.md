# Backlog

Non-phased suggestions and to-do items that are NOT yet scheduled. Move an item
into a `phase_<n>.md` when it is scheduled (and delete it from here); never
implement directly from here. Completed and scheduled work is not tracked here —
see the phase files, `CHANGELOG.md`, and git history. Roughly ordered
high → low within each group.

## Medium priority

- Save-as dialog for notebooks (user 2026-07-16): default the file-type
  filter to something sensible (not "all files") and make sure the `.ipynb`
  extension is applied/autofilled rather than left off.
- Rust kernel (evcxr_jupyter) treats all stderr as errors (user 2026-07-20):
  evcxr writes EVERYTHING to stderr — compile progress, `Compiling {crate}`
  lines, warnings — not just real errors, so running any Rust cell floods the
  output with spurious `ERROR:` entries ("ERROR: compiling {crate}" etc.).
  The convention evcxr follows is that stdout is reserved for actual program
  output while stderr carries compilation/diagnostic logs. Goal: distinguish
  normal Rust build/log chatter on stderr from genuine errors so the noise can
  be filtered out of (or de-emphasised in) the error output. Investigate:
  where the notebook maps a Jupyter `stream` message with `name: "stderr"`
  onto an error-styled output (`crates/repl/src/outputs/`, `cell.rs` output
  handling) — evcxr sends compile logs as `stream`/stderr, and real Rust
  errors come through as `error`/`execute_reply` with `status: "error"`, so
  gating the error styling on the actual message TYPE (not the stream name),
  and/or a kernel-language-aware filter for known-benign evcxr stderr
  patterns, may be enough. Confirm evcxr's actual message shapes first.

## Medium priority (cont.)

- Global-search result opens raw JSON, not the notebook (user 2026-07-16,
  DEFECT): Ctrl-Shift-F does include notebook content, but clicking a notebook
  result opens the `.ipynb` as raw JSON text instead of the `NotebookEditor`.
  Root cause (research 2026-07-16): clicking a project-search excerpt goes
  through `Editor::open_buffers_in_workspace` →
  `workspace.open_project_item::<Editor>` (`crates/editor/src/editor.rs:10111`),
  which uses the TYPE registry (hardcoded to text `Editor`) and bypasses the
  PATH registry that maps `.ipynb` → `NotebookEditor` (used by the file tree
  via `open_path`). Quick-win fix: when the buffer's file has a non-`Editor`
  path opener registered, route the open through `workspace.open_path` (opens
  the real notebook; loses the intra-file match jump — acceptable first cut).
  Must be done generically (editor can't depend on repl). Bigger follow-ups
  (separate items): jump to the matching cell; make search preview show cell
  content instead of raw JSON.
- Kernel picker can't distinguish "still discovering" from "nothing found"
  (user 2026-08-06, out of confirming bug #20): kernelspec + python-toolchain
  discovery is asynchronous, so a picker opened immediately after launch is
  legitimately empty for a moment and shows the generic "No matches" — which
  reads as "you have no Python interpreters". Show a discovery state instead
  (e.g. a "Searching for kernels…" row / spinner) while a refresh is in
  flight, so that once it settles "No matches" is trustworthy and actually
  means no interpreter was found. Needs an in-flight flag on `ReplStore`'s
  refreshes (`refresh_kernelspecs` / `refresh_python_kernelspecs`) that the
  picker delegate can read — the picker already observes the store and rebuilds
  its entries live (bug #20's fix), so this is only about what is displayed
  while empty.

## Low priority

- Kernel picker: show the env path under Jupyter-kernel entries the way
  Python Environment entries show theirs (user 2026-07-16) — registered
  venv kernelspecs currently give no clue which directory they point at.
- Clicking an output body could also select its cell (user 2026-07-16) —
  currently only the cell gutter/border selects the cell; clicking on an
  output's content doesn't (a plain click there should select the cell, while
  a drag still selects output text). Minor nicety; unclear it ever selected
  from the output body, so not filed as a regression.
- Arch Linux distribution (user 2026-07-16, explicitly deferred — "long
  finger"): proper pacman-managed install, i.e. a self-hosted pacman repo
  the user's machines can pull from, or an AUR package (paru-manageable).
  No AppImages. Needs the Linux bundle (`script/bundle-linux`) plus
  PKGBUILD/repo tooling, and updates handed to pacman (build with
  `ZED_UPDATE_EXPLANATION` so in-app auto-update stays off for the pacman
  build). Windows (phases 44-47) comes first.
- Upstream the fork's two generic changes as PRs to zed-industries/zed to
  permanently shrink the merge-conflict surface (phase 43 discovery):
  gpui `scroll_to_reveal_item_top_aligned` (`crates/gpui/src/elements/list.rs`)
  and `util::process` `spawn_interruptible`/Windows interrupt plumbing
  (`crates/util/src/process.rs`).

- Horizontal scrollbar for wide notebook outputs (user 2026-07-30): rich outputs
  (e.g. pandas DataFrames) render at full width but overflow the viewport and
  clip on the right with no way to scroll horizontally. Add a horizontal
  scrollbar / horizontal scroll to the cell output container for wide outputs
  (tables, wide text). Relates to bug #53 (text output too narrow) but is the
  opposite end — this is about outputs that are TOO wide to fit. Investigate the
  output container sizing in `crates/repl/src/outputs/` and `notebook/cell.rs`.

- Notebook-aware copy/paste with a separate notebook clipboard (user 2026-07-30):
  copying cells should paste as FULL CELLS (structure, types, outputs) into
  another notebook, but as PLAIN CODE — no JSON — into a normal text file or
  outside Zed. VS Code does this with two clipboards: an internal notebook
  clipboard holding the rich cell data, and the system clipboard holding just
  the code. Mirror that: keep the existing rich snapshot for notebook→notebook
  (phase 7/27 machinery) and additionally put a plain-text rendering on the
  SYSTEM clipboard. **Improvement over VS Code the user specifically wants:**
  when producing that plain-text form, prefix every line of a MARKDOWN cell with
  `# ` so pasting a mix of code and markdown cells into a `.py` yields valid
  Python with the prose as comments. (VS Code leaves markdown uncommented, so
  any non-heading markdown becomes a syntax error.)
- Setting to swap the right control sidebar for a VS Code-style TOP control bar
  (user 2026-07-30). Requirements the user gave:
  * A top version of the control bar, which ABSORBS the existing top-right
    kernel cluster (so there aren't two kernel indicators in the same place) and
    makes the sidebar's kernel selector redundant — remove it in this mode
    (note: that sidebar selector is also the one that crashes, bug #54).
  * Remove the right sidebar entirely in this mode.
  * With the sidebar gone, the cell-list scrollbar moves to the true right edge
    and renders properly (relates to bug #57 — the scrollbar currently overlaps
    the cell margin because the sidebar occupies that space).
  * A setting to choose between the two layouts.

- Accurate notebook scrollbar via full measurement (user 2026-07-30). The
  scrollbar thumb is wrong on open and SHRINKS as you scroll: a notebook that
  looked ~3 viewports long read as ~20+ once scrolled to the end.
  **Cause (confirmed):** the cell list is virtualized and an UNMEASURED item
  contributes `px(0.)` to the height tree (`gpui/src/elements/list.rs`,
  `ListItem::Unmeasured` summary). Total content height is therefore massively
  underestimated until items are laid out, and grows as you scroll — so the
  thumb shrinks. This is the SAME root cause as bug #45 (End landing short), and
  the user's diagnosis of that bug was right: End was effectively jumping to the
  end of the MEASURED content, then further each press as more got measured.
  (#45 itself is fixed — End now uses the index-anchored `scroll_to_end`, which
  walks backwards from the last item and so is immune to heights. This item is
  only about the SCROLLBAR being honest.)
  **Lever that already exists:** `ListState::measure_all()` — its own doc says
  "useful for ensuring that the scrollbar size is correct instead of based on
  only rendered elements". Already used by csv_preview, settings_ui and pickers.
  Applying it to the notebook's `cell_list` should give a correct thumb.
  **Cost / the setting the user suggested:** `measure_all` LAYS OUT every item
  on first layout (it does not paint them all), so it is O(cells) work on open —
  noticeable for a large notebook with heavy outputs, but much cheaper than full
  rendering. Do it as the user proposed: measure everything by DEFAULT, with a
  setting (e.g. `notebook_dynamic_render`, default off) to fall back to
  measure-as-you-scroll on low-power machines. Verify open time on the user's
  large Rust notebooks before settling the default.
- Alternative "follow running cell" modes (user 2026-07-31). Follow mode
  currently pins each cell near the TOP of the viewport as it starts
  (`follow_scroll_to`, called from `advance_run_queue` in `notebook_ui.rs`),
  which means you watch the cell's SOURCE — the part you already wrote and the
  least informative part while it runs. What you actually want to watch is the
  execution status and the output being produced.
  **The anchor is the STATUS LINE, not the output (user 2026-07-31, with
  screenshots).** This is not a head-vs-tail-of-output question — the user
  ruled that out. What the framing has to deliver is:
  * that there IS output accumulating (the notebook is visibly working), and
  * the cell's status/runtime footer — the `Running… 10.2s` / `✓ 51.9s` line
    that sits between the source and the output — so you can tell a cell that
    is still running with no output yet from one that has completed or failed.
  The wanted framing is the bottom of the source, the status footer, and the
  first chunk of output all on screen at once. Today's top-aligned framing
  shows only source, giving no signal that the notebook is even still alive.
  Implement by anchoring the reveal on the status footer rather than the cell
  top (`follow_scroll_to`), leaving a little source above it and as much output
  below as fits. Decide what to do when a cell is taller than the viewport —
  the footer is the priority, source above it is the first thing to sacrifice.
  Same underlying complaint as the reveal-target note in the cell-error
  navigation item above; whatever is chosen here should inform that, since both
  are "scroll so the USEFUL part is on screen". The user notes the proposed
  failure indicator in the top kernel strip would partly cover the
  "is-it-still-running" gap, but not fully — this is still worth doing.
  **Priority: explicitly deferred by the user (2026-07-31) — not needed now.**
- Notebook control buttons should focus the notebook (user 2026-07-30). Bug #51
  made the sidebar buttons act on their own notebook regardless of focus, but
  focus itself stays wherever it was (e.g. the project panel), so keyboard
  shortcuts still don't go to the notebook afterwards. Clicking Run All /
  Restart / Stop / etc. should ALSO move focus to that notebook, exactly as
  clicking a cell or a per-cell run button does. Small: focus the notebook's
  focus handle in the control-button listeners (`render_notebook_controls`).
- Kernel picker "Creating <name>…" row polish (user 2026-07-30): the row works
  but "could look a little better" visually. (The separate defect — it not
  refreshing live when the build completes — is bug #58.)

- Add `smooth_scrolling` to the GUI settings UI (user 2026-07-30). Phase 54
  added the setting to `default.json`, the schema and the docs, but NOT to the
  settings UI — so it's JSON-only today. It belongs in the existing **Editor →
  "Scrolling"** section (`crates/settings_ui/src/page_data.rs:1811`), right
  alongside its siblings `scroll_sensitivity`, `mouse_wheel_zoom` and
  `fast_scroll_sensitivity`, which all already have entries there — so its
  absence is a genuine gap, not an upstream convention. It is a GENERAL editor
  setting (it affects every file, not just notebooks), so it does not belong on
  the REPL & Notebooks page. Small: one `SettingsPageItem` with a
  `json_path: Some("smooth_scrolling")` + pick/place pair following the
  `mouse_wheel_zoom` boolean exactly. NOTE: `page_data.rs` is a known
  upstream-merge hotspot (phase 46) — keep the addition minimal and adjacent to
  the related entries.

- Remove the redundant kernel selector from the notebook's right sidebar (user
  2026-07-30). There are two kernel-picker triggers: the top-right kernel strip
  and one at the bottom of the right control sidebar. They share a single
  `PopoverMenuHandle`, so clicking the SIDEBAR one opens the popover anchored at
  the TOP-RIGHT trigger — visibly odd, and confirming the redundancy. Simplest
  fix is to drop the sidebar trigger and keep the top-right strip (which already
  shows kernel name + status). This is a subset of the larger "top control bar
  vs right sidebar" item above, but is worth doing on its own regardless of that.
