# Backlog

Non-phased suggestions and to-do items that are NOT yet scheduled. Move an item
into a `phase_<n>.md` when it is scheduled (and delete it from here); never
implement directly from here. Completed and scheduled work is not tracked here —
see the phase files, `CHANGELOG.md`, and git history. Roughly ordered
high → low within each group.

## Medium priority

- Notebook session persistence (user 2026-07-16): restore notebooks with the
  workspace like text buffers are. TWO gaps today: (1) SAVED notebooks that
  were open are not reopened on relaunch; (2) UNSAVED/untitled notebooks are
  lost (an unsaved text buffer from Ctrl-N IS restored, and because notebooks
  don't participate, closing the last window prompts to save the notebook
  instead of silently keeping it in the session like unsaved buffers).
  Implementation direction: `workspace::SerializableItem` for
  `NotebookEditor` — saved ones re-open by path; untitled ones serialize
  their nbformat JSON to the workspace DB the way unsaved buffers store
  their text.
- Save-as dialog for notebooks (user 2026-07-16): default the file-type
  filter to something sensible (not "all files") and make sure the `.ipynb`
  extension is applied/autofilled rather than left off.

## Medium priority (cont.)

- In-notebook search (Ctrl-F) (user 2026-07-16): searching within a notebook
  does nothing today. Root cause (research 2026-07-16): `NotebookEditor::as_searchable`
  returns `None` (`crates/repl/src/notebook/notebook_ui.rs`); the rest of the
  Ctrl-F pipeline is generic and works once an item is a `SearchableItem`.
  A notebook is N independent cell editors (not a multibuffer), so this is the
  first fan-out `SearchableItem`: `NotebookEditor` implements the
  `SearchableItem` trait (`crates/workspace/src/searchable.rs`) delegating each
  primitive to the per-cell `Entity<Editor>` (Match = (CellId, Range<Anchor>));
  `activate_match` selects the cell + `scroll_to_reveal_item_top_aligned` +
  focuses + delegates. Phase 1 = find/highlight/Next-Prev across cells (no
  replace); Phase 2 = replace + options. Est. medium (~1.5–3 days), all in
  `crates/repl/src/notebook/`.
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
- Go to running cell (user 2026-07-16): a "Go to running cell" action that
  reveals + selects + focuses the currently-executing cell, in the command
  palette AND as a right-sidebar button under Run All (greyed/disabled when
  nothing is running). Research 2026-07-16: small (~1–1.5 hr). Find the cell
  whose `is_executing()` is true (`cell.rs`) → `set_selected_index(index,
  true, …)` + `enter_command_mode`; add `notebook::GoToRunningCell` action
  (auto-appears in palette), sidebar button via `render_notebook_control` with
  `.disabled(running_cell_index(cx).is_none())` (mirrors the Interrupt
  button). Per-cell running spinner (also mentioned) ALREADY exists
  (phases 18/21) — no work there.
- Notebook scrollbar missing / hidden behind the right bar (user 2026-07-16):
  the notebook cell list has no visible scrollbar. Either there isn't one, or
  one is drawn but sits UNDER the right-hand control sidebar
  (`render_notebook_controls`) that runs down the right edge. Fix: add a
  scrollbar to the cell-list container (the `list()`/`ListState` `cell_list`
  in `NotebookEditor::render`, `notebook_ui.rs`) and, if one already exists,
  position it so it's not occluded — e.g. inset it left of the sidebar, or
  move the scrollbar/sidebar so they don't overlap. Check whether Zed's
  editor/list scrollbar component can be reused. Investigate first whether a
  hidden one is already there before adding a second.
- Global kernel busy/idle indicator (user 2026-07-16): a single indicator,
  visible regardless of scroll position, showing whether the kernel is idle or
  actively working — NOT the per-cell spinner (that exists). The top kernel
  strip already shows a status icon (`render_kernel_strip`, Idle=Circle/Success,
  Busy=ArrowCircle/Warning, Starting=Muted) — so this is about making that
  busy/idle state clear and prominent enough to read at a glance from anywhere
  (e.g. an animated spinner + label while Busy), since the strip is pinned at
  the top above the cells. Small; enhances the existing indicator.

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

