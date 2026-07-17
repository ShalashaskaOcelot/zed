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

- Newly-created kernel should be selected immediately, before it's ready
  (user 2026-07-16). When you "Create Python/Conda Environment" while a
  DIFFERENT kernel is already selected, there's a gap during creation where
  the notebook falls back to the old kernel — running a cell then starts the
  OLD kernel, and the new one auto-switches in only once it finishes building.
  Desired: on choosing create, immediately show the new env as this
  notebook's selected kernel (rendered greyed/disabled in the picker like a
  no-ipykernel entry until ready), and queue any run as Pending until the new
  kernel is built, then start it and run — i.e. the exact behaviour that
  already happens when creating from the "Select Kernel" (no-kernel) state.
  Implementation: seed a placeholder/selected spec + treat the create task
  like a pending-launch, reusing the `cells_awaiting_kernel_choice` /
  `promote_awaiting_cells` path. (Confirmed working from the no-kernel start
  state; only the switch-from-another-kernel case regresses to the old one.)

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

