# Phase 1 — Discovery (COMPLETE, archived 2026-07-08)

Goal: no implementation. Survey the current notebook implementation
(`crates/repl/src/notebook/`, `crates/repl/src/kernels/`,
`crates/repl/src/outputs/`, `crates/repl/src/session.rs`,
`crates/repl/src/repl_store.rs`, `crates/repl/src/components/kernel_options.rs`,
keymaps) and:

- [x] Check for items reported broken that are actually (partially) implemented,
      and items that look implemented but were reported missing — flag for review
- [x] Check what can be easily implemented
- [x] Check what is more work
- [x] File everything classed as a bug in `bugs.md`
- [x] Plan subsequent phases and backlog

## Findings

### Architecture ground truth

- The notebook editor is `NotebookEditor` in
  `crates/repl/src/notebook/notebook_ui.rs` (~1800 lines), cells in
  `crates/repl/src/notebook/cell.rs`. The whole file starts with
  `#![allow(unused, dead_code)]` — an experimental, intentionally
  unfinished feature.
- The notebook UI and the inline REPL (`session.rs`) are two separate,
  parallel kernel-lifecycle implementations. The REPL version is older and
  handles several things more carefully (proper restart sequencing, queueing
  executions while the kernel starts). The notebook side has not reached
  parity.
- All notebook actions are defined in `crates/zed_actions/src/lib.rs:911-949`
  (`pub mod notebook`), handled/registered in `notebook_ui.rs:1340-1471`,
  and bound in the three keymaps under contexts `NotebookEditor` and
  `NotebookEditor > Editor`, with nav ("command") mode gated by
  `notebook_mode == command` (set from the `NotebookMode` enum,
  `notebook_ui.rs:52-56`, exposed to the keymap at `notebook_ui.rs:1324-1334`).
- Notebook scrolling is a single GPUI `list()` backed by `ListState`
  (`notebook_ui.rs:98`, `:1267-1277`); each cell embeds its own
  `Entity<Editor>` with `SizingBehavior::SizeByContent` and scrollbars
  disabled, so cells have no internal scrolling and the `ListState` is the
  only scroll container.
- Kernel discovery uses Zed's toolchain infra backed by Microsoft `pet`
  (python-environment-tools) via `project.available_toolchains()`
  (`kernels/mod.rs:408-663`, `crates/languages/src/python.rs:1283-1342`).
  Project `.venv` detection already works. Selecting an env without
  ipykernel already auto-installs it (`repl_editor.rs:78-202`, uses
  `uv pip install` or `pip install` with toasts).

### Reported broken but actually (partially) implemented — flagged for review

1. **Restart kernel** — it is NOT shutdown-only. `restart_kernel`
   (`notebook_ui.rs:491-502`) force-kills the old kernel and calls
   `launch_kernel_with_spec` to start a new one. The observed
   "kills but never restarts" is the relaunch half failing due to races:
   fire-and-forget `force_shutdown().detach()`, connection file path keyed
   only on `entity_id` (`native_kernel.rs:144`) so it's reused across
   restarts while the old kernel's `Drop` deletes it
   (`native_kernel.rs:300-305`), plus the documented ephemeral-port TOCTOU
   (`native_kernel.rs:79-91`). Contrast the correct REPL sequence at
   `session.rs:873-913` (send `ShutdownRequest{restart:true}`, wait, await
   forced kill, then relaunch). → bugs.md #1
2. **Interrupt** — fully implemented at the protocol level
   (`InterruptRequest` over the control channel, `kernels/mod.rs:156-160`;
   handler `notebook_ui.rs:504-516`; bound to ctrl-c). The button is greyed
   out because it is enabled ONLY when `KernelStatus::Busy`
   (`notebook_ui.rs:1253-1256`), and an errored/dead kernel reports
   `Error`, not `Busy`. Ctrl-c "does nothing" because the handler
   early-returns unless `Kernel::RunningKernel` and swallows send failures
   with `.ok()`. → bugs.md #3
3. **The screenshot error** ("control recv … os error 10054" →
   "the kernel failed to launch") is fully traced: control-socket read
   failure (`kernels/mod.rs:144-147`) → recv task bails →
   `kernel_errored` (`mod.rs:190-197`) → `Kernel::ErroredLaunch` →
   `execute_cell` renders "the kernel failed to launch: …"
   (`notebook_ui.rs:543`) via `show_kernel_error` (`cell.rs:826-840`).
   The native (non-WSL) launch path connects immediately with no readiness
   wait — the WSL path (`wsl_kernel.rs:290-323`) waits 2s and checks for
   premature exit; native does neither. → bugs.md #6
4. **Run-button number** — confirmed it is the Jupyter `execution_count`
   from `ExecuteInput` messages (`cell.rs:651`, `:879-884`, rendered
   `:1039-1047`). Not a bug; working as intended.

### Confirmed missing / stub (matches user report)

- "More options" toolbar button: tooltip only, no `on_click`, no menu
  (`notebook_ui.rs:1124-1127`). A `CellControlType::CellOptions` variant
  exists but is never used (`cell.rs:37`). → bugs.md #4
- Output "..." button: bare `IconButton` with no handler (`cell.rs:924-939`).
  Note the per-output copy buttons in `outputs.rs:200-221` DO work. → bugs.md #5
- Run-cell with a dead kernel does NOT attempt to (re)start it —
  `execute_cell` (`notebook_ui.rs:518-567`) has no relaunch branch; it just
  renders an error output. → bugs.md #2
- No delete/split/join/change-type/copy/cut/paste/duplicate cell actions
  exist at all; no run-all-above / run-cell-and-below.
- No insert-above: `insert_cell_at_current_position` (`notebook_ui.rs:763`)
  hardcodes `selected_cell_index + 1`.
- No venv/conda creation anywhere user-facing (only an internal
  pylsp-hosting venv, `python.rs:1740-1786`).
- Nav-mode scroll: `ListState::scroll_to_reveal_item`
  (`crates/gpui/src/elements/list.rs:626-656`) reveals the nearest edge —
  moving down aligns the cell bottom, moving up aligns the top — with no
  cell-fits-in-view awareness. Matches the reported behaviour exactly.
- Edit-mode cursor is NOT followed by the notebook scroll: cell editors are
  `SizeByContent` (no internal scroll extent) and nothing wires editor
  cursor movement to `cell_list` scrolling; `Item::pixel_position_of_cursor`
  returns `None` (`notebook_ui.rs:1740`, TODO).

### Easy wins (implemented but unbound, or trivial)

- `NotebookMoveUp` / `NotebookMoveDown` — fully implemented "smart arrows"
  (move within a cell, cross cell boundary at first/last line;
  `notebook_ui.rs:1395`, `:1431`) but have NO keybinding.
- `select_first` / `select_last` handlers exist (`notebook_ui.rs:954`,
  `:967`) but Home/End are not bound.
- Interrupt button enablement fix is a one-liner class of change.
- Remove leftover `println!` debug lines (`notebook_ui.rs:740, 744, 753`).

### More substantial work

- Restart sequencing + unique connection files + launch readiness checks.
- Fit-aware scroll (likely a new/changed `ListState` method in gpui;
  primitives available: `bounds_for_item`, `logical_scroll_top`,
  `last_layout_bounds`).
- View-follows-cursor in edit mode (needs editor→notebook scroll wiring).
- Cell-level action set (delete/add-above/change-type/run-above/run-below)
  plus the two dead menus.
- Venv creation flow (template exists: `install_ipykernel_and_assign`
  background-command + toast pattern; pet re-scan picks new envs up).

### Other findings (not user-reported)

- `restart_kernel` never clears `execution_requests` (stale msg_id→cell map;
  `change_kernel` does clear it, `notebook_ui.rs:486`). → bugs.md #7
- Native process-exit watcher only transitions state on FAILED exit; a clean
  kernel exit leaves stale `RunningKernel` state (`native_kernel.rs:220-242`).
  → bugs.md #8
- `NotebookItem::is_dirty` hardcoded `false` (`notebook_ui.rs:1595-1598`) —
  unsaved structural/metadata changes won't prompt to save. → bugs.md #9
- `open_notebook` is a `println!` stub (`notebook_ui.rs:739-741`).
- No file watching for external notebook changes (TODO `notebook_ui.rs:1519`).
- Notebook doesn't queue executions while the kernel is starting (REPL does,
  `session.rs:683-793`).
- `kernelspec.interrupt_mode` is always `None`; message-based interrupt is
  the only mechanism used.

## Disposition

All tasks complete. Follow-on work planned in `phase_2.md` — `phase_5.md`;
everything else moved to `backlog.md`; bugs recorded in `bugs.md`.
