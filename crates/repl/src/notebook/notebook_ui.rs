use std::collections::BTreeSet;
use std::future::Future;
use std::ops::Range;
use std::time::{Duration, Instant};
use std::{path::PathBuf, sync::Arc};

use anyhow::{Context as _, Result, anyhow};
use collections::HashMap;
use feature_flags::{FeatureFlagAppExt as _, NotebookFeatureFlag};
use fs::MTime;
use futures::FutureExt;
use futures::channel::oneshot;
use futures::future::Shared;
use gpui::{
    AnyElement, App, ClipboardItem, Entity, EventEmitter, FocusHandle, Focusable, KeyContext,
    ListState, Point, PromptLevel, Task, TaskExt, WeakEntity, list, prelude::*,
};
use language::{Buffer, Language, LanguageRegistry};
use log;
use project::{Project, ProjectEntryId, ProjectPath};
use settings::{NotebookFollowMode, NotebookRunLandingMode, SeedQuerySetting, Settings as _};
use ui::{
    CommonAnimationExt, ScrollAxes, ScrollbarStyle, Scrollbars, Tooltip, WithScrollbar, prelude::*,
};
use workspace::item::{ItemEvent, SaveOptions, TabContentParams};
use workspace::notifications::NotificationId;
use workspace::searchable::{
    Direction, SearchEvent, SearchOptions, SearchToken, SearchableItem, SearchableItemHandle,
};
use workspace::{
    Item, ItemId, Open, Pane, ProjectItem, SerializableItem, Workspace, WorkspaceId,
    delete_unloaded_items,
};

use crate::notebook::persistence::{NotebookDb, SerializedNotebook};

use super::{
    Cell, CellEvent, CellExecutionStatus, CellPosition, CellToolbarAction, MarkdownCellEvent,
    RenderableCell, format_duration,
};

use nbformat::v4::CellId;
use serde_json;
use uuid::Uuid;

use crate::components::{KernelPickerDelegate, KernelSelector};
use crate::kernels::{
    Kernel, KernelSession, KernelSpecification, KernelStatus, NativeRunningKernel,
    PythonEnvKernelSpecification, RemoteRunningKernel, SshRunningKernel, WslRunningKernel,
};
use crate::notebook::MovementDirection;
use crate::notebook::env_name_modal::EnvNameModal;
use crate::repl_settings::ReplSettings;
use crate::repl_store::ReplStore;

use picker::Picker;
use runtimelib::{
    ExecuteRequest, ExecutionState, JupyterMessage, JupyterMessageContent, ReplyStatus,
    ShutdownRequest,
};
use ui::{ContextMenu, PopoverMenu, PopoverMenuHandle};
use util::ResultExt as _;
use zed_actions::editor::{MoveDown, MoveUp};
use zed_actions::notebook::{
    AddCellAbove, AddCellBelow, AddCodeBlock, AddMarkdownBlock, ClearCellOutputs, ClearOutputs,
    ConvertToCode, ConvertToMarkdown, CopyCell, CutCell, DeleteCell, DuplicateCell,
    EnterCommandMode, EnterEditMode, ExtendSelectionDown, ExtendSelectionToEnd,
    ExtendSelectionToStart, ExtendSelectionUp, GoToError, GoToRunningCell, InterruptKernel,
    JoinCells, MoveCellDown, MoveCellUp, NewNotebook, NextError, NotebookMoveDown, NotebookMoveUp,
    OpenNotebook, PasteCell, PasteCellAbove, PreviousError, RedoCellOp, ReloadNotebook,
    RestartKernel, Run, RunAll, RunAndAdvance, RunCellAndBelow, RunCellsAbove, SelectAllCells,
    SelectFirstCell, SelectLastCell, SplitCell, ToggleFollowRunningCell, UndoCellOp,
};

/// The failed-cell index to move to after `selected`, wrapping to the first
/// once past the last. `failed` must be ascending. `None` when nothing failed.
fn next_failed_index(failed: &[usize], selected: usize) -> Option<usize> {
    failed
        .iter()
        .find(|&&index| index > selected)
        .or_else(|| failed.first())
        .copied()
}

/// The failed-cell index to move to before `selected`, wrapping to the last
/// once past the first. `failed` must be ascending. `None` when nothing failed.
fn previous_failed_index(failed: &[usize], selected: usize) -> Option<usize> {
    failed
        .iter()
        .rev()
        .find(|&&index| index < selected)
        .or_else(|| failed.last())
        .copied()
}

/// Probe PATH for a conda-compatible frontend, preferring the most standard.
/// Returns the executable name to drive env creation with, or `None` when
/// none is installed. Running `--version` mirrors how the venv flow probes
/// `python`/`python3` (rather than depending on a `which`-style lookup).
async fn detect_conda_frontend() -> Option<&'static str> {
    for frontend in ["conda", "mamba", "micromamba"] {
        if let Ok(output) = util::command::new_command(frontend)
            .arg("--version")
            .output()
            .await
            && output.status.success()
        {
            return Some(frontend);
        }
    }
    None
}

/// Turn a display name into a Jupyter kernelspec name: lowercased, with any
/// character outside `[a-z0-9._-]` replaced by `-` (kernelspec names become
/// directory names, so they must be filesystem-safe).
fn sanitize_kernel_name(name: &str) -> String {
    name.to_lowercase()
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || matches!(character, '.' | '_' | '-') {
                character
            } else {
                '-'
            }
        })
        .collect()
}

/// A structural cell operation, stored so it can be undone/redone. Restored
/// cells are rebuilt from the serialized nbformat form (not resurrected
/// entities), so their subscriptions and language wiring are always fresh.
enum CellEdit {
    Inserted {
        index: usize,
        cell: nbformat::v4::Cell,
    },
    Deleted {
        index: usize,
        cell: nbformat::v4::Cell,
    },
    Moved {
        from: usize,
        to: usize,
    },
    Converted {
        index: usize,
        before: nbformat::v4::Cell,
        after: nbformat::v4::Cell,
    },
    /// Several edits applied as ONE logical operation (multi-cell delete,
    /// block move, multi-convert). Stored in applied order; undo replays them
    /// in reverse, redo replays them forward.
    Group(Vec<CellEdit>),
}

/// Whether the notebook is in command mode (navigating cells) or edit mode (editing a cell).
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum NotebookMode {
    Command,
    Edit,
}

#[derive(PartialEq, Eq)]
enum SelectionMode {
    SelectOnly,
    SelectAndMove,
}

pub(crate) const MEDIUM_SPACING_SIZE: f32 = 12.0;
// Wide enough for a three-digit execution number (`[169]`) in the cell gutter
// without wrapping; four digits will still wrap, which the user accepted as the
// right trade against pushing every cell's content further right.
pub(crate) const GUTTER_WIDTH: f32 = 38.0;
/// Hover group shared by every cell's root element, so gutters and toolbars
/// can show/hide on cell hover regardless of cell type.
pub(crate) const CELL_HOVER_GROUP: &str = "notebook-cell";

/// Marker for the "notebook changed on disk" conflict toast, shared between
/// showing it (external change under unsaved edits) and dismissing it (any
/// reload of the notebook resolves the conflict).
struct NotebookConflictToast;
pub(crate) const CODE_BLOCK_INSET: f32 = MEDIUM_SPACING_SIZE;
pub(crate) const CONTROL_SIZE: f32 = 20.0;

pub fn init(cx: &mut App) {
    if cx.has_flag::<NotebookFeatureFlag>() || std::env::var("LOCAL_NOTEBOOK_DEV").is_ok() {
        workspace::register_project_item::<NotebookEditor>(cx);
        workspace::register_serializable_item::<NotebookEditor>(cx);
    }

    cx.observe_flag::<NotebookFeatureFlag, _>({
        move |flag, cx| {
            if *flag {
                workspace::register_project_item::<NotebookEditor>(cx);
                workspace::register_serializable_item::<NotebookEditor>(cx);
            } else {
                // todo: there is no way to unregister a project item, so if the feature flag
                // gets turned off they need to restart Zed.
            }
        }
    })
    .detach();

    cx.observe_new(|workspace: &mut Workspace, _window, _cx| {
        workspace.register_action(|workspace, _: &NewNotebook, window, cx| {
            NotebookEditor::create_new_notebook(workspace, window, cx);
        });
    })
    .detach();
}

pub struct NotebookEditor {
    languages: Arc<LanguageRegistry>,
    project: Entity<Project>,
    worktree_id: project::WorktreeId,
    focus_handle: FocusHandle,
    notebook_item: Entity<NotebookItem>,
    notebook_language: Shared<Task<Option<Arc<Language>>>>,
    cell_list: ListState,
    notebook_mode: NotebookMode,
    selected_cell_index: usize,
    cell_order: Vec<CellId>,
    original_cell_order: Vec<CellId>,
    cell_map: HashMap<CellId, Cell>,
    kernel: Kernel,
    kernel_specification: Option<KernelSpecification>,
    /// Whether the CURRENT kernel launch reached the running state. Set false
    /// when a launch begins, true once it connects. Lets `execute_cell` tell a
    /// kernel that DIED after running (e.g. the Rust/evcxr kernel exits when
    /// interrupted, erroring the kernel — not a cell) from one whose LAUNCH
    /// failed: the former relaunches the remembered spec on the next run, the
    /// latter still prompts so a broken spec doesn't relaunch-loop (bug #31).
    kernel_reached_running: bool,
    execution_requests: HashMap<String, CellId>,
    pending_executions: Vec<CellId>,
    /// Cells the user tried to run while no kernel was selected. They are held
    /// (not spinning) while the kernel picker is open: promoted to
    /// `pending_executions` if a kernel is chosen, or cleared if it's dismissed.
    cells_awaiting_kernel_choice: Vec<CellId>,
    /// Display name of an environment currently being created (venv/conda) that
    /// will become this notebook's kernel once built (phase 48). While `Some`,
    /// the top strip shows this as the selected kernel and runs are held in
    /// `cells_awaiting_kernel_choice` instead of running on the previously
    /// selected kernel. Cleared on build success (the real kernel is selected),
    /// failure, or an explicit kernel pick.
    creating_kernel_name: Option<String>,
    /// Remaining cells of a multi-cell run (Run All / Run Above / Run Below),
    /// submitted ONE at a time so a failure can stop the rest.
    run_queue: Vec<CellId>,
    /// The code cell currently executing as part of a batch; we wait for its
    /// `ExecuteReply` before submitting the next queued cell.
    active_run_cell: Option<CellId>,
    /// Structural cell operations, for undo/redo (does not cover in-cell text
    /// edits, which the cell editors undo themselves).
    undo_stack: Vec<CellEdit>,
    redo_stack: Vec<CellEdit>,
    kernel_picker_handle: PopoverMenuHandle<Picker<KernelPickerDelegate>>,
    /// The .ipynb changed on disk while there were unsaved changes here (the
    /// conflict toast was shown). While set, saving prompts before
    /// overwriting the on-disk version. Cleared on reload or confirmed save.
    disk_changed_externally: bool,
    /// Execution state (outputs / execution counts) changed since the last
    /// save. Cell text edits and structural changes are tracked separately;
    /// without this, an executed-but-unedited notebook reported itself clean
    /// and an external save would silently auto-reload over the run results.
    execution_state_changed: bool,
    /// Exactly what we last wrote to disk (or loaded from it). Used to
    /// recognize our OWN save when the file watcher reports the file changed,
    /// even if new outputs arrived in the meantime — otherwise a save during
    /// a long-running cell would raise a spurious conflict toast.
    last_saved_disk_text: Option<String>,
    /// A batch run superseded an in-flight run: we interrupted the old run and
    /// must wait for the kernel to finish aborting (return to Idle) before
    /// submitting the new queue. Submitting during the kernel's "aborting"
    /// state would get the new requests aborted too.
    resume_run_queue_on_idle: bool,
    /// When on, the notebook follows the running cell as a batch run (Run All /
    /// Above / Below) advances, so execution visibly "walks" down the notebook.
    /// What "follow" does is `repl.notebook_follow_mode`: `minimal` pins each
    /// cell near the top of the viewport and never touches the selection, while
    /// `page` moves the selection with execution and scrolls a whole page at a
    /// time (see `follow_reveal`).
    follow_running_cell: bool,
    /// Set when this notebook was restored from a session with unsaved changes
    /// that are not on disk. The ordinary dirtiness signals can't see them: the
    /// restored cells become their own baseline, so `has_structural_changes` and
    /// `has_content_changes` both read clean even though the file on disk says
    /// something else. Cleared by `mark_as_saved`, like every other dirty flag.
    restored_unsaved_changes: bool,
    /// Multi-selection: every selected index INCLUDING the primary
    /// (`selected_cell_index`). Empty when only a single cell is selected.
    /// Index-based, so any structural change collapses the selection.
    selected_indices: BTreeSet<usize>,
    /// The fixed end of a shift-range selection (the cell selection started
    /// from). `None` means the anchor is the primary cell.
    selection_anchor: Option<usize>,
    /// Execution time already banked this kernel session. Time is banked when a
    /// run ENDS, so the tally pauses between cells instead of counting the time
    /// spent writing the next one. Summing the cells' own recorded durations
    /// would not work: a re-run notebook's completed cells still carry their
    /// previous run's durations.
    execution_time_banked: Duration,
    /// Set while a cell is executing; its elapsed time is added on top of
    /// `execution_time_banked` for display and banked when the run ends.
    execution_time_started_at: Option<Instant>,
    /// When the current kernel came up, for the uptime timer.
    kernel_started_at: Option<Instant>,
    /// The final uptime of a kernel that has stopped, kept on screen until a
    /// kernel starts again.
    kernel_uptime_frozen: Option<Duration>,
    /// Repeating notify that keeps the kernel strip's timers ticking. Ends
    /// itself once no timer is visible, so an idle notebook holds no wake-up
    /// loop; `runtime_timer_ticking` says whether it is still alive.
    _runtime_timer: Option<Task<()>>,
    runtime_timer_ticking: bool,
}

impl NotebookEditor {
    pub fn new(
        project: Entity<Project>,
        notebook_item: Entity<NotebookItem>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Self {
        let focus_handle = cx.focus_handle();

        let languages = project.read(cx).languages().clone();
        // An untitled notebook has no path; fall back to the first visible
        // worktree so kernel discovery / the picker still have a scope.
        let worktree_id = notebook_item
            .read(cx)
            .project_path
            .as_ref()
            .map(|project_path| project_path.worktree_id)
            .or_else(|| {
                project
                    .read(cx)
                    .visible_worktrees(cx)
                    .next()
                    .map(|worktree| worktree.read(cx).id())
            })
            .unwrap_or_else(|| project::WorktreeId::from_usize(0));

        let notebook_language = notebook_item.read(cx).notebook_language();
        let notebook_language = cx
            .spawn_in(window, async move |_, _| notebook_language.await)
            .shared();

        let mut cell_order = vec![]; // Vec<CellId>
        let mut cell_map = HashMap::default(); // HashMap<CellId, Cell>

        let cell_count = notebook_item.read(cx).notebook.cells.len();
        for index in 0..cell_count {
            let cell = notebook_item.read(cx).notebook.cells[index].clone();
            let cell_id = cell.id();
            cell_order.push(cell_id.clone());
            let cell_entity = Cell::load(&cell, &languages, notebook_language.clone(), window, cx);

            match &cell_entity {
                Cell::Code(code_cell) => {
                    let cell_id_for_focus = cell_id.clone();
                    cx.subscribe_in(code_cell, window, move |this, _cell, event, window, cx| {
                        match event {
                            CellEvent::Run(cell_id) => {
                                this.execute_cell(cell_id.clone(), window, cx)
                            }
                            CellEvent::FocusedIn(_) => {
                                this.select_cell_by_id(&cell_id_for_focus, cx)
                            }
                            CellEvent::ToolbarAction(cell_id, action) => {
                                this.handle_cell_toolbar_action(cell_id, *action, window, cx)
                            }
                            CellEvent::Stop(cell_id) => this.handle_cell_stop(cell_id, window, cx),
                            CellEvent::ModifiedClick { id, shift } => {
                                this.handle_modified_click(id, *shift, window, cx)
                            }
                            CellEvent::PlainClick { id } => this.handle_plain_click(id, window, cx),
                            CellEvent::MetadataChanged(_) => {
                                // Collapse state persists to the .ipynb, so it
                                // counts as unsaved changes.
                                this.execution_state_changed = true;
                            }
                        }
                    })
                    .detach();

                    let cell_id_for_editor = cell_id.clone();
                    let editor = code_cell.read(cx).editor().clone();
                    cx.subscribe_in(&editor, window, move |this, _editor, event, window, cx| {
                        this.on_cell_editor_event(&cell_id_for_editor, event, window, cx);
                    })
                    .detach();
                }
                Cell::Markdown(markdown_cell) => {
                    cx.subscribe(
                        markdown_cell,
                        move |_this, cell, event: &MarkdownCellEvent, cx| {
                            match event {
                                MarkdownCellEvent::FinishedEditing => {
                                    cell.update(cx, |cell, cx| {
                                        cell.reparse_markdown(cx);
                                    });
                                }
                                MarkdownCellEvent::Run(_cell_id) => {
                                    // run is handled separately by move_to_next_cell
                                    // Just reparse here
                                    cell.update(cx, |cell, cx| {
                                        cell.reparse_markdown(cx);
                                    });
                                }
                            }
                        },
                    )
                    .detach();

                    cx.subscribe_in(
                        markdown_cell,
                        window,
                        |this, _cell, event: &CellEvent, window, cx| match event {
                            CellEvent::ModifiedClick { id, shift } => {
                                this.handle_modified_click(id, *shift, window, cx)
                            }
                            CellEvent::PlainClick { id } => this.handle_plain_click(id, window, cx),
                            _ => {}
                        },
                    )
                    .detach();

                    let cell_id_for_editor = cell_id.clone();
                    let editor = markdown_cell.read(cx).editor().clone();
                    cx.subscribe_in(&editor, window, move |this, _editor, event, window, cx| {
                        this.on_cell_editor_event(&cell_id_for_editor, event, window, cx);
                    })
                    .detach();
                }
                Cell::Raw(raw_cell) => {
                    cx.subscribe_in(
                        raw_cell,
                        window,
                        |this, _cell, event: &CellEvent, window, cx| match event {
                            CellEvent::ModifiedClick { id, shift } => {
                                this.handle_modified_click(id, *shift, window, cx)
                            }
                            CellEvent::PlainClick { id } => this.handle_plain_click(id, window, cx),
                            _ => {}
                        },
                    )
                    .detach();
                }
            }

            cell_map.insert(cell_id.clone(), cell_entity);
        }

        let cell_count = cell_order.len();

        let cell_list = ListState::new(cell_count, gpui::ListAlignment::Top, px(1000.));

        let mut editor = Self {
            project,
            languages: languages.clone(),
            worktree_id,
            focus_handle,
            notebook_item: notebook_item.clone(),
            notebook_language,
            cell_list,
            notebook_mode: NotebookMode::Command,
            selected_cell_index: 0,
            cell_order: cell_order.clone(),
            original_cell_order: cell_order.clone(),
            cell_map: cell_map.clone(),
            kernel: Kernel::Shutdown,
            kernel_specification: None,
            kernel_reached_running: false,
            execution_requests: HashMap::default(),
            pending_executions: Vec::new(),
            cells_awaiting_kernel_choice: Vec::new(),
            creating_kernel_name: None,
            run_queue: Vec::new(),
            active_run_cell: None,
            follow_running_cell: false,
            restored_unsaved_changes: false,
            undo_stack: Vec::new(),
            redo_stack: Vec::new(),
            kernel_picker_handle: PopoverMenuHandle::default(),
            disk_changed_externally: false,
            execution_state_changed: false,
            // What we loaded IS what is on disk right now (nothing is on disk
            // for an untitled notebook).
            last_saved_disk_text: notebook_item
                .read(cx)
                .buffer
                .as_ref()
                .map(|buffer| buffer.read(cx).text()),
            resume_run_queue_on_idle: false,
            selected_indices: BTreeSet::new(),
            selection_anchor: None,
            execution_time_banked: Duration::ZERO,
            execution_time_started_at: None,
            kernel_started_at: None,
            kernel_uptime_frozen: None,
            _runtime_timer: None,
            runtime_timer_ticking: false,
        };
        // Lazy start: don't launch a kernel on open. Show the remembered
        // kernel's name if we can resolve one now (a real launch happens on
        // first run or explicit selection); otherwise the status bar shows
        // "Select Kernel" until the user picks or runs a cell. The resolution
        // itself happens in adopt_metadata_kernel_selection below.
        editor.refresh_language(cx);
        editor.refresh_kernelspecs(cx);

        cx.subscribe(&notebook_item, |this, _item, _event, cx| {
            this.refresh_language(cx);
        })
        .detach();

        // Reload the notebook when its .ipynb changes on disk (the project
        // auto-reloads the backing buffer and emits `Reloaded`). Untitled
        // notebooks have no backing file yet; the watch attaches on first
        // save-as.
        if let Some(buffer) = notebook_item.read(cx).buffer.clone() {
            editor.watch_backing_buffer(buffer, window, cx);
        }

        // Pre-select the notebook's saved kernelspec once discovery delivers a
        // match, so a reopened notebook shows its kernel as selected (and
        // lazy-starts it on first run) without re-picking (phase 25). Discovery
        // is async, so try now AND whenever the store updates; an explicit
        // in-session selection always wins — this only ever fills a void.
        cx.observe_in(
            &ReplStore::global(cx),
            window,
            |this, _store, window, cx| {
                this.adopt_metadata_kernel_selection(window, cx);
            },
        )
        .detach();
        editor.adopt_metadata_kernel_selection(window, cx);

        // Keep `notebook_mode` in sync with focus: when the notebook itself
        // (not a cell editor) holds focus, we are in command mode. This avoids
        // a stuck state where the mode flag and the actually-focused element
        // disagree and single-key shortcuts stop firing.
        cx.on_focus(&editor.focus_handle, window, |this, _window, cx| {
            this.notebook_mode = NotebookMode::Command;
            cx.notify();
        })
        .detach();

        editor
    }

    fn refresh_kernelspecs(&mut self, cx: &mut Context<Self>) {
        let store = ReplStore::global(cx);
        let project = self.project.clone();
        let worktree_id = self.worktree_id;

        let refresh_task = store.update(cx, |store, cx| {
            store.refresh_python_kernelspecs(worktree_id, &project, cx)
        });

        cx.background_spawn(refresh_task).detach_and_log_err(cx);
    }

    fn refresh_language(&mut self, cx: &mut Context<Self>) {
        let notebook_language = self.notebook_item.read(cx).notebook_language();
        let task = cx.spawn(async move |this, cx| {
            let language = notebook_language.await;
            if let Some(this) = this.upgrade() {
                this.update(cx, |this, cx| {
                    for cell in this.cell_map.values() {
                        if let Cell::Code(code_cell) = cell {
                            code_cell.update(cx, |cell, cx| {
                                cell.set_language(language.clone(), cx);
                            });
                        }
                    }
                });
            }
            language
        });
        self.notebook_language = task.shared();
    }

    fn has_structural_changes(&self) -> bool {
        self.cell_order != self.original_cell_order
    }

    fn has_content_changes(&self, cx: &App) -> bool {
        self.cell_map.values().any(|cell| cell.is_dirty(cx))
    }

    pub fn to_notebook(&self, cx: &App) -> nbformat::v4::Notebook {
        let cells: Vec<nbformat::v4::Cell> = self
            .cell_order
            .iter()
            .filter_map(|cell_id| {
                self.cell_map
                    .get(cell_id)
                    .map(|cell| cell.to_nbformat_cell(cx))
            })
            .collect();

        let metadata = self.notebook_item.read(cx).notebook.metadata.clone();

        nbformat::v4::Notebook {
            metadata,
            nbformat: 4,
            nbformat_minor: 5,
            cells,
        }
    }

    pub fn mark_as_saved(&mut self, cx: &mut Context<Self>) {
        self.original_cell_order = self.cell_order.clone();
        self.execution_state_changed = false;
        self.restored_unsaved_changes = false;

        for cell in self.cell_map.values() {
            match cell {
                Cell::Code(code_cell) => {
                    code_cell.update(cx, |code_cell, cx| {
                        let editor = code_cell.editor();
                        editor.update(cx, |editor, cx| {
                            editor.buffer().update(cx, |buffer, cx| {
                                if let Some(buf) = buffer.as_singleton() {
                                    buf.update(cx, |b, cx| {
                                        let version = b.version();
                                        b.did_save(version, None, cx);
                                    });
                                }
                            });
                        });
                    });
                }
                Cell::Markdown(markdown_cell) => {
                    markdown_cell.update(cx, |markdown_cell, cx| {
                        let editor = markdown_cell.editor();
                        editor.update(cx, |editor, cx| {
                            editor.buffer().update(cx, |buffer, cx| {
                                if let Some(buf) = buffer.as_singleton() {
                                    buf.update(cx, |b, cx| {
                                        let version = b.version();
                                        b.did_save(version, None, cx);
                                    });
                                }
                            });
                        });
                    });
                }
                Cell::Raw(_) => {}
            }
        }
        // Re-serialize now that the notebook is clean, so the hot-exit copy of
        // its unsaved contents is CLEARED. Without this the next restore would
        // resurrect pre-save edits over a file that already has them: a clean
        // notebook isn't in the close-time dirty set, so nothing else would
        // rewrite that row. (This is the one thing `Editor` gets from having
        // `EditorEvent::Saved` in its `should_serialize` set; the notebook's
        // single `()` event has to be emitted deliberately.)
        cx.emit(());
        cx.notify();
    }

    /// The kernel the notebook should use without any explicit choice yet:
    /// the active in-session selection, this NOTEBOOK's remembered pick, or
    /// one matching the notebook's saved metadata. All per-notebook (bug #30)
    /// — the worktree-level selection belongs to the inline REPL. Deliberately
    /// does NOT fall back to the "recommended"/global kernel — an unremembered
    /// notebook should prompt rather than silently start the wrong interpreter.
    fn remembered_kernel_spec(&self, cx: &App) -> Option<KernelSpecification> {
        // A remembered env that vanished from disk (deleted mid-session) must
        // never be silently launched or adopted — resolution treats it as
        // "nothing remembered" so every consumer prompts instead (phase 42).
        self.remembered_kernel_spec_any(cx)
            .filter(|spec| !Self::spec_interpreter_missing(spec))
    }

    /// The raw remembered-kernel resolution, WITHOUT the existence check —
    /// only for detecting a stale selection that needs discarding.
    fn remembered_kernel_spec_any(&self, cx: &App) -> Option<KernelSpecification> {
        if let Some(spec) = &self.kernel_specification {
            return Some(spec.clone());
        }
        if let Some(notebook_path) = &self.notebook_item.read(cx).path
            && let Some(spec) = ReplStore::global(cx)
                .read(cx)
                .notebook_kernelspec(notebook_path)
        {
            return Some(spec.clone());
        }
        self.metadata_matched_kernel_spec(cx)
    }

    /// Whether a locally-launched spec's interpreter has vanished (its env
    /// was deleted). A cheap stat at resolution time — no polling. Remote /
    /// WSL / SSH specs are never checked. Jupyter kernelspecs with a relative
    /// argv (e.g. plain "python") are trusted; registered venv kernelspecs
    /// use absolute paths, which are the ones that go stale.
    fn spec_interpreter_missing(spec: &KernelSpecification) -> bool {
        let interpreter = match spec {
            KernelSpecification::PythonEnv(env) => Some(env.path.clone()),
            KernelSpecification::Jupyter(local) => local
                .kernelspec
                .argv
                .first()
                .map(PathBuf::from)
                .filter(|path| path.is_absolute()),
            _ => None,
        };
        interpreter.is_some_and(|path| !path.exists())
    }

    /// Drop a kernel selection whose environment no longer exists, and kick a
    /// re-discovery so the picker/indicator stop pointing at a ghost env and
    /// a recreated same-name env resolves to its NEW interpreter (phase 42).
    fn discard_stale_kernel_selection(&mut self, cx: &mut Context<Self>) {
        self.kernel_specification = None;
        if let Some(path) = self.notebook_item.read(cx).path.clone() {
            ReplStore::global(cx).update(cx, |store, _| {
                store.clear_notebook_kernelspec(&path);
            });
        }
        self.refresh_kernelspecs(cx);
        ReplStore::global(cx).update(cx, |store, cx| {
            store.refresh_kernelspecs(cx).detach_and_log_err(cx);
        });
        cx.notify();
    }

    /// The discovered kernel matching the notebook's saved
    /// `metadata.kernelspec.name`, if any. Only kernels that are safe to start
    /// SILENTLY are eligible: remote-server specs are excluded (a run must
    /// never leave the machine without an explicit pick), as are Python envs
    /// missing ipykernel (the picker greys those out with a warning — silently
    /// adopting one would loop through launch failures without ever showing
    /// that warning).
    fn metadata_matched_kernel_spec(&self, cx: &App) -> Option<KernelSpecification> {
        let kernelspec = self
            .notebook_item
            .read(cx)
            .notebook
            .metadata
            .kernelspec
            .as_ref()?;
        let name = kernelspec.name.clone();
        ReplStore::global(cx)
            .read(cx)
            .kernel_specifications_for_worktree(self.worktree_id)
            .find(|spec| {
                spec.name().as_ref() == name
                    && spec.has_ipykernel()
                    && !matches!(
                        spec,
                        KernelSpecification::JupyterServer(_) | KernelSpecification::SshRemote(_)
                    )
            })
            .cloned()
    }

    /// If nothing has been selected this session, adopt the kernel this
    /// notebook remembers (its own in-session pick, else the kernel matching
    /// its saved `kernelspec` metadata) for DISPLAY (phase 25): the status bar
    /// shows it and the first run lazy-starts it, VS Code style — no
    /// re-picking after a full restart. Discovery is async, so this is called
    /// on open AND from a store observer; it no-ops until a matching spec is
    /// discovered, and a stale metadata name simply never matches.
    ///
    /// Deliberately PER-NOTEBOOK (bug #30): nothing here touches the
    /// worktree-level selection, which belongs to the inline REPL. Each
    /// notebook resolves its own pick/metadata, so neither opening nor picking
    /// in one notebook changes what a sibling notebook runs.
    fn adopt_metadata_kernel_selection(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if self.kernel_specification.is_some() {
            return;
        }
        if let Some(spec) = self.remembered_kernel_spec(cx) {
            log::info!(
                "notebook: pre-selected kernel '{}' (saved selection/metadata)",
                spec.name()
            );
            self.kernel_specification = Some(spec.clone());
            // Opt-in autostart (phase 34): launch the adopted kernel on open
            // instead of waiting for the first run. Only from a clean Shutdown
            // — never after a failed launch (bug #31's re-prompt rule), and
            // never a kernel that is already starting/running. Remote and
            // ipykernel-less specs were already excluded by the matcher.
            if ReplSettings::get_global(cx).notebook_autostart_kernel
                && matches!(self.kernel, Kernel::Shutdown)
            {
                log::info!("notebook: autostarting kernel '{}'", spec.name());
                self.launch_kernel_with_spec(spec, window, cx);
            }
            cx.notify();
        }
    }

    /// Launch the remembered kernel, or prompt for one if none is remembered.
    fn launch_kernel(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if let Some(spec) = self.remembered_kernel_spec(cx) {
            self.launch_kernel_with_spec(spec, window, cx);
        } else {
            // Nothing selected or remembered: prompt the user to choose a
            // kernel. Any cell that triggered this is already queued and will
            // run once a kernel is picked and ready. Deferred because `show`
            // fires the picker's `on_open`, which updates this notebook — and
            // this runs inside an update already (see the same fix elsewhere).
            let kernel_picker_handle = self.kernel_picker_handle.clone();
            window.defer(cx, move |window, cx| {
                kernel_picker_handle.show(window, cx);
            });
        }
    }

    /// The "Create Python Environment" flow from the kernel picker. Offers a
    /// workspace `.venv`, a venv at a chosen location, and — when a conda
    /// frontend is on PATH — a named conda environment. Conda detection is
    /// async, so the whole flow (detect → prompt → act) runs in one task.
    fn create_python_environment(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        self.kernel_picker_handle.hide(cx);

        let worktree_root = self
            .project
            .read(cx)
            .worktree_for_id(self.worktree_id, cx)
            .map(|worktree| worktree.read(cx).abs_path().to_path_buf());

        cx.spawn_in(window, async move |this, cx| {
            let conda = detect_conda_frontend().await;

            if worktree_root.is_none() && conda.is_none() {
                this.update_in(cx, |_this, window, cx| {
                    Self::show_env_toast(
                        window,
                        cx,
                        "Cannot create a Python environment: no project folder is open, and \
                         no conda/mamba/micromamba was found on PATH."
                            .to_string(),
                        false,
                    );
                })
                .ok();
                return anyhow::Ok(());
            }

            // The fast path stays one keypress — the first button (Enter)
            // creates the workspace `.venv`. Options only appear when they can
            // work: the venv options need a project folder; the conda option
            // needs a conda frontend.
            #[derive(Clone, Copy)]
            enum EnvChoice {
                Venv,
                ChooseLocation,
                Conda,
            }
            let mut labels: Vec<&str> = Vec::new();
            let mut choices: Vec<EnvChoice> = Vec::new();
            if worktree_root.is_some() {
                labels.push("Create .venv");
                choices.push(EnvChoice::Venv);
                labels.push("Choose Location…");
                choices.push(EnvChoice::ChooseLocation);
            }
            if conda.is_some() {
                labels.push("Create Conda Env…");
                choices.push(EnvChoice::Conda);
            }
            labels.push("Cancel");

            let detail = match (worktree_root.is_some(), conda.is_some()) {
                (true, true) => {
                    "\"Create .venv\" creates it in the project root. \
                     \"Choose Location…\" makes the selected folder the environment. \
                     \"Create Conda Env…\" creates a named conda environment."
                }
                (true, false) => {
                    "\"Create .venv\" creates it in the project root. \
                     \"Choose Location…\" makes the selected folder the environment."
                }
                _ => "\"Create Conda Env…\" creates a named conda environment.",
            };

            let answer = this.update_in(cx, |_this, window, cx| {
                window.prompt(
                    PromptLevel::Info,
                    "Create a Python environment?",
                    Some(detail),
                    &labels,
                    cx,
                )
            })?;
            let index = answer.await?;
            let Some(choice) = choices.get(index).copied() else {
                return anyhow::Ok(());
            };

            match choice {
                EnvChoice::Venv => {
                    let Some(root) = worktree_root.clone() else {
                        return anyhow::Ok(());
                    };
                    this.update_in(cx, |this, window, cx| {
                        this.create_python_environment_at(
                            root.join(".venv"),
                            ".venv".to_string(),
                            window,
                            cx,
                        );
                    })
                    .ok();
                }
                EnvChoice::ChooseLocation => {
                    let paths = cx.update(|_, cx| {
                        cx.prompt_for_paths(gpui::PathPromptOptions {
                            files: false,
                            directories: true,
                            multiple: false,
                            prompt: Some("Use as Environment".into()),
                        })
                    })?;
                    if let Ok(Ok(Some(mut paths))) = paths.await
                        && let Some(env_dir) = paths.pop()
                    {
                        let env_name = env_dir
                            .file_name()
                            .map(|name| name.to_string_lossy().to_string())
                            .unwrap_or_else(|| "venv".to_string());
                        this.update_in(cx, |this, window, cx| {
                            this.create_python_environment_at(env_dir, env_name, window, cx);
                        })
                        .ok();
                    }
                }
                EnvChoice::Conda => {
                    let Some(frontend) = conda else {
                        return anyhow::Ok(());
                    };
                    this.update_in(cx, |this, window, cx| {
                        this.create_conda_environment(frontend, window, cx);
                    })
                    .ok();
                }
            }
            anyhow::Ok(())
        })
        .detach_and_log_err(cx);
    }

    /// Prompt for a conda environment name (modal), then create it.
    fn create_conda_environment(
        &mut self,
        frontend: &'static str,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let Some(workspace) = Workspace::for_window(window, cx) else {
            Self::show_env_toast(
                window,
                cx,
                "Cannot create a conda environment: no workspace is open.".to_string(),
                false,
            );
            return;
        };

        let (tx, rx) = oneshot::channel::<String>();
        workspace.update(cx, |workspace, cx| {
            workspace.toggle_modal(window, cx, |window, cx| {
                EnvNameModal::new(
                    "New Conda Environment",
                    "Creates a conda environment with Python and ipykernel, then selects it.",
                    "environment name",
                    tx,
                    window,
                    cx,
                )
            });
        });

        cx.spawn_in(window, async move |this, cx| {
            if let Ok(name) = rx.await {
                this.update_in(cx, |this, window, cx| {
                    this.create_conda_environment_named(frontend, name, window, cx);
                })
                .ok();
            }
            anyhow::Ok(())
        })
        .detach_and_log_err(cx);
    }

    /// Create a named conda environment (`<frontend> create -y -n <name>
    /// python ipykernel`), resolve its interpreter via the frontend itself,
    /// register a kernelspec, and select it.
    fn create_conda_environment_named(
        &mut self,
        frontend: &'static str,
        name: String,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        // Make the env being built the notebook's pending selection so runs
        // queue for it instead of the previously selected kernel (phase 48).
        self.begin_creating_kernel(name.clone(), cx);

        struct CreateCondaEnv;
        let notification_id = NotificationId::unique::<CreateCondaEnv>();
        let workspace = Workspace::for_window(window, cx);
        if let Some(workspace) = &workspace {
            workspace.update(cx, |workspace, cx| {
                workspace.show_toast(
                    workspace::Toast::new(
                        notification_id.clone(),
                        format!(
                            "Creating conda environment {name} \
                             (installing Python + ipykernel; this can take a while)…"
                        ),
                    ),
                    cx,
                );
            });
        }
        let weak_workspace = workspace.map(|workspace| workspace.downgrade());

        let create_task = cx.background_spawn({
            let name = name.clone();
            async move {
                // `conda create` is a solver + download; failures (solver
                // conflicts, no channel) come back on stderr.
                let mut command = util::command::new_command(frontend);
                command.args(["create", "-y", "-n", &name, "python", "ipykernel"]);
                // micromamba ships with no default channels, so it needs one
                // to resolve `python`; conda/mamba use the user's configured
                // channels.
                if frontend == "micromamba" {
                    command.args(["-c", "conda-forge"]);
                }
                let output = command
                    .output()
                    .await
                    .with_context(|| format!("failed to run {frontend} create"))?;
                anyhow::ensure!(
                    output.status.success(),
                    "{frontend} create failed: {}",
                    String::from_utf8_lossy(&output.stderr)
                        .lines()
                        .last()
                        .unwrap_or("unknown error")
                );

                // Resolve the interpreter through the frontend rather than
                // guessing the per-platform layout (`envs/<name>/bin/python`
                // vs `envs\<name>\python.exe`).
                let python_output = util::command::new_command(frontend)
                    .args([
                        "run",
                        "-n",
                        &name,
                        "python",
                        "-c",
                        "import sys; print(sys.executable)",
                    ])
                    .output()
                    .await
                    .with_context(|| format!("failed to resolve the {name} interpreter"))?;
                anyhow::ensure!(
                    python_output.status.success(),
                    "could not resolve the {name} interpreter: {}",
                    String::from_utf8_lossy(&python_output.stderr)
                        .lines()
                        .last()
                        .unwrap_or("unknown error")
                );
                let python_path = PathBuf::from(
                    String::from_utf8_lossy(&python_output.stdout)
                        .trim()
                        .to_string(),
                );
                anyhow::ensure!(
                    !python_path.as_os_str().is_empty(),
                    "conda returned an empty interpreter path for {name}"
                );

                // Conda envs live outside the worktree, so register a
                // kernelspec for discovery next session (as the venv flow does
                // for out-of-worktree envs). Non-fatal on failure.
                let kernel_name = sanitize_kernel_name(&name);
                let result = util::command::new_command(python_path.to_string_lossy().as_ref())
                    .args(["-m", "ipykernel", "install", "--user", "--name"])
                    .arg(&kernel_name)
                    .arg("--display-name")
                    .arg(format!("Python ({name})"))
                    .output()
                    .await;
                let registration_warning = match result {
                    Ok(output) if output.status.success() => None,
                    Ok(output) => Some(
                        String::from_utf8_lossy(&output.stderr)
                            .lines()
                            .last()
                            .unwrap_or("unknown error")
                            .to_string(),
                    ),
                    Err(error) => Some(error.to_string()),
                };

                anyhow::Ok((python_path, registration_warning))
            }
        });

        self.finalize_env_creation(
            create_task,
            name,
            Some("Conda".to_string()),
            notification_id,
            weak_workspace,
            window,
            cx,
        );
    }

    /// Create (or reuse) a Python venv at `venv_dir`, install ipykernel into
    /// it, and select it as this notebook's kernel.
    fn create_python_environment_at(
        &mut self,
        venv_dir: PathBuf,
        env_name: String,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let fs = self.project.read(cx).fs().clone();
        let venv_python = if cfg!(windows) {
            venv_dir.join("Scripts").join("python.exe")
        } else {
            venv_dir.join("bin").join("python")
        };

        // An env outside the worktree is invisible to toolchain discovery on
        // the next start (bug #36), so register it as a real Jupyter
        // kernelspec, which discovery does find. Workspace `.venv`s are
        // discovered directly and skip this to avoid polluting the per-user
        // kernelspec list.
        let register_kernelspec = self
            .project
            .read(cx)
            .worktree_for_id(self.worktree_id, cx)
            .map(|worktree| worktree.read(cx).abs_path().to_path_buf())
            .is_none_or(|root| !venv_dir.starts_with(&root));

        // Make the env being built the notebook's pending selection so runs
        // queue for it instead of the previously selected kernel (phase 48).
        self.begin_creating_kernel(env_name.clone(), cx);

        struct CreatePythonEnv;
        let notification_id = NotificationId::unique::<CreatePythonEnv>();
        let workspace = Workspace::for_window(window, cx);
        if let Some(workspace) = &workspace {
            workspace.update(cx, |workspace, cx| {
                workspace.show_toast(
                    workspace::Toast::new(
                        notification_id.clone(),
                        format!("Creating {env_name} and installing ipykernel…"),
                    ),
                    cx,
                );
            });
        }
        let weak_workspace = workspace.map(|workspace| workspace.downgrade());

        let create_task = cx.background_spawn({
            let env_name = env_name.clone();
            async move {
            // Create the venv unless one already exists (reuse it if so).
            if !fs.is_file(&venv_python).await {
                let mut last_error = String::new();
                let mut created = false;
                for base_python in ["python3", "python"] {
                    match util::command::new_command(base_python)
                        .arg("-m")
                        .arg("venv")
                        .arg(&venv_dir)
                        .output()
                        .await
                    {
                        Ok(output) if output.status.success() => {
                            created = true;
                            break;
                        }
                        Ok(output) => {
                            last_error = String::from_utf8_lossy(&output.stderr)
                                .lines()
                                .last()
                                .unwrap_or("")
                                .to_string();
                        }
                        Err(error) => last_error = error.to_string(),
                    }
                }
                anyhow::ensure!(
                    created,
                    "could not create the environment (is Python installed and on PATH?): {last_error}"
                );
            }

            let output = util::command::new_command(venv_python.to_string_lossy().as_ref())
                .args(["-m", "pip", "install", "ipykernel"])
                .output()
                .await
                .context("failed to run pip install ipykernel")?;
            anyhow::ensure!(
                output.status.success(),
                "failed to install ipykernel: {}",
                String::from_utf8_lossy(&output.stderr)
                    .lines()
                    .last()
                    .unwrap_or("unknown error")
            );

            // Registration failure is non-fatal — the env still works for
            // this session — but the user must know it won't be listed next
            // start.
            let mut registration_warning = None;
            if register_kernelspec {
                let kernel_name = sanitize_kernel_name(&env_name);
                let result = util::command::new_command(venv_python.to_string_lossy().as_ref())
                    .args(["-m", "ipykernel", "install", "--user", "--name"])
                    .arg(&kernel_name)
                    .arg("--display-name")
                    .arg(format!("Python ({env_name})"))
                    .output()
                    .await;
                registration_warning = match result {
                    Ok(output) if output.status.success() => None,
                    Ok(output) => Some(
                        String::from_utf8_lossy(&output.stderr)
                            .lines()
                            .last()
                            .unwrap_or("unknown error")
                            .to_string(),
                    ),
                    Err(error) => Some(error.to_string()),
                };
            }

            anyhow::Ok((venv_python, registration_warning))
        }});

        self.finalize_env_creation(
            create_task,
            env_name,
            Some("venv".to_string()),
            notification_id,
            weak_workspace,
            window,
            cx,
        );
    }

    /// Shared tail of the env-creation flows (venv and conda): await the
    /// background create task, surface success/failure toasts, then build the
    /// kernel spec from the resolved interpreter, select it, and refresh the
    /// kernelspec list. `create_task` yields `(interpreter_path,
    /// registration_warning)`.
    fn finalize_env_creation(
        &mut self,
        create_task: Task<Result<(PathBuf, Option<String>)>>,
        env_name: String,
        environment_kind: Option<String>,
        notification_id: NotificationId,
        weak_workspace: Option<WeakEntity<Workspace>>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        cx.spawn_in(window, async move |this, cx| {
            match create_task.await {
                Ok((python_path, registration_warning)) => {
                    if let Some(weak_workspace) = &weak_workspace {
                        weak_workspace
                            .update(cx, |workspace, cx| {
                                workspace.dismiss_toast(&notification_id, cx);
                                let (message, autohide) = match &registration_warning {
                                    Some(warning) => (
                                        format!(
                                            "Created {env_name}, but couldn't register its \
                                             kernel for future sessions: {warning}"
                                        ),
                                        false,
                                    ),
                                    None => (
                                        format!("Created {env_name} and installed ipykernel"),
                                        true,
                                    ),
                                };
                                let toast = workspace::Toast::new(notification_id.clone(), message);
                                let toast = if autohide { toast.autohide() } else { toast };
                                workspace.show_toast(toast, cx);
                            })
                            .ok();
                    }
                    this.update_in(cx, |this, window, cx| {
                        // Only auto-select the built env if it's still the
                        // pending "creating" selection — the user may have
                        // picked a different kernel while it built (phase 48).
                        let still_ours =
                            this.creating_kernel_name.as_deref() == Some(env_name.as_str());
                        this.creating_kernel_name = None;
                        this.refresh_kernelspecs(cx);
                        if still_ours {
                            let spec = KernelSpecification::PythonEnv(
                                PythonEnvKernelSpecification::from_python_path(
                                    python_path,
                                    env_name.clone(),
                                    true,
                                    environment_kind.clone(),
                                ),
                            );
                            // Promotes the cells held during the build onto the
                            // new kernel (change_kernel → promote_awaiting_cells).
                            this.change_kernel(spec, window, cx);
                        }
                        cx.notify();
                    })
                    .ok();
                }
                Err(error) => {
                    if let Some(weak_workspace) = &weak_workspace {
                        weak_workspace
                            .update(cx, |workspace, cx| {
                                workspace.dismiss_toast(&notification_id, cx);
                                workspace.show_toast(
                                    workspace::Toast::new(
                                        notification_id.clone(),
                                        format!("Failed to create {env_name}: {error}"),
                                    ),
                                    cx,
                                );
                            })
                            .ok();
                    }
                    // Drop the "creating" selection and release the cells held
                    // for it back to Idle (unless the user already moved on to
                    // another kernel, which cleared the flag).
                    this.update_in(cx, |this, _window, cx| {
                        if this.creating_kernel_name.as_deref() == Some(env_name.as_str()) {
                            this.creating_kernel_name = None;
                            this.clear_awaiting_cells(cx);
                            cx.notify();
                        }
                    })
                    .ok();
                }
            }
        })
        .detach();
    }

    /// Enter the "creating a kernel" state (phase 48): the named env becomes
    /// this notebook's pending selection, so the top strip shows it and
    /// `execute_cell` holds runs until the build finishes (rather than running
    /// on the previously selected kernel).
    fn begin_creating_kernel(&mut self, name: String, cx: &mut Context<Self>) {
        self.creating_kernel_name = Some(name);
        cx.notify();
    }

    fn show_env_toast(
        window: &mut Window,
        cx: &mut Context<Self>,
        message: String,
        autohide: bool,
    ) {
        struct NotebookToast;
        let notification_id = NotificationId::unique::<NotebookToast>();
        if let Some(workspace) = Workspace::for_window(window, cx) {
            workspace.update(cx, |workspace, cx| {
                let toast = workspace::Toast::new(notification_id, message);
                let toast = if autohide { toast.autohide() } else { toast };
                workspace.show_toast(toast, cx);
            });
        }
    }

    /// A single empty code cell with a fresh id, used to seed a new or empty
    /// notebook so it opens with something to type into.
    fn empty_code_cell() -> nbformat::v4::Cell {
        nbformat::v4::Cell::Code {
            id: Uuid::new_v4().into(),
            metadata: Self::empty_cell_metadata(),
            execution_count: None,
            source: Vec::new(),
            outputs: Vec::new(),
        }
    }

    /// A minimal valid nbformat v4 notebook containing one empty code cell.
    /// Used when opening an empty `.ipynb` and by the "New Jupyter Notebook"
    /// command, mirroring VS Code (a new notebook is never truly empty).
    fn empty_notebook() -> Result<nbformat::v4::Notebook> {
        Ok(nbformat::v4::Notebook {
            nbformat: 4,
            nbformat_minor: 5,
            cells: vec![Self::empty_code_cell()],
            metadata: serde_json::from_str("{}")?,
        })
    }

    /// Open an untitled, session-only notebook seeded with the one-cell
    /// template (phase 33). Nothing touches disk until the first save, which
    /// routes through the save-as prompt. Notebooks created via the file
    /// browser's New File are unaffected — they are real files from creation.
    fn create_new_notebook(
        workspace: &mut Workspace,
        window: &mut Window,
        cx: &mut Context<Workspace>,
    ) {
        // Only meaningful when notebooks are enabled (the `.ipynb` project item
        // is registered under the same gate); otherwise the file would open as
        // raw JSON.
        if !cx.has_flag::<NotebookFeatureFlag>() && std::env::var("LOCAL_NOTEBOOK_DEV").is_err() {
            return;
        }

        let project = workspace.project().clone();

        let template = match Self::empty_notebook() {
            Ok(notebook) => notebook,
            Err(error) => {
                log::error!("notebook: failed to build the new-notebook template: {error}");
                return;
            }
        };

        let languages = project.read(cx).languages().clone();
        let notebook_item =
            cx.new(|_| NotebookItem::untitled(project.downgrade(), languages, template));
        let editor = cx.new(|cx| NotebookEditor::new(project, notebook_item, window, cx));
        workspace.add_item_to_active_pane(Box::new(editor), None, true, window, cx);
    }

    /// Parse `.ipynb` text into a v4 notebook, tolerating empty files, missing
    /// cell IDs, and legacy formats. Shared by open, reload, and external
    /// change handling. An empty/whitespace file yields a one-cell template so
    /// it opens as a usable notebook rather than a blank pane.
    fn parse_notebook_text(text: &str) -> Result<nbformat::v4::Notebook> {
        if text.trim().is_empty() {
            return Self::empty_notebook();
        }

        let parsed = match nbformat::parse_notebook(text) {
            Ok(notebook) => notebook,
            Err(_) => {
                // Pre-process to ensure cell IDs exist, then re-parse.
                let mut json: serde_json::Value = serde_json::from_str(text)?;
                if let Some(cells) = json.get_mut("cells").and_then(|c| c.as_array_mut()) {
                    for cell in cells {
                        if cell.get("id").is_none() {
                            cell["id"] = serde_json::Value::String(Uuid::new_v4().to_string());
                        }
                    }
                }
                nbformat::parse_notebook(&serde_json::to_string(&json)?)?
            }
        };

        Ok(match parsed {
            nbformat::Notebook::V4(notebook) => notebook,
            nbformat::Notebook::Legacy(legacy) => nbformat::upgrade_legacy_notebook(legacy)?,
            nbformat::Notebook::V3(v3) => nbformat::upgrade_v3_notebook(v3)?,
        })
    }

    /// Rebuild the notebook's cells from `notebook`, wiring up subscriptions
    /// and resetting execution state. Used by reload and external-change
    /// handling (NOT the initial `new`, which wires cells inline).
    fn reload_cells_from_notebook(
        &mut self,
        notebook: &nbformat::v4::Notebook,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let languages = self.languages.clone();
        let notebook_language = self.notebook_language.clone();

        let mut cell_order = Vec::new();
        let mut cell_map = HashMap::default();
        for cell in notebook.cells.iter() {
            let cell_id = cell.id();
            cell_order.push(cell_id.clone());
            let cell_entity = Cell::load(cell, &languages, notebook_language.clone(), window, cx);
            match &cell_entity {
                Cell::Code(code_cell) => {
                    self.wire_code_cell(cell_id.clone(), code_cell, window, cx)
                }
                Cell::Markdown(markdown_cell) => {
                    self.wire_markdown_cell(cell_id.clone(), markdown_cell, window, cx)
                }
                Cell::Raw(_) => {}
            }
            cell_map.insert(cell_id.clone(), cell_entity);
        }

        // Reset execution/queue state — the previous cells no longer exist.
        self.execution_requests.clear();
        self.pending_executions.clear();
        self.cells_awaiting_kernel_choice.clear();
        self.cancel_run_queue(cx);
        // We now reflect the on-disk content, so any prior conflict is moot:
        // clear the flags and take down the conflict toast (reloading from the
        // command palette must dismiss it too, not just the toast's button).
        self.disk_changed_externally = false;
        self.execution_state_changed = false;
        self.dismiss_conflict_toast(window, cx);

        self.cell_order = cell_order.clone();
        self.original_cell_order = cell_order;
        self.cell_map = cell_map;
        self.selected_cell_index = 0;
        self.notebook_mode = NotebookMode::Command;
        self.cell_list = ListState::new(self.cell_order.len(), gpui::ListAlignment::Top, px(1000.));

        self.notebook_item.update(cx, |item, _| {
            item.notebook = notebook.clone();
        });
        self.refresh_language(cx);
        cx.notify();
    }

    /// Put back the unsaved changes a previous session was holding, over the
    /// copy just loaded from disk (phase 69's hot exit).
    ///
    /// The cells are loaded exactly as a reload would load them, then the
    /// notebook is marked dirty explicitly: the restored cells ARE the baseline
    /// afterwards, so nothing else would report the difference from disk, and
    /// without that the tab would look clean, closing wouldn't prompt, and the
    /// next serialize would drop the very changes we just recovered.
    ///
    /// `saved_mtime` is what the file had when the session ended. If disk has
    /// moved on since, someone edited the notebook while Zed was closed, and
    /// these restored changes are no longer based on the current file — that is
    /// exactly the conflict `disk_changed_externally` exists for (bug #14), so
    /// saving will ask before overwriting instead of silently winning.
    fn restore_unsaved_notebook(
        &mut self,
        unsaved: &nbformat::v4::Notebook,
        saved_mtime: Option<MTime>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let disk_mtime = self
            .notebook_item
            .read(cx)
            .buffer
            .as_ref()
            .and_then(|buffer| buffer.read(cx).saved_mtime());
        self.reload_cells_from_notebook(unsaved, window, cx);
        self.restored_unsaved_changes = true;
        // Order matters: `reload_cells_from_notebook` clears the conflict flag
        // because it normally means "we now match disk", which is the opposite
        // of what just happened here.
        if let (Some(saved_mtime), Some(disk_mtime)) = (saved_mtime, disk_mtime)
            && saved_mtime != disk_mtime
        {
            self.disk_changed_externally = true;
        }
    }

    /// Take down the "notebook changed on disk" conflict toast. Called by any
    /// path that re-aligns us with disk — reload (command or toast button) and
    /// a confirmed overwrite save.
    fn dismiss_conflict_toast(&self, window: &mut Window, cx: &mut Context<Self>) {
        if let Some(workspace) = Workspace::for_window(window, cx) {
            workspace.update(cx, |workspace, cx| {
                workspace.dismiss_toast(&NotificationId::unique::<NotebookConflictToast>(), cx);
            });
        }
    }

    /// Watch the notebook's backing project buffer so external .ipynb changes
    /// reload it. Called at open for file-backed notebooks, and after the
    /// first save-as of an untitled one.
    fn watch_backing_buffer(
        &mut self,
        buffer: Entity<Buffer>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        cx.subscribe_in(&buffer, window, |this, buffer, event, window, cx| {
            match event {
                language::BufferEvent::Reloaded => this.handle_external_change(buffer, window, cx),
                // The file was deleted, recreated or renamed under us. The disk
                // state lives on the buffer's `File`, which we don't own, so the
                // only thing to do is repaint: `tab_content` re-reads it, and the
                // emitted event tells the pane the tab changed so the close
                // prompt and `close_on_file_delete` see it too.
                language::BufferEvent::FileHandleChanged => {
                    cx.emit(());
                    cx.notify();
                }
                _ => {}
            }
        })
        .detach();
    }

    /// The .ipynb changed on disk (the project auto-reloaded the backing
    /// buffer). Rebuild from the new content unless there are unsaved changes,
    /// in which case keep them and warn.
    fn handle_external_change(
        &mut self,
        buffer: &Entity<Buffer>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let disk_text = buffer.read(cx).text();

        // Fast path: byte-identical to what we last wrote — our own save
        // landing back via the file watcher.
        if let Some(last_saved) = &self.last_saved_disk_text
            && disk_text.trim() == last_saved.trim()
        {
            return;
        }

        // Parse the new on-disk content. Leave our in-memory state untouched if
        // it doesn't parse rather than raising a conflict over unreadable JSON.
        let disk_notebook = match Self::parse_notebook_text(&disk_text) {
            Ok(notebook) => notebook,
            Err(error) => {
                log::warn!("notebook: failed to parse externally-changed .ipynb: {error}");
                return;
            }
        };

        // Authoritative own-save / no-op guard: compare CONTENT, not text.
        // A save writes `to_string_pretty(to_notebook())`, but the buffer the
        // file watcher reloads can differ byte-for-byte from what we wrote
        // (final-newline handling, CRLF vs LF, JSON map key ordering) while
        // being semantically identical. Comparing the parsed structures as
        // JSON values ignores all of that — key ordering included — so our own
        // writes (e.g. a metadata-only collapse save) never raise a spurious
        // "changed on disk" conflict. Also take down any stale conflict toast
        // now that we're re-aligned with disk.
        let same_as_memory = match (
            serde_json::to_value(&disk_notebook),
            serde_json::to_value(self.to_notebook(cx)),
        ) {
            (Ok(disk), Ok(memory)) => disk == memory,
            _ => false,
        };
        if same_as_memory {
            self.disk_changed_externally = false;
            self.last_saved_disk_text = Some(disk_text);
            self.dismiss_conflict_toast(window, cx);
            return;
        }

        // A genuine external change. If we have unsaved edits, keep them and
        // warn rather than clobbering them with the on-disk version.
        if self.is_dirty(cx) {
            self.disk_changed_externally = true;
            let notification_id = NotificationId::unique::<NotebookConflictToast>();
            let this = cx.entity().downgrade();
            let project = self.project.clone();
            if let Some(workspace) = Workspace::for_window(window, cx) {
                workspace.update(cx, |workspace, cx| {
                    workspace.show_toast(
                        workspace::Toast::new(
                            notification_id,
                            "This notebook changed on disk, but you have unsaved changes \
                             here. Saving will ask before overwriting the on-disk version.",
                        )
                        .on_click(
                            "Reload (discard my changes)",
                            move |window, cx| {
                                this.update(cx, |this, cx| {
                                    this.reload(project.clone(), window, cx)
                                        .detach_and_log_err(cx);
                                })
                                .log_err();
                            },
                        ),
                        cx,
                    );
                });
            }
            return;
        }

        // No local edits — adopt the on-disk version.
        self.reload_cells_from_notebook(&disk_notebook, window, cx);
        self.last_saved_disk_text = Some(disk_text);
    }

    fn launch_kernel_with_spec(
        &mut self,
        spec: KernelSpecification,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let entity_id = cx.entity_id();
        let working_directory = self
            .project
            .read(cx)
            .worktree_for_id(self.worktree_id, cx)
            .map(|worktree| {
                let worktree = worktree.read(cx);
                let root = worktree.abs_path().to_path_buf();
                // A notebook opened or saved OUTSIDE the project lives in a
                // single-file worktree whose root is the FILE, not a directory.
                // Spawning the kernel with a file as its working directory
                // fails (on Windows: "The directory name is invalid",
                // os error 267), so use the containing directory instead.
                if worktree.is_single_file() {
                    root.parent()
                        .map(|parent| parent.to_path_buf())
                        .unwrap_or(root)
                } else {
                    root
                }
            })
            .unwrap_or_else(std::env::temp_dir);
        let fs = self.project.read(cx).fs().clone();
        // Weak: the kernel's tasks must not keep this editor (and therefore
        // the kernel process) alive after the tab closes (bug #35).
        let view = cx.entity().downgrade();

        // Both runtime timers are scoped to a kernel session, so a launch
        // (including a restart or a switch to another kernel) starts them over.
        self.reset_runtime_timers();
        self.kernel_specification = Some(spec.clone());

        self.notebook_item.update(cx, |item, cx| {
            let kernel_name = spec.name().to_string();
            let language = spec.language().to_string();

            let display_name = match &spec {
                KernelSpecification::Jupyter(s) => s.kernelspec.display_name.clone(),
                KernelSpecification::PythonEnv(s) => s.kernelspec.display_name.clone(),
                KernelSpecification::JupyterServer(s) => s.kernelspec.display_name.clone(),
                KernelSpecification::SshRemote(s) => s.kernelspec.display_name.clone(),
                KernelSpecification::WslRemote(s) => s.kernelspec.display_name.clone(),
            };

            let kernelspec_json = serde_json::json!({
                "display_name": display_name,
                "name": kernel_name,
                "language": language
            });

            if let Ok(k) = serde_json::from_value(kernelspec_json) {
                item.notebook.metadata.kernelspec = Some(k);
                cx.emit(());
            }
        });

        let kernel_task = match spec {
            KernelSpecification::Jupyter(local_spec) => NativeRunningKernel::new(
                local_spec,
                entity_id,
                working_directory,
                fs,
                view,
                window,
                cx,
            ),
            KernelSpecification::PythonEnv(env_spec) => NativeRunningKernel::new(
                env_spec.as_local_spec(),
                entity_id,
                working_directory,
                fs,
                view,
                window,
                cx,
            ),
            KernelSpecification::JupyterServer(remote_spec) => {
                RemoteRunningKernel::new(remote_spec, working_directory, view, window, cx)
            }

            KernelSpecification::SshRemote(spec) => {
                let project = self.project.clone();
                SshRunningKernel::new(spec, working_directory, project, view, window, cx)
            }
            KernelSpecification::WslRemote(spec) => {
                WslRunningKernel::new(spec, entity_id, working_directory, fs, view, window, cx)
            }
        };

        let pending_kernel = cx
            .spawn_in(window, async move |this, cx| {
                let kernel = kernel_task.await;

                match kernel {
                    Ok(kernel) => {
                        this.update_in(cx, |editor, window, cx| {
                            editor.kernel = Kernel::RunningKernel(kernel);
                            editor.kernel_reached_running = true;
                            // Uptime counts from the kernel actually being up,
                            // not from the launch request.
                            editor.kernel_started_at = Some(Instant::now());
                            editor.kernel_uptime_frozen = None;
                            cx.notify();
                            let queued = std::mem::take(&mut editor.pending_executions);
                            log::debug!(
                                "notebook: kernel ready; submitting {} queued cell(s) in order",
                                queued.len(),
                            );
                            for cell_id in queued {
                                editor.execute_cell(cell_id, window, cx);
                            }
                        })
                        .ok();
                    }
                    Err(err) => {
                        log::error!("Kernel failed to start: {:?}", err);
                        this.update_in(cx, |editor, window, cx| {
                            let error_message = err.to_string();
                            editor.kernel = Kernel::ErroredLaunch(error_message.clone());
                            // The launch failed, so no queued cell can run.
                            editor.cancel_run_queue(cx);
                            cx.notify();
                            for cell_id in std::mem::take(&mut editor.pending_executions) {
                                if let Some(Cell::Code(cell)) = editor.cell_map.get(&cell_id) {
                                    cell.update(cx, |cell, cx| {
                                        cell.show_kernel_error(
                                            &format!(
                                                "the kernel failed to launch: {error_message}"
                                            ),
                                            window,
                                            cx,
                                        );
                                    });
                                }
                            }
                        })
                        .ok();
                    }
                }
            })
            .shared();

        self.kernel = Kernel::StartingKernel(pending_kernel);
        // A fresh launch has not connected yet; until it does, an error is a
        // launch failure (prompt), not a died-after-running kernel (relaunch).
        self.kernel_reached_running = false;
        cx.notify();
    }

    // Note: Python environments are only detected as kernels if ipykernel is installed.
    // Users need to run `pip install ipykernel` (or `uv pip install ipykernel`) in their
    // virtual environment for it to appear in the kernel selector.
    // This happens because we have an ipykernel check inside the function python_env_kernel_specification in mod.rs L:121

    fn change_kernel(
        &mut self,
        spec: KernelSpecification,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        // Selecting a real kernel supersedes any in-progress "creating" pending
        // selection (phase 48): stop showing the creating env, and let the
        // build's completion see it was superseded so it won't re-select.
        self.creating_kernel_name = None;

        if let Kernel::RunningKernel(kernel) = &mut self.kernel {
            kernel.force_shutdown(window, cx).detach();
        }

        self.execution_requests.clear();
        // If this is a deliberate kernel switch (nothing was waiting on a
        // kernel choice), abort any in-progress batch and cancel in-flight
        // cells. If instead the user is picking a kernel to satisfy a run that
        // was waiting for one, keep the queue AND the cells' Pending status —
        // cancelling them here made every batch cell flash "Cancelled" between
        // picking a kernel and the kernel starting (bug #28).
        if self.cells_awaiting_kernel_choice.is_empty() {
            self.cancel_run_queue(cx);
            self.stop_executing_cells(cx);
        }

        // Remember the choice for THIS notebook only (bug #30): a pick here
        // must not change which kernel sibling notebooks or the inline REPL
        // resolve to. Cross-session persistence flows through the notebook's
        // own kernelspec metadata, written below on launch. An untitled
        // notebook has no path to key on — its own `kernel_specification`
        // field carries the pick for the session.
        if let Some(notebook_path) = self.notebook_item.read(cx).path.clone() {
            ReplStore::global(cx).update(cx, |store, cx| {
                store.set_notebook_kernelspec(notebook_path, spec.clone(), cx);
            });
        }

        // Any cell the user ran before picking a kernel should now run once
        // this kernel is ready.
        self.promote_awaiting_cells();

        self.launch_kernel_with_spec(spec, window, cx);
    }

    fn restart_kernel(&mut self, _: &RestartKernel, window: &mut Window, cx: &mut Context<Self>) {
        let Some(spec) = self.kernel_specification.clone() else {
            return;
        };

        // A restart is a new session: zero the timers now rather than when the
        // relaunch lands, so the old kernel's uptime doesn't keep climbing
        // through the shutdown wait.
        self.reset_runtime_timers();

        let kernel = std::mem::replace(&mut self.kernel, Kernel::Restarting);
        self.execution_requests.clear();
        self.cancel_run_queue(cx);
        self.stop_executing_cells(cx);
        // The restarted kernel's execution counter starts over at 1, so the
        // cells' `In [N]` numbers from the old session are stale — clear them
        // so a fresh run-through is visually distinct. Only when a session
        // actually existed, though: "restarting" a never-launched kernel
        // (possible since phase 25 pre-selects one on open) must not wipe the
        // counts loaded from disk or dirty the notebook.
        if matches!(
            kernel,
            Kernel::RunningKernel(_) | Kernel::StartingKernel(_) | Kernel::Restarting
        ) {
            self.execution_state_changed = true;
            for cell in self.cell_map.values() {
                if let Cell::Code(code_cell) = cell {
                    code_cell.update(cx, |code_cell, cx| {
                        code_cell.reset_execution_count();
                        cx.notify();
                    });
                }
            }
        }
        cx.notify();

        match kernel {
            Kernel::Restarting => {}
            starting @ Kernel::StartingKernel(_) => {
                // A launch is already in flight; let it finish rather than racing it.
                self.kernel = starting;
            }
            Kernel::RunningKernel(mut kernel) => {
                let mut request_tx = kernel.request_tx();

                cx.spawn_in(window, async move |this, cx| {
                    let message: JupyterMessage = ShutdownRequest { restart: true }.into();
                    request_tx.try_send(message).ok();

                    // Give the kernel a chance to exit gracefully and release
                    // its sockets before force-killing and relaunching.
                    cx.background_executor().timer(Duration::from_secs(1)).await;

                    if let Ok(forced) =
                        this.update_in(cx, |_, window, cx| kernel.force_shutdown(window, cx))
                    {
                        forced.await.log_err();
                    }
                    // Dropping the old kernel here cancels its message tasks
                    // and removes its connection file before the relaunch.
                    drop(kernel);

                    this.update_in(cx, |this, window, cx| {
                        this.launch_kernel_with_spec(spec, window, cx);
                    })
                    .ok();
                })
                .detach();
            }
            Kernel::ErroredLaunch(_) | Kernel::ShuttingDown | Kernel::Shutdown => {
                self.launch_kernel_with_spec(spec, window, cx);
            }
        }
    }

    /// Cancel every running/queued cell (kernel restart/loss/switch). The
    /// cells never completed, so they get a cancelled marker — NOT a completed
    /// tick with a bogus time.
    fn stop_executing_cells(&mut self, cx: &mut Context<Self>) {
        for cell in self.cell_map.values() {
            if let Cell::Code(code_cell) = cell {
                code_cell.update(cx, |cell, cx| {
                    if cell.is_execution_in_flight() {
                        cell.cancel_execution();
                        cx.notify();
                    }
                });
            }
        }
        // A cancelled cell still spent the time it ran for; bank it now rather
        // than letting the clock run on with nothing executing.
        self.sync_execution_clock(cx);
    }

    fn interrupt_kernel(
        &mut self,
        _: &InterruptKernel,
        _window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        // Interrupting stops the whole batch, not just the current cell.
        self.cancel_run_queue(cx);
        match &self.kernel {
            Kernel::RunningKernel(kernel) => {
                kernel.interrupt();
                cx.notify();
            }
            _ => {
                log::warn!("notebook: interrupt requested but no kernel is running");
            }
        }
    }

    fn execute_cell(&mut self, cell_id: CellId, window: &mut Window, cx: &mut Context<Self>) {
        // A kernel environment is being created (phase 48): hold the cell like
        // the awaiting-kernel-choice path (Pending, no spinner) — but WITHOUT
        // opening the picker — so it runs on the new kernel once built and
        // never starts/uses the previously selected kernel. `change_kernel`
        // (on build success) promotes these; `clear_awaiting_cells` (on build
        // failure) returns them to Idle.
        if self.creating_kernel_name.is_some() {
            if !self.cells_awaiting_kernel_choice.contains(&cell_id) {
                self.cells_awaiting_kernel_choice.push(cell_id.clone());
            }
            if let Some(Cell::Code(cell)) = self.cell_map.get(&cell_id) {
                cell.update(cx, |cell, cx| {
                    cell.mark_pending();
                    cx.notify();
                });
            }
            return;
        }

        let code = if let Some(Cell::Code(cell)) = self.cell_map.get(&cell_id) {
            let editor = cell.read(cx).editor().clone();
            let buffer = editor.read(cx).buffer().read(cx);
            buffer
                .as_singleton()
                .map(|b| b.read(cx).text())
                .unwrap_or_default()
        } else {
            return;
        };

        enum Disposition {
            Sent(String),
            /// Queued to run when the (launching/starting) kernel is ready;
            /// shows the running spinner now. `launch` starts a remembered
            /// kernel.
            Queued {
                launch: bool,
            },
            /// No kernel selected: prompt for one. The cell is held in
            /// `cells_awaiting_kernel_choice` WITHOUT a spinner, so dismissing
            /// the picker leaves it idle; it runs only if a kernel is chosen.
            Prompt,
            Failed(String),
        }

        // Computed before borrowing `self.kernel` mutably below.
        let has_remembered_kernel = self.remembered_kernel_spec(cx).is_some();
        let kernel_reached_running = self.kernel_reached_running;

        let disposition = match &mut self.kernel {
            Kernel::RunningKernel(kernel) => {
                let request = ExecuteRequest {
                    code,
                    ..Default::default()
                };
                let message: JupyterMessage = request.into();
                let msg_id = message.header.msg_id.clone();
                match kernel.request_tx().try_send(message) {
                    Ok(()) => Disposition::Sent(msg_id),
                    Err(err) => Disposition::Failed(format!(
                        "failed to send execute request to kernel (the kernel process may have died): {err}"
                    )),
                }
            }
            Kernel::StartingKernel(_) | Kernel::Restarting => Disposition::Queued { launch: false },
            Kernel::Shutdown => {
                if has_remembered_kernel {
                    Disposition::Queued { launch: true }
                } else {
                    Disposition::Prompt
                }
            }
            // The kernel is in an error state. Two cases:
            // - It DIED after running (e.g. the Rust/evcxr kernel exits when
            //   interrupted, which errors the KERNEL, not a cell). The spec is
            //   good, so relaunch it on this run — like `Shutdown` does — rather
            //   than asking which kernel to use.
            // - Its LAUNCH never connected (`kernel_reached_running == false`):
            //   silently relaunching would loop the error forever (e.g. a spec
            //   that can't start — bug #31), so re-prompt; an explicit pick
            //   replaces the broken selection.
            Kernel::ErroredLaunch(_) => {
                if kernel_reached_running && has_remembered_kernel {
                    Disposition::Queued { launch: true }
                } else {
                    Disposition::Prompt
                }
            }
            Kernel::ShuttingDown => Disposition::Failed("the kernel is shutting down".to_string()),
        };

        if let Disposition::Prompt = disposition {
            // If we are prompting because the remembered env VANISHED (the
            // validated resolution returned None while the raw one still has
            // a spec), drop the ghost selection and refresh discovery before
            // the picker opens (phase 42).
            if !has_remembered_kernel
                && self
                    .remembered_kernel_spec_any(cx)
                    .is_some_and(|spec| Self::spec_interpreter_missing(&spec))
            {
                self.discard_stale_kernel_selection(cx);
            }
            // Hold the cell (no spinner) and open the picker directly — NOT
            // via launch_kernel, whose remembered-spec fallback would relaunch
            // the very spec that just failed (bug #31). Both Prompt producers
            // want the picker: Shutdown-without-remembered has nothing to
            // launch, and ErroredLaunch must not relaunch. The cell runs if a
            // kernel is chosen (see change_kernel), or is cleared on dismiss.
            if !self.cells_awaiting_kernel_choice.contains(&cell_id) {
                self.cells_awaiting_kernel_choice.push(cell_id);
            }
            // Defer the picker open: `PopoverMenuHandle::show` synchronously
            // fires the picker's `on_open` callback, which re-enters
            // `NotebookEditor.update` (re-validating envs). Calling it inline
            // would nest an update inside this one (execute_cell already holds
            // the entity lease) and panic with a double-lease (bug #48). Use
            // `window.defer` — NOT `cx.defer_in`, which re-wraps the closure in
            // a `NotebookEditor` update and reintroduces the same nesting —
            // so `show` runs with no entity lease held.
            let kernel_picker_handle = self.kernel_picker_handle.clone();
            window.defer(cx, move |window, cx| {
                kernel_picker_handle.show(window, cx);
            });
            return;
        }

        if let Disposition::Queued { launch } = &disposition {
            if !self.pending_executions.contains(&cell_id) {
                self.pending_executions.push(cell_id.clone());
            }
            if *launch {
                self.launch_kernel(window, cx);
            }
        }

        if let Some(Cell::Code(cell)) = self.cell_map.get(&cell_id) {
            // Everything but Prompt mutates the cell's execution state
            // (queues a run or records an error), which is savable content.
            if !matches!(disposition, Disposition::Prompt) {
                self.execution_state_changed = true;
            }
            cell.update(cx, |cell, cx| {
                match &disposition {
                    Disposition::Failed(error) => {
                        if cell.has_outputs() {
                            cell.clear_outputs();
                        }
                        cell.show_kernel_error(error, window, cx);
                    }
                    // Submitted or queued cells are PENDING: no spinner, no
                    // timer, and the old output stays until the kernel
                    // actually starts the cell (its `execute_input` arrives —
                    // see `begin_running`). This keeps queued cells from
                    // accruing the wait behind a long-running cell.
                    Disposition::Queued { .. } => cell.mark_pending(),
                    // A Sent cell also anchors its submit time, so a very fast
                    // cell whose reply beats its `execute_input` still reports a
                    // duration (see `CodeCell::record_submitted`).
                    Disposition::Sent(_) => {
                        cell.mark_pending();
                        cell.record_submitted();
                    }
                    Disposition::Prompt => {}
                }
                cx.notify();
            });
        }

        match disposition {
            Disposition::Sent(msg_id) => {
                // No longer pending — it's been submitted to the kernel.
                self.pending_executions.retain(|id| id != &cell_id);
                self.execution_requests.insert(msg_id, cell_id);
            }
            Disposition::Queued { .. } | Disposition::Prompt => {}
            Disposition::Failed(error) => {
                log::error!("notebook: cannot execute cell: {error}");
            }
        }
    }

    /// Promote cells that were waiting for a kernel choice into the pending
    /// queue (they run once the newly-selected kernel is ready).
    fn promote_awaiting_cells(&mut self) {
        log::debug!(
            "notebook: promoting {} awaiting cell(s) into the pending queue ({} already pending)",
            self.cells_awaiting_kernel_choice.len(),
            self.pending_executions.len(),
        );
        for cell_id in std::mem::take(&mut self.cells_awaiting_kernel_choice) {
            if !self.pending_executions.contains(&cell_id) {
                self.pending_executions.push(cell_id);
            }
        }
    }

    /// Clear cells that were waiting for a kernel choice (picker dismissed
    /// without selecting). They and any batch queued behind them never reached
    /// a kernel, so they return to IDLE — not Cancelled (bug #28): nothing was
    /// cancelled mid-flight, the run simply never started.
    fn clear_awaiting_cells(&mut self, cx: &mut Context<Self>) {
        if !self.cells_awaiting_kernel_choice.is_empty() {
            log::debug!(
                "notebook: kernel picker dismissed; dropping {} awaiting cell(s)",
                self.cells_awaiting_kernel_choice.len(),
            );
            for cell_id in std::mem::take(&mut self.cells_awaiting_kernel_choice) {
                if let Some(Cell::Code(cell)) = self.cell_map.get(&cell_id) {
                    cell.update(cx, |cell, cx| {
                        cell.reset_execution_status();
                        cx.notify();
                    });
                }
            }
            self.abandon_run_queue(cx);
            cx.notify();
        }
    }

    /// Drop a run queue whose cells never reached a kernel (the kernel picker
    /// was dismissed without a selection): their statuses return to Idle.
    /// Interrupt/error/restart/kernel-loss paths use `cancel_run_queue`
    /// instead, which marks the queued cells Cancelled.
    fn abandon_run_queue(&mut self, cx: &mut Context<Self>) {
        for cell_id in std::mem::take(&mut self.run_queue) {
            if let Some(Cell::Code(cell)) = self.cell_map.get(&cell_id) {
                cell.update(cx, |cell, cx| {
                    cell.reset_execution_status();
                    cx.notify();
                });
            }
        }
        self.active_run_cell = None;
        self.resume_run_queue_on_idle = false;
    }

    fn get_selected_cell(&self) -> Option<&Cell> {
        self.cell_order
            .get(self.selected_cell_index)
            .and_then(|cell_id| self.cell_map.get(cell_id))
    }

    /// Whether Clear Outputs would do anything. Outputs are not the only thing
    /// it clears — execution counts, run durations, timestamps and status
    /// markers go with them — so a notebook of cells that ran silently still
    /// has something to clear (user 2026-08-06).
    fn has_execution_record(&self, _window: &mut Window, cx: &mut Context<Self>) -> bool {
        self.cell_map.values().any(|cell| {
            if let Cell::Code(code_cell) = cell {
                code_cell.read(cx).has_execution_record()
            } else {
                false
            }
        })
    }

    fn clear_outputs(&mut self, _window: &mut Window, cx: &mut Context<Self>) {
        self.execution_state_changed = true;
        for cell in self.cell_map.values() {
            if let Cell::Code(code_cell) = cell {
                code_cell.update(cx, |cell, cx| {
                    cell.clear_execution_record();
                    cx.notify();
                });
            }
        }
        cx.notify();
    }

    fn run_cells(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        self.run_cell_batch(self.cell_order.clone(), window, cx);
    }

    /// Run a batch of cells sequentially, stopping the remainder if one fails.
    fn run_cell_batch(&mut self, cells: Vec<CellId>, window: &mut Window, cx: &mut Context<Self>) {
        // A batch run (Run All / Run Above / Run Below) supersedes any run
        // already in flight: interrupt it and drop its routing so trailing
        // messages don't touch the new run. Without this, `advance_run_queue`
        // refuses to start while a cell is active, leaving the new batch stuck
        // Pending forever. (A single-cell run does NOT go through here — it
        // just queues at the kernel like Jupyter.)
        let superseded = self.active_run_cell.is_some() || !self.run_queue.is_empty();
        if superseded {
            if let Kernel::RunningKernel(kernel) = &self.kernel {
                kernel.interrupt();
            }
            self.cancel_run_queue(cx);
            self.stop_executing_cells(cx);
            self.execution_requests.clear();
        }

        // Every code cell in the batch shows as pending immediately — a
        // previously-executed cell's ✓ makes way for the pending marker, but
        // its OUTPUT stays until the cell actually re-executes.
        for cell_id in &cells {
            if let Some(Cell::Code(cell)) = self.cell_map.get(cell_id) {
                self.execution_state_changed = true;
                cell.update(cx, |cell, cx| {
                    cell.mark_pending();
                    cx.notify();
                });
            }
        }
        self.run_queue = cells;
        // When we superseded a run on a BUSY kernel, wait for the interrupted
        // run to finish aborting (kernel returns to Idle) before submitting —
        // otherwise these requests land in the kernel's "aborting" state and
        // come back Aborted (see the Status handling in `route`). If the
        // kernel is NOT busy, no further Status(idle) transition may ever
        // arrive, so waiting would deadlock the batch at "Pending" forever —
        // submit immediately instead.
        let kernel_busy = matches!(self.kernel.status(), KernelStatus::Busy);
        if superseded && kernel_busy {
            self.resume_run_queue_on_idle = true;
        } else {
            self.advance_run_queue(window, cx);
        }
    }

    /// Submit the next queued cell, if nothing from the batch is already
    /// running. Non-code cells are skipped. The next cell is submitted only
    /// once the current one's `ExecuteReply` arrives (see `route`), so a
    /// failure can cancel the rest.
    fn advance_run_queue(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if self.active_run_cell.is_some() {
            return;
        }
        while !self.run_queue.is_empty() {
            let cell_id = self.run_queue.remove(0);
            if matches!(self.cell_map.get(&cell_id), Some(Cell::Code(_))) {
                self.active_run_cell = Some(cell_id.clone());
                if self.follow_running_cell
                    && let Some(index) = self.cell_order.iter().position(|id| id == &cell_id)
                {
                    self.follow_reveal(index, window, cx);
                }
                self.execute_cell(cell_id, window, cx);
                return;
            }
            // Skip markdown/raw cells and continue to the next.
        }
    }

    /// Scroll (viewport only) so the cell at `index` sits near the top of the
    /// viewport, with a small margin of the preceding cell for context. Used by
    /// follow mode instead of a minimal reveal, which would land each newly
    /// running cell on the bottom edge as a Run All walks down. The margin
    /// scales with the viewport but is bounded so it stays "near the top" on
    /// large screens without dominating small ones; because it's a fixed offset
    /// above the running cell, a tall preceding output never pushes the running
    /// cell out of view.
    fn follow_scroll_to(&self, index: usize) {
        let viewport_height = self.cell_list.viewport_bounds().size.height;
        let margin = (viewport_height * 0.12).max(px(40.)).min(px(96.));
        self.cell_list.scroll_to_item_near_top(index, margin);
    }

    /// Follow the cell at `index` as it starts running, in whichever way
    /// `repl.notebook_follow_mode` asks for.
    ///
    /// `minimal` re-pins every cell near the top as it runs — a small, frequent
    /// movement. `page` instead makes the SELECTION carry the progress and
    /// leaves the viewport alone while the running cell is on screen, moving a
    /// whole page only when execution passes the fold. The two are deliberately
    /// opposite: `minimal` keeps the running cell in one place, `page` keeps the
    /// page in one place. Which reads better depends on whether you are watching
    /// the cell or the notebook.
    fn follow_reveal(&mut self, index: usize, window: &mut Window, cx: &mut Context<Self>) {
        match ReplSettings::get_global(cx).notebook_follow_mode {
            NotebookFollowMode::Minimal => self.follow_scroll_to(index),
            NotebookFollowMode::Page => {
                // Only the SELECTION moves here — it is this mode's progress
                // indicator, so it moves for every cell even when the viewport
                // doesn't. Safe against stealing a cursor because entering edit
                // mode turns follow off (`disable_follow_on_edit`).
                //
                // The viewport is NOT decided here. Where the page should sit
                // depends on cell heights, and the cell that just started
                // running has produced no output yet — so anything decided now
                // is decided from heights that are about to change.
                // `maintain_page_follow` re-checks it every frame instead.
                self.set_selected_index(index, false, window, cx);
                cx.notify();
            }
        }
    }

    /// Keep the running cell on screen in page mode. Called once per frame from
    /// the cell list, so it sees the heights the LAST layout measured.
    ///
    /// Deciding the page only when a cell starts running does not hold: the
    /// cells above it are still producing output, which grows them and pushes
    /// the running cell down and off the bottom (user 2026-08-11 — a notebook
    /// of `time.sleep` cells followed perfectly, one that displayed a DataFrame
    /// did not). Nothing can predict that at run start, because the output does
    /// not exist yet, so the only correct answer is to keep asking. The check is
    /// a cursor seek and re-scrolling to a page already in place is a no-op, so
    /// asking every frame is cheap and settles immediately.
    fn maintain_page_follow(&mut self, cx: &App) {
        if !self.follow_running_cell
            || ReplSettings::get_global(cx).notebook_follow_mode != NotebookFollowMode::Page
        {
            return;
        }
        let Some(index) = self.running_cell_index(cx) else {
            return;
        };
        if !self.cell_is_followed(index) {
            self.cell_list.scroll_to_item_top_clamped(index);
        }
    }

    /// Whether cell `index` counts as being followed — i.e. the viewport is
    /// already showing what there is to show of it.
    ///
    /// Normally that means ENTIRELY on screen: page-wise follow treats the last
    /// fully-visible cell as the page boundary, so a cell hanging off the bottom
    /// edge starts the next page rather than being watched with its output cut
    /// off. A cell TALLER than the viewport can never satisfy that, so for those
    /// any part being visible counts — otherwise every frame would drag the
    /// viewport back to the cell's top and you could never scroll down its own
    /// output while it ran.
    fn cell_is_followed(&self, index: usize) -> bool {
        let viewport = self.cell_list.viewport_bounds();
        // `bounds_for_item` yields `None` for a cell above the scroll position
        // or one too far below it to have been measured — neither is on screen.
        self.cell_list.bounds_for_item(index).is_some_and(|bounds| {
            if bounds.size.height > viewport.size.height {
                bounds.top() < viewport.bottom() && bounds.bottom() > viewport.top()
            } else {
                bounds.top() >= viewport.top() && bounds.bottom() <= viewport.bottom()
            }
        })
    }

    /// Follow mode is for WATCHING a run, so any move into edit mode ends it —
    /// otherwise page-wise follow would drag the selection out from under
    /// someone typing in a cell further down. Turning it back on is deliberate
    /// (the toggle), which is the point: you have stopped watching.
    ///
    /// This fires for the post-run landing too, when
    /// `repl.notebook_run_landing_mode` is `edit`: landing in a cell's editor
    /// after a run is still being in edit mode, and single-cell runs don't use
    /// follow mode anyway (only batch runs walk the queue).
    fn disable_follow_on_edit(&mut self, cx: &mut Context<Self>) {
        if self.follow_running_cell {
            self.follow_running_cell = false;
            cx.notify();
        }
    }

    /// Reconcile the execution clock with whether a cell is running right now.
    /// Called at every run start/finish as well as on the display tick, so the
    /// banked total is exact rather than quantised to the tick interval — and
    /// so it stays correct while the timer is switched off in settings (no tick
    /// runs then).
    fn sync_execution_clock(&mut self, cx: &App) {
        match (
            self.running_cell_index(cx).is_some(),
            self.execution_time_started_at,
        ) {
            (true, None) => self.execution_time_started_at = Some(Instant::now()),
            (false, Some(started_at)) => {
                self.execution_time_banked += started_at.elapsed();
                self.execution_time_started_at = None;
            }
            _ => {}
        }
    }

    /// Time this kernel session has spent executing cells, including the cell
    /// running right now.
    fn execution_time_total(&self) -> Duration {
        self.execution_time_banked
            + self
                .execution_time_started_at
                .map_or(Duration::ZERO, |started_at| started_at.elapsed())
    }

    /// How long the kernel has been up — live while it runs, frozen at its
    /// final value once it stops, `None` before any kernel has started.
    fn kernel_uptime(&self) -> Option<Duration> {
        self.kernel_started_at
            .map(|started_at| started_at.elapsed())
            .or(self.kernel_uptime_frozen)
    }

    /// A new kernel session begins: both timers measure THIS session, so they
    /// start from zero on a launch, restart or kernel switch.
    fn reset_runtime_timers(&mut self) {
        self.execution_time_banked = Duration::ZERO;
        self.execution_time_started_at = None;
        self.kernel_started_at = None;
        self.kernel_uptime_frozen = None;
    }

    /// The kernel stopped: keep its final uptime on screen instead of dropping
    /// back to nothing, until a kernel starts again.
    fn freeze_kernel_uptime(&mut self) {
        if let Some(started_at) = self.kernel_started_at.take() {
            self.kernel_uptime_frozen = Some(started_at.elapsed());
        }
    }

    /// Whether a timer that CHANGES over time is on screen. A frozen uptime and
    /// a paused execution total are static, so they need no ticking.
    fn has_live_timer(&self, cx: &App) -> bool {
        let settings = ReplSettings::get_global(cx);
        (settings.notebook_show_execution_time && self.execution_time_started_at.is_some())
            || (settings.notebook_show_kernel_uptime && self.kernel_started_at.is_some())
    }

    /// Keep the strip's timers redrawing while one of them is live. Driven from
    /// `render`, so it covers a run starting, a kernel coming up and the
    /// settings being switched on mid-session alike; the task ends itself as
    /// soon as nothing is live, so an idle notebook holds no wake-up loop.
    fn ensure_runtime_timer(&mut self, cx: &mut Context<Self>) {
        if self.runtime_timer_ticking || !self.has_live_timer(cx) {
            return;
        }
        self.runtime_timer_ticking = true;
        self._runtime_timer = Some(cx.spawn(async move |this, cx| {
            loop {
                cx.background_executor()
                    .timer(Duration::from_millis(100))
                    .await;
                let still_live = this
                    .update(cx, |this, cx| {
                        this.sync_execution_clock(cx);
                        let live = this.has_live_timer(cx);
                        if !live {
                            this.runtime_timer_ticking = false;
                        }
                        // Notify either way: the tick that finds the timer no
                        // longer live is the one that paints its final value.
                        cx.notify();
                        live
                    })
                    .unwrap_or(false);
                if !still_live {
                    break;
                }
            }
        }));
    }

    /// The index (in `cell_order`) of the cell currently executing, if any.
    /// Backs both the Go to running cell action and its sidebar button's
    /// enabled state. Only one cell runs at a time (sequential queue), so the
    /// first executing cell is the running one.
    fn running_cell_index(&self, cx: &App) -> Option<usize> {
        self.cell_order.iter().position(|id| {
            matches!(
                self.cell_map.get(id),
                Some(Cell::Code(cell)) if cell.read(cx).is_executing()
            )
        })
    }

    /// Reveal, select, and focus the currently-executing cell. A no-op (not an
    /// error) when nothing is running.
    fn go_to_running_cell(
        &mut self,
        _: &GoToRunningCell,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if let Some(index) = self.running_cell_index(cx) {
            // Select without a cumulative-height reveal, then use the same
            // index-anchored near-top scroll as follow mode. A plain reveal
            // derives the scroll offset from the summed height of every cell
            // ABOVE the target, which is wrong when those cells are unmeasured
            // (large notebook just opened) or hold stale heights (their outputs
            // grew while off-screen during a long Run All) — it lands partway
            // instead of on the running cell. Anchoring on the target's index
            // paints downward from it and is immune to those heights.
            self.set_selected_index(index, false, window, cx);
            self.follow_scroll_to(index);
            self.enter_command_mode(window, cx);
            cx.notify();
        }
    }

    /// Indices (in `cell_order`) of every cell whose last execution failed.
    /// No run-scoping is needed: a cell re-run this session either raised again
    /// or is no longer `Failed`, and a cell queued behind one is `Pending`, so a
    /// `Failed` status is always truthful about that cell's last run.
    fn failed_cell_indices(&self, cx: &App) -> Vec<usize> {
        self.cell_order
            .iter()
            .enumerate()
            .filter_map(|(index, id)| match self.cell_map.get(id) {
                Some(Cell::Code(cell)) if cell.read(cx).has_failed() => Some(index),
                _ => None,
            })
            .collect()
    }

    /// Reveal, select, and focus a failed cell, landing on its ERROR rather
    /// than its first line — the traceback is the informative part, so the
    /// cell's tail is what gets shown.
    fn reveal_error_cell(&mut self, index: usize, window: &mut Window, cx: &mut Context<Self>) {
        self.set_selected_index(index, false, window, cx);
        self.cell_list.scroll_to_item_bottom_aligned(index);
        self.enter_command_mode(window, cx);
        cx.notify();
    }

    /// Jump to the failed cell. Under stop-on-error a batch has exactly one, so
    /// document order picks the cell that halted the run; `NextError` /
    /// `PreviousError` cover the several-failures case. A no-op when nothing
    /// has failed.
    fn go_to_error(&mut self, _: &GoToError, window: &mut Window, cx: &mut Context<Self>) {
        if let Some(&index) = self.failed_cell_indices(cx).first() {
            self.reveal_error_cell(index, window, cx);
        }
    }

    fn next_error(&mut self, _: &NextError, window: &mut Window, cx: &mut Context<Self>) {
        let failed = self.failed_cell_indices(cx);
        if let Some(index) = next_failed_index(&failed, self.selected_cell_index) {
            self.reveal_error_cell(index, window, cx);
        }
    }

    fn previous_error(&mut self, _: &PreviousError, window: &mut Window, cx: &mut Context<Self>) {
        let failed = self.failed_cell_indices(cx);
        if let Some(index) = previous_failed_index(&failed, self.selected_cell_index) {
            self.reveal_error_cell(index, window, cx);
        }
    }

    fn toggle_follow_running_cell(
        &mut self,
        _: &ToggleFollowRunningCell,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.follow_running_cell = !self.follow_running_cell;
        // Turning it on catches up with the running cell immediately (if any),
        // so the toggle's effect is visible right away rather than only on the
        // next queue advance. Page mode reveals unconditionally here: the point
        // of the toggle is to go and look, even if the cell happens to be on
        // screen already.
        if self.follow_running_cell
            && let Some(index) = self.running_cell_index(cx)
        {
            match ReplSettings::get_global(cx).notebook_follow_mode {
                NotebookFollowMode::Minimal => self.follow_scroll_to(index),
                NotebookFollowMode::Page => {
                    self.set_selected_index(index, false, window, cx);
                    // Unconditional: the point of switching follow ON is to go
                    // and look, even if the cell happens to be on screen.
                    self.cell_list.scroll_to_item_top_clamped(index);
                }
            }
        }
        cx.notify();
    }

    /// Abort any in-progress multi-cell run (e.g. on error, interrupt, kernel
    /// loss, or a structural change). Cells still waiting in the queue lose
    /// their pending marker — they will not run. The active cell is left to
    /// resolve via its kernel reply (or `stop_executing_cells` on kernel loss).
    fn cancel_run_queue(&mut self, cx: &mut Context<Self>) {
        for cell_id in std::mem::take(&mut self.run_queue) {
            if let Some(Cell::Code(cell)) = self.cell_map.get(&cell_id) {
                cell.update(cx, |cell, cx| {
                    cell.cancel_execution();
                    cx.notify();
                });
            }
        }
        self.active_run_cell = None;
        self.resume_run_queue_on_idle = false;
    }

    /// Whether a cell already has an execution in flight — actively running,
    /// queued in a batch, or waiting on a kernel start/choice. Used so
    /// repeatedly running a cell (e.g. shift-enter cycling past an already
    /// running/queued cell) doesn't double-queue or re-run it.
    ///
    /// This is driven by the cell's own Pending/Running status (which resolves
    /// to Finished/Cancelled when the run ends) rather than `execution_requests`
    /// — that map is not pruned per-cell, so using it would keep every
    /// previously-run cell "in flight" forever and block re-running.
    fn is_cell_in_flight(&self, cell_id: &CellId, cx: &App) -> bool {
        if self.active_run_cell.as_ref() == Some(cell_id)
            || self.run_queue.contains(cell_id)
            || self.pending_executions.contains(cell_id)
            || self.cells_awaiting_kernel_choice.contains(cell_id)
        {
            return true;
        }
        matches!(
            self.cell_map.get(cell_id),
            Some(Cell::Code(cell)) if cell.read(cx).is_execution_in_flight()
        )
    }

    /// Handle a per-cell stop (gutter stop button). Scoped to the cell:
    /// - the actively-running cell → interrupt the kernel (halts the batch, as
    ///   an interrupt always has);
    /// - a queued (pending) batch cell → drop just that cell from the queue and
    ///   cancel its status, leaving the rest of the batch to run;
    /// - a single running cell (not part of a batch) → interrupt; a
    ///   pending/awaiting single cell → cancel and forget it.
    fn handle_cell_stop(&mut self, cell_id: &CellId, window: &mut Window, cx: &mut Context<Self>) {
        if self.active_run_cell.as_ref() == Some(cell_id) {
            self.interrupt_kernel(&InterruptKernel, window, cx);
            return;
        }
        if self.run_queue.contains(cell_id) {
            self.run_queue.retain(|id| id != cell_id);
            if let Some(Cell::Code(cell)) = self.cell_map.get(cell_id) {
                cell.update(cx, |cell, cx| {
                    cell.cancel_execution();
                    cx.notify();
                });
            }
            return;
        }
        // Not a batch cell: either a single cell running at the kernel, or one
        // pending/awaiting a kernel.
        let running = matches!(
            self.cell_map.get(cell_id),
            Some(Cell::Code(cell)) if cell.read(cx).execution_status() == CellExecutionStatus::Running
        );
        if running {
            self.interrupt_kernel(&InterruptKernel, window, cx);
        } else {
            self.pending_executions.retain(|id| id != cell_id);
            self.cells_awaiting_kernel_choice.retain(|id| id != cell_id);
            if let Some(Cell::Code(cell)) = self.cell_map.get(cell_id) {
                cell.update(cx, |cell, cx| {
                    cell.cancel_execution();
                    cx.notify();
                });
            }
        }
    }

    fn run_current_cell(&mut self, _: &Run, window: &mut Window, cx: &mut Context<Self>) {
        // Capture the mode BEFORE running, for the `remember` landing mode.
        let was_edit_mode = self.notebook_mode == NotebookMode::Edit;
        // Run on a multi-selection executes every selected cell, in order.
        if self.has_multi_selection() {
            let cells: Vec<CellId> = self
                .effective_selection()
                .into_iter()
                .filter_map(|index| self.cell_order.get(index).cloned())
                .collect();
            self.run_cell_batch(cells, window, cx);
            self.apply_post_run_landing(was_edit_mode, window, cx);
            return;
        }
        let Some(cell_id) = self.cell_order.get(self.selected_cell_index).cloned() else {
            return;
        };
        let Some(cell) = self.cell_map.get(&cell_id) else {
            return;
        };
        match cell {
            Cell::Code(_) => {
                // Don't re-run a cell whose execution is already in flight.
                if !self.is_cell_in_flight(&cell_id, cx) {
                    self.execute_cell(cell_id, window, cx);
                }
            }
            Cell::Markdown(markdown_cell) => {
                // for markdown, finish editing
                let is_editing = markdown_cell.read(cx).is_editing();
                if is_editing {
                    markdown_cell.update(cx, |cell, cx| {
                        cell.run(cx);
                    });
                }
            }
            Cell::Raw(_) => {}
        }
        self.apply_post_run_landing(was_edit_mode, window, cx);
    }

    fn run_and_advance(&mut self, _: &RunAndAdvance, window: &mut Window, cx: &mut Context<Self>) {
        // Capture the mode BEFORE running, for the `remember` landing mode.
        let was_edit_mode = self.notebook_mode == NotebookMode::Edit;
        if let Some(cell_id) = self.cell_order.get(self.selected_cell_index).cloned() {
            if let Some(cell) = self.cell_map.get(&cell_id) {
                match cell {
                    Cell::Code(_) => {
                        // Don't re-run a cell whose execution is already in
                        // flight — cycling shift-enter past a running/queued
                        // cell must leave it untouched.
                        if !self.is_cell_in_flight(&cell_id, cx) {
                            self.execute_cell(cell_id, window, cx);
                        }
                    }
                    Cell::Markdown(markdown_cell) => {
                        if markdown_cell.read(cx).is_editing() {
                            markdown_cell.update(cx, |cell, cx| {
                                cell.run(cx);
                            });
                        }
                    }
                    Cell::Raw(_) => {}
                }
            }
        }

        let is_last_cell = self.selected_cell_index == self.cell_count().saturating_sub(1);
        if is_last_cell {
            // Adds AND selects a fresh cell below (in command mode).
            self.add_code_block(window, cx);
        } else {
            self.advance_in_command_mode(window, cx);
        }
        self.apply_post_run_landing(was_edit_mode, window, cx);
    }

    /// Land in the configured mode after running a cell
    /// (`repl.notebook_run_landing_mode`): always command, always edit, or the
    /// mode the run was triggered from (`remember`).
    fn apply_post_run_landing(
        &mut self,
        was_edit_mode: bool,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let want_edit = match ReplSettings::get_global(cx).notebook_run_landing_mode {
            NotebookRunLandingMode::Command => false,
            NotebookRunLandingMode::Edit => true,
            NotebookRunLandingMode::Remember => was_edit_mode,
        };
        // Entering edit mode focuses the cell's editor, which would dismiss an
        // open kernel picker (dropping the cells awaiting the kernel choice) —
        // fall back to command mode in that case.
        if want_edit && !self.kernel_picker_handle.is_deployed() {
            self.enter_edit_mode(&EnterEditMode, window, cx);
        } else {
            self.enter_command_mode(window, cx);
        }
    }

    /// Select the cell that owns a toolbar button — so index-based actions
    /// target it even when the toolbar was shown on hover of a non-selected
    /// cell — then perform the requested action.
    fn handle_cell_toolbar_action(
        &mut self,
        cell_id: &CellId,
        action: CellToolbarAction,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let Some(index) = self.cell_order.iter().position(|id| id == cell_id) else {
            return;
        };
        self.selected_cell_index = index;
        self.notebook_mode = NotebookMode::Command;
        self.focus_handle.focus(window, cx);

        match action {
            CellToolbarAction::Run => self.run_current_cell(&Run, window, cx),
            CellToolbarAction::RunAbove => self.run_cells_above(&RunCellsAbove, window, cx),
            CellToolbarAction::RunBelow => self.run_cell_and_below(&RunCellAndBelow, window, cx),
            CellToolbarAction::AddBelow => self.add_cell_below(&AddCellBelow, window, cx),
            CellToolbarAction::Delete => self.delete_cell(&DeleteCell, window, cx),
        }
    }

    fn enter_edit_mode(&mut self, _: &EnterEditMode, window: &mut Window, cx: &mut Context<Self>) {
        self.notebook_mode = NotebookMode::Edit;
        self.disable_follow_on_edit(cx);
        if let Some(cell_id) = self.cell_order.get(self.selected_cell_index) {
            if let Some(cell) = self.cell_map.get(cell_id) {
                match cell {
                    Cell::Code(code_cell) => {
                        let editor = code_cell.read(cx).editor().clone();
                        window.focus(&editor.focus_handle(cx), cx);
                    }
                    Cell::Markdown(markdown_cell) => {
                        markdown_cell.update(cx, |cell, cx| {
                            cell.set_editing(true);
                            cx.notify();
                        });
                        let editor = markdown_cell.read(cx).editor().clone();
                        window.focus(&editor.focus_handle(cx), cx);
                    }
                    Cell::Raw(_) => {}
                }
            }
        }
        cx.notify();
    }

    fn enter_command_mode(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        self.notebook_mode = NotebookMode::Command;
        // Don't steal focus from an open kernel picker — grabbing focus
        // dismisses the popover, and the dismiss callback drops the cells
        // waiting on the kernel choice (this skipped the first cell when
        // shift-enter opened the picker and then advanced).
        if !self.kernel_picker_handle.is_deployed() {
            self.focus_handle.focus(window, cx);
        }
        cx.notify();
    }

    fn handle_enter_command_mode(
        &mut self,
        _: &EnterCommandMode,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        // Esc while ALREADY in command mode collapses a multi-cell selection
        // back to just the primary cell (phase 32) — VS Code style. Esc from
        // edit mode only switches modes; a fresh multi-selection made in
        // command mode stays until a second Esc.
        if self.notebook_mode == NotebookMode::Command && self.has_multi_selection() {
            self.collapse_selection();
        }
        self.enter_command_mode(window, cx);
    }

    /// Reload the notebook from disk. Prompts first when there are unsaved
    /// changes, since they would be lost.
    fn handle_reload_notebook(
        &mut self,
        _: &ReloadNotebook,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let project = self.project.clone();
        if self.is_dirty(cx) {
            let answer = window.prompt(
                PromptLevel::Warning,
                "Reload this notebook from disk?",
                Some("Your unsaved changes will be lost."),
                &["Reload", "Cancel"],
                cx,
            );
            cx.spawn_in(window, async move |this, cx| {
                if answer.await != Ok(0) {
                    return anyhow::Ok(());
                }
                this.update_in(cx, |this, window, cx| this.reload(project, window, cx))?
                    .await
            })
            .detach_and_log_err(cx);
        } else {
            self.reload(project, window, cx).detach_and_log_err(cx);
        }
    }

    /// Clear the outputs of the selected cell (the per-output "..." menu's
    /// Clear Output only clears one output).
    fn clear_selected_cell_outputs(
        &mut self,
        _: &ClearCellOutputs,
        _window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        for index in self.effective_selection() {
            if let Some(cell_id) = self.cell_order.get(index)
                && let Some(Cell::Code(cell)) = self.cell_map.get(cell_id)
            {
                self.execution_state_changed = true;
                cell.update(cx, |cell, cx| {
                    cell.clear_execution_record();
                    cx.notify();
                });
            }
        }
    }

    /// Advances to the next cell while staying in command mode (used by RunAndAdvance and shift-enter).
    fn advance_in_command_mode(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let count = self.cell_count();
        if count == 0 {
            return;
        }
        if self.selected_cell_index < count - 1 {
            self.selected_cell_index += 1;
            self.cell_list
                .scroll_to_reveal_item(self.selected_cell_index);
        }
        self.notebook_mode = NotebookMode::Command;
        // See enter_command_mode: focusing while the kernel picker is open
        // dismisses it and drops the awaiting cells.
        if !self.kernel_picker_handle.is_deployed() {
            self.focus_handle.focus(window, cx);
        }
        cx.notify();
    }

    fn open_notebook(&mut self, _: &OpenNotebook, window: &mut Window, cx: &mut Context<Self>) {
        window.dispatch_action(Box::new(Open::DEFAULT), cx);
    }

    /// Whether the multi-selection is one contiguous block. Block moves only
    /// support contiguous selections (a discontiguous move is ambiguous).
    fn selection_is_contiguous(&self) -> bool {
        let selection = self.effective_selection();
        match (selection.first(), selection.last()) {
            (Some(first), Some(last)) => last - first + 1 == selection.len(),
            _ => false,
        }
    }

    fn move_cell_up(&mut self, _window: &mut Window, cx: &mut Context<Self>) {
        let selection = self.effective_selection();
        let Some(&first) = selection.first() else {
            return;
        };
        if first == 0 {
            return;
        }
        if selection.len() == 1 {
            let from = self.selected_cell_index;
            let to = from - 1;
            self.cell_order.swap(from, to);
            self.selected_cell_index = to;
            self.collapse_selection();
            self.record_edit(CellEdit::Moved { from, to });
            cx.notify();
            return;
        }
        if !self.selection_is_contiguous() {
            log::info!("notebook: move ignored for a discontiguous multi-selection");
            return;
        }
        // Shift the whole block up one: move each member (top-down) one slot
        // up. Grouped so undo restores the block in one step.
        let primary = self.selected_cell_index;
        let anchor = self.selection_anchor;
        let mut edits = Vec::with_capacity(selection.len());
        for &index in &selection {
            self.raw_move_cell(index, index - 1);
            edits.push(CellEdit::Moved {
                from: index,
                to: index - 1,
            });
        }
        self.record_edit(CellEdit::Group(edits));
        // Re-establish the (shifted) selection that raw_move_cell collapsed.
        self.selected_cell_index = primary - 1;
        self.selection_anchor = anchor.map(|a| a - 1);
        self.selected_indices = selection.iter().map(|index| index - 1).collect();
        cx.notify();
    }

    fn move_cell_down(&mut self, _window: &mut Window, cx: &mut Context<Self>) {
        let selection = self.effective_selection();
        let count = self.cell_order.len();
        let Some(&last) = selection.last() else {
            return;
        };
        if count == 0 || last >= count - 1 {
            return;
        }
        if selection.len() == 1 {
            let from = self.selected_cell_index;
            let to = from + 1;
            self.cell_order.swap(from, to);
            self.selected_cell_index = to;
            self.collapse_selection();
            self.record_edit(CellEdit::Moved { from, to });
            cx.notify();
            return;
        }
        if !self.selection_is_contiguous() {
            log::info!("notebook: move ignored for a discontiguous multi-selection");
            return;
        }
        // Shift the block down one: move each member bottom-up.
        let primary = self.selected_cell_index;
        let anchor = self.selection_anchor;
        let mut edits = Vec::with_capacity(selection.len());
        for &index in selection.iter().rev() {
            self.raw_move_cell(index, index + 1);
            edits.push(CellEdit::Moved {
                from: index,
                to: index + 1,
            });
        }
        self.record_edit(CellEdit::Group(edits));
        self.selected_cell_index = primary + 1;
        self.selection_anchor = anchor.map(|a| a + 1);
        self.selected_indices = selection.iter().map(|index| index + 1).collect();
        cx.notify();
    }

    /// Inserts a cell at `index` (clamped), updates the list, and selects it.
    fn insert_cell(
        &mut self,
        index: usize,
        cell_id: CellId,
        cell: Cell,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let index = index.min(self.cell_order.len());
        self.cell_order.insert(index, cell_id.clone());
        self.cell_map.insert(cell_id, cell);
        self.selected_cell_index = index;
        // Indices shifted — a stale multi-selection would select wrong cells.
        self.collapse_selection();
        self.cell_list.splice(index..index, 1);
        // Best-effort synchronous reveal. This alone is NOT enough: the
        // just-spliced item is Unmeasured (zero height in the list's sum
        // tree), so this reveal computes a scroll that leaves the new cell's
        // top at the viewport's bottom edge — out of sight (bug #25).
        self.cell_list.scroll_to_reveal_item(index);
        // Re-reveal once the item has a real height. Next-frame callbacks run
        // at the START of a frame tick, BEFORE that frame's layout, so hop two
        // frames: the first frame's layout measures the item (it is within the
        // list's overdraw), and the second hop reveals with the true height —
        // then notifies so the corrected scroll actually paints.
        let list = self.cell_list.clone();
        let view = cx.entity_id();
        window.on_next_frame(move |window, _| {
            window.on_next_frame(move |_, cx| {
                list.scroll_to_reveal_item(index);
                cx.notify(view);
            });
        });
    }

    /// Index just after the selected cell (or 0 when the notebook is empty).
    fn index_below_selection(&self) -> usize {
        if self.cell_order.is_empty() {
            0
        } else {
            self.selected_cell_index + 1
        }
    }

    fn empty_cell_metadata() -> nbformat::v4::CellMetadata {
        serde_json::from_str("{}").expect("empty object should parse")
    }

    /// Reads the current editor text of a cell (may differ from its saved
    /// source if it has unsaved edits).
    fn cell_source_text(&self, cell_id: &CellId, cx: &App) -> String {
        let Some(cell) = self.cell_map.get(cell_id) else {
            return String::new();
        };
        let Some(editor) = cell.editor(cx) else {
            return String::new();
        };
        editor
            .read(cx)
            .buffer()
            .read(cx)
            .as_singleton()
            .map(|buffer| buffer.read(cx).text())
            .unwrap_or_default()
    }

    fn wire_code_cell(
        &mut self,
        cell_id: CellId,
        code_cell: &Entity<super::CodeCell>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let cell_id_for_run = cell_id.clone();
        cx.subscribe_in(
            code_cell,
            window,
            move |this, _cell, event, window, cx| match event {
                CellEvent::Run(cell_id) => this.execute_cell(cell_id.clone(), window, cx),
                CellEvent::FocusedIn(_) => this.select_cell_by_id(&cell_id_for_run, cx),
                CellEvent::ToolbarAction(cell_id, action) => {
                    this.handle_cell_toolbar_action(cell_id, *action, window, cx)
                }
                CellEvent::Stop(cell_id) => this.handle_cell_stop(cell_id, window, cx),
                CellEvent::ModifiedClick { id, shift } => {
                    this.handle_modified_click(id, *shift, window, cx)
                }
                CellEvent::PlainClick { id } => this.handle_plain_click(id, window, cx),
                CellEvent::MetadataChanged(_) => {
                    this.execution_state_changed = true;
                }
            },
        )
        .detach();

        let cell_id_for_editor = cell_id;
        let editor = code_cell.read(cx).editor().clone();
        cx.subscribe_in(&editor, window, move |this, _editor, event, window, cx| {
            this.on_cell_editor_event(&cell_id_for_editor, event, window, cx);
        })
        .detach();
    }

    fn wire_markdown_cell(
        &mut self,
        cell_id: CellId,
        markdown_cell: &Entity<super::MarkdownCell>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        cx.subscribe(
            markdown_cell,
            move |_this, cell, event: &MarkdownCellEvent, cx| match event {
                MarkdownCellEvent::FinishedEditing | MarkdownCellEvent::Run(_) => {
                    cell.update(cx, |cell, cx| {
                        cell.reparse_markdown(cx);
                    });
                }
            },
        )
        .detach();

        cx.subscribe_in(
            markdown_cell,
            window,
            |this, _cell, event: &CellEvent, window, cx| match event {
                CellEvent::ModifiedClick { id, shift } => {
                    this.handle_modified_click(id, *shift, window, cx)
                }
                CellEvent::PlainClick { id } => this.handle_plain_click(id, window, cx),
                _ => {}
            },
        )
        .detach();

        let cell_id_for_editor = cell_id;
        let editor = markdown_cell.read(cx).editor().clone();
        cx.subscribe_in(&editor, window, move |this, _editor, event, window, cx| {
            this.on_cell_editor_event(&cell_id_for_editor, event, window, cx);
        })
        .detach();
    }

    fn build_code_cell(
        &mut self,
        source: String,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> (CellId, Entity<super::CodeCell>) {
        let new_cell_id: CellId = Uuid::new_v4().into();
        let notebook_language = self.notebook_language.clone();
        let code_cell = cx.new(|cx| {
            super::CodeCell::new(
                super::CellSource::None,
                new_cell_id.clone(),
                Self::empty_cell_metadata(),
                source,
                notebook_language,
                window,
                cx,
            )
        });
        self.wire_code_cell(new_cell_id.clone(), &code_cell, window, cx);
        (new_cell_id, code_cell)
    }

    fn build_markdown_cell(
        &mut self,
        source: String,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> (CellId, Entity<super::MarkdownCell>) {
        let new_cell_id: CellId = Uuid::new_v4().into();
        let languages = self.languages.clone();
        let markdown_cell = cx.new(|cx| {
            super::MarkdownCell::new(
                new_cell_id.clone(),
                Self::empty_cell_metadata(),
                source,
                languages,
                window,
                cx,
            )
        });
        self.wire_markdown_cell(new_cell_id.clone(), &markdown_cell, window, cx);
        (new_cell_id, markdown_cell)
    }

    fn add_markdown_block(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let (cell_id, markdown_cell) = self.build_markdown_cell(String::new(), window, cx);
        let index = self.index_below_selection();
        self.insert_cell(
            index,
            cell_id.clone(),
            Cell::Markdown(markdown_cell),
            window,
            cx,
        );
        self.record_new_cell(index, &cell_id, cx);
        // Select the new cell in command mode (VS Code-style: press Enter to
        // edit). Staying in command mode keeps single-key shortcuts working.
        self.enter_command_mode(window, cx);
    }

    fn add_code_block(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let index = self.index_below_selection();
        self.add_code_cell_at(index, window, cx);
    }

    fn add_code_cell_at(&mut self, index: usize, window: &mut Window, cx: &mut Context<Self>) {
        let (cell_id, code_cell) = self.build_code_cell(String::new(), window, cx);
        self.insert_cell(index, cell_id.clone(), Cell::Code(code_cell), window, cx);
        self.record_new_cell(index, &cell_id, cx);
        self.enter_command_mode(window, cx);
    }

    /// Record a just-inserted cell (by id) as an undoable insertion.
    fn record_new_cell(&mut self, index: usize, cell_id: &CellId, cx: &mut Context<Self>) {
        if let Some(cell) = self.cell_map.get(cell_id) {
            let serialized = cell.to_nbformat_cell(cx);
            self.record_edit(CellEdit::Inserted {
                index,
                cell: serialized,
            });
        }
    }

    fn add_cell_above(&mut self, _: &AddCellAbove, window: &mut Window, cx: &mut Context<Self>) {
        let index = self.selected_cell_index.min(self.cell_order.len());
        self.add_code_cell_at(index, window, cx);
    }

    fn add_cell_below(&mut self, _: &AddCellBelow, window: &mut Window, cx: &mut Context<Self>) {
        self.add_code_block(window, cx);
    }

    fn delete_cell(&mut self, _: &DeleteCell, window: &mut Window, cx: &mut Context<Self>) {
        let targets = self.effective_selection();
        if targets.is_empty() {
            return;
        }
        // Deleting the whole notebook must still "do something": we delete the
        // cells and drop in one fresh empty code cell (below), never leaving an
        // empty notebook with nowhere to type.
        let deleting_all = targets.len() >= self.cell_order.len();

        // Delete bottom-up so earlier indices stay valid; group the edits so
        // undo restores the whole selection in one step.
        let mut edits = Vec::with_capacity(targets.len() + 1);
        for &index in targets.iter().rev() {
            let Some(cell_id) = self.cell_order.get(index).cloned() else {
                continue;
            };
            // Capture the cell (with live content) for undo before removing it.
            let serialized = self
                .cell_map
                .get(&cell_id)
                .map(|cell| cell.to_nbformat_cell(cx));
            self.raw_remove_cell(index, cx);
            if let Some(serialized) = serialized {
                edits.push(CellEdit::Deleted {
                    index,
                    cell: serialized,
                });
            }
        }

        // Replace an emptied notebook with a fresh code cell, in the SAME undo
        // group — so undo removes the fresh cell and restores the originals.
        if deleting_all {
            let (cell_id, code_cell) = self.build_code_cell(String::new(), window, cx);
            self.insert_cell(0, cell_id.clone(), Cell::Code(code_cell), window, cx);
            if let Some(cell) = self.cell_map.get(&cell_id) {
                edits.push(CellEdit::Inserted {
                    index: 0,
                    cell: cell.to_nbformat_cell(cx),
                });
            }
        }

        match edits.len() {
            0 => {}
            1 => self.record_edit(edits.remove(0)),
            _ => self.record_edit(CellEdit::Group(edits)),
        }

        self.notebook_mode = NotebookMode::Command;
        self.focus_handle.focus(window, cx);
        self.cell_list
            .scroll_to_reveal_item(self.selected_cell_index);
        cx.notify();
    }

    fn copy_cell(&mut self, _: &CopyCell, _window: &mut Window, cx: &mut Context<Self>) {
        // An active mouse selection inside an output wins over cell copy, so
        // drag-selecting output text and hitting the copy shortcut copies that
        // text. Click-away clears output selections, so at most one exists.
        if let Some(text) = self.output_selection_text(cx) {
            cx.write_to_clipboard(ClipboardItem::new_string(text));
            return;
        }
        self.copy_cells_to_clipboard(cx);
    }

    /// Serializes the selected cell(s) to the clipboard — the cell-level copy
    /// used by both copy (when no output text is selected) and cut.
    fn copy_cells_to_clipboard(&mut self, cx: &mut Context<Self>) {
        let cells: Vec<nbformat::v4::Cell> = self
            .effective_selection()
            .into_iter()
            .filter_map(|index| self.cell_order.get(index))
            .filter_map(|cell_id| self.cell_map.get(cell_id))
            .map(|cell| cell.to_nbformat_cell(cx))
            .collect();
        if cells.is_empty() {
            return;
        }
        // A single cell keeps the original single-object format (compatible
        // with older copies); a multi-selection serializes as a JSON array.
        let json = if cells.len() == 1 {
            serde_json::to_string(&cells[0])
        } else {
            serde_json::to_string(&cells)
        };
        match json {
            Ok(json) => cx.write_to_clipboard(ClipboardItem::new_string(json)),
            Err(error) => log::error!("notebook: failed to copy cell(s): {error}"),
        }
    }

    /// Text of the active in-place output selection anywhere in the notebook,
    /// if one exists.
    fn output_selection_text(&self, cx: &App) -> Option<String> {
        self.cell_map.values().find_map(|cell| match cell {
            Cell::Code(code_cell) => code_cell
                .read(cx)
                .outputs()
                .iter()
                .find_map(|output| output.selection_text(cx)),
            _ => None,
        })
    }

    fn cut_cell(&mut self, _: &CutCell, window: &mut Window, cx: &mut Context<Self>) {
        self.copy_cells_to_clipboard(cx);
        self.delete_cell(&DeleteCell, window, cx);
    }

    /// A copy of `cell` with `source` replaced and any execution record
    /// (count + outputs) cleared; id and metadata are preserved.
    fn nbformat_cell_with_source(cell: &nbformat::v4::Cell, text: &str) -> nbformat::v4::Cell {
        let source: Vec<String> = text.lines().map(|line| format!("{line}\n")).collect();
        match cell {
            nbformat::v4::Cell::Code { id, metadata, .. } => nbformat::v4::Cell::Code {
                id: id.clone(),
                metadata: metadata.clone(),
                execution_count: None,
                source,
                outputs: Vec::new(),
            },
            nbformat::v4::Cell::Markdown {
                id,
                metadata,
                attachments,
                ..
            } => nbformat::v4::Cell::Markdown {
                id: id.clone(),
                metadata: metadata.clone(),
                source,
                attachments: attachments.clone(),
            },
            nbformat::v4::Cell::Raw { id, metadata, .. } => nbformat::v4::Cell::Raw {
                id: id.clone(),
                metadata: metadata.clone(),
                source,
            },
        }
    }

    /// Split the selected cell at the cursor into two cells of the same type
    /// (Jupyter's ctrl-shift-minus), as one undo group. The top half keeps the
    /// cell's id and metadata (collapse state); the bottom half gets a fresh
    /// identity; the execution record is cleared on both.
    fn split_cell(&mut self, _: &SplitCell, window: &mut Window, cx: &mut Context<Self>) {
        if self.notebook_mode != NotebookMode::Edit {
            return;
        }
        let index = self.selected_cell_index;
        let Some(cell_id) = self.cell_order.get(index) else {
            return;
        };
        let Some(cell) = self.cell_map.get(cell_id) else {
            return;
        };
        let Some(editor) = cell.editor(cx).cloned() else {
            return;
        };
        let original = cell.to_nbformat_cell(cx);

        let (text, offset) = editor.update(cx, |editor, cx| {
            let snapshot = editor.display_snapshot(cx);
            let offset = editor
                .selections
                .newest::<multi_buffer::MultiBufferOffset>(&snapshot)
                .head()
                .0;
            (editor.text(cx), offset)
        });
        // Splitting at a line boundary should not leave a stray blank line on
        // either half, so one newline at the split point is absorbed.
        let top_text = text[..offset].strip_suffix('\n').unwrap_or(&text[..offset]);
        let bottom_text = text[offset..].strip_prefix('\n').unwrap_or(&text[offset..]);

        let top = Self::nbformat_cell_with_source(&original, top_text);
        let mut bottom = Self::nbformat_cell_with_source(&original, bottom_text);
        match &mut bottom {
            nbformat::v4::Cell::Code { id, metadata, .. }
            | nbformat::v4::Cell::Raw { id, metadata, .. } => {
                *id = Uuid::new_v4().into();
                *metadata = Self::empty_cell_metadata();
            }
            nbformat::v4::Cell::Markdown {
                id,
                metadata,
                attachments,
                ..
            } => {
                *id = Uuid::new_v4().into();
                *metadata = Self::empty_cell_metadata();
                *attachments = None;
            }
        }

        self.raw_replace_cell(index, top.clone(), window, cx);
        self.raw_insert_cell(index + 1, bottom.clone(), window, cx);
        self.record_edit(CellEdit::Group(vec![
            CellEdit::Converted {
                index,
                before: original,
                after: top,
            },
            CellEdit::Inserted {
                index: index + 1,
                cell: bottom,
            },
        ]));

        // Continue editing in the bottom half, cursor at its start (Jupyter
        // behavior). `raw_insert_cell` already selected it.
        self.enter_edit_mode(&EnterEditMode, window, cx);
        self.cell_list
            .scroll_to_reveal_item(self.selected_cell_index);
        cx.notify();
    }

    /// Join the contiguous multi-selection (or the selected cell with the one
    /// below) into one cell: sources concatenated, outputs cleared, one undo
    /// group. Restricted to cells of the same type.
    fn join_cells(&mut self, _: &JoinCells, window: &mut Window, cx: &mut Context<Self>) {
        let selection = self.effective_selection();
        let indices: Vec<usize> = if selection.len() > 1 {
            if !selection.windows(2).all(|pair| pair[1] == pair[0] + 1) {
                Self::show_env_toast(
                    window,
                    cx,
                    "Join Cells needs a contiguous selection".to_string(),
                    true,
                );
                return;
            }
            selection
        } else {
            let index = self.selected_cell_index;
            if index + 1 >= self.cell_order.len() {
                return;
            }
            vec![index, index + 1]
        };

        let snapshots: Vec<nbformat::v4::Cell> = indices
            .iter()
            .filter_map(|index| self.cell_order.get(*index))
            .filter_map(|cell_id| self.cell_map.get(cell_id))
            .map(|cell| cell.to_nbformat_cell(cx))
            .collect();
        if snapshots.len() != indices.len() {
            return;
        }
        let first_discriminant = std::mem::discriminant(&snapshots[0]);
        if snapshots
            .iter()
            .any(|cell| std::mem::discriminant(cell) != first_discriminant)
        {
            Self::show_env_toast(
                window,
                cx,
                "Only cells of the same type can be joined".to_string(),
                true,
            );
            return;
        }

        let merged_text = snapshots
            .iter()
            .map(|cell| cell.source().concat().trim_end_matches('\n').to_string())
            .collect::<Vec<_>>()
            .join("\n\n");
        let merged = Self::nbformat_cell_with_source(&snapshots[0], &merged_text);

        let first_index = indices[0];
        self.raw_replace_cell(first_index, merged.clone(), window, cx);
        let mut edits = vec![CellEdit::Converted {
            index: first_index,
            before: snapshots[0].clone(),
            after: merged,
        }];
        // Each removal happens at the same index because the remaining cells
        // shift up; the recorded order replays correctly forward (redo) and
        // reversed (undo).
        for snapshot in &snapshots[1..] {
            self.raw_remove_cell(first_index + 1, cx);
            edits.push(CellEdit::Deleted {
                index: first_index + 1,
                cell: snapshot.clone(),
            });
        }
        self.record_edit(CellEdit::Group(edits));

        self.selected_cell_index = first_index;
        self.collapse_selection();
        self.notebook_mode = NotebookMode::Command;
        self.focus_handle.focus(window, cx);
        self.cell_list.scroll_to_reveal_item(first_index);
        cx.notify();
    }

    /// Parse nbformat cell(s) from clipboard text, if present: either a single
    /// cell object or an array of cells (multi-selection copy).
    fn clipboard_cells(cx: &mut Context<Self>) -> Vec<nbformat::v4::Cell> {
        let Some(text) = cx.read_from_clipboard().and_then(|item| item.text()) else {
            return Vec::new();
        };
        if let Ok(cell) = serde_json::from_str::<nbformat::v4::Cell>(&text) {
            return vec![cell];
        }
        serde_json::from_str::<Vec<nbformat::v4::Cell>>(&text).unwrap_or_default()
    }

    fn paste_cell(&mut self, _: &PasteCell, window: &mut Window, cx: &mut Context<Self>) {
        self.paste_cells_at(self.index_below_selection(), window, cx);
    }

    fn paste_cell_above(
        &mut self,
        _: &PasteCellAbove,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        // Insert before the primary cell (index 0 when the notebook is empty).
        self.paste_cells_at(self.selected_cell_index, window, cx);
    }

    /// Insert the clipboard cell(s) starting at `base`, as one undo operation.
    fn paste_cells_at(&mut self, base: usize, window: &mut Window, cx: &mut Context<Self>) {
        let cells = Self::clipboard_cells(cx);
        if cells.is_empty() {
            return;
        }
        let mut edits = Vec::with_capacity(cells.len());
        for (offset, cell) in cells.into_iter().enumerate() {
            edits.push(self.insert_nbformat_cell(base + offset, cell, window, cx));
        }
        match edits.len() {
            0 => {}
            1 => self.record_edit(edits.remove(0)),
            _ => self.record_edit(CellEdit::Group(edits)),
        }
    }

    fn duplicate_cell(&mut self, _: &DuplicateCell, window: &mut Window, cx: &mut Context<Self>) {
        let Some(cell_id) = self.cell_order.get(self.selected_cell_index) else {
            return;
        };
        let Some(cell) = self.cell_map.get(cell_id) else {
            return;
        };
        let nbformat_cell = cell.to_nbformat_cell(cx);
        let index = self.selected_cell_index + 1;
        let edit = self.insert_nbformat_cell(index, nbformat_cell, window, cx);
        self.record_edit(edit);
    }

    /// Build a live cell from an nbformat cell (with a fresh id), wire it, and
    /// insert it at `index`. Shared by paste and duplicate.
    fn insert_nbformat_cell(
        &mut self,
        index: usize,
        cell: nbformat::v4::Cell,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> CellEdit {
        // A pasted/duplicated cell must get a fresh id, or it would collide
        // with the source cell's id.
        let new_cell_id: CellId = Uuid::new_v4().into();
        let cell = match cell {
            nbformat::v4::Cell::Markdown {
                metadata,
                source,
                attachments,
                ..
            } => nbformat::v4::Cell::Markdown {
                id: new_cell_id,
                metadata,
                source,
                attachments,
            },
            nbformat::v4::Cell::Code {
                metadata,
                execution_count,
                source,
                outputs,
                ..
            } => nbformat::v4::Cell::Code {
                id: new_cell_id,
                metadata,
                execution_count,
                source,
                outputs,
            },
            nbformat::v4::Cell::Raw {
                metadata, source, ..
            } => nbformat::v4::Cell::Raw {
                id: new_cell_id,
                metadata,
                source,
            },
        };

        self.raw_insert_cell(index, cell.clone(), window, cx);
        self.notebook_mode = NotebookMode::Command;
        self.focus_handle.focus(window, cx);
        cx.notify();
        CellEdit::Inserted { index, cell }
    }

    // --- Structural primitives (no undo recording, no focus/mode side effects) ---

    /// Build a live cell from `cell` (PRESERVING its id), wire it, and insert at
    /// `index`. Selects the inserted cell.
    fn raw_insert_cell(
        &mut self,
        index: usize,
        cell: nbformat::v4::Cell,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let cell_id = cell.id().clone();
        let languages = self.languages.clone();
        let notebook_language = self.notebook_language.clone();
        let cell_entity = Cell::load(&cell, &languages, notebook_language, window, cx);
        match &cell_entity {
            Cell::Code(code_cell) => self.wire_code_cell(cell_id.clone(), code_cell, window, cx),
            Cell::Markdown(markdown_cell) => {
                self.wire_markdown_cell(cell_id.clone(), markdown_cell, window, cx)
            }
            Cell::Raw(_) => {}
        }
        self.insert_cell(index, cell_id, cell_entity, window, cx);
    }

    /// Remove the cell at `index`, cleaning up execution/queue state.
    fn raw_remove_cell(&mut self, index: usize, cx: &mut Context<Self>) {
        if index >= self.cell_order.len() {
            return;
        }
        let cell_id = self.cell_order.remove(index);
        self.cell_map.remove(&cell_id);
        self.execution_requests
            .retain(|_, mapped| mapped != &cell_id);
        self.pending_executions.retain(|mapped| mapped != &cell_id);
        self.cells_awaiting_kernel_choice
            .retain(|mapped| mapped != &cell_id);
        if self.active_run_cell.as_ref() == Some(&cell_id) || self.run_queue.contains(&cell_id) {
            self.cancel_run_queue(cx);
        }
        self.cell_list.splice(index..index + 1, 0);
        self.selected_cell_index = index.min(self.cell_order.len().saturating_sub(1));
        self.collapse_selection();
    }

    /// Move the cell at `from` to `to` (count unchanged; no list splice needed).
    fn raw_move_cell(&mut self, from: usize, to: usize) {
        if from >= self.cell_order.len() || to >= self.cell_order.len() {
            return;
        }
        let cell_id = self.cell_order.remove(from);
        self.cell_order.insert(to, cell_id);
        self.selected_cell_index = to;
        // Callers that move a multi-selected block re-establish the selection
        // themselves after all the moves.
        self.collapse_selection();
    }

    /// Replace the cell at `index` with a fresh live cell built from `cell`.
    fn raw_replace_cell(
        &mut self,
        index: usize,
        cell: nbformat::v4::Cell,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if index >= self.cell_order.len() {
            return;
        }
        let old_id = self.cell_order[index].clone();
        self.cell_map.remove(&old_id);
        self.execution_requests
            .retain(|_, mapped| mapped != &old_id);
        self.pending_executions.retain(|mapped| mapped != &old_id);
        self.cells_awaiting_kernel_choice
            .retain(|mapped| mapped != &old_id);

        let cell_id = cell.id().clone();
        let languages = self.languages.clone();
        let notebook_language = self.notebook_language.clone();
        let cell_entity = Cell::load(&cell, &languages, notebook_language, window, cx);
        match &cell_entity {
            Cell::Code(code_cell) => self.wire_code_cell(cell_id.clone(), code_cell, window, cx),
            Cell::Markdown(markdown_cell) => {
                self.wire_markdown_cell(cell_id.clone(), markdown_cell, window, cx)
            }
            Cell::Raw(_) => {}
        }
        self.cell_order[index] = cell_id.clone();
        self.cell_map.insert(cell_id, cell_entity);
        self.cell_list.splice(index..index + 1, 1);
        self.selected_cell_index = index;
    }

    fn record_edit(&mut self, edit: CellEdit) {
        self.undo_stack.push(edit);
        self.redo_stack.clear();
    }

    /// Reverse one edit (recursively for groups, whose members are undone in
    /// reverse of the order they were applied).
    fn apply_undo_edit(&mut self, edit: &CellEdit, window: &mut Window, cx: &mut Context<Self>) {
        match edit {
            CellEdit::Inserted { index, .. } => self.raw_remove_cell(*index, cx),
            CellEdit::Deleted { index, cell } => {
                self.raw_insert_cell(*index, cell.clone(), window, cx)
            }
            CellEdit::Moved { from, to } => self.raw_move_cell(*to, *from),
            CellEdit::Converted { index, before, .. } => {
                self.raw_replace_cell(*index, before.clone(), window, cx)
            }
            CellEdit::Group(edits) => {
                for edit in edits.iter().rev() {
                    self.apply_undo_edit(edit, window, cx);
                }
            }
        }
    }

    /// Re-apply one edit (recursively for groups, in applied order).
    fn apply_redo_edit(&mut self, edit: &CellEdit, window: &mut Window, cx: &mut Context<Self>) {
        match edit {
            CellEdit::Inserted { index, cell } => {
                self.raw_insert_cell(*index, cell.clone(), window, cx)
            }
            CellEdit::Deleted { index, .. } => self.raw_remove_cell(*index, cx),
            CellEdit::Moved { from, to } => self.raw_move_cell(*from, *to),
            CellEdit::Converted { index, after, .. } => {
                self.raw_replace_cell(*index, after.clone(), window, cx)
            }
            CellEdit::Group(edits) => {
                for edit in edits {
                    self.apply_redo_edit(edit, window, cx);
                }
            }
        }
    }

    fn undo_cell_op(&mut self, _: &UndoCellOp, window: &mut Window, cx: &mut Context<Self>) {
        let Some(edit) = self.undo_stack.pop() else {
            return;
        };
        self.apply_undo_edit(&edit, window, cx);
        self.redo_stack.push(edit);
        self.after_undo_redo(window, cx);
    }

    fn redo_cell_op(&mut self, _: &RedoCellOp, window: &mut Window, cx: &mut Context<Self>) {
        let Some(edit) = self.redo_stack.pop() else {
            return;
        };
        self.apply_redo_edit(&edit, window, cx);
        self.undo_stack.push(edit);
        self.after_undo_redo(window, cx);
    }

    fn after_undo_redo(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        self.notebook_mode = NotebookMode::Command;
        self.focus_handle.focus(window, cx);
        if !self.cell_order.is_empty() {
            self.cell_list
                .scroll_to_reveal_item_top_aligned(self.selected_cell_index);
        }
        cx.notify();
    }

    fn convert_to_markdown(
        &mut self,
        _: &ConvertToMarkdown,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.convert_selected_cell(true, window, cx);
    }

    fn convert_to_code(&mut self, _: &ConvertToCode, window: &mut Window, cx: &mut Context<Self>) {
        self.convert_selected_cell(false, window, cx);
    }

    fn convert_selected_cell(
        &mut self,
        to_markdown: bool,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        // Converting collapses the multi-selection (cell entities are rebuilt),
        // so capture the target indices first and group the edits for undo.
        let selection = self.effective_selection();
        let primary = self.selected_cell_index;
        let anchor = self.selection_anchor;
        let multi = selection.len() > 1;
        let mut edits = Vec::with_capacity(selection.len());
        for index in &selection {
            if let Some(edit) = self.convert_cell_at(*index, to_markdown, window, cx) {
                edits.push(edit);
            }
        }
        match edits.len() {
            0 => return,
            1 => self.record_edit(edits.remove(0)),
            _ => self.record_edit(CellEdit::Group(edits)),
        }
        if multi {
            // Conversion keeps cells in place — restore the selection.
            self.selected_cell_index = primary;
            self.selection_anchor = anchor;
            self.selected_indices = selection.into_iter().collect();
        }

        self.notebook_mode = NotebookMode::Command;
        self.focus_handle.focus(window, cx);
        cx.notify();
    }

    /// Convert one cell (by index) to markdown/code, returning the undo edit.
    /// No-op (None) when the cell is already the requested type.
    fn convert_cell_at(
        &mut self,
        index: usize,
        to_markdown: bool,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Option<CellEdit> {
        let cell_id = self.cell_order.get(index).cloned()?;
        let is_markdown = matches!(self.cell_map.get(&cell_id), Some(Cell::Markdown(_)));
        if is_markdown == to_markdown {
            return None;
        }

        let before = self
            .cell_map
            .get(&cell_id)
            .map(|cell| cell.to_nbformat_cell(cx))?;
        let source = self.cell_source_text(&cell_id, cx);
        let new_cell_id: CellId = Uuid::new_v4().into();
        let after = if to_markdown {
            nbformat::v4::Cell::Markdown {
                id: new_cell_id,
                metadata: Self::empty_cell_metadata(),
                source: vec![source],
                attachments: None,
            }
        } else {
            nbformat::v4::Cell::Code {
                id: new_cell_id,
                metadata: Self::empty_cell_metadata(),
                execution_count: None,
                source: vec![source],
                outputs: vec![],
            }
        };

        self.raw_replace_cell(index, after.clone(), window, cx);
        Some(CellEdit::Converted {
            index,
            before,
            after,
        })
    }

    fn run_cells_above(&mut self, _: &RunCellsAbove, window: &mut Window, cx: &mut Context<Self>) {
        let end = self.selected_cell_index.min(self.cell_order.len());
        let cells: Vec<CellId> = self.cell_order[..end].to_vec();
        self.run_cell_batch(cells, window, cx);
    }

    fn run_cell_and_below(
        &mut self,
        _: &RunCellAndBelow,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let start = self.selected_cell_index;
        if start >= self.cell_order.len() {
            return;
        }
        let cells: Vec<CellId> = self.cell_order[start..].to_vec();
        self.run_cell_batch(cells, window, cx);
    }

    fn cell_count(&self) -> usize {
        self.cell_map.len()
    }

    fn selected_index(&self) -> usize {
        self.selected_cell_index
    }

    /// Collapse any multi-selection down to the primary cell.
    fn collapse_selection(&mut self) {
        self.selected_indices.clear();
        self.selection_anchor = None;
    }

    /// The full selection, sorted: the multi-selection when active, otherwise
    /// just the primary cell.
    fn effective_selection(&self) -> Vec<usize> {
        if self.selected_indices.len() > 1 {
            self.selected_indices.iter().copied().collect()
        } else {
            vec![self.selected_cell_index]
        }
    }

    fn has_multi_selection(&self) -> bool {
        self.selected_indices.len() > 1
    }

    fn is_index_selected(&self, index: usize) -> bool {
        if self.selected_indices.len() > 1 {
            self.selected_indices.contains(&index)
        } else {
            index == self.selected_cell_index
        }
    }

    /// Replace the selection with the contiguous range anchor..=primary.
    fn select_range(&mut self, anchor: usize, primary: usize) {
        self.selection_anchor = Some(anchor);
        self.selected_cell_index = primary;
        self.selected_indices = (anchor.min(primary)..=anchor.max(primary)).collect();
    }

    /// Extend the shift-range selection one cell down/up (shift-down/up in
    /// command mode).
    fn extend_selection(&mut self, direction: i32, window: &mut Window, cx: &mut Context<Self>) {
        let count = self.cell_count();
        if count == 0 {
            return;
        }
        let anchor = self.selection_anchor.unwrap_or(self.selected_cell_index);
        let primary = if direction > 0 {
            (self.selected_cell_index + 1).min(count - 1)
        } else {
            self.selected_cell_index.saturating_sub(1)
        };
        self.select_range(anchor, primary);
        self.cell_list.scroll_to_reveal_item(primary);
        self.notebook_mode = NotebookMode::Command;
        if !self.kernel_picker_handle.is_deployed() {
            self.focus_handle.focus(window, cx);
        }
        cx.notify();
    }

    /// Extend the shift-range selection all the way to the first or last cell
    /// (shift-home / shift-end in command mode). The primary moves to that
    /// boundary cell, like the equivalent text-editing motion.
    fn extend_selection_to_boundary(
        &mut self,
        to_end: bool,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let count = self.cell_count();
        if count == 0 {
            return;
        }
        let anchor = self.selection_anchor.unwrap_or(self.selected_cell_index);
        let primary = if to_end { count - 1 } else { 0 };
        self.select_range(anchor, primary);
        self.cell_list.scroll_to_reveal_item_top_aligned(primary);
        self.notebook_mode = NotebookMode::Command;
        if !self.kernel_picker_handle.is_deployed() {
            self.focus_handle.focus(window, cx);
        }
        cx.notify();
    }

    /// Select every cell as one contiguous range (ctrl/cmd-a in command mode):
    /// anchor at the first cell, primary at the last. Does not scroll — the
    /// viewport stays where it is, like select-all in a text editor.
    fn select_all_cells(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let count = self.cell_count();
        if count == 0 {
            return;
        }
        self.select_range(0, count - 1);
        self.notebook_mode = NotebookMode::Command;
        if !self.kernel_picker_handle.is_deployed() {
            self.focus_handle.focus(window, cx);
        }
        cx.notify();
    }

    /// A shift- or ctrl/cmd-click on a cell (see `CellEvent::ModifiedClick`).
    fn handle_modified_click(
        &mut self,
        cell_id: &CellId,
        shift: bool,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let Some(index) = self.cell_order.iter().position(|id| id == cell_id) else {
            return;
        };
        if shift {
            // Range from the anchor (or current primary) to the clicked cell.
            let anchor = self.selection_anchor.unwrap_or(self.selected_cell_index);
            self.select_range(anchor, index);
        } else {
            // ctrl/cmd-click: toggle the cell in a discontiguous selection.
            let mut set = if self.selected_indices.len() > 1 {
                self.selected_indices.clone()
            } else {
                BTreeSet::from([self.selected_cell_index])
            };
            if set.contains(&index) && set.len() > 1 {
                set.remove(&index);
                self.selected_cell_index = *set.iter().next().unwrap_or(&0);
            } else {
                set.insert(index);
                self.selected_cell_index = index;
            }
            self.selection_anchor = Some(self.selected_cell_index);
            if set.len() > 1 {
                self.selected_indices = set;
            } else {
                self.collapse_selection();
            }
        }
        self.notebook_mode = NotebookMode::Command;
        if !self.kernel_picker_handle.is_deployed() {
            self.focus_handle.focus(window, cx);
        }
        cx.notify();
    }

    fn select_cell_by_id(&mut self, cell_id: &CellId, cx: &mut Context<Self>) {
        if let Some(index) = self.cell_order.iter().position(|id| id == cell_id) {
            self.selected_cell_index = index;
            self.collapse_selection();
            self.notebook_mode = NotebookMode::Edit;
            self.disable_follow_on_edit(cx);
            cx.notify();
        }
    }

    /// A plain (unmodified) click on a cell outside its editor — the gutter,
    /// margins, or output area (see `CellEvent::PlainClick`). Selects just this
    /// cell and enters command mode. A click that lands on the editor still
    /// focuses it and enters edit mode via the editor's own focus event, which
    /// fires after this (the classifier does not stop propagation for plain
    /// clicks), so this only "wins" for clicks that miss the editor.
    fn handle_plain_click(
        &mut self,
        cell_id: &CellId,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if let Some(index) = self.cell_order.iter().position(|id| id == cell_id) {
            self.selected_cell_index = index;
            self.collapse_selection();
            self.enter_command_mode(window, cx);
        }
    }

    pub fn set_selected_index(
        &mut self,
        index: usize,
        jump_to_index: bool,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.selected_cell_index = index;
        self.collapse_selection();
        let current_index = self.selected_cell_index;

        // in the future we may have some `on_cell_change` event that we want to fire here

        if jump_to_index {
            self.jump_to_cell(current_index, window, cx);
        }
    }

    fn select_next(
        &mut self,
        _: &menu::SelectNext,
        selection_mode: SelectionMode,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let count = self.cell_count();
        if count > 0 {
            let index = self.selected_index();
            let ix = if index == count - 1 {
                count - 1
            } else {
                index + 1
            };
            // Command-mode cell navigation (`SelectOnly`) reveals the whole
            // cell, top-aligned — you are choosing a CELL, so showing as much
            // of it as possible is what you want. Crossing a cell boundary
            // while EDITING (`SelectAndMove`) is different: you are following a
            // LINE, so snapping the new cell to the top throws away your place
            // for no reason. Skip the reveal there and let the cursor move
            // below emit `SelectionsChanged`, which runs the ordinary
            // minimal follow — it scrolls only if the line being moved to is
            // actually out of view, and only far enough to show it.
            let reveal_whole_cell = selection_mode == SelectionMode::SelectOnly;
            self.set_selected_index(ix, reveal_whole_cell, window, cx);

            if selection_mode == SelectionMode::SelectAndMove
                && let Some(cell) = self.get_selected_cell()
            {
                cell.move_to(MovementDirection::Start, window, cx);
            }

            cx.notify();
        }
    }

    fn select_previous(
        &mut self,
        _: &menu::SelectPrevious,
        selection_mode: SelectionMode,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let count = self.cell_count();
        if count > 0 {
            let index = self.selected_index();
            let ix = if index == 0 { 0 } else { index - 1 };
            // Command-mode cell navigation (`SelectOnly`) reveals the whole
            // cell, top-aligned — you are choosing a CELL, so showing as much
            // of it as possible is what you want. Crossing a cell boundary
            // while EDITING (`SelectAndMove`) is different: you are following a
            // LINE, so snapping the new cell to the top throws away your place
            // for no reason. Skip the reveal there and let the cursor move
            // below emit `SelectionsChanged`, which runs the ordinary
            // minimal follow — it scrolls only if the line being moved to is
            // actually out of view, and only far enough to show it.
            let reveal_whole_cell = selection_mode == SelectionMode::SelectOnly;
            self.set_selected_index(ix, reveal_whole_cell, window, cx);

            if selection_mode == SelectionMode::SelectAndMove
                && let Some(cell) = self.get_selected_cell()
            {
                cell.move_to(MovementDirection::End, window, cx);
            }

            cx.notify();
        }
    }

    pub fn select_first(
        &mut self,
        _: &SelectFirstCell,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let count = self.cell_count();
        if count > 0 {
            self.set_selected_index(0, true, window, cx);
            cx.notify();
        }
    }

    pub fn select_last(&mut self, _: &SelectLastCell, window: &mut Window, cx: &mut Context<Self>) {
        let count = self.cell_count();
        if count > 0 {
            // Select the last cell WITHOUT a cumulative-height reveal (which
            // lands short when cells above are unmeasured or hold stale heights
            // in a large notebook), then anchor on the end: `scroll_to_end`
            // walks backwards from the last item, so it reaches the true bottom
            // regardless of measurement state.
            self.set_selected_index(count - 1, false, window, cx);
            self.cell_list.scroll_to_end();
            cx.notify();
        }
    }

    /// Shared handling for a cell editor's events: track focus for selection,
    /// and follow the cursor while editing.
    fn on_cell_editor_event(
        &mut self,
        cell_id: &CellId,
        event: &editor::EditorEvent,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        match event {
            editor::EditorEvent::Focused => self.select_cell_by_id(cell_id, cx),
            editor::EditorEvent::SelectionsChanged { .. } => {
                if self.notebook_mode == NotebookMode::Edit
                    && let Some(index) = self.cell_order.iter().position(|id| id == cell_id)
                    && index == self.selected_cell_index
                {
                    self.follow_cursor_in_cell(index, window, cx);
                }
            }
            _ => {}
        }
    }

    fn jump_to_cell(&mut self, index: usize, _window: &mut Window, _cx: &mut Context<Self>) {
        // Top-align a cell that doesn't fit the viewport (regardless of travel
        // direction), otherwise minimally reveal it.
        self.cell_list.scroll_to_reveal_item_top_aligned(index);
    }

    /// Keep the cursor visible while editing a tall cell: scroll the notebook
    /// only when the cursor actually sits outside the viewport.
    ///
    /// The cursor's position is derived from the editor's LAID-OUT bounds and
    /// the cursor's row, never from where the cursor was painted. A painted
    /// position only exists for a cursor that was on screen, so relying on it
    /// meant that once the cursor left the viewport there was nothing left to
    /// tell us it had — the follow went permanently dead, and no amount of
    /// further arrow presses recovered it. Cell editors are `SizeByContent`,
    /// with no internal scroll, so their height is exactly their rows: the row
    /// maps linearly onto those bounds, which are known whether or not the row
    /// is visible.
    fn follow_cursor_in_cell(
        &mut self,
        index: usize,
        _window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let Some(cell) = self
            .cell_order
            .get(index)
            .and_then(|id| self.cell_map.get(id))
        else {
            return;
        };
        let Some(editor) = cell.editor(cx).cloned() else {
            return;
        };

        if self.cell_list.bounds_for_item(index).is_none() {
            // Not currently laid out (e.g. scrolled far away): reveal it.
            self.cell_list.scroll_to_reveal_item_top_aligned(index);
            return;
        }
        let viewport = self.cell_list.viewport_bounds();
        if viewport.size.height <= px(0.) {
            return;
        }

        let (cursor_row, total_rows) = editor.update(cx, |editor, cx| {
            let snapshot = editor.display_snapshot(cx);
            let cursor_row = snapshot
                .max_point()
                .row()
                .min(editor.selections.newest_display(&snapshot).head().row());
            (cursor_row.0, snapshot.max_point().row().0)
        });

        // The editor's own laid-out bounds, which exist regardless of what is
        // currently on screen.
        let Some(editor_bounds) = editor.read(cx).last_bounds().copied() else {
            return;
        };
        let rows = (total_rows + 1).max(1) as f32;
        let line_height = editor_bounds.size.height / rows;
        let cursor_top = editor_bounds.top() + line_height * cursor_row as f32;
        let cursor_bottom = cursor_top + line_height;

        // Scroll ONLY when the cursor's line is actually outside the viewport,
        // and then only far enough to bring that one line fully in. Any lead
        // here would move the viewport while the cursor is still perfectly
        // visible, which is disorienting while editing and (with the pointer
        // held) compounds into runaway scrolling.
        if cursor_top < viewport.top() {
            self.cell_list.scroll_by(cursor_top - viewport.top());
        } else if cursor_bottom > viewport.bottom() {
            self.cell_list.scroll_by(cursor_bottom - viewport.bottom());
        }
    }

    fn button_group(_window: &mut Window, cx: &mut Context<Self>) -> Div {
        v_flex()
            .gap(DynamicSpacing::Base04.rems(cx))
            .items_center()
            .w(px(CONTROL_SIZE + 4.0))
            .overflow_hidden()
            .rounded(px(5.))
            .bg(cx.theme().colors().title_bar_background)
            .p_px()
            .border_1()
            .border_color(cx.theme().colors().border)
    }

    fn render_notebook_control(
        id: impl Into<SharedString>,
        icon: IconName,
        _window: &mut Window,
        _cx: &mut Context<Self>,
    ) -> IconButton {
        let id: ElementId = ElementId::Name(id.into());
        IconButton::new(id, icon).width(px(CONTROL_SIZE))
    }

    fn render_notebook_controls(
        &self,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> impl IntoElement {
        let has_execution_record = self.has_execution_record(window, cx);
        let has_running_cell = self.running_cell_index(cx).is_some();
        let following = self.follow_running_cell;

        v_flex()
            .max_w(px(CONTROL_SIZE + 4.0))
            .items_center()
            .gap(DynamicSpacing::Base16.rems(cx))
            .justify_between()
            .flex_none()
            .h_full()
            .py(DynamicSpacing::Base12.px(cx))
            .child(
                v_flex()
                    .gap(DynamicSpacing::Base08.rems(cx))
                    .child(
                        Self::button_group(window, cx)
                            .child(
                                Self::render_notebook_control(
                                    "run-all-cells",
                                    IconName::PlayFilled,
                                    window,
                                    cx,
                                )
                                .tooltip(move |_window, cx| {
                                    Tooltip::for_action("Execute all cells", &RunAll, cx)
                                })
                                .on_click(cx.listener(
                                    |this, _, window, cx| {
                                        this.focus_notebook(window, cx);
                                        this.run_cells(window, cx);
                                    },
                                )),
                            )
                            .child(
                                Self::render_notebook_control(
                                    "run-cells-above",
                                    IconName::ArrowUpRight,
                                    window,
                                    cx,
                                )
                                .tooltip(move |_window, cx| {
                                    Tooltip::for_action("Run cells above", &RunCellsAbove, cx)
                                })
                                .on_click(cx.listener(
                                    |this, _, window, cx| {
                                        this.focus_notebook(window, cx);
                                        this.run_cells_above(&RunCellsAbove, window, cx);
                                    },
                                )),
                            )
                            .child(
                                Self::render_notebook_control(
                                    "run-cell-and-below",
                                    IconName::ArrowDownRight,
                                    window,
                                    cx,
                                )
                                .tooltip(move |_window, cx| {
                                    Tooltip::for_action("Run cell and below", &RunCellAndBelow, cx)
                                })
                                .on_click(cx.listener(
                                    |this, _, window, cx| {
                                        this.focus_notebook(window, cx);
                                        this.run_cell_and_below(&RunCellAndBelow, window, cx);
                                    },
                                )),
                            )
                            .child(
                                Self::render_notebook_control(
                                    "clear-all-outputs",
                                    IconName::ListX,
                                    window,
                                    cx,
                                )
                                .disabled(!has_execution_record)
                                .tooltip(move |_window, cx| {
                                    Tooltip::for_action("Clear all outputs", &ClearOutputs, cx)
                                })
                                .on_click(cx.listener(
                                    |this, _, window, cx| {
                                        this.focus_notebook(window, cx);
                                        this.clear_outputs(window, cx);
                                    },
                                )),
                            ),
                    )
                    .child(
                        Self::button_group(window, cx)
                            .child(
                                Self::render_notebook_control(
                                    "go-to-running-cell",
                                    IconName::Crosshair,
                                    window,
                                    cx,
                                )
                                .disabled(!has_running_cell)
                                .tooltip(move |_window, cx| {
                                    Tooltip::for_action("Go to running cell", &GoToRunningCell, cx)
                                })
                                .on_click(cx.listener(
                                    |this, _, window, cx| {
                                        this.focus_notebook(window, cx);
                                        this.go_to_running_cell(&GoToRunningCell, window, cx);
                                    },
                                )),
                            )
                            .child(
                                Self::render_notebook_control(
                                    "follow-running-cell",
                                    IconName::Eye,
                                    window,
                                    cx,
                                )
                                .toggle_state(following)
                                .tooltip(move |_window, cx| {
                                    Tooltip::for_action(
                                        "Follow running cell",
                                        &ToggleFollowRunningCell,
                                        cx,
                                    )
                                })
                                .on_click(cx.listener(
                                    |this, _, window, cx| {
                                        this.focus_notebook(window, cx);
                                        this.toggle_follow_running_cell(
                                            &ToggleFollowRunningCell,
                                            window,
                                            cx,
                                        );
                                    },
                                )),
                            ),
                    )
                    .child(
                        Self::button_group(window, cx)
                            .child(
                                Self::render_notebook_control(
                                    "move-cell-up",
                                    IconName::ArrowUp,
                                    window,
                                    cx,
                                )
                                .tooltip(move |_window, cx| {
                                    Tooltip::for_action("Move cell up", &MoveCellUp, cx)
                                })
                                .on_click(cx.listener(
                                    |this, _, window, cx| {
                                        this.focus_notebook(window, cx);
                                        this.move_cell_up(window, cx);
                                    },
                                )),
                            )
                            .child(
                                Self::render_notebook_control(
                                    "move-cell-down",
                                    IconName::ArrowDown,
                                    window,
                                    cx,
                                )
                                .tooltip(move |_window, cx| {
                                    Tooltip::for_action("Move cell down", &MoveCellDown, cx)
                                })
                                .on_click(cx.listener(
                                    |this, _, window, cx| {
                                        this.focus_notebook(window, cx);
                                        this.move_cell_down(window, cx);
                                    },
                                )),
                            ),
                    )
                    .child(
                        Self::button_group(window, cx)
                            .child(
                                Self::render_notebook_control(
                                    "new-markdown-cell",
                                    IconName::Plus,
                                    window,
                                    cx,
                                )
                                .tooltip(move |_window, cx| {
                                    Tooltip::for_action("Add markdown block", &AddMarkdownBlock, cx)
                                })
                                .on_click(cx.listener(
                                    |this, _, window, cx| {
                                        this.focus_notebook(window, cx);
                                        this.add_markdown_block(window, cx);
                                    },
                                )),
                            )
                            .child(
                                Self::render_notebook_control(
                                    "new-code-cell",
                                    IconName::Code,
                                    window,
                                    cx,
                                )
                                .tooltip(move |_window, cx| {
                                    Tooltip::for_action("Add code block", &AddCodeBlock, cx)
                                })
                                .on_click(cx.listener(
                                    |this, _, window, cx| {
                                        this.focus_notebook(window, cx);
                                        this.add_code_block(window, cx);
                                    },
                                )),
                            ),
                    ),
            )
            .child(
                v_flex()
                    .gap(DynamicSpacing::Base08.rems(cx))
                    .items_center()
                    .child(
                        PopoverMenu::new("notebook-more-menu")
                            .trigger_with_tooltip(
                                Self::render_notebook_control(
                                    "more-menu",
                                    IconName::Ellipsis,
                                    window,
                                    cx,
                                ),
                                Tooltip::text("More options"),
                            )
                            .menu(move |window, cx| {
                                Some(ContextMenu::build(window, cx, |menu, _, _| {
                                    menu.action("Run Cells Above", Box::new(RunCellsAbove))
                                        .action("Run Cell and Below", Box::new(RunCellAndBelow))
                                        .separator()
                                        .action("Add Cell Above", Box::new(AddCellAbove))
                                        .action("Add Cell Below", Box::new(AddCellBelow))
                                        .action("Move Cell Up", Box::new(MoveCellUp))
                                        .action("Move Cell Down", Box::new(MoveCellDown))
                                        .separator()
                                        .action("Convert to Code", Box::new(ConvertToCode))
                                        .action("Convert to Markdown", Box::new(ConvertToMarkdown))
                                        .action("Split Cell", Box::new(SplitCell))
                                        .action("Join Cells", Box::new(JoinCells))
                                        .separator()
                                        .action("Copy Cell", Box::new(CopyCell))
                                        .action("Cut Cell", Box::new(CutCell))
                                        .action("Paste Cell", Box::new(PasteCell))
                                        .action("Paste Cell Above", Box::new(PasteCellAbove))
                                        .action("Duplicate Cell", Box::new(DuplicateCell))
                                        .separator()
                                        .action("Undo Cell Change", Box::new(UndoCellOp))
                                        .action("Redo Cell Change", Box::new(RedoCellOp))
                                        .separator()
                                        .action("Clear Cell Outputs", Box::new(ClearCellOutputs))
                                        .action("Clear All Outputs", Box::new(ClearOutputs))
                                        .action("Delete Cell", Box::new(DeleteCell))
                                        .separator()
                                        .action("Reload Notebook", Box::new(ReloadNotebook))
                                }))
                            }),
                    )
                    // Kernel lifecycle controls, moved here from the removed
                    // bottom bar (phase 30).
                    .child(
                        Self::button_group(window, cx)
                            .child(
                                Self::render_notebook_control(
                                    "restart-kernel",
                                    IconName::RotateCw,
                                    window,
                                    cx,
                                )
                                .tooltip(|_window, cx| {
                                    Tooltip::for_action("Restart Kernel", &RestartKernel, cx)
                                })
                                .on_click(cx.listener(
                                    |this, _, window, cx| {
                                        this.focus_notebook(window, cx);
                                        this.restart_kernel(&RestartKernel, window, cx);
                                    },
                                )),
                            )
                            .child(
                                Self::render_notebook_control(
                                    "interrupt-kernel",
                                    IconName::Stop,
                                    window,
                                    cx,
                                )
                                .disabled(!self.kernel.status().is_connected())
                                .tooltip(|_window, cx| {
                                    Tooltip::for_action("Interrupt Kernel", &InterruptKernel, cx)
                                })
                                .on_click(cx.listener(
                                    |this, _, window, cx| {
                                        this.focus_notebook(window, cx);
                                        this.interrupt_kernel(&InterruptKernel, window, cx);
                                    },
                                )),
                            ),
                    ), // The sidebar's own kernel-picker trigger used to sit
                       // here. It shared a single `PopoverMenuHandle` with the
                       // top-right strip, so clicking it opened the popover
                       // anchored at the OTHER trigger; the strip already shows
                       // kernel name + status, so one entry point is enough.
            )
    }

    /// Sidebar controls already act on THIS notebook whatever has focus (bug
    /// #51), but they must also MOVE focus here — otherwise the keyboard keeps
    /// talking to whatever was focused before (typically the project panel the
    /// notebook was opened from), so shortcuts silently do nothing right after
    /// you clicked a notebook button. A click from INSIDE the notebook (a cell
    /// editor has focus) is left alone, so pressing a control mid-edit doesn't
    /// throw you out of the cell.
    fn focus_notebook(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if !self.focus_handle.contains_focused(window, cx) {
            self.focus_handle.focus(window, cx);
        }
    }

    /// Slim strip ABOVE the cells (phase 30): the kernel cluster — status
    /// icon + name, acting as the kernel-picker trigger — right-aligned, a
    /// light take on VS Code's notebook top bar. Replaces the old bottom
    /// status bar; Restart/Interrupt moved into the right sidebar.
    fn render_kernel_strip(
        &self,
        _window: &mut Window,
        cx: &mut Context<Self>,
    ) -> impl IntoElement {
        // While an env is being created (phase 48) it is the pending selection:
        // show its name with a Starting status, ahead of the old kernel's.
        let creating = self.creating_kernel_name.is_some();
        let kernel_status = if creating {
            KernelStatus::Starting
        } else {
            self.kernel.status()
        };
        let kernel_name = self
            .creating_kernel_name
            .clone()
            .or_else(|| {
                self.kernel_specification
                    .as_ref()
                    .map(|spec| spec.name().to_string())
            })
            .unwrap_or_else(|| "Select Kernel".to_string());

        let (status_icon, status_color) = match &kernel_status {
            KernelStatus::Idle => (IconName::Circle, Color::Success),
            KernelStatus::Busy => (IconName::ArrowCircle, Color::Info),
            KernelStatus::Starting => (IconName::ArrowCircle, Color::Muted),
            KernelStatus::Error => (IconName::XCircle, Color::Error),
            KernelStatus::ShuttingDown => (IconName::ArrowCircle, Color::Muted),
            KernelStatus::Shutdown => (IconName::Circle, Color::Muted),
            KernelStatus::Restarting => (IconName::ArrowCircle, Color::Warning),
        };
        // "Something is happening" states spin; every settled state (including
        // every terminal one) is static, so the animation can never outlive the
        // work it stands for. Derived from the CURRENT status each render — no
        // latched flag to get stuck on.
        let kernel_working = matches!(
            kernel_status,
            KernelStatus::Busy
                | KernelStatus::Starting
                | KernelStatus::Restarting
                | KernelStatus::ShuttingDown
        );
        let status_icon = Icon::new(status_icon)
            .size(IconSize::Small)
            .color(status_color);
        // The strip is pinned above the cells, so this doubles as the notebook's
        // global busy/idle indicator: a static icon alone was too easy to miss
        // from across the screen, hence the motion and the spelled-out state.
        let status_indicator = if kernel_working {
            status_icon.with_rotate_animation(2).into_any_element()
        } else {
            status_icon.into_any_element()
        };
        // A grey dot already reads as "not running", so spelling out "Shutdown"
        // next to it is noise (user 2026-08-06). Every other state earns its
        // label.
        let status_label = match &kernel_status {
            KernelStatus::Shutdown => None,
            status => Some(status.to_string()),
        };

        let worktree_id = self.worktree_id;
        let kernel_picker_handle = self.kernel_picker_handle.clone();
        let view = cx.entity().downgrade();
        let view_for_dismiss = view.clone();
        let view_for_create = view.clone();
        let view_for_open = view.clone();

        // The strip is pinned above the cells, so a failure shown here is
        // readable at any scroll position — which is the point: after a Run All
        // the cell that stopped it is usually off-screen.
        let failed_count = self.failed_cell_indices(cx).len();

        let settings = ReplSettings::get_global(cx);
        // Both timers are opt-in and share the strip with the failure count and
        // the kernel selector, so they render as short muted labels with the
        // full name in a tooltip rather than spelling it out inline.
        let execution_time = settings
            .notebook_show_execution_time
            .then(|| format_duration(self.execution_time_total()));
        let kernel_uptime = settings
            .notebook_show_kernel_uptime
            .then(|| self.kernel_uptime().map(format_duration))
            .flatten();

        // No background band: the strip reads as dead space at the top of the
        // notebook with just the kernel cluster in the corner (user 2026-07-14).
        h_flex()
            .w_full()
            .flex_none()
            .px_2()
            .py_0p5()
            .gap_2()
            .items_center()
            .justify_end()
            .when_some(execution_time, |el, elapsed| {
                el.child(
                    div()
                        .id("notebook-execution-time")
                        .child(
                            Label::new(format!("Exec {elapsed}"))
                                .size(LabelSize::Small)
                                .color(Color::Muted),
                        )
                        .tooltip(Tooltip::text(
                            "Time this kernel session has spent executing cells",
                        )),
                )
            })
            .when_some(kernel_uptime, |el, uptime| {
                el.child(
                    div()
                        .id("notebook-kernel-uptime")
                        .child(
                            Label::new(format!("Up {uptime}"))
                                .size(LabelSize::Small)
                                .color(Color::Muted),
                        )
                        .tooltip(Tooltip::text("How long the kernel has been running")),
                )
            })
            .when(failed_count > 0, |el| {
                el.child(
                    Button::new(
                        "go-to-error",
                        if failed_count == 1 {
                            "1 cell failed".to_string()
                        } else {
                            format!("{failed_count} cells failed")
                        },
                    )
                    .start_icon(
                        Icon::new(IconName::XCircle)
                            .size(IconSize::Small)
                            .color(Color::Error),
                    )
                    .label_size(LabelSize::Small)
                    .style(ButtonStyle::Transparent)
                    .tooltip(Tooltip::for_action_title("Go to Error", &GoToError))
                    .on_click(cx.listener(|this, _, window, cx| {
                        this.go_to_error(&GoToError, window, cx);
                    })),
                )
            })
            .child(
                h_flex()
                    .id("notebook-kernel-status")
                    .gap_1()
                    .items_center()
                    .child(status_indicator)
                    .when_some(status_label, |el, label| {
                        el.child(Label::new(label).size(LabelSize::Small).color(status_color))
                    })
                    .tooltip(Tooltip::text(format!(
                        "Kernel {kernel_name} is {}",
                        kernel_status.to_string()
                    ))),
            )
            .child(
                KernelSelector::new(
                    Box::new(move |spec: KernelSpecification, window, cx| {
                        if let Some(view) = view.upgrade() {
                            view.update(cx, |this, cx| {
                                this.change_kernel(spec, window, cx);
                            });
                        }
                    }),
                    worktree_id,
                    // No status icon on the button: the animated indicator to
                    // its left carries the status now, and two icons for one
                    // state read as two different things.
                    Button::new("kernel-selector", kernel_name.clone())
                        .label_size(LabelSize::Small),
                    Tooltip::text(format!(
                        "Kernel: {} ({}). Click to change.",
                        kernel_name,
                        kernel_status.to_string()
                    )),
                )
                // The picker reflects THIS notebook's kernel, not the
                // worktree-level selection (bug #30).
                .with_selected(self.kernel_specification.clone())
                // While an env is building, show it as a greyed selected entry
                // (phase 49); suppresses the old kernel's checkmark.
                .with_creating(self.creating_kernel_name.clone())
                .with_dismiss(Box::new(move |_window, cx| {
                    if let Some(view) = view_for_dismiss.upgrade() {
                        view.update(cx, |this, cx| {
                            this.clear_awaiting_cells(cx);
                        });
                    }
                }))
                .with_create_env(std::rc::Rc::new(move |window, cx| {
                    if let Some(view) = view_for_create.upgrade() {
                        view.update(cx, |this, cx| {
                            this.create_python_environment(window, cx);
                        });
                    }
                }))
                // Opening the picker re-validates the environments (phase
                // 42): python envs are re-discovered, and a selection whose
                // env vanished is dropped so the checkmark/indicator don't
                // point at a ghost.
                .with_on_open(std::rc::Rc::new(move |_window, cx| {
                    if let Some(view) = view_for_open.upgrade() {
                        view.update(cx, |this, cx| {
                            if this
                                .kernel_specification
                                .as_ref()
                                .is_some_and(|spec| Self::spec_interpreter_missing(spec))
                            {
                                this.discard_stale_kernel_selection(cx);
                            } else {
                                this.refresh_kernelspecs(cx);
                            }
                        });
                    }
                }))
                .with_handle(kernel_picker_handle),
            )
    }

    fn cell_list(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let view = cx.entity();
        // gpui can't read editor settings, so the notebook opts its list in.
        // Re-applied each render so toggling the setting takes effect live.
        self.cell_list
            .set_smooth_scroll(editor::EditorSettings::get_global(cx).smooth_scrolling);
        // Re-check the page against the heights the last layout measured, so
        // output growing in the cells above can't quietly carry the running cell
        // off the bottom of the screen.
        self.maintain_page_follow(cx);
        list(self.cell_list.clone(), move |index, window, cx| {
            view.update(cx, |this, cx| {
                let cell_id = &this.cell_order[index];
                let cell = this.cell_map.get(cell_id).unwrap();
                this.render_cell(index, cell, window, cx).into_any_element()
            })
        })
        .size_full()
    }

    fn cell_position(&self, index: usize) -> CellPosition {
        match index {
            0 => CellPosition::First,
            index if index == self.cell_count() - 1 => CellPosition::Last,
            _ => CellPosition::Middle,
        }
    }

    fn render_cell(
        &self,
        index: usize,
        cell: &Cell,
        _window: &mut Window,
        cx: &mut Context<Self>,
    ) -> impl IntoElement {
        let cell_position = self.cell_position(index);

        let is_selected = self.is_index_selected(index);

        match cell {
            Cell::Code(cell) => {
                cell.update(cx, |cell, _cx| {
                    cell.set_selected(is_selected)
                        .set_cell_position(cell_position);
                });
                cell.clone().into_any_element()
            }
            Cell::Markdown(cell) => {
                cell.update(cx, |cell, _cx| {
                    cell.set_selected(is_selected)
                        .set_cell_position(cell_position);
                });
                cell.clone().into_any_element()
            }
            Cell::Raw(cell) => {
                cell.update(cx, |cell, _cx| {
                    cell.set_selected(is_selected)
                        .set_cell_position(cell_position);
                });
                cell.clone().into_any_element()
            }
        }
    }
}

impl Render for NotebookEditor {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        self.ensure_runtime_timer(cx);

        let mut key_context = KeyContext::new_with_defaults();
        key_context.add("NotebookEditor");
        key_context.set(
            "notebook_mode",
            match self.notebook_mode {
                NotebookMode::Command => "command",
                NotebookMode::Edit => "edit",
            },
        );

        v_flex()
            .size_full()
            .key_context(key_context)
            .track_focus(&self.focus_handle)
            .on_action(cx.listener(|this, _: &OpenNotebook, window, cx| {
                this.open_notebook(&OpenNotebook, window, cx)
            }))
            .on_action(
                cx.listener(|this, _: &ClearOutputs, window, cx| this.clear_outputs(window, cx)),
            )
            .on_action(
                cx.listener(|this, _: &Run, window, cx| this.run_current_cell(&Run, window, cx)),
            )
            .on_action(
                cx.listener(|this, action, window, cx| this.run_and_advance(action, window, cx)),
            )
            .on_action(cx.listener(|this, _: &RunAll, window, cx| this.run_cells(window, cx)))
            .on_action(cx.listener(Self::go_to_running_cell))
            .on_action(cx.listener(Self::toggle_follow_running_cell))
            .on_action(cx.listener(Self::go_to_error))
            .on_action(cx.listener(Self::next_error))
            .on_action(cx.listener(Self::previous_error))
            .on_action(
                cx.listener(|this, _: &MoveCellUp, window, cx| this.move_cell_up(window, cx)),
            )
            .on_action(
                cx.listener(|this, _: &MoveCellDown, window, cx| this.move_cell_down(window, cx)),
            )
            .on_action(cx.listener(|this, _: &AddMarkdownBlock, window, cx| {
                this.add_markdown_block(window, cx)
            }))
            .on_action(
                cx.listener(|this, _: &AddCodeBlock, window, cx| this.add_code_block(window, cx)),
            )
            .on_action(
                cx.listener(|this, action, window, cx| this.add_cell_above(action, window, cx)),
            )
            .on_action(
                cx.listener(|this, action, window, cx| this.add_cell_below(action, window, cx)),
            )
            .on_action(cx.listener(|this, action, window, cx| this.delete_cell(action, window, cx)))
            .on_action(cx.listener(|this, action, window, cx| this.split_cell(action, window, cx)))
            .on_action(cx.listener(|this, action, window, cx| this.join_cells(action, window, cx)))
            .on_action(cx.listener(|this, action, window, cx| this.copy_cell(action, window, cx)))
            .on_action(cx.listener(|this, action, window, cx| this.cut_cell(action, window, cx)))
            .on_action(cx.listener(|this, action, window, cx| this.paste_cell(action, window, cx)))
            .on_action(
                cx.listener(|this, action, window, cx| this.paste_cell_above(action, window, cx)),
            )
            .on_action(
                cx.listener(|this, action, window, cx| this.duplicate_cell(action, window, cx)),
            )
            .on_action(
                cx.listener(|this, action, window, cx| this.undo_cell_op(action, window, cx)),
            )
            .on_action(
                cx.listener(|this, action, window, cx| this.redo_cell_op(action, window, cx)),
            )
            .on_action(
                cx.listener(|this, action, window, cx| this.convert_to_code(action, window, cx)),
            )
            .on_action(
                cx.listener(|this, action, window, cx| {
                    this.convert_to_markdown(action, window, cx)
                }),
            )
            .on_action(
                cx.listener(|this, action, window, cx| this.run_cells_above(action, window, cx)),
            )
            .on_action(
                cx.listener(|this, action, window, cx| this.run_cell_and_below(action, window, cx)),
            )
            .on_action(
                cx.listener(|this, action, window, cx| this.enter_edit_mode(action, window, cx)),
            )
            .on_action(cx.listener(|this, action, window, cx| {
                this.handle_enter_command_mode(action, window, cx)
            }))
            .on_action(cx.listener(|this, action, window, cx| {
                this.select_next(action, SelectionMode::SelectOnly, window, cx)
            }))
            .on_action(cx.listener(|this, action, window, cx| {
                this.select_previous(action, SelectionMode::SelectOnly, window, cx)
            }))
            .on_action(cx.listener(Self::select_first))
            .on_action(cx.listener(Self::select_last))
            .on_action(cx.listener(|this, _: &ExtendSelectionDown, window, cx| {
                this.extend_selection(1, window, cx)
            }))
            .on_action(cx.listener(|this, _: &ExtendSelectionUp, window, cx| {
                this.extend_selection(-1, window, cx)
            }))
            .on_action(cx.listener(|this, _: &ExtendSelectionToStart, window, cx| {
                this.extend_selection_to_boundary(false, window, cx)
            }))
            .on_action(cx.listener(|this, _: &ExtendSelectionToEnd, window, cx| {
                this.extend_selection_to_boundary(true, window, cx)
            }))
            .on_action(
                cx.listener(|this, _: &SelectAllCells, window, cx| {
                    this.select_all_cells(window, cx)
                }),
            )
            .on_action(cx.listener(|this, _: &MoveDown, window, cx| {
                this.select_next(
                    &Default::default(),
                    SelectionMode::SelectAndMove,
                    window,
                    cx,
                );
            }))
            .on_action(cx.listener(|this, _: &MoveUp, window, cx| {
                this.select_previous(
                    &Default::default(),
                    SelectionMode::SelectAndMove,
                    window,
                    cx,
                );
            }))
            .on_action(cx.listener(|this, _: &NotebookMoveDown, window, cx| {
                let Some(cell) = this.get_selected_cell() else {
                    return;
                };

                let Some(editor) = cell.editor(cx).cloned() else {
                    return;
                };

                let is_at_last_line = editor.update(cx, |editor, cx| {
                    let display_snapshot = editor.display_snapshot(cx);
                    let selections = editor.selections.all_display(&display_snapshot);
                    if let Some(selection) = selections.last() {
                        let head = selection.head();
                        let cursor_row = head.row();
                        let max_row = display_snapshot.max_point().row();

                        cursor_row >= max_row
                    } else {
                        false
                    }
                });

                if is_at_last_line {
                    this.select_next(
                        &Default::default(),
                        SelectionMode::SelectAndMove,
                        window,
                        cx,
                    );
                } else {
                    editor.update(cx, |editor, cx| {
                        editor.move_down(&Default::default(), window, cx);
                    });
                }
            }))
            .on_action(cx.listener(|this, _: &NotebookMoveUp, window, cx| {
                let Some(cell) = this.get_selected_cell() else {
                    return;
                };

                let Some(editor) = cell.editor(cx).cloned() else {
                    return;
                };

                let is_at_first_line = editor.update(cx, |editor, cx| {
                    let display_snapshot = editor.display_snapshot(cx);
                    let selections = editor.selections.all_display(&display_snapshot);
                    if let Some(selection) = selections.first() {
                        let head = selection.head();
                        let cursor_row = head.row();

                        cursor_row.0 == 0
                    } else {
                        false
                    }
                });

                if is_at_first_line {
                    this.select_previous(
                        &Default::default(),
                        SelectionMode::SelectAndMove,
                        window,
                        cx,
                    );
                } else {
                    editor.update(cx, |editor, cx| {
                        editor.move_up(&Default::default(), window, cx);
                    });
                }
            }))
            .on_action(
                cx.listener(|this, action, window, cx| this.restart_kernel(action, window, cx)),
            )
            .on_action(
                cx.listener(|this, action, window, cx| this.interrupt_kernel(action, window, cx)),
            )
            .on_action(cx.listener(|this, action, window, cx| {
                this.handle_reload_notebook(action, window, cx)
            }))
            .on_action(cx.listener(|this, action, window, cx| {
                this.clear_selected_cell_outputs(action, window, cx)
            }))
            // Kernel strip on top, cells below it (phase 30) — the old bottom
            // status bar is gone; restart/interrupt live in the sidebar.
            .child(self.render_kernel_strip(window, cx))
            .child(
                // `.flex_1()` (not `.h_full()`) sizes this row to the height
                // left after the kernel strip; `.h_full()` here would take the
                // whole notebook height and let siblings overlap the row's
                // bottom, hiding the last cell (see bug #25).
                h_flex()
                    .flex_1()
                    .w_full()
                    .min_h_0()
                    .gap_2()
                    .child({
                        // Vertical scrollbar tracking the cell-list `ListState`.
                        // Attached to the list column (a flex sibling left of the
                        // `gap_2` and the control bar), so it sits at the list's
                        // right edge without being occluded by the sidebar.
                        // Visibility follows the user's editor scrollbar setting.
                        let scrollbars =
                            Scrollbars::for_settings::<editor::EditorSettingsScrollbarProxy>()
                                .show_along(ScrollAxes::Vertical)
                                // Match the editor's scrollbar (wider, full-width
                                // thumb, no side gaps) rather than the default
                                // Regular style, which read as too thin/floaty in
                                // the notebook. See phase 56.
                                .style(ScrollbarStyle::Editor)
                                .tracked_scroll_handle(&self.cell_list);
                        div()
                            .flex_1()
                            .h_full()
                            .child(self.cell_list(window, cx))
                            .custom_scrollbars(scrollbars, window, cx)
                    })
                    .child(self.render_notebook_controls(window, cx)),
            )
    }
}

impl Focusable for NotebookEditor {
    fn focus_handle(&self, _: &App) -> FocusHandle {
        self.focus_handle.clone()
    }
}

// Intended to be a NotebookBuffer
pub struct NotebookItem {
    // The file-backing fields are all `Some` together for a notebook opened
    // from (or saved to) disk, and all `None` for an untitled notebook
    // created by "New Jupyter Notebook" (phase 33), which only gains them on
    // its first save-as.
    path: Option<PathBuf>,
    project_path: Option<ProjectPath>,
    languages: Arc<LanguageRegistry>,
    // Raw notebook data
    notebook: nbformat::v4::Notebook,
    // The entry id observed at open time. Only a FALLBACK for `entry_id` —
    // saving rewrites the file via a temp-file rename, which can replace the
    // worktree entry under a new id; advertising the stale id would defeat
    // the pane's already-open dedup and let the same notebook open in
    // multiple tabs (bug #34).
    id: Option<ProjectEntryId>,
    project: WeakEntity<Project>,
    // The underlying project buffer for the .ipynb file. Retained so the
    // project keeps watching the file and emits `Reloaded` on external change.
    buffer: Option<Entity<Buffer>>,
}

impl project::ProjectItem for NotebookItem {
    /// Match `.ipynb` by the worktree-relative path (the fast common case),
    /// falling back to the ABSOLUTE path when the relative one carries no
    /// extension. A file saved/opened OUTSIDE any worktree gets a single-file
    /// worktree rooted at the file, whose relative path is empty — without this
    /// fallback the extension gate failed and the notebook opened as raw JSON in
    /// a plain text editor instead of the notebook UI.
    fn claims_path(project: &Entity<Project>, path: &ProjectPath, cx: &App) -> bool {
        path.path.extension() == Some("ipynb")
            || project
                .read(cx)
                .absolute_path(path, cx)
                .as_deref()
                .and_then(|abs_path| abs_path.extension())
                .and_then(|extension| extension.to_str())
                == Some("ipynb")
    }

    fn try_open(
        project: &Entity<Project>,
        path: &ProjectPath,
        cx: &mut App,
    ) -> Option<Task<anyhow::Result<Entity<Self>>>> {
        let path = path.clone();
        let project = project.clone();
        let languages = project.read(cx).languages().clone();

        if Self::claims_path(&project, &path, cx) {
            Some(cx.spawn(async move |cx| {
                let abs_path = project
                    .read_with(cx, |project, cx| project.absolute_path(&path, cx))
                    .with_context(|| format!("finding the absolute path of {path:?}"))?;

                let buffer = project
                    .update(cx, |project, cx| project.open_buffer(path.clone(), cx))
                    .await?;
                let file_content = buffer.read_with(cx, |buffer, _| buffer.text());

                let notebook = NotebookEditor::parse_notebook_text(&file_content)?;

                // A MISSING entry is not a failure. The notebook opens fine
                // without one — `id` is only a fallback for `entry_id`, which
                // re-resolves from the path anyway — and insisting on it meant a
                // notebook whose file had been deleted could not be opened at
                // all. Session restore then dropped the tab silently, taking any
                // unsaved changes it was holding with it (bug #74).
                let id = project.update(cx, |project, cx| {
                    project.entry_for_path(&path, cx).map(|entry| entry.id)
                });

                Ok(cx.new(|_| NotebookItem {
                    path: Some(abs_path),
                    project_path: Some(path),
                    languages,
                    notebook,
                    id,
                    project: project.downgrade(),
                    buffer: Some(buffer),
                }))
            }))
        } else {
            None
        }
    }

    fn entry_id(&self, cx: &App) -> Option<ProjectEntryId> {
        let project_path = self.project_path.as_ref()?;
        self.project
            .upgrade()
            .and_then(|project| project.read(cx).entry_for_path(project_path, cx))
            .map(|entry| entry.id)
            .or(self.id)
    }

    fn project_path(&self, _: &App) -> Option<ProjectPath> {
        self.project_path.clone()
    }

    fn is_dirty(&self) -> bool {
        // TODO: Track if notebook metadata or structure has changed
        false
    }
}

impl NotebookItem {
    /// An untitled, session-only notebook (phase 33): lives purely in memory
    /// until the first save-as attaches it to a file.
    pub fn untitled(
        project: WeakEntity<Project>,
        languages: Arc<LanguageRegistry>,
        notebook: nbformat::v4::Notebook,
    ) -> Self {
        NotebookItem {
            path: None,
            project_path: None,
            languages,
            notebook,
            id: None,
            project,
            buffer: None,
        }
    }

    fn is_untitled(&self) -> bool {
        self.path.is_none()
    }

    pub fn language_name(&self) -> Option<String> {
        self.notebook
            .metadata
            .language_info
            .as_ref()
            .map(|l| l.name.clone())
            .or(self
                .notebook
                .metadata
                .kernelspec
                .as_ref()
                .and_then(|spec| spec.language.clone()))
    }

    pub fn notebook_language(&self) -> impl Future<Output = Option<Arc<Language>>> + use<> {
        let language_name = self.language_name();
        let languages = self.languages.clone();

        async move {
            if let Some(language_name) = language_name {
                languages.language_for_name(&language_name).await.ok()
            } else {
                None
            }
        }
    }
}

impl EventEmitter<()> for NotebookItem {}

impl EventEmitter<()> for NotebookEditor {}

impl EventEmitter<SearchEvent> for NotebookEditor {}

/// A single Ctrl-F match inside a notebook. A notebook is N independent cell
/// editors (not a multibuffer), so a match is a cell plus a range within THAT
/// cell's editor buffer — the range is only valid for `cell_id`'s editor and
/// must never be handed to another cell.
#[derive(Clone)]
pub struct NotebookSearchMatch {
    cell_id: CellId,
    range: Range<editor::Anchor>,
}

impl NotebookEditor {
    /// Ordered (cell, editor) pairs for every cell that has a source editor
    /// (code + markdown; raw cells have none), in document order.
    fn ordered_cell_editors(&self, cx: &App) -> Vec<(CellId, Entity<editor::Editor>)> {
        self.cell_order
            .iter()
            .filter_map(|id| {
                self.cell_map
                    .get(id)
                    .and_then(|cell| cell.editor(cx).cloned())
                    .map(|editor| (id.clone(), editor))
            })
            .collect()
    }

    fn selected_cell_editor(&self, cx: &App) -> Option<Entity<editor::Editor>> {
        self.cell_order
            .get(self.selected_cell_index)
            .and_then(|id| self.cell_map.get(id))
            .and_then(|cell| cell.editor(cx).cloned())
    }

    fn cell_index_of(&self, cell_id: &CellId) -> Option<usize> {
        self.cell_order.iter().position(|id| id == cell_id)
    }
}

impl SearchableItem for NotebookEditor {
    type Match = NotebookSearchMatch;

    fn supported_options(&self) -> SearchOptions {
        // Part 1: find / highlight / navigate only. Replace and selection-scoped
        // search are deferred to a follow-up phase.
        SearchOptions {
            case: true,
            word: true,
            regex: true,
            replacement: false,
            selection: false,
            select_all: true,
            find_in_results: false,
        }
    }

    fn clear_matches(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        // Only signal invalidation when a cell actually had matches to clear.
        // Emitting unconditionally hangs the app: the buffer search bar reacts
        // to MatchesInvalidated by re-running its update, which calls back into
        // clear_matches — so an unconditional emit is an infinite clear/emit
        // loop the moment the notebook becomes the active searchable item.
        // (Editor guards its emit the same way — only when a highlight was
        // actually removed.)
        let mut had_matches = false;
        for (_, editor) in self.ordered_cell_editors(cx) {
            editor.update(cx, |editor, cx| {
                if !SearchableItem::get_matches(editor, window, cx).0.is_empty() {
                    had_matches = true;
                }
                SearchableItem::clear_matches(editor, window, cx);
            });
        }
        if had_matches {
            cx.emit(SearchEvent::MatchesInvalidated);
        }
    }

    fn update_matches(
        &mut self,
        matches: &[Self::Match],
        active_match_index: Option<usize>,
        token: SearchToken,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        for (cell_id, editor) in self.ordered_cell_editors(cx) {
            let cell_ranges: Vec<Range<editor::Anchor>> = matches
                .iter()
                .filter(|m| m.cell_id == cell_id)
                .map(|m| m.range.clone())
                .collect();
            // Translate the global active index to this cell's local index (its
            // position among this cell's matches), when the active match is here.
            let local_active = active_match_index.and_then(|global| {
                matches
                    .get(global)
                    .filter(|m| m.cell_id == cell_id)
                    .map(|_| {
                        matches[..global]
                            .iter()
                            .filter(|m| m.cell_id == cell_id)
                            .count()
                    })
            });
            editor.update(cx, |editor, cx| {
                SearchableItem::update_matches(
                    editor,
                    &cell_ranges,
                    local_active,
                    token,
                    window,
                    cx,
                );
            });
        }
    }

    fn query_suggestion(
        &mut self,
        seed_query_override: Option<SeedQuerySetting>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> String {
        self.selected_cell_editor(cx)
            .map(|editor| {
                editor.update(cx, |editor, cx| {
                    SearchableItem::query_suggestion(editor, seed_query_override, window, cx)
                })
            })
            .unwrap_or_default()
    }

    fn activate_match(
        &mut self,
        index: usize,
        matches: &[Self::Match],
        token: SearchToken,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let Some(search_match) = matches.get(index) else {
            return;
        };
        let cell_id = search_match.cell_id.clone();
        let range = search_match.range.clone();
        let Some(cell_index) = self.cell_index_of(&cell_id) else {
            return;
        };
        // Select the owning cell, select the range inside its editor, then reveal
        // the cell via the index-anchored scroll (immune to unmeasured/stale cell
        // heights — see bug #45), so a match in a far cell actually lands on it.
        self.set_selected_index(cell_index, false, window, cx);
        if let Some(editor) = self
            .cell_map
            .get(&cell_id)
            .and_then(|cell| cell.editor(cx).cloned())
        {
            editor.update(cx, |editor, cx| {
                SearchableItem::activate_match(
                    editor,
                    0,
                    std::slice::from_ref(&range),
                    token,
                    window,
                    cx,
                );
            });
        }
        self.follow_scroll_to(cell_index);
        cx.notify();
    }

    fn select_matches(
        &mut self,
        matches: &[Self::Match],
        token: SearchToken,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        for (cell_id, editor) in self.ordered_cell_editors(cx) {
            let cell_ranges: Vec<Range<editor::Anchor>> = matches
                .iter()
                .filter(|m| m.cell_id == cell_id)
                .map(|m| m.range.clone())
                .collect();
            if cell_ranges.is_empty() {
                continue;
            }
            editor.update(cx, |editor, cx| {
                SearchableItem::select_matches(editor, &cell_ranges, token, window, cx);
            });
        }
    }

    fn replace(
        &mut self,
        _: &Self::Match,
        _: &project::search::SearchQuery,
        _token: SearchToken,
        _window: &mut Window,
        _: &mut Context<Self>,
    ) {
        // Replace is not offered in part 1 (`supported_options().replacement`
        // is false), so this is never called.
    }

    fn find_matches(
        &mut self,
        query: Arc<project::search::SearchQuery>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Task<Vec<Self::Match>> {
        let mut cell_tasks: Vec<(CellId, Task<Vec<Range<editor::Anchor>>>)> = Vec::new();
        for (cell_id, editor) in self.ordered_cell_editors(cx) {
            let task = editor.update(cx, |editor, cx| {
                SearchableItem::find_matches(editor, query.clone(), window, cx)
            });
            cell_tasks.push((cell_id, task));
        }
        cx.background_spawn(async move {
            let mut all_matches = Vec::new();
            for (cell_id, task) in cell_tasks {
                for range in task.await {
                    all_matches.push(NotebookSearchMatch {
                        cell_id: cell_id.clone(),
                        range,
                    });
                }
            }
            all_matches
        })
    }

    fn active_match_index(
        &mut self,
        direction: Direction,
        matches: &[Self::Match],
        _token: SearchToken,
        _window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Option<usize> {
        if matches.is_empty() {
            return None;
        }
        let selected = self.selected_cell_index;

        // If the selected cell has matches, resolve the current match relative to
        // that cell's cursor using the per-editor binary search, then map its
        // local index back to the global index.
        if let Some(cell_id) = self.cell_order.get(selected).cloned() {
            let local: Vec<(usize, Range<editor::Anchor>)> = matches
                .iter()
                .enumerate()
                .filter(|(_, m)| m.cell_id == cell_id)
                .map(|(global, m)| (global, m.range.clone()))
                .collect();
            if !local.is_empty()
                && let Some(editor) = self
                    .cell_map
                    .get(&cell_id)
                    .and_then(|cell| cell.editor(cx).cloned())
            {
                let (cursor, snapshot) = editor.update(cx, |editor, cx| {
                    let snapshot = editor.buffer().read(cx).snapshot(cx);
                    let cursor = editor.selections.newest_anchor().head();
                    (cursor, snapshot)
                });
                let ranges: Vec<Range<editor::Anchor>> =
                    local.iter().map(|(_, range)| range.clone()).collect();
                if let Some(local_index) =
                    editor::items::active_match_index(direction, &ranges, &cursor, &snapshot)
                {
                    return local.get(local_index).map(|(global, _)| *global);
                }
            }
        }

        // Otherwise fall back to the nearest match by cell position.
        match direction {
            Direction::Next => matches
                .iter()
                .position(|m| {
                    self.cell_index_of(&m.cell_id)
                        .is_some_and(|ci| ci >= selected)
                })
                .or(Some(0)),
            Direction::Prev => matches
                .iter()
                .rposition(|m| {
                    self.cell_index_of(&m.cell_id)
                        .is_some_and(|ci| ci <= selected)
                })
                .or(Some(matches.len() - 1)),
        }
    }
}

impl Item for NotebookEditor {
    type Event = ();

    fn can_split(&self) -> bool {
        true
    }

    fn clone_on_split(
        &self,
        _workspace_id: Option<workspace::WorkspaceId>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Task<Option<Entity<Self>>>
    where
        Self: Sized,
    {
        Task::ready(Some(cx.new(|cx| {
            Self::new(self.project.clone(), self.notebook_item.clone(), window, cx)
        })))
    }

    fn buffer_kind(&self, _: &App) -> workspace::item::ItemBufferKind {
        workspace::item::ItemBufferKind::Singleton
    }

    fn for_each_project_item(
        &self,
        cx: &App,
        f: &mut dyn FnMut(gpui::EntityId, &dyn project::ProjectItem),
    ) {
        f(self.notebook_item.entity_id(), self.notebook_item.read(cx))
    }

    fn tab_content_text(&self, _detail: usize, cx: &App) -> SharedString {
        // Derive the tab label from the ABSOLUTE path, not the worktree-relative
        // path: a notebook saved OUTSIDE any worktree lives in a single-file
        // worktree rooted at the file itself, so its relative path is empty and
        // `project_path.path.file_name()` is `None` (which fell back to
        // "Untitled" even though the file was named and saved). `path` is set on
        // both save and open, and is `None` only when genuinely untitled.
        self.notebook_item
            .read(cx)
            .path
            .as_ref()
            .and_then(|path| path.file_name())
            .map(|name| name.to_string_lossy().into_owned().into())
            .unwrap_or_else(|| SharedString::from("Untitled"))
    }

    fn suggested_filename(&self, cx: &App) -> SharedString {
        if self.notebook_item.read(cx).is_untitled() {
            "Untitled.ipynb".into()
        } else {
            self.tab_content_text(0, cx)
        }
    }

    /// Insist on `.ipynb`, whatever the user typed over the suggested name.
    ///
    /// Save-as can only ever write nbformat JSON, and the extension is the only
    /// thing that makes the result open as a notebook again — the path registry
    /// keys on it, and the save dialog can't be filtered to steer the choice.
    /// So a name with any other extension (or none) gets `.ipynb` APPENDED
    /// rather than replaced: `data.json` becomes `data.json.ipynb`, which keeps
    /// whatever the user typed intact and visible instead of quietly discarding
    /// part of their filename.
    fn adjust_save_as_path(&self, path: PathBuf, _cx: &App) -> PathBuf {
        if path
            .extension()
            .is_some_and(|extension| extension.eq_ignore_ascii_case("ipynb"))
        {
            return path;
        }
        let mut name = path.file_name().unwrap_or_default().to_os_string();
        name.push(".ipynb");
        path.with_file_name(name)
    }

    fn tab_content(&self, params: TabContentParams, _window: &Window, cx: &App) -> AnyElement {
        Label::new(self.tab_content_text(params.detail.unwrap_or(0), cx))
            .single_line()
            .color(params.text_color())
            .when(params.preview, |this| this.italic())
            // Struck through once the file is gone from disk, the same as a text
            // file's tab. The indicator is per-item — `TabContentParams` carries
            // no deleted flag and the default `tab_content` has no notion of one
            // — so it has to be applied here (phase 69).
            .when(Item::has_deleted_file(self, cx), |this| {
                this.strikethrough()
            })
            .into_any_element()
    }

    /// Whether the notebook's file has been deleted from disk while it stayed
    /// open. Drives the strikethrough above, the "this file has been deleted"
    /// prompt on close, and the `close_on_file_delete` setting — none of which
    /// worked for notebooks while this defaulted to `false`.
    fn has_deleted_file(&self, cx: &App) -> bool {
        self.notebook_item
            .read(cx)
            .buffer
            .as_ref()
            .and_then(|buffer| buffer.read(cx).file())
            .is_some_and(|file| file.disk_state().is_deleted())
    }

    /// The notebook's single `()` event means "something about this item that
    /// the tab shows has changed" — it is saved, or its file appeared/vanished.
    /// Mapping it to `UpdateTab` is what makes the pane re-read the title, the
    /// dirty state and `has_deleted_file`; without it the tab still repainted on
    /// ordinary redraws, but the deleted-file close prompt and the
    /// `close_on_file_delete` setting never ran for a notebook.
    ///
    /// Deliberately NOT mapped to `ItemEvent::Edit`: that schedules autosave,
    /// and `()` is not an edit signal — the notebook emits it on save.
    fn to_item_events(_event: &Self::Event, f: &mut dyn FnMut(ItemEvent)) {
        f(ItemEvent::UpdateTab);
        f(ItemEvent::UpdateBreadcrumbs);
    }

    fn tab_icon(&self, _window: &Window, _cx: &App) -> Option<Icon> {
        Some(IconName::Book.into())
    }

    fn show_toolbar(&self) -> bool {
        false
    }

    fn pixel_position_of_cursor(&self, cx: &App) -> Option<Point<Pixels>> {
        let cell_id = self.cell_order.get(self.selected_cell_index)?;
        let editor = self.cell_map.get(cell_id)?.editor(cx)?;
        editor.read(cx).pixel_position_of_cursor(cx)
    }

    fn as_searchable(
        &self,
        handle: &Entity<Self>,
        _: &App,
    ) -> Option<Box<dyn SearchableItemHandle>> {
        Some(Box::new(handle.clone()))
    }

    fn set_nav_history(
        &mut self,
        _: workspace::ItemNavHistory,
        _window: &mut Window,
        _: &mut Context<Self>,
    ) {
        // TODO
    }

    fn can_save(&self, cx: &App) -> bool {
        // An untitled notebook has nowhere to save TO yet — the workspace
        // routes it through the save-as prompt instead (phase 33).
        !self.notebook_item.read(cx).is_untitled()
    }

    fn can_save_as(&self, _cx: &App) -> bool {
        true
    }

    fn save(
        &mut self,
        _options: SaveOptions,
        project: Entity<Project>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Task<Result<()>> {
        let notebook = self.to_notebook(cx);
        let Some(path) = self.notebook_item.read(cx).path.clone() else {
            // Untitled: can_save() is false, so the workspace goes through the
            // save-as prompt instead of here.
            return Task::ready(Err(anyhow::anyhow!(
                "an untitled notebook must be saved via save-as"
            )));
        };
        let fs = project.read(cx).fs().clone();

        if !self.disk_changed_externally {
            self.mark_as_saved(cx);
            return cx.spawn(async move |this, cx| {
                let json = serde_json::to_string_pretty(&notebook)
                    .context("Failed to serialize notebook")?;
                // Recorded so the file watcher's reload event for this write
                // is recognized as our own save (see handle_external_change).
                this.update(cx, |this, _| {
                    this.last_saved_disk_text = Some(json.clone());
                })?;
                fs.atomic_write(path, json).await?;
                Ok(())
            });
        }

        // The file changed on disk while we had unsaved changes — confirm
        // before overwriting the external version.
        let answer = window.prompt(
            PromptLevel::Warning,
            "This notebook changed on disk since it was loaded.",
            Some("Saving will overwrite the on-disk changes with your version."),
            &["Overwrite", "Cancel"],
            cx,
        );

        cx.spawn_in(window, async move |this, cx| {
            if answer.await != Ok(0) {
                return Ok(());
            }
            let json =
                serde_json::to_string_pretty(&notebook).context("Failed to serialize notebook")?;
            this.update_in(cx, |this, window, cx| {
                this.disk_changed_externally = false;
                this.last_saved_disk_text = Some(json.clone());
                this.mark_as_saved(cx);
                // Overwriting the disk version resolves the conflict just like
                // a reload does, so take down the conflict toast.
                this.dismiss_conflict_toast(window, cx);
            })?;
            fs.atomic_write(path, json).await?;
            Ok(())
        })
    }

    fn save_as(
        &mut self,
        project: Entity<Project>,
        path: ProjectPath,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Task<Result<()>> {
        let notebook = self.to_notebook(cx);
        let fs = project.read(cx).fs().clone();

        let abs_path = project.read(cx).absolute_path(&path, cx);

        self.mark_as_saved(cx);

        cx.spawn_in(window, async move |this, cx| {
            let abs_path = abs_path.context("Failed to get absolute path")?;
            let json =
                serde_json::to_string_pretty(&notebook).context("Failed to serialize notebook")?;
            this.update(cx, |this, _| {
                this.last_saved_disk_text = Some(json.clone());
            })?;
            fs.atomic_write(abs_path.clone(), json).await?;

            // Attach the notebook to its new file (phase 33): from here on it
            // behaves like any opened notebook — tab title, dedup by entry,
            // external-change watch, per-notebook kernel memory. This also
            // re-points an already file-backed notebook that was save-as'd
            // elsewhere.
            let buffer = this
                .update(cx, |this, cx| {
                    this.project
                        .update(cx, |project, cx| project.open_buffer(path.clone(), cx))
                })?
                .await?;
            this.update_in(cx, |this, window, cx| {
                let entry_id = this
                    .project
                    .read(cx)
                    .entry_for_path(&path, cx)
                    .map(|entry| entry.id);
                this.worktree_id = path.worktree_id;
                this.notebook_item.update(cx, |item, cx| {
                    item.path = Some(abs_path);
                    item.project_path = Some(path);
                    item.id = entry_id;
                    item.buffer = Some(buffer.clone());
                    cx.emit(());
                });
                this.watch_backing_buffer(buffer, window, cx);
                cx.notify();
            })?;
            Ok(())
        })
    }

    fn reload(
        &mut self,
        _project: Entity<Project>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Task<Result<()>> {
        let Some(project_path) = self.notebook_item.read(cx).project_path.clone() else {
            // Untitled: nothing on disk to reload from.
            return Task::ready(Ok(()));
        };

        cx.spawn_in(window, async move |this, cx| {
            let buffer = this
                .update(cx, |this, cx| {
                    this.project
                        .update(cx, |project, cx| project.open_buffer(project_path, cx))
                })?
                .await?;

            let file_content = buffer.read_with(cx, |buffer, _| buffer.text());
            let notebook = Self::parse_notebook_text(&file_content)?;

            this.update_in(cx, |this, window, cx| {
                this.reload_cells_from_notebook(&notebook, window, cx);
                this.last_saved_disk_text = Some(file_content);
            })?;

            Ok(())
        })
    }

    fn is_dirty(&self, cx: &App) -> bool {
        self.restored_unsaved_changes
            || self.execution_state_changed
            || self.has_structural_changes()
            || self.has_content_changes(cx)
    }
}

impl ProjectItem for NotebookEditor {
    type Item = NotebookItem;

    fn for_project_item(
        project: Entity<Project>,
        _pane: Option<&Pane>,
        item: Entity<Self::Item>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Self {
        Self::new(project, item, window, cx)
    }
}

impl SerializableItem for NotebookEditor {
    fn serialized_item_kind() -> &'static str {
        "NotebookEditor"
    }

    fn cleanup(
        workspace_id: WorkspaceId,
        alive_items: Vec<ItemId>,
        _window: &mut Window,
        cx: &mut App,
    ) -> Task<Result<()>> {
        delete_unloaded_items(
            alive_items,
            workspace_id,
            "notebook_editors",
            &NotebookDb::global(cx),
            cx,
        )
    }

    fn serialize(
        &mut self,
        workspace: &mut Workspace,
        item_id: ItemId,
        _closing: bool,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Option<Task<Result<()>>> {
        let workspace_id = workspace.database_id()?;
        let abs_path = self.notebook_item.read(cx).path.clone();
        // Hot exit, on the same rule as `SerializedEditor`: keep the notebook's
        // JSON whenever it has UNSAVED changes, so quitting never loses them,
        // and keep only the path when it is clean, so restore just reopens the
        // file. Gating on DIRTY rather than on untitled is the whole point —
        // previously a saved notebook stored no contents at all and its unsaved
        // edits died with the process (phase 69).
        let notebook = Item::is_dirty(self, cx).then(|| self.to_notebook(cx));
        // Nothing to restore from: an untitled notebook with no content to keep.
        if abs_path.is_none() && notebook.is_none() {
            return None;
        }
        let mtime = self
            .notebook_item
            .read(cx)
            .buffer
            .as_ref()
            .and_then(|buffer| buffer.read(cx).saved_mtime());
        let db = NotebookDb::global(cx);
        Some(cx.spawn_in(window, async move |_this, cx| {
            cx.background_spawn(async move {
                // Encode off the main thread. `to_notebook` has to run on it
                // (it reads the cell entities), but a notebook carrying image
                // outputs is megabytes of base64 and this runs on every edit,
                // throttled only to `SERIALIZATION_THROTTLE_TIME`.
                let contents = notebook
                    .as_ref()
                    .and_then(|notebook| serde_json::to_string(notebook).log_err());
                if abs_path.is_none() && contents.is_none() {
                    return Ok(());
                }
                db.save_serialized_notebook(
                    item_id,
                    workspace_id,
                    SerializedNotebook {
                        abs_path,
                        contents,
                        mtime,
                    },
                )
                .await
                .context("failed to save serialized notebook")
            })
            .await
        }))
    }

    fn deserialize(
        project: Entity<Project>,
        _workspace: WeakEntity<Workspace>,
        workspace_id: WorkspaceId,
        item_id: ItemId,
        window: &mut Window,
        cx: &mut App,
    ) -> Task<Result<Entity<Self>>> {
        let serialized = match NotebookDb::global(cx).get_serialized_notebook(item_id, workspace_id)
        {
            Ok(Some(serialized)) => serialized,
            Ok(None) => {
                return Task::ready(Err(anyhow!(
                    "no serialized notebook for item {item_id} in workspace {workspace_id:?}"
                )));
            }
            Err(error) => return Task::ready(Err(error)),
        };

        if let Some(abs_path) = serialized.abs_path {
            // Saved: reopen by path via the normal notebook open route so the
            // file is loaded and watched exactly as a fresh open would be.
            let existing_project_path = project.update(cx, |project, cx| {
                project
                    .find_worktree(&abs_path, cx)
                    .map(|(worktree, path)| ProjectPath {
                        worktree_id: worktree.read(cx).id(),
                        path,
                    })
            });
            let saved_mtime = serialized.mtime;
            let unsaved_contents = serialized.contents;
            window.spawn(cx, async move |cx| {
                // Held until the notebook is open. An INVISIBLE worktree is
                // kept only WEAKLY by the worktree store (`WorktreeStore::add`),
                // so it survives just as long as someone holds a strong handle:
                // letting this drop before the open would destroy the worktree
                // out from under it and fail with "no such worktree". `Pane`'s
                // out-of-project save-as has the same guard for the same reason.
                let created_worktree;
                let project_path = match existing_project_path {
                    Some(project_path) => {
                        created_worktree = None;
                        project_path
                    }
                    None => {
                        // Not in any worktree: a notebook opened from outside
                        // every project root lives in an INVISIBLE single-file
                        // worktree, and only VISIBLE worktrees are saved as the
                        // workspace's roots — so at restore time that worktree
                        // does not exist yet. Recreate it the way
                        // `Project::open_local_buffer` does for a loose text
                        // file, instead of giving up and dropping the tab.
                        let (worktree, relative_path) = project
                            .update(cx, |project, cx| {
                                project.find_or_create_worktree(&abs_path, false, cx)
                            })
                            .await
                            .with_context(|| format!("opening loose notebook {abs_path:?}"))?;
                        let project_path = ProjectPath {
                            worktree_id: worktree.read_with(cx, |worktree, _| worktree.id()),
                            path: relative_path,
                        };
                        created_worktree = Some(worktree);
                        project_path
                    }
                };
                let open_task = cx.update(|_, cx| {
                    <NotebookItem as project::ProjectItem>::try_open(&project, &project_path, cx)
                })?;
                let Some(open_task) = open_task else {
                    anyhow::bail!("not a notebook path: {abs_path:?}");
                };
                let notebook_item = open_task
                    .await
                    .context("failed to open serialized notebook by path")?;
                // The opened buffer's `File` holds the worktree from here on.
                drop(created_worktree);
                // Contents are stored only for a notebook with UNSAVED changes,
                // so anything here is newer than what was just loaded from disk.
                let unsaved = unsaved_contents
                    .as_deref()
                    .and_then(|contents| NotebookEditor::parse_notebook_text(contents).log_err());
                cx.update(|window, cx| {
                    cx.new(|cx| {
                        let mut editor = NotebookEditor::new(project, notebook_item, window, cx);
                        if let Some(unsaved) = unsaved {
                            editor.restore_unsaved_notebook(&unsaved, saved_mtime, window, cx);
                        }
                        editor
                    })
                })
            })
        } else if let Some(contents) = serialized.contents {
            // Untitled: rebuild the item from the stored nbformat JSON.
            window.spawn(cx, async move |cx| {
                let notebook = NotebookEditor::parse_notebook_text(&contents)
                    .context("failed to parse serialized untitled notebook")?;
                cx.update(|window, cx| {
                    let languages = project.read(cx).languages().clone();
                    let notebook_item = cx
                        .new(|_| NotebookItem::untitled(project.downgrade(), languages, notebook));
                    cx.new(|cx| {
                        let mut editor = NotebookEditor::new(project, notebook_item, window, cx);
                        // Contents are only stored for a notebook with unsaved
                        // changes, so a restored untitled notebook has content
                        // that exists NOWHERE on disk — it is unsaved by
                        // definition and has to say so. Rebuilding it makes the
                        // restored cells their own baseline, which otherwise
                        // reads as clean until the next keystroke (bug #75).
                        editor.restored_unsaved_changes = true;
                        editor
                    })
                })
            })
        } else {
            // Untitled and clean at quit, so nothing was worth keeping: come
            // back as a fresh notebook rather than losing the tab. (`Editor`
            // does the same for a path-less, contents-less row.)
            window.spawn(cx, async move |cx| {
                let notebook = NotebookEditor::empty_notebook()
                    .context("failed to build an empty notebook template")?;
                cx.update(|window, cx| {
                    let languages = project.read(cx).languages().clone();
                    let notebook_item = cx
                        .new(|_| NotebookItem::untitled(project.downgrade(), languages, notebook));
                    cx.new(|cx| NotebookEditor::new(project, notebook_item, window, cx))
                })
            })
        }
    }

    fn should_serialize(&self, _event: &Self::Event) -> bool {
        // The notebook has a single `()` event, so there is no event set to
        // filter on the way `Editor` filters `EditorEvent`. The cost control
        // lives in `serialize` instead: it encodes the notebook's JSON only
        // when there are unsaved changes, and does the encode off the main
        // thread. `()` is emitted deliberately and rarely — see `mark_as_saved`.
        true
    }
}

impl KernelSession for NotebookEditor {
    fn route(&mut self, message: &JupyterMessage, window: &mut Window, cx: &mut Context<Self>) {
        // Handle kernel status updates (these are broadcast to all)
        if let JupyterMessageContent::Status(status) = &message.content {
            self.kernel.set_execution_state(&status.execution_state);
            // A batch that superseded an in-flight run waited for the
            // interrupted run to finish aborting; now that the kernel is idle
            // again, submit the new queue.
            if self.resume_run_queue_on_idle
                && matches!(status.execution_state, ExecutionState::Idle)
                && self.active_run_cell.is_none()
                && !self.run_queue.is_empty()
            {
                self.resume_run_queue_on_idle = false;
                self.advance_run_queue(window, cx);
            }
            cx.notify();
        }

        if let JupyterMessageContent::KernelInfoReply(reply) = &message.content {
            self.kernel.set_kernel_info(reply);

            if let Ok(language_info) = serde_json::from_value::<nbformat::v4::LanguageInfo>(
                serde_json::to_value(&reply.language_info).unwrap(),
            ) {
                self.notebook_item.update(cx, |item, cx| {
                    item.notebook.metadata.language_info = Some(language_info);
                    cx.emit(());
                });
            }
            cx.notify();
        }

        // Handle cell-specific messages
        if let Some(parent_header) = &message.parent_header {
            if let Some(cell_id) = self.execution_requests.get(&parent_header.msg_id) {
                if let Some(Cell::Code(cell)) = self.cell_map.get(cell_id) {
                    // Outputs / execution counts arriving are savable state —
                    // but only for messages that actually change it. Status
                    // (busy/idle) broadcasts and replies must not re-dirty a
                    // notebook that was just saved.
                    if matches!(
                        &message.content,
                        JupyterMessageContent::StreamContent(_)
                            | JupyterMessageContent::DisplayData(_)
                            | JupyterMessageContent::ExecuteResult(_)
                            | JupyterMessageContent::ExecuteInput(_)
                            | JupyterMessageContent::ErrorOutput(_)
                    ) {
                        self.execution_state_changed = true;
                    }
                    cell.update(cx, |cell, cx| {
                        cell.handle_message(message, window, cx);
                    });
                }
            }
        }

        // Advance (or stop) a multi-cell run when the active cell finishes.
        if let JupyterMessageContent::ExecuteReply(reply) = &message.content {
            let finished_cell = message
                .parent_header
                .as_ref()
                .and_then(|header| self.execution_requests.get(&header.msg_id).cloned());
            if let Some(finished_cell) = finished_cell
                && self.active_run_cell.as_ref() == Some(&finished_cell)
            {
                self.active_run_cell = None;
                if matches!(reply.status, ReplyStatus::Error | ReplyStatus::Aborted) {
                    // Stop-on-error (or the kernel aborting after an
                    // interrupt): cancel the rest of the batch, clearing the
                    // queued cells' pending markers.
                    if !self.run_queue.is_empty() {
                        log::info!(
                            "notebook: cell errored/aborted; cancelling {} queued cell(s)",
                            self.run_queue.len()
                        );
                        self.cancel_run_queue(cx);
                    }
                } else {
                    self.advance_run_queue(window, cx);
                }
            }
        }

        // Cell statuses have just been updated from this message, so this is
        // where a cell starts or stops running as far as the notebook is
        // concerned — keep the execution clock in step with it.
        self.sync_execution_clock(cx);
    }

    fn kernel_errored(&mut self, error_message: String, cx: &mut Context<Self>) {
        // Errors from a kernel that is being torn down are expected; don't
        // clobber the Restarting/ShuttingDown state with ErroredLaunch.
        if self.kernel.is_shutting_down() {
            log::info!("notebook: ignoring kernel error during shutdown/restart: {error_message}");
            return;
        }
        self.kernel = Kernel::ErroredLaunch(error_message);
        self.execution_requests.clear();
        self.cancel_run_queue(cx);
        self.stop_executing_cells(cx);
        self.freeze_kernel_uptime();
        cx.notify();
    }

    fn kernel_exited(&mut self, cx: &mut Context<Self>) {
        if self.kernel.is_shutting_down() {
            return;
        }
        self.kernel = Kernel::Shutdown;
        self.execution_requests.clear();
        self.cancel_run_queue(cx);
        self.stop_executing_cells(cx);
        self.freeze_kernel_uptime();
        cx.notify();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::LocalKernelSpecification;
    use gpui::TestAppContext;
    use jupyter_protocol::JupyterKernelspec;
    use project::Fs as _;
    use project::{FakeFs, Project, ProjectItem as _};
    use serde_json::json;
    use settings::SettingsStore;
    use util::path;
    use util::rel_path::rel_path;

    #[test]
    fn test_format_duration_tiers() {
        assert_eq!(format_duration(Duration::from_millis(0)), "0ms");
        assert_eq!(format_duration(Duration::from_millis(142)), "142ms");
        assert_eq!(format_duration(Duration::from_millis(999)), "999ms");
        assert_eq!(format_duration(Duration::from_millis(1000)), "1.0s");
        assert_eq!(format_duration(Duration::from_millis(1830)), "1.8s");
        assert_eq!(format_duration(Duration::from_secs(59)), "59.0s");
        assert_eq!(format_duration(Duration::from_secs(60)), "1m 0.0s");
        assert_eq!(format_duration(Duration::from_millis(184_200)), "3m 4.2s");
        assert_eq!(format_duration(Duration::from_secs(3599)), "59m 59.0s");
        assert_eq!(format_duration(Duration::from_secs(3600)), "1h 0m 0.0s");
        assert_eq!(format_duration(Duration::from_secs(9163)), "2h 32m 43.0s");
    }

    #[test]
    fn test_error_navigation_wraps_in_document_order() {
        let failed = [2usize, 5, 9];

        assert_eq!(next_failed_index(&failed, 0), Some(2));
        assert_eq!(next_failed_index(&failed, 2), Some(5));
        assert_eq!(next_failed_index(&failed, 5), Some(9));
        // Past the last failure, wrap to the first.
        assert_eq!(next_failed_index(&failed, 9), Some(2));
        assert_eq!(next_failed_index(&failed, 42), Some(2));

        assert_eq!(previous_failed_index(&failed, 9), Some(5));
        assert_eq!(previous_failed_index(&failed, 5), Some(2));
        // Before the first failure, wrap to the last.
        assert_eq!(previous_failed_index(&failed, 2), Some(9));
        assert_eq!(previous_failed_index(&failed, 0), Some(9));

        // A single failure is its own next and previous, so repeatedly
        // pressing either key keeps landing on it rather than doing nothing.
        assert_eq!(next_failed_index(&[4], 4), Some(4));
        assert_eq!(previous_failed_index(&[4], 4), Some(4));

        assert_eq!(next_failed_index(&[], 0), None);
        assert_eq!(previous_failed_index(&[], 0), None);
    }

    const NOTEBOOK_WITH_ONE_CODE_CELL: &str = r#"{
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5,
        "cells": [
            {
                "cell_type": "code",
                "id": "cell-one",
                "metadata": {},
                "execution_count": null,
                "outputs": [],
                "source": ["print('hello')"]
            }
        ]
    }"#;

    /// When the remembered kernel's interpreter no longer exists (deleted
    /// env / Python uninstalled), running a cell must not launch it or leave
    /// the cell stuck executing: the stale selection is dropped and the cell
    /// is held for a kernel prompt instead (phase 42).
    #[gpui::test]
    async fn test_run_cell_with_missing_interpreter_prompts(cx: &mut TestAppContext) {
        cx.update(|cx| {
            let settings_store = SettingsStore::test(cx);
            cx.set_global(settings_store);
            theme_settings::init(theme::LoadThemes::JustBase, cx);
            editor::init(cx);
        });

        let fs = FakeFs::new(cx.executor());
        fs.insert_tree(
            path!("/notebooks"),
            json!({ "test.ipynb": NOTEBOOK_WITH_ONE_CODE_CELL }),
        )
        .await;

        let project = Project::test(fs.clone(), [path!("/notebooks").as_ref()], cx).await;
        cx.update(|cx| ReplStore::init(fs.clone(), cx));

        let worktree_id = project.read_with(cx, |project, cx| {
            project.worktrees(cx).next().unwrap().read(cx).id()
        });

        // Select a kernel whose interpreter doesn't exist, simulating a machine
        // where Python isn't installed properly. This is the same path the
        // kernel picker uses.
        let missing_interpreter = path!("/nonexistent/python3");
        let broken_spec = KernelSpecification::Jupyter(LocalKernelSpecification {
            name: "python3".to_string(),
            path: PathBuf::from(missing_interpreter),
            kernelspec: JupyterKernelspec {
                argv: vec![
                    missing_interpreter.to_string(),
                    "-m".to_string(),
                    "ipykernel_launcher".to_string(),
                    "-f".to_string(),
                    "{connection_file}".to_string(),
                ],
                display_name: "Python 3".to_string(),
                language: "python".to_string(),
                interrupt_mode: None,
                metadata: None,
                env: None,
            },
        });
        cx.update(|cx| {
            ReplStore::global(cx).update(cx, |store, cx| {
                store.set_notebook_kernelspec(
                    PathBuf::from(path!("/notebooks/test.ipynb")),
                    broken_spec,
                    cx,
                );
            })
        });

        let notebook_item = cx
            .update(|cx| {
                NotebookItem::try_open(
                    &project,
                    &ProjectPath {
                        worktree_id,
                        path: rel_path("test.ipynb").into(),
                    },
                    cx,
                )
                .expect("ipynb files should be openable as notebooks")
            })
            .await
            .expect("notebook should parse");

        // Don't render the notebook UI itself: its animated kernel status icon
        // schedules a new frame on every render, which makes `run_until_parked`
        // spin forever in tests. The editor entity is created inside an empty
        // window instead; we are testing execution behavior, not rendering.
        let cx = cx.add_empty_window();

        // Launching a kernel probes real TCP ports on localhost, which the
        // deterministic test scheduler cannot drive.
        cx.executor().allow_parking();

        let editor = cx.update(|window, cx| {
            cx.new(|cx| NotebookEditor::new(project.clone(), notebook_item, window, cx))
        });

        // Lazy start: creating the editor does NOT launch a kernel; it stays
        // shut down until a cell is run (or a kernel is explicitly selected).
        editor.read_with(cx, |editor, _| {
            assert!(
                matches!(editor.kernel, Kernel::Shutdown),
                "kernel should not start on open, instead status is: {}",
                editor.kernel.status().to_string()
            );
        });

        // Run the (only) cell via the production action handler. Since phase
        // 42, a remembered kernel whose interpreter no longer exists is NOT
        // launched — the run validates the env first, drops the stale
        // selection, and prompts for a kernel instead of erroring.
        editor.update_in(cx, |editor, window, cx| {
            editor.run_current_cell(&Run, window, cx);
        });

        editor.read_with(cx, |editor, cx| {
            assert!(
                matches!(editor.kernel, Kernel::Shutdown),
                "a vanished env must not be launched; kernel is: {}",
                editor.kernel.status().to_string()
            );
            assert!(
                editor.kernel_specification.is_none(),
                "the stale selection must be dropped"
            );
            assert!(
                ReplStore::global(cx)
                    .read(cx)
                    .notebook_kernelspec(std::path::Path::new(path!("/notebooks/test.ipynb")))
                    .is_none(),
                "the notebook's remembered pick must be forgotten"
            );

            let cell_id = editor.cell_order.first().expect("notebook has one cell");
            assert!(
                editor.cells_awaiting_kernel_choice.contains(cell_id),
                "the cell should be held awaiting a kernel choice (picker prompt)"
            );
            let Some(Cell::Code(cell)) = editor.cell_map.get(cell_id) else {
                panic!("expected a code cell");
            };
            assert!(
                !cell.read(cx).is_executing(),
                "cell must not be stuck in the executing state when the kernel is not running"
            );
        });
    }

    const NOTEBOOK_WITH_MIXED_CELLS: &str = r##"{
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5,
        "cells": [
            {
                "cell_type": "markdown",
                "id": "cell-md",
                "metadata": {},
                "source": ["# Heading\n", "Some text"]
            },
            {
                "cell_type": "code",
                "id": "cell-one",
                "metadata": {},
                "execution_count": 3,
                "outputs": [
                    {
                        "output_type": "stream",
                        "name": "stdout",
                        "text": ["hello\n"]
                    },
                    {
                        "output_type": "execute_result",
                        "execution_count": 3,
                        "metadata": {},
                        "data": { "text/plain": ["42"] }
                    }
                ],
                "source": ["print('hello')\n", "42"]
            },
            {
                "cell_type": "code",
                "id": "cell-two",
                "metadata": { "jupyter": { "source_hidden": true } },
                "execution_count": null,
                "outputs": [],
                "source": ["x = 1"]
            }
        ]
    }"##;

    /// Merely opening a notebook must not mark it dirty: nothing has changed
    /// until the user edits, runs, or toggles something (bug #32 repro).
    #[gpui::test]
    async fn test_opening_a_notebook_is_not_dirty(cx: &mut TestAppContext) {
        cx.update(|cx| {
            let settings_store = SettingsStore::test(cx);
            cx.set_global(settings_store);
            theme_settings::init(theme::LoadThemes::JustBase, cx);
            editor::init(cx);
        });

        let fs = FakeFs::new(cx.executor());
        fs.insert_tree(
            path!("/notebooks"),
            json!({ "test.ipynb": NOTEBOOK_WITH_MIXED_CELLS }),
        )
        .await;

        let project = Project::test(fs.clone(), [path!("/notebooks").as_ref()], cx).await;
        cx.update(|cx| ReplStore::init(fs.clone(), cx));

        let worktree_id = project.read_with(cx, |project, cx| {
            project.worktrees(cx).next().unwrap().read(cx).id()
        });

        let notebook_item = cx
            .update(|cx| {
                NotebookItem::try_open(
                    &project,
                    &ProjectPath {
                        worktree_id,
                        path: rel_path("test.ipynb").into(),
                    },
                    cx,
                )
                .expect("ipynb files should be openable as notebooks")
            })
            .await
            .expect("notebook should parse");

        let cx = cx.add_empty_window();
        let editor = cx.update(|window, cx| {
            cx.new(|cx| NotebookEditor::new(project.clone(), notebook_item, window, cx))
        });

        // Let async open work (language loading, kernelspec discovery
        // observers) settle before checking.
        cx.run_until_parked();

        editor.read_with(cx, |editor, cx| {
            assert!(
                !editor.execution_state_changed,
                "opening must not count as an execution-state change"
            );
            assert!(
                !editor.has_structural_changes(),
                "opening must not count as a structural change"
            );
            assert!(
                !editor.has_content_changes(cx),
                "opening must not make any cell buffer dirty"
            );
        });
    }

    /// A notebook opened from OUTSIDE every project root lives in an invisible
    /// single-file worktree, which is not part of the saved workspace roots — so
    /// at restore time its path belongs to no worktree and one has to be
    /// recreated, exactly as `Project::open_local_buffer` does for a loose text
    /// file.
    ///
    /// The trap this pins down: an invisible worktree is held only WEAKLY by the
    /// worktree store, so recreating one and then letting the handle go destroys
    /// it before anything can be opened in it. `deserialize` has to keep it
    /// alive until the notebook is open, and this test fails with "no such
    /// worktree" if it doesn't (bug #79).
    #[gpui::test]
    async fn test_loose_notebook_is_restored(cx: &mut TestAppContext) {
        cx.update(|cx| {
            let settings_store = SettingsStore::test(cx);
            cx.set_global(settings_store);
            theme_settings::init(theme::LoadThemes::JustBase, cx);
            editor::init(cx);
        });

        let fs = FakeFs::new(cx.executor());
        fs.insert_tree(
            path!("/elsewhere"),
            json!({ "loose.ipynb": NOTEBOOK_WITH_MIXED_CELLS }),
        )
        .await;
        // A project with NO worktree covering the notebook — the state a
        // restored session starts in for a loose file.
        let project = Project::test(fs.clone(), [], cx).await;
        cx.update(|cx| ReplStore::init(fs.clone(), cx));

        let abs_path = PathBuf::from(path!("/elsewhere/loose.ipynb"));
        assert!(
            project
                .update(cx, |project, cx| project.find_worktree(&abs_path, cx))
                .is_none(),
            "precondition: the notebook's path is in no worktree"
        );

        let workspace_db = cx.update(|cx| workspace::WorkspaceDb::global(cx));
        let workspace_id = workspace_db.next_id().await.unwrap();
        let notebook_db = cx.update(|cx| NotebookDb::global(cx));
        notebook_db
            .save_serialized_notebook(
                1234,
                workspace_id,
                SerializedNotebook {
                    abs_path: Some(abs_path.clone()),
                    contents: None,
                    mtime: None,
                },
            )
            .await
            .unwrap();

        let cx = cx.add_empty_window();
        let workspace =
            cx.update(|window, cx| cx.new(|cx| Workspace::test_new(project.clone(), window, cx)));

        let editor = cx
            .update(|window, cx| {
                NotebookEditor::deserialize(
                    project.clone(),
                    workspace.downgrade(),
                    workspace_id,
                    1234,
                    window,
                    cx,
                )
            })
            .await
            .expect("a loose notebook must be restored, not dropped");

        editor.read_with(cx, |editor, _| {
            assert_eq!(
                editor.cell_count(),
                NotebookEditor::parse_notebook_text(NOTEBOOK_WITH_MIXED_CELLS)
                    .unwrap()
                    .cells
                    .len(),
                "the restored notebook should hold the file's real content"
            );
        });
    }

    /// Hot exit (phase 69): unsaved changes restored over the copy read from
    /// disk must READ as unsaved — the restored cells become the baseline, so
    /// without an explicit marker the tab would look clean, closing wouldn't
    /// prompt, and the next serialize would drop the changes just recovered.
    #[gpui::test]
    async fn test_restored_unsaved_notebook_is_dirty_until_saved(cx: &mut TestAppContext) {
        cx.update(|cx| {
            let settings_store = SettingsStore::test(cx);
            cx.set_global(settings_store);
            theme_settings::init(theme::LoadThemes::JustBase, cx);
            editor::init(cx);
        });

        let fs = FakeFs::new(cx.executor());
        fs.insert_tree(
            path!("/notebooks"),
            json!({ "test.ipynb": NOTEBOOK_WITH_MIXED_CELLS }),
        )
        .await;
        let project = Project::test(fs.clone(), [path!("/notebooks").as_ref()], cx).await;
        cx.update(|cx| ReplStore::init(fs.clone(), cx));
        let worktree_id = project.read_with(cx, |project, cx| {
            project.worktrees(cx).next().unwrap().read(cx).id()
        });

        let notebook_item = cx
            .update(|cx| {
                NotebookItem::try_open(
                    &project,
                    &ProjectPath {
                        worktree_id,
                        path: rel_path("test.ipynb").into(),
                    },
                    cx,
                )
                .expect("ipynb files should be openable as notebooks")
            })
            .await
            .expect("notebook should parse");

        let cx = cx.add_empty_window();
        let editor = cx.update(|window, cx| {
            cx.new(|cx| NotebookEditor::new(project.clone(), notebook_item, window, cx))
        });
        cx.run_until_parked();

        editor.read_with(cx, |editor, cx| {
            assert!(
                !Item::is_dirty(editor, cx),
                "a notebook just opened from disk is clean"
            );
        });

        // What a previous session was holding: the same notebook with an extra
        // cell that never reached disk.
        let mut unsaved =
            NotebookEditor::parse_notebook_text(NOTEBOOK_WITH_MIXED_CELLS).expect("should parse");
        unsaved.cells.push(NotebookEditor::empty_code_cell());
        let restored_cell_count = unsaved.cells.len();

        editor.update_in(cx, |editor, window, cx| {
            editor.restore_unsaved_notebook(&unsaved, None, window, cx);
        });
        cx.run_until_parked();

        editor.read_with(cx, |editor, cx| {
            assert_eq!(
                editor.cell_count(),
                restored_cell_count,
                "the restored cells are what the user sees"
            );
            assert!(
                Item::is_dirty(editor, cx),
                "restored unsaved changes must mark the notebook dirty, even though \
                 they are now its own baseline"
            );
            assert!(
                !editor.disk_changed_externally,
                "the file did not move under us, so this is not a conflict"
            );
        });

        editor.update(cx, |editor, cx| editor.mark_as_saved(cx));
        editor.read_with(cx, |editor, cx| {
            assert!(
                !Item::is_dirty(editor, cx),
                "saving clears the restored-unsaved marker like every other dirty flag"
            );
        });
    }

    /// If the file moved on disk while Zed was closed, the restored changes are
    /// no longer based on it — that is the existing overwrite-conflict case, and
    /// the stored mtime is the only way to notice.
    #[gpui::test]
    async fn test_restored_unsaved_notebook_flags_a_stale_base(cx: &mut TestAppContext) {
        cx.update(|cx| {
            let settings_store = SettingsStore::test(cx);
            cx.set_global(settings_store);
            theme_settings::init(theme::LoadThemes::JustBase, cx);
            editor::init(cx);
        });

        let fs = FakeFs::new(cx.executor());
        fs.insert_tree(
            path!("/notebooks"),
            json!({ "test.ipynb": NOTEBOOK_WITH_MIXED_CELLS }),
        )
        .await;
        let project = Project::test(fs.clone(), [path!("/notebooks").as_ref()], cx).await;
        cx.update(|cx| ReplStore::init(fs.clone(), cx));
        let worktree_id = project.read_with(cx, |project, cx| {
            project.worktrees(cx).next().unwrap().read(cx).id()
        });

        let notebook_item = cx
            .update(|cx| {
                NotebookItem::try_open(
                    &project,
                    &ProjectPath {
                        worktree_id,
                        path: rel_path("test.ipynb").into(),
                    },
                    cx,
                )
                .expect("ipynb files should be openable as notebooks")
            })
            .await
            .expect("notebook should parse");

        let cx = cx.add_empty_window();
        let editor = cx.update(|window, cx| {
            cx.new(|cx| NotebookEditor::new(project.clone(), notebook_item, window, cx))
        });
        cx.run_until_parked();

        let unsaved =
            NotebookEditor::parse_notebook_text(NOTEBOOK_WITH_MIXED_CELLS).expect("should parse");

        // An mtime that cannot match what the file actually has.
        let stale = Some(fs::MTime::from_seconds_and_nanos(1, 0));
        editor.update_in(cx, |editor, window, cx| {
            editor.restore_unsaved_notebook(&unsaved, stale, window, cx);
        });
        cx.run_until_parked();

        editor.read_with(cx, |editor, _| {
            assert!(
                editor.disk_changed_externally,
                "a file edited while Zed was closed must raise the overwrite conflict \
                 rather than being silently overwritten by the restored changes"
            );
        });
    }

    /// An untitled notebook (phase 33) lives purely in memory: no file on
    /// disk, "Untitled" tab, save routed through save-as — and the first
    /// save-as attaches it to its new file so it behaves like an opened
    /// notebook from then on.
    #[gpui::test]
    async fn test_untitled_notebook_saves_via_save_as(cx: &mut TestAppContext) {
        cx.update(|cx| {
            let settings_store = SettingsStore::test(cx);
            cx.set_global(settings_store);
            theme_settings::init(theme::LoadThemes::JustBase, cx);
            editor::init(cx);
        });

        let fs = FakeFs::new(cx.executor());
        fs.insert_tree(path!("/notebooks"), json!({})).await;

        let project = Project::test(fs.clone(), [path!("/notebooks").as_ref()], cx).await;
        cx.update(|cx| ReplStore::init(fs.clone(), cx));

        let worktree_id = project.read_with(cx, |project, cx| {
            project.worktrees(cx).next().unwrap().read(cx).id()
        });

        let template = NotebookEditor::empty_notebook().expect("template should build");
        let languages = project.read_with(cx, |project, _| project.languages().clone());
        let notebook_item = cx.update(|cx| {
            cx.new(|_| NotebookItem::untitled(project.downgrade(), languages, template))
        });

        let cx = cx.add_empty_window();
        let editor = cx.update(|window, cx| {
            cx.new(|cx| NotebookEditor::new(project.clone(), notebook_item, window, cx))
        });
        cx.run_until_parked();

        assert!(
            !fs.is_file(std::path::Path::new(path!("/notebooks/Untitled.ipynb")))
                .await,
            "creating an untitled notebook must not write anything to disk"
        );
        editor.read_with(cx, |editor, cx| {
            assert!(
                !Item::can_save(editor, cx),
                "an untitled notebook has nowhere to save to"
            );
            assert!(Item::can_save_as(editor, cx));
            assert_eq!(Item::tab_content_text(editor, 0, cx), "Untitled");
            assert_eq!(Item::suggested_filename(editor, cx), "Untitled.ipynb");
            assert!(
                !Item::is_dirty(editor, cx),
                "a fresh untitled notebook starts clean"
            );
        });

        let new_path = ProjectPath {
            worktree_id,
            path: rel_path("analysis.ipynb").into(),
        };
        editor
            .update_in(cx, |editor, window, cx| {
                Item::save_as(editor, project.clone(), new_path, window, cx)
            })
            .await
            .expect("save-as should succeed");
        cx.run_until_parked();

        assert!(
            fs.is_file(std::path::Path::new(path!("/notebooks/analysis.ipynb")))
                .await,
            "save-as must create the chosen file"
        );
        editor.read_with(cx, |editor, cx| {
            assert!(
                Item::can_save(editor, cx),
                "after save-as the notebook is file-backed and saves in place"
            );
            assert_eq!(Item::tab_content_text(editor, 0, cx), "analysis.ipynb");
            assert!(
                !editor.notebook_item.read(cx).is_untitled(),
                "save-as must attach the notebook to its file"
            );
        });
    }

    /// Save-as can only write nbformat JSON, and the dialog can't be filtered
    /// to `.ipynb`, so whatever the user types has to end up with the extension
    /// that makes the file open as a notebook again.
    #[gpui::test]
    async fn test_save_as_path_always_gets_the_notebook_extension(cx: &mut TestAppContext) {
        cx.update(|cx| {
            let settings_store = SettingsStore::test(cx);
            cx.set_global(settings_store);
            theme_settings::init(theme::LoadThemes::JustBase, cx);
            editor::init(cx);
        });

        let fs = FakeFs::new(cx.executor());
        fs.insert_tree(path!("/notebooks"), json!({})).await;
        let project = Project::test(fs.clone(), [path!("/notebooks").as_ref()], cx).await;
        cx.update(|cx| ReplStore::init(fs.clone(), cx));

        let template = NotebookEditor::empty_notebook().expect("template should build");
        let languages = project.read_with(cx, |project, _| project.languages().clone());
        let notebook_item = cx.update(|cx| {
            cx.new(|_| NotebookItem::untitled(project.downgrade(), languages, template))
        });

        let cx = cx.add_empty_window();
        let editor = cx.update(|window, cx| {
            cx.new(|cx| NotebookEditor::new(project.clone(), notebook_item, window, cx))
        });
        cx.run_until_parked();

        editor.read_with(cx, |editor, cx| {
            let adjust = |path: &str| {
                Item::adjust_save_as_path(editor, PathBuf::from(path), cx)
                    .to_string_lossy()
                    .into_owned()
            };

            assert_eq!(
                adjust(path!("/notebooks/analysis")),
                path!("/notebooks/analysis.ipynb"),
                "a bare typed name is the case that used to produce an unopenable file"
            );
            assert_eq!(
                adjust(path!("/notebooks/analysis.ipynb")),
                path!("/notebooks/analysis.ipynb"),
                "a name that already ends in .ipynb is left exactly as typed"
            );
            assert_eq!(
                adjust(path!("/notebooks/ANALYSIS.IPYNB")),
                path!("/notebooks/ANALYSIS.IPYNB"),
                "the extension check is case-insensitive, so no doubling up"
            );
            assert_eq!(
                adjust(path!("/notebooks/data.json")),
                path!("/notebooks/data.json.ipynb"),
                "another extension is APPENDED to rather than replaced, so \
                 nothing the user typed is silently dropped"
            );
            assert_eq!(
                adjust(path!("/notebooks/analysis.v2")),
                path!("/notebooks/analysis.v2.ipynb"),
                "a version-suffixed name reads as an extension to the OS too, \
                 and still has to end up openable"
            );
        });
    }

    /// A cell's OUTPUTS must survive the snapshot round-trip used by
    /// copy/cut/paste and delete/undo: live cell → nbformat (what the
    /// clipboard and the undo stack store) → live cell → nbformat again
    /// (phase 27). Covers stream, plain-result, and rich (markdown) outputs —
    /// the rich kind is what bug #24 used to drop.
    #[gpui::test]
    async fn test_cell_outputs_round_trip_through_snapshot(cx: &mut TestAppContext) {
        cx.update(|cx| {
            let settings_store = SettingsStore::test(cx);
            cx.set_global(settings_store);
            theme_settings::init(theme::LoadThemes::JustBase, cx);
            editor::init(cx);
        });
        let cx = cx.add_empty_window();

        let cell_json = r##"{
            "cell_type": "code",
            "id": "cell-out",
            "metadata": {},
            "execution_count": 7,
            "outputs": [
                { "output_type": "stream", "name": "stdout", "text": ["hello\n"] },
                {
                    "output_type": "execute_result",
                    "execution_count": 7,
                    "metadata": {},
                    "data": { "text/plain": ["42"] }
                },
                {
                    "output_type": "display_data",
                    "metadata": {},
                    "data": { "text/markdown": ["**bold**"] }
                }
            ],
            "source": ["print('hello')"]
        }"##;
        let nbformat_cell: nbformat::v4::Cell =
            serde_json::from_str(cell_json).expect("fixture cell should parse");

        let languages = Arc::new(LanguageRegistry::test(cx.executor()));
        let notebook_language = Task::ready(None).shared();

        // Load exactly like paste/undo do (raw_insert_cell → Cell::load), then
        // snapshot exactly like copy/cut/delete do (to_nbformat_cell).
        let (loaded_count, resnapshotted) = cx.update(|window, cx| {
            let cell = Cell::load(&nbformat_cell, &languages, notebook_language, window, cx);
            let Cell::Code(code_cell) = &cell else {
                panic!("expected a code cell");
            };
            let loaded_count = code_cell.read(cx).outputs().len();
            (loaded_count, cell.to_nbformat_cell(cx))
        });

        assert_eq!(loaded_count, 3, "all three outputs should load");
        let nbformat::v4::Cell::Code {
            outputs,
            execution_count,
            ..
        } = resnapshotted
        else {
            panic!("expected a code cell");
        };
        assert_eq!(execution_count, Some(7), "execution count should survive");
        assert_eq!(
            outputs.len(),
            3,
            "all three outputs should serialize back (rich ones included)"
        );
    }

    /// The last-executed timestamps saved in `metadata.execution` (VS Code /
    /// Jupyter shape) must restore a loaded cell's run record — ✓ status,
    /// duration, completion time — and survive re-serialization (phase 28).
    #[gpui::test]
    async fn test_execution_timestamps_round_trip(cx: &mut TestAppContext) {
        cx.update(|cx| {
            let settings_store = SettingsStore::test(cx);
            cx.set_global(settings_store);
            theme_settings::init(theme::LoadThemes::JustBase, cx);
            editor::init(cx);
        });
        let cx = cx.add_empty_window();

        let cell_json = r##"{
            "cell_type": "code",
            "id": "cell-timed",
            "metadata": {
                "execution": {
                    "iopub.status.busy": "2026-07-14T10:00:00.000Z",
                    "iopub.status.idle": "2026-07-14T10:00:02.500Z",
                    "shell.execute_reply": "2026-07-14T10:00:02.500Z"
                }
            },
            "execution_count": 4,
            "outputs": [],
            "source": ["x = 1"]
        }"##;
        let nbformat_cell: nbformat::v4::Cell =
            serde_json::from_str(cell_json).expect("fixture cell should parse");

        let languages = Arc::new(LanguageRegistry::test(cx.executor()));
        let notebook_language = Task::ready(None).shared();

        let resnapshotted = cx.update(|window, cx| {
            let cell = Cell::load(&nbformat_cell, &languages, notebook_language, window, cx);
            let Cell::Code(code_cell) = &cell else {
                panic!("expected a code cell");
            };
            let code_cell = code_cell.read(cx);
            assert_eq!(
                code_cell.execution_status(),
                CellExecutionStatus::Finished,
                "a loaded cell with saved execution timestamps shows as run"
            );
            assert_eq!(
                code_cell.execution_duration(),
                Some(Duration::from_millis(2500)),
                "duration should be recovered from busy→idle"
            );
            cell.to_nbformat_cell(cx)
        });

        let nbformat::v4::Cell::Code { metadata, .. } = resnapshotted else {
            panic!("expected a code cell");
        };
        let execution = metadata
            .execution
            .expect("execution metadata should be written back on save");
        assert_eq!(
            execution.shell_execute_reply.as_deref(),
            Some("2026-07-14T10:00:02.500Z"),
            "completion timestamp should round-trip"
        );
        assert_eq!(
            execution.iopub_status_busy.as_deref(),
            Some("2026-07-14T10:00:00.000Z"),
            "derived start (completion − duration) should round-trip"
        );
    }

    /// An empty/whitespace `.ipynb` must open as a one-cell notebook (not a
    /// blank pane), and the generated template must round-trip back through the
    /// parser as valid nbformat.
    #[test]
    fn test_empty_notebook_template_round_trips() {
        for text in ["", "   \n\t", "\n"] {
            let notebook = NotebookEditor::parse_notebook_text(text)
                .expect("empty text should parse into the template");
            assert_eq!(
                notebook.cells.len(),
                1,
                "an empty notebook should be seeded with one cell"
            );
            assert!(
                matches!(notebook.cells[0], nbformat::v4::Cell::Code { .. }),
                "the seeded cell should be a code cell"
            );

            let serialized =
                serde_json::to_string_pretty(&notebook).expect("template should serialize");
            let reparsed = NotebookEditor::parse_notebook_text(&serialized)
                .expect("serialized template should re-parse");
            assert_eq!(reparsed.cells.len(), 1);
        }
    }
}
