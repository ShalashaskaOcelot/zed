#![allow(unused, dead_code)]
use std::collections::BTreeSet;
use std::future::Future;
use std::time::Duration;
use std::{path::PathBuf, sync::Arc};

use anyhow::{Context as _, Result};
use client::proto::ViewId;
use collections::HashMap;
use editor::{DisplayPoint, Editor};
use feature_flags::{FeatureFlagAppExt as _, NotebookFeatureFlag};
use futures::FutureExt;
use futures::future::Shared;
use gpui::{
    AnyElement, App, ClipboardItem, Entity, EventEmitter, FocusHandle, Focusable, KeyContext,
    ListScrollEvent, ListState, Point, PromptLevel, Task, TaskExt, actions, list, prelude::*,
};
use jupyter_protocol::JupyterKernelspec;
use language::{Buffer, Language, LanguageRegistry};
use log;
use project::{Project, ProjectEntryId, ProjectPath};
use settings::{NotebookRunLandingMode, Settings as _};
use ui::{CommonAnimationExt, Tooltip, prelude::*};
use workspace::item::{ItemEvent, SaveOptions, TabContentParams};
use workspace::notifications::NotificationId;
use workspace::searchable::SearchableItemHandle;
use workspace::{
    Item, ItemHandle, OpenOptions, OpenVisible, Pane, ProjectItem, ToolbarItemLocation, Workspace,
};

use super::{
    Cell, CellEvent, CellExecutionStatus, CellPosition, CellToolbarAction, MarkdownCellEvent,
    RenderableCell,
};

use nbformat::v4::CellId;
use nbformat::v4::Metadata as NotebookMetadata;
use serde_json;
use uuid::Uuid;

use crate::components::{KernelPickerDelegate, KernelSelector};
use crate::kernels::{
    Kernel, KernelSession, KernelSpecification, KernelStatus, LocalKernelSpecification,
    NativeRunningKernel, PythonEnvKernelSpecification, RemoteRunningKernel, SshRunningKernel,
    WslRunningKernel,
};
use crate::notebook::MovementDirection;
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
    EnterCommandMode, EnterEditMode, ExtendSelectionDown, ExtendSelectionUp, InterruptKernel,
    MoveCellDown, MoveCellUp, NewNotebook, NotebookMoveDown, NotebookMoveUp, OpenNotebook,
    PasteCell, PasteCellAbove, RedoCellOp, ReloadNotebook, RestartKernel, Run, RunAll, RunAndAdvance,
    RunCellAndBelow, RunCellsAbove, SelectFirstCell, SelectLastCell, UndoCellOp,
};

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

pub(crate) const MAX_TEXT_BLOCK_WIDTH: f32 = 9999.0;
pub(crate) const SMALL_SPACING_SIZE: f32 = 8.0;
pub(crate) const MEDIUM_SPACING_SIZE: f32 = 12.0;
pub(crate) const LARGE_SPACING_SIZE: f32 = 16.0;
pub(crate) const GUTTER_WIDTH: f32 = 30.0;
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
    }

    cx.observe_flag::<NotebookFeatureFlag, _>({
        move |flag, cx| {
            if *flag {
                workspace::register_project_item::<NotebookEditor>(cx);
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
    remote_id: Option<ViewId>,
    cell_list: ListState,
    notebook_mode: NotebookMode,
    selected_cell_index: usize,
    cell_order: Vec<CellId>,
    original_cell_order: Vec<CellId>,
    cell_map: HashMap<CellId, Cell>,
    kernel: Kernel,
    kernel_specification: Option<KernelSpecification>,
    execution_requests: HashMap<String, CellId>,
    pending_executions: Vec<CellId>,
    /// Cells the user tried to run while no kernel was selected. They are held
    /// (not spinning) while the kernel picker is open: promoted to
    /// `pending_executions` if a kernel is chosen, or cleared if it's dismissed.
    cells_awaiting_kernel_choice: Vec<CellId>,
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
    /// Multi-selection: every selected index INCLUDING the primary
    /// (`selected_cell_index`). Empty when only a single cell is selected.
    /// Index-based, so any structural change collapses the selection.
    selected_indices: BTreeSet<usize>,
    /// The fixed end of a shift-range selection (the cell selection started
    /// from). `None` means the anchor is the primary cell.
    selection_anchor: Option<usize>,
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
        let language_name = notebook_item.read(cx).language_name();
        let worktree_id = notebook_item.read(cx).project_path.worktree_id;

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
                            CellEvent::Stop(cell_id) => {
                                this.handle_cell_stop(cell_id, window, cx)
                            }
                            CellEvent::ModifiedClick { id, shift } => {
                                this.handle_modified_click(id, *shift, window, cx)
                            }
                            CellEvent::PlainClick { id } => {
                                this.handle_plain_click(id, window, cx)
                            }
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
                    cx.subscribe(&editor, move |this, _editor, event, cx| {
                        this.on_cell_editor_event(&cell_id_for_editor, event, cx);
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
                            CellEvent::PlainClick { id } => {
                                this.handle_plain_click(id, window, cx)
                            }
                            _ => {}
                        },
                    )
                    .detach();

                    let cell_id_for_editor = cell_id.clone();
                    let editor = markdown_cell.read(cx).editor().clone();
                    cx.subscribe(&editor, move |this, _editor, event, cx| {
                        this.on_cell_editor_event(&cell_id_for_editor, event, cx);
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
                            CellEvent::PlainClick { id } => {
                                this.handle_plain_click(id, window, cx)
                            }
                            _ => {}
                        },
                    )
                    .detach();
                }
            }

            cell_map.insert(cell_id.clone(), cell_entity);
        }

        let notebook_handle = cx.entity().downgrade();
        let cell_count = cell_order.len();

        let this = cx.entity();
        let cell_list = ListState::new(cell_count, gpui::ListAlignment::Top, px(1000.));

        let mut editor = Self {
            project,
            languages: languages.clone(),
            worktree_id,
            focus_handle,
            notebook_item: notebook_item.clone(),
            notebook_language,
            remote_id: None,
            cell_list,
            notebook_mode: NotebookMode::Command,
            selected_cell_index: 0,
            cell_order: cell_order.clone(),
            original_cell_order: cell_order.clone(),
            cell_map: cell_map.clone(),
            kernel: Kernel::Shutdown,
            kernel_specification: None,
            execution_requests: HashMap::default(),
            pending_executions: Vec::new(),
            cells_awaiting_kernel_choice: Vec::new(),
            run_queue: Vec::new(),
            active_run_cell: None,
            undo_stack: Vec::new(),
            redo_stack: Vec::new(),
            kernel_picker_handle: PopoverMenuHandle::default(),
            disk_changed_externally: false,
            execution_state_changed: false,
            // What we loaded IS what is on disk right now.
            last_saved_disk_text: Some(notebook_item.read(cx).buffer.read(cx).text()),
            resume_run_queue_on_idle: false,
            selected_indices: BTreeSet::new(),
            selection_anchor: None,
        };
        // Lazy start: don't launch a kernel on open. Show the remembered
        // kernel's name if we can resolve one now (a real launch happens on
        // first run or explicit selection); otherwise the status bar shows
        // "Select Kernel" until the user picks or runs a cell.
        editor.kernel_specification = editor.remembered_kernel_spec(cx);
        editor.refresh_language(cx);
        editor.refresh_kernelspecs(cx);

        cx.subscribe(&notebook_item, |this, _item, _event, cx| {
            this.refresh_language(cx);
        })
        .detach();

        // Reload the notebook when its .ipynb changes on disk (the project
        // auto-reloads the backing buffer and emits `Reloaded`).
        let buffer = notebook_item.read(cx).buffer.clone();
        cx.subscribe_in(&buffer, window, |this, buffer, event, window, cx| {
            if let language::BufferEvent::Reloaded = event {
                this.handle_external_change(buffer, window, cx);
            }
        })
        .detach();

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
        cx.notify();
    }

    /// The kernel the notebook should use without any explicit choice yet:
    /// an active in-session selection, a selection persisted for this worktree,
    /// or one matching the notebook's saved metadata. Deliberately does NOT
    /// fall back to the "recommended"/global kernel — an unremembered notebook
    /// should prompt rather than silently start the wrong interpreter.
    fn remembered_kernel_spec(&self, cx: &App) -> Option<KernelSpecification> {
        if let Some(spec) = &self.kernel_specification {
            return Some(spec.clone());
        }
        let store = ReplStore::global(cx);
        let store = store.read(cx);
        if let Some(spec) = store.selected_kernel(self.worktree_id) {
            return Some(spec.clone());
        }
        let kernelspec = self
            .notebook_item
            .read(cx)
            .notebook
            .metadata
            .kernelspec
            .as_ref()?;
        let name = kernelspec.name.clone();
        store
            .kernel_specifications_for_worktree(self.worktree_id)
            .find(|spec| spec.name().as_ref() == name)
            .cloned()
    }

    /// Launch the remembered kernel, or prompt for one if none is remembered.
    fn launch_kernel(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if let Some(spec) = self.remembered_kernel_spec(cx) {
            self.launch_kernel_with_spec(spec, window, cx);
        } else {
            // Nothing selected or remembered: prompt the user to choose a
            // kernel. Any cell that triggered this is already queued and will
            // run once a kernel is picked and ready.
            self.kernel_picker_handle.show(window, cx);
        }
    }

    /// Create a `.venv` in the worktree root, install ipykernel into it, and
    /// select it — the "Create Python Environment" flow from the kernel picker.
    fn create_python_environment(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        self.kernel_picker_handle.hide(cx);

        let Some(worktree_root) = self
            .project
            .read(cx)
            .worktree_for_id(self.worktree_id, cx)
            .map(|worktree| worktree.read(cx).abs_path().to_path_buf())
        else {
            Self::show_env_toast(
                window,
                cx,
                "Cannot create a Python environment: no project folder is open.".to_string(),
                false,
            );
            return;
        };

        let fs = self.project.read(cx).fs().clone();
        let venv_dir = worktree_root.join(".venv");
        let venv_python = if cfg!(windows) {
            venv_dir.join("Scripts").join("python.exe")
        } else {
            venv_dir.join("bin").join("python")
        };

        struct CreatePythonEnv;
        let notification_id = NotificationId::unique::<CreatePythonEnv>();
        let workspace = Workspace::for_window(window, cx);
        if let Some(workspace) = &workspace {
            workspace.update(cx, |workspace, cx| {
                workspace.show_toast(
                    workspace::Toast::new(
                        notification_id.clone(),
                        "Creating .venv and installing ipykernel…".to_string(),
                    ),
                    cx,
                );
            });
        }
        let weak_workspace = workspace.map(|workspace| workspace.downgrade());

        let create_task = cx.background_spawn(async move {
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
                    "could not create .venv (is Python installed and on PATH?): {last_error}"
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

            anyhow::Ok(venv_python)
        });

        cx.spawn_in(window, async move |this, cx| {
            let result = create_task.await;
            match result {
                Ok(venv_python) => {
                    if let Some(weak_workspace) = &weak_workspace {
                        weak_workspace
                            .update(cx, |workspace, cx| {
                                workspace.dismiss_toast(&notification_id, cx);
                                workspace.show_toast(
                                    workspace::Toast::new(
                                        notification_id.clone(),
                                        "Created .venv and installed ipykernel".to_string(),
                                    )
                                    .autohide(),
                                    cx,
                                );
                            })
                            .ok();
                    }
                    this.update_in(cx, |this, window, cx| {
                        let spec = KernelSpecification::PythonEnv(
                            PythonEnvKernelSpecification::from_python_path(
                                venv_python,
                                ".venv".to_string(),
                                true,
                                Some("venv".to_string()),
                            ),
                        );
                        this.change_kernel(spec, window, cx);
                        this.refresh_kernelspecs(cx);
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
                                        format!("Failed to create Python environment: {error}"),
                                    ),
                                    cx,
                                );
                            })
                            .ok();
                    }
                }
            }
        })
        .detach();
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

    /// Create a new `Untitled-N.ipynb` in the first visible worktree, seeded
    /// with the one-cell template, and open it as a notebook. Requires a folder
    /// to be open (there is nowhere to put the file otherwise).
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
        let fs = project.read(cx).fs().clone();

        let Some(worktree_root) = project
            .read(cx)
            .visible_worktrees(cx)
            .next()
            .map(|worktree| worktree.read(cx).abs_path().to_path_buf())
        else {
            struct NewNotebookToast;
            workspace.show_toast(
                workspace::Toast::new(
                    NotificationId::unique::<NewNotebookToast>(),
                    "Open a folder to create a new notebook.".to_string(),
                ),
                cx,
            );
            return;
        };

        let template = match Self::empty_notebook()
            .and_then(|notebook| Ok(serde_json::to_string_pretty(&notebook)?))
        {
            Ok(json) => json,
            Err(error) => {
                log::error!("notebook: failed to build the new-notebook template: {error}");
                return;
            }
        };

        cx.spawn_in(window, async move |workspace, cx| {
            // Pick a unique Untitled name so repeated invocations don't collide.
            let mut candidate = worktree_root.join("Untitled.ipynb");
            let mut index = 1;
            while fs.is_file(&candidate).await {
                candidate = worktree_root.join(format!("Untitled-{index}.ipynb"));
                index += 1;
            }

            fs.atomic_write(candidate.clone(), template).await?;

            workspace
                .update_in(cx, |workspace, window, cx| {
                    workspace.open_abs_path(
                        candidate,
                        OpenOptions {
                            visible: Some(OpenVisible::None),
                            ..Default::default()
                        },
                        window,
                        cx,
                    )
                })?
                .await?;
            anyhow::Ok(())
        })
        .detach_and_log_err(cx);
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
            .map(|worktree| worktree.read(cx).abs_path().to_path_buf())
            .unwrap_or_else(std::env::temp_dir);
        let fs = self.project.read(cx).fs().clone();
        let view = cx.entity();

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
        if let Kernel::RunningKernel(kernel) = &mut self.kernel {
            kernel.force_shutdown(window, cx).detach();
        }

        self.execution_requests.clear();
        // If this is a deliberate kernel switch (nothing was waiting on a
        // kernel choice), abort any in-progress batch. If instead the user is
        // picking a kernel to satisfy a batch that was waiting for one, keep
        // the queue so it runs on the new kernel.
        if self.cells_awaiting_kernel_choice.is_empty() {
            self.cancel_run_queue(cx);
        }
        self.stop_executing_cells(cx);

        // Persist the choice for this worktree so reopening the notebook (or
        // opening a sibling notebook) uses it instead of the global default.
        ReplStore::global(cx).update(cx, |store, cx| {
            store.set_active_kernelspec(self.worktree_id, spec.clone(), cx);
        });

        // Any cell the user ran before picking a kernel should now run once
        // this kernel is ready.
        self.promote_awaiting_cells();

        self.launch_kernel_with_spec(spec, window, cx);
    }

    fn restart_kernel(&mut self, _: &RestartKernel, window: &mut Window, cx: &mut Context<Self>) {
        let Some(spec) = self.kernel_specification.clone() else {
            return;
        };

        let kernel = std::mem::replace(&mut self.kernel, Kernel::Restarting);
        self.execution_requests.clear();
        self.cancel_run_queue(cx);
        self.stop_executing_cells(cx);
        // The restarted kernel's execution counter starts over at 1, so the
        // cells' `In [N]` numbers from the old session are stale — clear them
        // so a fresh run-through is visually distinct.
        self.execution_state_changed = true;
        for cell in self.cell_map.values() {
            if let Cell::Code(code_cell) = cell {
                code_cell.update(cx, |code_cell, cx| {
                    code_cell.reset_execution_count();
                    cx.notify();
                });
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
            Kernel::Shutdown | Kernel::ErroredLaunch(_) => {
                if has_remembered_kernel {
                    Disposition::Queued { launch: true }
                } else {
                    Disposition::Prompt
                }
            }
            Kernel::ShuttingDown => Disposition::Failed("the kernel is shutting down".to_string()),
        };

        if let Disposition::Prompt = disposition {
            // Hold the cell (no spinner) and open the picker. It will run if a
            // kernel is chosen (see change_kernel), or be cleared on dismiss.
            if !self.cells_awaiting_kernel_choice.contains(&cell_id) {
                self.cells_awaiting_kernel_choice.push(cell_id);
            }
            self.launch_kernel(window, cx);
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
    /// without selecting). They return to their idle state, and any batch that
    /// was waiting on the kernel choice is aborted.
    fn clear_awaiting_cells(&mut self, cx: &mut Context<Self>) {
        if !self.cells_awaiting_kernel_choice.is_empty() {
            log::debug!(
                "notebook: kernel picker dismissed; dropping {} awaiting cell(s)",
                self.cells_awaiting_kernel_choice.len(),
            );
            // Cells held for a kernel choice were marked Pending; dismissing the
            // picker means they won't run, so clear that status too (otherwise
            // the triggering cell stays stuck showing "Pending").
            for cell_id in std::mem::take(&mut self.cells_awaiting_kernel_choice) {
                if let Some(Cell::Code(cell)) = self.cell_map.get(&cell_id) {
                    cell.update(cx, |cell, cx| {
                        cell.cancel_execution();
                        cx.notify();
                    });
                }
            }
            self.cancel_run_queue(cx);
            cx.notify();
        }
    }

    fn get_selected_cell(&self) -> Option<&Cell> {
        self.cell_order
            .get(self.selected_cell_index)
            .and_then(|cell_id| self.cell_map.get(cell_id))
    }

    fn has_outputs(&self, window: &mut Window, cx: &mut Context<Self>) -> bool {
        self.cell_map.values().any(|cell| {
            if let Cell::Code(code_cell) = cell {
                code_cell.read(cx).has_outputs()
            } else {
                false
            }
        })
    }

    fn clear_outputs(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        self.execution_state_changed = true;
        for cell in self.cell_map.values() {
            if let Cell::Code(code_cell) = cell {
                code_cell.update(cx, |cell, cx| {
                    cell.clear_outputs();
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
                self.execute_cell(cell_id, window, cx);
                return;
            }
            // Skip markdown/raw cells and continue to the next.
        }
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
                    cell.clear_outputs();
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

    // Discussion can be done on this default implementation
    /// Moves focus to the next cell editor (used when already in edit mode).
    fn move_to_next_cell(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if !self.cell_order.is_empty() && self.selected_cell_index < self.cell_order.len() - 1 {
            self.selected_cell_index += 1;
            // focus the new cell's editor
            if let Some(cell_id) = self.cell_order.get(self.selected_cell_index) {
                if let Some(cell) = self.cell_map.get(cell_id) {
                    match cell {
                        Cell::Code(code_cell) => {
                            let editor = code_cell.read(cx).editor();
                            window.focus(&editor.focus_handle(cx), cx);
                        }
                        Cell::Markdown(markdown_cell) => {
                            // Don't auto-enter edit mode for next markdown cell
                            // Just select it
                        }
                        Cell::Raw(_) => {}
                    }
                }
            }
            cx.notify();
        } else {
            // in the end, could optionally create a new cell
            // For now, just stay on the current cell
        }
    }

    fn open_notebook(&mut self, _: &OpenNotebook, _window: &mut Window, _cx: &mut Context<Self>) {
        println!("Open notebook triggered");
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
    fn insert_cell(&mut self, index: usize, cell_id: CellId, cell: Cell) {
        let index = index.min(self.cell_order.len());
        self.cell_order.insert(index, cell_id.clone());
        self.cell_map.insert(cell_id, cell);
        self.selected_cell_index = index;
        // Indices shifted — a stale multi-selection would select wrong cells.
        self.collapse_selection();
        self.cell_list.splice(index..index, 1);
        self.cell_list.scroll_to_reveal_item(index);
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
        cx.subscribe(&editor, move |this, _editor, event, cx| {
            this.on_cell_editor_event(&cell_id_for_editor, event, cx);
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
        cx.subscribe(&editor, move |this, _editor, event, cx| {
            this.on_cell_editor_event(&cell_id_for_editor, event, cx);
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

    fn focus_cell_editor_in_edit_mode(
        &mut self,
        editor: Entity<Editor>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        window.focus(&editor.focus_handle(cx), cx);
        self.notebook_mode = NotebookMode::Edit;
        cx.notify();
    }

    fn add_markdown_block(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let (cell_id, markdown_cell) = self.build_markdown_cell(String::new(), window, cx);
        let index = self.index_below_selection();
        self.insert_cell(index, cell_id.clone(), Cell::Markdown(markdown_cell));
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
        self.insert_cell(index, cell_id.clone(), Cell::Code(code_cell));
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
            self.insert_cell(0, cell_id.clone(), Cell::Code(code_cell));
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

    fn cut_cell(&mut self, _: &CutCell, window: &mut Window, cx: &mut Context<Self>) {
        self.copy_cell(&CopyCell, window, cx);
        self.delete_cell(&DeleteCell, window, cx);
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

    fn paste_cell_above(&mut self, _: &PasteCellAbove, window: &mut Window, cx: &mut Context<Self>) {
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
        self.insert_cell(index, cell_id, cell_entity);
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
            cx.notify();
        }
    }

    /// A plain (unmodified) click on a cell outside its editor — the gutter,
    /// margins, or output area (see `CellEvent::PlainClick`). Selects just this
    /// cell and enters command mode. A click that lands on the editor still
    /// focuses it and enters edit mode via the editor's own focus event, which
    /// fires after this (the classifier does not stop propagation for plain
    /// clicks), so this only "wins" for clicks that miss the editor.
    fn handle_plain_click(&mut self, cell_id: &CellId, window: &mut Window, cx: &mut Context<Self>) {
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
            self.set_selected_index(ix, true, window, cx);

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
            self.set_selected_index(ix, true, window, cx);

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
            self.set_selected_index(count - 1, true, window, cx);
            cx.notify();
        }
    }

    /// Shared handling for a cell editor's events: track focus for selection,
    /// and follow the cursor while editing.
    fn on_cell_editor_event(
        &mut self,
        cell_id: &CellId,
        event: &editor::EditorEvent,
        cx: &mut Context<Self>,
    ) {
        match event {
            editor::EditorEvent::Focused => self.select_cell_by_id(cell_id, cx),
            editor::EditorEvent::SelectionsChanged { .. } => {
                if self.notebook_mode == NotebookMode::Edit
                    && let Some(index) = self.cell_order.iter().position(|id| id == cell_id)
                    && index == self.selected_cell_index
                {
                    self.follow_cursor_in_cell(index, cx);
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
    /// only when the cursor would fall outside the viewport (it does not keep
    /// the cursor centered). Cell editors are `SizeByContent` and have no
    /// internal scroll, so the outer list must follow the cursor.
    fn follow_cursor_in_cell(&mut self, index: usize, cx: &mut Context<Self>) {
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

        let (cursor_row, total_rows) = editor.update(cx, |editor, cx| {
            let snapshot = editor.display_snapshot(cx);
            let cursor_row = snapshot
                .max_point()
                .row()
                .min(editor.selections.newest_display(&snapshot).head().row());
            (cursor_row.0, snapshot.max_point().row().0)
        });

        let Some(cell_bounds) = self.cell_list.bounds_for_item(index) else {
            // Not currently laid out (e.g. scrolled far away): reveal it.
            self.cell_list.scroll_to_reveal_item_top_aligned(index);
            return;
        };
        let viewport = self.cell_list.viewport_bounds();
        if viewport.size.height <= px(0.) {
            return;
        }

        // Estimate the cursor's vertical position by its fractional row within
        // the cell's laid-out height. Approximate (the cell includes non-editor
        // chrome) but only used to decide when to nudge the viewport.
        let rows = (total_rows + 1).max(1) as f32;
        let fraction = (cursor_row as f32 + 0.5) / rows;
        let cursor_y = cell_bounds.top() + cell_bounds.size.height * fraction;

        let margin = px(24.);
        if cursor_y < viewport.top() + margin {
            self.cell_list
                .scroll_by(cursor_y - (viewport.top() + margin));
        } else if cursor_y > viewport.bottom() - margin {
            self.cell_list
                .scroll_by(cursor_y - (viewport.bottom() - margin));
        }
    }

    fn button_group(window: &mut Window, cx: &mut Context<Self>) -> Div {
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
        let has_outputs = self.has_outputs(window, cx);

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
                                .tooltip(move |window, cx| {
                                    Tooltip::for_action("Execute all cells", &RunAll, cx)
                                })
                                .on_click(|_, window, cx| {
                                    window.dispatch_action(Box::new(RunAll), cx);
                                }),
                            )
                            .child(
                                Self::render_notebook_control(
                                    "clear-all-outputs",
                                    IconName::ListX,
                                    window,
                                    cx,
                                )
                                .disabled(!has_outputs)
                                .tooltip(move |window, cx| {
                                    Tooltip::for_action("Clear all outputs", &ClearOutputs, cx)
                                })
                                .on_click(|_, window, cx| {
                                    window.dispatch_action(Box::new(ClearOutputs), cx);
                                }),
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
                                .tooltip(move |window, cx| {
                                    Tooltip::for_action("Move cell up", &MoveCellUp, cx)
                                })
                                .on_click(|_, window, cx| {
                                    window.dispatch_action(Box::new(MoveCellUp), cx);
                                }),
                            )
                            .child(
                                Self::render_notebook_control(
                                    "move-cell-down",
                                    IconName::ArrowDown,
                                    window,
                                    cx,
                                )
                                .tooltip(move |window, cx| {
                                    Tooltip::for_action("Move cell down", &MoveCellDown, cx)
                                })
                                .on_click(|_, window, cx| {
                                    window.dispatch_action(Box::new(MoveCellDown), cx);
                                }),
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
                                .tooltip(move |window, cx| {
                                    Tooltip::for_action("Add markdown block", &AddMarkdownBlock, cx)
                                })
                                .on_click(|_, window, cx| {
                                    window.dispatch_action(Box::new(AddMarkdownBlock), cx);
                                }),
                            )
                            .child(
                                Self::render_notebook_control(
                                    "new-code-cell",
                                    IconName::Code,
                                    window,
                                    cx,
                                )
                                .tooltip(move |window, cx| {
                                    Tooltip::for_action("Add code block", &AddCodeBlock, cx)
                                })
                                .on_click(|_, window, cx| {
                                    window.dispatch_action(Box::new(AddCodeBlock), cx);
                                }),
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
                    .child(Self::button_group(window, cx).child({
                        let kernel_status = self.kernel.status();
                        let (icon, icon_color) = match &kernel_status {
                            KernelStatus::Idle => (IconName::ReplNeutral, Color::Success),
                            KernelStatus::Busy => (IconName::ReplNeutral, Color::Warning),
                            KernelStatus::Starting => (IconName::ReplNeutral, Color::Muted),
                            KernelStatus::Error => (IconName::ReplNeutral, Color::Error),
                            KernelStatus::ShuttingDown => (IconName::ReplNeutral, Color::Muted),
                            KernelStatus::Shutdown => (IconName::ReplNeutral, Color::Disabled),
                            KernelStatus::Restarting => (IconName::ReplNeutral, Color::Warning),
                        };
                        let kernel_name = self
                            .kernel_specification
                            .as_ref()
                            .map(|spec| spec.name().to_string())
                            .unwrap_or_else(|| "Select Kernel".to_string());
                        IconButton::new("repl", icon)
                            .icon_color(icon_color)
                            .tooltip(move |window, cx| {
                                Tooltip::text(format!(
                                    "{} ({}). Click to change kernel.",
                                    kernel_name,
                                    kernel_status.to_string()
                                ))(window, cx)
                            })
                            .on_click(cx.listener(|this, _, window, cx| {
                                this.kernel_picker_handle.toggle(window, cx);
                            }))
                    })),
            )
    }

    fn render_kernel_status_bar(
        &self,
        _window: &mut Window,
        cx: &mut Context<Self>,
    ) -> impl IntoElement {
        let kernel_status = self.kernel.status();
        let kernel_name = self
            .kernel_specification
            .as_ref()
            .map(|spec| spec.name().to_string())
            .unwrap_or_else(|| "Select Kernel".to_string());

        let (status_icon, status_color) = match &kernel_status {
            KernelStatus::Idle => (IconName::Circle, Color::Success),
            KernelStatus::Busy => (IconName::ArrowCircle, Color::Warning),
            KernelStatus::Starting => (IconName::ArrowCircle, Color::Muted),
            KernelStatus::Error => (IconName::XCircle, Color::Error),
            KernelStatus::ShuttingDown => (IconName::ArrowCircle, Color::Muted),
            KernelStatus::Shutdown => (IconName::Circle, Color::Muted),
            KernelStatus::Restarting => (IconName::ArrowCircle, Color::Warning),
        };

        let is_spinning = matches!(
            kernel_status,
            KernelStatus::Busy
                | KernelStatus::Starting
                | KernelStatus::ShuttingDown
                | KernelStatus::Restarting
        );

        let status_icon_element = if is_spinning {
            Icon::new(status_icon)
                .size(IconSize::Small)
                .color(status_color)
                .with_rotate_animation(2)
                .into_any_element()
        } else {
            Icon::new(status_icon)
                .size(IconSize::Small)
                .color(status_color)
                .into_any_element()
        };

        let worktree_id = self.worktree_id;
        let kernel_picker_handle = self.kernel_picker_handle.clone();
        let view = cx.entity().downgrade();
        let view_for_dismiss = view.clone();
        let view_for_create = view.clone();

        h_flex()
            .w_full()
            .px_3()
            .py_1()
            .gap_2()
            .items_center()
            .justify_between()
            .bg(cx.theme().colors().status_bar_background)
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
                    Button::new("kernel-selector", kernel_name.clone())
                        .label_size(LabelSize::Small)
                        .start_icon(
                            Icon::new(status_icon)
                                .size(IconSize::Small)
                                .color(status_color),
                        ),
                    Tooltip::text(format!(
                        "Kernel: {} ({}). Click to change.",
                        kernel_name,
                        kernel_status.to_string()
                    )),
                )
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
                .with_handle(kernel_picker_handle),
            )
            .child(
                h_flex()
                    .gap_1()
                    .child(
                        IconButton::new("restart-kernel", IconName::RotateCw)
                            .icon_size(IconSize::Small)
                            .tooltip(|window, cx| {
                                Tooltip::for_action("Restart Kernel", &RestartKernel, cx)
                            })
                            .on_click(cx.listener(|this, _, window, cx| {
                                this.restart_kernel(&RestartKernel, window, cx);
                            })),
                    )
                    .child(
                        IconButton::new("interrupt-kernel", IconName::Stop)
                            .icon_size(IconSize::Small)
                            .disabled(!kernel_status.is_connected())
                            .tooltip(|window, cx| {
                                Tooltip::for_action("Interrupt Kernel", &InterruptKernel, cx)
                            })
                            .on_click(cx.listener(|this, _, window, cx| {
                                this.interrupt_kernel(&InterruptKernel, window, cx);
                            })),
                    ),
            )
    }

    fn cell_list(&self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let view = cx.entity();
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
        window: &mut Window,
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
            .child(
                // `.flex_1()` (not `.h_full()`) sizes this row to the height
                // left after the status bar; `.h_full()` here would take the
                // whole notebook height and let the status bar overlap the
                // row's bottom, hiding the last cell (see bug #25).
                h_flex()
                    .flex_1()
                    .w_full()
                    .min_h_0()
                    .gap_2()
                    .child(div().flex_1().h_full().child(self.cell_list(window, cx)))
                    .child(self.render_notebook_controls(window, cx)),
            )
            .child(self.render_kernel_status_bar(window, cx))
    }
}

impl Focusable for NotebookEditor {
    fn focus_handle(&self, _: &App) -> FocusHandle {
        self.focus_handle.clone()
    }
}

// Intended to be a NotebookBuffer
pub struct NotebookItem {
    path: PathBuf,
    project_path: ProjectPath,
    languages: Arc<LanguageRegistry>,
    // Raw notebook data
    notebook: nbformat::v4::Notebook,
    // Store our version of the notebook in memory (cell_order, cell_map)
    id: ProjectEntryId,
    // The underlying project buffer for the .ipynb file. Retained so the
    // project keeps watching the file and emits `Reloaded` on external change.
    buffer: Entity<Buffer>,
}

impl project::ProjectItem for NotebookItem {
    fn try_open(
        project: &Entity<Project>,
        path: &ProjectPath,
        cx: &mut App,
    ) -> Option<Task<anyhow::Result<Entity<Self>>>> {
        let path = path.clone();
        let project = project.clone();
        let fs = project.read(cx).fs().clone();
        let languages = project.read(cx).languages().clone();

        if path.path.extension().unwrap_or_default() == "ipynb" {
            Some(cx.spawn(async move |cx| {
                let abs_path = project
                    .read_with(cx, |project, cx| project.absolute_path(&path, cx))
                    .with_context(|| format!("finding the absolute path of {path:?}"))?;

                let buffer = project
                    .update(cx, |project, cx| project.open_buffer(path.clone(), cx))
                    .await?;
                let file_content = buffer.read_with(cx, |buffer, _| buffer.text());

                let notebook = NotebookEditor::parse_notebook_text(&file_content)?;

                let id = project
                    .update(cx, |project, cx| {
                        project.entry_for_path(&path, cx).map(|entry| entry.id)
                    })
                    .context("Entry not found")?;

                Ok(cx.new(|_| NotebookItem {
                    path: abs_path,
                    project_path: path,
                    languages,
                    notebook,
                    id,
                    buffer,
                }))
            }))
        } else {
            None
        }
    }

    fn entry_id(&self, _: &App) -> Option<ProjectEntryId> {
        Some(self.id)
    }

    fn project_path(&self, _: &App) -> Option<ProjectPath> {
        Some(self.project_path.clone())
    }

    fn is_dirty(&self) -> bool {
        // TODO: Track if notebook metadata or structure has changed
        false
    }
}

impl NotebookItem {
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

// pub struct NotebookControls {
//     pane_focused: bool,
//     active_item: Option<Box<dyn ItemHandle>>,
//     // subscription: Option<Subscription>,
// }

// impl NotebookControls {
//     pub fn new() -> Self {
//         Self {
//             pane_focused: false,
//             active_item: Default::default(),
//             // subscription: Default::default(),
//         }
//     }
// }

// impl EventEmitter<ToolbarItemEvent> for NotebookControls {}

// impl Render for NotebookControls {
//     fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
//         div().child("notebook controls")
//     }
// }

// impl ToolbarItemView for NotebookControls {
//     fn set_active_pane_item(
//         &mut self,
//         active_pane_item: Option<&dyn workspace::ItemHandle>,
//         window: &mut Window, cx: &mut Context<Self>,
//     ) -> workspace::ToolbarItemLocation {
//         cx.notify();
//         self.active_item = None;

//         let Some(item) = active_pane_item else {
//             return ToolbarItemLocation::Hidden;
//         };

//         ToolbarItemLocation::PrimaryLeft
//     }

//     fn pane_focus_update(&mut self, pane_focused: bool, _window: &mut Window, _cx: &mut Context<Self>) {
//         self.pane_focused = pane_focused;
//     }
// }

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
        self.notebook_item
            .read(cx)
            .project_path
            .path
            .file_name()
            .map(|s| s.to_string())
            .unwrap_or_default()
            .into()
    }

    fn tab_content(&self, params: TabContentParams, window: &Window, cx: &App) -> AnyElement {
        Label::new(self.tab_content_text(params.detail.unwrap_or(0), cx))
            .single_line()
            .color(params.text_color())
            .when(params.preview, |this| this.italic())
            .into_any_element()
    }

    fn tab_icon(&self, _window: &Window, _cx: &App) -> Option<Icon> {
        Some(IconName::Book.into())
    }

    fn show_toolbar(&self) -> bool {
        false
    }

    // TODO
    fn pixel_position_of_cursor(&self, _: &App) -> Option<Point<Pixels>> {
        None
    }

    // TODO
    fn as_searchable(&self, _: &Entity<Self>, _: &App) -> Option<Box<dyn SearchableItemHandle>> {
        None
    }

    fn set_nav_history(
        &mut self,
        _: workspace::ItemNavHistory,
        _window: &mut Window,
        _: &mut Context<Self>,
    ) {
        // TODO
    }

    fn can_save(&self, _cx: &App) -> bool {
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
        let path = self.notebook_item.read(cx).path.clone();
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
        _window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Task<Result<()>> {
        let notebook = self.to_notebook(cx);
        let fs = project.read(cx).fs().clone();

        let abs_path = project.read(cx).absolute_path(&path, cx);

        self.mark_as_saved(cx);

        cx.spawn(async move |this, cx| {
            let abs_path = abs_path.context("Failed to get absolute path")?;
            let json =
                serde_json::to_string_pretty(&notebook).context("Failed to serialize notebook")?;
            this.update(cx, |this, _| {
                this.last_saved_disk_text = Some(json.clone());
            })?;
            fs.atomic_write(abs_path, json).await?;
            Ok(())
        })
    }

    fn reload(
        &mut self,
        _project: Entity<Project>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Task<Result<()>> {
        let project_path = self.notebook_item.read(cx).project_path.clone();

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
        self.execution_state_changed
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
        cx.notify();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gpui::TestAppContext;
    use project::{FakeFs, Project, ProjectItem as _};
    use serde_json::json;
    use settings::SettingsStore;
    use util::path;
    use util::rel_path::rel_path;

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

    /// When the configured interpreter doesn't exist (e.g. Python isn't installed),
    /// running a cell must not leave it stuck in the executing state. It should
    /// instead surface the kernel launch error as an error output on the cell.
    #[gpui::test]
    async fn test_run_cell_with_missing_interpreter_shows_error(cx: &mut TestAppContext) {
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
                store.set_active_kernelspec(worktree_id, broken_spec, cx);
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

        // Run the (only) cell via the production action handler. This launches
        // the remembered (broken) kernel and queues the execution.
        editor.update_in(cx, |editor, window, cx| {
            editor.run_current_cell(&Run, window, cx);
        });

        // Wait for the launch task, which fails because the interpreter cannot
        // be spawned.
        let pending_kernel = editor.read_with(cx, |editor, _| match &editor.kernel {
            Kernel::StartingKernel(task) => task.clone(),
            _ => panic!("running a cell should launch the remembered kernel"),
        });
        pending_kernel.await;

        editor.read_with(cx, |editor, cx| {
            let cell_id = editor.cell_order.first().expect("notebook has one cell");
            let Some(Cell::Code(cell)) = editor.cell_map.get(cell_id) else {
                panic!("expected a code cell");
            };
            let cell = cell.read(cx);

            assert!(
                !cell.is_executing(),
                "cell must not be stuck in the executing state when the kernel is not running"
            );

            let nbformat::v4::Cell::Code { outputs, .. } = cell.to_nbformat_cell(cx) else {
                panic!("expected a code cell");
            };
            match outputs.as_slice() {
                [nbformat::v4::Output::Error(error)] => {
                    assert_eq!(error.ename, "Kernel Error");
                    let traceback = error.traceback.join("\n");
                    assert!(
                        traceback.contains("the kernel failed to launch"),
                        "error output should explain why the cell could not run, got: {traceback}"
                    );
                }
                other => panic!("expected a single error output, got: {other:?}"),
            }
        });
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
