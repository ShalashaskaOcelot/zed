use std::future::Future;
use std::{path::PathBuf, sync::Arc};

use anyhow::{Context as _, Result};
use client::proto::ViewId;
use collections::HashMap;
use feature_flags::{FeatureFlagAppExt as _, NotebookFeatureFlag};
use futures::FutureExt;
use gpui::{
    AnyElement, App, Entity, EventEmitter, FocusHandle, Focusable, ListState,
    Point, RetainAllImageCache, Task, TextStyleRefinement, actions, list, prelude::*,
};
use language::{Language, LanguageRegistry};
use project::{Project, ProjectEntryId, ProjectPath};
use settings::Settings as _;
use ui::{Tooltip, prelude::*};
use util::ResultExt;
use workspace::item::{SaveOptions, TabContentParams};
use workspace::searchable::SearchableItemHandle;
use workspace::{Item, Pane, ProjectItem};

use crate::{Kernel, KernelSpecification};
use crate::outputs::Output;
use runtimelib::{ExecuteRequest, JupyterMessage, JupyterMessageContent};

use super::{Cell, CellPosition, RenderableCell, RunnableCell};

use nbformat::v4::CellId;

/// Setup keyboard shortcut handlers on each cell editor
/// These handlers intercept keys BEFORE the editor's default handlers
pub fn setup_cell_editor_actions(
    editor: &mut editor::Editor,
    cell_id: CellId,
    notebook_handle: gpui::WeakEntity<NotebookEditor>,
) {
    // Register handler for Ctrl+Enter
    editor.register_action({
        let notebook_handle = notebook_handle.clone();
        let cell_id = cell_id.clone();
        move |_: &RunSelectedCell, window, cx| {
            log::info!("Editor action: RunSelectedCell triggered");
            if let Some(notebook) = notebook_handle.upgrade() {
                notebook.update(cx, |notebook, cx| {
                    // Find this cell and run it
                    if let Some(Cell::Code(code_cell)) = notebook.cell_map.get(&cell_id) {
                        notebook.execute_cell(cell_id.clone(), code_cell.clone(), window, cx);
                    }
                });
            }
        }
    }).detach();

    // Register handler for Shift+Enter
    editor.register_action({
        let notebook_handle = notebook_handle.clone();
        let cell_id = cell_id.clone();
        move |_: &RunSelectedCellAndMoveNext, window, cx| {
            log::info!("Editor action: RunSelectedCellAndMoveNext triggered");
            if let Some(notebook) = notebook_handle.upgrade() {
                notebook.update(cx, |notebook, cx| {
                    // Find this cell and run it
                    if let Some(Cell::Code(code_cell)) = notebook.cell_map.get(&cell_id) {
                        notebook.execute_cell(cell_id.clone(), code_cell.clone(), window, cx);
                        // Move to next cell and scroll
                        let cell_index = notebook.cell_order.iter().position(|id| id == &cell_id);
                        if let Some(index) = cell_index {
                            let next_index = (index + 1).min(notebook.cell_count() - 1);
                            if next_index != index {
                                notebook.selected_cell_index = next_index;
                                notebook.cell_list.scroll_to_reveal_item(next_index);
                                window.focus(&notebook.focus_handle);
                                cx.notify();
                            }
                        }
                    }
                });
            }
        }
    }).detach();
}

actions!(
    notebook,
    [
        /// Opens a Jupyter notebook file.
        OpenNotebook,
        /// Runs all cells in the notebook.
        RunAll,
        /// Runs the currently selected cell.
        RunSelectedCell,
        /// Runs the currently selected cell and moves to the next cell.
        RunSelectedCellAndMoveNext,
        /// Enters edit mode for the selected cell.
        EnterEditMode,
        /// Clears all cell outputs.
        ClearOutputs,
        /// Moves the current cell up.
        MoveCellUp,
        /// Moves the current cell down.
        MoveCellDown,
        /// Adds a new markdown cell.
        AddMarkdownBlock,
        /// Adds a new code cell.
        AddCodeBlock,
        /// Blurs all editors to exit edit mode.
        BlurAllEditors,
    ]
);

pub(crate) const MAX_TEXT_BLOCK_WIDTH: f32 = 9999.0;
pub(crate) const SMALL_SPACING_SIZE: f32 = 8.0;
pub(crate) const MEDIUM_SPACING_SIZE: f32 = 12.0;
pub(crate) const LARGE_SPACING_SIZE: f32 = 16.0;
pub(crate) const GUTTER_WIDTH: f32 = 19.0;
pub(crate) const CODE_BLOCK_INSET: f32 = MEDIUM_SPACING_SIZE;
pub(crate) const CONTROL_SIZE: f32 = 20.0;

pub fn init(cx: &mut App) {
    if cx.has_flag::<NotebookFeatureFlag>() || std::env::var("LOCAL_NOTEBOOK_DEV").is_ok() {
        workspace::register_project_item::<NotebookEditor>(cx);
    }

    cx.observe_flag::<NotebookFeatureFlag, _>({
        move |is_enabled, cx| {
            if is_enabled {
                workspace::register_project_item::<NotebookEditor>(cx);
            } else {
                // todo: there is no way to unregister a project item, so if the feature flag
                // gets turned off they need to restart Zed.
            }
        }
    })
    .detach();
}

pub struct NotebookEditor {
    languages: Arc<LanguageRegistry>,
    project: Entity<Project>,

    focus_handle: FocusHandle,
    notebook_item: Entity<NotebookItem>,

    remote_id: Option<ViewId>,
    cell_list: ListState,

    selected_cell_index: usize,
    cell_order: Vec<CellId>,
    cell_map: HashMap<CellId, Cell>,

    // Kernel for executing code
    kernel: Kernel,
    kernel_specification: Option<KernelSpecification>,
    // Map message IDs to cell IDs for tracking execution responses
    pending_executions: HashMap<String, CellId>,
    // Task for receiving kernel messages
    _message_task: Option<Task<()>>,
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

        let notebook_language = notebook_item.read(cx).notebook_language();
        let notebook_language = cx
            .spawn_in(window, async move |_, _| notebook_language.await)
            .shared();

        let mut cell_order = vec![]; // Vec<CellId>
        let mut cell_map = HashMap::default(); // HashMap<CellId, Cell>

        for (index, cell) in notebook_item
            .read(cx)
            .notebook
            .clone()
            .cells
            .iter()
            .enumerate()
        {
            let cell_id = cell.id();
            cell_order.push(cell_id.clone());
            cell_map.insert(
                cell_id.clone(),
                Cell::load(cell, &languages, notebook_language.clone(), window, cx),
            );
        }

        let notebook_handle = cx.entity().downgrade();
        let cell_count = cell_order.len();

        // Setup editor actions on all code cells
        for (cell_id, cell) in &cell_map {
            if let Cell::Code(code_cell) = cell {
                let cell_id = cell_id.clone();
                let notebook_handle = notebook_handle.clone();
                let editor = code_cell.read(cx).editor.clone();
                editor.update(cx, |editor, _cx| {
                    setup_cell_editor_actions(editor, cell_id, notebook_handle);
                });
            }
        }

        log::info!("NotebookEditor::new - loaded {} cells", cell_count);

        let this = cx.entity();
        // Use a reasonable estimated cell height (100px) so cells are rendered
        // This is just an estimate for scrolling performance, actual cells size themselves
        let cell_list = ListState::new(cell_count, gpui::ListAlignment::Top, px(100.));

        let mut editor = Self {
            project,
            languages: languages.clone(),
            focus_handle,
            notebook_item,
            remote_id: None,
            cell_list,
            selected_cell_index: 0,
            cell_order: cell_order.clone(),
            cell_map: cell_map.clone(),
            kernel: Kernel::Shutdown,
            kernel_specification: None,
            pending_executions: HashMap::default(),
            _message_task: None,
        };

        // Try to initialize kernel
        editor.initialize_kernel(window, cx);

        editor
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
        for cell in self.cell_map.values() {
            if let Cell::Code(code_cell) = cell {
                code_cell.update(cx, |cell, _cx| {
                    cell.clear_outputs();
                });
            }
        }
    }

    fn run_cells(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        // Check if kernel is running
        if !matches!(self.kernel, Kernel::RunningKernel(_)) {
            eprintln!("Cannot run cells: kernel is not running (status: {:?})", self.kernel.status());
            return;
        }

        // Execute all code cells in order
        for cell_id in self.cell_order.clone() {
            if let Some(Cell::Code(code_cell)) = self.cell_map.get(&cell_id) {
                self.execute_cell(cell_id.clone(), code_cell.clone(), window, cx);
            }
        }
    }

    fn run_selected_cell(&mut self, _: &RunSelectedCell, window: &mut Window, cx: &mut Context<Self>) {
        // Check if kernel is running
        if !matches!(self.kernel, Kernel::RunningKernel(_)) {
            eprintln!("Cannot run cell: kernel is not running");
            return;
        }

        // Get the selected cell
        let selected_index = self.selected_cell_index;
        if let Some(cell_id) = self.cell_order.get(selected_index) {
            if let Some(Cell::Code(code_cell)) = self.cell_map.get(cell_id) {
                self.execute_cell(cell_id.clone(), code_cell.clone(), window, cx);
            }
        }
    }

    fn run_selected_cell_and_move_next(&mut self, _: &RunSelectedCellAndMoveNext, window: &mut Window, cx: &mut Context<Self>) {
        // Run the cell first
        self.run_selected_cell(&RunSelectedCell, window, cx);

        // Then move to next cell
        let next_index = (self.selected_cell_index + 1).min(self.cell_count() - 1);
        if next_index != self.selected_cell_index {
            self.selected_cell_index = next_index;
            // Scroll to reveal the newly selected cell
            self.cell_list.scroll_to_reveal_item(next_index);
            // Focus notebook to exit edit mode
            window.focus(&self.focus_handle);
            cx.notify();
        }
    }

    fn enter_edit_mode(&mut self, _: &EnterEditMode, window: &mut Window, cx: &mut Context<Self>) {
        // Focus the selected cell's editor if it's a code cell
        let selected_index = self.selected_cell_index;
        if let Some(cell_id) = self.cell_order.get(selected_index) {
            if let Some(Cell::Code(code_cell)) = self.cell_map.get(cell_id) {
                let editor = code_cell.read(cx).editor.clone();
                window.focus(&editor.read(cx).focus_handle(cx));
            }
        }
    }

    fn key_down(&mut self, event: &gpui::KeyDownEvent, window: &mut Window, cx: &mut Context<Self>) {
        let is_ctrl_enter = event.keystroke.key == "enter" && event.keystroke.modifiers.control;
        let is_shift_enter = event.keystroke.key == "enter" && event.keystroke.modifiers.shift;

        // Find which cell's editor is focused (if any)
        let mut focused_cell_info: Option<(CellId, usize)> = None;
        for (index, cell_id) in self.cell_order.iter().enumerate() {
            if let Some(Cell::Code(code_cell)) = self.cell_map.get(cell_id) {
                let editor = code_cell.read(cx).editor.clone();
                if editor.read(cx).focus_handle(cx).is_focused(window) {
                    focused_cell_info = Some((cell_id.clone(), index));
                    break;
                }
            }
        }

        if is_ctrl_enter || is_shift_enter {
            log::info!("NotebookEditor::key_down: Matched Ctrl/Shift+Enter");

            // Stop propagation immediately to prevent editor from adding newline
            cx.stop_propagation();

            if let Some((cell_id, cell_index)) = focused_cell_info {
                // An editor is focused - run that cell
                log::info!("NotebookEditor::key_down: Running focused cell at index {}", cell_index);
                if let Some(Cell::Code(code_cell)) = self.cell_map.get(&cell_id) {
                    self.execute_cell(cell_id, code_cell.clone(), window, cx);

                    // Shift+Enter moves to next cell
                    if is_shift_enter {
                        let next_index = (cell_index + 1).min(self.cell_count() - 1);
                        if next_index != cell_index {
                            log::info!("NotebookEditor::key_down: Moving focus to next cell {}", next_index);
                            self.focus_cell(next_index, window, cx);
                        }
                    }
                }
            } else {
                // No editor focused - run selected cell
                log::info!("NotebookEditor::key_down: Running selected cell {}", self.selected_cell_index);
                let selected_index = self.selected_cell_index;
                if let Some(cell_id) = self.cell_order.get(selected_index) {
                    if let Some(Cell::Code(code_cell)) = self.cell_map.get(cell_id) {
                        self.execute_cell(cell_id.clone(), code_cell.clone(), window, cx);

                        // Shift+Enter moves to next cell
                        if is_shift_enter {
                            let next_index = (selected_index + 1).min(self.cell_count() - 1);
                            if next_index != selected_index {
                                log::info!("NotebookEditor::key_down: Moving focus to next cell {}", next_index);
                                self.focus_cell(next_index, window, cx);
                            }
                        }
                    }
                }
            }
        }
        // For regular Enter, don't stop propagation - let it reach the editor for newlines
    }

    fn focus_cell(&mut self, index: usize, window: &mut Window, cx: &mut Context<Self>) {
        if index >= self.cell_count() {
            return;
        }

        self.selected_cell_index = index;

        // Don't focus the editor - just select the cell
        // This way Shift+Enter moves selection without entering edit mode
        // Focus the notebook instead to blur any editors
        window.focus(&self.focus_handle);

        cx.notify();
    }

    fn execute_cell(
        &mut self,
        cell_id: CellId,
        code_cell: Entity<super::CodeCell>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let Kernel::RunningKernel(ref kernel) = self.kernel else {
            eprintln!("Cannot execute cell: kernel not running");
            return;
        };

        // Clear previous outputs before execution
        code_cell.update(cx, |cell, _cx| {
            cell.clear_outputs();
        });

        // Get the code from the cell's editor buffer (which reflects current edits)
        let code = code_cell.read(cx).text(cx);

        if code.trim().is_empty() {
            return;
        }

        // Create execute request
        let execute_request = ExecuteRequest {
            code,
            ..ExecuteRequest::default()
        };

        let message: JupyterMessage = execute_request.into();
        let msg_id = message.header.msg_id.clone();

        // Track this execution
        self.pending_executions.insert(msg_id.clone(), cell_id);

        // Send to kernel
        if let Err(e) = kernel.request_tx().try_send(message) {
            eprintln!("Failed to send execute request: {:?}", e);
            self.pending_executions.remove(&msg_id);
        }
    }

    fn open_notebook(&mut self, _: &OpenNotebook, _window: &mut Window, _cx: &mut Context<Self>) {
        println!("Open notebook triggered");
    }

    fn move_cell_up(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let current_index = self.selected_cell_index;

        // Can't move the first cell up
        if current_index == 0 {
            return;
        }

        // Swap with the cell above
        self.cell_order.swap(current_index, current_index - 1);
        self.selected_cell_index = current_index - 1;

        cx.notify();
    }

    fn move_cell_down(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let current_index = self.selected_cell_index;
        let cell_count = self.cell_count();

        // Can't move the last cell down
        if current_index >= cell_count - 1 {
            return;
        }

        // Swap with the cell below
        self.cell_order.swap(current_index, current_index + 1);
        self.selected_cell_index = current_index + 1;

        cx.notify();
    }

    fn add_markdown_block(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        use super::cell::MarkdownCell;
        use uuid::Uuid;

        let cell_id = CellId::from(Uuid::new_v4());
        let source = String::new();
        let metadata = serde_json::from_str("{}").unwrap();

        let markdown_cell = cx.new(|cx| {
            let markdown_parsing_task = {
                let languages = self.languages.clone();
                let source = source.clone();

                cx.spawn_in(window, async move |this, cx| {
                    let parsed_markdown = cx
                        .background_spawn(async move {
                            markdown_preview::markdown_parser::parse_markdown(&source, None, Some(languages)).await
                        })
                        .await;

                    this.update(cx, |cell: &mut MarkdownCell, cx| {
                        cell.parsed_markdown = Some(parsed_markdown);
                        cx.notify();
                    })
                    .log_err();
                })
            };

            MarkdownCell {
                markdown_parsing_task,
                image_cache: RetainAllImageCache::new(cx),
                languages: self.languages.clone(),
                id: cell_id.clone(),
                metadata,
                source: source.clone(),
                parsed_markdown: None,
                selected: false,
                cell_position: None,
            }
        });

        // Insert after the selected cell
        let insert_index = self.selected_cell_index + 1;
        self.cell_order.insert(insert_index, cell_id.clone());
        self.cell_map.insert(cell_id, Cell::Markdown(markdown_cell));

        // Update list state with new count
        self.cell_list.reset(self.cell_order.len());

        // Select the newly added cell
        self.selected_cell_index = insert_index;

        cx.notify();
    }

    fn add_code_block(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        use super::cell::CodeCell;
        use editor::{Editor, EditorMode, MultiBuffer};
        use language::Buffer;
        use theme::ThemeSettings;
        use uuid::Uuid;

        let cell_id = CellId::from(Uuid::new_v4());
        let text = String::new();
        let metadata = serde_json::from_str("{}").unwrap();

        let notebook_language = self.notebook_item.read(cx).notebook_language();
        let notebook_language = cx
            .spawn_in(window, async move |_, _| notebook_language.await)
            .shared();

        let code_cell = cx.new(|cx| {
            let buffer = cx.new(|cx| Buffer::local(text.clone(), cx));
            let multi_buffer = cx.new(|cx| MultiBuffer::singleton(buffer.clone(), cx));

            let editor_view = cx.new(|cx| {
                let mut editor = Editor::new(
                    EditorMode::AutoHeight {
                        min_lines: 1,
                        max_lines: Some(1024),
                    },
                    multi_buffer,
                    None,
                    window,
                    cx,
                );

                let theme = ThemeSettings::get_global(cx);

                let refinement = TextStyleRefinement {
                    font_family: Some(theme.buffer_font.family.clone()),
                    font_size: Some(theme.buffer_font_size(cx).into()),
                    color: Some(cx.theme().colors().editor_foreground),
                    background_color: Some(gpui::transparent_black()),
                    ..Default::default()
                };

                editor.set_text(text.clone(), window, cx);
                editor.set_show_gutter(false, cx);
                editor.set_text_style_refinement(refinement);

                editor
            });

            let language_task = cx.spawn_in(window, async move |_this, cx| {
                let language = notebook_language.await;

                buffer.update(cx, |buffer, cx| {
                    buffer.set_language(language.clone(), cx);
                })
                .log_err();
            });

            CodeCell {
                id: cell_id.clone(),
                metadata,
                execution_count: None,
                source: text,
                editor: editor_view,
                outputs: vec![],
                selected: false,
                cell_position: None,
                language_task,
            }
        });

        // Setup editor actions on the new cell
        let notebook_handle = cx.entity().downgrade();
        let cell_id_for_action = cell_id.clone();
        let editor = code_cell.read(cx).editor.clone();
        editor.update(cx, |editor, _cx| {
            setup_cell_editor_actions(editor, cell_id_for_action, notebook_handle);
        });

        // Insert after the selected cell
        let insert_index = self.selected_cell_index + 1;
        self.cell_order.insert(insert_index, cell_id.clone());
        self.cell_map.insert(cell_id, Cell::Code(code_cell));

        // Update list state with new count
        self.cell_list.reset(self.cell_order.len());

        // Select the newly added cell
        self.selected_cell_index = insert_index;

        cx.notify();
    }

    fn initialize_kernel(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        // Get kernel specification from notebook metadata or use default Python
        let kernel_spec = self.get_kernel_specification(cx);

        if let Some(kernel_spec) = kernel_spec {
            self.kernel_specification = Some(kernel_spec.clone());
            self.start_kernel(kernel_spec, window, cx);
        }
    }

    fn find_python_in_venv(&self, cx: &App) -> Option<PathBuf> {
        let notebook_item = self.notebook_item.read(cx);
        let notebook_dir = notebook_item.path.parent()?;

        // Try common venv locations
        for venv_name in &["venv", ".venv", "env", ".env"] {
            let venv_python = notebook_dir.join(venv_name).join("bin").join("python");
            if venv_python.exists() {
                log::info!("Found venv Python at: {:?}", venv_python);
                return Some(venv_python);
            }
            // Also try python3
            let venv_python3 = notebook_dir.join(venv_name).join("bin").join("python3");
            if venv_python3.exists() {
                log::info!("Found venv Python at: {:?}", venv_python3);
                return Some(venv_python3);
            }
        }

        log::warn!("No venv found in notebook directory, using system Python");
        None
    }

    fn get_kernel_specification(&self, cx: &App) -> Option<KernelSpecification> {
        // For now, we'll try to get the kernelspec from the notebook metadata
        // In the future, this could present a picker to the user
        let notebook_item = self.notebook_item.read(cx);
        let kernelspec = notebook_item.notebook.metadata.kernelspec.as_ref()?;

        // Try to find venv Python first, fall back to system python
        let python_path = self.find_python_in_venv(cx)
            .unwrap_or_else(|| PathBuf::from("python3"));

        log::info!("Using Python for kernel: {:?}", python_path);

        use crate::kernels::LocalKernelSpecification;

        Some(KernelSpecification::Jupyter(LocalKernelSpecification {
            name: kernelspec.name.clone(),
            path: PathBuf::from("jupyter"),
            kernelspec: jupyter_protocol::JupyterKernelspec {
                argv: vec![
                    python_path.to_string_lossy().to_string(),
                    "-m".to_string(),
                    "ipykernel_launcher".to_string(),
                    "-f".to_string(),
                    "{connection_file}".to_string(),
                ],
                display_name: kernelspec.display_name.clone(),
                language: kernelspec.language.clone().unwrap_or_else(|| "python".to_string()),
                interrupt_mode: None,
                metadata: None,
                env: None,
            },
        }))
    }

    fn start_kernel(
        &mut self,
        kernel_spec: KernelSpecification,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        use crate::kernels::{NativeRunningKernel, RemoteRunningKernel};
        use std::env::temp_dir;

        let fs = self.project.read(cx).fs().clone();
        let entity_id = cx.entity().entity_id();

        // Get working directory from notebook path or use temp
        let working_directory = self.notebook_item
            .read(cx)
            .path
            .parent()
            .map(|p| p.to_path_buf())
            .unwrap_or_else(temp_dir);

        let notebook_handle = cx.entity();

        // Start the kernel based on type
        let kernel = match kernel_spec.clone() {
            KernelSpecification::Jupyter(kernel_specification)
            | KernelSpecification::PythonEnv(kernel_specification) => NativeRunningKernel::new(
                kernel_specification,
                entity_id,
                working_directory,
                fs,
                notebook_handle.clone(),
                window,
                cx,
            ),
            KernelSpecification::Remote(remote_kernel_specification) => RemoteRunningKernel::new(
                remote_kernel_specification,
                working_directory,
                notebook_handle.clone(),
                window,
                cx,
            ),
        };

        let pending_kernel = cx
            .spawn(async move |this, cx| {
                let kernel = kernel.await;

                match kernel {
                    Ok(kernel) => {
                        this.update(cx, |notebook, cx| {
                            notebook.kernel = Kernel::RunningKernel(kernel);
                            cx.notify();
                        })
                        .ok();
                    }
                    Err(err) => {
                        this.update(cx, |notebook, cx| {
                            notebook.kernel = Kernel::ErroredLaunch(err.to_string());
                            cx.notify();
                        })
                        .ok();
                    }
                }
            })
            .shared();

        self.kernel = Kernel::StartingKernel(pending_kernel);
        cx.notify();
    }

    /// Route incoming Jupyter messages to the appropriate cell
    /// Internal implementation that works with &mut App
    fn route_internal(&mut self, message: &JupyterMessage, window: &mut Window, cx: &mut App) {
        // Handle status messages
        match &message.content {
            JupyterMessageContent::Status(status) => {
                self.kernel.set_execution_state(&status.execution_state);
                // Note: caller needs to notify
                return;
            }
            JupyterMessageContent::KernelInfoReply(reply) => {
                self.kernel.set_kernel_info(reply);
                // Note: caller needs to notify
                return;
            }
            _ => {}
        }

        // Route messages to cells based on parent message ID
        let parent_message_id = match message.parent_header.as_ref() {
            Some(header) => &header.msg_id,
            None => return,
        };

        // Find the cell that this message is for
        let cell_id = match self.pending_executions.get(parent_message_id) {
            Some(id) => id.clone(),
            None => return,
        };

        let cell = match self.cell_map.get(&cell_id) {
            Some(Cell::Code(code_cell)) => code_cell,
            _ => return,
        };

        // Handle different message types
        match &message.content {
            JupyterMessageContent::ExecuteReply(reply) => {
                // Update execution count
                cell.update(cx, |cell, _cx| {
                    cell.set_execution_count(reply.execution_count.0 as i32);
                });

                // Remove from pending executions
                self.pending_executions.remove(parent_message_id);
                // Note: caller needs to notify
            }
            JupyterMessageContent::ExecuteResult(result) => {
                // Add output to cell
                let output = Output::new(&result.data, None, window, cx);
                cell.update(cx, |cell, _cx| {
                    cell.add_output(output);
                });
                // Note: caller needs to notify
            }
            JupyterMessageContent::DisplayData(display_data) => {
                // Add output to cell
                let output = Output::new(&display_data.data, None, window, cx);
                cell.update(cx, |cell, _cx| {
                    cell.add_output(output);
                });
                // Note: caller needs to notify
            }
            JupyterMessageContent::StreamContent(stream) => {
                // Add stream output
                use crate::outputs::plain::TerminalOutput;
                let terminal_output = cx.new(|cx| TerminalOutput::from(&stream.text, window, cx));
                let output = Output::Stream {
                    content: terminal_output,
                };
                cell.update(cx, |cell, _cx| {
                    cell.add_output(output);
                });
                // Note: caller needs to notify
            }
            JupyterMessageContent::ErrorOutput(error) => {
                // Add error output
                use crate::outputs::{plain::TerminalOutput, user_error::ErrorView};
                let traceback = cx.new(|cx| {
                    TerminalOutput::from(&error.traceback.join("\n"), window, cx)
                });
                let error_view = ErrorView {
                    ename: error.ename.clone(),
                    evalue: error.evalue.clone(),
                    traceback,
                };
                let output = Output::ErrorOutput(error_view);
                cell.update(cx, |cell, _cx| {
                    cell.add_output(output);
                });
                // Note: caller needs to notify
            }
            _ => {}
        }
    }

    fn cell_count(&self) -> usize {
        self.cell_map.len()
    }

    fn selected_index(&self) -> usize {
        self.selected_cell_index
    }

    fn has_focused_editor(&self, window: &Window, cx: &App) -> bool {
        let result = self.cell_map.values().any(|cell| {
            if let Cell::Code(code_cell) = cell {
                let is_focused = code_cell.read(cx).editor.read(cx).focus_handle(cx).is_focused(window);
                if is_focused {
                    log::debug!("Editor in cell is focused");
                }
                is_focused
            } else {
                false
            }
        });
        log::debug!("has_focused_editor returning: {}", result);
        result
    }

    pub fn set_selected_index(
        &mut self,
        index: usize,
        jump_to_index: bool,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        // let previous_index = self.selected_cell_index;
        self.selected_cell_index = index;
        let current_index = self.selected_cell_index;

        // in the future we may have some `on_cell_change` event that we want to fire here

        if jump_to_index {
            self.jump_to_cell(current_index, window, cx);
        }
    }

    pub fn select_next(
        &mut self,
        _: &menu::SelectNext,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        // Don't move selection if an editor has focus
        let has_focus = self.has_focused_editor(window, cx);
        log::debug!("select_next: has_focused_editor={}, current_index={}", has_focus, self.selected_cell_index);

        if has_focus {
            return;
        }

        let count = self.cell_count();
        if count > 0 {
            let index = self.selected_index();
            let ix = if index == count - 1 {
                count - 1
            } else {
                index + 1
            };
            log::debug!("select_next: moving from {} to {}", index, ix);
            self.set_selected_index(ix, true, window, cx);
            cx.notify();
        }
    }

    pub fn select_previous(
        &mut self,
        _: &menu::SelectPrevious,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        // Don't move selection if an editor has focus
        if self.has_focused_editor(window, cx) {
            return;
        }

        let count = self.cell_count();
        if count > 0 {
            let index = self.selected_index();
            let ix = if index == 0 { 0 } else { index - 1 };
            self.set_selected_index(ix, true, window, cx);
            cx.notify();
        }
    }

    pub fn select_first(
        &mut self,
        _: &menu::SelectFirst,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        // Don't move selection if an editor has focus
        if self.has_focused_editor(window, cx) {
            return;
        }

        let count = self.cell_count();
        if count > 0 {
            self.set_selected_index(0, true, window, cx);
            cx.notify();
        }
    }

    pub fn select_last(
        &mut self,
        _: &menu::SelectLast,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        // Don't move selection if an editor has focus
        if self.has_focused_editor(window, cx) {
            return;
        }

        let count = self.cell_count();
        if count > 0 {
            self.set_selected_index(count - 1, true, window, cx);
            cx.notify();
        }
    }

    fn jump_to_cell(&mut self, index: usize, _window: &mut Window, _cx: &mut Context<Self>) {
        self.cell_list.scroll_to_reveal_item(index);
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
                    .child(Self::render_notebook_control(
                        "more-menu",
                        IconName::Ellipsis,
                        window,
                        cx,
                    ))
                    .child(
                        Self::button_group(window, cx)
                            .child(IconButton::new("repl", IconName::ReplNeutral)),
                    ),
            )
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
    ) -> AnyElement {
        let cell_position = self.cell_position(index);

        let is_selected = index == self.selected_cell_index;

        let cell_element = match cell {
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
        };

        // Wrap in a div with mouse_down handler to select the cell
        div()
            .id(("cell-wrapper", index))
            .w_full()
            .on_mouse_down(gpui::MouseButton::Left, cx.listener(move |this, _event, window, cx| {
                log::info!("Cell wrapper mouse_down: selecting cell {} and focusing notebook", index);
                // Select the cell and focus the notebook to blur any editors
                // This ensures run button actions go to the notebook handler
                // If user clicks the editor, it will re-focus itself immediately after
                this.selected_cell_index = index;
                window.focus(&this.focus_handle);
                cx.notify();
            }))
            .child(cell_element)
            .into_any_element()
    }
}

impl crate::kernels::MessageRouter for NotebookEditor {
    fn route(&mut self, message: &JupyterMessage, window: &mut Window, cx: &mut App) {
        // Call the internal implementation that works with &mut App
        self.route_internal(message, window, cx);
        // Note: We can't call cx.notify() here because we only have &mut App
        // The notification must happen in the update_in closure that calls this
    }

    fn kernel_errored(&mut self, error_message: String, _window: &mut Window, _cx: &mut App) {
        self.kernel = crate::Kernel::ErroredLaunch(error_message);
        // Note: We can't call cx.notify() here because we only have &mut App
        // The notification must happen in the update_in closure that calls this
    }
}

impl Render for NotebookEditor {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        div()
            .key_context("NotebookEditor")
            .track_focus(&self.focus_handle)
            .on_key_down(cx.listener(Self::key_down))
            .on_action(cx.listener(|this, &OpenNotebook, window, cx| {
                this.open_notebook(&OpenNotebook, window, cx)
            }))
            .on_action(
                cx.listener(|this, &ClearOutputs, window, cx| this.clear_outputs(window, cx)),
            )
            .on_action(cx.listener(|this, &RunAll, window, cx| this.run_cells(window, cx)))
            .on_action(cx.listener(Self::run_selected_cell))
            .on_action(cx.listener(Self::run_selected_cell_and_move_next))
            .on_action(cx.listener(Self::enter_edit_mode))
            .on_action(cx.listener(|this, &BlurAllEditors, window, cx| {
                // Focus the notebook to blur all editors
                window.focus(&this.focus_handle);
            }))
            .on_action(cx.listener(|this, &MoveCellUp, window, cx| this.move_cell_up(window, cx)))
            .on_action(
                cx.listener(|this, &MoveCellDown, window, cx| this.move_cell_down(window, cx)),
            )
            .on_action(cx.listener(|this, &AddMarkdownBlock, window, cx| {
                this.add_markdown_block(window, cx)
            }))
            .on_action(
                cx.listener(|this, &AddCodeBlock, window, cx| this.add_code_block(window, cx)),
            )
            .on_action(cx.listener(Self::select_next))
            .on_action(cx.listener(Self::select_previous))
            .on_action(cx.listener(Self::select_first))
            .on_action(cx.listener(Self::select_last))
            .size_full()
            .flex()
            .flex_row()
            .items_start()
            .px(DynamicSpacing::Base12.px(cx))
            .gap(DynamicSpacing::Base12.px(cx))
            .bg(cx.theme().colors().tab_bar_background)
            .child(
                v_flex()
                    .id("notebook-cells")
                    .flex_1()
                    .h_full()
                    .child(
                        list(
                            self.cell_list.clone(),
                            cx.processor(|this, ix, window, cx| {
                                this.cell_order
                                    .get(ix)
                                    .and_then(|cell_id| this.cell_map.get(cell_id))
                                    .map(|cell| this.render_cell(ix, cell, window, cx))
                                    .unwrap_or_else(|| div().into_any())
                            }),
                        )
                        .size_full()
                    ),
            )
            .child(self.render_notebook_controls(window, cx))
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
                    .read_with(cx, |project, cx| project.absolute_path(&path, cx))?
                    .with_context(|| format!("finding the absolute path of {path:?}"))?;

                // todo: watch for changes to the file
                let file_content = fs.load(abs_path.as_path()).await?;

                // Handle empty files by creating a default notebook
                let notebook = if file_content.trim().is_empty() {
                    // Create an empty notebook with default metadata
                    nbformat::v4::Notebook {
                        cells: vec![],
                        metadata: nbformat::v4::Metadata {
                            kernelspec: None,
                            language_info: None,
                            authors: None,
                            additional: std::collections::HashMap::new(),
                        },
                        nbformat: 4,
                        nbformat_minor: 5,
                    }
                } else {
                    let parsed_notebook = nbformat::parse_notebook(&file_content);
                    match parsed_notebook {
                        Ok(nbformat::Notebook::V4(notebook)) => notebook,
                        // 4.1 - 4.4 are converted to 4.5
                        Ok(nbformat::Notebook::Legacy(legacy_notebook)) => {
                            // TODO: Decide if we want to mutate the notebook by including Cell IDs
                            // and any other conversions
                            nbformat::upgrade_legacy_notebook(legacy_notebook)?
                        }
                        // Bad notebooks and notebooks v4.0 and below are not supported
                        Err(e) => {
                            anyhow::bail!("Failed to parse notebook: {:?}", e);
                        }
                    }
                };

                let id = project
                    .update(cx, |project, cx| {
                        project.entry_for_path(&path, cx).map(|entry| entry.id)
                    })?
                    .context("Entry not found")?;

                cx.new(|_| NotebookItem {
                    path: abs_path,
                    project_path: path,
                    languages,
                    notebook,
                    id,
                })
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

    fn tab_content(&self, params: TabContentParams, window: &Window, cx: &App) -> AnyElement {
        Label::new(self.tab_content_text(params.detail.unwrap_or(0), cx))
            .single_line()
            .color(params.text_color())
            .when(params.preview, |this| this.italic())
            .into_any_element()
    }

    fn tab_content_text(&self, _detail: usize, cx: &App) -> SharedString {
        let path = &self.notebook_item.read(cx).path;
        let title = path
            .file_name()
            .unwrap_or_else(|| path.as_os_str())
            .to_string_lossy()
            .to_string();
        title.into()
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
    fn as_searchable(&self, _: &Entity<Self>) -> Option<Box<dyn SearchableItemHandle>> {
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
        true  // Always allow saving notebooks
    }

    fn save(
        &mut self,
        _options: SaveOptions,
        project: Entity<Project>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Task<Result<()>> {
        let notebook_item = self.notebook_item.clone();
        let fs = project.read(cx).fs().clone();

        // Collect cell data from editors
        let mut cells = Vec::new();
        for cell_id in &self.cell_order {
            if let Some(cell) = self.cell_map.get(cell_id) {
                match cell {
                    Cell::Code(code_cell) => {
                        let code_cell = code_cell.read(cx);
                        let source = code_cell.text(cx);
                        let cell_data = nbformat::v4::Cell::Code {
                            id: code_cell.id.clone(),
                            metadata: code_cell.metadata.clone(),
                            execution_count: code_cell.execution_count,
                            source: source.lines().map(|s| s.to_string()).collect(),
                            outputs: vec![], // We don't save outputs for now
                        };
                        cells.push(cell_data);
                    }
                    Cell::Markdown(md_cell) => {
                        let md_cell = md_cell.read(cx);
                        let cell_data = nbformat::v4::Cell::Markdown {
                            id: md_cell.id.clone(),
                            metadata: md_cell.metadata.clone(),
                            source: md_cell.source.lines().map(|s| s.to_string()).collect(),
                            attachments: None,
                        };
                        cells.push(cell_data);
                    }
                    Cell::Raw(raw_cell) => {
                        let raw_cell = raw_cell.read(cx);
                        let cell_data = nbformat::v4::Cell::Raw {
                            id: raw_cell.id().clone(),
                            metadata: raw_cell.metadata().clone(),
                            source: raw_cell.source().lines().map(|s| s.to_string()).collect(),
                        };
                        cells.push(cell_data);
                    }
                }
            }
        }

        cx.spawn(async move |_, cx| {
            let path = notebook_item.read_with(cx, |item, _| item.path.clone())?;
            let metadata = notebook_item.read_with(cx, |item, _| item.notebook.metadata.clone())?;
            let nbformat_version = notebook_item.read_with(cx, |item, _| item.notebook.nbformat)?;
            let nbformat_minor = notebook_item.read_with(cx, |item, _| item.notebook.nbformat_minor)?;

            // Create updated notebook
            let updated_notebook = nbformat::v4::Notebook {
                cells,
                metadata,
                nbformat: nbformat_version,
                nbformat_minor,
            };

            // Serialize to JSON
            let json = serde_json::to_string_pretty(&updated_notebook)?;

            // Write to file
            fs.save(&path, &json.into(), Default::default()).await?;

            Ok(())
        })
    }

    fn save_as(
        &mut self,
        _project: Entity<Project>,
        _path: ProjectPath,
        _window: &mut Window,
        _cx: &mut Context<Self>,
    ) -> Task<Result<()>> {
        Task::ready(Err(anyhow::anyhow!("save_as not yet implemented for notebooks")))
    }

    fn reload(
        &mut self,
        _project: Entity<Project>,
        _window: &mut Window,
        _cx: &mut Context<Self>,
    ) -> Task<Result<()>> {
        Task::ready(Err(anyhow::anyhow!("reload not yet implemented for notebooks")))
    }

    fn is_dirty(&self, cx: &App) -> bool {
        self.cell_map.values().any(|cell| {
            if let Cell::Code(code_cell) = cell {
                code_cell.read(cx).is_dirty(cx)
            } else {
                false
            }
        })
    }
}

// TODO: Implement this to allow us to persist to the database, etc:
// impl SerializableItem for NotebookEditor {}

impl ProjectItem for NotebookEditor {
    type Item = NotebookItem;

    fn for_project_item(
        project: Entity<Project>,
        _: Option<&Pane>,
        item: Entity<Self::Item>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Self
    where
        Self: Sized,
    {
        Self::new(project, item, window, cx)
    }
}
