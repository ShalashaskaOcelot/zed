use std::sync::Arc;
use std::time::{Duration, Instant};

use editor::{Editor, EditorMode, MultiBuffer, SizingBehavior};
use futures::future::Shared;
use gpui::{
    App, ClipboardItem, Entity, EventEmitter, Focusable, Hsla, InteractiveElement,
    RetainAllImageCache, StatefulInteractiveElement, Task, prelude::*,
};
use language::{Buffer, Language, LanguageRegistry};
use markdown::{Markdown, MarkdownElement, MarkdownFont, MarkdownStyle};
use nbformat::v4::{CellId, CellMetadata, CellType};
use runtimelib::{JupyterMessage, JupyterMessageContent, ReplyStatus};
use settings::Settings as _;
use ui::{CommonAnimationExt, ContextMenu, IconButtonShape, PopoverMenu, Tooltip, prelude::*};
use util::ResultExt;
use zed_actions::notebook::{
    AddCellBelow, DeleteCell, RunCellAndBelow, RunCellsAbove,
};

use crate::{
    notebook::{CELL_HOVER_GROUP, CODE_BLOCK_INSET, GUTTER_WIDTH},
    outputs::{Output, plain, plain::TerminalOutput, user_error::ErrorView},
    repl_settings::ReplSettings,
};

#[derive(Copy, Clone, PartialEq, PartialOrd)]
pub enum CellPosition {
    First,
    Middle,
    Last,
}

pub enum CellControlType {
    RunCell,
    RerunCell,
    StopCell,
    ClearCell,
    CellOptions,
    CollapseCell,
    ExpandCell,
}

pub enum CellEvent {
    Run(CellId),
    FocusedIn(CellId),
    /// A per-cell toolbar button was clicked. The notebook selects this cell
    /// (by id) and then performs the action, so the button always acts on its
    /// own cell even when the toolbar is shown on hover of a non-selected cell.
    ToolbarAction(CellId, CellToolbarAction),
    /// The gutter stop button was clicked on a running/queued cell: interrupt
    /// it if running, or drop it from the queue if only pending — scoped to
    /// this cell rather than the whole kernel/batch.
    Stop(CellId),
    /// The cell was clicked with a selection modifier held: shift extends the
    /// contiguous selection from the anchor to this cell; ctrl/cmd (`!shift`)
    /// toggles this cell in a discontiguous multi-selection.
    ModifiedClick { id: CellId, shift: bool },
    /// The cell's gutter (the run-button / accent-bar strip, or the output
    /// gutter) was plain-clicked. Selects just this cell and drops into command
    /// mode. Emitted only from the gutter, never the editor, so clicking the
    /// cell body still focuses the editor into edit mode.
    PlainClick { id: CellId },
    /// Savable cell metadata changed (e.g. input/output collapse state, which
    /// persists to the .ipynb): the notebook should count as dirty.
    MetadataChanged(CellId),
}

/// Capture-phase mouse-down classifier shared by every cell root: a left click
/// with a selection modifier held becomes a `ModifiedClick` (`Some(shift)`) and
/// stops propagating, so it doesn't also focus the cell's editor. A plain left
/// click returns `None` and passes through untouched, so clicking the editor
/// focuses it (edit mode); plain selection is instead driven by a gutter click
/// (see `CellEvent::PlainClick`), which never overlaps the editor.
fn selection_modifiers(event: &gpui::MouseDownEvent) -> Option<bool> {
    if event.button != gpui::MouseButton::Left {
        return None;
    }
    if event.modifiers.shift {
        Some(true)
    } else if event.modifiers.secondary() {
        Some(false)
    } else {
        None
    }
}

/// Actions offered by the per-cell hover/selection toolbar. Each maps to an
/// existing notebook action; the toolbar is purely a discoverable surface.
#[derive(Clone, Copy)]
pub enum CellToolbarAction {
    Run,
    RunAbove,
    RunBelow,
    AddBelow,
    Delete,
}

pub enum MarkdownCellEvent {
    FinishedEditing,
    Run(CellId),
}

impl CellControlType {
    fn icon_name(&self) -> IconName {
        match self {
            CellControlType::RunCell => IconName::PlayFilled,
            CellControlType::RerunCell => IconName::ArrowCircle,
            CellControlType::StopCell => IconName::Stop,
            CellControlType::ClearCell => IconName::ListX,
            CellControlType::CellOptions => IconName::Ellipsis,
            CellControlType::CollapseCell => IconName::ChevronDown,
            CellControlType::ExpandCell => IconName::ChevronRight,
        }
    }
    fn id(&self) -> &'static str {
        match self {
            CellControlType::RunCell => "CellControlType::RunCell",
            CellControlType::RerunCell => "CellControlType::RerunCell",
            CellControlType::StopCell => "CellControlType::StopCell",
            CellControlType::ClearCell => "CellControlType::ClearCell",
            CellControlType::CellOptions => "CellControlType::CellOptions",
            CellControlType::CollapseCell => "CellControlType::CollapseCell",
            CellControlType::ExpandCell => "CellControlType::ExpandCell",
        }
    }
}

pub struct CellControl {
    button: IconButton,
}

impl CellControl {
    fn new(id: impl Into<SharedString>, control_type: CellControlType) -> Self {
        let icon_name = control_type.icon_name();
        let id = id.into();
        let button = IconButton::new(id, icon_name)
            .icon_size(IconSize::Small)
            .shape(IconButtonShape::Square);
        Self { button }
    }
}

impl Clickable for CellControl {
    fn on_click(
        self,
        handler: impl Fn(&gpui::ClickEvent, &mut Window, &mut App) + 'static,
    ) -> Self {
        let button = self.button.on_click(handler);
        Self { button }
    }

    fn cursor_style(self, _cursor_style: gpui::CursorStyle) -> Self {
        self
    }
}

/// A notebook cell
#[derive(Clone)]
pub enum Cell {
    Code(Entity<CodeCell>),
    Markdown(Entity<MarkdownCell>),
    Raw(Entity<RawCell>),
}

pub(crate) enum MovementDirection {
    Start,
    End,
}

fn convert_outputs(
    outputs: &Vec<nbformat::v4::Output>,
    window: &mut Window,
    cx: &mut App,
) -> Vec<Output> {
    outputs
        .iter()
        .map(|output| match output {
            nbformat::v4::Output::Stream { text, .. } => Output::Stream {
                content: cx.new(|cx| TerminalOutput::from(&text.0, window, cx)),
            },
            nbformat::v4::Output::DisplayData(display_data) => {
                Output::new(&display_data.data, None, window, cx)
            }
            nbformat::v4::Output::ExecuteResult(execute_result) => {
                Output::new(&execute_result.data, None, window, cx)
            }
            nbformat::v4::Output::Error(error) => Output::ErrorOutput(ErrorView {
                ename: error.ename.clone(),
                evalue: error.evalue.clone(),
                traceback: cx
                    .new(|cx| TerminalOutput::from(&error.traceback.join("\n"), window, cx)),
            }),
        })
        .collect()
}

impl Cell {
    pub fn id(&self, cx: &App) -> CellId {
        match self {
            Cell::Code(code_cell) => code_cell.read(cx).id().clone(),
            Cell::Markdown(markdown_cell) => markdown_cell.read(cx).id().clone(),
            Cell::Raw(raw_cell) => raw_cell.read(cx).id().clone(),
        }
    }

    pub fn current_source(&self, cx: &App) -> String {
        match self {
            Cell::Code(code_cell) => code_cell.read(cx).current_source(cx),
            Cell::Markdown(markdown_cell) => markdown_cell.read(cx).current_source(cx),
            Cell::Raw(raw_cell) => raw_cell.read(cx).source.clone(),
        }
    }

    pub fn to_nbformat_cell(&self, cx: &App) -> nbformat::v4::Cell {
        match self {
            Cell::Code(code_cell) => code_cell.read(cx).to_nbformat_cell(cx),
            Cell::Markdown(markdown_cell) => markdown_cell.read(cx).to_nbformat_cell(cx),
            Cell::Raw(raw_cell) => raw_cell.read(cx).to_nbformat_cell(),
        }
    }

    pub fn is_dirty(&self, cx: &App) -> bool {
        match self {
            Cell::Code(code_cell) => code_cell.read(cx).is_dirty(cx),
            Cell::Markdown(markdown_cell) => markdown_cell.read(cx).is_dirty(cx),
            Cell::Raw(_) => false,
        }
    }

    pub fn load(
        cell: &nbformat::v4::Cell,
        languages: &Arc<LanguageRegistry>,
        notebook_language: Shared<Task<Option<Arc<Language>>>>,
        window: &mut Window,
        cx: &mut App,
    ) -> Self {
        match cell {
            nbformat::v4::Cell::Markdown {
                id,
                metadata,
                source,
                ..
            } => {
                let source = source.concat();

                let entity = cx.new(|cx| {
                    MarkdownCell::new(
                        id.clone(),
                        metadata.clone(),
                        source,
                        languages.clone(),
                        window,
                        cx,
                    )
                });

                Cell::Markdown(entity)
            }
            nbformat::v4::Cell::Code {
                id,
                metadata,
                execution_count,
                source,
                outputs,
            } => {
                let text = source.concat();
                let outputs = convert_outputs(outputs, window, cx);

                Cell::Code(cx.new(|cx| {
                    CodeCell::new(
                        CellSource::Existing {
                            execution_count: *execution_count,
                            outputs,
                        },
                        id.clone(),
                        metadata.clone(),
                        text,
                        notebook_language,
                        window,
                        cx,
                    )
                }))
            }
            nbformat::v4::Cell::Raw {
                id,
                metadata,
                source,
            } => Cell::Raw(cx.new(|_| RawCell {
                id: id.clone(),
                metadata: metadata.clone(),
                source: source.concat(),
                selected: false,
                cell_position: None,
            })),
        }
    }

    pub(crate) fn move_to(&self, direction: MovementDirection, window: &mut Window, cx: &mut App) {
        fn move_in_editor(
            editor: &Entity<Editor>,
            direction: MovementDirection,
            window: &mut Window,
            cx: &mut App,
        ) {
            editor.update(cx, |editor, cx| {
                match direction {
                    MovementDirection::Start => {
                        editor.move_to_beginning(&Default::default(), window, cx);
                    }
                    MovementDirection::End => {
                        editor.move_to_end(&Default::default(), window, cx);
                    }
                }
                editor.focus_handle(cx).focus(window, cx);
            })
        }

        match self {
            Cell::Code(cell) => {
                cell.update(cx, |cell, cx| {
                    move_in_editor(&cell.editor, direction, window, cx)
                });
            }
            Cell::Markdown(cell) => {
                cell.update(cx, |cell, cx| {
                    cell.set_editing(true);
                    move_in_editor(&cell.editor, direction, window, cx);

                    cx.notify();
                });
            }
            _ => {}
        }
    }

    pub(crate) fn editor<'a>(&'a self, cx: &'a App) -> Option<&'a Entity<Editor>> {
        match self {
            Cell::Code(cell) => Some(cell.read(cx).editor()),
            Cell::Markdown(cell) => Some(cell.read(cx).editor()),
            _ => None,
        }
    }
}

pub trait RenderableCell: Render {
    const CELL_TYPE: CellType;

    fn id(&self) -> &CellId;
    fn cell_type(&self) -> CellType;
    fn metadata(&self) -> &CellMetadata;
    fn source(&self) -> &String;
    fn selected(&self) -> bool;
    fn set_selected(&mut self, selected: bool) -> &mut Self;
    fn selected_bg_color(&self, _window: &mut Window, cx: &mut Context<Self>) -> Hsla {
        if self.selected() {
            let mut color = cx.theme().colors().element_hover;
            color.fade_out(0.5);
            color
        } else {
            // Not sure if this is correct, previous was TODO: this is wrong
            gpui::transparent_black()
        }
    }
    fn control(&self, _window: &mut Window, _cx: &mut Context<Self>) -> Option<CellControl> {
        None
    }

    fn cell_position_spacer(
        &self,
        is_first: bool,
        _window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Option<impl IntoElement> {
        let cell_position = self.cell_position();

        if (cell_position == Some(&CellPosition::First) && is_first)
            || (cell_position == Some(&CellPosition::Last) && !is_first)
        {
            Some(div().flex().w_full().h(DynamicSpacing::Base12.px(cx)))
        } else {
            None
        }
    }

    /// The indicator bar at the far-left edge of the gutter (VS Code style):
    /// an accent bar on the selected cell, a grey bar on the hovered cell,
    /// nothing otherwise. Kept at the edge so it never clips the gutter
    /// controls.
    fn gutter_indicator_bar(&self, cx: &mut Context<Self>) -> Div {
        let is_selected = self.selected();
        div()
            .absolute()
            .left_0()
            .top_0()
            .h_full()
            .w(px(3.))
            .rounded_full()
            .when(is_selected, |this| this.bg(cx.theme().colors().icon_accent))
            .when(!is_selected, |this| {
                this.group_hover(CELL_HOVER_GROUP, |style| {
                    style.bg(cx.theme().colors().border)
                })
            })
    }

    fn gutter(&self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let is_selected = self.selected();
        div()
            .relative()
            .h_full()
            .w(px(GUTTER_WIDTH))
            .child(self.gutter_indicator_bar(cx))
            .when_some(self.control(window, cx), |this, control| {
                this.child(
                    div()
                        .absolute()
                        .top(px(CODE_BLOCK_INSET - 2.0))
                        // nudged right of the accent bar so the control reads
                        // centered in the gutter rather than hugging the bar
                        .left(px(7.))
                        .flex()
                        .flex_none()
                        .w(px(GUTTER_WIDTH - 7.0))
                        .h(px(GUTTER_WIDTH + 12.0))
                        .items_center()
                        .justify_center()
                        // VS Code style: the control only shows on the
                        // selected or hovered cell
                        .when(!is_selected, |this| {
                            this.invisible()
                                .group_hover(CELL_HOVER_GROUP, |style| style.visible())
                        })
                        .child(control.button),
                )
            })
    }

    fn cell_position(&self) -> Option<&CellPosition>;
    fn set_cell_position(&mut self, position: CellPosition) -> &mut Self;
}

pub trait RunnableCell: RenderableCell {
    fn execution_count(&self) -> Option<i32>;
    fn set_execution_count(&mut self, count: i32) -> &mut Self;
    fn run(&mut self, window: &mut Window, cx: &mut Context<Self>) -> ();
}

pub struct MarkdownCell {
    id: CellId,
    metadata: CellMetadata,
    image_cache: Entity<RetainAllImageCache>,
    source: String,
    editor: Entity<Editor>,
    markdown: Entity<Markdown>,
    editing: bool,
    selected: bool,
    cell_position: Option<CellPosition>,
    _editor_subscription: gpui::Subscription,
}

impl EventEmitter<MarkdownCellEvent> for MarkdownCell {}
impl EventEmitter<CellEvent> for MarkdownCell {}
impl EventEmitter<CellEvent> for RawCell {}

impl MarkdownCell {
    pub fn new(
        id: CellId,
        metadata: CellMetadata,
        source: String,
        languages: Arc<LanguageRegistry>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Self {
        let buffer = cx.new(|cx| Buffer::local(source.clone(), cx));
        let multi_buffer = cx.new(|cx| MultiBuffer::singleton(buffer.clone(), cx));

        let markdown_language = languages.language_for_name("Markdown");
        cx.spawn_in(window, async move |_this, cx| {
            if let Some(markdown) = markdown_language.await.log_err() {
                buffer.update(cx, |buffer, cx| {
                    buffer.set_language(Some(markdown), cx);
                });
            }
        })
        .detach();

        let editor = cx.new(|cx| {
            let mut editor = Editor::new(
                EditorMode::Full {
                    scale_ui_elements_with_buffer_font_size: false,
                    show_active_line_background: false,
                    sizing_behavior: SizingBehavior::SizeByContent,
                },
                multi_buffer,
                None,
                window,
                cx,
            );

            editor.set_show_gutter(false, cx);
            editor.set_use_modal_editing(true);
            editor.disable_mouse_wheel_zoom();
            editor.disable_scrollbars_and_minimap(window, cx);
            editor
        });

        let markdown = cx.new(|cx| Markdown::new(source.clone().into(), None, None, cx));

        let editor_subscription =
            cx.subscribe(&editor, move |this, _editor, event, cx| match event {
                editor::EditorEvent::Blurred => {
                    if this.editing {
                        this.editing = false;
                        cx.emit(MarkdownCellEvent::FinishedEditing);
                        cx.notify();
                    }
                }
                _ => {}
            });

        let start_editing = source.is_empty();
        Self {
            id,
            metadata,
            image_cache: RetainAllImageCache::new(cx),
            source,
            editor,
            markdown,
            editing: start_editing,
            selected: false,
            cell_position: None,
            _editor_subscription: editor_subscription,
        }
    }

    pub fn editor(&self) -> &Entity<Editor> {
        &self.editor
    }

    pub fn current_source(&self, cx: &App) -> String {
        let editor = self.editor.read(cx);
        let buffer = editor.buffer().read(cx);
        buffer
            .as_singleton()
            .map(|b| b.read(cx).text())
            .unwrap_or_default()
    }

    pub fn is_dirty(&self, cx: &App) -> bool {
        self.editor.read(cx).buffer().read(cx).is_dirty(cx)
    }

    pub fn to_nbformat_cell(&self, cx: &App) -> nbformat::v4::Cell {
        let source = self.current_source(cx);
        let source_lines: Vec<String> = source.lines().map(|l| format!("{}\n", l)).collect();

        nbformat::v4::Cell::Markdown {
            id: self.id.clone(),
            metadata: self.metadata.clone(),
            source: source_lines,
            attachments: None,
        }
    }

    pub fn is_editing(&self) -> bool {
        self.editing
    }

    pub fn set_editing(&mut self, editing: bool) {
        self.editing = editing;
    }

    pub fn reparse_markdown(&mut self, cx: &mut Context<Self>) {
        let editor = self.editor.read(cx);
        let buffer = editor.buffer().read(cx);
        let source = buffer
            .as_singleton()
            .map(|b| b.read(cx).text())
            .unwrap_or_default();

        self.source = source.clone();
        self.markdown.update(cx, |markdown, cx| {
            markdown.reset(source.into(), cx);
        });
    }

    /// Called when user presses Shift+Enter or Ctrl+Enter while editing.
    /// Finishes editing and signals to move to the next cell.
    pub fn run(&mut self, cx: &mut Context<Self>) {
        if self.editing {
            self.editing = false;
            cx.emit(MarkdownCellEvent::FinishedEditing);
            cx.emit(MarkdownCellEvent::Run(self.id.clone()));
            cx.notify();
        }
    }
}

impl RenderableCell for MarkdownCell {
    const CELL_TYPE: CellType = CellType::Markdown;

    fn id(&self) -> &CellId {
        &self.id
    }

    fn cell_type(&self) -> CellType {
        CellType::Markdown
    }

    fn metadata(&self) -> &CellMetadata {
        &self.metadata
    }

    fn source(&self) -> &String {
        &self.source
    }

    fn selected(&self) -> bool {
        self.selected
    }

    fn set_selected(&mut self, selected: bool) -> &mut Self {
        self.selected = selected;
        self
    }

    fn control(&self, _window: &mut Window, _: &mut Context<Self>) -> Option<CellControl> {
        None
    }

    fn cell_position(&self) -> Option<&CellPosition> {
        self.cell_position.as_ref()
    }

    fn set_cell_position(&mut self, cell_position: CellPosition) -> &mut Self {
        self.cell_position = Some(cell_position);
        self
    }
}

impl Render for MarkdownCell {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        // If editing, show the editor
        if self.editing {
            return v_flex()
                .size_full()
                .group(CELL_HOVER_GROUP)
                .capture_any_mouse_down(cx.listener(|this, event, _window, cx| {
                    if let Some(shift) = selection_modifiers(event) {
                        cx.emit(CellEvent::ModifiedClick {
                            id: this.id.clone(),
                            shift,
                        });
                        cx.stop_propagation();
                    }
                }))
                .children(self.cell_position_spacer(true, window, cx))
                .child(
                    h_flex()
                        .w_full()
                        .pr_6()
                        .rounded_xs()
                        .items_start()
                        .gap(DynamicSpacing::Base08.rems(cx))
                        .bg(self.selected_bg_color(window, cx))
                        .child(self.gutter(window, cx))
                        .child(
                            div()
                                .key_context("NotebookCellEditor")
                                .flex_1()
                                .p_3()
                                .bg(cx.theme().colors().editor_background)
                                .rounded_sm()
                                .child(self.editor.clone())
                                .on_mouse_down(
                                    gpui::MouseButton::Left,
                                    cx.listener(|_this, _event, _window, _cx| {
                                        // Prevent the click from propagating
                                    }),
                                ),
                        ),
                )
                .children(self.cell_position_spacer(false, window, cx));
        }

        // Preview mode - show rendered markdown

        let style = MarkdownStyle::themed(MarkdownFont::Preview, window, cx);

        v_flex()
            .size_full()
            .group(CELL_HOVER_GROUP)
            .capture_any_mouse_down(cx.listener(|this, event, _window, cx| {
                if let Some(shift) = selection_modifiers(event) {
                    cx.emit(CellEvent::ModifiedClick {
                        id: this.id.clone(),
                        shift,
                    });
                    cx.stop_propagation();
                }
            }))
            .children(self.cell_position_spacer(true, window, cx))
            .child(
                h_flex()
                    .w_full()
                    .pr_6()
                    .rounded_xs()
                    .items_start()
                    .gap(DynamicSpacing::Base08.rems(cx))
                    .bg(self.selected_bg_color(window, cx))
                    .child(self.gutter(window, cx))
                    .child(
                        v_flex()
                            .image_cache(self.image_cache.clone())
                            .id("markdown-content")
                            .size_full()
                            .flex_1()
                            .p_3()
                            .font_ui(cx)
                            .text_size(TextSize::Default.rems(cx))
                            .cursor_pointer()
                            .on_click(cx.listener(|this, _event, window, cx| {
                                this.editing = true;
                                window.focus(&this.editor.focus_handle(cx), cx);
                                cx.notify();
                            }))
                            .child(MarkdownElement::new(self.markdown.clone(), style)),
                    ),
            )
            .children(self.cell_position_spacer(false, window, cx))
    }
}

/// Lifecycle of a code cell's execution. `Pending` (queued, waiting for the
/// kernel to reach it) is distinct from `Running` (the kernel is actively
/// executing it) so queued cells don't show a running spinner or accrue the
/// wait time behind a long-running cell. Three terminal states: `Finished`
/// (ran to completion, ✓), `Failed` (ran and raised, red ✕ — VS Code style),
/// and `Cancelled` (interrupt/restart/abort before completion, muted ✕) so a
/// cell that never ran doesn't get a completed tick.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum CellExecutionStatus {
    #[default]
    Idle,
    Pending,
    Running,
    Finished,
    Failed,
    Cancelled,
}

pub struct CodeCell {
    id: CellId,
    metadata: CellMetadata,
    execution_count: Option<i32>,
    source: String,
    editor: Entity<editor::Editor>,
    outputs: Vec<Output>,
    selected: bool,
    cell_position: Option<CellPosition>,
    _language_task: Task<()>,
    execution_start_time: Option<Instant>,
    /// When the execute request was dispatched to the kernel. Used as a timing
    /// fallback for very fast cells whose shell `ExecuteReply` beats the iopub
    /// `ExecuteInput` (so `execution_start_time` never gets set); without it
    /// such cells finish showing a ✓ but no duration.
    submitted_at: Option<Instant>,
    execution_duration: Option<Duration>,
    execution_status: CellExecutionStatus,
    /// Repeating notify task that keeps the live elapsed-time label ticking
    /// while the cell is Running. Dropped (cancelling it) when the run ends.
    _run_timer: Option<Task<()>>,
    /// Input (code editor) collapsed. Persisted to the .ipynb as
    /// `metadata.jupyter.source_hidden` (VS Code / Jupyter compatible).
    source_collapsed: bool,
    /// Outputs collapsed. Persisted as `metadata.jupyter.outputs_hidden`.
    outputs_collapsed: bool,
}

impl EventEmitter<CellEvent> for CodeCell {}

pub(super) enum CellSource {
    /// Crate a new empty cell
    None,
    /// Backed by an existing notebook cell
    Existing {
        execution_count: Option<i32>,
        outputs: Vec<Output>,
    },
}

impl CellSource {
    fn into_outputs(self) -> (Option<i32>, Vec<Output>) {
        match self {
            CellSource::Existing {
                execution_count,
                outputs,
            } => (execution_count, outputs),
            CellSource::None => Default::default(),
        }
    }
}

impl CodeCell {
    pub(super) fn new(
        cell_source: CellSource,
        id: CellId,
        metadata: CellMetadata,
        source: String,
        notebook_language: Shared<Task<Option<Arc<Language>>>>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Self {
        let buffer = cx.new(|cx| Buffer::local(source.clone(), cx));
        let multi_buffer = cx.new(|cx| MultiBuffer::singleton(buffer.clone(), cx));

        let editor = cx.new(|cx| {
            let mut editor = Editor::new(
                EditorMode::Full {
                    scale_ui_elements_with_buffer_font_size: false,
                    show_active_line_background: false,
                    sizing_behavior: SizingBehavior::SizeByContent,
                },
                multi_buffer,
                None,
                window,
                cx,
            );

            editor.disable_mouse_wheel_zoom();
            editor.disable_scrollbars_and_minimap(window, cx);
            editor.set_text(source.clone(), window, cx);
            editor.set_show_gutter(false, cx);
            editor.set_use_modal_editing(true);
            editor
        });

        let language_task = cx.spawn_in(window, async move |_this, cx| {
            let language = notebook_language.await;
            buffer.update(cx, |buffer, cx| {
                buffer.set_language(language.clone(), cx);
            });
        });

        let (execution_count, outputs) = cell_source.into_outputs();

        let source_collapsed = metadata
            .jupyter
            .as_ref()
            .and_then(|jupyter| jupyter.source_hidden)
            .unwrap_or(false);
        let outputs_collapsed = metadata
            .jupyter
            .as_ref()
            .and_then(|jupyter| jupyter.outputs_hidden)
            .unwrap_or(false);

        Self {
            id,
            metadata,
            execution_count,
            source,
            editor,
            outputs,
            selected: false,
            cell_position: None,
            execution_start_time: None,
            submitted_at: None,
            execution_duration: None,
            execution_status: CellExecutionStatus::Idle,
            _run_timer: None,
            source_collapsed,
            outputs_collapsed,
            _language_task: language_task,
        }
    }

    pub fn set_language(&mut self, language: Option<Arc<Language>>, cx: &mut Context<Self>) {
        self.editor.update(cx, |editor, cx| {
            editor.buffer().update(cx, |buffer, cx| {
                if let Some(buffer) = buffer.as_singleton() {
                    buffer.update(cx, |buffer, cx| {
                        buffer.set_language(language, cx);
                    });
                }
            });
        });
    }

    pub fn editor(&self) -> &Entity<editor::Editor> {
        &self.editor
    }

    pub fn current_source(&self, cx: &App) -> String {
        let editor = self.editor.read(cx);
        let buffer = editor.buffer().read(cx);
        buffer
            .as_singleton()
            .map(|b| b.read(cx).text())
            .unwrap_or_default()
    }

    pub fn is_dirty(&self, cx: &App) -> bool {
        self.editor.read(cx).buffer().read(cx).is_dirty(cx)
    }

    pub fn to_nbformat_cell(&self, cx: &App) -> nbformat::v4::Cell {
        let source = self.current_source(cx);
        let source_lines: Vec<String> = source.lines().map(|l| format!("{}\n", l)).collect();

        let outputs = self.outputs_to_nbformat(cx);

        nbformat::v4::Cell::Code {
            id: self.id.clone(),
            metadata: self.metadata_with_visibility(),
            execution_count: self.execution_count,
            source: source_lines,
            outputs,
        }
    }

    /// The cell metadata with the current collapse state written into
    /// `jupyter.source_hidden` / `jupyter.outputs_hidden` (VS Code / Jupyter
    /// compatible). Expanded state omits the keys to keep the file clean,
    /// preserving any other `jupyter.*` fields.
    fn metadata_with_visibility(&self) -> CellMetadata {
        let mut metadata = self.metadata.clone();
        let mut jupyter =
            metadata
                .jupyter
                .take()
                .unwrap_or(nbformat::v4::JupyterCellMetadata {
                    source_hidden: None,
                    outputs_hidden: None,
                    additional: Default::default(),
                });
        jupyter.source_hidden = self.source_collapsed.then_some(true);
        jupyter.outputs_hidden = self.outputs_collapsed.then_some(true);
        let keep = jupyter.source_hidden.is_some()
            || jupyter.outputs_hidden.is_some()
            || !jupyter.additional.is_empty();
        metadata.jupyter = keep.then_some(jupyter);
        metadata
    }

    fn toggle_source_collapsed(&mut self, cx: &mut Context<Self>) {
        self.source_collapsed = !self.source_collapsed;
        cx.emit(CellEvent::MetadataChanged(self.id.clone()));
        cx.notify();
    }

    fn toggle_outputs_collapsed(&mut self, cx: &mut Context<Self>) {
        self.outputs_collapsed = !self.outputs_collapsed;
        cx.emit(CellEvent::MetadataChanged(self.id.clone()));
        cx.notify();
    }

    fn outputs_to_nbformat(&self, cx: &App) -> Vec<nbformat::v4::Output> {
        self.outputs
            .iter()
            .filter_map(|output| output.to_nbformat(cx))
            .collect()
    }

    pub fn has_outputs(&self) -> bool {
        !self.outputs.is_empty()
    }

    pub fn clear_outputs(&mut self) {
        // Only the outputs — NOT the recorded duration. `begin_running` clears
        // outputs at the start of every run (including a late iopub
        // `execute_input` that arrives AFTER a fast cell's shell `execute_reply`
        // already finished it); wiping the duration here would erase the time
        // `finish_execution` just computed, leaving a ✓ with no time. The
        // duration is reset explicitly by `mark_pending` / `begin_running` /
        // `cancel_execution` when a run genuinely (re)starts.
        self.outputs.clear();
    }

    /// Concatenated plain text of the cell's text-bearing outputs (stdout/plain
    /// results and error tracebacks), for the "Copy Output" menu action.
    fn outputs_as_text(&self, cx: &App) -> String {
        let mut parts = Vec::new();
        for output in &self.outputs {
            match output {
                Output::Plain { content, .. } | Output::Stream { content } => {
                    parts.push(content.read(cx).full_text(cx));
                }
                Output::ErrorOutput(error_view) => {
                    parts.push(error_view.traceback.read(cx).full_text(cx));
                }
                _ => {}
            }
        }
        parts.join("\n")
    }

    /// Mark the cell as queued for execution: the previous tick/time make way
    /// for a pending indicator, but the OUTPUT is kept until the cell actually
    /// re-executes (see `begin_running`). The timer does NOT start here — a
    /// pending cell must not accrue the wait time behind earlier cells.
    pub fn mark_pending(&mut self) {
        self.execution_status = CellExecutionStatus::Pending;
        self.execution_start_time = None;
        self.submitted_at = None;
        self.execution_duration = None;
        self._run_timer = None;
    }

    /// Record the instant the execute request was dispatched to the kernel.
    /// Serves as a timing fallback if the iopub `ExecuteInput` (which sets the
    /// precise `execution_start_time`) never arrives before the cell finishes.
    pub fn record_submitted(&mut self) {
        self.submitted_at = Some(Instant::now());
    }

    /// The kernel started executing this cell (its `execute_input` arrived):
    /// drop the previous run's outputs so the new ones replace them, and — if
    /// the run hasn't already resolved — start the timer and go Running.
    ///
    /// The shell `ExecuteReply` and the iopub `ExecuteInput` travel on separate
    /// channels, so for a fast cell the reply (which finishes the cell) can
    /// arrive BEFORE the input. In that case the cell is already Finished, and
    /// we must NOT resurrect it into Running (which left cells stuck spinning).
    /// The input still precedes this run's outputs on iopub, so clearing here
    /// is safe either way.
    pub fn begin_running(&mut self, cx: &mut Context<Self>) {
        self.clear_outputs();
        if matches!(
            self.execution_status,
            CellExecutionStatus::Finished
                | CellExecutionStatus::Failed
                | CellExecutionStatus::Cancelled
        ) {
            return;
        }
        self.execution_status = CellExecutionStatus::Running;
        self.execution_start_time = Some(Instant::now());
        self.execution_duration = None;
        // Tick the live elapsed-time label while running. The task ends itself
        // when the status leaves Running, and is dropped (cancelled) by the
        // terminal transitions as well.
        self._run_timer = Some(cx.spawn(async move |this, cx| {
            loop {
                cx.background_executor()
                    .timer(Duration::from_millis(100))
                    .await;
                let still_running = this
                    .update(cx, |cell, cx| {
                        let running = cell.execution_status == CellExecutionStatus::Running;
                        if running {
                            cx.notify();
                        }
                        running
                    })
                    .unwrap_or(false);
                if !still_running {
                    break;
                }
            }
        }));
    }

    pub fn finish_execution(&mut self) {
        self.complete_execution(CellExecutionStatus::Finished);
    }

    /// The cell ran and raised (its `ExecuteReply` came back with an Error
    /// status): terminal like `finish_execution`, but marked as a failure
    /// (red ✕) instead of a completed ✓.
    pub fn fail_execution(&mut self) {
        self.complete_execution(CellExecutionStatus::Failed);
    }

    fn complete_execution(&mut self, final_status: CellExecutionStatus) {
        self._run_timer = None;
        // An interrupted cell was already marked Cancelled (KeyboardInterrupt
        // on iopub); its ExecuteReply — which reports Error for an interrupt —
        // must not flip it to a ✓ or a red ✕.
        if self.execution_status == CellExecutionStatus::Cancelled {
            return;
        }
        // Prefer the precise start (iopub `execute_input`); fall back to the
        // submit time for fast cells whose reply beat their input, so they
        // still show a (near-exact) duration rather than none.
        if let Some(start_time) = self.execution_start_time.take().or(self.submitted_at.take()) {
            self.execution_duration = Some(start_time.elapsed());
        }
        self.submitted_at = None;
        self.execution_status = final_status;
    }

    /// A queued run was abandoned before ever reaching a kernel (the kernel
    /// picker was dismissed): back to Idle, as if the run had not been
    /// requested. Unlike `cancel_execution` there is no Cancelled marker —
    /// nothing was actually cancelled mid-flight (bug #28).
    pub fn reset_execution_status(&mut self) {
        if self.execution_status == CellExecutionStatus::Pending {
            self.execution_status = CellExecutionStatus::Idle;
            self.execution_start_time = None;
            self.submitted_at = None;
            self.execution_duration = None;
            self._run_timer = None;
        }
    }

    /// The cell never completed (interrupt/restart/kernel loss/aborted batch):
    /// no completed tick and no bogus time.
    pub fn cancel_execution(&mut self) {
        if matches!(
            self.execution_status,
            CellExecutionStatus::Pending | CellExecutionStatus::Running
        ) {
            self.execution_status = CellExecutionStatus::Cancelled;
            self.execution_start_time = None;
            self.submitted_at = None;
            self.execution_duration = None;
            self._run_timer = None;
        }
    }

    pub fn is_executing(&self) -> bool {
        self.execution_status == CellExecutionStatus::Running
    }

    /// Running or queued: an execution is in flight for this cell.
    pub fn is_execution_in_flight(&self) -> bool {
        matches!(
            self.execution_status,
            CellExecutionStatus::Pending | CellExecutionStatus::Running
        )
    }

    pub fn execution_status(&self) -> CellExecutionStatus {
        self.execution_status
    }

    /// Forget the kernel-session execution number (`In [N]`). Used on kernel
    /// restart: the new session's counter starts at 1, so the old numbers are
    /// stale.
    pub fn reset_execution_count(&mut self) {
        self.execution_count = None;
    }

    /// Displays a kernel-level failure (e.g. the kernel failed to launch because
    /// Python is not installed) as an error output on this cell, so the user gets
    /// feedback instead of a spinner that never resolves.
    pub fn show_kernel_error(
        &mut self,
        error_message: &str,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.outputs.push(Output::ErrorOutput(ErrorView {
            ename: "Kernel Error".to_string(),
            evalue: "cell could not be executed".to_string(),
            traceback: cx.new(|cx| TerminalOutput::from(error_message, window, cx)),
        }));
        self.execution_start_time = None;
        self.submitted_at = None;
        // The cell never ran — no completed tick, no time.
        self.execution_status = CellExecutionStatus::Cancelled;
        cx.notify();
    }

    pub fn execution_duration(&self) -> Option<Duration> {
        self.execution_duration
    }

    /// The small status line for the cell's last/current run: spinner while
    /// running, a queued indicator while pending, ✓ + time when finished, a
    /// red ✕ + time when the cell raised, and a muted ✕ when cancelled.
    /// `None` for an idle cell.
    fn execution_status_element(&self, cx: &App) -> Option<AnyElement> {
        let label = |text: String, cx: &App| {
            div()
                .text_xs()
                .text_color(cx.theme().colors().text_muted)
                .child(text)
        };
        let element = match self.execution_status {
            CellExecutionStatus::Idle => return None,
            CellExecutionStatus::Pending => h_flex()
                .gap_1()
                .items_center()
                .child(
                    Icon::new(IconName::Clock)
                        .size(IconSize::XSmall)
                        .color(Color::Muted),
                )
                .child(label("Pending...".to_string(), cx)),
            CellExecutionStatus::Running => {
                // Live elapsed time, kept ticking by `_run_timer`'s notifies.
                let running_label = match self.execution_start_time {
                    Some(start_time) => {
                        format!("Running... {}", Self::format_duration(start_time.elapsed()))
                    }
                    None => "Running...".to_string(),
                };
                h_flex()
                    .gap_1()
                    .items_center()
                    .child(
                        Icon::new(IconName::ArrowCircle)
                            .size(IconSize::XSmall)
                            .color(Color::Warning)
                            .with_rotate_animation(2),
                    )
                    .child(label(running_label, cx))
            }
            CellExecutionStatus::Finished => h_flex()
                .gap_1()
                .items_center()
                .child(
                    Icon::new(IconName::Check)
                        .size(IconSize::XSmall)
                        .color(Color::Success),
                )
                .when_some(
                    self.execution_duration.map(Self::format_duration),
                    |this, duration_text| this.child(label(duration_text, cx)),
                ),
            CellExecutionStatus::Failed => h_flex()
                .gap_1()
                .items_center()
                .child(
                    Icon::new(IconName::XCircle)
                        .size(IconSize::XSmall)
                        .color(Color::Error),
                )
                .when_some(
                    self.execution_duration.map(Self::format_duration),
                    |this, duration_text| this.child(label(duration_text, cx)),
                ),
            CellExecutionStatus::Cancelled => h_flex()
                .gap_1()
                .items_center()
                .child(
                    Icon::new(IconName::XCircle)
                        .size(IconSize::XSmall)
                        .color(Color::Muted),
                )
                .child(label("Cancelled".to_string(), cx)),
        };
        Some(element.into_any_element())
    }

    fn format_duration(duration: Duration) -> String {
        let total_secs = duration.as_secs_f64();
        if total_secs < 1.0 {
            format!("{:.0}ms", duration.as_millis())
        } else if total_secs < 60.0 {
            format!("{:.1}s", total_secs)
        } else {
            let minutes = (total_secs / 60.0).floor() as u64;
            let secs = total_secs % 60.0;
            format!("{}m {:.1}s", minutes, secs)
        }
    }

    /// A floating toolbar of the most common cell actions, shown in the cell's
    /// top-right when it is selected or hovered. Each button emits a
    /// `CellEvent::ToolbarAction`; the notebook selects this cell and then runs
    /// the matching action, so the buttons are just a discoverable surface over
    /// the existing keyboard/menu actions.
    fn cell_toolbar(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let button = |name: &'static str, icon: IconName, action: CellToolbarAction| {
            IconButton::new(name, icon)
                .icon_size(IconSize::Small)
                .shape(IconButtonShape::Square)
                .on_click(cx.listener(move |this, _, _window, cx| {
                    cx.emit(CellEvent::ToolbarAction(this.id.clone(), action));
                }))
        };

        let collapse_label: SharedString = if self.source_collapsed {
            "Expand Input".into()
        } else {
            "Collapse Input".into()
        };

        h_flex()
            .gap_0p5()
            .p_0p5()
            .rounded_md()
            .border_1()
            .border_color(cx.theme().colors().border)
            .bg(cx.theme().colors().element_background)
            .child(
                IconButton::new(
                    "cell-collapse-input",
                    if self.source_collapsed {
                        IconName::ChevronRight
                    } else {
                        IconName::ChevronDown
                    },
                )
                .icon_size(IconSize::Small)
                .shape(IconButtonShape::Square)
                .on_click(cx.listener(|this, _, _window, cx| {
                    this.toggle_source_collapsed(cx);
                }))
                .tooltip(move |window, cx| Tooltip::text(collapse_label.clone())(window, cx)),
            )
            .child(
                button(
                    "cell-run-above",
                    IconName::ArrowUp,
                    CellToolbarAction::RunAbove,
                )
                .tooltip(|_window, cx| Tooltip::for_action("Run cells above", &RunCellsAbove, cx)),
            )
            .child(
                button(
                    "cell-run-below",
                    IconName::ArrowDown,
                    CellToolbarAction::RunBelow,
                )
                .tooltip(|_window, cx| {
                    Tooltip::for_action("Run cell and below", &RunCellAndBelow, cx)
                }),
            )
            .child(
                button(
                    "cell-add-below",
                    IconName::Plus,
                    CellToolbarAction::AddBelow,
                )
                .tooltip(|_window, cx| Tooltip::for_action("Add cell below", &AddCellBelow, cx)),
            )
            .child(
                button("cell-delete", IconName::Trash, CellToolbarAction::Delete)
                    .tooltip(|_window, cx| Tooltip::for_action("Delete cell", &DeleteCell, cx)),
            )
    }

    pub fn handle_message(
        &mut self,
        message: &JupyterMessage,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        match &message.content {
            JupyterMessageContent::StreamContent(stream) => {
                self.outputs.push(Output::Stream {
                    content: cx.new(|cx| TerminalOutput::from(&stream.text, window, cx)),
                });
            }
            JupyterMessageContent::DisplayData(display_data) => {
                self.outputs
                    .push(Output::new(&display_data.data, None, window, cx));
            }
            JupyterMessageContent::ExecuteResult(execute_result) => {
                self.outputs
                    .push(Output::new(&execute_result.data, None, window, cx));
            }
            JupyterMessageContent::ExecuteInput(input) => {
                // The kernel started executing THIS cell: only now does it
                // become Running and start its timer (queued cells must not
                // accrue the wait behind earlier cells), and only now are the
                // previous outputs dropped.
                self.begin_running(cx);
                self.execution_count = serde_json::to_value(&input.execution_count)
                    .ok()
                    .and_then(|v| v.as_i64())
                    .map(|v| v as i32);
            }
            JupyterMessageContent::ExecuteReply(reply) => {
                match reply.status {
                    // A kernel aborts the requests queued behind an error or
                    // interrupt without executing them — those cells were
                    // never run, so they must not get a completed tick.
                    ReplyStatus::Aborted => self.cancel_execution(),
                    ReplyStatus::Error => {
                        // An interrupt's reply also reports Error
                        // (KeyboardInterrupt). The iopub error usually lands
                        // first and marks the cell Cancelled, but shell and
                        // iopub can reorder — recognize the interrupt from
                        // the reply itself so a user stop never shows as a
                        // red ✕ failure. Real errors → red ✕.
                        let interrupted = reply
                            .error
                            .as_ref()
                            .is_some_and(|error| error.ename == "KeyboardInterrupt");
                        if interrupted {
                            self.cancel_execution();
                        } else {
                            self.fail_execution();
                        }
                    }
                    _ => self.finish_execution(),
                }
            }
            JupyterMessageContent::ErrorOutput(error) => {
                // An interrupt shows as a KeyboardInterrupt error: the cell was
                // stopped, not completed — mark it Cancelled (muted ✕), keeping
                // the traceback visible. Real errors get the red ✕ via their
                // ExecuteReply's Error status (`fail_execution`).
                if error.ename == "KeyboardInterrupt" {
                    self.cancel_execution();
                }
                self.outputs.push(Output::ErrorOutput(ErrorView {
                    ename: error.ename.clone(),
                    evalue: error.evalue.clone(),
                    traceback: cx
                        .new(|cx| TerminalOutput::from(&error.traceback.join("\n"), window, cx)),
                }));
            }
            _ => {}
        }
        cx.notify();
    }

    pub fn gutter_output(&self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let is_selected = self.selected();
        div()
            .relative()
            .h_full()
            .w(px(GUTTER_WIDTH))
            // Selecting from the output gutter mirrors the input gutter.
            .on_mouse_down(
                gpui::MouseButton::Left,
                cx.listener(|this, _event, _window, cx| {
                    cx.emit(CellEvent::PlainClick { id: this.id.clone() });
                }),
            )
            .child(self.gutter_indicator_bar(cx))
            .when(self.has_outputs(), |this| {
                this.child(
                    div()
                        .absolute()
                        .top(px(CODE_BLOCK_INSET - 2.0))
                        // nudged right of the accent bar so the control reads
                        // centered in the gutter rather than hugging the bar
                        .left(px(7.))
                        .flex()
                        .flex_none()
                        .w(px(GUTTER_WIDTH - 7.0))
                        .h(px(GUTTER_WIDTH + 12.0))
                        .items_center()
                        .justify_center()
                        .when(!is_selected, |this| {
                            this.invisible()
                                .group_hover(CELL_HOVER_GROUP, |style| style.visible())
                        })
                        .child(
                            PopoverMenu::new("cell-output-menu")
                                .trigger_with_tooltip(
                                    IconButton::new("control", IconName::Ellipsis)
                                        .icon_size(IconSize::Small),
                                    Tooltip::text("Output options"),
                                )
                                .menu({
                                    let cell = cx.entity();
                                    move |window, cx| {
                                        let text = cell.read(cx).outputs_as_text(cx);
                                        let collapsed = cell.read(cx).outputs_collapsed;
                                        let cell = cell.clone();
                                        Some(ContextMenu::build(window, cx, move |menu, _, _| {
                                            menu.entry("Copy Output", None, {
                                                let text = text.clone();
                                                move |_, cx| {
                                                    cx.write_to_clipboard(
                                                        ClipboardItem::new_string(text.clone()),
                                                    );
                                                }
                                            })
                                            .separator()
                                            .entry(
                                                if collapsed {
                                                    "Expand Output"
                                                } else {
                                                    "Collapse Output"
                                                },
                                                None,
                                                {
                                                    let cell = cell.clone();
                                                    move |_, cx| {
                                                        cell.update(cx, |cell, cx| {
                                                            cell.toggle_outputs_collapsed(cx);
                                                        });
                                                    }
                                                },
                                            )
                                            .entry(
                                                "Clear Output",
                                                None,
                                                move |_, cx| {
                                                    cell.update(cx, |cell, cx| {
                                                        cell.clear_outputs();
                                                        cx.notify();
                                                    });
                                                },
                                            )
                                        }))
                                    }
                                }),
                        ),
                )
            })
    }
}

impl RenderableCell for CodeCell {
    const CELL_TYPE: CellType = CellType::Code;

    fn id(&self) -> &CellId {
        &self.id
    }

    fn cell_type(&self) -> CellType {
        CellType::Code
    }

    fn metadata(&self) -> &CellMetadata {
        &self.metadata
    }

    fn source(&self) -> &String {
        &self.source
    }

    fn control(&self, _window: &mut Window, cx: &mut Context<Self>) -> Option<CellControl> {
        // Running or queued: the button interrupts. Otherwise it runs.
        let control_type = if self.is_execution_in_flight() {
            CellControlType::StopCell
        } else if self.has_outputs() {
            CellControlType::RerunCell
        } else {
            CellControlType::RunCell
        };

        Some(
            CellControl::new(control_type.id(), control_type).on_click(cx.listener(
                move |this, _, window, cx| {
                    if this.is_execution_in_flight() {
                        // Scoped stop: the notebook interrupts this cell if it's
                        // running, or un-queues it if only pending, leaving the
                        // rest of a batch running.
                        cx.emit(CellEvent::Stop(this.id.clone()));
                    } else {
                        this.run(window, cx);
                    }
                },
            )),
        )
    }

    fn selected(&self) -> bool {
        self.selected
    }

    fn set_selected(&mut self, selected: bool) -> &mut Self {
        self.selected = selected;
        self
    }

    fn cell_position(&self) -> Option<&CellPosition> {
        self.cell_position.as_ref()
    }

    fn set_cell_position(&mut self, cell_position: CellPosition) -> &mut Self {
        self.cell_position = Some(cell_position);
        self
    }

    fn gutter(&self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let execution_count = self.execution_count;
        // The stop button on a running/queued cell must stay visible even when
        // the cell is neither selected nor hovered.
        let always_show_control = self.selected() || self.is_execution_in_flight();

        div()
            .relative()
            .h_full()
            .w(px(GUTTER_WIDTH))
            // A plain click on the gutter selects the cell in command mode
            // (the run button on top still runs — its click just also selects).
            // Kept off the editor so clicking the cell body enters edit mode.
            .on_mouse_down(
                gpui::MouseButton::Left,
                cx.listener(|this, _event, _window, cx| {
                    cx.emit(CellEvent::PlainClick { id: this.id.clone() });
                }),
            )
            .child(self.gutter_indicator_bar(cx))
            .when_some(self.control(window, cx), |this, control| {
                this.child(
                    v_flex()
                        .absolute()
                        .top(px(CODE_BLOCK_INSET - 2.0))
                        // nudged right of the accent bar so the control reads
                        // centered in the gutter rather than hugging the bar
                        .left(px(7.))
                        .w(px(GUTTER_WIDTH - 7.0))
                        .items_center()
                        .gap_0p5()
                        // VS Code style: a bare hovering run button, only shown
                        // on the selected or hovered cell (or while running)
                        .child(
                            div()
                                .when(!always_show_control, |this| {
                                    this.invisible()
                                        .group_hover(CELL_HOVER_GROUP, |style| style.visible())
                                })
                                .child(control.button),
                        )
                        // Jupyter-style execution number (`In [N]`): the count is
                        // the kernel's session-global execution counter, not a
                        // per-cell run tally.
                        .when_some(execution_count, |this, count| {
                            this.child(
                                div()
                                    .text_xs()
                                    .text_color(cx.theme().colors().text_muted)
                                    .child(format!("[{count}]")),
                            )
                        }),
                )
            })
    }
}

impl RunnableCell for CodeCell {
    fn run(&mut self, _window: &mut Window, cx: &mut Context<Self>) {
        cx.emit(CellEvent::Run(self.id.clone()));
    }

    fn execution_count(&self) -> Option<i32> {
        self.execution_count
            .and_then(|count| if count > 0 { Some(count) } else { None })
    }

    fn set_execution_count(&mut self, count: i32) -> &mut Self {
        self.execution_count = Some(count);
        self
    }
}

impl Render for CodeCell {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let output_max_height = ReplSettings::get_global(cx).output_max_height_lines;
        let output_max_height = if output_max_height > 0 {
            Some(window.line_height() * output_max_height as f32)
        } else {
            None
        };
        let output_max_width =
            plain::max_width_for_columns(ReplSettings::get_global(cx).max_columns, window, cx);
        // get the language from the editor's buffer
        let language_name = self
            .editor
            .read(cx)
            .buffer()
            .read(cx)
            .as_singleton()
            .and_then(|buffer| buffer.read(cx).language())
            .map(|lang| lang.name().to_string());

        let is_selected = self.selected();

        v_flex()
            .size_full()
            .group(CELL_HOVER_GROUP)
            .capture_any_mouse_down(cx.listener(|this, event, _window, cx| {
                if let Some(shift) = selection_modifiers(event) {
                    cx.emit(CellEvent::ModifiedClick {
                        id: this.id.clone(),
                        shift,
                    });
                    cx.stop_propagation();
                }
            }))
            // TODO: Move base cell render into trait impl so we don't have to repeat this
            .children(self.cell_position_spacer(true, window, cx))
            // Editor portion
            .child(
                h_flex()
                    .w_full()
                    .pr_6()
                    .rounded_xs()
                    .items_start()
                    .gap(DynamicSpacing::Base08.rems(cx))
                    .bg(self.selected_bg_color(window, cx))
                    .child(self.gutter(window, cx))
                    .child(
                        div().py_1p5().w_full().child(
                            div()
                                .relative()
                                .flex()
                                .flex_col()
                                .size_full()
                                .flex_1()
                                .py_3()
                                .px_5()
                                .rounded_lg()
                                .border_1()
                                .border_color(cx.theme().colors().border)
                                .bg(cx.theme().colors().editor_background)
                                .map(|this| {
                                    if self.source_collapsed {
                                        // Collapsed input: a one-line summary
                                        // of the source; click to expand.
                                        let first_line = self
                                            .current_source(cx)
                                            .lines()
                                            .next()
                                            .unwrap_or_default()
                                            .to_string();
                                        this.child(
                                            div()
                                                .id("collapsed-input")
                                                .w_full()
                                                .cursor_pointer()
                                                .text_color(cx.theme().colors().text_muted)
                                                .child(format!("{first_line} ⋯"))
                                                .on_click(cx.listener(|this, _, _window, cx| {
                                                    this.toggle_source_collapsed(cx);
                                                })),
                                        )
                                    } else {
                                        this.child(
                                            div()
                                                .key_context("NotebookCellEditor")
                                                .w_full()
                                                .child(self.editor.clone()),
                                        )
                                    }
                                })
                                // VS Code-style cell status bar: the execution
                                // status + time sit INSIDE the cell, in its
                                // bottom-left corner.
                                .when_some(
                                    self.execution_status_element(cx),
                                    |this, status_element| {
                                        this.child(
                                            h_flex().mt_2().justify_start().child(status_element),
                                        )
                                    },
                                )
                                // per-cell action toolbar in the top-right,
                                // shown when the cell is selected or hovered
                                .child(
                                    div()
                                        .absolute()
                                        .top_1()
                                        .right_2()
                                        .when(!is_selected, |this| {
                                            this.invisible()
                                                .group_hover(CELL_HOVER_GROUP, |style| {
                                                    style.visible()
                                                })
                                        })
                                        .child(self.cell_toolbar(cx)),
                                )
                                // lang badge in the bottom-right corner (moved
                                // out of the top-right to make room for the
                                // toolbar)
                                .when_some(language_name, |this, name| {
                                    this.child(
                                        div()
                                            .absolute()
                                            .bottom_1()
                                            .right_2()
                                            .px_2()
                                            .py_0p5()
                                            .rounded_md()
                                            .bg(cx.theme().colors().element_background.opacity(0.7))
                                            .text_xs()
                                            .text_color(cx.theme().colors().text_muted)
                                            .child(name),
                                    )
                                }),
                        ),
                    ),
            )
            .when(self.has_outputs(), |this| {
                this.child(
                    h_flex()
                        .w_full()
                        .pr_6()
                        .rounded_xs()
                        .items_start()
                        .gap(DynamicSpacing::Base08.rems(cx))
                        .bg(self.selected_bg_color(window, cx))
                        .child(self.gutter_output(window, cx))
                        .child(
                            div().py_1p5().w_full().child(
                                v_flex()
                                    .size_full()
                                    .flex_1()
                                    .py_3()
                                    .px_5()
                                    .rounded_lg()
                                    .border_1()
                                    .map(|this| {
                                        if self.outputs_collapsed {
                                            // Collapsed output: a slim
                                            // placeholder; click to expand.
                                            this.child(
                                                div()
                                                    .id("collapsed-output")
                                                    .w_full()
                                                    .cursor_pointer()
                                                    .text_color(cx.theme().colors().text_muted)
                                                    .text_size(TextSize::Small.rems(cx))
                                                    .child("Output collapsed ⋯")
                                                    .on_click(cx.listener(
                                                        |this, _, _window, cx| {
                                                            this.toggle_outputs_collapsed(cx);
                                                        },
                                                    )),
                                            )
                                        } else {
                                            this.child(
                                                div()
                                                    .id((
                                                        ElementId::from(self.id.to_string()),
                                                        "output-scroll",
                                                    ))
                                                    .w_full()
                                                    .when_some(output_max_width, |div, max_width| {
                                                        div.max_w(max_width).overflow_x_scroll()
                                                    })
                                                    .when_some(
                                                        output_max_height,
                                                        |div, max_height| {
                                                            div.max_h(max_height).overflow_y_scroll()
                                                        },
                                                    )
                                                    .children(self.outputs.iter().map(|output| {
                                                        div().children(output.content(window, cx))
                                                    })),
                                            )
                                        }
                                    }),
                            ),
                        ),
                )
            })
            // TODO: Move base cell render into trait impl so we don't have to repeat this
            .children(self.cell_position_spacer(false, window, cx))
    }
}

pub struct RawCell {
    id: CellId,
    metadata: CellMetadata,
    source: String,
    selected: bool,
    cell_position: Option<CellPosition>,
}

impl RawCell {
    pub fn to_nbformat_cell(&self) -> nbformat::v4::Cell {
        let source_lines: Vec<String> = self.source.lines().map(|l| format!("{}\n", l)).collect();

        nbformat::v4::Cell::Raw {
            id: self.id.clone(),
            metadata: self.metadata.clone(),
            source: source_lines,
        }
    }
}

impl RenderableCell for RawCell {
    const CELL_TYPE: CellType = CellType::Raw;

    fn id(&self) -> &CellId {
        &self.id
    }

    fn cell_type(&self) -> CellType {
        CellType::Raw
    }

    fn metadata(&self) -> &CellMetadata {
        &self.metadata
    }

    fn source(&self) -> &String {
        &self.source
    }

    fn selected(&self) -> bool {
        self.selected
    }

    fn set_selected(&mut self, selected: bool) -> &mut Self {
        self.selected = selected;
        self
    }

    fn cell_position(&self) -> Option<&CellPosition> {
        self.cell_position.as_ref()
    }

    fn set_cell_position(&mut self, cell_position: CellPosition) -> &mut Self {
        self.cell_position = Some(cell_position);
        self
    }
}

impl Render for RawCell {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        v_flex()
            .size_full()
            .group(CELL_HOVER_GROUP)
            .capture_any_mouse_down(cx.listener(|this, event, _window, cx| {
                if let Some(shift) = selection_modifiers(event) {
                    cx.emit(CellEvent::ModifiedClick {
                        id: this.id.clone(),
                        shift,
                    });
                    cx.stop_propagation();
                }
            }))
            // TODO: Move base cell render into trait impl so we don't have to repeat this
            .children(self.cell_position_spacer(true, window, cx))
            .child(
                h_flex()
                    .w_full()
                    .pr_2()
                    .rounded_xs()
                    .items_start()
                    .gap(DynamicSpacing::Base08.rems(cx))
                    .bg(self.selected_bg_color(window, cx))
                    .child(self.gutter(window, cx))
                    .child(
                        div()
                            .flex()
                            .size_full()
                            .flex_1()
                            .p_3()
                            .font_ui(cx)
                            .text_size(TextSize::Default.rems(cx))
                            .child(self.source.clone()),
                    ),
            )
            // TODO: Move base cell render into trait impl so we don't have to repeat this
            .children(self.cell_position_spacer(false, window, cx))
    }
}
