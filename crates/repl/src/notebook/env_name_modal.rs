use editor::Editor;
use futures::channel::oneshot;
use gpui::{DismissEvent, Entity, EventEmitter, FocusHandle, Focusable, rems};
use ui::{Headline, HeadlineSize, Label, LabelSize, prelude::*};
use workspace::ModalView;

/// A small modal that prompts for a single-line name (e.g. a conda environment
/// name) and returns the trimmed, non-empty value through a oneshot channel on
/// confirm. Dismissing without confirming drops the sender, so the awaiting
/// task sees the channel close and simply does nothing.
pub struct EnvNameModal {
    title: SharedString,
    description: SharedString,
    editor: Entity<Editor>,
    tx: Option<oneshot::Sender<String>>,
}

impl EnvNameModal {
    pub fn new(
        title: impl Into<SharedString>,
        description: impl Into<SharedString>,
        placeholder: impl Into<SharedString>,
        tx: oneshot::Sender<String>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Self {
        let placeholder = placeholder.into();
        let editor = cx.new(|cx| {
            let mut editor = Editor::single_line(window, cx);
            editor.set_placeholder_text(placeholder.as_ref(), window, cx);
            editor
        });
        Self {
            title: title.into(),
            description: description.into(),
            editor,
            tx: Some(tx),
        }
    }

    fn cancel(&mut self, _: &menu::Cancel, _window: &mut Window, cx: &mut Context<Self>) {
        cx.emit(DismissEvent);
    }

    fn confirm(&mut self, _: &menu::Confirm, _window: &mut Window, cx: &mut Context<Self>) {
        let name = self.editor.read(cx).text(cx).trim().to_string();
        // Ignore a confirm on an empty field so the user isn't dropped into a
        // failed `conda create` with no name.
        if name.is_empty() {
            return;
        }
        if let Some(tx) = self.tx.take() {
            tx.send(name).ok();
        }
        cx.emit(DismissEvent);
    }
}

impl EventEmitter<DismissEvent> for EnvNameModal {}
impl ModalView for EnvNameModal {}
impl Focusable for EnvNameModal {
    fn focus_handle(&self, cx: &App) -> FocusHandle {
        self.editor.focus_handle(cx)
    }
}

impl Render for EnvNameModal {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        v_flex()
            .key_context("EnvNamePrompt")
            .on_action(cx.listener(Self::cancel))
            .on_action(cx.listener(Self::confirm))
            .elevation_2(cx)
            .w(rems(30.))
            .child(
                v_flex()
                    .p_3()
                    .gap_1()
                    .child(Headline::new(self.title.clone()).size(HeadlineSize::XSmall))
                    .child(
                        Label::new(self.description.clone())
                            .size(LabelSize::Small)
                            .color(Color::Muted),
                    ),
            )
            .child(
                div()
                    .p_3()
                    .border_t_1()
                    .border_color(cx.theme().colors().border_variant)
                    .bg(cx.theme().colors().editor_background)
                    .child(self.editor.clone()),
            )
    }
}
