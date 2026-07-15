use crate::KERNEL_DOCS_URL;
use crate::kernels::KernelSpecification;
use crate::repl_store::ReplStore;

use gpui::{AnyView, DismissEvent, FontWeight, SharedString, Task};
use picker::{Picker, PickerDelegate};
use project::WorktreeId;
use std::sync::Arc;
use ui::{ListItem, ListItemSpacing, PopoverMenu, PopoverMenuHandle, PopoverTrigger, prelude::*};

type OnSelect = Box<dyn Fn(KernelSpecification, &mut Window, &mut App)>;
type OnDismiss = Box<dyn Fn(&mut Window, &mut App)>;
type OnCreateEnv = std::rc::Rc<dyn Fn(&mut Window, &mut App)>;

#[derive(Clone)]
pub enum KernelPickerEntry {
    SectionHeader(SharedString),
    Kernel {
        spec: KernelSpecification,
        is_recommended: bool,
    },
}

fn build_grouped_entries(
    store: &ReplStore,
    worktree_id: WorktreeId,
    selected_kernel: Option<&KernelSpecification>,
) -> Vec<KernelPickerEntry> {
    let mut entries = Vec::new();
    let mut recommended_entry: Option<KernelPickerEntry> = None;
    let mut found_selected = false;

    let mut python_envs = Vec::new();
    let mut jupyter_kernels = Vec::new();
    let mut wsl_kernels = Vec::new();
    let mut remote_kernels = Vec::new();

    for spec in store.kernel_specifications_for_worktree(worktree_id) {
        let is_recommended = store.is_recommended_kernel(worktree_id, spec);
        let is_selected = selected_kernel.map_or(false, |s| s == spec);

        if is_selected {
            recommended_entry = Some(KernelPickerEntry::Kernel {
                spec: spec.clone(),
                is_recommended: true,
            });
            found_selected = true;
        } else if is_recommended && !found_selected {
            recommended_entry = Some(KernelPickerEntry::Kernel {
                spec: spec.clone(),
                is_recommended: true,
            });
        }

        match spec {
            KernelSpecification::PythonEnv(_) => {
                python_envs.push(KernelPickerEntry::Kernel {
                    spec: spec.clone(),
                    is_recommended,
                });
            }
            KernelSpecification::Jupyter(_) => {
                jupyter_kernels.push(KernelPickerEntry::Kernel {
                    spec: spec.clone(),
                    is_recommended,
                });
            }
            KernelSpecification::JupyterServer(_) | KernelSpecification::SshRemote(_) => {
                remote_kernels.push(KernelPickerEntry::Kernel {
                    spec: spec.clone(),
                    is_recommended,
                });
            }
            KernelSpecification::WslRemote(_) => {
                wsl_kernels.push(KernelPickerEntry::Kernel {
                    spec: spec.clone(),
                    is_recommended,
                });
            }
        }
    }

    // Sort Python envs: has_ipykernel first, then by name
    python_envs.sort_by(|a, b| {
        let (spec_a, spec_b) = match (a, b) {
            (
                KernelPickerEntry::Kernel { spec: sa, .. },
                KernelPickerEntry::Kernel { spec: sb, .. },
            ) => (sa, sb),
            _ => return std::cmp::Ordering::Equal,
        };
        spec_b
            .has_ipykernel()
            .cmp(&spec_a.has_ipykernel())
            .then_with(|| spec_a.name().cmp(&spec_b.name()))
    });

    // Recommended section
    if let Some(rec) = recommended_entry {
        entries.push(KernelPickerEntry::SectionHeader("Recommended".into()));
        entries.push(rec);
    }

    // Python Environments section
    if !python_envs.is_empty() {
        entries.push(KernelPickerEntry::SectionHeader(
            "Python Environments".into(),
        ));
        entries.extend(python_envs);
    }

    // Jupyter Kernels section
    if !jupyter_kernels.is_empty() {
        entries.push(KernelPickerEntry::SectionHeader("Jupyter Kernels".into()));
        entries.extend(jupyter_kernels);
    }

    // WSL Kernels section
    if !wsl_kernels.is_empty() {
        entries.push(KernelPickerEntry::SectionHeader("WSL Kernels".into()));
        entries.extend(wsl_kernels);
    }

    // Remote section
    if !remote_kernels.is_empty() {
        entries.push(KernelPickerEntry::SectionHeader("Remote Servers".into()));
        entries.extend(remote_kernels);
    }

    entries
}

#[derive(IntoElement)]
pub struct KernelSelector<T, TT>
where
    T: PopoverTrigger + ButtonCommon,
    TT: Fn(&mut Window, &mut App) -> AnyView + 'static,
{
    handle: Option<PopoverMenuHandle<Picker<KernelPickerDelegate>>>,
    on_select: OnSelect,
    on_dismiss: Option<OnDismiss>,
    on_create_env: Option<OnCreateEnv>,
    trigger: T,
    tooltip: TT,
    info_text: Option<SharedString>,
    worktree_id: WorktreeId,
    /// When set, this is the current selection shown in the picker (checkmark
    /// + Recommended override) instead of the store's worktree-level
    /// selection. Notebooks pass their own per-notebook kernel here (bug #30).
    selected_override: Option<Option<KernelSpecification>>,
}

pub struct KernelPickerDelegate {
    all_entries: Vec<KernelPickerEntry>,
    filtered_entries: Vec<KernelPickerEntry>,
    selected_kernelspec: Option<KernelSpecification>,
    selected_index: usize,
    on_select: OnSelect,
    on_dismiss: Option<OnDismiss>,
    on_create_env: Option<OnCreateEnv>,
}

impl<T, TT> KernelSelector<T, TT>
where
    T: PopoverTrigger + ButtonCommon,
    TT: Fn(&mut Window, &mut App) -> AnyView + 'static,
{
    pub fn new(on_select: OnSelect, worktree_id: WorktreeId, trigger: T, tooltip: TT) -> Self {
        KernelSelector {
            on_select,
            on_dismiss: None,
            on_create_env: None,
            handle: None,
            trigger,
            tooltip,
            info_text: None,
            worktree_id,
            selected_override: None,
        }
    }

    /// Show `selected` as the picker's current selection instead of the
    /// store's worktree-level selection (which belongs to the inline REPL).
    pub fn with_selected(mut self, selected: Option<KernelSpecification>) -> Self {
        self.selected_override = Some(selected);
        self
    }

    /// Called when the user chooses "Create Python Environment" in the picker
    /// footer. The callback is responsible for starting the creation flow and
    /// dismissing the picker.
    pub fn with_create_env(mut self, on_create_env: OnCreateEnv) -> Self {
        self.on_create_env = Some(on_create_env);
        self
    }

    pub fn with_handle(mut self, handle: PopoverMenuHandle<Picker<KernelPickerDelegate>>) -> Self {
        self.handle = Some(handle);
        self
    }

    /// Called when the picker is dismissed (both on selection and on cancel).
    /// On selection, `on_select` runs first, so a dismiss handler that clears
    /// pending state won't undo a just-made selection.
    pub fn with_dismiss(mut self, on_dismiss: OnDismiss) -> Self {
        self.on_dismiss = Some(on_dismiss);
        self
    }

    pub fn with_info_text(mut self, text: impl Into<SharedString>) -> Self {
        self.info_text = Some(text.into());
        self
    }
}

impl KernelPickerDelegate {
    fn first_selectable_index(entries: &[KernelPickerEntry]) -> usize {
        entries
            .iter()
            .position(|e| matches!(e, KernelPickerEntry::Kernel { .. }))
            .unwrap_or(0)
    }

    fn next_selectable_index(&self, from: usize, direction: i32) -> usize {
        let len = self.filtered_entries.len();
        if len == 0 {
            return 0;
        }

        let mut index = from as i32 + direction;
        while index >= 0 && (index as usize) < len {
            if matches!(
                self.filtered_entries.get(index as usize),
                Some(KernelPickerEntry::Kernel { .. })
            ) {
                return index as usize;
            }
            index += direction;
        }

        from
    }
}

impl PickerDelegate for KernelPickerDelegate {
    type ListItem = ListItem;

    fn name() -> &'static str {
        "kernel picker"
    }

    fn match_count(&self) -> usize {
        self.filtered_entries.len()
    }

    fn selected_index(&self) -> usize {
        self.selected_index
    }

    fn set_selected_index(&mut self, ix: usize, _: &mut Window, cx: &mut Context<Picker<Self>>) {
        if matches!(
            self.filtered_entries.get(ix),
            Some(KernelPickerEntry::SectionHeader(_))
        ) {
            let forward = self.next_selectable_index(ix, 1);
            if forward != ix {
                self.selected_index = forward;
            } else {
                self.selected_index = self.next_selectable_index(ix, -1);
            }
        } else {
            self.selected_index = ix;
        }

        if let Some(KernelPickerEntry::Kernel { spec, .. }) =
            self.filtered_entries.get(self.selected_index)
        {
            self.selected_kernelspec = Some(spec.clone());
        }
        cx.notify();
    }

    fn placeholder_text(&self, _window: &mut Window, _cx: &mut App) -> Arc<str> {
        "Select a kernel...".into()
    }

    fn update_matches(
        &mut self,
        query: String,
        _window: &mut Window,
        _cx: &mut Context<Picker<Self>>,
    ) -> Task<()> {
        if query.is_empty() {
            self.filtered_entries = self.all_entries.clone();
        } else {
            let query_lower = query.to_lowercase();
            let mut filtered = Vec::new();
            let mut pending_header: Option<KernelPickerEntry> = None;

            for entry in &self.all_entries {
                match entry {
                    KernelPickerEntry::SectionHeader(_) => {
                        pending_header = Some(entry.clone());
                    }
                    KernelPickerEntry::Kernel { spec, .. } => {
                        if spec.name().to_lowercase().contains(&query_lower) {
                            if let Some(header) = pending_header.take() {
                                filtered.push(header);
                            }
                            filtered.push(entry.clone());
                        }
                    }
                }
            }

            self.filtered_entries = filtered;
        }

        self.selected_index = Self::first_selectable_index(&self.filtered_entries);
        if let Some(KernelPickerEntry::Kernel { spec, .. }) =
            self.filtered_entries.get(self.selected_index)
        {
            self.selected_kernelspec = Some(spec.clone());
        }

        Task::ready(())
    }

    fn separators_after_indices(&self) -> Vec<usize> {
        let mut separators = Vec::new();
        for (index, entry) in self.filtered_entries.iter().enumerate() {
            if matches!(entry, KernelPickerEntry::SectionHeader(_)) && index > 0 {
                separators.push(index - 1);
            }
        }
        separators
    }

    fn confirm(&mut self, _secondary: bool, window: &mut Window, cx: &mut Context<Picker<Self>>) {
        if let Some(KernelPickerEntry::Kernel { spec, .. }) =
            self.filtered_entries.get(self.selected_index)
        {
            (self.on_select)(spec.clone(), window, cx);
            cx.emit(DismissEvent);
        }
    }

    fn dismissed(&mut self, window: &mut Window, cx: &mut Context<Picker<Self>>) {
        if let Some(on_dismiss) = &self.on_dismiss {
            on_dismiss(window, cx);
        }
    }

    fn render_match(
        &self,
        ix: usize,
        selected: bool,
        _: &mut Window,
        cx: &mut Context<Picker<Self>>,
    ) -> Option<Self::ListItem> {
        let entry = self.filtered_entries.get(ix)?;

        match entry {
            KernelPickerEntry::SectionHeader(title) => Some(
                ListItem::new(ix)
                    .inset(true)
                    .spacing(ListItemSpacing::Dense)
                    .selectable(false)
                    .child(
                        Label::new(title.clone())
                            .size(LabelSize::Small)
                            .weight(FontWeight::SEMIBOLD)
                            .color(Color::Muted),
                    ),
            ),
            KernelPickerEntry::Kernel {
                spec,
                is_recommended,
            } => {
                let is_currently_selected = self.selected_kernelspec.as_ref() == Some(spec);
                let icon = spec.icon(cx);
                let has_ipykernel = spec.has_ipykernel();

                let subtitle = match spec {
                    KernelSpecification::Jupyter(_) => None,
                    KernelSpecification::WslRemote(_) => Some(spec.path().to_string()),
                    KernelSpecification::PythonEnv(_)
                    | KernelSpecification::JupyterServer(_)
                    | KernelSpecification::SshRemote(_) => {
                        let env_kind = spec.environment_kind_label();
                        let path = spec.path();
                        match env_kind {
                            Some(kind) => Some(format!("{} \u{2013} {}", kind, path)),
                            None => Some(path.to_string()),
                        }
                    }
                };

                Some(
                    ListItem::new(ix)
                        .inset(true)
                        .spacing(ListItemSpacing::Sparse)
                        .toggle_state(selected)
                        .child(
                            h_flex()
                                .w_full()
                                .gap_3()
                                .when(!has_ipykernel, |flex| flex.opacity(0.5))
                                .child(icon.color(Color::Default).size(IconSize::Medium))
                                .child(
                                    v_flex()
                                        .flex_grow_1()
                                        .overflow_x_hidden()
                                        .gap_0p5()
                                        .child(
                                            h_flex()
                                                .gap_1()
                                                .child(
                                                    div()
                                                        .overflow_x_hidden()
                                                        .flex_shrink_1()
                                                        .text_ellipsis()
                                                        .child(
                                                            Label::new(spec.name())
                                                                .weight(FontWeight::MEDIUM)
                                                                .size(LabelSize::Default),
                                                        ),
                                                )
                                                .when(*is_recommended, |flex| {
                                                    flex.child(
                                                        Label::new("Recommended")
                                                            .size(LabelSize::XSmall)
                                                            .color(Color::Accent),
                                                    )
                                                })
                                                .when(!has_ipykernel, |flex| {
                                                    flex.child(
                                                        Label::new("ipykernel not installed")
                                                            .size(LabelSize::XSmall)
                                                            .color(Color::Warning),
                                                    )
                                                }),
                                        )
                                        .when_some(subtitle, |flex, subtitle| {
                                            flex.child(
                                                div().overflow_x_hidden().text_ellipsis().child(
                                                    Label::new(subtitle)
                                                        .size(LabelSize::Small)
                                                        .color(Color::Muted),
                                                ),
                                            )
                                        }),
                                ),
                        )
                        .when(is_currently_selected, |item| {
                            item.end_slot(
                                Icon::new(IconName::Check)
                                    .color(Color::Accent)
                                    .size(IconSize::Small),
                            )
                        }),
                )
            }
        }
    }

    fn render_footer(
        &self,
        _: &mut Window,
        cx: &mut Context<Picker<Self>>,
    ) -> Option<gpui::AnyElement> {
        Some(
            h_flex()
                .w_full()
                .border_t_1()
                .border_color(cx.theme().colors().border_variant)
                .p_1()
                .gap_4()
                .child(
                    Button::new("kernel-docs", "Kernel Docs")
                        .end_icon(
                            Icon::new(IconName::ArrowUpRight)
                                .size(IconSize::Small)
                                .color(Color::Muted),
                        )
                        .on_click(move |_, _, cx| cx.open_url(KERNEL_DOCS_URL)),
                )
                .when_some(self.on_create_env.clone(), |this, on_create_env| {
                    this.child(
                        Button::new("create-python-env", "Create Python Environment")
                            .start_icon(
                                Icon::new(IconName::Plus)
                                    .size(IconSize::Small)
                                    .color(Color::Muted),
                            )
                            .on_click(move |_, window, cx| on_create_env(window, cx)),
                    )
                })
                .into_any(),
        )
    }
}

impl<T, TT> RenderOnce for KernelSelector<T, TT>
where
    T: PopoverTrigger + ButtonCommon,
    TT: Fn(&mut Window, &mut App) -> AnyView + 'static,
{
    fn render(self, window: &mut Window, cx: &mut App) -> impl IntoElement {
        let store_entity = ReplStore::global(cx);
        store_entity.update(cx, |store, cx| store.ensure_kernelspecs(cx));
        let store = store_entity.read(cx);

        let selected_kernelspec = match &self.selected_override {
            Some(selected) => selected.clone(),
            None => store.active_kernelspec(self.worktree_id, None, cx),
        };
        let all_entries =
            build_grouped_entries(store, self.worktree_id, selected_kernelspec.as_ref());
        let selected_index = all_entries
            .iter()
            .position(|entry| {
                if let KernelPickerEntry::Kernel { spec, .. } = entry {
                    selected_kernelspec.as_ref() == Some(spec)
                } else {
                    false
                }
            })
            .unwrap_or_else(|| KernelPickerDelegate::first_selectable_index(&all_entries));

        let selected_for_rebuild = selected_kernelspec.clone();
        let delegate = KernelPickerDelegate {
            on_select: self.on_select,
            on_dismiss: self.on_dismiss,
            on_create_env: self.on_create_env,
            all_entries: all_entries.clone(),
            filtered_entries: all_entries,
            selected_kernelspec,
            selected_index,
        };

        let worktree_id = self.worktree_id;
        let picker_view = cx.new(|cx| {
            // Kernelspec / toolchain discovery is asynchronous, so the picker
            // may be built (and opened) before any kernels are known. Rebuild
            // the entries whenever the store updates, so kernels stream into
            // an already-open picker instead of it staying empty (bug #20).
            cx.observe_in(
                &store_entity,
                window,
                move |picker: &mut Picker<KernelPickerDelegate>, store, window, cx| {
                    let entries = build_grouped_entries(
                        store.read(cx),
                        worktree_id,
                        selected_for_rebuild.as_ref(),
                    );
                    if picker.delegate.selected_kernelspec.is_none() {
                        picker.delegate.selected_index =
                            KernelPickerDelegate::first_selectable_index(&entries);
                    }
                    picker.delegate.all_entries = entries;
                    // Re-applies the current query over the new entries.
                    picker.refresh(window, cx);
                },
            )
            .detach();

            Picker::list(delegate, window, cx)
                .list_measure_all()
                .popover()
        });

        PopoverMenu::new("kernel-switcher")
            .menu(move |_window, _cx| Some(picker_view.clone()))
            .trigger_with_tooltip(self.trigger, self.tooltip)
            .attach(gpui::Anchor::BottomLeft)
            .when_some(self.handle, |menu, handle| menu.with_handle(handle))
    }
}
