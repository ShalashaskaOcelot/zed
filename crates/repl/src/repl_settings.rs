use settings::{NotebookFollowMode, NotebookRunLandingMode, RegisterSetting, Settings};

/// Settings for configuring REPL display and behavior.
#[derive(Clone, Debug, RegisterSetting)]
pub struct ReplSettings {
    /// Maximum number of lines to keep in REPL's scrollback buffer.
    /// Clamped with [4, 256] range.
    ///
    /// Default: 32
    pub max_lines: usize,
    /// Maximum number of columns to keep in REPL's scrollback buffer.
    /// Clamped with [20, 512] range.
    ///
    /// Default: 128
    pub max_columns: usize,
    /// Whether to show small single-line outputs inline instead of in a block.
    ///
    /// Default: true
    pub inline_output: bool,
    /// Maximum number of characters for an output to be shown inline.
    /// Only applies when `inline_output` is true.
    ///
    /// Default: 50
    pub inline_output_max_length: usize,
    /// Maximum number of lines of output to display before scrolling.
    /// Set to 0 to disable output height limits.
    ///
    /// Default: 0
    pub output_max_height_lines: usize,
    /// Which mode a notebook lands in after running a cell (ctrl-enter /
    /// shift-enter): always command, always edit, or whatever mode the run
    /// was triggered from.
    ///
    /// Default: command
    pub notebook_run_landing_mode: NotebookRunLandingMode,
    /// Whether to show WHEN a cell was last executed (a timestamp next to the
    /// ✓/✕ and duration), VS Code style.
    ///
    /// Default: true
    pub notebook_show_last_executed: bool,
    /// Whether to start a notebook's remembered kernel on open instead of
    /// waiting for the first run (opt-in; lazy start is the default).
    ///
    /// Default: false
    pub notebook_autostart_kernel: bool,
    /// Whether to show a running total of time spent executing cells next to
    /// the kernel status.
    ///
    /// Default: false
    pub notebook_show_execution_time: bool,
    /// Whether to show how long the kernel has been running next to the kernel
    /// status.
    ///
    /// Default: false
    pub notebook_show_kernel_uptime: bool,
    /// How the notebook scrolls while "follow running cell" is on: pin each
    /// cell near the top as it runs, or move a page at a time and let the
    /// selection carry the progress.
    ///
    /// Default: minimal
    pub notebook_follow_mode: NotebookFollowMode,
}

impl Settings for ReplSettings {
    fn from_settings(content: &settings::SettingsContent) -> Self {
        let repl = content.repl.as_ref().unwrap();

        Self {
            max_lines: repl.max_lines.unwrap(),
            max_columns: repl.max_columns.unwrap(),
            inline_output: repl.inline_output.unwrap_or(true),
            inline_output_max_length: repl.inline_output_max_length.unwrap_or(50),
            output_max_height_lines: repl.output_max_height_lines.unwrap_or(0),
            notebook_run_landing_mode: repl.notebook_run_landing_mode.unwrap_or_default(),
            notebook_show_last_executed: repl.notebook_show_last_executed.unwrap_or(true),
            notebook_autostart_kernel: repl.notebook_autostart_kernel.unwrap_or(false),
            notebook_show_execution_time: repl.notebook_show_execution_time.unwrap_or(false),
            notebook_show_kernel_uptime: repl.notebook_show_kernel_uptime.unwrap_or(false),
            notebook_follow_mode: repl.notebook_follow_mode.unwrap_or_default(),
        }
    }
}
