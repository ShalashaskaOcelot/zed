# Phase 34 — Notebook & kernel configuration

Kind: **mixed** (new settings + a flow change). Not yet started — this is a
plan. Promoted from the backlog to keep 5 phases in rotation after phase 30
completed. Bundles the remaining configuration/environment backlog items.

Primary files: `crates/repl/src/repl_settings.rs`,
`crates/settings_content/src/settings_content.rs`,
`crates/repl/src/notebook/notebook_ui.rs` (autostart hook, venv flow).

## Tasks

- [ ] Kernel autostart on notebook open, setting-gated and OPT-IN (e.g.
      `notebook_autostart_kernel`, default false): when on and a kernel is
      remembered (pick or metadata match), launch it on open instead of on
      first run. Lazy start stays the default.
- [ ] Let "Create Python Environment" choose the location (user 2026-07-08):
      default stays the workspace `.venv` (Enter/OK just works), but allow
      picking a different directory for user-central venvs.
- [ ] Dedicated REPL / Notebook section in the GUI settings UI (user
      2026-07-11): surface the notebook settings (landing mode, last-executed
      timestamp, autostart, output limits) as a grouped, discoverable section
      rather than settings.json-only. Investigate how Zed's settings UI
      registers sections; if registration turns out to be a large separate
      framework effort, report back and re-scope rather than hacking it in.

## Risks / gaps

- Autostart must respect the bug #31 rule: never autostart a spec whose launch
  already failed, and never autostart remote-server specs silently.
- The venv location picker needs a directory-picker flow; keep the fast path
  (Enter = workspace .venv) frictionless.

## Verification

- `cargo check -p repl` + clippy clean; tests pass.
- User test: autostart off by default; on → remembered kernel starts on open;
  venv creation offers a location; settings UI shows the notebook group.
