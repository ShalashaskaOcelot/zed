# Building a Windows installer for this fork (local, unsigned, user-level)

This is the reproducible procedure for producing a working Windows installer of
**this fork** on your own Windows machine — no CI, no code signing, and no
Zed-Industries services. It is the foundation the release pipeline (Drone →
Gitea, a later phase) automates; get it working by hand first.

The installer Zed produces is already **user-level**: it needs no admin rights,
installs under `%LOCALAPPDATA%\Programs`, and only writes `HKCU`. Nothing in
this procedure changes that.

> **Channel note.** In-repo `crates/zed/RELEASE_CHANNEL` is `dev`, so the build
> below produces a **"Zed Dev"** build (its own AppId, icon, and install dir,
> side-by-side with any real Zed you have installed). Dev-channel builds
> **never auto-update** — that is expected and fine until the fork's
> auto-update work lands.

---

## 1. Prerequisites

Install these once. Several paths are **hardcoded** in
`script/bundle-windows.ps1`; where a version or path is exact, it is called out
— a mismatch fails the build.

| Component | Required version / path | Notes |
|---|---|---|
| **Visual Studio 2022** | Any edition — Community, Professional, Enterprise, or **Build Tools** | Discovered via `vswhere`; you no longer need the Community edition specifically. |
| ↳ MSVC x64 build tools | "MSVC v143 - VS 2022 C++ x64/x86 build tools" | The C++ toolchain the Rust MSVC target links against. |
| ↳ Spectre-mitigated libs | "MSVC v143 … Spectre-mitigated libs (Latest)" | Required by the build; a plain MSVC install without these fails to link. |
| ↳ C++ CMake tools | "C++ CMake tools for Windows" | Some native dependencies build via CMake. |
| **Windows 11 SDK** | **`10.0.26100`** (exact) | `makeAppx.exe` is invoked from the hardcoded path `C:\Program Files (x86)\Windows Kits\10\bin\10.0.26100.0\x64`. A different SDK build number will not be found. |
| **rustup** | Toolchain **1.95.0** | Auto-pinned by `rust-toolchain.toml`; rustup installs it on first build. Just have rustup itself present. |
| **Inno Setup 6** | Installed at `C:\Program Files (x86)\Inno Setup 6\ISCC.exe` (exact) | The compiler path is hardcoded. Install to the default location. |

### System configuration

- **Enable long paths.** Both are needed:
  - Registry: set `HKLM\SYSTEM\CurrentControlSet\Control\FileSystem` →
    `LongPathsEnabled` (DWORD) to `1`, e.g. in an elevated PowerShell:
    ```powershell
    New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" `
      -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
    ```
  - Git: `git config --global core.longpaths true` (or `--local` in the clone).
- **`RUSTFLAGS` must be unset.** A globally-exported `RUSTFLAGS` changes the
  build configuration and breaks the bundle. Clear it in the shell you build
  from: `Remove-Item Env:\RUSTFLAGS -ErrorAction SilentlyContinue`.
- **Outbound HTTPS to `github.com`.** The build downloads the AMD GPU Services
  (AGS) SDK and ConPTY from GitHub at bundle time, plus crates from crates.io.
  A locked-down or proxied network that blocks these will fail the bundle.
- **Free disk:** roughly **45–100 GB**. A clean release build of the whole
  workspace is large; the high end assumes the `target/` directory persists.

---

## 2. Build

From the **repo root**, in **PowerShell** (`pwsh`), with `RUSTFLAGS` unset and
`CI` **not** set (a plain interactive shell — do not set `$env:CI`):

```powershell
./script/bundle-windows.ps1
```

Optionally pass `-Install` to launch the resulting installer when the build
finishes:

```powershell
./script/bundle-windows.ps1 -Install
```

With `CI` unset the script automatically **skips** code signing, Sentry symbol
upload, and the signing-secret environment-variable check — so no Azure / cert
configuration is required for a local build.

The first build is slow (full release compile with thin-LTO and a single
codegen unit; the final link is RAM-heavy). Subsequent builds reuse `target/`.

---

## 3. What "done" looks like

On success the script prints `Build successful` and the installer is written to:

```
target\Zed-x86_64.exe
```

(`Zed-<arch>.exe` — `aarch64` Windows is out of scope for this fork; build
x86_64.)

Run that `.exe` to install. Because it is unsigned and downloaded/produced
outside the Microsoft store, **SmartScreen will warn** on first run — that is
the expected cost of an unsigned installer; choose "More info → Run anyway".
The install lands under `%LOCALAPPDATA%\Programs\Zed Dev`, adds a Start-menu
entry, an optional desktop icon, and puts `zed` on the per-user PATH — all
without a UAC prompt. Uninstall from **Apps & Features** (per-user).

---

## 4. Troubleshooting

- **"Could not locate a Visual Studio 2022 installation…"** — VS 2022 (any of
  Community / Professional / Enterprise / Build Tools) is not installed, or
  `vswhere` cannot see it. Confirm with
  `& "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe" -latest -products * -property installationPath`.
- **`makeAppx.exe` not found** — the Windows 11 SDK `10.0.26100` is missing;
  install exactly that SDK build.
- **`ISCC.exe` not found** — install Inno Setup 6 to its default location.
- **Link failures / odd codegen** — check `RUSTFLAGS` is unset in the current
  shell.
- **Path-too-long errors during checkout or build** — long paths are not
  enabled (registry *and* `git config core.longpaths true`).
- **Network errors fetching AGS or ConPTY** — outbound HTTPS to `github.com`
  is blocked.
