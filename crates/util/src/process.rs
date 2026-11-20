use anyhow::{Context as _, Result};
use std::process::Stdio;

/// A wrapper around `smol::process::Child` that attempts to clean up subprocess
/// trees: on Unix by using process groups, and on Windows by using job objects.
///
/// On Windows, if job setup and assignment succeed, dropping this struct
/// closes the job handle and terminates processes in the job. If either step
/// fails, the direct child remains usable but descendants are not tracked.
/// A descendant spawned before job assignment can also escape the job.
pub struct Child {
    process: smol::process::Child,
    #[cfg(windows)]
    job: Option<windows_job::JobObject>,
    #[cfg(windows)]
    interrupt_event: Option<windows_interrupt::InterruptEvent>,
}

impl std::ops::Deref for Child {
    type Target = smol::process::Child;

    fn deref(&self) -> &Self::Target {
        &self.process
    }
}

impl std::ops::DerefMut for Child {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.process
    }
}

impl Child {
    #[cfg(not(windows))]
    pub fn spawn(
        command: std::process::Command,
        stdin: Stdio,
        stdout: Stdio,
        stderr: Stdio,
    ) -> Result<Self> {
        Self::spawn_impl(command, stdin, stdout, stderr)
    }

    /// Like [`Child::spawn`], but the resulting child can be interrupted with
    /// [`Child::interrupt`]. On Unix any child can be interrupted (SIGINT to
    /// its process group), so this is identical to `spawn`; the distinct
    /// method exists for the Windows path, which must set up an interrupt
    /// event at spawn time.
    #[cfg(not(windows))]
    pub fn spawn_interruptible(
        command: std::process::Command,
        stdin: Stdio,
        stdout: Stdio,
        stderr: Stdio,
    ) -> Result<Self> {
        Self::spawn_impl(command, stdin, stdout, stderr)
    }

    #[cfg(not(windows))]
    fn spawn_impl(
        mut command: std::process::Command,
        stdin: Stdio,
        stdout: Stdio,
        stderr: Stdio,
    ) -> Result<Self> {
        crate::set_pre_exec_to_start_new_session(&mut command);
        let mut command = smol::process::Command::from(command);
        let process = command
            .stdin(stdin)
            .stdout(stdout)
            .stderr(stderr)
            .spawn()
            .with_context(|| {
                format!(
                    "failed to spawn command {}",
                    crate::redact::redact_command(&format!("{command:?}"))
                )
            })?;
        Ok(Self { process })
    }

    #[cfg(windows)]
    pub fn spawn(
        command: std::process::Command,
        stdin: Stdio,
        stdout: Stdio,
        stderr: Stdio,
    ) -> Result<Self> {
        Self::spawn_impl(command, stdin, stdout, stderr, false)
    }

    /// Like [`Child::spawn`], but the resulting child can be interrupted with
    /// [`Child::interrupt`]. On Windows this creates an inheritable interrupt
    /// event and passes it to the child via `JPY_INTERRUPT_EVENT`, which is
    /// how Jupyter kernels are interrupted on Windows.
    #[cfg(windows)]
    pub fn spawn_interruptible(
        command: std::process::Command,
        stdin: Stdio,
        stdout: Stdio,
        stderr: Stdio,
    ) -> Result<Self> {
        Self::spawn_impl(command, stdin, stdout, stderr, true)
    }

    #[cfg(windows)]
    fn spawn_impl(
        command: std::process::Command,
        stdin: Stdio,
        stdout: Stdio,
        stderr: Stdio,
        interruptible: bool,
    ) -> Result<Self> {
        let interrupt_event = if interruptible {
            match windows_interrupt::InterruptEvent::new() {
                Ok(event) => Some(event),
                Err(error) => {
                    log::error!("failed to create process interrupt event: {error:#}");
                    None
                }
            }
        } else {
            None
        };

        let mut command = smol::process::Command::from(command);
        if let Some(event) = &interrupt_event {
            // The child inherits this event handle with the same numeric value
            // because it is created inheritable; ipykernel's Windows parent
            // poller waits on it and raises KeyboardInterrupt when signaled.
            command.env("JPY_INTERRUPT_EVENT", event.env_value());
        }
        let process = command
            .stdin(stdin)
            .stdout(stdout)
            .stderr(stderr)
            .spawn()
            .with_context(|| {
                format!(
                    "failed to spawn command {}",
                    crate::redact::redact_command(&format!("{command:?}"))
                )
            })?;

        // Assign the child to a job object configured to kill the entire
        // process tree when the last job handle is closed, so descendants
        // (e.g. node workers and MCP servers spawned by agent servers) are
        // reaped even if the direct child doesn't clean them up. Any process
        // the child spawns after this assignment is automatically part of the
        // job.
        //
        // There is a small race: descendants the child spawns between the
        // `spawn()` call returning and the assignment below escape the job.
        // Closing it fully would require creating the process suspended
        // (`CREATE_SUSPENDED`), assigning it, then resuming it, which the
        // std/smol process APIs don't support without reimplementing process
        // creation. The window is normally short, but descendants created in
        // it are not tracked by the job.
        let job = windows_job::JobObject::new()
            .and_then(|job| {
                job.assign_process(process.id())?;
                Ok(job)
            })
            .map_err(|error| {
                log::error!("failed to assign spawned process to a job object: {error:#}");
            })
            .ok();

        Ok(Self {
            process,
            job,
            interrupt_event,
        })
    }

    /// Consumes the child, draining its stdout/stderr and waiting for it to
    /// exit, then returns the collected output.
    pub async fn output(self) -> Result<std::process::Output> {
        // NOTE: Keep `self` alive across this await, do not destructure it to
        // pull `process` out first. On Windows that drops the job object early,
        // which triggers `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE` and kills the
        // child before `output()` finishes collecting its stdout/stderr.
        Ok(self.process.output().await?)
    }

    #[cfg(not(windows))]
    pub fn kill(&mut self) -> Result<()> {
        let pid = self.process.id();
        unsafe {
            libc::killpg(pid as i32, libc::SIGKILL);
        }
        Ok(())
    }

    #[cfg(windows)]
    pub fn kill(&mut self) -> Result<()> {
        if let Some(job) = &self.job {
            job.terminate()
        } else {
            self.process.kill()?;
            Ok(())
        }
    }

    /// Sends an interrupt to the child without killing it. On Unix this is
    /// SIGINT to the child's process group (the child starts a new session at
    /// spawn, so its process-group id equals its pid). On Windows the child
    /// must have been spawned with [`Child::spawn_interruptible`]; a real
    /// CTRL_C console event is delivered to the child's console when
    /// possible, with the interrupt event as fallback. Jupyter kernels turn
    /// either into a `KeyboardInterrupt`.
    #[cfg(not(windows))]
    pub fn interrupt(&self) -> Result<()> {
        let pid = self.process.id();
        let result = unsafe { libc::killpg(pid as i32, libc::SIGINT) };
        if result != 0 {
            return Err(std::io::Error::last_os_error())
                .context("failed to send SIGINT to child process group");
        }
        Ok(())
    }

    #[cfg(windows)]
    pub fn interrupt(&self) -> Result<()> {
        // Prefer a real CTRL_C console event: CPython sets the hidden event
        // that wakes main-thread C blockers (`time.sleep`, `input()`) only in
        // its OS-level signal handler (`signal_handler` in
        // Modules/signalmodule.c), which runs for real console events. The
        // JPY interrupt event merely makes ipykernel's poller call
        // `_thread.interrupt_main()`, which trips the between-bytecodes flag
        // without setting that event — pure-Python loops stop, blocking C
        // calls do not. The console path fails harmlessly when this process
        // already owns a console or the child has none; the event then
        // covers those cases (at bytecode-boundary promptness).
        let console_error = match windows_interrupt::send_ctrl_c(self.process.id()) {
            Ok(()) => return Ok(()),
            Err(error) => error,
        };
        match &self.interrupt_event {
            Some(event) => {
                log::debug!(
                    "console CTRL_C failed ({console_error:#}); \
                     falling back to the interrupt event"
                );
                event.signal()
            }
            None => Err(console_error.context("process was not spawned with interrupt support")),
        }
    }

    /// Returns job activity, or `None` when descendant cleanup is unavailable.
    ///
    /// The count includes descendants after the direct child exits. `None`
    /// means job setup failed, not that the process tree is empty.
    ///
    /// # Errors
    ///
    /// Returns an error if Windows cannot query the owned job's activity.
    #[cfg(windows)]
    pub fn has_running_processes(&self) -> Result<Option<bool>> {
        self.job
            .as_ref()
            .map(windows_job::JobObject::has_running_processes)
            .transpose()
    }
}

#[cfg(windows)]
mod windows_interrupt {
    use crate::ResultExt as _;
    use anyhow::{Context as _, Result};
    use windows::Win32::{
        Foundation::{CloseHandle, HANDLE},
        Security::SECURITY_ATTRIBUTES,
        System::Console::{
            AttachConsole, CTRL_C_EVENT, FreeConsole, GenerateConsoleCtrlEvent,
            SetConsoleCtrlHandler,
        },
        System::Threading::{CreateEventW, SetEvent},
    };

    /// Delivers a real CTRL_C console event to the console of the process
    /// `pid`. Zed spawns children with `CREATE_NO_WINDOW`, so each gets its
    /// own hidden console shared only with its descendants — attaching to it
    /// and broadcasting CTRL_C reaches exactly that process tree, mirroring
    /// Unix `killpg(SIGINT)`.
    ///
    /// `GenerateConsoleCtrlEvent` can only signal the CALLER's console, and a
    /// process can be attached to at most one console, so this must
    /// temporarily attach to the child's console. `AttachConsole` fails if
    /// this process already owns a console (never freeing a console we did
    /// not create) or if the child has none — callers fall back to the
    /// interrupt event in those cases.
    pub(crate) fn send_ctrl_c(pid: u32) -> Result<()> {
        // Console attachment is process-global state; concurrent interrupts
        // of different children must not interleave attach/detach.
        static CONSOLE_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
        let _guard = CONSOLE_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());

        // While attached, this process receives the CTRL_C too, and the
        // DEFAULT console handler exits the process. Permanently ignoring
        // CTRL_C is safe for a GUI process and avoids a window where the
        // event could be delivered after a handler reset.
        static IGNORE_CTRL_C: std::sync::OnceLock<Result<(), windows::core::Error>> =
            std::sync::OnceLock::new();
        IGNORE_CTRL_C
            .get_or_init(|| unsafe { SetConsoleCtrlHandler(None, true) })
            .as_ref()
            .map_err(|error| anyhow::anyhow!("failed to ignore CTRL_C in this process: {error}"))?;

        unsafe {
            AttachConsole(pid).context("failed to attach to the child's console")?;
            let result = GenerateConsoleCtrlEvent(CTRL_C_EVENT, 0)
                .context("failed to send CTRL_C_EVENT to the child's console");
            FreeConsole().log_err();
            result
        }
    }

    /// Test-only variant: the test harness process owns a console (so
    /// `AttachConsole` would fail); detach from it first and re-attach to the
    /// parent's console afterwards.
    #[cfg(test)]
    pub(crate) fn send_ctrl_c_detaching_own_console(pid: u32) -> Result<()> {
        use windows::Win32::System::Console::ATTACH_PARENT_PROCESS;

        unsafe {
            FreeConsole().log_err();
        }
        let result = send_ctrl_c(pid);
        unsafe {
            AttachConsole(ATTACH_PARENT_PROCESS).log_err();
        }
        result
    }

    /// A Win32 auto-reset event used to interrupt a locally-spawned Jupyter
    /// kernel. The handle is created inheritable and its numeric value is
    /// passed to the kernel via the `JPY_INTERRUPT_EVENT` environment variable;
    /// ipykernel's Windows parent poller waits on it and raises
    /// `KeyboardInterrupt` in the kernel when it is signaled. This mirrors how
    /// jupyter_client interrupts kernels on Windows, where message-based
    /// interrupts over the control channel are not honored.
    pub(crate) struct InterruptEvent(HANDLE);

    // SAFETY: event handles can be used from any thread.
    unsafe impl Send for InterruptEvent {}
    unsafe impl Sync for InterruptEvent {}

    impl InterruptEvent {
        pub(crate) fn new() -> Result<Self> {
            unsafe {
                let attributes = SECURITY_ATTRIBUTES {
                    nLength: std::mem::size_of::<SECURITY_ATTRIBUTES>() as u32,
                    lpSecurityDescriptor: std::ptr::null_mut(),
                    bInheritHandle: true.into(),
                };
                let handle = CreateEventW(
                    Some(&attributes),
                    false, // auto-reset
                    false, // initially non-signaled
                    windows::core::PCWSTR::null(),
                )
                .context("failed to create interrupt event")?;
                Ok(Self(handle))
            }
        }

        /// The handle value to hand to the child via `JPY_INTERRUPT_EVENT`.
        pub(crate) fn env_value(&self) -> String {
            (self.0.0 as isize).to_string()
        }

        pub(crate) fn signal(&self) -> Result<()> {
            unsafe { SetEvent(self.0).context("failed to signal interrupt event") }
        }
    }

    impl Drop for InterruptEvent {
        fn drop(&mut self) {
            unsafe {
                CloseHandle(self.0).ok();
            }
        }
    }
}

#[cfg(windows)]
mod windows_job {
    use crate::ResultExt as _;
    use anyhow::{Context as _, Result};
    use windows::Win32::{
        Foundation::{CloseHandle, HANDLE},
        System::{
            JobObjects::{
                AssignProcessToJobObject, CreateJobObjectW, JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
                JOBOBJECT_BASIC_ACCOUNTING_INFORMATION, JOBOBJECT_EXTENDED_LIMIT_INFORMATION,
                JobObjectBasicAccountingInformation, JobObjectExtendedLimitInformation,
                QueryInformationJobObject, SetInformationJobObject, TerminateJobObject,
            },
            Threading::{OpenProcess, PROCESS_SET_QUOTA, PROCESS_TERMINATE},
        },
    };

    /// A Win32 job object configured with `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`:
    /// all processes assigned to the job (and their descendants) are terminated
    /// when the last handle to the job is closed, which happens when this struct
    /// is dropped, or when the OS closes the owning process's handles after it
    /// exits for any reason.
    pub(crate) struct JobObject(HANDLE);

    // SAFETY: Job object handles can be used from any thread.
    unsafe impl Send for JobObject {}
    unsafe impl Sync for JobObject {}

    impl JobObject {
        pub(crate) fn new() -> Result<Self> {
            unsafe {
                let job =
                    Self(CreateJobObjectW(None, None).context("failed to create job object")?);
                let mut info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION::default();
                info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
                SetInformationJobObject(
                    job.0,
                    JobObjectExtendedLimitInformation,
                    &info as *const _ as *const _,
                    size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
                )
                .context("failed to set job object limits")?;
                Ok(job)
            }
        }

        pub(crate) fn assign_process(&self, pid: u32) -> Result<()> {
            unsafe {
                let process = OpenProcess(PROCESS_SET_QUOTA | PROCESS_TERMINATE, false, pid)
                    .context("failed to open process")?;
                let result = AssignProcessToJobObject(self.0, process)
                    .context("failed to assign process to job object");
                CloseHandle(process).log_err();
                result
            }
        }

        pub(crate) fn terminate(&self) -> Result<()> {
            unsafe { TerminateJobObject(self.0, 1).context("failed to terminate job object") }
        }

        pub(crate) fn has_running_processes(&self) -> Result<bool> {
            let mut accounting = JOBOBJECT_BASIC_ACCOUNTING_INFORMATION::default();
            // SAFETY: self owns a valid job handle, and accounting is writable
            // for the exact buffer size passed to this synchronous query.
            unsafe {
                QueryInformationJobObject(
                    Some(self.0),
                    JobObjectBasicAccountingInformation,
                    &mut accounting as *mut _ as *mut _,
                    size_of::<JOBOBJECT_BASIC_ACCOUNTING_INFORMATION>() as u32,
                    None,
                )
                .context("failed to query active job processes")?;
            }
            Ok(accounting.ActiveProcesses > 0)
        }
    }

    impl Drop for JobObject {
        fn drop(&mut self) {
            unsafe {
                CloseHandle(self.0).log_err();
            }
        }
    }
}

#[cfg(all(test, windows))]
mod windows_tests {
    use super::*;
    use std::time::{Duration, Instant};

    #[test]
    fn test_unmanaged_child_reports_unavailable_job_activity() {
        let mut command = gpui_util::new_std_command("ping.exe");
        command.args(["-n", "60", "127.0.0.1"]);
        let process = smol::process::Command::from(command)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .kill_on_drop(true)
            .spawn()
            .expect("failed to spawn unmanaged process");
        let mut child = Child { process, job: None };

        let pid = child.id();
        assert!(
            process_is_alive(pid),
            "direct child should still be running"
        );
        assert_eq!(
            child.has_running_processes().expect("failed to query job"),
            None
        );
        child
            .kill()
            .expect("failed to kill unassigned direct child");
        smol::block_on(child.status()).expect("failed to reap direct child");
        assert_process_exits(pid, "direct child should be terminated");
    }

    #[test]
    fn test_job_tracks_grandchild_after_direct_child_exits() {
        let temp_dir = tempfile::tempdir().unwrap();
        let pid_file = temp_dir.path().join("grandchild_pid");
        let mut command = gpui_util::new_std_command("powershell.exe");
        command.args(["-NoProfile", "-Command"]).arg(format!(
            "$p = Start-Process -FilePath ping.exe -ArgumentList @('-n','60','127.0.0.1') -PassThru -WindowStyle Hidden; \
             Set-Content -LiteralPath '{}' -Value $p.Id",
            pid_file.display()
        ));
        let mut child = Child::spawn(command, Stdio::null(), Stdio::null(), Stdio::null())
            .expect("failed to spawn powershell");
        let status = smol::block_on(child.status()).expect("failed to await powershell");
        assert!(status.success(), "powershell failed: {status}");

        let grandchild_pid = std::fs::read_to_string(&pid_file)
            .expect("powershell did not record grandchild pid")
            .trim()
            .parse::<u32>()
            .expect("invalid grandchild pid");
        assert!(process_is_alive(grandchild_pid));
        assert_eq!(
            child.has_running_processes().expect("failed to query job"),
            Some(true),
            "job should remain active after its direct child exits"
        );

        child.kill().expect("failed to terminate job");
        let deadline = Instant::now() + Duration::from_secs(2);
        while child.has_running_processes().expect("failed to query job") != Some(false) {
            assert!(Instant::now() < deadline, "job did not become inactive");
            std::thread::sleep(Duration::from_millis(50));
        }
        assert_process_exits(grandchild_pid, "job grandchild should be terminated");
    }

    /// Spawns a process tree `powershell -> ping` via `Child::spawn` and
    /// returns the `Child` along with the pid of the grandchild (`ping`).
    fn spawn_process_tree(temp_dir: &std::path::Path) -> (Child, u32) {
        let pid_file = temp_dir.join("grandchild_pid");
        let mut command = std::process::Command::new("powershell.exe");
        command.args(["-NoProfile", "-Command"]).arg(format!(
            "$p = Start-Process -FilePath ping.exe -ArgumentList @('-n','60','127.0.0.1') -PassThru -WindowStyle Hidden; \
             Set-Content -LiteralPath '{}' -Value $p.Id; \
             Wait-Process -Id $p.Id",
            pid_file.display()
        ));
        let child = Child::spawn(command, Stdio::null(), Stdio::null(), Stdio::null())
            .expect("failed to spawn powershell");

        let deadline = Instant::now() + Duration::from_secs(5);
        let grandchild_pid = loop {
            if let Ok(contents) = std::fs::read_to_string(&pid_file)
                && let Ok(pid) = contents.trim().parse::<u32>()
            {
                break pid;
            }
            assert!(
                Instant::now() < deadline,
                "timed out waiting for grandchild pid file"
            );
            std::thread::sleep(Duration::from_millis(50));
        };
        assert!(
            process_is_alive(grandchild_pid),
            "grandchild should be alive after spawning"
        );
        (child, grandchild_pid)
    }

    fn process_is_alive(pid: u32) -> bool {
        use windows::Win32::{
            Foundation::{CloseHandle, STILL_ACTIVE},
            System::Threading::{
                GetExitCodeProcess, OpenProcess, PROCESS_QUERY_LIMITED_INFORMATION,
            },
        };

        unsafe {
            let Ok(handle) = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, false, pid) else {
                return false;
            };
            let mut exit_code = 0u32;
            let alive = GetExitCodeProcess(handle, &mut exit_code).is_ok()
                && exit_code == STILL_ACTIVE.0 as u32;
            CloseHandle(handle).expect("failed to close process handle");
            alive
        }
    }

    fn assert_process_exits(pid: u32, message: &str) {
        let deadline = Instant::now() + Duration::from_secs(2);
        while process_is_alive(pid) {
            assert!(Instant::now() < deadline, "{message} (pid {pid})");
            std::thread::sleep(Duration::from_millis(100));
        }
    }

    #[test]
    fn test_console_ctrl_c_interrupts_child() {
        // ping.exe knows nothing about the JPY interrupt event, so it only
        // stops if the real CTRL_C console event lands; this exercises the
        // console path end-to-end. Spawn via `new_std_command` to get
        // CREATE_NO_WINDOW like production kernels — ping gets its own hidden
        // console, so the CTRL_C cannot reach the test runner's console.
        let mut command = crate::command::new_std_command("ping.exe");
        command.args(["-n", "60", "127.0.0.1"]);
        let child = Child::spawn_interruptible(command, Stdio::null(), Stdio::null(), Stdio::null())
            .expect("failed to spawn ping");
        let pid = child.id();
        assert!(process_is_alive(pid), "ping should be alive after spawning");

        windows_interrupt::send_ctrl_c_detaching_own_console(pid)
            .expect("failed to send console CTRL_C");

        assert_process_exits(pid, "ping should exit after a console CTRL_C");
        drop(child);
    }

    #[test]
    fn test_kill_terminates_grandchildren() {
        let temp_dir = tempfile::tempdir().unwrap();
        let (mut child, grandchild_pid) = spawn_process_tree(temp_dir.path());

        child.kill().expect("failed to kill child");

        assert_process_exits(
            grandchild_pid,
            "grandchild should be terminated after killing the child",
        );
    }

    #[test]
    fn test_drop_terminates_grandchildren() {
        let temp_dir = tempfile::tempdir().unwrap();
        let (child, grandchild_pid) = spawn_process_tree(temp_dir.path());

        drop(child);

        assert_process_exits(
            grandchild_pid,
            "grandchild should be terminated after dropping the child",
        );
    }
}
