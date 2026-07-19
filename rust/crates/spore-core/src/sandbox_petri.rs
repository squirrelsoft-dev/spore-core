//! MicroVM sandbox provider backed by the petri runtime (feature `petri`).
//!
//! [`PetriSandboxProvider`] splits the two halves of [`SandboxProvider`]:
//!
//! - **Path resolution / file I/O** delegates to an embedded
//!   [`WorkspaceScopedSandbox`], so file tools do host-side I/O against the
//!   granted workspace directory with the exact same boundary policy.
//! - **Command execution** dispatches into a petri microVM whose guest sees
//!   the same workspace over virtio-fs at `/workspace`. The VM boundary is the
//!   process/network isolation; there is no sync layer.
//!
//! The VM is created lazily on the first `execute_command` and torn down via
//! [`PetriSandboxProvider::shutdown`] (or best-effort on drop). Boot hangs are
//! handled with a create timeout plus one abandon-and-recreate retry; every
//! sandbox is tagged with a session id in its metadata so orphans from a
//! crashed host process can be reaped with [`reap_session`].
//!
//! Every teardown path above needs the host process to still be alive to run
//! it, so none of them fire when it is killed outright (SIGKILL, force-quit,
//! a panic). A VM that outlives its host reparents to init and keeps holding
//! its disk open, which breaks the *next* run rather than the one that leaked
//! it. Sandboxes therefore also record their owner's pid, letting
//! [`reap_orphans`] collect VMs whose owner is gone — call it at host startup,
//! where any such VM is by definition an orphan.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Duration;

use petri::instance::GUEST_WORKSPACE_PATH;
use petri::{
    CommandOptions as PetriCommandOptions, HostBackend, InstanceId, MacosBackend, Sandbox,
    SandboxOptions, Status,
};

use crate::harness::{
    BoxFut, CommandOutput, CommandProgressSink, IsolationMode, NetworkPolicy, Operation,
    SandboxProvider, SandboxViolation, TruncatedOutput,
};
use crate::model::ToolCall;
use crate::sandbox::{
    sanitize_command_output, BuildError, WorkspaceConfig, WorkspaceScopedSandbox,
};

/// Metadata key stamped on every sandbox this module creates, holding the
/// caller's session tag. [`reap_session`] filters on it.
pub const SESSION_METADATA_KEY: &str = "spore-session";

/// Metadata key stamped on every sandbox this module creates, holding the pid
/// of the host process that owns it. [`reap_orphans`] filters on it. The
/// instance's own `pid` field is the VM helper's and stays alive after the
/// owner dies, so it cannot tell an orphan from a working VM; this can.
pub const HOST_PID_METADATA_KEY: &str = "spore-host-pid";

/// Command posture for a generated petri policy. Petri's guest enforces an
/// allowlist on the *dispatched binary*, so postures below
/// [`Unrestricted`](PetriCommandPolicy::Unrestricted) reject shells — spore's
/// `bash_command` tool (`/bin/sh -c …`) only works under `Unrestricted`, while
/// the argv-style `exec` tool works at every level. The VM boundary is the
/// real isolation at every posture.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PetriCommandPolicy {
    /// Inspection commands only (`ls`, `cat`, `rg`, …). No mutation, no shell.
    ReadOnly,
    /// Read-only plus common build/file-mutation tools (`cargo`, `git`,
    /// `make`, …). Still no arbitrary shell.
    Edit,
    /// Arbitrary command execution, including shells. The microVM (plus the
    /// network toggle) is the only boundary.
    Unrestricted,
}

/// A generated petri policy: the command posture crossed with spore's
/// [`NetworkPolicy`], rendered to petri's policy TOML by [`Self::to_toml`].
///
/// Petri's network axis is currently a boolean (`network_enabled`); host
/// allowlists land with petri's ADR 0002. Until then
/// [`NetworkPolicy::Allowlist`] maps to *enabled* — callers that need a hard
/// no-egress guarantee must use [`NetworkPolicy::None`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PetriPolicy {
    pub command: PetriCommandPolicy,
    pub network: NetworkPolicy,
    /// Per-command wall-clock ceiling enforced by the guest.
    pub max_runtime_secs: u64,
    /// Per-command captured-output ceiling enforced by the guest.
    pub max_output_bytes: u64,
}

impl PetriPolicy {
    /// A policy with petri-template-like defaults for the given posture.
    pub fn new(command: PetriCommandPolicy, network: NetworkPolicy) -> Self {
        Self {
            command,
            network,
            max_runtime_secs: 600,
            max_output_bytes: 4_194_304,
        }
    }

    /// Render the policy as petri policy TOML. The command level boots at its
    /// ceiling (`default == max`) because spore never issues `set_mode`
    /// escalations.
    pub fn to_toml(&self) -> String {
        let network_enabled = !matches!(self.network, NetworkPolicy::None);
        let command_section = match self.command {
            PetriCommandPolicy::ReadOnly => concat!(
                "default = \"read_only\"\n",
                "max = \"read_only\"\n",
                "read_only = [\"ls\", \"cat\", \"rg\", \"grep\", \"find\", \"head\", \"tail\", ",
                "\"wc\", \"stat\", \"file\", \"tree\", \"diff\"]\n",
            )
            .to_string(),
            PetriCommandPolicy::Edit => concat!(
                "default = \"edit\"\n",
                "max = \"edit\"\n",
                "read_only = [\"ls\", \"cat\", \"rg\", \"grep\", \"find\", \"head\", \"tail\", ",
                "\"wc\", \"stat\", \"file\", \"tree\", \"diff\"]\n",
                "edit = [\"cargo\", \"rustc\", \"git\", \"make\", \"sed\", \"tee\", \"mkdir\", ",
                "\"cp\", \"mv\", \"touch\", \"rm\", \"tar\", \"npm\", \"pnpm\", \"node\", ",
                "\"python3\", \"pip3\"]\n",
            )
            .to_string(),
            PetriCommandPolicy::Unrestricted => "default = \"yolo\"\nmax = \"yolo\"\n".to_string(),
        };
        format!(
            "# Generated by spore-core sandbox_petri — do not edit.\n\
             [policy]\n\
             network_enabled = {network_enabled}\n\
             max_runtime_secs = {}\n\
             max_output_bytes = {}\n\
             workspace_path = \"{GUEST_WORKSPACE_PATH}\"\n\
             \n\
             [policy.command]\n\
             {command_section}",
            self.max_runtime_secs, self.max_output_bytes,
        )
    }

    /// Write the rendered TOML into `dir` and return the file path. Creates
    /// `dir` if missing.
    pub fn write_to(&self, dir: &Path) -> std::io::Result<PathBuf> {
        std::fs::create_dir_all(dir)?;
        let path = dir.join("spore-petri-policy.toml");
        std::fs::write(&path, self.to_toml())?;
        Ok(path)
    }
}

/// VM lifecycle knobs for [`PetriSandboxProvider`].
#[derive(Debug, Clone)]
pub struct PetriVmOptions {
    /// Image bundle path. `None` uses the backend's default base image (or the
    /// loopback fallback in dev).
    pub image: Option<PathBuf>,
    /// Root directory for per-VM scratch overlays. See
    /// [`PetriVmOptions::with_scratch_root`].
    pub scratch_root: Option<PathBuf>,
    /// Policy TOML applied at boot (see [`PetriPolicy::write_to`]).
    pub policy_path: PathBuf,
    /// Network policy recorded in [`IsolationMode::MicroVm`]. Enforcement
    /// happens in the guest via the policy file; this field is what
    /// observability and middleware route on.
    pub network: NetworkPolicy,
    /// How long one `Sandbox::create` may take before the boot is presumed
    /// hung, the instance is abandoned (torn down in the background), and one
    /// fresh create is attempted.
    pub create_timeout: Duration,
    /// Session tag stamped into sandbox metadata for orphan reaping.
    pub session_tag: String,
}

impl PetriVmOptions {
    pub fn new(policy_path: impl Into<PathBuf>, session_tag: impl Into<String>) -> Self {
        Self {
            image: None,
            scratch_root: None,
            policy_path: policy_path.into(),
            network: NetworkPolicy::None,
            create_timeout: Duration::from_secs(120),
            session_tag: session_tag.into(),
        }
    }

    pub fn with_image(mut self, image: impl Into<PathBuf>) -> Self {
        self.image = Some(image.into());
        self
    }

    /// Boot the image read-only under a per-VM ephemeral scratch overlay,
    /// rooted at `scratch_root`. Requires [`with_image`](Self::with_image).
    ///
    /// Without this the VM attaches the image bundle's disk *directly and
    /// writably*: every run mutates the shared base image, and two VMs on one
    /// image corrupt each other. With it, the base is read-only and each VM
    /// gets its own overlay under `scratch_root/<vm-id>`, discarded on
    /// shutdown.
    ///
    /// The overlay is served in-process, so it dies with this process — a VM
    /// that outlives its host loses its disk instead of holding the base image
    /// open (see [`reap_orphans`] for collecting the VM itself).
    pub fn with_scratch_root(mut self, scratch_root: impl Into<PathBuf>) -> Self {
        self.scratch_root = Some(scratch_root.into());
        self
    }

    pub fn with_network(mut self, network: NetworkPolicy) -> Self {
        self.network = network;
        self
    }

    pub fn with_create_timeout(mut self, timeout: Duration) -> Self {
        self.create_timeout = timeout;
        self
    }
}

/// The microVM sandbox provider. Generic over petri's [`HostBackend`] so tests
/// can exercise the full wiring against an in-memory backend; production use
/// is [`PetriSandbox`] (the [`MacosBackend`] alias) via
/// [`PetriSandboxProvider::macos`].
pub struct PetriSandboxProvider<B: HostBackend + Clone + Send + Sync + 'static> {
    backend: B,
    /// Host-side path policy — identical boundary semantics to the canonical
    /// workspace sandbox. Its root doubles as the VM's mounted workspace.
    host: WorkspaceScopedSandbox,
    options: PetriVmOptions,
    vm: tokio::sync::Mutex<Option<Sandbox<B>>>,
}

/// The production provider type: petri's macOS Virtualization.framework
/// backend.
pub type PetriSandbox = PetriSandboxProvider<MacosBackend>;

impl PetriSandbox {
    /// Build a provider over petri's macOS backend. `state_dir` holds petri's
    /// instance state; `helper_binary` is the bundled `petri-vz` helper.
    pub fn macos(
        state_dir: impl Into<PathBuf>,
        helper_binary: impl Into<PathBuf>,
        workspace: WorkspaceConfig,
        options: PetriVmOptions,
    ) -> Result<Self, BuildError> {
        Self::with_backend(
            MacosBackend::new(state_dir, helper_binary),
            workspace,
            options,
        )
    }
}

impl<B: HostBackend + Clone + Send + Sync + 'static> PetriSandboxProvider<B> {
    /// Build a provider over an explicit backend (tests, alternate backends).
    pub fn with_backend(
        backend: B,
        workspace: WorkspaceConfig,
        options: PetriVmOptions,
    ) -> Result<Self, BuildError> {
        let host = WorkspaceScopedSandbox::new(workspace)?;
        Ok(Self {
            backend,
            host,
            options,
            vm: tokio::sync::Mutex::new(None),
        })
    }

    /// The sandbox id of the running VM, if one has been created.
    pub async fn vm_id(&self) -> Option<String> {
        self.vm
            .lock()
            .await
            .as_ref()
            .map(|sb| sb.id().as_str().to_string())
    }

    /// Tear down the VM if one is running. Idempotent. Call this on run
    /// completion/cancellation; `Drop` only makes a best-effort attempt.
    pub async fn shutdown(&self) {
        let sandbox = self.vm.lock().await.take();
        if let Some(sandbox) = sandbox {
            let _ = tokio::task::spawn_blocking(move || sandbox.kill()).await;
        }
    }

    /// Get the running VM handle, creating it on first use. Applies the
    /// create timeout with one abandon-and-recreate retry for boot hangs; an
    /// abandoned instance is torn down in the background.
    async fn ensure_vm(&self) -> Result<Sandbox<B>, SandboxViolation> {
        let mut slot = self.vm.lock().await;
        if let Some(sandbox) = slot.as_ref() {
            return Ok(sandbox.clone());
        }

        let mut last_error = String::from("petri VM boot timed out");
        for _attempt in 0..2 {
            let id = generate_vm_id(&self.options.session_tag);
            let mut sandbox_options =
                SandboxOptions::new(self.host.workspace_root(), &self.options.policy_path)
                    .with_id(id.clone())
                    .with_metadata(BTreeMap::from([
                        (
                            SESSION_METADATA_KEY.to_string(),
                            self.options.session_tag.clone(),
                        ),
                        (
                            HOST_PID_METADATA_KEY.to_string(),
                            std::process::id().to_string(),
                        ),
                    ]));
            if let Some(image) = &self.options.image {
                sandbox_options = sandbox_options.with_image(image.clone());
            }
            if let Some(scratch_root) = &self.options.scratch_root {
                // Keyed by VM id, not session: the retry below mints a fresh id,
                // and an abandoned boot may still be holding its overlay.
                sandbox_options = sandbox_options.with_scratch_overlay(scratch_root.join(&id));
            }
            let backend = self.backend.clone();
            let create =
                tokio::task::spawn_blocking(move || Sandbox::create(backend, sandbox_options));
            match tokio::time::timeout(self.options.create_timeout, create).await {
                Ok(Ok(Ok(sandbox))) => {
                    *slot = Some(sandbox.clone());
                    return Ok(sandbox);
                }
                Ok(Ok(Err(e))) => {
                    return Err(SandboxViolation::ExecSpawnFailed {
                        command: "petri sandbox create".to_string(),
                        message: e.to_string(),
                    });
                }
                Ok(Err(join_error)) => {
                    return Err(SandboxViolation::ExecSpawnFailed {
                        command: "petri sandbox create".to_string(),
                        message: format!("create task panicked: {join_error}"),
                    });
                }
                Err(_elapsed) => {
                    // Boot hang: the blocking create may still complete later,
                    // so reap the abandoned instance in the background and try
                    // once more with a fresh id.
                    last_error = format!(
                        "petri VM '{id}' did not boot within {:?}; abandoned",
                        self.options.create_timeout
                    );
                    let backend = self.backend.clone();
                    if let Ok(instance_id) = InstanceId::new(id) {
                        tokio::task::spawn_blocking(move || {
                            let _ = backend.teardown(&instance_id);
                        });
                    }
                }
            }
        }
        Err(SandboxViolation::ExecSpawnFailed {
            command: "petri sandbox create".to_string(),
            message: last_error,
        })
    }

    /// Map a working directory onto the guest workspace mount. Models mix
    /// host and guest views freely, so all three spellings are accepted:
    /// `None` (and the workspace root itself) become the guest workspace, a
    /// guest-style `/workspace/...` path passes through, a relative path
    /// joins the mount, and a host path under the root translates to
    /// `/workspace/<relative>`. Anything else — including `..` traversal in
    /// the guest/relative forms — is a path escape.
    fn guest_cwd(&self, working_dir: Option<&Path>) -> Result<PathBuf, SandboxViolation> {
        use std::path::Component;
        let root = self.host.workspace_root();
        let Some(dir) = working_dir else {
            return Ok(PathBuf::from(GUEST_WORKSPACE_PATH));
        };
        let escape = || SandboxViolation::PathEscape {
            path: dir.display().to_string(),
        };
        let has_parent_component = dir.components().any(|c| matches!(c, Component::ParentDir));
        if dir.starts_with(GUEST_WORKSPACE_PATH) {
            if has_parent_component {
                return Err(escape());
            }
            return Ok(dir.to_path_buf());
        }
        if dir.is_relative() {
            if has_parent_component {
                return Err(escape());
            }
            return Ok(Path::new(GUEST_WORKSPACE_PATH).join(dir));
        }
        let relative = dir.strip_prefix(root).map_err(|_| escape())?;
        Ok(Path::new(GUEST_WORKSPACE_PATH).join(relative))
    }
}

impl<B: HostBackend + Clone + Send + Sync + 'static> SandboxProvider for PetriSandboxProvider<B> {
    fn validate<'a>(&'a self, call: &'a ToolCall) -> BoxFut<'a, Result<(), SandboxViolation>> {
        self.host.validate(call)
    }

    fn execute_command<'a>(
        &'a self,
        command: &'a str,
        args: &'a [String],
        working_dir: Option<&'a Path>,
        timeout: Option<Duration>,
    ) -> BoxFut<'a, Result<CommandOutput, SandboxViolation>> {
        self.execute_command_streaming(command, args, working_dir, timeout, None)
    }

    /// Dispatch into the guest. v1 buffers output — petri's dispatch protocol
    /// returns complete streams, so `progress` is accepted but unused until
    /// petri grows incremental output.
    fn execute_command_streaming<'a>(
        &'a self,
        command: &'a str,
        args: &'a [String],
        working_dir: Option<&'a Path>,
        timeout: Option<Duration>,
        _progress: Option<CommandProgressSink>,
    ) -> BoxFut<'a, Result<CommandOutput, SandboxViolation>> {
        Box::pin(async move {
            let cwd = self.guest_cwd(working_dir)?;
            let sandbox = self.ensure_vm().await?;
            // Petri's guest requires a bare executable name (resolved via the
            // guest's PATH) and rejects anything with a path separator. Host
            // paths like the bash tool's `/bin/sh` are meaningless inside the
            // VM anyway, so dispatch the basename.
            let command_owned = Path::new(command)
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or(command)
                .to_string();
            // Models routinely embed the HOST workspace path in scripts (it
            // is the path their prompt names). Inside the guest that path
            // does not exist — the workspace is mounted at /workspace — so
            // rewrite host-root references onto the mount. A literal host
            // path is never correct in-guest, so the substitution cannot
            // break a working script.
            let host_root = self.host.workspace_root().display().to_string();
            let guest_args: Vec<String> = args
                .iter()
                .map(|a| a.replace(&host_root, GUEST_WORKSPACE_PATH))
                .collect();
            let petri_options = PetriCommandOptions {
                cwd: Some(cwd),
                args: guest_args,
                timeout_ms: timeout.map(|t| t.as_millis() as u64),
                ..PetriCommandOptions::default()
            };
            let dispatch_command = command_owned.clone();
            let result = tokio::task::spawn_blocking(move || {
                sandbox.commands().run(dispatch_command, petri_options)
            })
            .await
            .map_err(|join_error| SandboxViolation::ExecSpawnFailed {
                command: command_owned.clone(),
                message: format!("dispatch task panicked: {join_error}"),
            })?
            .map_err(|e| SandboxViolation::ExecSpawnFailed {
                command: command_owned.clone(),
                message: e.to_string(),
            })?;

            let error_message = result
                .error
                .as_ref()
                .map(|e| e.message.clone())
                .unwrap_or_default();
            match result.status {
                Status::Success | Status::Failure => Ok(CommandOutput {
                    stdout: sanitize_command_output(&result.stdout),
                    stderr: sanitize_command_output(&result.stderr),
                    exit_code: result.exit_code.unwrap_or(-1),
                    timed_out: false,
                    truncated: result.output_truncated,
                }),
                Status::Timeout => {
                    let mut stderr = sanitize_command_output(&result.stderr);
                    if !stderr.is_empty() {
                        stderr.push('\n');
                    }
                    let secs = timeout.map(|t| t.as_secs()).unwrap_or(0);
                    stderr.push_str(&format!("command timed out after {secs}s"));
                    Ok(CommandOutput {
                        stdout: sanitize_command_output(&result.stdout),
                        stderr,
                        exit_code: -1,
                        timed_out: true,
                        truncated: result.output_truncated,
                    })
                }
                Status::Rejected => Err(SandboxViolation::DisallowedCommand {
                    command: format!("{command_owned} (petri policy: {error_message})"),
                }),
                Status::Cancelled | Status::Malformed => Err(SandboxViolation::ExecSpawnFailed {
                    command: command_owned,
                    message: format!("petri dispatch {:?}: {error_message}", result.status),
                }),
            }
        })
    }

    fn handle_large_output<'a>(
        &'a self,
        content: String,
        call_id: &'a str,
        head_tokens: u32,
        tail_tokens: u32,
    ) -> BoxFut<'a, TruncatedOutput> {
        self.host
            .handle_large_output(content, call_id, head_tokens, tail_tokens)
    }

    fn resolve_path<'a>(
        &'a self,
        path: &'a str,
        operation: Operation,
    ) -> BoxFut<'a, Result<PathBuf, SandboxViolation>> {
        self.host.resolve_path(path, operation)
    }

    fn isolation_mode(&self) -> IsolationMode {
        IsolationMode::MicroVm {
            image: self
                .options
                .image
                .as_ref()
                .map(|p| p.display().to_string())
                .unwrap_or_default(),
            network: self.options.network.clone(),
        }
    }

    fn workspace_root(&self) -> &Path {
        self.host.workspace_root()
    }

    fn max_write_size(&self) -> Option<u64> {
        self.host.max_write_size()
    }
}

impl<B: HostBackend + Clone + Send + Sync + 'static> Drop for PetriSandboxProvider<B> {
    /// Best-effort teardown for providers dropped without [`shutdown`]. The
    /// kill runs on a detached thread so drop never blocks an async runtime;
    /// [`reap_session`] covers the case where the process dies first.
    fn drop(&mut self) {
        if let Some(sandbox) = self.vm.get_mut().take() {
            std::thread::spawn(move || {
                let _ = sandbox.kill();
            });
        }
    }
}

/// Whether the petri runtime looks usable: the helper binary exists and, when
/// given, the image bundle path exists. A cheap doctor probe — it does not
/// boot anything.
pub fn petri_available(helper_binary: &Path, image: Option<&Path>) -> bool {
    helper_binary.is_file() && image.is_none_or(|p| p.exists())
}

/// Tear down every sandbox whose metadata carries `session_tag` under
/// [`SESSION_METADATA_KEY`] — orphans left by a previous host process. Returns
/// the number of instances reaped. Blocking; wrap in `spawn_blocking` from
/// async contexts.
pub fn reap_session<B: HostBackend>(backend: &B, session_tag: &str) -> usize {
    let Ok(handles) = backend.list() else {
        return 0;
    };
    let mut reaped = 0;
    for handle in handles {
        if handle
            .metadata
            .get(SESSION_METADATA_KEY)
            .map(String::as_str)
            == Some(session_tag)
            && backend.teardown(&handle.id).is_ok()
        {
            reaped += 1;
        }
    }
    reaped
}

/// Tear down every sandbox this module created whose owning host process is
/// no longer alive — VMs leaked by a host that died without running its
/// teardown. Returns the number of instances reaped. Blocking; wrap in
/// `spawn_blocking` from async contexts.
///
/// Safe to call while other hosts are running: a sandbox whose owner is still
/// alive is left alone, so concurrent instances never reap each other. Call it
/// at startup, before creating any sandbox of your own.
///
/// A sandbox with no recorded owner predates [`HOST_PID_METADATA_KEY`] and is
/// treated as an orphan — nothing tracks it, so no live host can lose a VM it
/// is using.
pub fn reap_orphans<B: HostBackend>(backend: &B) -> usize {
    let Ok(handles) = backend.list() else {
        return 0;
    };
    let mut reaped = 0;
    for handle in handles {
        if !handle.metadata.contains_key(SESSION_METADATA_KEY) {
            continue; // not ours to reap
        }
        let owner_alive = handle
            .metadata
            .get(HOST_PID_METADATA_KEY)
            .and_then(|pid| pid.parse::<u32>().ok())
            .is_some_and(host_process_alive);
        if owner_alive {
            continue;
        }
        if backend.teardown(&handle.id).is_ok() {
            reaped += 1;
        }
    }
    reaped
}

/// Delete scratch overlays left behind by VMs that are no longer running,
/// under a `scratch_root` previously given to
/// [`PetriVmOptions::with_scratch_root`]. Returns the number removed.
/// Blocking; wrap in `spawn_blocking` from async contexts.
///
/// `Sandbox::kill` discards a scratch on the way out, so this only cleans up
/// after hosts that died without getting the chance — where the overlay
/// survives holding however many gigabytes the guest had written.
///
/// **Call this after [`reap_orphans`], never before.** It spares any scratch
/// belonging to an instance the backend still lists, which is what stops it
/// deleting a *live* VM's disk out from under a concurrently running host.
/// Reaping first is what makes that list mean "still alive" rather than "not
/// yet cleaned up".
pub fn reap_orphan_scratch<B: HostBackend>(backend: &B, scratch_root: &Path) -> usize {
    let live: std::collections::HashSet<String> = backend
        .list()
        .map(|handles| {
            handles
                .into_iter()
                .map(|handle| handle.id.as_str().to_string())
                .collect()
        })
        .unwrap_or_default();

    let Ok(entries) = std::fs::read_dir(scratch_root) else {
        return 0; // no scratch root yet — nothing has booted with an overlay
    };
    let mut reaped = 0;
    for entry in entries.flatten() {
        // Directories are named for the VM that owns them (`ensure_vm`).
        let owned_by_live_vm = entry
            .file_name()
            .to_str()
            .is_some_and(|name| live.contains(name));
        if !owned_by_live_vm && std::fs::remove_dir_all(entry.path()).is_ok() {
            reaped += 1;
        }
    }
    reaped
}

/// What a startup reap collected.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct ReapCounts {
    /// Orphaned VMs torn down.
    pub vms: usize,
    /// Orphaned scratch overlays deleted.
    pub scratch: usize,
}

impl ReapCounts {
    /// Whether anything was collected — i.e. a previous host did leak.
    pub fn any(&self) -> bool {
        self.vms > 0 || self.scratch > 0
    }
}

/// [`reap_orphans`] + [`reap_orphan_scratch`] against petri's macOS backend —
/// the production pairing, for callers that hold the same `state_dir` /
/// `helper_binary` they would pass to [`PetriSandbox::macos`] and have no
/// reason to name a backend type. Pass the same `scratch_root` given to
/// [`PetriVmOptions::with_scratch_root`], or `None` if overlays are not in use.
///
/// Does both halves in the only safe order, so callers cannot get it wrong.
pub fn reap_macos_orphans(
    state_dir: impl Into<PathBuf>,
    helper_binary: impl Into<PathBuf>,
    scratch_root: Option<&Path>,
) -> ReapCounts {
    let backend = MacosBackend::new(state_dir, helper_binary);
    let vms = reap_orphans(&backend);
    // Strictly after the VM reap: only then does `list` mean "still alive", the
    // property that keeps a live VM's overlay safe.
    let scratch = scratch_root.map_or(0, |root| reap_orphan_scratch(&backend, root));
    ReapCounts { vms, scratch }
}

/// Whether `pid` names a live process.
///
/// Pids are recycled, so a dead owner's pid can be reused by an unrelated
/// process and report alive. That only costs us a missed reap this time
/// around, which the next startup retries — the failure we must avoid is the
/// opposite one, reporting a live owner dead and killing a VM in use.
#[cfg(unix)]
fn host_process_alive(pid: u32) -> bool {
    // kill(0, …) addresses the caller's whole process group rather than one
    // process, which would report our own group alive and strand the orphan.
    if pid == 0 {
        return false;
    }
    // SAFETY: kill(2) takes a pid and a signal number and has no
    // memory-safety preconditions. Signal 0 runs the error checks without
    // delivering anything.
    if unsafe { libc::kill(pid as libc::pid_t, 0) } == 0 {
        return true;
    }
    // EPERM means the process exists but belongs to another user; only ESRCH
    // proves it is gone.
    std::io::Error::last_os_error().raw_os_error() == Some(libc::EPERM)
}

#[cfg(not(unix))]
fn host_process_alive(_pid: u32) -> bool {
    // No cheap liveness probe here, so assume alive and never reap: leaking a
    // VM is recoverable, killing a live one is not.
    true
}

fn generate_vm_id(session_tag: &str) -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let safe_tag: String = session_tag
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect();
    format!("spore-{safe_tag}-{nanos}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use petri::instance::{InstanceConfig, InstanceHandle, LifecycleState};
    use petri::{DispatchRequest, DispatchResult};
    use std::sync::{Arc, Mutex};
    use tempfile::TempDir;

    /// In-memory [`HostBackend`] recording lifecycle + dispatch traffic so the
    /// provider's wiring is testable without a VM.
    #[derive(Debug, Clone, Default)]
    struct FakeBackend {
        state: Arc<Mutex<FakeState>>,
    }

    #[derive(Debug, Default)]
    struct FakeState {
        instances: Vec<InstanceHandle>,
        create_calls: usize,
        teardown_calls: Vec<String>,
        /// Ids torn down before their (still in-flight) create completed. A
        /// real backend registers state early enough that teardown lands; the
        /// fake models the same outcome by suppressing the late registration.
        tombstones: std::collections::HashSet<String>,
        last_request: Option<DispatchRequest>,
        next_result: Option<DispatchResult>,
        create_delay: Option<Duration>,
        /// The most recent config handed to `create`, so tests can assert how
        /// the VM was asked to boot (image bundle vs NBD direct boot).
        last_config: Option<InstanceConfig>,
    }

    impl FakeBackend {
        fn set_next_result(&self, result: DispatchResult) {
            self.state.lock().unwrap().next_result = Some(result);
        }

        fn set_create_delay(&self, delay: Duration) {
            self.state.lock().unwrap().create_delay = Some(delay);
        }

        fn last_config(&self) -> InstanceConfig {
            self.state
                .lock()
                .unwrap()
                .last_config
                .clone()
                .expect("create was called")
        }
    }

    impl HostBackend for FakeBackend {
        fn name(&self) -> &str {
            "fake"
        }

        fn create(&self, config: InstanceConfig) -> petri::Result<InstanceHandle> {
            let delay = self.state.lock().unwrap().create_delay.take();
            if let Some(delay) = delay {
                std::thread::sleep(delay);
            }
            let mut state = self.state.lock().unwrap();
            state.create_calls += 1;
            state.last_config = Some(config.clone());
            let handle = InstanceHandle {
                id: config.id,
                backend: config.backend,
                state: LifecycleState::Ready,
                metadata: config.metadata,
            };
            if !state.tombstones.contains(handle.id.as_str()) {
                state.instances.push(handle.clone());
            }
            Ok(handle)
        }

        fn list(&self) -> petri::Result<Vec<InstanceHandle>> {
            Ok(self.state.lock().unwrap().instances.clone())
        }

        fn dispatch(
            &self,
            _instance_id: &InstanceId,
            request: DispatchRequest,
        ) -> petri::Result<DispatchResult> {
            let mut state = self.state.lock().unwrap();
            state.last_request = Some(request.clone());
            Ok(state.next_result.take().unwrap_or_else(|| {
                DispatchResult::process(
                    request.id,
                    Status::Success,
                    1,
                    "ok\n".to_string(),
                    String::new(),
                    Some(0),
                    false,
                )
            }))
        }

        fn stop(&self, _instance_id: &InstanceId) -> petri::Result<()> {
            Ok(())
        }

        fn teardown(&self, instance_id: &InstanceId) -> petri::Result<()> {
            let mut state = self.state.lock().unwrap();
            state.instances.retain(|h| &h.id != instance_id);
            state.teardown_calls.push(instance_id.as_str().to_string());
            state.tombstones.insert(instance_id.as_str().to_string());
            Ok(())
        }
    }

    fn provider(backend: FakeBackend, root: &Path) -> PetriSandboxProvider<FakeBackend> {
        PetriSandboxProvider::with_backend(
            backend,
            WorkspaceConfig::scoped(root.to_path_buf()),
            PetriVmOptions::new(root.join("policy.toml"), "test-session"),
        )
        .unwrap()
    }

    // ------------------------------------------------------------------
    // Policy TOML generation
    // ------------------------------------------------------------------

    #[test]
    fn policy_read_only_no_network() {
        let toml = PetriPolicy::new(PetriCommandPolicy::ReadOnly, NetworkPolicy::None).to_toml();
        assert!(toml.contains("network_enabled = false"));
        assert!(toml.contains("default = \"read_only\""));
        assert!(toml.contains("max = \"read_only\""));
        assert!(!toml.contains("yolo"));
    }

    #[test]
    fn policy_edit_boots_at_its_ceiling() {
        let toml = PetriPolicy::new(PetriCommandPolicy::Edit, NetworkPolicy::None).to_toml();
        assert!(toml.contains("default = \"edit\""));
        assert!(toml.contains("max = \"edit\""));
        assert!(toml.contains("\"cargo\""));
    }

    #[test]
    fn policy_unrestricted_full_network() {
        let toml =
            PetriPolicy::new(PetriCommandPolicy::Unrestricted, NetworkPolicy::Full).to_toml();
        assert!(toml.contains("network_enabled = true"));
        assert!(toml.contains("default = \"yolo\""));
    }

    #[test]
    fn policy_allowlist_maps_to_network_enabled() {
        // Petri has no host-allowlist axis yet; Allowlist must not silently
        // become no-network (bots that need one API host would break), so it
        // maps to enabled.
        let toml = PetriPolicy::new(
            PetriCommandPolicy::Unrestricted,
            NetworkPolicy::Allowlist {
                hosts: vec!["api.example.com".into()],
            },
        )
        .to_toml();
        assert!(toml.contains("network_enabled = true"));
    }

    #[test]
    fn policy_write_to_creates_file() {
        let dir = TempDir::new().unwrap();
        let path = PetriPolicy::new(PetriCommandPolicy::ReadOnly, NetworkPolicy::None)
            .write_to(&dir.path().join("policies"))
            .unwrap();
        let written = std::fs::read_to_string(&path).unwrap();
        assert!(written.contains("[policy.command]"));
    }

    // ------------------------------------------------------------------
    // VM lifecycle
    // ------------------------------------------------------------------

    #[tokio::test]
    async fn vm_is_created_lazily_and_reused() {
        let dir = TempDir::new().unwrap();
        let backend = FakeBackend::default();
        let sb = provider(backend.clone(), dir.path());

        assert_eq!(backend.state.lock().unwrap().create_calls, 0);
        sb.execute_command("echo", &["a".into()], None, None)
            .await
            .unwrap();
        sb.execute_command("echo", &["b".into()], None, None)
            .await
            .unwrap();
        assert_eq!(backend.state.lock().unwrap().create_calls, 1);
    }

    #[tokio::test]
    async fn vm_is_tagged_with_the_session() {
        let dir = TempDir::new().unwrap();
        let backend = FakeBackend::default();
        let sb = provider(backend.clone(), dir.path());
        sb.execute_command("echo", &[], None, None).await.unwrap();
        let state = backend.state.lock().unwrap();
        let handle = state.instances.first().unwrap();
        assert_eq!(
            handle
                .metadata
                .get(SESSION_METADATA_KEY)
                .map(String::as_str),
            Some("test-session")
        );
        assert!(handle.id.as_str().starts_with("spore-test-session-"));
    }

    #[tokio::test]
    async fn shutdown_tears_the_vm_down() {
        let dir = TempDir::new().unwrap();
        let backend = FakeBackend::default();
        let sb = provider(backend.clone(), dir.path());
        sb.execute_command("echo", &[], None, None).await.unwrap();
        assert_eq!(backend.state.lock().unwrap().instances.len(), 1);
        sb.shutdown().await;
        assert!(backend.state.lock().unwrap().instances.is_empty());
        // Idempotent.
        sb.shutdown().await;
    }

    #[tokio::test]
    async fn hung_boot_is_abandoned_and_recreated() {
        let dir = TempDir::new().unwrap();
        let backend = FakeBackend::default();
        // First create sleeps far past the timeout; the retry create is fast.
        backend.set_create_delay(Duration::from_millis(500));
        let sb = PetriSandboxProvider::with_backend(
            backend.clone(),
            WorkspaceConfig::scoped(dir.path().to_path_buf()),
            PetriVmOptions::new(dir.path().join("policy.toml"), "test-session")
                .with_create_timeout(Duration::from_millis(50)),
        )
        .unwrap();
        let out = sb.execute_command("echo", &[], None, None).await.unwrap();
        assert_eq!(out.exit_code, 0);
        // Give the abandoned create + background teardown time to finish,
        // then verify only the retry's VM survived.
        tokio::time::sleep(Duration::from_millis(700)).await;
        let state = backend.state.lock().unwrap();
        assert_eq!(state.create_calls, 2, "expected abandon + retry");
        assert_eq!(state.instances.len(), 1, "abandoned VM was not reaped");
    }

    // ------------------------------------------------------------------
    // Dispatch mapping
    // ------------------------------------------------------------------

    #[tokio::test]
    async fn cwd_defaults_to_the_guest_workspace() {
        let dir = TempDir::new().unwrap();
        let backend = FakeBackend::default();
        let sb = provider(backend.clone(), dir.path());
        sb.execute_command("ls", &[], None, None).await.unwrap();
        let state = backend.state.lock().unwrap();
        let request = state.last_request.clone().unwrap();
        let args = request.args.unwrap();
        assert_eq!(args["cwd"], GUEST_WORKSPACE_PATH);
        assert_eq!(args["command"], "ls");
    }

    #[tokio::test]
    async fn command_paths_dispatch_as_bare_executable_names() {
        // Petri's guest rejects commands containing a path separator; spore's
        // bash tool dispatches "/bin/sh". The provider must send the basename.
        let dir = TempDir::new().unwrap();
        let backend = FakeBackend::default();
        let sb = provider(backend.clone(), dir.path());
        sb.execute_command("/bin/sh", &["-c".into(), "echo hi".into()], None, None)
            .await
            .unwrap();
        let state = backend.state.lock().unwrap();
        let args = state.last_request.clone().unwrap().args.unwrap();
        assert_eq!(args["command"], "sh");
        assert_eq!(args["argv"], serde_json::json!(["-c", "echo hi"]));
    }

    #[tokio::test]
    async fn host_subdirectory_maps_onto_the_guest_mount() {
        let dir = TempDir::new().unwrap();
        let root = std::fs::canonicalize(dir.path()).unwrap();
        std::fs::create_dir(root.join("sub")).unwrap();
        let backend = FakeBackend::default();
        let sb = provider(backend.clone(), &root);
        sb.execute_command("ls", &[], Some(&root.join("sub")), None)
            .await
            .unwrap();
        let state = backend.state.lock().unwrap();
        let args = state.last_request.clone().unwrap().args.unwrap();
        assert_eq!(args["cwd"], format!("{GUEST_WORKSPACE_PATH}/sub"));
    }

    #[tokio::test]
    async fn host_workspace_paths_in_script_args_are_rewritten_to_the_mount() {
        // The regression: a model writes `cd /Users/me/Downloads && ...` in a
        // bash_command script because that is the path its prompt names. The
        // host root must be rewritten onto the guest mount before dispatch.
        let dir = TempDir::new().unwrap();
        let root = std::fs::canonicalize(dir.path()).unwrap();
        let backend = FakeBackend::default();
        let sb = provider(backend.clone(), &root);
        let script = format!(
            "cd {} && mv {}/a.txt {}/sub/a.txt",
            root.display(),
            root.display(),
            root.display()
        );
        sb.execute_command("/bin/sh", &["-c".into(), script], None, None)
            .await
            .unwrap();
        let state = backend.state.lock().unwrap();
        let args = state.last_request.clone().unwrap().args.unwrap();
        let argv = args["argv"].as_array().unwrap();
        let sent = argv[1].as_str().unwrap();
        assert_eq!(
            sent,
            format!(
                "cd {gw} && mv {gw}/a.txt {gw}/sub/a.txt",
                gw = GUEST_WORKSPACE_PATH
            )
        );
        assert!(!sent.contains(root.to_str().unwrap()));
    }

    #[tokio::test]
    async fn guest_style_and_relative_working_dirs_are_accepted() {
        let dir = TempDir::new().unwrap();
        let backend = FakeBackend::default();
        let sb = provider(backend.clone(), dir.path());
        sb.execute_command("ls", &[], Some(Path::new("/workspace/sub")), None)
            .await
            .unwrap();
        {
            let state = backend.state.lock().unwrap();
            let args = state.last_request.clone().unwrap().args.unwrap();
            assert_eq!(args["cwd"], format!("{GUEST_WORKSPACE_PATH}/sub"));
        }
        sb.execute_command("ls", &[], Some(Path::new("sub")), None)
            .await
            .unwrap();
        let state = backend.state.lock().unwrap();
        let args = state.last_request.clone().unwrap().args.unwrap();
        assert_eq!(args["cwd"], format!("{GUEST_WORKSPACE_PATH}/sub"));
    }

    #[tokio::test]
    async fn guest_style_working_dir_with_traversal_is_rejected() {
        let dir = TempDir::new().unwrap();
        let backend = FakeBackend::default();
        let sb = provider(backend.clone(), dir.path());
        for bad in ["/workspace/../etc", "sub/../../etc"] {
            let err = sb
                .execute_command("ls", &[], Some(Path::new(bad)), None)
                .await
                .unwrap_err();
            assert!(matches!(err, SandboxViolation::PathEscape { .. }), "{bad}");
        }
    }

    #[tokio::test]
    async fn working_dir_outside_the_root_is_a_path_escape() {
        let dir = TempDir::new().unwrap();
        let backend = FakeBackend::default();
        let sb = provider(backend.clone(), dir.path());
        let err = sb
            .execute_command("ls", &[], Some(Path::new("/etc")), None)
            .await
            .unwrap_err();
        assert!(matches!(err, SandboxViolation::PathEscape { .. }));
        // The escape is caught before any VM is booted.
        assert_eq!(backend.state.lock().unwrap().create_calls, 0);
    }

    #[tokio::test]
    async fn rejected_dispatch_is_a_disallowed_command() {
        let dir = TempDir::new().unwrap();
        let backend = FakeBackend::default();
        backend.set_next_result(DispatchResult::process(
            "r".to_string(),
            Status::Rejected,
            1,
            String::new(),
            String::new(),
            None,
            false,
        ));
        let sb = provider(backend.clone(), dir.path());
        let err = sb
            .execute_command("rm", &["-rf".into()], None, None)
            .await
            .unwrap_err();
        match err {
            SandboxViolation::DisallowedCommand { command } => {
                assert!(command.contains("rm"), "{command}");
            }
            other => panic!("expected DisallowedCommand, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn guest_timeout_maps_to_timed_out_output() {
        let dir = TempDir::new().unwrap();
        let backend = FakeBackend::default();
        backend.set_next_result(DispatchResult::process(
            "r".to_string(),
            Status::Timeout,
            1,
            "partial".to_string(),
            String::new(),
            None,
            false,
        ));
        let sb = provider(backend.clone(), dir.path());
        let out = sb
            .execute_command("sleep", &["99".into()], None, Some(Duration::from_secs(1)))
            .await
            .unwrap();
        assert!(out.timed_out);
        assert_eq!(out.stdout, "partial");
        assert!(out.stderr.contains("timed out after 1s"));
    }

    #[tokio::test]
    async fn non_zero_exit_is_a_normal_output_not_an_error() {
        let dir = TempDir::new().unwrap();
        let backend = FakeBackend::default();
        backend.set_next_result(DispatchResult::process(
            "r".to_string(),
            Status::Failure,
            1,
            String::new(),
            "boom\n".to_string(),
            Some(1),
            false,
        ));
        let sb = provider(backend.clone(), dir.path());
        let out = sb.execute_command("false", &[], None, None).await.unwrap();
        assert_eq!(out.exit_code, 1);
        assert_eq!(out.stderr, "boom\n");
        assert!(!out.timed_out);
    }

    // ------------------------------------------------------------------
    // Path policy delegation + identity
    // ------------------------------------------------------------------

    #[tokio::test]
    async fn resolve_path_still_enforces_the_workspace_boundary() {
        let dir = TempDir::new().unwrap();
        let sb = provider(FakeBackend::default(), dir.path());
        let err = sb
            .resolve_path("../etc/passwd", Operation::Read)
            .await
            .unwrap_err();
        assert!(matches!(err, SandboxViolation::PathEscape { .. }));
    }

    #[tokio::test]
    async fn isolation_mode_reports_micro_vm() {
        let dir = TempDir::new().unwrap();
        let backend = FakeBackend::default();
        let sb = PetriSandboxProvider::with_backend(
            backend,
            WorkspaceConfig::scoped(dir.path().to_path_buf()),
            PetriVmOptions::new(dir.path().join("policy.toml"), "s")
                .with_image("/images/base.petri")
                .with_network(NetworkPolicy::Full),
        )
        .unwrap();
        match sb.isolation_mode() {
            IsolationMode::MicroVm { image, network } => {
                assert_eq!(image, "/images/base.petri");
                assert_eq!(network, NetworkPolicy::Full);
            }
            other => panic!("expected MicroVm, got {other:?}"),
        }
    }

    // ------------------------------------------------------------------
    // Doctor probe + orphan reaping
    // ------------------------------------------------------------------

    #[test]
    fn petri_available_checks_helper_and_image() {
        let dir = TempDir::new().unwrap();
        let helper = dir.path().join("petri-vz");
        assert!(!petri_available(&helper, None));
        std::fs::write(&helper, "stub").unwrap();
        assert!(petri_available(&helper, None));
        assert!(!petri_available(
            &helper,
            Some(&dir.path().join("missing.img"))
        ));
        let image = dir.path().join("base.img");
        std::fs::write(&image, "img").unwrap();
        assert!(petri_available(&helper, Some(&image)));
    }

    #[tokio::test]
    async fn reap_session_kills_only_matching_orphans() {
        let dir = TempDir::new().unwrap();
        let backend = FakeBackend::default();
        // One VM from "our" session, one from another.
        let ours = provider(backend.clone(), dir.path());
        ours.execute_command("echo", &[], None, None).await.unwrap();
        // Leak it (simulate a crashed host) by forgetting the handle.
        {
            let mut slot = ours.vm.lock().await;
            let _ = slot.take().map(std::mem::forget);
        }
        let other = PetriSandboxProvider::with_backend(
            backend.clone(),
            WorkspaceConfig::scoped(dir.path().to_path_buf()),
            PetriVmOptions::new(dir.path().join("policy.toml"), "other-session"),
        )
        .unwrap();
        other
            .execute_command("echo", &[], None, None)
            .await
            .unwrap();

        assert_eq!(reap_session(&backend, "test-session"), 1);
        let state = backend.state.lock().unwrap();
        assert_eq!(state.instances.len(), 1);
        assert_eq!(
            state.instances[0]
                .metadata
                .get(SESSION_METADATA_KEY)
                .map(String::as_str),
            Some("other-session")
        );
    }

    /// Model a host that died without tearing its VM down: forget the handle
    /// so `Drop` can't kill the sandbox behind the reaper's back.
    async fn leak_vm(sb: &PetriSandboxProvider<FakeBackend>) {
        let mut slot = sb.vm.lock().await;
        let _ = slot.take().map(std::mem::forget);
    }

    /// A pid that is certainly dead: spawn a child and reap it.
    fn dead_pid() -> u32 {
        let mut child = std::process::Command::new("/bin/echo")
            .stdout(std::process::Stdio::null())
            .spawn()
            .expect("spawn /bin/echo");
        let pid = child.id();
        child.wait().expect("wait for /bin/echo");
        pid
    }

    /// Overwrite the owner pid recorded on the one instance the fake holds.
    fn set_owner(backend: &FakeBackend, owner: Option<u32>) {
        let mut state = backend.state.lock().unwrap();
        let metadata = &mut state.instances[0].metadata;
        match owner {
            Some(pid) => {
                metadata.insert(HOST_PID_METADATA_KEY.to_string(), pid.to_string());
            }
            None => {
                metadata.remove(HOST_PID_METADATA_KEY);
            }
        }
    }

    /// A minimal direct-bootable image bundle. The disk must be a whole number
    /// of 64 KiB blocks or petri rejects the geometry.
    fn image_bundle(root: &Path) -> PathBuf {
        let bundle = root.join("bundle");
        std::fs::create_dir_all(&bundle).unwrap();
        std::fs::write(bundle.join("root.img"), vec![0u8; 128 * 1024]).unwrap();
        std::fs::write(bundle.join("vmlinuz"), b"kernel").unwrap();
        std::fs::write(bundle.join("initrd.img"), b"initrd").unwrap();
        std::fs::write(
            bundle.join("petri-image.json"),
            r#"{
                "architecture": "aarch64",
                "kernel": "vmlinuz",
                "disk": "root.img",
                "initrd": "initrd.img",
                "kernel_command_line": "root=/dev/vda1 rw"
            }"#,
        )
        .unwrap();
        bundle
    }

    #[tokio::test]
    async fn scratch_root_boots_the_vm_over_an_overlay_not_the_bare_image() {
        let dir = TempDir::new().unwrap();
        let bundle = image_bundle(dir.path());
        let backend = FakeBackend::default();
        let sb = PetriSandboxProvider::with_backend(
            backend.clone(),
            WorkspaceConfig::scoped(dir.path().to_path_buf()),
            PetriVmOptions::new(dir.path().join("policy.toml"), "test-session")
                .with_image(&bundle)
                .with_scratch_root(dir.path().join("scratch")),
        )
        .unwrap();
        sb.execute_command("echo", &[], None, None).await.unwrap();

        let config = backend.last_config();
        // Direct boot off the NBD export, and crucially no writable image
        // attachment — that is what stops runs from mutating the base.
        assert!(config.direct_boot.is_some(), "expected an NBD direct boot");
        assert!(config.image.is_none(), "base image attached writably");

        let vm_id = sb.vm_id().await.expect("vm id");
        assert!(dir
            .path()
            .join("scratch")
            .join(vm_id)
            .join("scratch.data")
            .exists());
    }

    #[tokio::test]
    async fn without_a_scratch_root_the_image_still_attaches_directly() {
        let dir = TempDir::new().unwrap();
        let bundle = image_bundle(dir.path());
        let backend = FakeBackend::default();
        let sb = PetriSandboxProvider::with_backend(
            backend.clone(),
            WorkspaceConfig::scoped(dir.path().to_path_buf()),
            PetriVmOptions::new(dir.path().join("policy.toml"), "test-session").with_image(&bundle),
        )
        .unwrap();
        sb.execute_command("echo", &[], None, None).await.unwrap();

        let config = backend.last_config();
        assert_eq!(config.image.as_deref(), Some(bundle.as_path()));
        assert!(config.direct_boot.is_none());
    }

    #[tokio::test]
    async fn vm_records_the_owning_host_pid() {
        let dir = TempDir::new().unwrap();
        let backend = FakeBackend::default();
        let sb = provider(backend.clone(), dir.path());
        sb.execute_command("echo", &[], None, None).await.unwrap();
        let state = backend.state.lock().unwrap();
        assert_eq!(
            state.instances[0].metadata.get(HOST_PID_METADATA_KEY),
            Some(&std::process::id().to_string())
        );
    }

    #[tokio::test]
    async fn reap_orphans_spares_a_vm_whose_owner_is_alive() {
        let dir = TempDir::new().unwrap();
        let backend = FakeBackend::default();
        let sb = provider(backend.clone(), dir.path());
        // Owned by this very process, so the reaper must leave it running —
        // this is what keeps concurrent hosts from reaping each other.
        sb.execute_command("echo", &[], None, None).await.unwrap();

        assert_eq!(reap_orphans(&backend), 0);
        assert_eq!(backend.state.lock().unwrap().instances.len(), 1);
    }

    #[tokio::test]
    async fn reap_orphans_collects_a_vm_whose_owner_is_gone() {
        let dir = TempDir::new().unwrap();
        let backend = FakeBackend::default();
        let sb = provider(backend.clone(), dir.path());
        sb.execute_command("echo", &[], None, None).await.unwrap();
        set_owner(&backend, Some(dead_pid()));
        leak_vm(&sb).await;

        assert_eq!(reap_orphans(&backend), 1);
        assert!(backend.state.lock().unwrap().instances.is_empty());
    }

    #[tokio::test]
    async fn reap_orphans_collects_a_vm_with_no_recorded_owner() {
        let dir = TempDir::new().unwrap();
        let backend = FakeBackend::default();
        let sb = provider(backend.clone(), dir.path());
        sb.execute_command("echo", &[], None, None).await.unwrap();
        // Predates HOST_PID_METADATA_KEY: nothing tracks it, so it is an orphan.
        set_owner(&backend, None);
        leak_vm(&sb).await;

        assert_eq!(reap_orphans(&backend), 1);
        assert!(backend.state.lock().unwrap().instances.is_empty());
    }

    #[tokio::test]
    async fn reap_orphans_ignores_instances_it_did_not_create() {
        let backend = FakeBackend::default();
        // An untagged instance belongs to another petri user entirely.
        backend
            .state
            .lock()
            .unwrap()
            .instances
            .push(InstanceHandle {
                id: InstanceId::new("someone-elses-vm").unwrap(),
                backend: "fake".to_string(),
                state: LifecycleState::Ready,
                metadata: BTreeMap::new(),
            });

        assert_eq!(reap_orphans(&backend), 0);
        assert_eq!(backend.state.lock().unwrap().instances.len(), 1);
    }

    #[tokio::test]
    async fn reap_orphan_scratch_spares_a_live_vms_overlay() {
        let dir = TempDir::new().unwrap();
        let bundle = image_bundle(dir.path());
        let scratch_root = dir.path().join("scratch");
        let backend = FakeBackend::default();
        let sb = PetriSandboxProvider::with_backend(
            backend.clone(),
            WorkspaceConfig::scoped(dir.path().to_path_buf()),
            PetriVmOptions::new(dir.path().join("policy.toml"), "test-session")
                .with_image(&bundle)
                .with_scratch_root(&scratch_root),
        )
        .unwrap();
        sb.execute_command("echo", &[], None, None).await.unwrap();
        let live_vm = sb.vm_id().await.expect("vm id");

        // Deleting this would pull the disk out from under a running VM.
        assert_eq!(reap_orphan_scratch(&backend, &scratch_root), 0);
        assert!(scratch_root.join(&live_vm).join("scratch.data").exists());
    }

    #[tokio::test]
    async fn reap_orphan_scratch_collects_a_dead_vms_overlay() {
        let dir = TempDir::new().unwrap();
        let scratch_root = dir.path().join("scratch");
        let backend = FakeBackend::default();
        // A host that died mid-run leaves its overlay behind, still holding
        // everything the guest had written.
        let orphan = scratch_root.join("spore-test-session-999");
        std::fs::create_dir_all(&orphan).unwrap();
        std::fs::write(orphan.join("scratch.data"), vec![0u8; 4096]).unwrap();

        assert_eq!(reap_orphan_scratch(&backend, &scratch_root), 1);
        assert!(!orphan.exists());
    }

    #[test]
    fn reap_orphan_scratch_tolerates_a_missing_scratch_root() {
        let dir = TempDir::new().unwrap();
        let backend = FakeBackend::default();
        assert_eq!(
            reap_orphan_scratch(&backend, &dir.path().join("never-booted")),
            0
        );
    }

    #[test]
    fn host_process_alive_distinguishes_live_pids_from_dead_ones() {
        assert!(host_process_alive(std::process::id()));
        assert!(!host_process_alive(dead_pid()));
        // kill(2) reads 0 as "my whole process group", which would report a
        // live owner for an instance that recorded no real pid.
        assert!(!host_process_alive(0));
    }

    // ------------------------------------------------------------------
    // Live VM (opt-in)
    // ------------------------------------------------------------------

    /// End-to-end against a real petri microVM. Requires the petri-vz helper
    /// and a base image:
    /// `SPORE_PETRI_STATE_DIR=… SPORE_PETRI_HELPER=… SPORE_PETRI_IMAGE=… \
    ///  cargo test --features petri -- --ignored live_vm`
    #[tokio::test]
    #[ignore = "boots a real petri microVM; needs SPORE_PETRI_{STATE_DIR,HELPER,IMAGE}"]
    async fn live_vm_runs_a_command_in_the_guest() {
        let (Ok(state_dir), Ok(helper), Ok(image)) = (
            std::env::var("SPORE_PETRI_STATE_DIR"),
            std::env::var("SPORE_PETRI_HELPER"),
            std::env::var("SPORE_PETRI_IMAGE"),
        ) else {
            panic!("set SPORE_PETRI_STATE_DIR / SPORE_PETRI_HELPER / SPORE_PETRI_IMAGE");
        };
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("marker.txt"), "from-the-host").unwrap();
        let policy = PetriPolicy::new(PetriCommandPolicy::Unrestricted, NetworkPolicy::None)
            .write_to(dir.path())
            .unwrap();
        let sb = PetriSandbox::macos(
            state_dir,
            helper,
            WorkspaceConfig::scoped(dir.path().to_path_buf()),
            PetriVmOptions::new(policy, "live-test").with_image(image),
        )
        .unwrap();
        let out = sb
            .execute_command(
                "/bin/sh",
                &["-c".into(), "cat /workspace/marker.txt".into()],
                None,
                Some(Duration::from_secs(60)),
            )
            .await
            .unwrap();
        assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
        assert!(out.stdout.contains("from-the-host"));
        sb.shutdown().await;
    }
}
