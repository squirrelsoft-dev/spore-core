# Design note: sandbox isolation backends

> **Status:** proposal / design — not yet implemented. This is an engineering design note
> (internals), distinct from the user-facing [concept pages](../concepts/). It records the plan and
> the rationale for implementing real isolation backends behind `IsolationMode`.
>
> **Supersedes/extends:** the `Bubblewrap`/`Docker` stubs and issue #6.

## Problem

`IsolationMode` (`rust/crates/spore-core/src/harness.rs:4612`) advertises four isolation modes, but
only two are real:

- `WorkspaceScoped` — **path enforcement only**. Confines file *paths* to the workspace root, but
  `exec`/`bash` run **uncontained on the host** (full privileges + network). This is the default.
- `None` — no isolation (dangerous-gated, issue #34).
- `Bubblewrap { profile }` and `Docker { image, network }` — **stubs**. Both arms in
  `WorkspaceScopedSandbox::execute_command` (`sandbox.rs:334-345`) return `DisallowedCommand`.

So today an agent that can run commands is not actually contained. This note designs the real
backends, adds a new strongest tier, and corrects the model that the stubs encode.

## Isolation tiers

Three tiers, weakest to strongest. Each has a backend per platform.

| Tier | `IsolationMode` | Linux backend | macOS backend | External dep | Contains process? | Hostname allowlist? |
|------|-----------------|---------------|---------------|--------------|-------------------|---------------------|
| Path-only | `WorkspaceScoped` | — (host) | — (host) | none | ❌ | ❌ |
| **OS process sandbox** | `OsSandbox` *(was `Bubblewrap`)* | `bwrap` | Seatbelt (`sandbox-exec`) | none | ✅ | ❌ (none/full only) |
| **Container** | `Docker` | Docker | Docker Desktop | docker | ✅ | ✅ |
| **MicroVM** | `MicroVM` *(new)* | petri (Firecracker, later) | petri (Virtualization.framework) | `petri` crate | ✅✅ | ✅ |

The big realization that drives everything below: **the strength of network control is a function
of whether the tier owns a network stack.** See [Network model](#network-model).

## Terminology: "bubblewrap" → OS process sandbox

In this project "bubblewrap" is used as shorthand for **a lightweight OS-native process sandbox**,
not the Linux `bwrap` tool specifically. The tier dispatches by OS:

- **Linux** → `bwrap` (bubblewrap): namespace-based isolation.
- **macOS** → **Seatbelt** via `sandbox-exec -f profile.sb`: kernel MAC-layer policy.

Because the wire `kind` tag is serialized across all four languages and would say `bubblewrap` even
while running Seatbelt on macOS, the variant should be **renamed to an OS-neutral name** —
recommended `OsSandbox` (wire tag `os_sandbox`) — with `bwrap`/`seatbelt` selected at runtime via
`cfg!(target_os)`. "bubblewrap" remains fine as colloquial/doc shorthand. *(Open decision — see
below.)*

The current `BwrapProfile {}` placeholder (`harness.rs:4651`) becomes a **neutral policy
descriptor** that each backend lowers to its native form — workspace root (rw), extra read-only
binds, network policy, allowed exec. This mirrors how petri lowers a neutral TOML policy to
in-guest nftables.

## Architecture: stateful providers vs. command-wrapping

The stubs assume every backend is a **stateless command transform** wired into
`WorkspaceScopedSandbox::execute_command`'s `match`. That is right for the OS-sandbox tier and wrong
for the heavier tiers:

- **`OsSandbox`** *is* a stateless wrap — prefix the argv with `bwrap …` (Linux) or run under
  `sandbox-exec -f <generated.sb>` (macOS). It can stay inside `WorkspaceScopedSandbox`, gated by
  `isolation_mode`, reusing the existing path canonicalization (the workspace it binds *is* the
  enforced root).
- **`Docker` and `MicroVM` are stateful.** A petri `Sandbox` is a **live VM** with a real lifecycle
  (`Sandbox::create → commands().run → kill`); a useful Docker backend wants a **persistent
  container** so `npm install` then `npm test` share state. Both own a handle, must serialize
  dispatches, manage teardown-on-drop, and translate host `working_dir` → guest/container
  `/workspace`. That does not fit a stateless `match` arm.

**Decision:** heavy tiers become their own `SandboxProvider` implementations
(`PetriSandboxProvider`, `DockerSandboxProvider`); `IsolationMode` stays the routing/observability
tag (`isolation_mode()` still reports `MicroVM { … }`). The OS-sandbox tier stays inside
`WorkspaceScopedSandbox`.

This is coherent because spore-core's **file tools are already workspace-confined**, and the
container/VM mounts that same host workspace (Docker bind mount; petri virtio-fs at `/workspace`,
read-write, live). File ops run host-side against the shared mount; `exec`/`bash` route into the
guest — same bytes, no sync layer.

## Network model

Network control is the part that most distinguishes the tiers, so it is specified here in full.
"On/off" is solid everywhere; **hostname allowlisting is a stack-owning-tier capability only.**

### On/off — two different mechanisms

The OS-sandbox backends achieve "off" by opposite means:

| Backend | Mechanism | "Off" | "On" |
|---------|-----------|-------|------|
| **bwrap** | **Isolation by construction** — new network namespace (`CLONE_NEWNET`) | `--unshare-net`: the process gets an empty netns — no `eth0`, no routes, nothing to connect *through* | omit `--unshare-net` → inherits host netns |
| **Seatbelt** | **Mediation by policy** — kernel MAC hooks allow/deny syscalls | `(deny network*)`: interfaces remain visible but `connect()`/`bind()` are refused (EPERM) | `(allow network*)` |

**Verification (observed on macOS arm64 + Docker Desktop 28.5.1, Debian trixie guest):**
- bwrap: host netns shows `eth0`; inside `--unshare-net` no external interface exists.
- Seatbelt: under `(deny network*)`, `en0` is still visible with its real IP, but an outbound
  `connect()` is refused in 0 ms (no SYN leaves); the unsandboxed control reaches HTTP 200.

One asymmetry to document in the Seatbelt profile generator: bwrap-off still brings up an isolated
`lo`, so **localhost keeps working**; Seatbelt `(deny network*)` blocks loopback too unless the
profile adds `(allow network* (local …))`. Special-case localhost if tools depend on it.

### Hostname allowlist — DNS proxy + default-deny firewall

The VM/container model is **not** "a DNS proxy." It is a DNS proxy **and** a firewall, because DNS
alone is bypassable (a workload can hardcode an IP or use DoH/DoT over 443):

1. The workload's resolver is pointed at a DNS proxy the sandbox controls (own `/etc/resolv.conf`,
   or a port-53 redirect).
2. The proxy resolves only allowlisted names; for an allowed name it returns the real IP **and pins
   that IP into an `nftables` allow-set**.
3. `nftables` runs **default-deny egress** — only IPs the proxy just blessed get out. Hardcoded IPs
   and alternate resolvers are dropped.

The precondition is **a private network namespace/stack in the data path** between workload and
outside world. Even then it is good-faith, not airtight (petri's own ADR 0002 says so) — the
default-deny firewall, not the DNS filter, is what makes it hold.

### Why the lightweight tier cannot do hostname allowlists

- **bwrap** *has* the seam (`--unshare-net` gives a private netns) but supplies none of the
  furniture. A real allowlist means bolting on a `veth`/`slirp4netns`/`pasta` stack + the DNS proxy
  + nftables — i.e. **reconstructing the container network path**, losing the "lightweight,
  zero-dependency" property. Not worth building; point users at Docker/petri.
- **Seatbelt** *cannot* — it creates no netns; the process shares the host's single stack and
  Seatbelt only allow/denies syscalls. There is no per-process data-path seam to insert a resolver
  or firewall into; macOS `pf` is host-global, not per-process, and not driven by Seatbelt. The
  ceiling is the coarse `(remote ip "1.2.3.4:443")` IP:port filter — no hostnames, since names
  resolve to changing IPs at connect time and DNS egress would itself have to be allowed.

### Per-tier capability and the `NetworkPolicy` decision

| Tier | Owns a netns/stack? | `None` | `Full` | `Allowlist{hosts}` |
|------|---------------------|--------|--------|--------------------|
| `OsSandbox` (bwrap) | yes, unfurnished | ✅ `--unshare-net` | ✅ omit unshare | ⚠️ only by rebuilding the container stack — **rejected** |
| `OsSandbox` (Seatbelt) | no (shared host stack) | ✅ `(deny network*)` | ✅ `(allow network*)` | ❌ impossible |
| `Docker` | yes (per-container netns) | ✅ | ✅ | ✅ DNS proxy + nftables in the container |
| `MicroVM` (petri) | yes (VM NIC) | ✅ | ✅ | ✅ in-guest DNS proxy + nftables |

**Decision:** on the `OsSandbox` tier, `NetworkPolicy` offers **`None`** and **`Full`** and
**errors on `Allowlist`** — "hostname allowlist requires `microvm` or `docker`." Silent downgrade
(either direction) is a footgun; an explicit error tells the caller to pick a stronger tier.
`Allowlist` is a container/VM-tier capability, full stop.

## Per-backend implementation notes

**`bwrap` (Linux, OsSandbox).** Wrap argv: `bwrap --bind <workspace> /workspace --ro-bind <ro …> …
--unshare-all [--share-net unless network none] --uid <n> --dev /dev -- <cmd> <args>`. Lower the
neutral profile to flags. Requires unprivileged userns on the host (or setuid bwrap).

**Seatbelt (macOS, OsSandbox).** Generate a `.sb` SBPL profile from the neutral profile
(`(allow default)`, `(deny file-write*)` + `(allow file-write* (subpath <workspace>))`, network
clause, localhost special-case) and run `sandbox-exec -f <profile> -- <cmd> <args>`. Note:
`sandbox-exec`/`sandbox_init` are **deprecated by Apple but fully functional** and the de-facto
userspace sandbox on macOS (Chromium, Claude Code's own macOS sandbox use Seatbelt). The blessed App
Sandbox needs entitlements + a signed bundle and cannot wrap arbitrary commands, so `sandbox-exec`
is the pragmatic choice — document the dependency on a deprecated-but-stable interface.

**Docker (`DockerSandboxProvider`).** Persistent container: `docker run -d --rm --network <…> -v
<workspace>:/workspace -w /workspace <image>` on create; `docker exec` per command; `docker rm -f`
on drop. Network: `--network none` / default bridge / custom network + DNS-filter sidecar for
allowlist. Map host `working_dir` → `/workspace/…`.

**petri (`PetriSandboxProvider`, MicroVM).** Build on petri's first-party `Sandbox` client (Rust
crate in-process; CLI-wrapper clients for TS/Py/Go) — **not** raw CLI/protocol. Lifecycle:
`Sandbox::create(backend, {workspace, policy})` → `commands().run(cmd, {cwd, args, env, timeout_ms,
max_output_bytes})` → `kill()`. One provider instance = one VM for its lifetime. Required handling
(per petri's `docs/spore-core-integration.md`): supply both workspace + policy; tag with session
`metadata` to reap orphans via `list({metadata}) + kill`; **map non-zero exit to a normal
`CommandOutput`, not an error** (only transport/usage/protocol faults are errors); serialize
dispatches per VM; apply a create-timeout + abandon-recreate for the known boot-hang flake; pass env
explicitly (guest launches workloads with a clean environment); guarantee teardown on all paths.
Guest is Linux (Debian trixie arm64); `cwd` must canonicalize inside `/workspace`.

### The `IsolationMode::MicroVM` variant

Pure data, so **parity-safe to add to all four languages immediately**; the *provider* (which pulls
the `petri` crate) hides behind an optional feature. Proposed shape:

```
MicroVM { image: Option<PathBuf>, network: NetworkPolicy, /* limits, command allowlist … */ }
```

This is **not** dangerous-gated — MicroVM is the *strongest* tier, the opposite of `None`/`Yolo`.

## Feature gating & dependencies

- `OsSandbox`, `Docker` — no external Rust deps (shell out to `bwrap`/`sandbox-exec`/`docker`). Wire
  variants always present.
- `MicroVM` — the enum variant is always present (parity); `PetriSandboxProvider` lives behind an
  optional cargo feature (proposed `petri`, default-off) with a **git dependency** on
  `squirrelsoft-dev/petri` (`petri` is not published to crates.io). This is *not* the `dangerous`
  gate.

## Testing matrix

Every tier is verifiable on a macOS arm64 dev box (this was checked, not assumed):

| Backend | How verified | Notes |
|---------|--------------|-------|
| Seatbelt | **natively** via `sandbox-exec` | filesystem confinement + network deny proven; no Docker/VM needed |
| bwrap | **relaxed Linux container** (`--cap-add SYS_ADMIN --security-opt seccomp=unconfined --security-opt apparmor=unconfined`, or `--privileged`) | default containers block nested unprivileged userns; relaxing them lets bwrap nest. Verifies **wiring**, not the host-hardening posture (which wants a bare Linux host / Linux CI) |
| Docker | Docker Desktop directly | real verification |
| petri/MicroVM | petri macOS backend, or `PETRI_MACOS_BACKEND_FALLBACK=loopback` | loopback runs `petri-guest` as a local TCP process for wiring tests without VM prereqs |

Add a containerized bwrap test lane (Linux CI) for the OS-sandbox tier.

## Wire-format / parity impact

Adding `MicroVM` and renaming `Bubblewrap → OsSandbox` changes the serialized `IsolationMode`
(`#[serde(tag = "kind")]`) across Rust/TS/Py/Go. Land Rust first (reference) and verify, then file
the TS/Py/Go parity issue — per project convention. Check fixtures under
`fixtures/sandbox_violations/` and any `IsolationMode` wire fixtures for the renamed/added variant.

## Sequencing

Recommended, Rust-first then parity:

1. **`OsSandbox`** (bwrap + Seatbelt). No external deps, native + locally verifiable on both target
   platforms, and it builds the neutral-profile-lowering abstraction the other tiers reuse.
2. **`Docker`**. Stateful provider; real local verification.
3. **`MicroVM`/petri**. Strongest isolation, most turnkey contract, but carries the git dependency.

*(petri-first is also defensible — it has the readiest integration contract. Open decision.)*

## Open decisions

1. **Variant name** — `OsSandbox` (recommended) vs. keep the literal `bubblewrap` wire tag.
2. **Sequencing** — `OsSandbox`-first (recommended) vs. petri-first.
3. **`Allowlist` on `OsSandbox`** — error (recommended) vs. silent downgrade to `None`.
4. **petri git dependency** behind an optional feature — confirm acceptable.

## References

- spore-core: `IsolationMode` `harness.rs:4612`; `SandboxProvider` `harness.rs:4448`; stubs
  `sandbox.rs:334-345`; `NetworkPolicy` `harness.rs:4638`; `dangerous` gating `sandbox.rs:143` +
  `Cargo.toml`; fixtures `fixtures/sandbox_violations/`.
- Canonical spec: [`harness-engineering-concepts.md`](../harness-engineering-concepts.md);
  user-facing [modes-and-approval](../concepts/modes-and-approval.md), [tools](../concepts/tools.md).
- petri: repo `squirrelsoft-dev/petri`; `docs/spore-core-integration.md`, `docs/policy-config.md`,
  `docs/vsock-dispatch-protocol.md`, `docs/workspace-contract.md`, ADR 0002 (network filtering).
- Tracking: issue #6 (process isolation); issue #34 (`dangerous` gating).
