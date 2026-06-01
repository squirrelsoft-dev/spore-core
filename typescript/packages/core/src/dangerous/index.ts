/**
 * `@spore/core/dangerous` — explicit opt-in entry point for safety footguns
 * (spore-core issue #34).
 *
 * The default barrel (`@spore/core`) deliberately does NOT expose these. They
 * are reachable only by importing this subpath, which is the idiomatic
 * TypeScript analogue of the Rust `dangerous` Cargo feature: in a default build
 * neither `Mode::Yolo` nor `IsolationMode::None` can be constructed, so reaching
 * them requires a conscious, code-review-visible act.
 *
 * What is gated:
 *   - `Mode.Yolo`  — full autonomy, no approval gates. Exposed here as the
 *     {@link DangerousMode} value `"yolo"` plus constructors that produce its
 *     prompt chunk / approval policy / standard-library chunk.
 *   - `IsolationMode.None` — no path enforcement. Exposed here as the
 *     {@link DangerousIsolationMode} value {@link noneIsolationMode} plus the
 *     {@link dangerousWorkspaceSandbox} constructor that wires it into a
 *     `WorkspaceScopedSandbox`.
 *
 * What is NOT gated (per maintainer decision on issue #34):
 *   - The `ApprovalPolicy` value `"none"` — only its entry point `Mode.Yolo` is.
 *
 * Wire tags are unchanged: the dangerous mode still serializes as `"yolo"` and
 * the dangerous isolation mode still serializes as `{ "kind": "none" }`.
 *
 * Intended for benchmarking, evals, and local development only. Do not enable
 * in production deployments.
 */

import type { DangerousIsolationMode, WorkspaceConfig } from "../harness/types.js";
import {
  type ApprovalPolicy,
  type ChunkId,
  type ChunkValidationError,
  type ComposedPrompt,
  type DangerousMode,
  type PromptChunk,
  anyModeApprovalPolicy,
  anyModeDefaultToolPhase,
  anyModePromptChunk,
  standardChunks,
  StandardPromptChunkRegistry,
} from "../prompt-chunk-registry/index.js";
import { WorkspaceScopedSandbox } from "../sandbox/index.js";
import type { TaskPhase } from "../tool-registry/types.js";

export type { DangerousMode, DangerousIsolationMode };

// ============================================================================
// Mode.Yolo
// ============================================================================

/**
 * The dangerous mode value. Equivalent to the Rust `Mode::Yolo`. The brand is
 * minted here (the single trusted source); default callers cannot forge it.
 */
export const DANGEROUS_MODE_YOLO = "yolo" as DangerousMode;

/** Build the Yolo prompt chunk (slot `mode`, cache `static`). */
export function yoloPromptChunk(): PromptChunk {
  return anyModePromptChunk(DANGEROUS_MODE_YOLO);
}

/** Approval policy implied by Yolo mode — `"none"` (full autonomy). */
export function yoloApprovalPolicy(): ApprovalPolicy {
  return anyModeApprovalPolicy(DANGEROUS_MODE_YOLO);
}

/** Initial task phase implied by Yolo mode — `"execution"`. */
export function yoloDefaultToolPhase(): TaskPhase {
  return anyModeDefaultToolPhase(DANGEROUS_MODE_YOLO);
}

/**
 * Compose a prompt with the dangerous Yolo mode. The default
 * {@link StandardPromptChunkRegistry.compose} cannot name `"yolo"`; this wrapper
 * reaches the registry's internal `composeAny` path.
 */
export function composeWithYolo(
  registry: StandardPromptChunkRegistry,
  role: ChunkId,
  capabilities: ChunkId[],
  skills: ChunkId[],
): { ok: true; composed: ComposedPrompt } | { ok: false; errors: ChunkValidationError[] } {
  return registry.composeAny(role, DANGEROUS_MODE_YOLO, capabilities, skills);
}

/**
 * The standard chunk library plus the dangerous Yolo mode chunk. Use this in
 * place of {@link standardChunks} when the registry must serve a Yolo
 * composition. Mirrors the Rust default build appending `Mode::Yolo` under the
 * `dangerous` feature.
 */
export function dangerousStandardChunks(): PromptChunk[] {
  return [...standardChunks(), yoloPromptChunk()];
}

// ============================================================================
// IsolationMode.None
// ============================================================================

/**
 * The dangerous isolation mode value — no path enforcement. The brand is minted
 * here (the single trusted source); default callers cannot forge it.
 */
export function noneIsolationMode(): DangerousIsolationMode {
  return { kind: "none" } as DangerousIsolationMode;
}

/**
 * Construct a {@link WorkspaceScopedSandbox} with the dangerous
 * {@link noneIsolationMode}. Emits a construction-time warning. This is the only
 * supported way to build a sandbox with no isolation; the default constructor
 * cannot name `{ kind: "none" }`.
 */
export function dangerousWorkspaceSandbox(
  config: WorkspaceConfig,
  mode: DangerousIsolationMode = noneIsolationMode(),
): WorkspaceScopedSandbox {
  return WorkspaceScopedSandbox.unsafeWithMode(config, mode);
}
