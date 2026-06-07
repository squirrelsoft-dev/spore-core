/**
 * ExecutionRegistry — runtime resolution of serializable strategy handles
 * (Composable Execution A.3, issue #120; part of the #117–#131 refactor).
 *
 * # Types
 * - {@link ExecutionRegistry} — five maps of collaborators keyed by string:
 *   `agents`, `toolsets`, `schemas`, `verifiers`, `custom` (custom strategies).
 *   Trait objects never serialize, so the registry is NOT (de)serialized (no
 *   `toJSON`/`fromJSON`).
 * - {@link ExecutionRegistryBuilder} — fluent assembler mirroring
 *   {@link HarnessBuilder}.
 * - {@link StrategyResolution} — the result of resolving a {@link StrategyRef}:
 *   either a built-in {@link LoopStrategy} or a custom {@link RunStrategy}.
 * - {@link EscalationMode} — the HITL-vs-AFK config knob (PRD goal #7), defined
 *   in `./types.ts`.
 *
 * # Methods
 * - `resolveAgent` / `resolveToolset` / `resolveSchema` / `resolveVerifier` —
 *   read-only, synchronous pure lookups. Each `*Ref` type maps to exactly ONE
 *   map ({@link SchemaRef} → `schemas`). Return `undefined` on a miss.
 * - `resolveStrategy` — a `built_in` ref resolves to the built-in tree; a
 *   `custom` key looks up `custom` and THROWS the recoverable
 *   {@link StrategyNotFound} when the key is absent.
 * - `registerStrategy` — register a custom {@link RunStrategy} at startup.
 * - `validate` — walks a {@link Task}'s strategy tree, throwing the FIRST
 *   unresolved handle as {@link UnresolvedHandle} (or {@link StrategyNotFound}
 *   for a missing custom key). Called at the entry of `StandardHarness.run` so
 *   an unresolved handle is a STARTUP error, before the first turn.
 *
 * # Resolutions applied (do not re-litigate — pinned in #120)
 * - **Scope = ADDITIVE (Option B).** This slice ADDS the registry +
 *   `escalationMode` to {@link HarnessConfig}; it does NOT remove the four
 *   single-collaborator fields nor touch executor consumption sites. The
 *   registry coexists with the deprecated fields this slice and is not yet read
 *   by the run bodies (#123/#124).
 * - Exactly FIVE maps (no sixth).
 * - {@link EscalationMode} has NO default value baked into the type; the builder
 *   picks an explicit default (`surface_to_human`).
 */

import type { Agent } from "../agent/interface.js";
import type { Verifier } from "../verifier/types.js";

import { InvalidConfiguration, StrategyNotFound, UnresolvedHandle } from "./types.js";
import type {
  AgentRef,
  LoopStrategy,
  RunStrategy,
  SchemaRef,
  StrategyRef,
  Task,
  ToolRegistry,
  ToolsetRef,
} from "./types.js";

/**
 * The result of resolving a {@link StrategyRef} against an
 * {@link ExecutionRegistry}: either the built-in {@link LoopStrategy} tree or
 * the custom {@link RunStrategy} looked up in the registry's `custom` map.
 */
export type StrategyResolution =
  | { kind: "built_in"; strategy: LoopStrategy }
  | { kind: "custom"; strategy: RunStrategy };

/**
 * Runtime resolver mapping serializable string handles (and `StrategyRef`
 * `custom` keys) to concrete collaborators. See the module docs for the full
 * type/method/rule documentation.
 *
 * Trait objects never serialize, so this type is NOT (de)serialized. Build one
 * with {@link ExecutionRegistry.builder} or {@link ExecutionRegistry.empty}.
 */
export class ExecutionRegistry {
  private readonly agents: Map<string, Agent>;
  private readonly toolsets: Map<string, ToolRegistry>;
  private readonly schemas: Map<string, unknown>;
  private readonly verifiers: Map<string, Verifier>;
  private readonly custom: Map<string, RunStrategy>;

  /** Internal — use {@link ExecutionRegistry.empty} or
   *  {@link ExecutionRegistry.builder}. */
  constructor(maps?: {
    agents?: Map<string, Agent>;
    toolsets?: Map<string, ToolRegistry>;
    schemas?: Map<string, unknown>;
    verifiers?: Map<string, Verifier>;
    custom?: Map<string, RunStrategy>;
  }) {
    this.agents = maps?.agents ?? new Map();
    this.toolsets = maps?.toolsets ?? new Map();
    this.schemas = maps?.schemas ?? new Map();
    this.verifiers = maps?.verifiers ?? new Map();
    this.custom = maps?.custom ?? new Map();
  }

  /** An empty registry (no entries in any of the five maps). */
  static empty(): ExecutionRegistry {
    return new ExecutionRegistry();
  }

  /** Start a fluent {@link ExecutionRegistryBuilder}. */
  static builder(): ExecutionRegistryBuilder {
    return new ExecutionRegistryBuilder();
  }

  /**
   * True when no entries exist in any of the five maps. Lets the harness skip
   * startup validation for callers that never wire a registry (Option B
   * additive scope — they still use the deprecated single-collaborator fields).
   */
  isEmpty(): boolean {
    return (
      this.agents.size === 0 &&
      this.toolsets.size === 0 &&
      this.schemas.size === 0 &&
      this.verifiers.size === 0 &&
      this.custom.size === 0
    );
  }

  /**
   * Consume this registry into a builder preserving all existing entries, so a
   * caller (e.g. {@link HarnessBuilder}'s per-key convenience setters) can add
   * more before re-building. Entries are copied into fresh maps so mutating the
   * builder never aliases this registry.
   */
  toBuilder(): ExecutionRegistryBuilder {
    const b = new ExecutionRegistryBuilder();
    for (const [k, v] of this.agents) b.agent(k, v);
    for (const [k, v] of this.toolsets) b.toolset(k, v);
    for (const [k, v] of this.schemas) b.schema(k, v);
    for (const [k, v] of this.verifiers) b.verifier(k, v);
    for (const [k, v] of this.custom) b.registerStrategy(k, v);
    return b;
  }

  /** Resolve an {@link AgentRef} to its registered agent, or `undefined`. */
  resolveAgent(ref: AgentRef): Agent | undefined {
    return this.agents.get(ref);
  }

  /** Resolve a {@link ToolsetRef} to its registered toolset, or `undefined`. */
  resolveToolset(ref: ToolsetRef): ToolRegistry | undefined {
    return this.toolsets.get(ref);
  }

  /** Resolve a {@link SchemaRef} to its registered JSON schema, or `undefined`.
   *  ({@link SchemaRef} maps to the `schemas` map.) */
  resolveSchema(ref: SchemaRef): unknown | undefined {
    return this.schemas.get(ref);
  }

  /** Resolve a verifier key to its registered verifier, or `undefined`. */
  resolveVerifier(key: string): Verifier | undefined {
    return this.verifiers.get(key);
  }

  /**
   * Resolve a {@link StrategyRef}: a `built_in` ref returns the built-in tree;
   * a `custom` key looks up the `custom` map and THROWS the recoverable
   * {@link StrategyNotFound} when the key is absent.
   */
  resolveStrategy(ref: StrategyRef): StrategyResolution {
    switch (ref.kind) {
      case "built_in":
        return { kind: "built_in", strategy: ref.value };
      case "custom": {
        const strategy = this.custom.get(ref.value);
        if (strategy === undefined) {
          throw new StrategyNotFound(ref.value);
        }
        return { kind: "custom", strategy };
      }
    }
  }

  /** Register (or replace, last-wins) a custom strategy under `key`. */
  registerStrategy(key: string, strategy: RunStrategy): void {
    this.custom.set(key, strategy);
  }

  /**
   * Validate that every handle referenced by `task.loop_strategy` resolves
   * against this registry. Walks the strategy tree and THROWS the FIRST
   * unresolved handle as {@link UnresolvedHandle} (or {@link StrategyNotFound}
   * for a missing custom key). Returns normally when the whole tree resolves.
   * Called at the entry of `StandardHarness.run` so an unresolved handle is a
   * startup error.
   */
  validate(task: Task): void {
    this.walkStrategy(task.loop_strategy);
  }

  /** Recursive tree-walk over a {@link LoopStrategy}, checking every child
   *  handle. Throws on the first unresolved handle (depth-first). */
  private walkStrategy(ls: LoopStrategy): void {
    switch (ls.kind) {
      case "react":
        this.checkAgent(ls.agent);
        this.checkToolset(ls.toolset);
        if (ls.output !== undefined) {
          this.checkSchema(ls.output);
        }
        return;
      case "plan_execute":
        // A.5 (#124, Q3): the `plan` slot is STRUCTURED — it must yield a task
        // graph. A bare `ReAct` there needs an output schema.
        ExecutionRegistry.checkStructuredSlot(ls.plan, "plan");
        this.walkStrategy(ls.plan);
        this.walkStrategy(ls.execute);
        return;
      case "self_verifying":
        // A.5: the `inner` (worker) slot is STRUCTURED — its result must be
        // evaluable. A bare `ReAct` worker needs an output schema.
        ExecutionRegistry.checkStructuredSlot(ls.inner, "worker");
        this.walkStrategy(ls.inner);
        // The evaluator is a SchemaRef (the evaluator schema handle).
        this.checkSchema(ls.evaluator);
        return;
      case "ralph":
        this.walkStrategy(ls.inner);
        this.checkAgent(ls.agent);
        return;
      case "hill_climbing":
        // A.5: the `inner` (propose) slot is STRUCTURED — it must yield a
        // candidate. A bare `ReAct` proposer needs an output schema.
        ExecutionRegistry.checkStructuredSlot(ls.inner, "propose");
        this.walkStrategy(ls.inner);
        // The evaluator is an AgentRef (the metric-evaluator agent).
        this.checkAgent(ls.evaluator);
        return;
    }
  }

  /**
   * A.5 output-contract enforcement (#124, Q3): a bare `ReAct` feeding a
   * STRUCTURED slot (`plan` ⇒ task graph, `propose` ⇒ candidate, `worker` ⇒
   * evaluable result) MUST declare `ReAct.output`. A combinator child carries
   * its own contract, so this check applies only to the leaf. Throws
   * {@link InvalidConfiguration} naming the offending slot.
   */
  private static checkStructuredSlot(slot: LoopStrategy, slotName: string): void {
    if (slot.kind === "react" && slot.output === undefined) {
      throw new InvalidConfiguration(
        `a bare ReAct in the structured \`${slotName}\` slot requires ` +
          "`output = Some(schema)` so the slot yields a typed result",
      );
    }
  }

  private checkAgent(ref: AgentRef): void {
    if (!this.agents.has(ref)) {
      throw new UnresolvedHandle("agent", ref);
    }
  }

  private checkToolset(ref: ToolsetRef): void {
    if (!this.toolsets.has(ref)) {
      throw new UnresolvedHandle("toolset", ref);
    }
  }

  private checkSchema(ref: SchemaRef): void {
    if (!this.schemas.has(ref)) {
      throw new UnresolvedHandle("schema", ref);
    }
  }
}

/** Fluent assembler for an {@link ExecutionRegistry}, mirroring
 *  {@link HarnessBuilder}. */
export class ExecutionRegistryBuilder {
  private readonly agents = new Map<string, Agent>();
  private readonly toolsets = new Map<string, ToolRegistry>();
  private readonly schemas = new Map<string, unknown>();
  private readonly verifiers = new Map<string, Verifier>();
  private readonly custom = new Map<string, RunStrategy>();

  /** Register an agent under `key`. */
  agent(key: string, agent: Agent): this {
    this.agents.set(key, agent);
    return this;
  }

  /** Register a toolset under `key`. */
  toolset(key: string, toolset: ToolRegistry): this {
    this.toolsets.set(key, toolset);
    return this;
  }

  /** Register a JSON schema under `key`. */
  schema(key: string, schema: unknown): this {
    this.schemas.set(key, schema);
    return this;
  }

  /** Register a verifier under `key`. */
  verifier(key: string, verifier: Verifier): this {
    this.verifiers.set(key, verifier);
    return this;
  }

  /** Register a custom strategy under `key`. */
  registerStrategy(key: string, strategy: RunStrategy): this {
    this.custom.set(key, strategy);
    return this;
  }

  /** Finish and return the assembled {@link ExecutionRegistry}. */
  build(): ExecutionRegistry {
    return new ExecutionRegistry({
      agents: this.agents,
      toolsets: this.toolsets,
      schemas: this.schemas,
      verifiers: this.verifiers,
      custom: this.custom,
    });
  }
}
