/**
 * Lifecycle hook system (spore-core issue #69).
 *
 * Public surface: the {@link Hook} / {@link HookChain} interfaces, the 17
 * {@link HookEvent}s with their classification predicates, the tagged
 * {@link HookDecision} union, {@link HookError}, the per-event
 * {@link HookContext}, and the concrete {@link StandardHookChain},
 * {@link FunctionHook}, {@link CommandHook}.
 */
export * from "./types.js";
export * from "./standard.js";
