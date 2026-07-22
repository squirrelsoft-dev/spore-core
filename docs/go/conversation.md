# Conversation, events, and control flow

`StandardHarness.Run` executes a `Task` through its configured `LoopStrategy` and returns a `RunResult`.

Common result kinds:

- `RunSuccess`: terminal model output is available in `Output`.
- `RunFailure`: terminal halt is available in `Reason`.
- `RunPause`: the run paused for human/help/consultation flow.
- `RunEscalate`: a harness signal was emitted, such as `SignalPlanRecorded`.

## Streaming events

Pass a stream sink on run options to observe model/tool progress. Harness stream events include model turns, tool calls/results, tool progress previews, output-schema retries/violations, user messages, turn-end usage, and run terminal events.

All event variants may carry `NodeAttr` metadata when emitted from nested strategies:

- `Kind`: strategy/node kind.
- `AgentID`: leaf agent ID when available.
- `Depth`: nesting depth.
- `Path`: strategy path from wrapper nodes.

## Tool progress

Command-running tools can report live previews through `ToolContext.CommandProgress(call)`. Sandboxes that implement the streaming seam emit tail previews while preserving final stdout/stderr in the normal tool result.

## Defaults that preserve legacy behavior

Several parity features are opt-in so default-constructed Go configs remain stable:

- `WithToolCallNudge(true)`: nudge execute-step models to use tools before claiming completion.
- `WithPlanEmptyMaxRetries(n)`: re-ask an empty planner up to `n` times before `HaltEmptyPlan`; retries force the plan JSON schema, suppress tools, and carry discarded-attempt usage forward.
- `WithToolSearch(config)`: advertise core tools plus namespace/search headers, then statefully load searched/opened tools with description-budget LRU eviction.
- `WithChunkProvider(provider)`: presence-only prompt chunks such as the SelfVerifying role-evaluator chunk.
