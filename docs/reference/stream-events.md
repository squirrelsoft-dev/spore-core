# Reference: StreamEvent

> Every stream-event variant and how they're ordered. Language-agnostic. To consume them in Rust,
> see [rust/building-a-tool](../rust/building-a-tool.md); concept background in
> [architecture](../concepts/architecture.md).

A run optionally takes a **stream sink** — a callback the harness invokes with a `StreamEvent` as
the run progresses. Streaming is a *delivery mechanism for the UI*, not an operational concern:
the harness loop itself works on complete turn results. Only the model interface and the agent
ever touch stream events; everything else works on complete values. If you don't supply a sink,
nothing changes about how the run executes.

Events come in two granularities — **coarse** (turn- and call-level milestones) and **delta**
(token-by-token fragments). A simple UI can listen to the coarse events alone; a live-typing UI
also consumes the deltas.

## Coarse events

| Event | Fields | Meaning |
|-------|--------|---------|
| `TurnStart` | `turn` | A new model turn began. |
| `TurnEnd` | `turn` | The turn finished. |
| `ToolCall` | `call_id`, `name`, `args` | The model requested a tool call (fully accumulated arguments). |
| `ToolResult` | `call_id`, `is_error`, `content` | A tool call finished; correlate to its `ToolCall` by `call_id`. |
| `FinalResponse` | `content` | The model produced its final answer for the run. |
| `BudgetWarning` | `limit_type` | A budget threshold (e.g. tokens, time) was crossed. |
| `UserMessage` | `content` | A tool surfaced an out-of-band message to the user (the send-message tool), rather than a normal tool result. |

## Delta events (token-level)

These stream the pieces of a turn as they arrive. Each correlates to a content block by `index`,
and tool-argument fragments correlate to a call by `call_id`.

| Event | Fields | Meaning |
|-------|--------|---------|
| `BlockStart` | `index`, `block` | A content block opened (the first delta for that index). |
| `TextDelta` | `content` | A fragment of streamed answer text. |
| `ReasoningDelta` | `content` | A fragment of streamed reasoning/thinking text. |
| `ToolCallStart` | `index`, `call_id`, `name` | A tool-use block opened. `name` may be empty if the model stream doesn't surface it before the arguments — it's recovered on the coarse `ToolCall`. |
| `ToolArgsDelta` | `call_id`, `partial_json` | A fragment of the tool call's argument JSON. |
| `BlockStop` | `index` | A content block closed. |

## Ordering

Within a turn, a tool call streams in this order:

```
ToolCallStart → ToolArgsDelta* → BlockStop → ToolCall (coarse, fully accumulated)
```

That is: the block opens, its argument JSON arrives in fragments, the block closes, and finally
the coarse `ToolCall` carries the complete, parsed arguments. Correlate the fragments to the final
call by `call_id`. Text and reasoning blocks follow the same `BlockStart → *Delta* → BlockStop`
shape.

## Consuming them

- **Minimal UI** — listen for `TurnStart` to show progress, `ToolCall` / `ToolResult` to show
  activity, and `FinalResponse` for the answer.
- **Live-typing UI** — append `TextDelta` (and optionally `ReasoningDelta`) as they arrive;
  buffer tool arguments from `ToolArgsDelta` keyed by `call_id` if you want to show a tool call
  forming.
- **Reasoning** — thinking fragments arrive as `ReasoningDelta`; the corresponding reasoning
  content is preserved in message history and passed back on subsequent requests, so don't treat
  it as throwaway.

The stream is for display. Operational decisions — termination, error routing, tool dispatch —
happen off the complete turn result the harness assembles, regardless of whether you're listening.
