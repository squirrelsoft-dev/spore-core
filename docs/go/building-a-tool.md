# Building a Go tool

Implement `sporecore.Tool`:

```go
type MyTool struct{}

func (MyTool) Name() string { return "my_tool" }
func (MyTool) IsSubagentTool() bool { return false }
func (MyTool) MayProduceLargeOutput() bool { return false }
func (MyTool) Execute(ctx context.Context, call sporecore.ToolCall, sandbox sporecore.SandboxProvider, toolCtx *sporecore.ToolContext) sporecore.ToolOutput {
    // Parse call.Input, validate, do work through sandbox/toolCtx only.
    return sporecore.ToolOutput{Kind: sporecore.ToolOutputSuccess, Content: "ok"}
}
```

Register it with a schema:

```go
reg := sporecore.NewStandardToolRegistry()
err := reg.Register(MyTool{}, sporecore.RegistryToolSchema{
    Name: "my_tool",
    Description: "Does one thing",
    Parameters: json.RawMessage(`{"type":"object"}`),
})
```

Use `ToolContext` for run-local or durable state:

- `Get` / `Put`: ephemeral session-keyed state.
- `GetDurable` / `PutDurable`: project-keyed durable state (`ProjectNamespace` when configured, otherwise session fallback).
- `CommandProgress(call)`: optional per-call live preview sink for command-running tools.

Run commands through `sporecore.ExecuteCommandStreaming(ctx, sandbox, ...)` to get live progress when the sandbox supports it and safe fallback otherwise.
