# Go quickstart

Install the Go module from this repository path:

```go
import sporecore "github.com/squirrelsoft-dev/spore-core/go/spore-core"
```

Create a task with a loop strategy, configure a harness, and run it:

```go
agent := sporecore.NewMockAgent("example") // replace with a real Agent
agent.Push(sporecore.NewFinalResponse("done", sporecore.TokenUsage{}))

cfg := sporecore.HarnessConfig{
    Agent:             agent,
    ToolRegistry:      sporecore.NewScriptedToolRegistry(),
    Sandbox:           sporecore.AllowAllSandbox{},
    ContextManager:    sporecore.NoopContextManager{},
    TerminationPolicy: sporecore.AlwaysContinuePolicy{},
    EscalationMode:    sporecore.AutonomousEscalation(),
}
h := sporecore.NewStandardHarness(cfg)

task := sporecore.NewTask(
    "Do the work",
    sporecore.SessionID("session-1"),
    sporecore.ReActStrategy(10),
)
result := h.Run(context.Background(), sporecore.NewHarnessRunOptions(task))
```

For catalogue tools, use `StandardToolRegistry` and the standard tool constructors from `go/spore-core/tools`. The harness injects per-run `ToolContext` storage automatically for catalogue registries.
