// Package scenarios assembles the end-to-end CLI agent scenario suite (issue
// #57). It holds the reusable wiring shared by the e2e-agent command binary AND
// the hermetic integration tests, so a live run (ollama.ModelInterface +
// ModelAgent) and an offline run (MockAgent + ScriptedToolRegistry) drive the
// SAME code path. BuildScenario takes the agent + tool registry + context
// manager as parameters, so the only difference between live and mock mode is
// which components you inject.
//
// ## Architectural gaps closed here vs. already-present in Go
//
//   - RealToolRegistry bridge — NOT needed in Go. The harness ToolRegistry seam
//     IS sporecore.ToolRegistry (Dispatch(ctx, call, sandbox)), so a
//     StandardToolRegistry plugs directly into HarnessConfig.ToolRegistry, and
//     the loop's dispatchAndUnwrap already maps a DispatchError onto a
//     recoverable ToolOutputError (and never halts on the recoverable
//     FailingTool). BuildRealToolRegistry returns the StandardToolRegistry
//     directly.
//   - SchemaInjectingContextManager — needed. The StandardCompactionAdapter's
//     Assemble returns a Context with no Tools, so without this decorator the
//     live model never sees any tools and can never emit a tool call. The
//     decorator injects the registry's schemas (sorted by name) while forwarding
//     every other seam, including the optional CompactingContextManager and
//     TokenBudgetReader seams so live compaction still fires.
//   - FailingTool (flaky_op, recoverable) — lives in the tools package
//     (tools.FailingTool); registered here.
//   - Complete-on-final-response termination — NOT needed. Go ships
//     sporecore.AlwaysContinuePolicy, which the harness already interprets as
//     "accept the final response and succeed"; reuse it.
package scenarios

import (
	"context"
	"sort"

	sporecore "github.com/squirrelsoft-dev/spore-core/go/spore-core"
	"github.com/squirrelsoft-dev/spore-core/go/spore-core/contextmgr"
	"github.com/squirrelsoft-dev/spore-core/go/spore-core/tools"
)

// ============================================================================
// ScenarioID — parsed from the CLI arg s1..s4
// ============================================================================

// ScenarioID identifies one of the four end-to-end scenarios.
type ScenarioID string

const (
	// S1 — multi-step / multi-tool: read input.txt -> uppercase -> write
	// output.txt -> read back + confirm.
	S1 ScenarioID = "s1"
	// S2 — multi-turn: run twice with the same SessionID, carrying state.
	S2 ScenarioID = "s2"
	// S3 — live compaction: a seeded small window + long history fires the
	// compaction adapter mid-run.
	S3 ScenarioID = "s3"
	// S4 — tool failure + recovery: call flaky_op (recoverable), then write a
	// recovery file.
	S4 ScenarioID = "s4"
)

// ParseScenarioID parses "s1".."s4" (case-insensitive). The bool is false for
// an unrecognized id.
func ParseScenarioID(s string) (ScenarioID, bool) {
	switch toLowerTrim(s) {
	case "s1":
		return S1, true
	case "s2":
		return S2, true
	case "s3":
		return S3, true
	case "s4":
		return S4, true
	default:
		return "", false
	}
}

func toLowerTrim(s string) string {
	// Avoid importing strings twice; trim spaces and lowercase ASCII.
	start, end := 0, len(s)
	for start < end && (s[start] == ' ' || s[start] == '\t' || s[start] == '\n') {
		start++
	}
	for end > start && (s[end-1] == ' ' || s[end-1] == '\t' || s[end-1] == '\n') {
		end--
	}
	b := []byte(s[start:end])
	for i, c := range b {
		if c >= 'A' && c <= 'Z' {
			b[i] = c + 32
		}
	}
	return string(b)
}

// Prompt returns the default prompt that drives this scenario.
func (id ScenarioID) Prompt() string {
	switch id {
	case S1:
		return "Read the file input.txt, transform its contents to UPPERCASE, write the " +
			"result to output.txt, then read output.txt back to confirm it was written. " +
			"Use the read_file, write_file, and bash_command tools. When done, reply DONE."
	case S2:
		return "Create a file notes.md containing a TODO list with one item: 'set up the " +
			"project'. Use write_file. Reply DONE when written."
	case S3:
		return "Summarize the long conversation so far and continue working on the deploy of " +
			"the payment service. Reply DONE when finished."
	case S4:
		return "Call the flaky_op tool. If it fails, do not give up: write a file " +
			"recovered.txt explaining that flaky_op failed and how you adapted, using " +
			"write_file. Reply DONE when finished."
	default:
		return ""
	}
}

// ============================================================================
// BuildRealToolRegistry — the shared real-tool catalog
// ============================================================================

// BuildRealToolRegistry builds a StandardToolRegistry populated with the real
// read/write/list/bash tools plus the recoverable FailingTool. Shared by every
// scenario so the agent always sees the same catalog. Registration failures are
// programming errors (duplicate / invalid schema) and panic.
func BuildRealToolRegistry() *sporecore.StandardToolRegistry {
	reg := sporecore.NewStandardToolRegistry()
	must := func(err error) {
		if err != nil {
			panic(err)
		}
	}
	must(reg.Register(tools.NewReadFileTool(), tools.NewReadFileTool().Schema()))
	must(reg.Register(tools.NewWriteFileTool(), tools.NewWriteFileTool().Schema()))
	must(reg.Register(tools.NewListDirTool(), tools.NewListDirTool().Schema()))
	must(reg.Register(tools.NewBashCommandTool(), tools.NewBashCommandTool().Schema()))
	must(reg.Register(tools.NewFailingTool(), tools.NewFailingTool().Schema()))
	return reg
}

// ModelSchemas returns the model-facing tool schemas for a registry, sorted by
// name (for cache stability and deterministic injection).
func ModelSchemas(reg *sporecore.StandardToolRegistry) []sporecore.ToolSchema {
	registry := reg.ActiveSchemas(nil) // already sorted by name
	out := make([]sporecore.ToolSchema, 0, len(registry))
	for _, s := range registry {
		out = append(out, s.ToModelSchema())
	}
	return out
}

// ============================================================================
// SchemaInjectingContextManager — fills Assemble().Tools from the registry
// ============================================================================

// SchemaInjectingContextManager decorates a sporecore.ContextManager, delegating
// every seam method to the inner manager but injecting the registry's tool
// schemas into Assemble().Tools. The compaction adapter's Assemble returns an
// empty tool list, so without this decorator the model never sees any tools and
// can never emit a tool call in live mode.
//
// The decorator forwards the OPTIONAL CompactingContextManager and
// TokenBudgetReader seams (when the inner manager implements them) so live
// compaction and the #57 token-budget span stamping still work behind it.
type SchemaInjectingContextManager struct {
	inner sporecore.ContextManager
	tools []sporecore.ToolSchema
}

// NewSchemaInjectingContextManager wraps inner, injecting toolSchemas (sorted by
// name) into every assembled context.
func NewSchemaInjectingContextManager(inner sporecore.ContextManager, toolSchemas []sporecore.ToolSchema) *SchemaInjectingContextManager {
	cp := append([]sporecore.ToolSchema(nil), toolSchemas...)
	sort.Slice(cp, func(i, j int) bool { return cp[i].Name < cp[j].Name })
	return &SchemaInjectingContextManager{inner: inner, tools: cp}
}

// Assemble delegates to the inner manager, then overwrites the assembled Tools
// with the injected schemas.
func (m *SchemaInjectingContextManager) Assemble(ctx context.Context, session *sporecore.SessionState, task *sporecore.Task) sporecore.Context {
	c := m.inner.Assemble(ctx, session, task)
	c.Tools = append([]sporecore.ToolSchema(nil), m.tools...)
	return c
}

// AppendToolResult forwards to the inner manager.
func (m *SchemaInjectingContextManager) AppendToolResult(ctx context.Context, session *sporecore.SessionState, result *sporecore.HarnessToolResult) {
	m.inner.AppendToolResult(ctx, session, result)
}

// AppendUserMessage forwards to the inner manager.
func (m *SchemaInjectingContextManager) AppendUserMessage(ctx context.Context, session *sporecore.SessionState, text string) {
	m.inner.AppendUserMessage(ctx, session, text)
}

// ShouldCompact forwards to the inner manager.
func (m *SchemaInjectingContextManager) ShouldCompact(session *sporecore.SessionState) bool {
	return m.inner.ShouldCompact(session)
}

// PrepareCompactionTurn forwards to the inner manager's compaction seam when it
// implements CompactingContextManager; otherwise reports nothing to compact.
func (m *SchemaInjectingContextManager) PrepareCompactionTurn(session *sporecore.SessionState) (*sporecore.CompactionTurn, bool) {
	if cm, ok := m.inner.(sporecore.CompactingContextManager); ok {
		return cm.PrepareCompactionTurn(session)
	}
	return nil, false
}

// InjectMissingItems forwards to the inner compaction seam (no-op otherwise).
func (m *SchemaInjectingContextManager) InjectMissingItems(c *sporecore.Context, missing []string) {
	if cm, ok := m.inner.(sporecore.CompactingContextManager); ok {
		cm.InjectMissingItems(c, missing)
	}
}

// ApplyCompaction forwards to the inner compaction seam (no-op otherwise).
func (m *SchemaInjectingContextManager) ApplyCompaction(session *sporecore.SessionState, summary string) {
	if cm, ok := m.inner.(sporecore.CompactingContextManager); ok {
		cm.ApplyCompaction(session, summary)
	}
}

// TokenBudgetUsed forwards to the inner TokenBudgetReader seam so the harness
// can stamp the real post-compaction budget (issue #57).
func (m *SchemaInjectingContextManager) TokenBudgetUsed(session *sporecore.SessionState) (uint32, bool) {
	if r, ok := m.inner.(sporecore.TokenBudgetReader); ok {
		return r.TokenBudgetUsed(session)
	}
	return 0, false
}

var (
	_ sporecore.ContextManager           = (*SchemaInjectingContextManager)(nil)
	_ sporecore.CompactingContextManager = (*SchemaInjectingContextManager)(nil)
	_ sporecore.TokenBudgetReader        = (*SchemaInjectingContextManager)(nil)
)

// ============================================================================
// Compaction state seeding (S3)
// ============================================================================

// SeedCompactionState seeds a harness SessionState with rich compaction state
// for S3: a small window, a budget near the threshold, and a history longer than
// preserve_recent_n so compaction fires mid-run. The session can then compact,
// continue, and compact again because the token-accounting fix decrements the
// budget on each compaction.
func SeedCompactionState(
	session *sporecore.SessionState,
	taskInstruction string,
	sessionID sporecore.SessionID,
	taskID sporecore.TaskID,
	windowLimit uint32,
	tokenBudgetUsed uint32,
	historyLen int,
) {
	rich := contextmgr.NewSessionState(sessionID, taskID, taskInstruction)
	rich.WindowLimit = windowLimit
	rich.TokenBudgetUsed = tokenBudgetUsed
	hist := make([]sporecore.Message, 0, historyLen)
	for i := 0; i < historyLen; i++ {
		role := sporecore.RoleUser
		if i%2 == 1 {
			role = sporecore.RoleAssistant
		}
		hist = append(hist, sporecore.Message{
			Role: role,
			Content: sporecore.NewTextContent(
				"history message: progress notes on the payment service deploy with " +
					"enough content to carry a meaningful token estimate for reclamation",
			),
		})
	}
	rich.MessageHistory = hist
	contextmgr.SeedRichState(session, &rich)
}

// ============================================================================
// Scenario assembly
// ============================================================================

// BuildScenario assembles a StandardHarness for a scenario from injected
// components. Generic over the agent and tool registry so live mode
// (ollama.ModelInterface/ModelAgent + StandardToolRegistry) and mock mode
// (MockAgent + ScriptedToolRegistry) share one code path.
//
// toolSchemas are injected into every assembled context (sorted by name) via the
// SchemaInjectingContextManager. Pass the registry's schemas in live mode, or nil
// in mock mode where the scripted agent does not need them.
//
// observability is optional: the command passes a durable-outbox observer; the
// hermetic tests pass an in-memory observer to assert spans; nil runs with none.
func BuildScenario(
	agent sporecore.Agent,
	toolRegistry sporecore.ToolRegistry,
	sandbox sporecore.SandboxProvider,
	contextManager sporecore.ContextManager,
	termination sporecore.TerminationPolicy,
	toolSchemas []sporecore.ToolSchema,
	verifier sporecore.CompactionVerifier,
	maxCompactionAttempts uint32,
	observability sporecore.HarnessObserver,
) *sporecore.StandardHarness {
	cm := NewSchemaInjectingContextManager(contextManager, toolSchemas)
	return sporecore.NewStandardHarness(sporecore.HarnessConfig{
		Agent:                 agent,
		ToolRegistry:          toolRegistry,
		Sandbox:               sandbox,
		ContextManager:        cm,
		TerminationPolicy:     termination,
		CompactionVerifier:    verifier,
		MaxCompactionAttempts: maxCompactionAttempts,
		Observability:         observability,
	})
}
