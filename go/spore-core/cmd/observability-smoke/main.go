// Command observability-smoke is the Go counterpart of the Rust
// `examples/observability_smoke.rs` recipe (issues #33 / #50).
//
// It drives a durable-outbox ObservabilityProvider directly: emits a turn span,
// a child tool-call span, and a terminal session summary, writing JSONL under
// <root>/sessions/. When SPORE_OTLP_ENDPOINT is set, the same spans are ALSO
// forwarded to Tempo over OTLP gRPC, grouped under the JSONL trace_id.
//
// Usage (against the local stack in observability/):
//
//	SPORE_OTLP_ENDPOINT=http://localhost:4317 \
//	  go run ./cmd/observability-smoke
//
// It prints the session id, the on-disk trace path, and the trace_id. Verify
// the grouped trace landed in Tempo with:
//
//	curl -s http://localhost:3200/api/traces/<trace_id> | jq '.batches | length'
//
// and check the "Spore" folder in Grafana. With SPORE_OTLP_ENDPOINT unset it
// writes JSONL only (no Tempo forwarding).
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/squirrelsoft-dev/spore-core/go/spore-core/guideregistry"
	obs "github.com/squirrelsoft-dev/spore-core/go/spore-core/observability"
)

func main() {
	now := time.Now().UTC()
	// Unique per-run session id so repeated runs don't collide in Tempo.
	sessionRaw := "live-smoke-" + strings.NewReplacer(":", "-", ".", "-").Replace(now.Format("2006-01-02T15:04:05.000Z07:00"))
	sessionID := obs.SessionID(sessionRaw)
	taskID := obs.TaskID("smoke-task")

	root := ".spore"
	if wd, err := os.Getwd(); err == nil {
		root = filepath.Join(wd, ".spore")
	}

	endpoint := os.Getenv("SPORE_OTLP_ENDPOINT")
	if strings.TrimSpace(endpoint) == "" {
		fmt.Fprintln(os.Stderr,
			"note: SPORE_OTLP_ENDPOINT is unset — writing JSONL only (no Tempo forwarding).\n"+
				"set SPORE_OTLP_ENDPOINT=http://localhost:4317 to forward to the local stack.")
	}

	provider := obs.NewOutboxObservabilityProvider(obs.NewOutboxConfig(root))

	startedAt := obs.Timestamp(now.Format("2006-01-02T15:04:05.000Z07:00"))
	cacheRead := uint32(1600)
	cacheWrite := uint32(0)

	// Turn span (root).
	turnBase := obs.NewRoot(obs.SpanID("turn-1"), sessionID, taskID, obs.SpanKindTurn, startedAt)
	turnBase.Finish(startedAt, obs.NewStatusOk(), 2100)
	provider.EmitTurn(obs.TurnSpan{
		Base:               turnBase,
		TurnNumber:         1,
		InputTokens:        1820,
		OutputTokens:       140,
		CacheReadTokens:    &cacheRead,
		CacheWriteTokens:   &cacheWrite,
		CostUSD:            0.0123,
		StopReason:         obs.StopReason("tool_use"),
		ToolCallsRequested: 1,
	})

	// Tool-call span (child of the turn).
	toolBase := obs.NewChild(obs.SpanID("toolcall-1"), turnBase, obs.SpanKindToolCall, startedAt)
	toolBase.Finish(startedAt, obs.NewStatusOk(), 35)
	provider.EmitToolCall(obs.ToolCallSpan{
		Base:                toolBase,
		ToolName:            "read_file",
		CallID:              "call-1",
		ParametersSizeBytes: 24,
		OutputSizeBytes:     128,
		Truncated:           false,
	})

	// Terminal outcome + flush (writes the session summary line + force-flushes
	// OTLP best-effort).
	provider.SetSessionOutcome(sessionID, guideregistry.NewOutcomeSuccess())
	if err := provider.FlushSession(context.Background(), sessionID); err != nil {
		fmt.Fprintf(os.Stderr, "flush failed: %v\n", err)
		os.Exit(1)
	}

	tracePath := filepath.Join(root, "sessions", string(sessionID), "trace.jsonl")
	traceID, kinds := summarize(tracePath)

	fmt.Printf("session_id : %s\n", sessionID)
	fmt.Printf("trace_path : %s\n", tracePath)
	fmt.Printf("trace_id   : %s\n", traceID)
	fmt.Printf("span kinds : %v\n", kinds)
	if strings.TrimSpace(endpoint) != "" {
		fmt.Printf("verify     : curl -s http://localhost:3200/api/traces/%s | jq '.batches | length'\n", traceID)
	}
}

func summarize(path string) (traceID string, kinds []string) {
	raw, err := os.ReadFile(path)
	if err != nil {
		fmt.Fprintf(os.Stderr, "read %s: %v\n", path, err)
		os.Exit(1)
	}
	for _, line := range strings.Split(strings.TrimRight(string(raw), "\n"), "\n") {
		if line == "" {
			continue
		}
		var m map[string]any
		if err := json.Unmarshal([]byte(line), &m); err != nil {
			continue
		}
		if traceID == "" {
			if t, ok := m["trace_id"].(string); ok {
				traceID = t
			}
		}
		if k, ok := m["kind"].(string); ok {
			kinds = append(kinds, k)
		}
	}
	return traceID, kinds
}
