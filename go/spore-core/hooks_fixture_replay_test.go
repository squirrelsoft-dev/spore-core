package sporecore

import (
	"bytes"
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

// hooksFixturePath resolves a file under fixtures/hooks/.
func hooksFixturePath(t *testing.T, name string) string {
	t.Helper()
	_, this, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("runtime.Caller failed")
	}
	return filepath.Join(filepath.Dir(this), "..", "..", "fixtures", "hooks", name)
}

func readFixture(t *testing.T, name string) []byte {
	t.Helper()
	raw, err := os.ReadFile(hooksFixturePath(t, name))
	if err != nil {
		t.Fatalf("read fixture %s: %v", name, err)
	}
	return raw
}

// ── hook_decision_wire.json ────────────────────────────────────────────────
// Serializing each variant must reproduce `json` byte-for-byte (compacted),
// and deserializing `json` must round-trip.
func TestFixtureHookDecisionWire(t *testing.T) {
	var fixture struct {
		Cases []struct {
			Name string          `json:"name"`
			JSON json.RawMessage `json:"json"`
		} `json:"cases"`
	}
	if err := json.Unmarshal(readFixture(t, "hook_decision_wire.json"), &fixture); err != nil {
		t.Fatal(err)
	}
	if len(fixture.Cases) != 5 {
		t.Fatalf("expected 5 cases, got %d", len(fixture.Cases))
	}
	for _, c := range fixture.Cases {
		t.Run(c.Name, func(t *testing.T) {
			var d HookDecision
			if err := json.Unmarshal(c.JSON, &d); err != nil {
				t.Fatalf("unmarshal: %v", err)
			}
			got, err := json.Marshal(d)
			if err != nil {
				t.Fatal(err)
			}
			want := compact(t, c.JSON)
			if !bytes.Equal(got, want) {
				t.Fatalf("got %s, want %s", got, want)
			}
		})
	}
}

// ── command_handler_io.json ─────────────────────────────────────────────────
// Pins the CommandHook stdin shape (event + per-event context) and the parsed
// decision / error for each case.
func TestFixtureCommandHandlerIO(t *testing.T) {
	var fixture struct {
		Cases []struct {
			Name             string          `json:"name"`
			Event            HookEvent       `json:"event"`
			ExpectedStdin    json.RawMessage `json:"expected_stdin"`
			Stdout           string          `json:"stdout"`
			ExitCode         int             `json:"exit_code"`
			ExpectedDecision json.RawMessage `json:"expected_decision"`
			ExpectedError    string          `json:"expected_error"`
		} `json:"cases"`
	}
	if err := json.Unmarshal(readFixture(t, "command_handler_io.json"), &fixture); err != nil {
		t.Fatal(err)
	}

	for _, c := range fixture.Cases {
		t.Run(c.Name, func(t *testing.T) {
			hctx := stopContextFromFixtureEvent(t, c.Event)

			// 1. Verify the stdin payload the harness would send matches
			//    expected_stdin (when the fixture pins it).
			if len(c.ExpectedStdin) > 0 {
				payload, err := hctx.toPayload()
				if err != nil {
					t.Fatal(err)
				}
				stdin, err := json.Marshal(struct {
					Event   HookEvent       `json:"event"`
					Context json.RawMessage `json:"context"`
				}{c.Event, payload})
				if err != nil {
					t.Fatal(err)
				}
				// Compare semantically: the harness emits map keys in sorted
				// order, while the fixture pins an author-chosen key order. JSON
				// object key order is not significant on the wire.
				if !jsonEqual(t, stdin, c.ExpectedStdin) {
					t.Fatalf("stdin\n got: %s\nwant: %s", stdin, compact(t, c.ExpectedStdin))
				}
			}

			// 2. Drive a real CommandHook whose script emits the fixture's
			//    stdout and exit code, and assert the parsed decision / error.
			hook := scriptHook(t, c.Event, c.Stdout, c.ExitCode)
			chain := NewStandardHookChain()
			if err := chain.Register(hook); err != nil {
				t.Fatal(err)
			}
			outcome, err := chain.Fire(context.Background(), hctx)

			switch c.ExpectedError {
			case "":
				if err != nil {
					t.Fatalf("unexpected error: %v", err)
				}
				var want HookDecision
				if err := json.Unmarshal(c.ExpectedDecision, &want); err != nil {
					t.Fatal(err)
				}
				assertOutcomeMatchesDecision(t, outcome, want)
			case "command_failed":
				var he *HookError
				if !asHookError(err, &he) || he.Kind != HookErrCommandFailed {
					t.Fatalf("err = %v, want CommandFailed", err)
				}
			case "command_output_invalid":
				var he *HookError
				if !asHookError(err, &he) || he.Kind != HookErrCommandOutputInvalid {
					t.Fatalf("err = %v, want CommandOutputInvalid", err)
				}
			default:
				t.Fatalf("unknown expected_error %q", c.ExpectedError)
			}
		})
	}
}

// stopContextFromFixtureEvent builds the HookContext matching the fixture's
// pinned stdin for the given event.
func stopContextFromFixtureEvent(t *testing.T, event HookEvent) *HookContext {
	t.Helper()
	switch event {
	case HookEventStop:
		out := TurnOutput{Text: "I'm done", HadToolCalls: false}
		instr := "make the tests pass"
		return &HookContext{
			Event:           HookEventStop,
			SessionID:       SessionID("sess-1"),
			TurnNumber:      3,
			LastOutput:      &out,
			TaskInstruction: &instr,
			SessionState:    nil, // fixture pins session_state: null
		}
	case HookEventPreToolUse:
		input := json.RawMessage(`{"path":"/etc/passwd"}`)
		return &HookContext{
			Event:      HookEventPreToolUse,
			SessionID:  SessionID("sess-1"),
			TurnNumber: 1,
			ToolName:   "read_file",
			ToolInput:  &input,
		}
	default:
		t.Fatalf("unhandled fixture event %q", event)
		return nil
	}
}

// scriptHook builds a CommandHook backed by a shell script that consumes stdin,
// emits stdout, and exits with exitCode.
func scriptHook(t *testing.T, event HookEvent, stdout string, exitCode int) *CommandHook {
	t.Helper()
	script := filepath.Join(t.TempDir(), "hook.sh")
	body := "#!/bin/sh\ncat >/dev/null\n"
	if stdout != "" {
		body += "printf '%s' " + shellQuote(stdout) + "\n"
	}
	body += "exit " + itoaFull(exitCode) + "\n"
	if err := os.WriteFile(script, []byte(body), 0o755); err != nil {
		t.Fatal(err)
	}
	return NewCommandHook("fixture", []HookEvent{event}, "sh", []string{script})
}

func assertOutcomeMatchesDecision(t *testing.T, outcome FireOutcome, want HookDecision) {
	t.Helper()
	switch want.Decision {
	case HookDecisionBlock:
		if outcome.Kind != FireBlock || outcome.Reason != want.Reason {
			t.Fatalf("outcome = %+v, want block %q", outcome, want.Reason)
		}
	case HookDecisionDeny:
		if outcome.Kind != FireDeny || outcome.Reason != want.Reason {
			t.Fatalf("outcome = %+v, want deny %q", outcome, want.Reason)
		}
	case HookDecisionContinue:
		if outcome.Kind != FireContinue {
			t.Fatalf("outcome = %+v, want continue", outcome)
		}
	default:
		t.Fatalf("unhandled expected decision %q", want.Decision)
	}
}

// ── pre_tool_use_mutation.json ──────────────────────────────────────────────
// Replays an ordered chain of decisions against a tool_input and asserts the
// final (mutated) input or a deny.
func TestFixturePreToolUseMutation(t *testing.T) {
	var fixture struct {
		Cases []struct {
			Name          string            `json:"name"`
			ToolName      string            `json:"tool_name"`
			ToolInput     json.RawMessage   `json:"tool_input"`
			HookDecisions []json.RawMessage `json:"hook_decisions"`
			Expected      struct {
				Outcome   string          `json:"outcome"`
				ToolInput json.RawMessage `json:"tool_input"`
				Reason    string          `json:"reason"`
			} `json:"expected"`
		} `json:"cases"`
	}
	if err := json.Unmarshal(readFixture(t, "pre_tool_use_mutation.json"), &fixture); err != nil {
		t.Fatal(err)
	}

	for _, c := range fixture.Cases {
		t.Run(c.Name, func(t *testing.T) {
			chain := NewStandardHookChain()
			for i, raw := range c.HookDecisions {
				var d HookDecision
				if err := json.Unmarshal(raw, &d); err != nil {
					t.Fatal(err)
				}
				name := "h" + itoaFull(i)
				if err := chain.Register(fnHook(name, []HookEvent{HookEventPreToolUse}, d)); err != nil {
					t.Fatal(err)
				}
			}
			input := append(json.RawMessage(nil), compact(t, c.ToolInput)...)
			hctx := &HookContext{
				Event:      HookEventPreToolUse,
				SessionID:  SessionID("sess-1"),
				TurnNumber: 1,
				ToolName:   c.ToolName,
				ToolInput:  &input,
			}
			outcome, err := chain.Fire(context.Background(), hctx)
			if err != nil {
				t.Fatal(err)
			}
			switch c.Expected.Outcome {
			case "continue":
				if outcome.Kind != FireContinue {
					t.Fatalf("outcome = %+v, want continue", outcome)
				}
				if !jsonEqual(t, input, c.Expected.ToolInput) {
					t.Fatalf("tool_input = %s, want %s", input, compact(t, c.Expected.ToolInput))
				}
			case "deny":
				if outcome.Kind != FireDeny || outcome.Reason != c.Expected.Reason {
					t.Fatalf("outcome = %+v, want deny %q", outcome, c.Expected.Reason)
				}
			default:
				t.Fatalf("unknown outcome %q", c.Expected.Outcome)
			}
		})
	}
}

// ── stop_block_basic.json ───────────────────────────────────────────────────
// Replays a sequence of Stop-hook decisions through the per-run block-count /
// cap logic (mirroring StandardHarness.fireStopHooks) and asserts the honored
// block count and how termination was reached.
func TestFixtureStopBlockBasic(t *testing.T) {
	var fixture struct {
		Cases []struct {
			Name          string            `json:"name"`
			MaxStopBlocks uint32            `json:"max_stop_blocks"`
			HookDecisions []json.RawMessage `json:"hook_decisions"`
			Expected      struct {
				Blocks       int    `json:"blocks"`
				TerminatedBy string `json:"terminated_by"`
			} `json:"expected"`
		} `json:"cases"`
	}
	if err := json.Unmarshal(readFixture(t, "stop_block_basic.json"), &fixture); err != nil {
		t.Fatal(err)
	}

	for _, c := range fixture.Cases {
		t.Run(c.Name, func(t *testing.T) {
			// Decode the scripted decisions, fired one per simulated termination
			// attempt.
			decisions := make([]HookDecision, len(c.HookDecisions))
			for i, raw := range c.HookDecisions {
				if err := json.Unmarshal(raw, &decisions[i]); err != nil {
					t.Fatal(err)
				}
			}
			cap := c.MaxStopBlocks

			var blocks uint32
			terminatedBy := ""
			// Walk the decision sequence: each is the Stop-chain's outcome at
			// one termination attempt. A block under the cap consumes one block
			// and continues; once blocks == cap the next block is ignored and
			// the loop terminates ("cap"); a continue terminates ("continue").
			for _, d := range decisions {
				if d.Decision == HookDecisionBlock {
					if blocks >= cap {
						terminatedBy = "cap"
						break
					}
					blocks++
					continue
				}
				// Continue (or any non-block) → normal termination.
				terminatedBy = "continue"
				break
			}
			if terminatedBy == "" {
				// Ran out of scripted decisions while still blocking under cap;
				// treat the final state as cap-bound termination.
				terminatedBy = "cap"
			}

			if int(blocks) != c.Expected.Blocks {
				t.Fatalf("blocks = %d, want %d", blocks, c.Expected.Blocks)
			}
			if terminatedBy != c.Expected.TerminatedBy {
				t.Fatalf("terminated_by = %q, want %q", terminatedBy, c.Expected.TerminatedBy)
			}
		})
	}
}

// jsonEqual reports whether two JSON documents are semantically equal
// (key order and insignificant whitespace ignored).
func jsonEqual(t *testing.T, a, b json.RawMessage) bool {
	t.Helper()
	var av, bv any
	if err := json.Unmarshal(a, &av); err != nil {
		t.Fatalf("unmarshal a: %v", err)
	}
	if err := json.Unmarshal(b, &bv); err != nil {
		t.Fatalf("unmarshal b: %v", err)
	}
	ab, _ := json.Marshal(av)
	bb, _ := json.Marshal(bv)
	return bytes.Equal(ab, bb)
}

// compact returns the compacted (canonical) form of a JSON document.
func compact(t *testing.T, raw json.RawMessage) []byte {
	t.Helper()
	var buf bytes.Buffer
	if err := json.Compact(&buf, raw); err != nil {
		t.Fatalf("compact: %v", err)
	}
	return buf.Bytes()
}

func itoaFull(n int) string {
	if n == 0 {
		return "0"
	}
	neg := n < 0
	if neg {
		n = -n
	}
	var b []byte
	for n > 0 {
		b = append([]byte{byte('0' + n%10)}, b...)
		n /= 10
	}
	if neg {
		b = append([]byte{'-'}, b...)
	}
	return string(b)
}

func shellQuote(s string) string {
	return "'" + bytesReplaceAll(s, "'", `'\''`) + "'"
}

func bytesReplaceAll(s, old, new string) string {
	return string(bytes.ReplaceAll([]byte(s), []byte(old), []byte(new)))
}
