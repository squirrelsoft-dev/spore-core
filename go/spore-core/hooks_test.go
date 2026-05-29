package sporecore

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func sid() SessionID { return SessionID("s1") }

func fnHook(name string, events []HookEvent, d HookDecision) *FunctionHook {
	return NewFunctionHook(name, events, func(_ context.Context, _ *HookContext) (HookDecision, error) {
		return d, nil
	})
}

// ── R3 / R25: registration-order firing, per-event filtering ───────────────
func TestFiresInRegistrationOrder(t *testing.T) {
	var mu sync.Mutex
	var order []string
	chain := NewStandardHookChain()
	for _, label := range []string{"a", "b", "c"} {
		label := label
		if err := chain.Register(NewFunctionHook(label, []HookEvent{HookEventPostTurn},
			func(_ context.Context, _ *HookContext) (HookDecision, error) {
				mu.Lock()
				order = append(order, label)
				mu.Unlock()
				return Continue(), nil
			})); err != nil {
			t.Fatal(err)
		}
	}
	// A hook for a different event must not fire.
	if err := chain.Register(fnHook("z", []HookEvent{HookEventPreTurn}, Continue())); err != nil {
		t.Fatal(err)
	}
	out := TurnOutput{}
	hctx := &HookContext{Event: HookEventPostTurn, SessionID: sid(), TurnNumber: 1, Output: &out}
	if _, err := chain.Fire(context.Background(), hctx); err != nil {
		t.Fatal(err)
	}
	if got := order; len(got) != 3 || got[0] != "a" || got[1] != "b" || got[2] != "c" {
		t.Fatalf("order = %v, want [a b c]", got)
	}
}

// ── R1 / R16: pre-hook mutation in place ────────────────────────────────────
func TestPreToolUseMutatesInputInPlace(t *testing.T) {
	chain := NewStandardHookChain()
	if err := chain.Register(NewFunctionHook("mut", []HookEvent{HookEventPreToolUse},
		func(_ context.Context, hctx *HookContext) (HookDecision, error) {
			*hctx.ToolInput = json.RawMessage(`{"path":"/safe"}`)
			return Continue(), nil
		})); err != nil {
		t.Fatal(err)
	}
	input := json.RawMessage(`{"path":"/etc/passwd"}`)
	hctx := &HookContext{Event: HookEventPreToolUse, SessionID: sid(), TurnNumber: 1, ToolName: "read_file", ToolInput: &input}
	out, err := chain.Fire(context.Background(), hctx)
	if err != nil {
		t.Fatal(err)
	}
	if out.Kind != FireContinue {
		t.Fatalf("outcome = %v", out.Kind)
	}
	if string(input) != `{"path":"/safe"}` {
		t.Fatalf("input = %s", input)
	}
}

// ── R2: pre-hook chain threads mutation to the next hook ────────────────────
func TestPreHookChainThreadsMutation(t *testing.T) {
	chain := NewStandardHookChain()
	if err := chain.Register(NewFunctionHook("first", []HookEvent{HookEventPreToolUse},
		func(_ context.Context, hctx *HookContext) (HookDecision, error) {
			*hctx.ToolInput = json.RawMessage(`{"v":1}`)
			return Continue(), nil
		})); err != nil {
		t.Fatal(err)
	}
	if err := chain.Register(NewFunctionHook("second", []HookEvent{HookEventPreToolUse},
		func(_ context.Context, hctx *HookContext) (HookDecision, error) {
			var probe struct {
				V int `json:"v"`
			}
			if err := json.Unmarshal(*hctx.ToolInput, &probe); err != nil {
				return HookDecision{}, err
			}
			*hctx.ToolInput = json.RawMessage([]byte(`{"v":` + itoa(probe.V+1) + `}`))
			return Continue(), nil
		})); err != nil {
		t.Fatal(err)
	}
	input := json.RawMessage(`{}`)
	hctx := &HookContext{Event: HookEventPreToolUse, SessionID: sid(), TurnNumber: 1, ToolName: "t", ToolInput: &input}
	if _, err := chain.Fire(context.Background(), hctx); err != nil {
		t.Fatal(err)
	}
	if string(input) != `{"v":2}` {
		t.Fatalf("input = %s, want {\"v\":2}", input)
	}
}

func itoa(n int) string { return string([]byte{byte('0' + n)}) }

// ── R6: Mutate decision replaces the mutable field ──────────────────────────
func TestMutateDecisionReplacesField(t *testing.T) {
	chain := NewStandardHookChain()
	if err := chain.Register(fnHook("m", []HookEvent{HookEventPreToolUse},
		Mutate(json.RawMessage(`{"replaced":true}`)))); err != nil {
		t.Fatal(err)
	}
	input := json.RawMessage(`{"orig":1}`)
	hctx := &HookContext{Event: HookEventPreToolUse, SessionID: sid(), TurnNumber: 1, ToolName: "t", ToolInput: &input}
	if _, err := chain.Fire(context.Background(), hctx); err != nil {
		t.Fatal(err)
	}
	if string(input) != `{"replaced":true}` {
		t.Fatalf("input = %s", input)
	}
}

// ── R15: PreToolUse deny ────────────────────────────────────────────────────
func TestPreToolUseDeny(t *testing.T) {
	chain := NewStandardHookChain()
	if err := chain.Register(fnHook("deny", []HookEvent{HookEventPreToolUse}, Deny("blocked path"))); err != nil {
		t.Fatal(err)
	}
	input := json.RawMessage(`{}`)
	hctx := &HookContext{Event: HookEventPreToolUse, SessionID: sid(), TurnNumber: 1, ToolName: "t", ToolInput: &input}
	out, err := chain.Fire(context.Background(), hctx)
	if err != nil {
		t.Fatal(err)
	}
	if out.Kind != FireDeny || out.Reason != "blocked path" {
		t.Fatalf("outcome = %+v", out)
	}
}

// ── R10 / R12: sync post-hook (Stop) block ──────────────────────────────────
func TestStopHookBlock(t *testing.T) {
	chain := NewStandardHookChain()
	if err := chain.Register(fnHook("verify", []HookEvent{HookEventStop}, Block("tests failing"))); err != nil {
		t.Fatal(err)
	}
	out := TurnOutput{}
	state := SessionState{}
	instr := "do it"
	hctx := &HookContext{Event: HookEventStop, SessionID: sid(), TurnNumber: 3, LastOutput: &out, TaskInstruction: &instr, SessionState: &state}
	outcome, err := chain.Fire(context.Background(), hctx)
	if err != nil {
		t.Fatal(err)
	}
	if outcome.Kind != FireBlock || outcome.Reason != "tests failing" {
		t.Fatalf("outcome = %+v", outcome)
	}
}

// ── R13: Stop all-continue terminates ───────────────────────────────────────
func TestStopHookAllContinue(t *testing.T) {
	chain := NewStandardHookChain()
	if err := chain.Register(fnHook("ok", []HookEvent{HookEventStop}, Continue())); err != nil {
		t.Fatal(err)
	}
	out := TurnOutput{}
	state := SessionState{}
	instr := "x"
	hctx := &HookContext{Event: HookEventStop, SessionID: sid(), TurnNumber: 1, LastOutput: &out, TaskInstruction: &instr, SessionState: &state}
	outcome, err := chain.Fire(context.Background(), hctx)
	if err != nil {
		t.Fatal(err)
	}
	if outcome.Kind != FireContinue {
		t.Fatalf("outcome = %+v", outcome)
	}
}

// ── R8: Stop registered async is rejected ───────────────────────────────────
func TestStopAsyncRejected(t *testing.T) {
	chain := NewStandardHookChain()
	hook := NewFunctionHook("s", []HookEvent{HookEventStop},
		func(_ context.Context, _ *HookContext) (HookDecision, error) { return Continue(), nil }).Async()
	err := chain.Register(hook)
	var he *HookError
	if !asHookError(err, &he) || he.Kind != HookErrSyncOnlyEvent {
		t.Fatalf("err = %v, want SyncOnlyEvent", err)
	}
}

// ── R9: OnPause registered sync is rejected ─────────────────────────────────
func TestOnPauseSyncRejected(t *testing.T) {
	chain := NewStandardHookChain()
	err := chain.Register(fnHook("p", []HookEvent{HookEventOnPause}, Continue()))
	var he *HookError
	if !asHookError(err, &he) || he.Kind != HookErrAsyncOnlyEvent {
		t.Fatalf("err = %v, want AsyncOnlyEvent", err)
	}
}

// ── R4 / R17 / R24: illegal Block on a non-blocking event rejected at fire ──
func TestIllegalBlockOnPostTurn(t *testing.T) {
	chain := NewStandardHookChain()
	if err := chain.Register(fnHook("bad", []HookEvent{HookEventPostTurn}, Block("no"))); err != nil {
		t.Fatal(err)
	}
	out := TurnOutput{}
	hctx := &HookContext{Event: HookEventPostTurn, SessionID: sid(), TurnNumber: 1, Output: &out}
	_, err := chain.Fire(context.Background(), hctx)
	var he *HookError
	if !asHookError(err, &he) || he.Kind != HookErrIllegalDecision {
		t.Fatalf("err = %v, want IllegalDecision", err)
	}
}

// ── R5: Deny outside PreToolUse/OnSubagentSpawn rejected ────────────────────
func TestDenyValidation(t *testing.T) {
	if err := Deny("x").ValidateFor(HookEventPreToolUse); err != nil {
		t.Fatalf("deny on pre_tool_use should be legal: %v", err)
	}
	if err := Deny("x").ValidateFor(HookEventOnSubagentSpawn); err != nil {
		t.Fatalf("deny on on_subagent_spawn should be legal: %v", err)
	}
	if err := Deny("x").ValidateFor(HookEventPreTurn); err == nil {
		t.Fatal("deny on pre_turn should be illegal")
	}
}

// ── R11: async fire-and-forget not awaited (no block, continues) ────────────
func TestAsyncPostHookFireAndForget(t *testing.T) {
	chain := NewStandardHookChain()
	var fired int32
	done := make(chan struct{})
	hook := NewFunctionHook("log", []HookEvent{HookEventPostTurn},
		func(_ context.Context, _ *HookContext) (HookDecision, error) {
			atomic.AddInt32(&fired, 1)
			close(done)
			// Even a Block here must not affect the outcome — async results are
			// swallowed.
			return Block("ignored"), nil
		}).Async()
	if err := chain.Register(hook); err != nil {
		t.Fatal(err)
	}
	out := TurnOutput{}
	hctx := &HookContext{Event: HookEventPostTurn, SessionID: sid(), TurnNumber: 1, Output: &out}
	outcome, err := chain.Fire(context.Background(), hctx)
	if err != nil {
		t.Fatal(err)
	}
	if outcome.Kind != FireContinue {
		t.Fatalf("outcome = %+v, want Continue (async result swallowed)", outcome)
	}
	// The goroutine still ran (fire-and-forget), just not awaited by Fire.
	select {
	case <-done:
	case <-time.After(2 * time.Second):
		t.Fatal("async hook never fired")
	}
	if atomic.LoadInt32(&fired) != 1 {
		t.Fatalf("fired = %d", fired)
	}
}

// ── R7: Inject aggregation ──────────────────────────────────────────────────
func TestInjectAggregatesNewlineJoined(t *testing.T) {
	chain := NewStandardHookChain()
	if err := chain.Register(fnHook("i1", []HookEvent{HookEventPreTurn}, Inject("one"))); err != nil {
		t.Fatal(err)
	}
	if err := chain.Register(fnHook("i2", []HookEvent{HookEventPreTurn}, Inject("two"))); err != nil {
		t.Fatal(err)
	}
	cb := ContextBlock{}
	hctx := &HookContext{Event: HookEventPreTurn, SessionID: sid(), TurnNumber: 1, ContextBlock: &cb}
	outcome, err := chain.Fire(context.Background(), hctx)
	if err != nil {
		t.Fatal(err)
	}
	if outcome.Kind != FireInject || outcome.Context != "one\ntwo" {
		t.Fatalf("outcome = %+v", outcome)
	}
}

// ── R23: FunctionHook runs the closure (and mutates in place) ───────────────
func TestFunctionHookRuns(t *testing.T) {
	chain := NewStandardHookChain()
	hook := NewFunctionHook("f", []HookEvent{HookEventOnLoopStart},
		func(_ context.Context, hctx *HookContext) (HookDecision, error) {
			*hctx.TaskInstruction += " [checked]"
			return Continue(), nil
		})
	if err := chain.Register(hook); err != nil {
		t.Fatal(err)
	}
	instr := "do work"
	cfg := HarnessConfig{}
	hctx := &HookContext{Event: HookEventOnLoopStart, SessionID: sid(), TaskInstruction: &instr, Config: &cfg}
	if _, err := chain.Fire(context.Background(), hctx); err != nil {
		t.Fatal(err)
	}
	if instr != "do work [checked]" {
		t.Fatalf("instr = %q", instr)
	}
}

// ── R18-R21: CommandHook stdin/stdout roundtrip ─────────────────────────────
func TestCommandHookRoundtrip(t *testing.T) {
	dir := t.TempDir()
	script := filepath.Join(dir, "hook.sh")
	if err := os.WriteFile(script,
		[]byte("#!/bin/sh\ncat >/dev/null\necho '{\"decision\":\"block\",\"reason\":\"cmd says no\"}'\n"),
		0o755); err != nil {
		t.Fatal(err)
	}
	hook := NewCommandHook("cmd", []HookEvent{HookEventStop}, "sh", []string{script})
	chain := NewStandardHookChain()
	if err := chain.Register(hook); err != nil {
		t.Fatal(err)
	}
	out := TurnOutput{}
	state := SessionState{}
	instr := "x"
	hctx := &HookContext{Event: HookEventStop, SessionID: sid(), TurnNumber: 1, LastOutput: &out, TaskInstruction: &instr, SessionState: &state}
	outcome, err := chain.Fire(context.Background(), hctx)
	if err != nil {
		t.Fatal(err)
	}
	if outcome.Kind != FireBlock || outcome.Reason != "cmd says no" {
		t.Fatalf("outcome = %+v", outcome)
	}
}

// ── R20: CommandHook nonzero exit → CommandFailed ───────────────────────────
func TestCommandHookNonzeroExitErrors(t *testing.T) {
	hook := NewCommandHook("cmd", []HookEvent{HookEventStop}, "sh", []string{"-c", "exit 7"})
	chain := NewStandardHookChain()
	if err := chain.Register(hook); err != nil {
		t.Fatal(err)
	}
	out := TurnOutput{}
	state := SessionState{}
	instr := "x"
	hctx := &HookContext{Event: HookEventStop, SessionID: sid(), TurnNumber: 1, LastOutput: &out, TaskInstruction: &instr, SessionState: &state}
	_, err := chain.Fire(context.Background(), hctx)
	var he *HookError
	if !asHookError(err, &he) || he.Kind != HookErrCommandFailed || he.Code != 7 {
		t.Fatalf("err = %v, want CommandFailed code 7", err)
	}
}

// ── R21: CommandHook malformed stdout → CommandOutputInvalid ────────────────
func TestCommandHookMalformedStdout(t *testing.T) {
	hook := NewCommandHook("cmd", []HookEvent{HookEventStop}, "sh", []string{"-c", "cat >/dev/null; echo 'not json'"})
	chain := NewStandardHookChain()
	if err := chain.Register(hook); err != nil {
		t.Fatal(err)
	}
	out := TurnOutput{}
	state := SessionState{}
	instr := "x"
	hctx := &HookContext{Event: HookEventStop, SessionID: sid(), TurnNumber: 1, LastOutput: &out, TaskInstruction: &instr, SessionState: &state}
	_, err := chain.Fire(context.Background(), hctx)
	var he *HookError
	if !asHookError(err, &he) || he.Kind != HookErrCommandOutputInvalid {
		t.Fatalf("err = %v, want CommandOutputInvalid", err)
	}
}

// ── HookDecision wire format pinned ─────────────────────────────────────────
func TestHookDecisionWireFormat(t *testing.T) {
	cases := []struct {
		d    HookDecision
		want string
	}{
		{Continue(), `{"decision":"continue"}`},
		{Block("r"), `{"decision":"block","reason":"r"}`},
		{Inject("c"), `{"decision":"inject","context":"c"}`},
		{Deny("d"), `{"decision":"deny","reason":"d"}`},
		{Mutate(json.RawMessage(`{"k":1}`)), `{"decision":"mutate","data":{"k":1}}`},
	}
	for _, c := range cases {
		b, err := json.Marshal(c.d)
		if err != nil {
			t.Fatal(err)
		}
		if string(b) != c.want {
			t.Fatalf("marshal = %s, want %s", b, c.want)
		}
		// Round-trip.
		var back HookDecision
		if err := json.Unmarshal(b, &back); err != nil {
			t.Fatal(err)
		}
		reb, _ := json.Marshal(back)
		if string(reb) != c.want {
			t.Fatalf("round-trip = %s, want %s", reb, c.want)
		}
	}
}

// ── Deferred-event fire methods work in isolation ───────────────────────────
func TestDeferredOnPlanCreatedMutates(t *testing.T) {
	chain := NewStandardHookChain()
	if err := chain.Register(NewFunctionHook("plan", []HookEvent{HookEventOnPlanCreated},
		func(_ context.Context, hctx *HookContext) (HookDecision, error) {
			hctx.Plan.Tasks = append(hctx.Plan.Tasks, "extra")
			return Continue(), nil
		})); err != nil {
		t.Fatal(err)
	}
	plan := PlanArtifact{Tasks: []string{"a"}}
	hctx := &HookContext{Event: HookEventOnPlanCreated, SessionID: sid(), Plan: &plan}
	if _, err := chain.Fire(context.Background(), hctx); err != nil {
		t.Fatal(err)
	}
	if len(plan.Tasks) != 2 || plan.Tasks[1] != "extra" {
		t.Fatalf("tasks = %v", plan.Tasks)
	}
}

func TestDeferredSubagentSpawnDeny(t *testing.T) {
	chain := NewStandardHookChain()
	if err := chain.Register(fnHook("ss", []HookEvent{HookEventOnSubagentSpawn}, Deny("no spawn"))); err != nil {
		t.Fatal(err)
	}
	child := "child task"
	hctx := &HookContext{Event: HookEventOnSubagentSpawn, SessionID: sid(), TaskInstruction: &child, Strategy: "react"}
	outcome, err := chain.Fire(context.Background(), hctx)
	if err != nil {
		t.Fatal(err)
	}
	if outcome.Kind != FireDeny || outcome.Reason != "no spawn" {
		t.Fatalf("outcome = %+v", outcome)
	}
}

func TestPreCompactMutatesHints(t *testing.T) {
	chain := NewStandardHookChain()
	if err := chain.Register(NewFunctionHook("pc", []HookEvent{HookEventPreCompact},
		func(_ context.Context, hctx *HookContext) (HookDecision, error) {
			hctx.PreserveHints.KeepRecentFileList = false
			return Continue(), nil
		})); err != nil {
		t.Fatal(err)
	}
	hints := CompactionPreserveHints{KeepRecentFileList: true}
	hctx := &HookContext{Event: HookEventPreCompact, SessionID: sid(), PreserveHints: &hints}
	if _, err := chain.Fire(context.Background(), hctx); err != nil {
		t.Fatal(err)
	}
	if hints.KeepRecentFileList {
		t.Fatal("KeepRecentFileList should be false")
	}
}

// ── OnResume mutate-via-decision (deferred event) ───────────────────────────
func TestOnResumeMutateDecision(t *testing.T) {
	chain := NewStandardHookChain()
	if err := chain.Register(fnHook("r", []HookEvent{HookEventOnResume}, Mutate(json.RawMessage(`"resumed task"`)))); err != nil {
		t.Fatal(err)
	}
	instr := "old"
	ps := PausedState{}
	hctx := &HookContext{Event: HookEventOnResume, SessionID: sid(), TaskInstruction: &instr, PausedState: &ps}
	if _, err := chain.Fire(context.Background(), hctx); err != nil {
		t.Fatal(err)
	}
	if instr != "resumed task" {
		t.Fatalf("instr = %q", instr)
	}
}

// ── Harness wiring: Stop block → inject → continue, then terminate ──────────
func TestHarnessStopHookBlocksThenContinues(t *testing.T) {
	a := NewMockAgent("t")
	a.Push(NewFinalResponse("first attempt", turnUsage()))
	a.Push(NewFinalResponse("second attempt", turnUsage()))

	var fires int32
	chain := NewStandardHookChain()
	if err := chain.Register(NewFunctionHook("verify", []HookEvent{HookEventStop},
		func(_ context.Context, hctx *HookContext) (HookDecision, error) {
			// First Stop blocks (reason injected, loop continues); second
			// Stop allows termination.
			if atomic.AddInt32(&fires, 1) == 1 {
				if hctx.LastOutput.Text != "first attempt" {
					t.Errorf("first stop saw %q", hctx.LastOutput.Text)
				}
				return Block("not done yet"), nil
			}
			return Continue(), nil
		})); err != nil {
		t.Fatal(err)
	}

	cfg := standardCfg(a)
	cfg.Hooks = chain
	h := NewStandardHarness(cfg)
	r := h.Run(context.Background(), NewHarnessRunOptions(reactTask(5)))
	if r.Kind != RunSuccess {
		t.Fatalf("kind=%q reason=%+v", r.Kind, r.Reason)
	}
	if r.Output != "second attempt" {
		t.Fatalf("output = %q, want second attempt", r.Output)
	}
	if r.Turns != 2 {
		t.Fatalf("turns = %d, want 2 (block forced one extra turn)", r.Turns)
	}
	if atomic.LoadInt32(&fires) != 2 {
		t.Fatalf("stop fired %d times, want 2", fires)
	}
}

// ── Harness wiring: all-continue terminates on the first attempt ────────────
func TestHarnessStopHookAllContinueTerminates(t *testing.T) {
	a := NewMockAgent("t")
	a.Push(NewFinalResponse("done", turnUsage()))
	chain := NewStandardHookChain()
	if err := chain.Register(fnHook("ok", []HookEvent{HookEventStop}, Continue())); err != nil {
		t.Fatal(err)
	}
	cfg := standardCfg(a)
	cfg.Hooks = chain
	h := NewStandardHarness(cfg)
	r := h.Run(context.Background(), NewHarnessRunOptions(reactTask(5)))
	if r.Kind != RunSuccess || r.Output != "done" || r.Turns != 1 {
		t.Fatalf("got %+v", r)
	}
}

// ── Harness wiring: MaxStopBlocks caps consecutive blocks ───────────────────
func TestHarnessStopHookMaxBlocksCap(t *testing.T) {
	a := NewMockAgent("t")
	// Queue plenty of final responses; the cap, not the agent, ends the run.
	for i := 0; i < 10; i++ {
		a.Push(NewFinalResponse("attempt", turnUsage()))
	}
	var fires int32
	chain := NewStandardHookChain()
	if err := chain.Register(NewFunctionHook("always-block", []HookEvent{HookEventStop},
		func(_ context.Context, _ *HookContext) (HookDecision, error) {
			atomic.AddInt32(&fires, 1)
			return Block("never satisfied"), nil
		})); err != nil {
		t.Fatal(err)
	}
	cfg := standardCfg(a)
	cfg.Hooks = chain
	cfg.MaxStopBlocks = 3
	h := NewStandardHarness(cfg)
	r := h.Run(context.Background(), NewHarnessRunOptions(reactTask(20)))
	if r.Kind != RunSuccess {
		t.Fatalf("kind=%q reason=%+v", r.Kind, r.Reason)
	}
	// 3 blocks honored (turns 1-3 forced continues), the 4th Stop fire hits the
	// cap and terminates: 4 fires total, 4 turns.
	if atomic.LoadInt32(&fires) != 4 {
		t.Fatalf("stop fired %d times, want 4 (3 honored blocks + 1 cap hit)", fires)
	}
	if r.Turns != 4 {
		t.Fatalf("turns = %d, want 4", r.Turns)
	}
}

// ── Harness wiring: nil hook chain terminates normally ──────────────────────
func TestHarnessNoHooksTerminatesNormally(t *testing.T) {
	a := NewMockAgent("t")
	a.Push(NewFinalResponse("done", turnUsage()))
	h := NewStandardHarness(standardCfg(a)) // no Hooks configured
	r := h.Run(context.Background(), NewHarnessRunOptions(reactTask(5)))
	if r.Kind != RunSuccess || r.Output != "done" || r.Turns != 1 {
		t.Fatalf("got %+v", r)
	}
}

func asHookError(err error, target **HookError) bool {
	he, ok := err.(*HookError)
	if ok {
		*target = he
	}
	return ok
}
