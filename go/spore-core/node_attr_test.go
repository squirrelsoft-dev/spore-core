package sporecore

import (
	"context"
	"encoding/json"
	"testing"
)

// reactTaskWithAgent builds a bare ReAct task whose leaf carries a non-empty
// worker agent handle (so the leaf's NodeAttr stamps a non-empty agent_id).
func reactTaskWithAgent(max uint32, handle string) Task {
	s := ReActStrategy(max)
	s.ReActCfg.Agent = AgentRef(handle)
	return NewTask("do something", SessionID("s1"), s)
}

// TestNodeAttrOmittedWhenNil confirms an un-attributed event serializes WITHOUT
// a `node` key — byte-identical to pre-§E1 (mirrors Rust serde's
// skip_serializing_if = "Option::is_none").
func TestNodeAttrOmittedWhenNil(t *testing.T) {
	ev := HarnessStreamEvent{Kind: HarnessStreamTurnStart, Turn: 1}
	b, err := json.Marshal(ev)
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]json.RawMessage
	if err := json.Unmarshal(b, &m); err != nil {
		t.Fatal(err)
	}
	if _, ok := m["node"]; ok {
		t.Fatalf("un-attributed event must omit `node`; got %s", b)
	}
	// Spot-check a second variant.
	ev2 := HarnessStreamEvent{Kind: HarnessStreamFinalResponse, Content: "done"}
	b2, _ := json.Marshal(ev2)
	var m2 map[string]json.RawMessage
	json.Unmarshal(b2, &m2)
	if _, ok := m2["node"]; ok {
		t.Fatalf("un-attributed final_response must omit `node`; got %s", b2)
	}
}

// TestNodeAttrSerialization confirms a present NodeAttr serializes the
// kind/depth always, agent_id only when non-empty, and path only when
// non-empty (mirrors Rust serde skip_serializing_if on agent_id/path).
func TestNodeAttrSerialization(t *testing.T) {
	ev := HarnessStreamEvent{
		Kind: HarnessStreamTurnStart,
		Turn: 1,
		Node: &NodeAttr{Kind: "react", AgentID: "worker", Depth: 0, Path: []string{"react"}},
	}
	b, err := json.Marshal(ev)
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]json.RawMessage
	if err := json.Unmarshal(b, &m); err != nil {
		t.Fatal(err)
	}
	if string(m["node"]) == "" {
		t.Fatalf("node key missing; got %s", b)
	}
	var na NodeAttr
	if err := json.Unmarshal(m["node"], &na); err != nil {
		t.Fatal(err)
	}
	if na.Kind != "react" || na.AgentID != "worker" || na.Depth != 0 || len(na.Path) != 1 || na.Path[0] != "react" {
		t.Fatalf("unexpected NodeAttr: %+v", na)
	}
	// Empty agent_id ⇒ omitted; empty path ⇒ omitted.
	ev2 := HarnessStreamEvent{
		Kind: HarnessStreamTurnEnd,
		Turn: 2,
		Node: &NodeAttr{Kind: "plan_execute", Depth: 1, Path: nil},
	}
	b2, _ := json.Marshal(ev2)
	var m2 map[string]json.RawMessage
	json.Unmarshal(b2, &m2)
	var na2 NodeAttr
	json.Unmarshal(m2["node"], &na2)
	if na2.AgentID != "" {
		t.Fatalf("empty agent_id must be omitted; got %s", b2)
	}
	if len(na2.Path) != 0 {
		t.Fatalf("empty path must be omitted; got %s", b2)
	}
	if na2.Kind != "plan_execute" || na2.Depth != 1 {
		t.Fatalf("kind/depth must remain: %+v", na2)
	}
}

// TestLeafStampsReactNodeAttr confirms a bare ReAct leaf stamps every emitted
// event with kind="react" and the resolved worker agent handle (mirrors Rust
// react.rs:2125 wrap_node_sink(cx.stream, "react", agent_id)).
func TestLeafStampsReactNodeAttr(t *testing.T) {
	a := NewMockAgent("t")
	a.Push(NewFinalResponse("done", turnUsage()))
	cfg := standardCfg(a).WithRegistryAgent("worker", a)
	h := NewStandardHarness(cfg)
	var nodes []*NodeAttr
	opts := NewHarnessRunOptions(reactTaskWithAgent(5, "worker"))
	opts.OnStream = func(ev HarnessStreamEvent) {
		if ev.Node != nil {
			nodes = append(nodes, ev.Node)
		}
	}
	h.Run(context.Background(), opts)
	if len(nodes) == 0 {
		t.Fatalf("no NodeAttr stamped on leaf events")
	}
	for _, n := range nodes {
		if n.Kind != "react" {
			t.Fatalf("leaf node kind = %q, want \"react\"", n.Kind)
		}
		if n.AgentID != "worker" {
			t.Fatalf("leaf node agent_id = %q, want \"worker\"", n.AgentID)
		}
		if n.Depth != 0 {
			t.Fatalf("leaf node depth = %d, want 0", n.Depth)
		}
		if len(n.Path) != 1 || n.Path[0] != "react" {
			t.Fatalf("leaf node path = %v, want [\"react\"]", n.Path)
		}
	}
}

// TestWrapNodeSinkPathDepth confirms wrapNodeSink stamps an unstamped event
// with kind/agent_id/seeds path (depth 0), and a combinator wrapper PREPENDS
// its kind to path and bumps depth (mirrors Rust wrap_node_sink).
func TestWrapNodeSinkPathDepth(t *testing.T) {
	var captured []HarnessStreamEvent
	parent := func(ev HarnessStreamEvent) { captured = append(captured, ev) }
	// Combinator wrapper (outermost).
	planWrap := wrapNodeSink(parent, "plan_execute", "")
	// Leaf wrapper (innermost) — installed inside the combinator.
	leafWrap := wrapNodeSink(planWrap, "react", "worker")

	leafWrap(HarnessStreamEvent{Kind: HarnessStreamTurnStart, Turn: 1})
	if len(captured) != 1 {
		t.Fatalf("captured %d events, want 1", len(captured))
	}
	n := captured[0].Node
	if n == nil {
		t.Fatal("node not stamped")
	}
	// Leaf set kind/agent_id/depth=0/path=[react]; combinator prepended
	// plan_execute and bumped depth to 1.
	if n.Kind != "react" {
		t.Fatalf("kind = %q, want \"react\" (innermost leaf wins)", n.Kind)
	}
	if n.AgentID != "worker" {
		t.Fatalf("agent_id = %q, want \"worker\"", n.AgentID)
	}
	if n.Depth != 1 {
		t.Fatalf("depth = %d, want 1 (one combinator hop)", n.Depth)
	}
	if len(n.Path) != 2 || n.Path[0] != "plan_execute" || n.Path[1] != "react" {
		t.Fatalf("path = %v, want [plan_execute, react]", n.Path)
	}
}

// TestWrapNodeSinkNilParentReturnsNil confirms a nil parent ⇒ no wrapper (an
// un-attributed run stays byte-identical, no allocation).
func TestWrapNodeSinkNilParentReturnsNil(t *testing.T) {
	if g := wrapNodeSink(nil, "react", "worker"); g != nil {
		t.Fatalf("wrapNodeSink(nil, ...) = %v, want nil", g)
	}
}
