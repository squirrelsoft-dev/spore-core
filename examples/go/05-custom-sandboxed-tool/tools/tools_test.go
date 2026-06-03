package tools

import (
	"context"
	"encoding/json"
	"testing"

	sporecore "github.com/squirrelsoft-dev/spore-core/go/spore-core"
)

const testSession = sporecore.SessionID("sess-test")

// newCtx returns a ToolContext backed by a fresh in-memory run store — the same
// store the builder defaults to when .Tool() tools are present.
func newCtx() (*sporecore.ToolContext, sporecore.ToolRunStore) {
	store := sporecore.NewInMemoryToolRunStore()
	return sporecore.NewToolContext(testSession, store, nil), store
}

func call(name string, input string) sporecore.ToolCall {
	return sporecore.ToolCall{ID: "c1", Name: name, Input: json.RawMessage(input)}
}

// remember stores the value under the fact: prefix.
func TestRememberStoresUnderFactPrefix(t *testing.T) {
	ctx, store := newCtx()
	out := RememberTool{}.Execute(context.Background(), call("remember", `{"key":"habitat","value":"coastal waters"}`), nil, ctx)

	if out.Kind != sporecore.ToolOutputSuccess {
		t.Fatalf("Kind = %v, want success (message=%q)", out.Kind, out.Message)
	}
	if out.Content != "remembered habitat" {
		t.Errorf("Content = %q, want %q", out.Content, "remembered habitat")
	}

	// The raw store must carry the value under the namespaced key, JSON-encoded.
	raw, found, err := store.Get(context.Background(), testSession, FactPrefix+"habitat")
	if err != nil || !found {
		t.Fatalf("store.Get(fact:habitat) found=%v err=%v, want found", found, err)
	}
	var got string
	if err := json.Unmarshal(raw, &got); err != nil {
		t.Fatalf("stored value not a JSON string: %v", err)
	}
	if got != "coastal waters" {
		t.Errorf("stored value = %q, want %q", got, "coastal waters")
	}
	// And it must NOT be stored under the bare key.
	if _, found, _ := store.Get(context.Background(), testSession, "habitat"); found {
		t.Error("value leaked into the un-prefixed key 'habitat'")
	}
}

// recall returns the value a prior remember stored.
func TestRecallReturnsStoredValue(t *testing.T) {
	ctx, _ := newCtx()
	if out := (RememberTool{}).Execute(context.Background(), call("remember", `{"key":"diet","value":"crabs and shrimp"}`), nil, ctx); out.Kind != sporecore.ToolOutputSuccess {
		t.Fatalf("setup remember failed: %q", out.Message)
	}

	out := RecallTool{}.Execute(context.Background(), call("recall", `{"key":"diet"}`), nil, ctx)
	if out.Kind != sporecore.ToolOutputSuccess {
		t.Fatalf("Kind = %v, want success (message=%q)", out.Kind, out.Message)
	}
	if out.Content != "crabs and shrimp" {
		t.Errorf("Content = %q, want %q", out.Content, "crabs and shrimp")
	}
}

// recall on an unknown key is a recoverable error with the spec'd message.
func TestRecallMissingKeyIsRecoverableError(t *testing.T) {
	ctx, _ := newCtx()
	out := RecallTool{}.Execute(context.Background(), call("recall", `{"key":"nope"}`), nil, ctx)

	if out.Kind != sporecore.ToolOutputError {
		t.Fatalf("Kind = %v, want error", out.Kind)
	}
	if !out.Recoverable {
		t.Error("missing-key error must be recoverable")
	}
	if want := "no fact stored under 'nope'"; out.Message != want {
		t.Errorf("Message = %q, want %q", out.Message, want)
	}
}

// Bad arguments are recoverable errors, not panics or fatal errors.
func TestArgErrors(t *testing.T) {
	ctx, _ := newCtx()
	cases := []struct {
		name string
		tool sporecore.Tool
		args string
	}{
		{"remember missing key", RememberTool{}, `{"value":"v"}`},
		{"remember missing value", RememberTool{}, `{"key":"k"}`},
		{"remember non-string key", RememberTool{}, `{"key":123,"value":"v"}`},
		{"remember non-string value", RememberTool{}, `{"key":"k","value":123}`},
		{"remember empty input", RememberTool{}, ``},
		{"recall missing key", RecallTool{}, `{}`},
		{"recall non-string key", RecallTool{}, `{"key":123}`},
		{"recall empty input", RecallTool{}, ``},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			out := tc.tool.Execute(context.Background(), call(tc.tool.Name(), tc.args), nil, ctx)
			if out.Kind != sporecore.ToolOutputError {
				t.Fatalf("Kind = %v, want error", out.Kind)
			}
			if !out.Recoverable {
				t.Error("argument error must be recoverable")
			}
		})
	}
}

// Schema names must match Name(), and the read-only/idempotent annotations must
// reflect each tool's contract: remember mutates (not read-only), recall reads.
func TestSchemaAnnotations(t *testing.T) {
	rem := RememberTool{}
	if rem.Schema().Name != rem.Name() {
		t.Errorf("remember schema name %q != Name() %q", rem.Schema().Name, rem.Name())
	}
	if rem.Schema().Annotations.ReadOnly {
		t.Error("remember must NOT be read-only — it mutates shared state")
	}

	rec := RecallTool{}
	if rec.Schema().Name != rec.Name() {
		t.Errorf("recall schema name %q != Name() %q", rec.Schema().Name, rec.Name())
	}
	if !rec.Schema().Annotations.ReadOnly {
		t.Error("recall must be read-only")
	}
	if !rec.Schema().Annotations.Idempotent {
		t.Error("recall must be idempotent")
	}

	// Schemas must be valid JSON objects with the required fields.
	for _, s := range []sporecore.RegistryToolSchema{rem.Schema(), rec.Schema()} {
		var probe map[string]any
		if err := json.Unmarshal(s.Parameters, &probe); err != nil {
			t.Errorf("%s parameters are not valid JSON: %v", s.Name, err)
		}
	}
}

// A store failure surfaces as a recoverable error (not a panic). Drive it with a
// run store whose Put always fails.
func TestRememberStoreFailureIsRecoverable(t *testing.T) {
	ctx := sporecore.NewToolContext(testSession, failingStore{}, nil)
	out := RememberTool{}.Execute(context.Background(), call("remember", `{"key":"k","value":"v"}`), nil, ctx)
	if out.Kind != sporecore.ToolOutputError || !out.Recoverable {
		t.Fatalf("got Kind=%v Recoverable=%v, want recoverable error", out.Kind, out.Recoverable)
	}
}

type failingStore struct{}

func (failingStore) Get(context.Context, sporecore.SessionID, string) (json.RawMessage, bool, error) {
	return nil, false, errBoom
}
func (failingStore) Put(context.Context, sporecore.SessionID, string, json.RawMessage) error {
	return errBoom
}

var errBoom = errString("boom")

type errString string

func (e errString) Error() string { return string(e) }
