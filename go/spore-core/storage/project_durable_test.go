package storage_test

import (
	"context"
	"encoding/json"
	"path/filepath"
	"testing"

	sporecore "github.com/squirrelsoft-dev/spore-core/go/spore-core"
	"github.com/squirrelsoft-dev/spore-core/go/spore-core/storage"
)

// Integration tests for #142 against the REAL FileSystemStorageProvider — these
// live in package storage_test (which may import storage) because the root
// sporecore package cannot import storage (a cycle). They prove the durable
// task_list / Ralph checkpoint, keyed by the stable project namespace, survive
// BOTH a Ralph window reset (fresh session id) AND a process restart (a fresh
// provider over the same on-disk root). Every test uses t.TempDir().

// AC2/AC3 (#142): the task_list keyed by project_id survives a window reset
// in-process AND a process restart. A first "window" (session A) writes the list
// under the project namespace; a second window (fresh session B) over a FRESH
// FileSystemStorageProvider on the same root reads it verbatim — never an empty
// list.
func TestProjectTaskListSurvivesWindowAndProcess(t *testing.T) {
	root := filepath.Join(t.TempDir(), "spore-142")
	project := storage.ProjectIDFromCanonicalPath("/work/audit-repo")
	ns := project.Namespace()
	list := json.RawMessage(`{"tasks":[{"id":1,"description":"discover","status":"completed","blockers":[]}],"next_id":2}`)

	// Window 1 (session A, "process 1"): write the durable list under the project
	// namespace, NOT session A.
	{
		p := storage.NewFileSystemStorageProvider(root)
		if err := p.Put(context.Background(), ns, sporecore.TaskListExtrasKey, list); err != nil {
			t.Fatalf("window 1 put: %v", err)
		}
		// Session A's own namespace must NOT carry the durable list.
		if _, found, _ := p.Get(context.Background(), sporecore.SessionID("sess-A"), sporecore.TaskListExtrasKey); found {
			t.Fatal("durable list must not be keyed by the window's session id")
		}
	}

	// Window 2 (fresh session B, "process 2"): a brand-new provider over the SAME
	// root reads window 1's list under the SAME project namespace.
	p2 := storage.NewFileSystemStorageProvider(root)
	got, found, err := p2.Get(context.Background(), ns, sporecore.TaskListExtrasKey)
	if err != nil || !found {
		t.Fatalf("cross-process read: found=%v err=%v", found, err)
	}
	var want, gotList map[string]any
	if err := json.Unmarshal(list, &want); err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(got, &gotList); err != nil {
		t.Fatal(err)
	}
	if gotList["next_id"] != want["next_id"] {
		t.Fatalf("cross-process list diverged: got %s", got)
	}
}

// AC4 (#142): the active-run lifecycle survives a process restart through the
// FileSystemStorageProvider. Start a run, restart the "process" (fresh provider),
// resume under the same tag, then complete — and a final fresh provider sees the
// archived slot.
func TestActiveRunLifecycleSurvivesProcessRestart(t *testing.T) {
	root := filepath.Join(t.TempDir(), "spore-142-active")
	project := storage.ProjectIDFromCanonicalPath("/work/audit-repo")
	ctx := context.Background()

	// Process 1: start a new run.
	{
		p := storage.NewFileSystemStorageProvider(root)
		dec, err := storage.StartOrResumeActiveRun(ctx, p, project, "job-1", storage.Timestamp("T1"))
		if err != nil {
			t.Fatal(err)
		}
		if dec != storage.ActiveRunStartedNew {
			t.Fatalf("process 1 start = %q, want started_new", dec)
		}
	}

	// Process 2: a fresh provider over the same root resumes the SAME tag.
	{
		p := storage.NewFileSystemStorageProvider(root)
		dec, err := storage.StartOrResumeActiveRun(ctx, p, project, "job-1", storage.Timestamp("T2"))
		if err != nil {
			t.Fatal(err)
		}
		if dec != storage.ActiveRunResumed {
			t.Fatalf("process 2 resume = %q, want resumed (the live slot survived the restart)", dec)
		}
		if err := storage.CompleteActiveRun(ctx, p, project); err != nil {
			t.Fatal(err)
		}
	}

	// Process 3: the archived (completed) slot is durable; a same-tag start is fresh.
	p := storage.NewFileSystemStorageProvider(root)
	slot, err := storage.LoadActiveRun(ctx, p, project)
	if err != nil || slot == nil {
		t.Fatalf("completed slot must persist: %+v err=%v", slot, err)
	}
	if slot.Status != storage.ActiveRunStatusCompleted {
		t.Fatalf("slot status = %q, want completed", slot.Status)
	}
	dec, err := storage.StartOrResumeActiveRun(ctx, p, project, "job-1", storage.Timestamp("T3"))
	if err != nil {
		t.Fatal(err)
	}
	if dec != storage.ActiveRunStartedNew {
		t.Fatalf("post-complete same-tag start = %q, want started_new", dec)
	}
}

// AC2 (#142): a Ralph harness drives completion off the DURABLE project store
// (not the .spore/ filesystem). A COMPLETE progress checkpoint pre-written to the
// FileSystemStorageProvider under the project namespace makes the Ralph loop
// succeed in a single window — proving the rewired readers consult the store, and
// that a checkpoint written by a prior "process" (fresh provider, same root)
// drives the current run.
func TestRalphCompletionDrivenByDurableStore(t *testing.T) {
	root := filepath.Join(t.TempDir(), "spore-142-ralph")
	project := storage.ProjectIDFromCanonicalPath("/work/audit-repo")
	ns := project.Namespace()

	// "Process 1": write a COMPLETE progress checkpoint, then forget the provider.
	{
		p := storage.NewFileSystemStorageProvider(root)
		if err := p.Put(context.Background(), ns, sporecore.RalphProgressKey, json.RawMessage(`{"complete":true,"remaining":[]}`)); err != nil {
			t.Fatal(err)
		}
	}

	// "Process 2": a fresh provider over the same root backs a Ralph harness. The
	// pre-existing complete checkpoint must make the loop succeed immediately.
	p := storage.NewFileSystemStorageProvider(root)
	agent := finalAgent("window done")
	cfg := sporecore.HarnessConfig{
		Agent:             agent,
		ToolRegistry:      sporecore.NewScriptedToolRegistry(),
		Sandbox:           sporecore.AllowAllSandbox{},
		ContextManager:    recordingCM{},
		TerminationPolicy: sporecore.AlwaysContinuePolicy{},
		RunStore:          p,
		ProjectNamespace:  ns,
		MaxResets:         3,
	}
	cfg = cfg.WithRegistryAgent("ralph-agent", agent)
	h := sporecore.NewStandardHarness(cfg)
	task := sporecore.NewTask("build the thing", sporecore.SessionID("ralph-1"),
		sporecore.RalphStrategy(sporecore.RalphConfig{
			Inner: sporecore.PtrStrategy(sporecore.ReActStrategy(^uint32(0))),
			Agent: sporecore.AgentRef("ralph-agent"),
		}))
	r := h.Run(context.Background(), sporecore.NewHarnessRunOptions(task))
	if r.Kind != sporecore.RunSuccess {
		t.Fatalf("Ralph must succeed off the durable complete checkpoint, got %+v", r)
	}
}
