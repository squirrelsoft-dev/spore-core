// Tests for the #142 project-scoped durable storage: ProjectID derivation (incl.
// the canonicalize-first FS constructors, the /a/b vs /a_b distinct-id case, and
// fixture replay), the active-run lifecycle (new/resume/complete + every error
// path), and the cross-window + cross-process durability survival (incl. fixture
// replay). All FS-touching tests use t.TempDir() and never write to ~/.spore.

package storage

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"testing"

	sporecore "github.com/squirrelsoft-dev/spore-core/go/spore-core"
)

// ── Derivation: pure, deterministic, same algorithm as WorkspaceID ───────────

func TestProjectIDIsPureAndMatchesWorkspaceID(t *testing.T) {
	const path = "/Users/sbeardsley/dev/spore-core"
	a := ProjectIDFromCanonicalPath(path)
	b := ProjectIDFromCanonicalPath(path)
	if a != b {
		t.Fatalf("non-deterministic: %q != %q", a, b)
	}
	// The pure derivation delegates to the SAME algorithm as WorkspaceID — the two
	// must be byte-identical for the same input (decision 4: reuse, don't duplicate).
	if string(a) != WorkspaceIDFromCanonicalPath(path).String() {
		t.Fatalf("ProjectID %q diverged from WorkspaceID %q for same path",
			a, WorkspaceIDFromCanonicalPath(path))
	}
}

func TestProjectIDForm(t *testing.T) {
	p := ProjectIDFromCanonicalPath("/Users/sbeardsley/dev/spore-core")
	const prefix = "spore-core-"
	s := p.String()
	if len(s) != len(prefix)+8 || s[:len(prefix)] != prefix {
		t.Fatalf("unexpected form: %q", s)
	}
}

func TestProjectIDRootCollapsesToRootBasename(t *testing.T) {
	p := ProjectIDFromCanonicalPath("/")
	if p.String()[:5] != "root-" {
		t.Fatalf("root basename = %q, want root-…", p)
	}
}

func TestProjectIDSanitizesSpecialChars(t *testing.T) {
	p := ProjectIDFromCanonicalPath("/Users/me/My Project (v2)!")
	const prefix = "my-project-v2-"
	if p.String()[:len(prefix)] != prefix {
		t.Fatalf("sanitize = %q, want %s…", p, prefix)
	}
}

func TestProjectIDIgnoresTrailingSlash(t *testing.T) {
	a := ProjectIDFromCanonicalPath("/Users/sbeardsley/dev/spore-core")
	b := ProjectIDFromCanonicalPath("/Users/sbeardsley/dev/spore-core/")
	if a != b {
		t.Fatalf("trailing slash changed id: %q != %q", a, b)
	}
}

// ── /a/b vs /a_b: distinct ids (the documented collision policy) ─────────────

// The 8-hex SHA-256 suffix of the FULL canonical path resolves the slug
// collision a naive slashes->underscores scheme would suffer: /a/b and /a_b have
// different canonical strings, hence different hashes, hence distinct ids.
func TestProjectIDSlashVsUnderscoreAreDistinct(t *testing.T) {
	ab := ProjectIDFromCanonicalPath("/a/b")
	a_b := ProjectIDFromCanonicalPath("/a_b")
	if ab == a_b {
		t.Fatalf("/a/b and /a_b collided: both %q", ab)
	}
	// The basenames differ too (b vs a-b), so the slug prefixes differ.
	if ab.String()[:2] != "b-" {
		t.Fatalf("/a/b basename = %q, want b-…", ab)
	}
	if a_b.String()[:4] != "a-b-" {
		t.Fatalf("/a_b basename = %q, want a-b-…", a_b)
	}
}

// ── FS-touching constructors: canonicalize FIRST ────────────────────────────

// ProjectIDFromPath resolves a RELATIVE path to its absolute canonical form
// before deriving — so the id is the same as deriving from the absolute path.
func TestProjectIDFromPathCanonicalizesRelative(t *testing.T) {
	dir := t.TempDir()
	sub := filepath.Join(dir, "repo")
	if err := os.MkdirAll(sub, 0o755); err != nil {
		t.Fatal(err)
	}
	// Run from the tempdir so "repo" is a valid relative path.
	wd, _ := os.Getwd()
	defer func() { _ = os.Chdir(wd) }()
	if err := os.Chdir(dir); err != nil {
		t.Fatal(err)
	}

	fromRel, err := ProjectIDFromPath("repo")
	if err != nil {
		t.Fatalf("from relative: %v", err)
	}
	fromAbs, err := ProjectIDFromPath(sub)
	if err != nil {
		t.Fatalf("from absolute: %v", err)
	}
	if fromRel != fromAbs {
		t.Fatalf("relative %q != absolute %q — canonicalization did not normalize",
			fromRel, fromAbs)
	}
}

// ProjectIDFromPath follows symlinks before deriving — a symlink and its target
// canonicalize to the same path, hence the same id.
func TestProjectIDFromPathFollowsSymlinks(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("symlink creation is privilege-gated on Windows")
	}
	dir := t.TempDir()
	target := filepath.Join(dir, "real")
	if err := os.MkdirAll(target, 0o755); err != nil {
		t.Fatal(err)
	}
	link := filepath.Join(dir, "link")
	if err := os.Symlink(target, link); err != nil {
		t.Skipf("symlink unsupported here: %v", err)
	}
	viaLink, err := ProjectIDFromPath(link)
	if err != nil {
		t.Fatalf("via link: %v", err)
	}
	viaTarget, err := ProjectIDFromPath(target)
	if err != nil {
		t.Fatalf("via target: %v", err)
	}
	if viaLink != viaTarget {
		t.Fatalf("symlink %q != target %q — symlink not resolved", viaLink, viaTarget)
	}
}

// macOS is case-insensitive: a path and its differently-cased twin canonicalize
// to the same on-disk path, hence the same id. Gated on GOOS because the
// behaviour is filesystem-dependent.
func TestProjectIDFromPathMacOSCaseInsensitive(t *testing.T) {
	if runtime.GOOS != "darwin" {
		t.Skip("case-insensitivity is a macOS (HFS+/APFS default) behaviour")
	}
	dir := t.TempDir()
	sub := filepath.Join(dir, "MixedCase")
	if err := os.MkdirAll(sub, 0o755); err != nil {
		t.Fatal(err)
	}
	upper, err := ProjectIDFromPath(sub)
	if err != nil {
		t.Fatalf("upper: %v", err)
	}
	lower, err := ProjectIDFromPath(filepath.Join(dir, "mixedcase"))
	if err != nil {
		// Some APFS volumes are case-sensitive; skip rather than fail.
		t.Skipf("volume appears case-sensitive: %v", err)
	}
	if upper != lower {
		t.Skipf("volume appears case-sensitive: %q != %q", upper, lower)
	}
}

// The canonicalize-failure path: a non-existent path returns a wrapped error and
// the empty id.
func TestProjectIDFromPathNonexistentErrors(t *testing.T) {
	dir := t.TempDir()
	_, err := ProjectIDFromPath(filepath.Join(dir, "does-not-exist"))
	if err == nil {
		t.Fatal("expected an error for a non-existent path")
	}
}

func TestProjectIDFromCwd(t *testing.T) {
	dir := t.TempDir()
	wd, _ := os.Getwd()
	defer func() { _ = os.Chdir(wd) }()
	if err := os.Chdir(dir); err != nil {
		t.Fatal(err)
	}
	fromCwd, err := ProjectIDFromCwd()
	if err != nil {
		t.Fatalf("from cwd: %v", err)
	}
	fromPath, err := ProjectIDFromPath(dir)
	if err != nil {
		t.Fatalf("from path: %v", err)
	}
	if fromCwd != fromPath {
		t.Fatalf("cwd %q != path %q", fromCwd, fromPath)
	}
}

// ── Namespace seam ───────────────────────────────────────────────────────────

func TestProjectIDNamespaceIsTheIDString(t *testing.T) {
	p := ProjectIDFromCanonicalPath("/work/audit-repo")
	if string(p.Namespace()) != p.String() {
		t.Fatalf("namespace %q != id %q", p.Namespace(), p)
	}
}

// ── Fixture replay: project_id_derivation.json ───────────────────────────────

func TestProjectIDDerivationFixtureReplay(t *testing.T) {
	b, err := os.ReadFile(fixturePath("project_id_derivation.json"))
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	var cases []struct {
		CanonicalPath     string `json:"canonical_path"`
		ExpectedProjectID string `json:"expected_project_id"`
	}
	must(t, json.Unmarshal(b, &cases))
	if len(cases) < 4 {
		t.Fatalf("fixture should carry several rows, got %d", len(cases))
	}
	for _, c := range cases {
		got := ProjectIDFromCanonicalPath(c.CanonicalPath)
		if got.String() != c.ExpectedProjectID {
			t.Fatalf("path %q: got %q want %q", c.CanonicalPath, got, c.ExpectedProjectID)
		}
	}
}

// ── Active-run lifecycle: new / resume / complete + error paths ──────────────

func TestActiveRunStartsNewThenResumesSameTag(t *testing.T) {
	run := NewInMemoryStorageProvider()
	project := ProjectIDFromCanonicalPath("/work/audit-repo")
	ts := Timestamp("2026-06-12T00:00:00Z")

	// First start under a tag => StartedNew, slot written Active.
	dec, err := StartOrResumeActiveRun(context.Background(), run, project, "job-1", ts)
	must(t, err)
	if dec != ActiveRunStartedNew {
		t.Fatalf("first start = %q, want started_new", dec)
	}
	// A second start under the SAME tag while Active => Resumed (slot intact).
	dec, err = StartOrResumeActiveRun(context.Background(), run, project, "job-1", Timestamp("LATER"))
	must(t, err)
	if dec != ActiveRunResumed {
		t.Fatalf("same-tag start = %q, want resumed", dec)
	}
	slot, err := LoadActiveRun(context.Background(), run, project)
	must(t, err)
	if slot == nil || slot.StartedAt != ts {
		t.Fatalf("resume must leave the slot intact, got %+v", slot)
	}
}

func TestActiveRunDifferentTagStartsFresh(t *testing.T) {
	run := NewInMemoryStorageProvider()
	project := ProjectIDFromCanonicalPath("/work/audit-repo")

	_, err := StartOrResumeActiveRun(context.Background(), run, project, "job-1", Timestamp("T1"))
	must(t, err)
	// A DIFFERENT tag => a fresh Active slot stamped with the new timestamp.
	dec, err := StartOrResumeActiveRun(context.Background(), run, project, "job-2", Timestamp("T2"))
	must(t, err)
	if dec != ActiveRunStartedNew {
		t.Fatalf("different-tag start = %q, want started_new", dec)
	}
	slot, err := LoadActiveRun(context.Background(), run, project)
	must(t, err)
	if slot == nil || slot.RunTag != "job-2" || slot.StartedAt != Timestamp("T2") {
		t.Fatalf("different tag must mint a fresh slot, got %+v", slot)
	}
}

func TestActiveRunCompleteThenSameTagStartsFresh(t *testing.T) {
	run := NewInMemoryStorageProvider()
	project := ProjectIDFromCanonicalPath("/work/audit-repo")

	_, err := StartOrResumeActiveRun(context.Background(), run, project, "job-1", Timestamp("T1"))
	must(t, err)
	must(t, CompleteActiveRun(context.Background(), run, project))

	slot, err := LoadActiveRun(context.Background(), run, project)
	must(t, err)
	if slot == nil || slot.Status != ActiveRunStatusCompleted {
		t.Fatalf("complete must flip status, got %+v", slot)
	}
	// After complete, even the SAME tag starts fresh (not a resume).
	dec, err := StartOrResumeActiveRun(context.Background(), run, project, "job-1", Timestamp("T2"))
	must(t, err)
	if dec != ActiveRunStartedNew {
		t.Fatalf("post-complete same-tag start = %q, want started_new", dec)
	}
	slot, err = LoadActiveRun(context.Background(), run, project)
	must(t, err)
	if slot.Status != ActiveRunStatusActive || slot.StartedAt != Timestamp("T2") {
		t.Fatalf("post-complete start must be Active+fresh, got %+v", slot)
	}
}

func TestLoadActiveRunAbsentIsNil(t *testing.T) {
	run := NewInMemoryStorageProvider()
	project := ProjectIDFromCanonicalPath("/work/empty")
	slot, err := LoadActiveRun(context.Background(), run, project)
	must(t, err)
	if slot != nil {
		t.Fatalf("absent slot must be nil, got %+v", slot)
	}
}

// A malformed slot is treated as "no live run" — the next start mints a fresh one
// rather than erroring.
func TestLoadActiveRunMalformedIsNil(t *testing.T) {
	run := NewInMemoryStorageProvider()
	project := ProjectIDFromCanonicalPath("/work/audit-repo")
	must(t, run.Put(context.Background(), project.Namespace(), ActiveRunKey, json.RawMessage(`{not json`)))
	slot, err := LoadActiveRun(context.Background(), run, project)
	must(t, err)
	if slot != nil {
		t.Fatalf("malformed slot must read as nil, got %+v", slot)
	}
	// The next start mints fresh over the garbage.
	dec, err := StartOrResumeActiveRun(context.Background(), run, project, "job", Timestamp("T"))
	must(t, err)
	if dec != ActiveRunStartedNew {
		t.Fatalf("start over malformed = %q, want started_new", dec)
	}
}

func TestCompleteActiveRunNoSlotIsNoOp(t *testing.T) {
	run := NewInMemoryStorageProvider()
	project := ProjectIDFromCanonicalPath("/work/empty")
	if err := CompleteActiveRun(context.Background(), run, project); err != nil {
		t.Fatalf("complete with no slot must be a no-op, got %v", err)
	}
	slot, err := LoadActiveRun(context.Background(), run, project)
	must(t, err)
	if slot != nil {
		t.Fatalf("complete must not create a slot, got %+v", slot)
	}
}

func TestActiveRunSurvivesStartError(t *testing.T) {
	// A store whose Put fails surfaces the error from StartOrResumeActiveRun.
	run := failingPutStore{NewInMemoryStorageProvider()}
	project := ProjectIDFromCanonicalPath("/work/audit-repo")
	_, err := StartOrResumeActiveRun(context.Background(), run, project, "job", Timestamp("T"))
	if err == nil {
		t.Fatal("expected the Put error to propagate")
	}
}

// failingPutStore wraps a RunStore but errors on Put — to exercise the
// store-error path of the active-run helpers.
type failingPutStore struct{ RunStore }

func (failingPutStore) Put(context.Context, SessionID, string, json.RawMessage) error {
	return os.ErrPermission
}

// ── Cross-window + cross-process durability: fixture replay ──────────────────

// AC5 (#142): the durable task_list keyed by project_id survives a Ralph window
// reset (fresh session each window, same project) AND a process restart (a fresh
// FileSystemStorageProvider over the same root re-reads it). Pinned by
// fixtures/storage/project_durable_survival.json.
func TestProjectDurableSurvivalFixtureReplay(t *testing.T) {
	raw, err := os.ReadFile(fixturePath("project_durable_survival.json"))
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	var fx struct {
		ProjectCanonicalPath string `json:"project_canonical_path"`
		ExpectedProjectID    string `json:"expected_project_id"`
		RunKey               string `json:"run_key"`
		Window1              struct {
			SessionID string          `json:"session_id"`
			TaskList  json.RawMessage `json:"task_list"`
		} `json:"window_1"`
		Window2 struct {
			SessionID        string          `json:"session_id"`
			ExpectedTaskList json.RawMessage `json:"expected_task_list"`
		} `json:"window_2"`
		CrossProcess struct {
			ExpectedTaskList json.RawMessage `json:"expected_task_list"`
		} `json:"cross_process"`
	}
	must(t, json.Unmarshal(raw, &fx))

	// The project id the fixture pins.
	project := ProjectIDFromCanonicalPath(fx.ProjectCanonicalPath)
	if project.String() != fx.ExpectedProjectID {
		t.Fatalf("project id = %q, want %q", project, fx.ExpectedProjectID)
	}
	ns := project.Namespace()

	// A durable on-disk root shared across "windows" and "processes".
	root := t.TempDir()

	// Window 1: a real provider over the root writes the list under the project
	// namespace (NOT the window's session id).
	provider1 := NewFileSystemStorageProvider(root)
	must(t, provider1.Put(context.Background(), ns, fx.RunKey, fx.Window1.TaskList))

	// Window 2: a DIFFERENT session id (mirrors NewSessionID() per Ralph window),
	// SAME project namespace, SAME provider — must read window 1's list verbatim.
	got2, found, err := provider1.Get(context.Background(), ns, fx.RunKey)
	must(t, err)
	if !found {
		t.Fatal("window 2: project-keyed list not found")
	}
	if !jsonEqual(t, got2, fx.Window2.ExpectedTaskList) {
		t.Fatalf("window 2 read %s, want %s", got2, fx.Window2.ExpectedTaskList)
	}
	// Sanity: window 2's OWN session id sees nothing (ephemeral keys are empty).
	if _, foundSession, _ := provider1.Get(context.Background(), SessionID(fx.Window2.SessionID), fx.RunKey); foundSession {
		t.Fatal("window 2's session id should NOT see the project-keyed list")
	}

	// Cross-process: a BRAND-NEW provider over the SAME on-disk root reads the
	// same bytes under the same project namespace (crash-resume durability).
	provider2 := NewFileSystemStorageProvider(root)
	got3, found, err := provider2.Get(context.Background(), ns, fx.RunKey)
	must(t, err)
	if !found {
		t.Fatal("cross-process: project-keyed list not found on a fresh provider")
	}
	if !jsonEqual(t, got3, fx.CrossProcess.ExpectedTaskList) {
		t.Fatalf("cross-process read %s, want %s", got3, fx.CrossProcess.ExpectedTaskList)
	}
}

func jsonEqual(t *testing.T, a, b json.RawMessage) bool {
	t.Helper()
	var av, bv any
	must(t, json.Unmarshal(a, &av))
	must(t, json.Unmarshal(b, &bv))
	return reflect.DeepEqual(av, bv)
}

// compile-time: the namespace projection yields a sporecore.SessionID.
var _ sporecore.SessionID = ProjectID("x").Namespace()

// The Ralph checkpoint key literals defined in the root sporecore package (which
// cannot import storage) MUST agree with the storage-package constants, because
// the namespace-reuse seam keys the SAME (project namespace, key) RunStore axis
// from both sides. This pins that agreement (forewarning #7-adjacent).
func TestRalphKeyLiteralsAgreeAcrossPackages(t *testing.T) {
	if RalphProgressKey != sporecore.RalphProgressKey {
		t.Fatalf("RalphProgressKey divergence: storage=%q sporecore=%q",
			RalphProgressKey, sporecore.RalphProgressKey)
	}
	if RalphFeatureListKey != sporecore.RalphFeatureListKey {
		t.Fatalf("RalphFeatureListKey divergence: storage=%q sporecore=%q",
			RalphFeatureListKey, sporecore.RalphFeatureListKey)
	}
}
