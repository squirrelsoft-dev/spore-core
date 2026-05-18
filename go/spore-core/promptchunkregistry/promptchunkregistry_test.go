package promptchunkregistry

import (
	"errors"
	"strings"
	"testing"

	sporecore "github.com/squirrelsoft-dev/spore-core/go/spore-core"
)

func registryWithRole(t *testing.T, id ChunkID) *StandardPromptChunkRegistry {
	t.Helper()
	r := NewStandardPromptChunkRegistry()
	if err := r.Register(NewPromptChunk(id, "you are a test agent", ChunkSlotRole, CacheBlockStatic)); err != nil {
		t.Fatalf("seed register: %v", err)
	}
	return r
}

// ── Register: error cases ───────────────────────────────────────────────────

func TestRegister_ErrorCases(t *testing.T) {
	cases := []struct {
		name      string
		setup     func(r *StandardPromptChunkRegistry)
		chunk     PromptChunk
		assertErr func(t *testing.T, err error)
	}{
		{
			name: "duplicate_id",
			setup: func(r *StandardPromptChunkRegistry) {
				_ = r.Register(NewPromptChunk("x", "hello", ChunkSlotCapability, CacheBlockStatic))
			},
			chunk: NewPromptChunk("x", "world", ChunkSlotCapability, CacheBlockStatic),
			assertErr: func(t *testing.T, err error) {
				var dup *ErrDuplicateID
				if !errors.As(err, &dup) {
					t.Fatalf("expected *ErrDuplicateID, got %T: %v", err, err)
				}
				if dup.ID != "x" {
					t.Errorf("want id=x, got %q", dup.ID)
				}
			},
		},
		{
			name:  "empty_content",
			chunk: NewPromptChunk("x", "   ", ChunkSlotCapability, CacheBlockStatic),
			assertErr: func(t *testing.T, err error) {
				var bad *InvalidSlotError
				if !errors.As(err, &bad) {
					t.Fatalf("expected *InvalidSlotError, got %T: %v", err, err)
				}
			},
		},
		{
			name:  "budget_slot_rejects_static",
			chunk: NewPromptChunk("b", "budget warning", ChunkSlotBudget, CacheBlockStatic),
			assertErr: func(t *testing.T, err error) {
				var c *ConflictingCacheBlockError
				if !errors.As(err, &c) {
					t.Fatalf("expected *ConflictingCacheBlockError, got %T: %v", err, err)
				}
				if c.Slot != ChunkSlotBudget || c.Expected != CacheBlockPerTurn || c.Actual != CacheBlockStatic {
					t.Errorf("unexpected fields: %+v", c)
				}
			},
		},
		{
			name:  "ephemeral_slot_rejects_per_session",
			chunk: NewPromptChunk("e", "ephemeral", ChunkSlotEphemeral, CacheBlockPerSession),
			assertErr: func(t *testing.T, err error) {
				var c *ConflictingCacheBlockError
				if !errors.As(err, &c) {
					t.Fatalf("expected *ConflictingCacheBlockError, got %T: %v", err, err)
				}
			},
		},
		{
			name:  "role_slot_rejects_per_session",
			chunk: NewPromptChunk("r", "role", ChunkSlotRole, CacheBlockPerSession),
			assertErr: func(t *testing.T, err error) {
				var c *ConflictingCacheBlockError
				if !errors.As(err, &c) {
					t.Fatalf("expected *ConflictingCacheBlockError, got %T: %v", err, err)
				}
			},
		},
		{
			name:  "mode_slot_rejects_per_turn",
			chunk: NewPromptChunk("m", "mode", ChunkSlotMode, CacheBlockPerTurn),
			assertErr: func(t *testing.T, err error) {
				var c *ConflictingCacheBlockError
				if !errors.As(err, &c) {
					t.Fatalf("expected *ConflictingCacheBlockError, got %T: %v", err, err)
				}
			},
		},
	}

	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			r := NewStandardPromptChunkRegistry()
			if tc.setup != nil {
				tc.setup(r)
			}
			err := r.Register(tc.chunk)
			if err == nil {
				t.Fatal("expected error, got nil")
			}
			tc.assertErr(t, err)
		})
	}
}

// ── Compose ─────────────────────────────────────────────────────────────────

func TestCompose_MissingRoleReturnsError(t *testing.T) {
	r := NewStandardPromptChunkRegistry()
	_, err := r.Compose("missing", ModeYolo, nil, nil)
	if err == nil {
		t.Fatal("expected error")
	}
	verrs := AsValidationErrors(err)
	found := false
	for _, e := range verrs {
		var m *MissingRequiredSlotError
		if errors.As(e, &m) && m.Slot == ChunkSlotRole {
			found = true
		}
	}
	if !found {
		t.Fatalf("expected MissingRequiredSlot{Role} in %v", verrs)
	}
}

func TestCompose_IncludesModeChunkFromEnum(t *testing.T) {
	r := registryWithRole(t, "role-test")
	composed, err := r.Compose("role-test", ModePlan, nil, nil)
	if err != nil {
		t.Fatalf("compose: %v", err)
	}
	var modeID ChunkID
	for _, c := range composed.Chunks {
		if c.Slot == ChunkSlotMode {
			modeID = c.ID
		}
	}
	if modeID != "mode-plan" {
		t.Fatalf("expected mode-plan, got %q", modeID)
	}
}

func TestCompose_OrdersBySlot(t *testing.T) {
	r := registryWithRole(t, "role-test")
	if err := r.Register(NewPromptChunk("cap-1", "cap one", ChunkSlotCapability, CacheBlockStatic)); err != nil {
		t.Fatal(err)
	}
	if err := r.Register(NewPromptChunk("skill-1", "skill one", ChunkSlotSkill, CacheBlockStatic)); err != nil {
		t.Fatal(err)
	}
	composed, err := r.Compose("role-test", ModeAutoEdit, []ChunkID{"cap-1"}, []ChunkID{"skill-1"})
	if err != nil {
		t.Fatalf("compose: %v", err)
	}
	want := []ChunkSlot{ChunkSlotRole, ChunkSlotMode, ChunkSlotCapability, ChunkSlotSkill}
	if len(composed.Chunks) != len(want) {
		t.Fatalf("len mismatch: got %d, want %d", len(composed.Chunks), len(want))
	}
	for i, w := range want {
		if composed.Chunks[i].Slot != w {
			t.Errorf("chunks[%d].Slot = %q, want %q", i, composed.Chunks[i].Slot, w)
		}
	}
}

func TestCompose_PreservesArgOrderWithinSlot(t *testing.T) {
	r := registryWithRole(t, "role-test")
	for _, id := range []ChunkID{"cap-a", "cap-b", "cap-c"} {
		if err := r.Register(NewPromptChunk(id, string(id), ChunkSlotCapability, CacheBlockStatic)); err != nil {
			t.Fatal(err)
		}
	}
	composed, err := r.Compose("role-test", ModeYolo, []ChunkID{"cap-c", "cap-a", "cap-b"}, nil)
	if err != nil {
		t.Fatalf("compose: %v", err)
	}
	var caps []ChunkID
	for _, c := range composed.Chunks {
		if c.Slot == ChunkSlotCapability {
			caps = append(caps, c.ID)
		}
	}
	want := []ChunkID{"cap-c", "cap-a", "cap-b"}
	for i, w := range want {
		if caps[i] != w {
			t.Errorf("caps[%d]=%q, want %q", i, caps[i], w)
		}
	}
}

// ── Block hashes ────────────────────────────────────────────────────────────

func TestBlockHashes_StableForIdenticalContent(t *testing.T) {
	a, err := registryWithRole(t, "role-test").Compose("role-test", ModeYolo, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	b, err := registryWithRole(t, "role-test").Compose("role-test", ModeYolo, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if a.Block1Hash != b.Block1Hash {
		t.Errorf("block1 hashes differ: %d vs %d", a.Block1Hash, b.Block1Hash)
	}
	if a.Block2Hash != b.Block2Hash {
		t.Errorf("block2 hashes differ: %d vs %d", a.Block2Hash, b.Block2Hash)
	}
}

func TestBlock1Hash_ChangesWhenContentChanges(t *testing.T) {
	a, err := registryWithRole(t, "role-test").Compose("role-test", ModeYolo, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	r2 := NewStandardPromptChunkRegistry()
	if err := r2.Register(NewPromptChunk("role-test", "DIFFERENT ROLE CONTENT", ChunkSlotRole, CacheBlockStatic)); err != nil {
		t.Fatal(err)
	}
	b, err := r2.Compose("role-test", ModeYolo, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if a.Block1Hash == b.Block1Hash {
		t.Error("block1 hash should change when role content changes")
	}
}

// ── Validate ────────────────────────────────────────────────────────────────

func TestValidate_FlagsPerTurnChunkInStaticBlock(t *testing.T) {
	r := NewStandardPromptChunkRegistry()
	composed := &ComposedPrompt{
		Chunks: []PromptChunk{
			NewPromptChunk("role-x", "x", ChunkSlotRole, CacheBlockStatic),
			ModeYolo.PromptChunk(),
			// Budget chunk with Static cache block — simulates a bug.
			{ID: "bad-budget", Content: "b", Slot: ChunkSlotBudget, CacheBlock: CacheBlockStatic},
		},
	}
	errs := r.Validate(composed)
	found := false
	for _, e := range errs {
		var pe *PerTurnChunkInStaticBlockError
		if errors.As(e, &pe) && pe.ID == "bad-budget" {
			found = true
		}
	}
	if !found {
		t.Fatalf("expected PerTurnChunkInStaticBlock for bad-budget, got %v", errs)
	}
}

func TestValidate_FlagsMoreThanOneModeChunk(t *testing.T) {
	r := NewStandardPromptChunkRegistry()
	composed := &ComposedPrompt{
		Chunks: []PromptChunk{
			NewPromptChunk("role-x", "x", ChunkSlotRole, CacheBlockStatic),
			ModeYolo.PromptChunk(),
			ModeAlwaysAsk.PromptChunk(),
		},
	}
	errs := r.Validate(composed)
	found := false
	for _, e := range errs {
		var c *ConflictingModeChunksError
		if errors.As(e, &c) {
			found = true
		}
	}
	if !found {
		t.Fatalf("expected ConflictingModeChunks, got %v", errs)
	}
}

func TestValidate_FlagsMissingRoleSlot(t *testing.T) {
	r := NewStandardPromptChunkRegistry()
	composed := &ComposedPrompt{
		Chunks: []PromptChunk{ModeYolo.PromptChunk()},
	}
	errs := r.Validate(composed)
	found := false
	for _, e := range errs {
		var m *MissingRequiredSlotError
		if errors.As(e, &m) && m.Slot == ChunkSlotRole {
			found = true
		}
	}
	if !found {
		t.Fatalf("expected MissingRequiredSlot{Role}, got %v", errs)
	}
}

// ── Get ─────────────────────────────────────────────────────────────────────

func TestGet_ReturnsRegisteredChunk(t *testing.T) {
	r := registryWithRole(t, "role-x")
	c, ok := r.Get("role-x")
	if !ok {
		t.Fatal("expected found")
	}
	if c.ID != "role-x" {
		t.Errorf("id mismatch: %q", c.ID)
	}
	if _, ok := r.Get("nope"); ok {
		t.Error("expected not found")
	}
}

// ── Mode helpers ────────────────────────────────────────────────────────────

func TestMode_ApprovalPolicyMatchesSpec(t *testing.T) {
	cases := []struct {
		m    Mode
		want ApprovalPolicy
	}{
		{ModeAlwaysAsk, ApprovalPolicyAlwaysAsk},
		{ModeAutoEdit, ApprovalPolicyAutoExplain},
		{ModePlan, ApprovalPolicyPlanOnly},
		{ModeSafeAuto, ApprovalPolicySafeAuto},
		{ModeYolo, ApprovalPolicyNone},
	}
	for _, c := range cases {
		if got := c.m.ApprovalPolicy(); got != c.want {
			t.Errorf("%s: got %q, want %q", c.m, got, c.want)
		}
	}
}

func TestMode_DefaultToolPhase(t *testing.T) {
	if got := ModePlan.DefaultToolPhase(); got != sporecore.PhasePlanning {
		t.Errorf("Plan: %q", got)
	}
	for _, m := range []Mode{ModeAlwaysAsk, ModeAutoEdit, ModeSafeAuto, ModeYolo} {
		if got := m.DefaultToolPhase(); got != sporecore.PhaseExecution {
			t.Errorf("%s: %q", m, got)
		}
	}
}

func TestMode_PromptChunkIDsAndPrefixes(t *testing.T) {
	cases := []struct {
		m      Mode
		id     ChunkID
		prefix string
	}{
		{ModeAlwaysAsk, "mode-always-ask", "Mode: AlwaysAsk."},
		{ModeAutoEdit, "mode-auto-edit", "Mode: AutoEdit."},
		{ModePlan, "mode-plan", "Mode: Plan."},
		{ModeSafeAuto, "mode-safe-auto", "Mode: SafeAuto."},
		{ModeYolo, "mode-yolo", "Mode: Yolo."},
	}
	for _, c := range cases {
		pc := c.m.PromptChunk()
		if pc.ID != c.id {
			t.Errorf("%s: id=%q, want %q", c.m, pc.ID, c.id)
		}
		if !strings.HasPrefix(pc.Content, c.prefix) {
			t.Errorf("%s: content %q missing prefix %q", c.m, pc.Content, c.prefix)
		}
		if pc.Slot != ChunkSlotMode || pc.CacheBlock != CacheBlockStatic {
			t.Errorf("%s: slot=%q cache=%q", c.m, pc.Slot, pc.CacheBlock)
		}
	}
}

// ── ComposedPrompt rendering ────────────────────────────────────────────────

func TestComposedPrompt_RenderJoinsChunks(t *testing.T) {
	r := registryWithRole(t, "role-test")
	composed, err := r.Compose("role-test", ModeYolo, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if composed.HasRendered() {
		t.Error("expected no cached render initially")
	}
	rendered := composed.Render()
	if !strings.Contains(rendered, "you are a test agent") {
		t.Error("missing role content")
	}
	if !strings.Contains(rendered, "Mode: Yolo") {
		t.Error("missing mode content")
	}
	if !composed.HasRendered() {
		t.Error("expected cached render after Render()")
	}
}

// ── Standard library smoke ──────────────────────────────────────────────────

func TestStandardChunks_RegisterCleanly(t *testing.T) {
	r := NewStandardPromptChunkRegistry()
	if err := r.RegisterStandardChunks(); err != nil {
		t.Fatalf("register standard: %v", err)
	}
	if _, ok := r.Get("role-coding-agent"); !ok {
		t.Error("missing role-coding-agent")
	}
	if _, ok := r.Get("capability-bash"); !ok {
		t.Error("missing capability-bash")
	}
	if _, ok := r.Get("skill-testing"); !ok {
		t.Error("missing skill-testing")
	}
}

func TestCompose_WithStandardChunks_CodingAgent(t *testing.T) {
	r := NewStandardPromptChunkRegistry()
	if err := r.RegisterStandardChunks(); err != nil {
		t.Fatal(err)
	}
	composed, err := r.Compose(
		"role-coding-agent",
		ModeSafeAuto,
		[]ChunkID{"capability-bash", "capability-filesystem", "capability-git"},
		[]ChunkID{"skill-testing", "skill-security-review"},
	)
	if err != nil {
		t.Fatal(err)
	}
	if want := 1 + 1 + 3 + 2; len(composed.Chunks) != want {
		t.Errorf("len=%d, want %d", len(composed.Chunks), want)
	}
	if composed.Chunks[0].Slot != ChunkSlotRole {
		t.Errorf("first slot = %q", composed.Chunks[0].Slot)
	}
	if composed.Chunks[1].Slot != ChunkSlotMode || composed.Chunks[1].ID != "mode-safe-auto" {
		t.Errorf("second chunk = %+v", composed.Chunks[1])
	}
}
