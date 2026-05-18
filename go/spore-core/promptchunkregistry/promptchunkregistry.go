// Package promptchunkregistry — issue #24 `PromptChunkRegistry`: named,
// cacheable prompt chunks composed into a deterministic system prompt at
// harness construction time.
//
// See `docs/harness-engineering-concepts.md` § "Cache Architecture" and the
// spec issue for the rules this package enforces.
//
// Rules enforced
//   - Each chunk has a unique ChunkID. Duplicate registration is
//     ErrDuplicateID.
//   - ChunkSlotBudget and ChunkSlotEphemeral are always PerTurn.
//     Registering with a different cache block is a ConflictingCacheBlockError.
//   - ChunkSlotRole and ChunkSlotMode are always Static. Anything else is a
//     ConflictingCacheBlockError.
//   - Chunk content must not be empty (InvalidSlotError).
//   - Compose requires the named role chunk to be registered and resolves
//     the mode chunk via Mode.PromptChunk. Capability and skill chunks must
//     already be registered.
//   - The composed chunk list is ordered by slot (Role → Mode → Capability
//     → Skill → Task → Environment → PriorSession → Budget → Ephemeral) and,
//     within a slot, by registration / argument order.
//   - Block1Hash is the FNV-64a digest of all Static chunk contents.
//     Block2Hash is the digest of all PerSession chunk contents.
//   - Validate rejects compositions with a PerTurn chunk in Block 1,
//     missing required slots (Role and Mode), or more than one Mode chunk.
//   - Mode is permanent for the life of a harness instance — there is no
//     mutation API.
package promptchunkregistry

import (
	"errors"
	"fmt"
	"hash/fnv"
	"sort"
	"strings"
	"sync"

	sporecore "github.com/squirrelsoft-dev/spore-core/go/spore-core"
)

// ============================================================================
// Identity
// ============================================================================

// ChunkID is the stable identifier for a chunk.
type ChunkID string

// ============================================================================
// Enums
// ============================================================================

// ChunkSlot identifies the section the chunk lands in.
type ChunkSlot string

const (
	ChunkSlotRole         ChunkSlot = "role"
	ChunkSlotMode         ChunkSlot = "mode"
	ChunkSlotCapability   ChunkSlot = "capability"
	ChunkSlotSkill        ChunkSlot = "skill"
	ChunkSlotTask         ChunkSlot = "task"
	ChunkSlotEnvironment  ChunkSlot = "environment"
	ChunkSlotPriorSession ChunkSlot = "prior_session"
	ChunkSlotBudget       ChunkSlot = "budget"
	ChunkSlotEphemeral    ChunkSlot = "ephemeral"
)

// order returns the render order for a slot.
func (s ChunkSlot) order() int {
	switch s {
	case ChunkSlotRole:
		return 0
	case ChunkSlotMode:
		return 1
	case ChunkSlotCapability:
		return 2
	case ChunkSlotSkill:
		return 3
	case ChunkSlotTask:
		return 4
	case ChunkSlotEnvironment:
		return 5
	case ChunkSlotPriorSession:
		return 6
	case ChunkSlotBudget:
		return 7
	case ChunkSlotEphemeral:
		return 8
	default:
		return 99
	}
}

// CacheBlock identifies which cache layer holds a chunk.
type CacheBlock string

const (
	CacheBlockStatic     CacheBlock = "static"
	CacheBlockPerSession CacheBlock = "per_session"
	CacheBlockPerTurn    CacheBlock = "per_turn"
)

// ApprovalPolicy is the enforcement policy implied by a Mode.
type ApprovalPolicy string

const (
	// ApprovalPolicyAlwaysAsk — every action requires approval before execution.
	ApprovalPolicyAlwaysAsk ApprovalPolicy = "always_ask"
	// ApprovalPolicyAutoExplain — actions proceed automatically; narrate after.
	ApprovalPolicyAutoExplain ApprovalPolicy = "auto_explain"
	// ApprovalPolicyPlanOnly — planning only; file edits blocked.
	ApprovalPolicyPlanOnly ApprovalPolicy = "plan_only"
	// ApprovalPolicySafeAuto — auto for Low/Medium; High/Critical require approval.
	ApprovalPolicySafeAuto ApprovalPolicy = "safe_auto"
	// ApprovalPolicyNone — full autonomy.
	ApprovalPolicyNone ApprovalPolicy = "none"
)

// Mode is the operating mode of the harness. Permanent for the life of the
// harness instance.
type Mode string

const (
	ModeAlwaysAsk Mode = "always_ask"
	ModeAutoEdit  Mode = "auto_edit"
	ModePlan      Mode = "plan"
	ModeSafeAuto  Mode = "safe_auto"
	ModeYolo      Mode = "yolo"
)

// PromptChunk returns the standard chunk for this mode. Lands in
// ChunkSlotMode in Block 1; always Static.
func (m Mode) PromptChunk() PromptChunk {
	var id ChunkID
	var content string
	switch m {
	case ModeAlwaysAsk:
		id = "mode-always-ask"
		content = "Mode: AlwaysAsk. Describe your plan and wait for explicit approval before taking any action."
	case ModeAutoEdit:
		id = "mode-auto-edit"
		content = "Mode: AutoEdit. Edit files freely. Explain the changes you make after they are done."
	case ModePlan:
		id = "mode-plan"
		content = "Mode: Plan. Produce a plan only. Do not edit files or execute mutating tools."
	case ModeSafeAuto:
		id = "mode-safe-auto"
		content = "Mode: SafeAuto. Auto-execute Low and Medium risk actions. High and Critical actions require approval."
	case ModeYolo:
		id = "mode-yolo"
		content = "Mode: Yolo. Full autonomy. No approval gates."
	}
	return NewPromptChunk(id, content, ChunkSlotMode, CacheBlockStatic)
}

// ApprovalPolicy returns the enforcement policy implied by the mode.
func (m Mode) ApprovalPolicy() ApprovalPolicy {
	switch m {
	case ModeAlwaysAsk:
		return ApprovalPolicyAlwaysAsk
	case ModeAutoEdit:
		return ApprovalPolicyAutoExplain
	case ModePlan:
		return ApprovalPolicyPlanOnly
	case ModeSafeAuto:
		return ApprovalPolicySafeAuto
	case ModeYolo:
		return ApprovalPolicyNone
	default:
		return ApprovalPolicyAlwaysAsk
	}
}

// DefaultToolPhase returns the initial task phase implied by this mode.
func (m Mode) DefaultToolPhase() sporecore.TaskPhase {
	if m == ModePlan {
		return sporecore.PhasePlanning
	}
	return sporecore.PhaseExecution
}

// ============================================================================
// Records
// ============================================================================

// PromptChunk is a named, cacheable text fragment.
type PromptChunk struct {
	ID         ChunkID    `json:"id"`
	Content    string     `json:"content"`
	Slot       ChunkSlot  `json:"slot"`
	CacheBlock CacheBlock `json:"cache_block"`
	Hash       uint64     `json:"hash"`
}

// NewPromptChunk builds a chunk and computes its content hash. Use this
// instead of a struct literal so the hash stays in sync with content.
func NewPromptChunk(id ChunkID, content string, slot ChunkSlot, cacheBlock CacheBlock) PromptChunk {
	return PromptChunk{
		ID:         id,
		Content:    content,
		Slot:       slot,
		CacheBlock: cacheBlock,
		Hash:       hashContent(content),
	}
}

// ComposedPrompt is the result of Compose. The harness stores one of these
// per agent and reuses it for the life of the instance.
type ComposedPrompt struct {
	Chunks      []PromptChunk `json:"chunks"`
	Block1Hash  uint64        `json:"block_1_hash"`
	Block2Hash  uint64        `json:"block_2_hash"`
	rendered    string
	hasRendered bool
}

// Render returns the deterministic rendering of the chunk list (chunks joined
// by a blank line). Caches the result; safe to call repeatedly.
func (c *ComposedPrompt) Render() string {
	if !c.hasRendered {
		parts := make([]string, len(c.Chunks))
		for i, ch := range c.Chunks {
			parts[i] = ch.Content
		}
		c.rendered = strings.Join(parts, "\n\n")
		c.hasRendered = true
	}
	return c.rendered
}

// RenderedStr returns the cached render or an empty string when not yet
// rendered. Useful in read-only contexts.
func (c *ComposedPrompt) RenderedStr() string {
	if !c.hasRendered {
		return ""
	}
	return c.rendered
}

// HasRendered reports whether Render has been called.
func (c *ComposedPrompt) HasRendered() bool { return c.hasRendered }

// RecomputeHashes recomputes the (block1, block2) digests from the current
// chunk slice.
func (c *ComposedPrompt) RecomputeHashes() (uint64, uint64) {
	return computeBlockHashes(c.Chunks)
}

// ============================================================================
// Errors
// ============================================================================

// ChunkError is the interface satisfied by all registration errors.
type ChunkError interface {
	error
	chunkError()
}

// ErrDuplicateID is returned when registering a chunk whose id already exists.
type ErrDuplicateID struct{ ID ChunkID }

func (e *ErrDuplicateID) Error() string { return fmt.Sprintf("duplicate chunk id: %q", e.ID) }
func (e *ErrDuplicateID) chunkError()   {}

// InvalidSlotError is returned for malformed chunks (e.g. empty content).
type InvalidSlotError struct {
	ID     ChunkID
	Reason string
}

func (e *InvalidSlotError) Error() string {
	return fmt.Sprintf("invalid slot for chunk %q: %s", e.ID, e.Reason)
}
func (e *InvalidSlotError) chunkError() {}

// ConflictingCacheBlockError is returned when a chunk's CacheBlock is invalid
// for its Slot.
type ConflictingCacheBlockError struct {
	ID       ChunkID
	Slot     ChunkSlot
	Expected CacheBlock
	Actual   CacheBlock
}

func (e *ConflictingCacheBlockError) Error() string {
	return fmt.Sprintf(
		"conflicting cache block for chunk %q in slot %q: expected %q, got %q",
		e.ID, e.Slot, e.Expected, e.Actual,
	)
}
func (e *ConflictingCacheBlockError) chunkError() {}

// NotFoundError is returned when a chunk id cannot be resolved.
type NotFoundError struct{ ID ChunkID }

func (e *NotFoundError) Error() string { return fmt.Sprintf("chunk not found: %q", e.ID) }
func (e *NotFoundError) chunkError()   {}

// ChunkValidationError is the interface satisfied by all composition validation
// errors.
type ChunkValidationError interface {
	error
	chunkValidationError()
}

// PerTurnChunkInStaticBlockError flags a PerTurn-slot chunk that landed in
// Block 1.
type PerTurnChunkInStaticBlockError struct{ ID ChunkID }

func (e *PerTurnChunkInStaticBlockError) Error() string {
	return fmt.Sprintf("per-turn chunk %q placed in the Static block", e.ID)
}
func (e *PerTurnChunkInStaticBlockError) chunkValidationError() {}

// MissingRequiredSlotError flags a missing required slot (Role or Mode).
type MissingRequiredSlotError struct{ Slot ChunkSlot }

func (e *MissingRequiredSlotError) Error() string {
	return fmt.Sprintf("required slot %q is missing from the composition", e.Slot)
}
func (e *MissingRequiredSlotError) chunkValidationError() {}

// ConflictingModeChunksError flags more than one Mode chunk in a composition.
type ConflictingModeChunksError struct{ IDs []ChunkID }

func (e *ConflictingModeChunksError) Error() string {
	return fmt.Sprintf("more than one Mode chunk in the composition: %v", e.IDs)
}
func (e *ConflictingModeChunksError) chunkValidationError() {}

// ValidationErrors aggregates multiple validation errors so Compose can return
// them via a single error value (use errors.As to unwrap individual entries).
type ValidationErrors struct{ Errors []ChunkValidationError }

func (v *ValidationErrors) Error() string {
	parts := make([]string, len(v.Errors))
	for i, e := range v.Errors {
		parts[i] = e.Error()
	}
	return "composition validation failed: " + strings.Join(parts, "; ")
}

// Unwrap exposes the contained validation errors to errors.Is / errors.As.
func (v *ValidationErrors) Unwrap() []error {
	out := make([]error, len(v.Errors))
	for i, e := range v.Errors {
		out[i] = e
	}
	return out
}

// ============================================================================
// Interface
// ============================================================================

// PromptChunkRegistry is the consumer-side interface for chunk registries.
type PromptChunkRegistry interface {
	// Register a chunk. Validates slot/cache-block compatibility before storing.
	Register(chunk PromptChunk) error

	// Compose chunks for a given agent configuration. Called once at harness
	// construction; the result is cached on the harness instance.
	Compose(role ChunkID, mode Mode, capabilities []ChunkID, skills []ChunkID) (ComposedPrompt, error)

	// Validate a composition. Returns nil when valid.
	Validate(composed *ComposedPrompt) []ChunkValidationError

	// Get looks up a chunk by id. Returns false when not found.
	Get(id ChunkID) (PromptChunk, bool)
}

// ============================================================================
// Standard implementation
// ============================================================================

// StandardPromptChunkRegistry is the reference, in-memory PromptChunkRegistry.
// The harness owns a single shared instance; chunks are typically registered
// at startup and never mutated afterwards.
type StandardPromptChunkRegistry struct {
	mu     sync.Mutex
	chunks map[ChunkID]PromptChunk
	order  []ChunkID
}

// NewStandardPromptChunkRegistry returns an empty registry.
func NewStandardPromptChunkRegistry() *StandardPromptChunkRegistry {
	return &StandardPromptChunkRegistry{
		chunks: make(map[ChunkID]PromptChunk),
	}
}

// RegisterStandardChunks registers every chunk in the standard library.
// Returns the first error if any chunk fails validation (e.g. on collision).
func (r *StandardPromptChunkRegistry) RegisterStandardChunks() error {
	for _, c := range StandardChunks() {
		if err := r.Register(c); err != nil {
			return err
		}
	}
	return nil
}

// Register a chunk.
func (r *StandardPromptChunkRegistry) Register(chunk PromptChunk) error {
	if err := validateSlotAndCacheBlock(chunk); err != nil {
		return err
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if _, ok := r.chunks[chunk.ID]; ok {
		return &ErrDuplicateID{ID: chunk.ID}
	}
	r.chunks[chunk.ID] = chunk
	r.order = append(r.order, chunk.ID)
	return nil
}

// Compose builds a ComposedPrompt for the given agent configuration.
func (r *StandardPromptChunkRegistry) Compose(
	role ChunkID,
	mode Mode,
	capabilities []ChunkID,
	skills []ChunkID,
) (ComposedPrompt, error) {
	r.mu.Lock()
	defer r.mu.Unlock()

	var errs []ChunkValidationError
	chosen := make([]PromptChunk, 0, 2+len(capabilities)+len(skills))

	// Role
	if c, ok := r.chunks[role]; ok && c.Slot == ChunkSlotRole {
		chosen = append(chosen, c)
	} else {
		errs = append(errs, &MissingRequiredSlotError{Slot: ChunkSlotRole})
	}

	// Mode — always sourced from the enum.
	chosen = append(chosen, mode.PromptChunk())

	// Capabilities
	for _, id := range capabilities {
		if c, ok := r.chunks[id]; ok && c.Slot == ChunkSlotCapability {
			chosen = append(chosen, c)
		} else {
			errs = append(errs, &MissingRequiredSlotError{Slot: ChunkSlotCapability})
		}
	}

	// Skills
	for _, id := range skills {
		if c, ok := r.chunks[id]; ok && c.Slot == ChunkSlotSkill {
			chosen = append(chosen, c)
		} else {
			errs = append(errs, &MissingRequiredSlotError{Slot: ChunkSlotSkill})
		}
	}

	if len(errs) > 0 {
		return ComposedPrompt{}, &ValidationErrors{Errors: errs}
	}

	// Stable sort by slot order; within a slot, insertion order is preserved.
	sort.SliceStable(chosen, func(i, j int) bool {
		return chosen[i].Slot.order() < chosen[j].Slot.order()
	})

	h1, h2 := computeBlockHashes(chosen)
	composed := ComposedPrompt{
		Chunks:     chosen,
		Block1Hash: h1,
		Block2Hash: h2,
	}

	if vErrs := r.validateLocked(&composed); len(vErrs) > 0 {
		return ComposedPrompt{}, &ValidationErrors{Errors: vErrs}
	}

	return composed, nil
}

// Validate runs the composition rules over a ComposedPrompt.
func (r *StandardPromptChunkRegistry) Validate(composed *ComposedPrompt) []ChunkValidationError {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.validateLocked(composed)
}

func (r *StandardPromptChunkRegistry) validateLocked(composed *ComposedPrompt) []ChunkValidationError {
	var errs []ChunkValidationError

	// Block 1 must not contain PerTurn-slot chunks marked Static.
	for _, c := range composed.Chunks {
		if c.CacheBlock == CacheBlockStatic &&
			(c.Slot == ChunkSlotBudget || c.Slot == ChunkSlotEphemeral) {
			errs = append(errs, &PerTurnChunkInStaticBlockError{ID: c.ID})
		}
	}

	// Required slots: Role and Mode.
	for _, required := range []ChunkSlot{ChunkSlotRole, ChunkSlotMode} {
		found := false
		for _, c := range composed.Chunks {
			if c.Slot == required {
				found = true
				break
			}
		}
		if !found {
			errs = append(errs, &MissingRequiredSlotError{Slot: required})
		}
	}

	// Exactly one Mode chunk.
	var modeIDs []ChunkID
	for _, c := range composed.Chunks {
		if c.Slot == ChunkSlotMode {
			modeIDs = append(modeIDs, c.ID)
		}
	}
	if len(modeIDs) > 1 {
		errs = append(errs, &ConflictingModeChunksError{IDs: modeIDs})
	}

	return errs
}

// Get returns the chunk with the given id.
func (r *StandardPromptChunkRegistry) Get(id ChunkID) (PromptChunk, bool) {
	r.mu.Lock()
	defer r.mu.Unlock()
	c, ok := r.chunks[id]
	return c, ok
}

// Compile-time interface conformance check.
var _ PromptChunkRegistry = (*StandardPromptChunkRegistry)(nil)

// ============================================================================
// Helpers
// ============================================================================

func hashContent(content string) uint64 {
	h := fnv.New64a()
	_, _ = h.Write([]byte(content))
	return h.Sum64()
}

func computeBlockHashes(chunks []PromptChunk) (uint64, uint64) {
	// Group by cache block, then sort by id for deterministic combination.
	type entry struct {
		id   ChunkID
		hash uint64
	}
	var block1, block2 []entry
	for _, c := range chunks {
		switch c.CacheBlock {
		case CacheBlockStatic:
			block1 = append(block1, entry{c.ID, c.Hash})
		case CacheBlockPerSession:
			block2 = append(block2, entry{c.ID, c.Hash})
		case CacheBlockPerTurn:
			// not hashed
		}
	}
	sortByID := func(s []entry) {
		sort.Slice(s, func(i, j int) bool { return s[i].id < s[j].id })
	}
	sortByID(block1)
	sortByID(block2)

	combine := func(s []entry) uint64 {
		h := fnv.New64a()
		for _, e := range s {
			_, _ = h.Write([]byte(e.id))
			var buf [8]byte
			v := e.hash
			for i := 0; i < 8; i++ {
				buf[i] = byte(v >> (8 * i))
			}
			_, _ = h.Write(buf[:])
		}
		return h.Sum64()
	}
	return combine(block1), combine(block2)
}

func validateSlotAndCacheBlock(chunk PromptChunk) error {
	if strings.TrimSpace(chunk.Content) == "" {
		return &InvalidSlotError{ID: chunk.ID, Reason: "content must not be empty"}
	}
	switch chunk.Slot {
	case ChunkSlotBudget, ChunkSlotEphemeral:
		if chunk.CacheBlock != CacheBlockPerTurn {
			return &ConflictingCacheBlockError{
				ID:       chunk.ID,
				Slot:     chunk.Slot,
				Expected: CacheBlockPerTurn,
				Actual:   chunk.CacheBlock,
			}
		}
	case ChunkSlotRole, ChunkSlotMode:
		if chunk.CacheBlock != CacheBlockStatic {
			return &ConflictingCacheBlockError{
				ID:       chunk.ID,
				Slot:     chunk.Slot,
				Expected: CacheBlockStatic,
				Actual:   chunk.CacheBlock,
			}
		}
	}
	return nil
}

// ============================================================================
// Standard chunk library
// ============================================================================

// StandardChunks returns the standard chunk library shipped by spore-core.
// Users register additional chunks for their domain.
func StandardChunks() []PromptChunk {
	out := make([]PromptChunk, 0, 20)

	roles := []struct{ id, content string }{
		{"role-coding-agent", "You are an expert software engineer. Read carefully, change deliberately, and verify your work."},
		{"role-evaluator", "You are a fresh evaluator. You did not write the code under review. Default to FAIL."},
		{"role-planner", "You are a planning specialist. Decompose tasks into small, verifiable steps."},
		{"role-rag-assistant", "You are a retrieval-augmented assistant. Always cite the source document for any claim."},
		{"role-sql-agent", "You are a SQL specialist. Prefer read-only queries; never DROP without explicit approval."},
	}
	for _, r := range roles {
		out = append(out, NewPromptChunk(ChunkID(r.id), r.content, ChunkSlotRole, CacheBlockStatic))
	}

	// Modes — derived from the enum so PromptChunk() and StandardChunks() agree.
	for _, m := range []Mode{ModeAlwaysAsk, ModeAutoEdit, ModePlan, ModeSafeAuto, ModeYolo} {
		out = append(out, m.PromptChunk())
	}

	capabilities := []struct{ id, content string }{
		{"capability-bash", "Capability: bash. You can run shell commands inside the sandbox."},
		{"capability-filesystem", "Capability: filesystem. You can read and write files inside the workspace."},
		{"capability-git", "Capability: git. You can stage, commit, and inspect history."},
		{"capability-browser", "Capability: browser. You can fetch web pages and follow links."},
		{"capability-subagent", "Capability: subagent. You can delegate work to a child harness."},
		{"capability-sql", "Capability: sql. You can issue queries against the configured database."},
	}
	for _, c := range capabilities {
		out = append(out, NewPromptChunk(ChunkID(c.id), c.content, ChunkSlotCapability, CacheBlockStatic))
	}

	skills := []struct{ id, content string }{
		{"skill-testing", "Skill: always run the test suite after changes and report results."},
		{"skill-decomposition", "Skill: break large tasks into small, independently verifiable steps."},
		{"skill-security-review", "Skill: review changes for injection, auth, and secret-leak issues before commit."},
		{"skill-citation", "Skill: cite the source document for every claim drawn from retrieved context."},
	}
	for _, s := range skills {
		out = append(out, NewPromptChunk(ChunkID(s.id), s.content, ChunkSlotSkill, CacheBlockStatic))
	}

	return out
}

// AsValidationErrors unwraps an error returned by Compose into the underlying
// slice of validation errors, or nil if the error is not a ValidationErrors.
func AsValidationErrors(err error) []ChunkValidationError {
	var v *ValidationErrors
	if errors.As(err, &v) {
		return v.Errors
	}
	return nil
}
