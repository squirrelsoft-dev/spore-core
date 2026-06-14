// Package contextmgr — issue #7 `ContextManager`: assemble and maintain the
// context window.
//
// Builds per-turn context from a pre-computed Block-1 ComposedPrompt,
// per-session metadata (Block 2), and per-turn ephemera (Block 3). Tracks
// token usage, compacts on threshold, offloads large tool results via the
// SandboxProvider, and injects just-in-time skill chunks.
//
// See `docs/harness-engineering-concepts.md` § "ContextManager" and
// § "Cache Architecture" for the cross-language rules this package enforces.
//
// The interface defined here is the canonical issue-#7 surface. The narrower
// stub of the same name in the top-level `sporecore` package is the
// placeholder used by `StandardHarness` while the wider rewrite lands; this
// package lives under a different import path on purpose to avoid name
// collisions during the transition.
package contextmgr

import (
	"context"
	"fmt"
	"hash/fnv"
	"regexp"
	"sort"
	"strings"
	"sync"

	sporecore "github.com/squirrelsoft-dev/spore-core/go/spore-core"
	"github.com/squirrelsoft-dev/spore-core/go/spore-core/promptchunkregistry"
)

// ============================================================================
// Forward-declared sibling types (issues #8, #9, #14)
// ============================================================================

// GuideID is the stable identifier for a guide or skill (issue #9).
type GuideID string

// Guide carries the rendered chunk and an identifier. Full lifecycle metadata
// lives with GuideRegistry (#9). Content must be stored in its final rendered
// form — the spec forbids reformatting at assembly time.
type Guide struct {
	ID      GuideID `json:"id"`
	Content string  `json:"content"`
}

// MemoryItem is forward-declared from MemoryProvider (#8).
type MemoryItem struct {
	Key     string `json:"key"`
	Content string `json:"content"`
}

// ComposedPrompt is forward-declared from PromptChunkRegistry (#14). Block 1
// is computed ONCE at harness startup. Rendered is the final byte-for-byte
// content; Block1Hash is a stable digest used by ContextManager to detect
// unexpected cache invalidation.
type ComposedPrompt struct {
	Rendered   string `json:"rendered"`
	Block1Hash uint64 `json:"block_1_hash"`
}

// CacheBlockHits is the per-block cache hit signal recorded into ContextMeta
// after each model response. Distinct from sporecore.CacheStats (#25), which
// carries token counts and costs parsed from the response.
type CacheBlockHits struct {
	StaticHit  *bool `json:"static_hit"`
	SessionHit *bool `json:"session_hit"`
	HistoryHit *bool `json:"history_hit"`
}

// CacheProvider, NullCacheProvider, and friends live in cache_provider.go
// (issue #25). The interface is the canonical surface; ContextManager calls
// Annotate after assembling a Context.

// ============================================================================
// Spec-defined types
// ============================================================================

// SegmentStability classifies a PromptSegment by its caching tier.
type SegmentStability string

const (
	// StabilityStatic — Block 1, computed once, permanent cache hit.
	StabilityStatic SegmentStability = "static"
	// StabilityPerSession — Block 2, stable within a session.
	StabilityPerSession SegmentStability = "per_session"
	// StabilityPerTurn — Block 3, may change every turn, never cached.
	StabilityPerTurn SegmentStability = "per_turn"
)

// PromptSegment is one named chunk of the system prompt.
type PromptSegment struct {
	Name            string           `json:"name"`
	Content         string           `json:"content"`
	Stability       SegmentStability `json:"stability"`
	CacheBreakpoint bool             `json:"cache_breakpoint,omitempty"`
}

// BreakpointInfo records the location of a provider cache breakpoint.
type BreakpointInfo struct {
	AfterSegment string `json:"after_segment"`
	TokenOffset  uint32 `json:"token_offset"`
}

// RenderedSystemPrompt is the assembled system prompt with breakpoint metadata
// and per-block hashes used for cache-stability invariants.
type RenderedSystemPrompt struct {
	Content          string           `json:"content"`
	Breakpoints      []BreakpointInfo `json:"breakpoints"`
	StaticBlockHash  uint64           `json:"static_block_hash"`
	SessionBlockHash uint64           `json:"session_block_hash"`
}

// CacheBlockStatus reports per-block cache hit/miss from a CacheProvider.
type CacheBlockStatus struct {
	StaticHit  *bool `json:"static_hit"`
	SessionHit *bool `json:"session_hit"`
	HistoryHit *bool `json:"history_hit"`
}

// ContextMeta carries per-turn metadata about the assembled Context.
type ContextMeta struct {
	SessionID      sporecore.SessionID `json:"session_id"`
	TurnNumber     uint32              `json:"turn_number"`
	ActivePhase    sporecore.TaskPhase `json:"active_phase"`
	GuidesLoaded   []GuideID           `json:"guides_loaded"`
	SkillsInjected []GuideID           `json:"skills_injected"`
	Compacted      bool                `json:"compacted"`
	CacheBlocks    CacheBlockStatus    `json:"cache_blocks"`
}

// Context is the assembled per-turn context.
//
// Distinct from sporecore.Context (the narrower bundle the agent treats as
// immutable input). Use ToRequest to project into a sporecore.ModelRequest.
type Context struct {
	SystemPrompt RenderedSystemPrompt   `json:"system_prompt"`
	Messages     []sporecore.Message    `json:"messages"`
	ToolSchemas  []sporecore.ToolSchema `json:"tool_schemas"`
	TokenCount   uint32                 `json:"token_count"`
	WindowLimit  uint32                 `json:"window_limit"`
	Utilization  float32                `json:"utilization"`
	Meta         ContextMeta            `json:"meta"`
}

// ToRequest projects to a sporecore.ModelRequest. The system prompt is
// prepended as a Role==system Message so existing agents can consume the
// assembled context without a new field.
func (c Context) ToRequest(params sporecore.ModelParams) sporecore.ModelRequest {
	msgs := make([]sporecore.Message, 0, len(c.Messages)+1)
	msgs = append(msgs, sporecore.Message{
		Role:    sporecore.RoleSystem,
		Content: sporecore.NewTextContent(c.SystemPrompt.Content),
	})
	msgs = append(msgs, c.Messages...)
	return sporecore.ModelRequest{
		Messages: msgs,
		Tools:    c.ToolSchemas,
		Params:   params,
		Stream:   false,
	}
}

// DefaultContextLength is the conservative fallback compaction window used
// when neither the caller's CompactionConfig.ContextLength nor the model's
// Provider().ContextWindow supplies a usable (> 0) value (issue #141).
//
// Deliberately small (8K, gemma-class) rather than the old 200K: when the real
// context length is unknown, assume a tight window so compaction still fires
// rather than silently never running.
const DefaultContextLength uint32 = 8000

// CompactionConfig controls when and how compaction runs.
type CompactionConfig struct {
	Threshold       float32 `json:"threshold"`
	PreserveRecentN uint32  `json:"preserve_recent_n"`
	HeadTailTokens  uint32  `json:"head_tail_tokens"`
	OffloadPath     string  `json:"offload_path"`
	// MaxCompactionAttempts bounds how many times the harness re-requests a
	// summary that fails post-compaction verification before accepting it
	// as-is and emitting a warn-level event. Default 2. The harness loop that
	// consumes this is deferred (issue #3); the verifier itself does not read
	// this field.
	MaxCompactionAttempts uint32 `json:"max_compaction_attempts"`
	// ContextLength is an optional caller override for the resolved compaction
	// window (issue #141). When non-nil and *ContextLength > 0, the resolver
	// (StandardContextManager.ResolveContextLength) uses it as the
	// WindowLimit. A nil pointer (the default) and an explicit pointer to 0
	// both fall through to the model's Provider().ContextWindow, then to
	// DefaultContextLength. Configured values are NOT clamped to the model's
	// real window.
	//
	// A *uint32 sentinel distinguishes absent (nil) from an explicit 0; both
	// fall through, but the pointer keeps the absent case serialized as a
	// MISSING key (omitempty), so an existing serialized CompactionConfig
	// stays byte-identical (no new key when unset).
	ContextLength *uint32 `json:"context_length,omitempty"`
}

// DefaultCompactionConfig returns the spec defaults.
func DefaultCompactionConfig() CompactionConfig {
	return CompactionConfig{
		Threshold:             0.80,
		PreserveRecentN:       8,
		HeadTailTokens:        512,
		OffloadPath:           ".spore/offload",
		MaxCompactionAttempts: 2,
	}
}

// SessionState is the canonical per-session bag of context inputs owned by
// ContextManager. Distinct from sporecore.SessionState (the harness-side
// opaque pause/resume bag).
type SessionState struct {
	SessionID               sporecore.SessionID `json:"session_id"`
	TaskID                  sporecore.TaskID    `json:"task_id"`
	TurnNumber              uint32              `json:"turn_number"`
	TaskInstruction         string              `json:"task_instruction"`
	Environment             string              `json:"environment"`
	PriorState              string              `json:"prior_state"`
	OperationalInstructions string              `json:"operational_instructions"`
	ActivePhase             sporecore.TaskPhase `json:"active_phase"`
	MessageHistory          []sporecore.Message `json:"message_history"`
	TokenBudgetUsed         uint32              `json:"token_budget_used"`
	WindowLimit             uint32              `json:"window_limit"`
	GuidesLoaded            []GuideID           `json:"guides_loaded"`
	// PendingSkillInjections are skills to inject in Block 3 on the next
	// Assemble. Cleared after each assemble — skills are ephemeral, one turn
	// only.
	PendingSkillInjections []Guide `json:"pending_skill_injections"`
	BudgetWarningActive    bool    `json:"budget_warning_active"`
	// OpenProblems feed the keep_open_problems hint (issue #47).
	OpenProblems []string `json:"open_problems"`
	// ArchitecturalDecisions feed the keep_architectural_decisions hint (#47).
	ArchitecturalDecisions []string `json:"architectural_decisions"`
	// RecentFiles feed the keep_recent_file_list hint (#47). Typed as []string,
	// not a path type — keeps tokenization byte-identical across languages (no
	// per-language path semantics).
	RecentFiles []string `json:"recent_files"`
	// ReasoningSummary feeds the keep_thinking_blocks hint (issue #47).
	ReasoningSummary string `json:"reasoning_summary"`
}

// NewSessionState constructs a SessionState with spec defaults.
//
// WindowLimit defaults to the conservative DefaultContextLength (8K) rather
// than the old hardcoded 200K (issue #141): when the real context length is
// unknown, assume a tight window so compaction still fires. The manager's
// SeedSession overrides this with the resolved window for a real run.
func NewSessionState(sessionID sporecore.SessionID, taskID sporecore.TaskID, instruction string) SessionState {
	return SessionState{
		SessionID:       sessionID,
		TaskID:          taskID,
		TaskInstruction: instruction,
		ActivePhase:     sporecore.PhaseExecution,
		WindowLimit:     DefaultContextLength,
	}
}

// ContextSources bundles per-turn inputs supplied by sibling components.
type ContextSources struct {
	Guides         []Guide                `json:"guides"`
	Memory         []MemoryItem           `json:"memory"`
	ToolSchemas    []sporecore.ToolSchema `json:"tool_schemas"`
	ComposedPrompt ComposedPrompt         `json:"composed_prompt"`
}

// CompactionRequest is the prepared-but-not-yet-executed compaction proposal.
type CompactionRequest struct {
	MessagesToCompact []sporecore.Message     `json:"messages_to_compact"`
	PreserveHints     CompactionPreserveHints `json:"preserve_hints"`
}

// CompactionPreserveHints tell the compaction summariser what must survive.
type CompactionPreserveHints struct {
	KeepArchitecturalDecisions bool `json:"keep_architectural_decisions"`
	KeepOpenProblems           bool `json:"keep_open_problems"`
	KeepCurrentTaskState       bool `json:"keep_current_task_state"`
	KeepRecentFileList         bool `json:"keep_recent_file_list"`
	// KeepThinkingBlocks defaults to true — never compact active reasoning.
	KeepThinkingBlocks bool `json:"keep_thinking_blocks"`
}

// DefaultPreserveHints returns the spec defaults (all true).
func DefaultPreserveHints() CompactionPreserveHints {
	return CompactionPreserveHints{
		KeepArchitecturalDecisions: true,
		KeepOpenProblems:           true,
		KeepCurrentTaskState:       true,
		KeepRecentFileList:         true,
		KeepThinkingBlocks:         true,
	}
}

// CompactionResult is the outcome of an executed compaction.
type CompactionResult struct {
	SummaryMessage  sporecore.Message `json:"summary_message"`
	TokensReclaimed uint32            `json:"tokens_reclaimed"`
	MessagesRemoved uint32            `json:"messages_removed"`
}

// ============================================================================
// Post-compaction verification (issue #29)
// ============================================================================

// CompactionVerificationResult is the outcome of a CompactionVerifier check.
type CompactionVerificationResult struct {
	Passed bool `json:"passed"`
	// MissingItems are preservation terms not found in the summary, in
	// first-occurrence order (already lowercased/normalized).
	MissingItems []string `json:"missing_items"`
	Detail       string   `json:"detail"`
}

// CompactionVerifier is a lightweight, synchronous post-compaction sensor. It
// runs after the agent produces a summary and before the harness accepts it.
// Implementations are purely computational and MUST NOT call the model — hence
// no context.Context parameter.
type CompactionVerifier interface {
	Verify(summary string, hints CompactionPreserveHints, sessionState *SessionState) CompactionVerificationResult
}

// keyTermSplit matches runs of any character that is not an ASCII lowercase
// letter or digit; used to tokenize a lowercased source string.
var keyTermSplit = regexp.MustCompile("[^a-z0-9]+")

// KeyTermVerifier is the standard CompactionVerifier. It extracts key terms
// from the session state per the enabled hints and checks they appear in the
// summary.
//
// All five hints contribute source terms, each gated on its hint and pushed in
// this fixed order (issue #47) — this order is the cross-language invariant that
// determines first-occurrence dedup:
//
//  1. KeepCurrentTaskState       → SessionState.TaskInstruction
//  2. KeepOpenProblems           → each SessionState.OpenProblems element
//  3. KeepArchitecturalDecisions → each SessionState.ArchitecturalDecisions element
//  4. KeepRecentFileList         → each SessionState.RecentFiles element
//  5. KeepThinkingBlocks         → SessionState.ReasoningSummary
//
// Each source string runs through the same extractTerms rule; an empty/unset
// field contributes no terms.
type KeyTermVerifier struct{}

// extractTerms tokenizes a source: lowercase, split on runs of non-[a-z0-9],
// drop empty tokens and tokens shorter than 4 chars.
func extractTerms(source string) []string {
	lower := strings.ToLower(source)
	var terms []string
	for _, tok := range keyTermSplit.Split(lower, -1) {
		if len(tok) < 4 {
			continue
		}
		terms = append(terms, tok)
	}
	return terms
}

// Verify implements CompactionVerifier.
func (KeyTermVerifier) Verify(summary string, hints CompactionPreserveHints, sessionState *SessionState) CompactionVerificationResult {
	// Step 1: collect source strings from enabled hints, each gated on its hint
	// and pushed in this fixed order (issue #47). This order is the
	// cross-language invariant that determines first-occurrence dedup.
	var sources []string
	if sessionState != nil {
		if hints.KeepCurrentTaskState {
			sources = append(sources, sessionState.TaskInstruction)
		}
		if hints.KeepOpenProblems {
			sources = append(sources, sessionState.OpenProblems...)
		}
		if hints.KeepArchitecturalDecisions {
			sources = append(sources, sessionState.ArchitecturalDecisions...)
		}
		if hints.KeepRecentFileList {
			sources = append(sources, sessionState.RecentFiles...)
		}
		if hints.KeepThinkingBlocks {
			sources = append(sources, sessionState.ReasoningSummary)
		}
	}

	// Step 2: build the term list, deduping preserving first-occurrence order.
	var terms []string
	seen := make(map[string]struct{})
	for _, source := range sources {
		for _, term := range extractTerms(source) {
			if _, ok := seen[term]; ok {
				continue
			}
			seen[term] = struct{}{}
			terms = append(terms, term)
		}
	}

	// Step 3: a term is present iff the lowercased summary contains it.
	// Initialize non-nil so JSON marshals to [] not null.
	summaryLower := strings.ToLower(summary)
	missing := []string{}
	for _, term := range terms {
		if !strings.Contains(summaryLower, term) {
			missing = append(missing, term)
		}
	}

	// Steps 4 + 5.
	total := len(terms)
	passed := len(missing) == 0
	var detail string
	if passed {
		detail = fmt.Sprintf("all %d key term(s) present", total)
	} else {
		detail = fmt.Sprintf("missing %d of %d key term(s): %s", len(missing), total, strings.Join(missing, ", "))
	}

	return CompactionVerificationResult{
		Passed:       passed,
		MissingItems: missing,
		Detail:       detail,
	}
}

// Compile-time interface check.
var _ CompactionVerifier = KeyTermVerifier{}

// ============================================================================
// Errors
// ============================================================================

// ContextErrorKind discriminates ContextError variants.
type ContextErrorKind string

const (
	// ErrKindTokenCountFailed — ModelInterface.CountTokens returned an error.
	ErrKindTokenCountFailed ContextErrorKind = "token_count_failed"
	// ErrKindCompactionFailed — compaction could not be prepared/applied.
	ErrKindCompactionFailed ContextErrorKind = "compaction_failed"
	// ErrKindAssemblyFailed — assemble could not complete.
	ErrKindAssemblyFailed ContextErrorKind = "assembly_failed"
	// ErrKindCacheHashMismatch — a Block 1/2 hash unexpectedly changed.
	ErrKindCacheHashMismatch ContextErrorKind = "cache_hash_mismatch"
)

// ContextError is the typed error returned by ContextManager methods.
//
// CacheHashMismatch carries the offending CacheBlock (not a raw string) plus
// the turn on which the mismatch was detected. Both Block 1
// (CacheBlockStatic) and Block 2 (CacheBlockPerSession) halt the run on a
// mid-session mismatch — they are treated consistently (issue #32). A Block-2
// change mid-session means session-stable content mutated and every
// subsequent turn would silently pay full input-token cost; rather than warn,
// the run stops so the caller can fix the source. Block 2 only halts when
// TurnNumber > 1 (the turn-1 assemble records the baseline). Estimated
// cache-cost-delta tracking (UnexpectedMiss) is a separate observability
// concern tracked in issue #90.
type ContextError struct {
	Kind   ContextErrorKind `json:"kind"`
	Reason string           `json:"reason,omitempty"`
	// Block is the cache block whose content hash unexpectedly changed
	// (CacheHashMismatch only). Typed as promptchunkregistry.CacheBlock.
	Block      promptchunkregistry.CacheBlock `json:"block,omitempty"`
	Expected   uint64                         `json:"expected,omitempty"`
	Actual     uint64                         `json:"actual,omitempty"`
	TurnNumber uint32                         `json:"turn_number,omitempty"`
	Err        error                          `json:"-"`
}

// Error implements error.
func (e *ContextError) Error() string {
	switch e.Kind {
	case ErrKindTokenCountFailed:
		if e.Err != nil {
			return fmt.Sprintf("token count failed: %s", e.Err)
		}
		return "token count failed"
	case ErrKindCompactionFailed:
		return fmt.Sprintf("compaction failed: %s", e.Reason)
	case ErrKindAssemblyFailed:
		return fmt.Sprintf("assembly failed: %s", e.Reason)
	case ErrKindCacheHashMismatch:
		return fmt.Sprintf("cache hash mismatch on block %s at turn %d: expected %d, got %d", e.Block, e.TurnNumber, e.Expected, e.Actual)
	default:
		return fmt.Sprintf("context error: %s", e.Kind)
	}
}

// Unwrap returns the underlying cause when present.
func (e *ContextError) Unwrap() error { return e.Err }

// Typed constructors.
func newTokenCountFailed(cause error) *ContextError {
	return &ContextError{Kind: ErrKindTokenCountFailed, Err: cause}
}
func newCompactionFailed(reason string) *ContextError {
	return &ContextError{Kind: ErrKindCompactionFailed, Reason: reason}
}
func newCacheHashMismatch(block promptchunkregistry.CacheBlock, expected, actual uint64, turnNumber uint32) *ContextError {
	return &ContextError{Kind: ErrKindCacheHashMismatch, Block: block, Expected: expected, Actual: actual, TurnNumber: turnNumber}
}

// ============================================================================
// Interface
// ============================================================================

// ContextManager is the canonical issue-#7 interface.
type ContextManager interface {
	Assemble(ctx context.Context, state *SessionState, sources *ContextSources) (*Context, error)
	AppendToolResult(ctx context.Context, state *SessionState, result *sporecore.HarnessToolResult, sandbox sporecore.SandboxProvider) error
	AppendResponse(state *SessionState, response string)
	ShouldCompact(state *SessionState) bool
	PrepareCompaction(state *SessionState) (*CompactionRequest, error)
	ApplyCompaction(state *SessionState, result CompactionResult) error
	InjectSkill(ctx *Context, skill *Guide) error
	RecordCacheResult(ctx *Context, stats CacheBlockHits)
}

// ============================================================================
// StandardContextManager
// ============================================================================

// StandardContextManager is the reference implementation. It enforces the
// spec assembly order: Block 1 from ComposedPrompt, Block 2 from
// SessionState, Block 3 from per-turn ephemera. Tool schemas are sorted by
// name. Token counts come from ModelInterface.CountTokens, never estimated.
type StandardContextManager struct {
	model                 sporecore.ModelInterface
	cacheProvider         CacheProvider
	compaction            CompactionConfig
	offloadThresholdBytes int

	mu          sync.Mutex
	staticHash  *uint64
	sessionHash *uint64
}

// NewStandardContextManager constructs a StandardContextManager.
func NewStandardContextManager(
	model sporecore.ModelInterface,
	cacheProvider CacheProvider,
	compaction CompactionConfig,
) *StandardContextManager {
	if cacheProvider == nil {
		cacheProvider = NullCacheProvider{}
	}
	return &StandardContextManager{
		model:                 model,
		cacheProvider:         cacheProvider,
		compaction:            compaction,
		offloadThresholdBytes: 32 * 1024,
	}
}

// WithOffloadThreshold returns a copy with the given offload threshold (in
// bytes). Inputs whose tool-result text exceeds this size will be passed
// through SandboxProvider.HandleLargeOutput.
func (m *StandardContextManager) WithOffloadThreshold(bytes int) *StandardContextManager {
	m.offloadThresholdBytes = bytes
	return m
}

// ResolveContextLength resolves the compaction window (issue #141). Fallback
// order:
//
//  1. the configured CompactionConfig.ContextLength when non-nil and > 0,
//  2. else the model's Provider().ContextWindow when > 0,
//  3. else DefaultContextLength (8K).
//
// A nil pointer (and an explicit pointer to 0) falls through to model
// metadata, then to the default. The configured value is NOT clamped to the
// model's real window — a larger configured value is used as-is.
func (m *StandardContextManager) ResolveContextLength() uint32 {
	if cl := m.compaction.ContextLength; cl != nil && *cl > 0 {
		return *cl
	}
	if m.model != nil {
		if window := m.model.Provider().ContextWindow; window > 0 {
			return window
		}
	}
	return DefaultContextLength
}

// SeedSession builds the initial SessionState for a run, seeding its
// WindowLimit from ResolveContextLength (issue #141).
//
// The manager owns seeding so the resolved window has a single production
// seam — callers get a SessionState whose WindowLimit already reflects the
// configured/model/default resolution rather than the bare NewSessionState
// constructor default.
func (m *StandardContextManager) SeedSession(sessionID sporecore.SessionID, taskID sporecore.TaskID, instruction string) SessionState {
	state := NewSessionState(sessionID, taskID, instruction)
	state.WindowLimit = m.ResolveContextLength()
	return state
}

// Assemble builds the per-turn Context.
func (m *StandardContextManager) Assemble(
	ctx context.Context,
	state *SessionState,
	sources *ContextSources,
) (*Context, error) {
	// ── BLOCK 1 hash check ──────────────────────────────────────────────
	staticHash := sources.ComposedPrompt.Block1Hash
	m.mu.Lock()
	if m.staticHash != nil {
		if *m.staticHash != staticHash {
			prev := *m.staticHash
			m.mu.Unlock()
			return nil, newCacheHashMismatch(promptchunkregistry.CacheBlockStatic, prev, staticHash, state.TurnNumber)
		}
	} else {
		h := staticHash
		m.staticHash = &h
	}
	m.mu.Unlock()

	// ── BLOCK 2 (PerSession) ─────────────────────────────────────────────
	segments := m.buildSessionSegments(state)
	sessionHash := segmentsHash(segments)
	m.mu.Lock()
	if m.sessionHash != nil && *m.sessionHash != sessionHash && state.TurnNumber > 1 {
		// Block 2 mid-session change: same as Block 1, halt the run rather than
		// warn (issue #32). The turn-1 assemble (guarded by TurnNumber > 1)
		// records the baseline and never halts.
		prev := *m.sessionHash
		m.mu.Unlock()
		return nil, newCacheHashMismatch(promptchunkregistry.CacheBlockPerSession, prev, sessionHash, state.TurnNumber)
	}
	{
		h := sessionHash
		m.sessionHash = &h
	}
	m.mu.Unlock()

	// ── BLOCK 3 (PerTurn, never cached) ──────────────────────────────────
	if state.BudgetWarningActive {
		segments = append(segments, PromptSegment{
			Name:      "budget_warning",
			Content:   fmt.Sprintf("[BUDGET] %d of %d tokens used.", state.TokenBudgetUsed, state.WindowLimit),
			Stability: StabilityPerTurn,
		})
	}
	for _, skill := range state.PendingSkillInjections {
		segments = append(segments, PromptSegment{
			Name:      fmt.Sprintf("skill:%s", skill.ID),
			Content:   skill.Content,
			Stability: StabilityPerTurn,
		})
	}

	// ── Render ───────────────────────────────────────────────────────────
	rendered, breakpoints := renderSegments(sources.ComposedPrompt.Rendered, segments)
	systemPrompt := RenderedSystemPrompt{
		Content:          rendered,
		Breakpoints:      breakpoints,
		StaticBlockHash:  staticHash,
		SessionBlockHash: sessionHash,
	}

	// ── Tool schemas: sort by name (cache stability) ─────────────────────
	toolSchemas := make([]sporecore.ToolSchema, len(sources.ToolSchemas))
	copy(toolSchemas, sources.ToolSchemas)
	sort.SliceStable(toolSchemas, func(i, j int) bool {
		return toolSchemas[i].Name < toolSchemas[j].Name
	})

	// ── Message history ──────────────────────────────────────────────────
	messages := make([]sporecore.Message, len(state.MessageHistory))
	copy(messages, state.MessageHistory)

	// ── Token count (from ModelInterface, not estimated) ─────────────────
	reqMsgs := make([]sporecore.Message, 0, len(messages)+1)
	reqMsgs = append(reqMsgs, sporecore.Message{
		Role:    sporecore.RoleSystem,
		Content: sporecore.NewTextContent(systemPrompt.Content),
	})
	reqMsgs = append(reqMsgs, messages...)
	req := sporecore.ModelRequest{
		Messages: reqMsgs,
		Tools:    toolSchemas,
		Params:   sporecore.ModelParams{},
		Stream:   false,
	}
	tokenCount, err := m.model.CountTokens(ctx, req)
	if err != nil {
		return nil, newTokenCountFailed(err)
	}

	var utilization float32
	if state.WindowLimit > 0 {
		utilization = float32(tokenCount) / float32(state.WindowLimit)
	}

	meta := ContextMeta{
		SessionID:    state.SessionID,
		TurnNumber:   state.TurnNumber,
		ActivePhase:  state.ActivePhase,
		GuidesLoaded: append([]GuideID(nil), state.GuidesLoaded...),
		SkillsInjected: func() []GuideID {
			ids := make([]GuideID, 0, len(state.PendingSkillInjections))
			for _, g := range state.PendingSkillInjections {
				ids = append(ids, g.ID)
			}
			return ids
		}(),
	}

	out := &Context{
		SystemPrompt: systemPrompt,
		Messages:     messages,
		ToolSchemas:  toolSchemas,
		TokenCount:   tokenCount,
		WindowLimit:  state.WindowLimit,
		Utilization:  utilization,
		Meta:         meta,
	}

	_ = m.cacheProvider.Annotate(out)
	return out, nil
}

// AppendToolResult appends a tool message to the session, head+tail
// truncating via SandboxProvider.HandleLargeOutput when the result text
// exceeds the configured threshold.
func (m *StandardContextManager) AppendToolResult(
	ctx context.Context,
	state *SessionState,
	result *sporecore.HarnessToolResult,
	sandbox sporecore.SandboxProvider,
) error {
	var text string
	switch result.Output.Kind {
	case sporecore.ToolOutputSuccess:
		text = result.Output.Content
	case sporecore.ToolOutputError:
		text = fmt.Sprintf("[error] %s", result.Output.Message)
	case sporecore.ToolOutputWaitingForHuman:
		text = "[waiting]"
	default:
		text = ""
	}

	finalText := text
	if len(text) > m.offloadThresholdBytes && sandbox != nil {
		truncated := sandbox.HandleLargeOutput(
			ctx, text, result.CallID,
			m.compaction.HeadTailTokens, m.compaction.HeadTailTokens,
		)
		finalText = formatTruncated(truncated.Content, truncated.FullRef)
	}

	state.MessageHistory = append(state.MessageHistory, sporecore.Message{
		Role:    sporecore.RoleTool,
		Content: sporecore.NewTextContent(finalText),
	})
	return nil
}

// AppendResponse appends an Assistant message to the session.
func (m *StandardContextManager) AppendResponse(state *SessionState, response string) {
	state.MessageHistory = append(state.MessageHistory, sporecore.Message{
		Role:    sporecore.RoleAssistant,
		Content: sporecore.NewTextContent(response),
	})
}

// ShouldCompact reports whether the session has crossed the compaction
// threshold (default 80%).
func (m *StandardContextManager) ShouldCompact(state *SessionState) bool {
	if state.WindowLimit == 0 {
		return false
	}
	util := float32(state.TokenBudgetUsed) / float32(state.WindowLimit)
	return util >= m.compaction.Threshold
}

// PrepareCompaction slices off all but the last PreserveRecentN messages and
// returns a CompactionRequest with default preserve hints.
func (m *StandardContextManager) PrepareCompaction(state *SessionState) (*CompactionRequest, error) {
	n := len(state.MessageHistory)
	keep := int(m.compaction.PreserveRecentN)
	if n <= keep {
		return &CompactionRequest{
			MessagesToCompact: nil,
			PreserveHints:     DefaultPreserveHints(),
		}, nil
	}
	cut := n - keep
	msgs := make([]sporecore.Message, cut)
	copy(msgs, state.MessageHistory[:cut])
	return &CompactionRequest{
		MessagesToCompact: msgs,
		PreserveHints:     DefaultPreserveHints(),
	}, nil
}

// ApplyCompaction replaces the old slice with summary + last PreserveRecentN
// messages, decrementing TokenBudgetUsed by tokens_reclaimed (saturating).
func (m *StandardContextManager) ApplyCompaction(state *SessionState, result CompactionResult) error {
	n := len(state.MessageHistory)
	keep := int(m.compaction.PreserveRecentN)
	if n <= keep {
		return newCompactionFailed("history shorter than preserve_recent_n")
	}
	cut := n - keep
	newHistory := make([]sporecore.Message, 0, keep+1)
	newHistory = append(newHistory, result.SummaryMessage)
	newHistory = append(newHistory, state.MessageHistory[cut:]...)
	state.MessageHistory = newHistory
	if result.TokensReclaimed >= state.TokenBudgetUsed {
		state.TokenBudgetUsed = 0
	} else {
		state.TokenBudgetUsed -= result.TokensReclaimed
	}
	return nil
}

// InjectSkill performs an ephemeral Block-3 injection: appends to the system
// prompt content but does NOT modify the message history and does NOT change
// Block 1 or Block 2 hashes.
func (m *StandardContextManager) InjectSkill(c *Context, skill *Guide) error {
	if c == nil || skill == nil {
		return &ContextError{Kind: ErrKindAssemblyFailed, Reason: "nil context or skill"}
	}
	content := c.SystemPrompt.Content
	if len(content) > 0 && content[len(content)-1] != '\n' {
		content += "\n"
	}
	content += fmt.Sprintf("[SKILL:%s]\n%s", skill.ID, skill.Content)
	c.SystemPrompt.Content = content
	c.Meta.SkillsInjected = append(c.Meta.SkillsInjected, skill.ID)
	return nil
}

// RecordCacheResult updates ContextMeta.CacheBlocks with the supplied stats.
func (m *StandardContextManager) RecordCacheResult(c *Context, stats CacheBlockHits) {
	c.Meta.CacheBlocks = CacheBlockStatus{
		StaticHit:  stats.StaticHit,
		SessionHit: stats.SessionHit,
		HistoryHit: stats.HistoryHit,
	}
}

// ── helpers ─────────────────────────────────────────────────────────────────

func (m *StandardContextManager) buildSessionSegments(state *SessionState) []PromptSegment {
	// Order is load-bearing for prefix-cache stability.
	return []PromptSegment{
		{Name: "task", Content: state.TaskInstruction, Stability: StabilityPerSession},
		{Name: "environment", Content: state.Environment, Stability: StabilityPerSession},
		{Name: "prior_state", Content: state.PriorState, Stability: StabilityPerSession},
		{Name: "operational", Content: state.OperationalInstructions, Stability: StabilityPerSession, CacheBreakpoint: true},
	}
}

func segmentsHash(segments []PromptSegment) uint64 {
	h := fnv.New64a()
	for _, s := range segments {
		_, _ = h.Write([]byte(s.Name))
		h.Write([]byte{0})
		_, _ = h.Write([]byte(s.Content))
		h.Write([]byte{0})
		_, _ = h.Write([]byte(s.Stability))
		h.Write([]byte{0})
		if s.CacheBreakpoint {
			h.Write([]byte{1})
		} else {
			h.Write([]byte{0})
		}
	}
	return h.Sum64()
}

// renderSegments concatenates Block 1 then each segment, returning the
// rendered content and the list of cache breakpoints. Block 1 always ends
// with an implicit breakpoint (the CacheProvider inserts the marker).
func renderSegments(block1 string, segments []PromptSegment) (string, []BreakpointInfo) {
	content := block1
	breakpoints := []BreakpointInfo{
		{AfterSegment: "__block_1__", TokenOffset: uint32(len(content) / 4)},
	}
	for _, seg := range segments {
		if len(content) > 0 && content[len(content)-1] != '\n' {
			content += "\n"
		}
		content += seg.Content
		if seg.CacheBreakpoint {
			breakpoints = append(breakpoints, BreakpointInfo{
				AfterSegment: seg.Name,
				TokenOffset:  uint32(len(content) / 4),
			})
		}
	}
	return content, breakpoints
}

func formatTruncated(headTail string, fullRef *sporecore.FileRef) string {
	if fullRef != nil {
		return fmt.Sprintf("%s\n\n[truncated; full output at %s (%d bytes)]", headTail, fullRef.Path, fullRef.ByteLen)
	}
	return fmt.Sprintf("%s\n\n[truncated]", headTail)
}

// Compile-time interface check.
var _ ContextManager = (*StandardContextManager)(nil)
