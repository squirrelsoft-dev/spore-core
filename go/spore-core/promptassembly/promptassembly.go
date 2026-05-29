// Package promptassembly — issue #79 `PromptAssemblyEngine`: conditional,
// provider-sourced prompt assembly that extends (does not replace) the #24
// `promptchunkregistry`.
//
// The shipped #24 `promptchunkregistry` package composes a static Block-1
// ComposedPrompt once at construction. This package builds *on top of* it:
// chunks are loaded from pluggable ChunkProviders and included conditionally
// based on mode, active tools, phase, agent type, trigger words, hook events,
// or arbitrary architect-defined predicates. The Static bucket is folded into
// a #24 ComposedPrompt (Block 1); PerSession / PerTurn chunks flow through the
// existing composed-prompt / PromptSegment machinery in `contextmgr`
// (decision A4 — no new public segment vectors on contextmgr.ContextSources).
//
// This package owns its OWN PromptChunk and ChunkProviderError — distinct from
// the #24 promptchunkregistry.PromptChunk / ChunkError, which are left
// untouched (decision A1). It is also the home of the minimal shared
// StorageScope enum (decision A2).
//
// # Import-cycle note
//
// The reused types (Mode, SegmentStability, ComposedPrompt, PromptSegment,
// ContextSources, Guide, MemoryItem) live in the `contextmgr` and
// `promptchunkregistry` subpackages, both of which import the root `sporecore`
// package. This package therefore lives in its own subpackage and imports
// those siblings plus root `sporecore`; the root package never imports back, so
// no cycle arises (the same pattern #80/#81 used for local newtypes).
//
// # Rules enforced (cross-language parity with the Rust reference)
//
//   - R1  Always always matches.
//   - R2  WhenMode(m) iff ctx.Mode == m.
//   - R3  WhenToolActive(t) iff ctx.ActiveToolNames contains t.
//   - R4  WhenToolCapability(t,c) iff ctx.ActiveCapabilities contains (t,c).
//   - R5  WhenPhase / WhenAgentType / WhenFeature (a feature matches iff present
//     in ctx.Features AND its value is true).
//   - R6  OnTrigger(words) iff some word is a substring of ctx.IncomingMessage
//     (an absent message never matches).
//   - R7  OnEvent(e) iff ctx.PendingEvents contains e.
//   - R8  All / Any / Not compose their children.
//   - R9  Custom(f) is evaluated by calling f(ctx).
//   - R10 Chunks are bucketed by SegmentStability (Static / PerSession / PerTurn).
//   - R11 Registration order is preserved within a bucket.
//   - R12 A ToolAffinity chunk is included iff its tool is active AND
//     (capability is empty OR that capability is active).
//   - R13 OnTrigger chunks whose trigger matches the incoming message are routed
//     to the PerTurn bucket.
//   - R14 OnEvent chunks are injected into PerTurn only when their event is pending.
//   - R15 The Block-1 hash is stable across two builds of an identical Static set.
//   - R16 cache_breakpoint injects a breakpoint after the chunk.
//   - R17 A tool that is not active yields no description chunk.
//   - R18 EmbeddedChunkProvider invalidate is a no-op; load returns the same set.
//   - R19 InMemoryChunkProvider returns the registered set; Set replaces it.
//   - R21 CompositeChunkProvider merges children in add order and propagates Invalidate.
//   - R25 HarnessBuilder defaults to an empty InMemoryChunkProvider; ChunkProvider /
//     Chunks override it.
//
// # A3 — Custom is first-class but unserialized
//
// ChunkConditionCustom wraps a func(*AssemblyContext) bool. It is the PRIMARY
// escape hatch for conditions that cannot be expressed with the serializable
// variants, and it is fully supported in the public API. However it CANNOT
// serialize and has no comparable identity, so:
//   - MarshalJSON emits null for a Custom node (it is omitted from the wire
//     form); UnmarshalJSON can therefore never produce a Custom.
//   - When marshalling All/Any/Not, Custom children are pruned from the wire
//     form (matching the Rust reference).
//   - Equal treats Custom as NEVER equal to anything (including another Custom).
//   - Custom is excluded from the shared byte-identical fixtures. Architects who
//     reach for Custom knowingly opt that chunk out of the cross-language
//     byte-identical contract — a deliberate, supported choice.
package promptassembly

import (
	"context"
	"encoding/json"
	"fmt"
	"sort"
	"strings"
	"sync"

	sporecore "github.com/squirrelsoft-dev/spore-core/go/spore-core"
	"github.com/squirrelsoft-dev/spore-core/go/spore-core/contextmgr"
	"github.com/squirrelsoft-dev/spore-core/go/spore-core/promptchunkregistry"
)

// ============================================================================
// StorageScope (A2)
// ============================================================================

// StorageScope is the minimal shared storage scope. This package is its home
// (decision A2); the scope-aware FileSystemChunkProvider that consumes it is
// deferred (A6). The zero value is StorageScopeProject.
type StorageScope string

const (
	StorageScopeUser    StorageScope = "user"
	StorageScopeProject StorageScope = "project"
	StorageScopeLocal   StorageScope = "local"
)

// DefaultStorageScope is the scope used when none is set (matches the Rust
// default of Project).
const DefaultStorageScope = StorageScopeProject

// ============================================================================
// ToolAffinity
// ============================================================================

// ToolAffinity binds a chunk to a tool (and optionally a sub-capability). The
// builder includes the chunk only when the tool — and capability, if any — is
// active. An empty Capability means "any capability of this tool".
type ToolAffinity struct {
	ToolName   string `json:"tool_name"`
	Capability string `json:"capability,omitempty"`
}

// NewToolAffinity builds a tool-only affinity (no capability gate).
func NewToolAffinity(toolName string) ToolAffinity {
	return ToolAffinity{ToolName: toolName}
}

// NewToolCapabilityAffinity builds an affinity gated on a tool capability.
func NewToolCapabilityAffinity(toolName, capability string) ToolAffinity {
	return ToolAffinity{ToolName: toolName, Capability: capability}
}

// toolAffinityWire mirrors ToolAffinity but always emits `capability` (as null
// when empty) so the wire form matches the Rust reference / fixtures, where
// capability is `Option<String>` serialized as a present key.
type toolAffinityWire struct {
	ToolName   string  `json:"tool_name"`
	Capability *string `json:"capability"`
}

// MarshalJSON emits capability as an explicit null when empty.
func (a ToolAffinity) MarshalJSON() ([]byte, error) {
	w := toolAffinityWire{ToolName: a.ToolName}
	if a.Capability != "" {
		c := a.Capability
		w.Capability = &c
	}
	return json.Marshal(w)
}

// UnmarshalJSON accepts both a present-and-null and an absent capability.
func (a *ToolAffinity) UnmarshalJSON(data []byte) error {
	var w toolAffinityWire
	if err := json.Unmarshal(data, &w); err != nil {
		return err
	}
	a.ToolName = w.ToolName
	if w.Capability != nil {
		a.Capability = *w.Capability
	} else {
		a.Capability = ""
	}
	return nil
}

// ============================================================================
// ChunkCondition
// ============================================================================

// CustomCondition is the predicate type behind ChunkConditionCustom.
type CustomCondition func(*AssemblyContext) bool

// ConditionKind discriminates the ChunkCondition variants. It doubles as the
// serialized `type` tag for the variants that serialize.
type ConditionKind string

const (
	KindAlways             ConditionKind = "always"
	KindWhenMode           ConditionKind = "when_mode"
	KindWhenToolActive     ConditionKind = "when_tool_active"
	KindWhenToolCapability ConditionKind = "when_tool_capability"
	KindWhenPhase          ConditionKind = "when_phase"
	KindWhenAgentType      ConditionKind = "when_agent_type"
	KindWhenFeature        ConditionKind = "when_feature"
	KindOnTrigger          ConditionKind = "on_trigger"
	KindOnEvent            ConditionKind = "on_event"
	KindAll                ConditionKind = "all"
	KindAny                ConditionKind = "any"
	KindNot                ConditionKind = "not"
	// KindCustom never appears on the wire (A3). It exists so Kind() is total.
	KindCustom ConditionKind = "custom"
)

// ChunkCondition is the condition primitive tree. Architects compose these; the
// framework evaluates them against an AssemblyContext via
// ContextSourcesBuilder.Evaluate.
//
// A ChunkCondition is a tagged union: the Kind field selects which of the
// payload fields is meaningful. Construct values with the New* helpers rather
// than struct literals. All variants serialize EXCEPT Custom (see the
// package-level A3 note).
type ChunkCondition struct {
	kind ConditionKind

	mode       promptchunkregistry.Mode
	tool       string
	capability string
	phase      sporecore.TaskPhase
	agentType  string
	feature    string
	words      []string
	event      sporecore.HookEvent
	children   []ChunkCondition // All / Any
	inner      *ChunkCondition  // Not
	custom     CustomCondition
}

// Kind returns the variant tag of this condition.
func (c ChunkCondition) Kind() ConditionKind { return c.kind }

// --- Constructors -----------------------------------------------------------

// Always builds an unconditional condition (R1).
func Always() ChunkCondition { return ChunkCondition{kind: KindAlways} }

// WhenMode matches when the assembly context's mode equals m (R2).
func WhenMode(m promptchunkregistry.Mode) ChunkCondition {
	return ChunkCondition{kind: KindWhenMode, mode: m}
}

// WhenToolActive matches when the named tool is active (R3).
func WhenToolActive(tool string) ChunkCondition {
	return ChunkCondition{kind: KindWhenToolActive, tool: tool}
}

// WhenToolCapability matches when (tool, capability) is active (R4).
func WhenToolCapability(tool, capability string) ChunkCondition {
	return ChunkCondition{kind: KindWhenToolCapability, tool: tool, capability: capability}
}

// WhenPhase matches when the context phase equals phase (R5).
func WhenPhase(phase sporecore.TaskPhase) ChunkCondition {
	return ChunkCondition{kind: KindWhenPhase, phase: phase}
}

// WhenAgentType matches when the context agent type equals agentType (R5).
func WhenAgentType(agentType string) ChunkCondition {
	return ChunkCondition{kind: KindWhenAgentType, agentType: agentType}
}

// WhenFeature matches when the named feature is present AND true (R5).
func WhenFeature(feature string) ChunkCondition {
	return ChunkCondition{kind: KindWhenFeature, feature: feature}
}

// OnTrigger matches when any word is a substring of the incoming message (R6).
func OnTrigger(words []string) ChunkCondition {
	return ChunkCondition{kind: KindOnTrigger, words: words}
}

// OnEvent matches when the event is present in the context's pending events (R7).
func OnEvent(event sporecore.HookEvent) ChunkCondition {
	return ChunkCondition{kind: KindOnEvent, event: event}
}

// All matches when every child matches (R8).
func All(children ...ChunkCondition) ChunkCondition {
	return ChunkCondition{kind: KindAll, children: children}
}

// Any matches when at least one child matches (R8).
func Any(children ...ChunkCondition) ChunkCondition {
	return ChunkCondition{kind: KindAny, children: children}
}

// Not inverts its child (R8).
func Not(inner ChunkCondition) ChunkCondition {
	return ChunkCondition{kind: KindNot, inner: &inner}
}

// Custom wraps an arbitrary predicate (A3 / R9). First-class but never
// serialized and never equal under Equal.
func Custom(f CustomCondition) ChunkCondition {
	return ChunkCondition{kind: KindCustom, custom: f}
}

// --- Equality ---------------------------------------------------------------

// Equal reports structural equality. Custom is NEVER equal to anything,
// including another Custom (A3 — closure identity is not comparable).
func (c ChunkCondition) Equal(other ChunkCondition) bool {
	if c.kind != other.kind {
		return false
	}
	switch c.kind {
	case KindAlways:
		return true
	case KindWhenMode:
		return c.mode == other.mode
	case KindWhenToolActive:
		return c.tool == other.tool
	case KindWhenToolCapability:
		return c.tool == other.tool && c.capability == other.capability
	case KindWhenPhase:
		return c.phase == other.phase
	case KindWhenAgentType:
		return c.agentType == other.agentType
	case KindWhenFeature:
		return c.feature == other.feature
	case KindOnTrigger:
		return stringSliceEqual(c.words, other.words)
	case KindOnEvent:
		return c.event == other.event
	case KindAll, KindAny:
		if len(c.children) != len(other.children) {
			return false
		}
		for i := range c.children {
			if !c.children[i].Equal(other.children[i]) {
				return false
			}
		}
		return true
	case KindNot:
		if c.inner == nil || other.inner == nil {
			return c.inner == other.inner
		}
		return c.inner.Equal(*other.inner)
	case KindCustom:
		// Never equal (A3).
		return false
	default:
		return false
	}
}

func stringSliceEqual(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// --- Serialization (A3) -----------------------------------------------------
//
// Wire form is an internally-tagged object keyed on `type`. A Custom node
// marshals to JSON null and is pruned from All/Any/Not children; an
// unmarshalled null/absent condition yields Always (matching the Rust
// reference, decision A3 and the encoding/json default-handling discipline #81
// established).

// conditionWire is the serializable mirror of ChunkCondition (everything but
// Custom). Field presence depends on the `type` tag.
type conditionWire struct {
	Type       ConditionKind            `json:"type"`
	Mode       promptchunkregistry.Mode `json:"mode,omitempty"`
	Tool       string                   `json:"tool,omitempty"`
	Capability string                   `json:"capability,omitempty"`
	Phase      sporecore.TaskPhase      `json:"phase,omitempty"`
	AgentType  string                   `json:"agent_type,omitempty"`
	Feature    string                   `json:"feature,omitempty"`
	Words      []string                 `json:"words,omitempty"`
	Event      sporecore.HookEvent      `json:"event,omitempty"`
	Conditions []conditionWire          `json:"conditions,omitempty"`
	Condition  *conditionWire           `json:"condition,omitempty"`
}

// toWire converts a condition to its wire form, returning ok=false for a Custom
// node (which is omitted entirely). Custom children of All/Any are pruned; a Not
// wrapping a Custom is itself pruned.
func (c ChunkCondition) toWire() (conditionWire, bool) {
	switch c.kind {
	case KindAlways:
		return conditionWire{Type: KindAlways}, true
	case KindWhenMode:
		return conditionWire{Type: KindWhenMode, Mode: c.mode}, true
	case KindWhenToolActive:
		return conditionWire{Type: KindWhenToolActive, Tool: c.tool}, true
	case KindWhenToolCapability:
		return conditionWire{Type: KindWhenToolCapability, Tool: c.tool, Capability: c.capability}, true
	case KindWhenPhase:
		return conditionWire{Type: KindWhenPhase, Phase: c.phase}, true
	case KindWhenAgentType:
		return conditionWire{Type: KindWhenAgentType, AgentType: c.agentType}, true
	case KindWhenFeature:
		return conditionWire{Type: KindWhenFeature, Feature: c.feature}, true
	case KindOnTrigger:
		return conditionWire{Type: KindOnTrigger, Words: c.words}, true
	case KindOnEvent:
		return conditionWire{Type: KindOnEvent, Event: c.event}, true
	case KindAll:
		return conditionWire{Type: KindAll, Conditions: pruneWire(c.children)}, true
	case KindAny:
		return conditionWire{Type: KindAny, Conditions: pruneWire(c.children)}, true
	case KindNot:
		if c.inner == nil {
			return conditionWire{}, false
		}
		w, ok := c.inner.toWire()
		if !ok {
			return conditionWire{}, false
		}
		return conditionWire{Type: KindNot, Condition: &w}, true
	case KindCustom:
		return conditionWire{}, false
	default:
		return conditionWire{}, false
	}
}

// pruneWire converts children to wire form, dropping any Custom (or
// Custom-only) nodes.
func pruneWire(children []ChunkCondition) []conditionWire {
	out := make([]conditionWire, 0, len(children))
	for _, ch := range children {
		if w, ok := ch.toWire(); ok {
			out = append(out, w)
		}
	}
	return out
}

func (w conditionWire) toCondition() ChunkCondition {
	switch w.Type {
	case KindAlways:
		return Always()
	case KindWhenMode:
		return WhenMode(w.Mode)
	case KindWhenToolActive:
		return WhenToolActive(w.Tool)
	case KindWhenToolCapability:
		return WhenToolCapability(w.Tool, w.Capability)
	case KindWhenPhase:
		return WhenPhase(w.Phase)
	case KindWhenAgentType:
		return WhenAgentType(w.AgentType)
	case KindWhenFeature:
		return WhenFeature(w.Feature)
	case KindOnTrigger:
		return OnTrigger(w.Words)
	case KindOnEvent:
		return OnEvent(w.Event)
	case KindAll:
		return ChunkCondition{kind: KindAll, children: wiresToConditions(w.Conditions)}
	case KindAny:
		return ChunkCondition{kind: KindAny, children: wiresToConditions(w.Conditions)}
	case KindNot:
		if w.Condition == nil {
			return Always()
		}
		return Not(w.Condition.toCondition())
	default:
		// Unknown / absent tag -> Always default (A3).
		return Always()
	}
}

func wiresToConditions(ws []conditionWire) []ChunkCondition {
	out := make([]ChunkCondition, 0, len(ws))
	for _, w := range ws {
		out = append(out, w.toCondition())
	}
	return out
}

// MarshalJSON emits the internally-tagged wire form, or null for a Custom node
// (A3).
func (c ChunkCondition) MarshalJSON() ([]byte, error) {
	w, ok := c.toWire()
	if !ok {
		return []byte("null"), nil
	}
	return json.Marshal(w)
}

// UnmarshalJSON parses the internally-tagged wire form. A null or absent node
// deserializes to Always (A3).
func (c *ChunkCondition) UnmarshalJSON(data []byte) error {
	trimmed := strings.TrimSpace(string(data))
	if trimmed == "null" || trimmed == "" {
		*c = Always()
		return nil
	}
	var w conditionWire
	if err := json.Unmarshal(data, &w); err != nil {
		return err
	}
	if w.Type == "" {
		*c = Always()
		return nil
	}
	*c = w.toCondition()
	return nil
}

// ============================================================================
// PromptChunk (this package's own — distinct from #24, decision A1)
// ============================================================================

// PromptChunk is the unit of conditional assembly content. Distinct from the
// #24 promptchunkregistry.PromptChunk: this carries a ChunkCondition, triggers,
// affinities, and a stability bucket rather than a slot.
//
// A Custom condition is omitted from the wire form (A3) and round-trips back to
// the Always default.
type PromptChunk struct {
	ID              string                      `json:"id"`
	Content         string                      `json:"content"`
	Stability       contextmgr.SegmentStability `json:"stability"`
	Condition       ChunkCondition              `json:"condition"`
	Triggers        []string                    `json:"triggers,omitempty"`
	ToolAffinity    *ToolAffinity               `json:"tool_affinity,omitempty"`
	AgentAffinity   string                      `json:"agent_affinity,omitempty"`
	CacheBreakpoint bool                        `json:"cache_breakpoint,omitempty"`
}

// promptChunkWire decouples the required-field defaulting that encoding/json
// does not do natively (#81 discipline): an absent `condition` must default to
// Always, and an absent `stability` is invalid (caller-validated upstream).
type promptChunkWire struct {
	ID              string                      `json:"id"`
	Content         string                      `json:"content"`
	Stability       contextmgr.SegmentStability `json:"stability"`
	Condition       *ChunkCondition             `json:"condition"`
	Triggers        []string                    `json:"triggers"`
	ToolAffinity    *ToolAffinity               `json:"tool_affinity"`
	AgentAffinity   string                      `json:"agent_affinity"`
	CacheBreakpoint bool                        `json:"cache_breakpoint"`
}

// UnmarshalJSON defaults an absent/null condition to Always (A3) and tolerates
// absent optional fields.
func (p *PromptChunk) UnmarshalJSON(data []byte) error {
	var w promptChunkWire
	if err := json.Unmarshal(data, &w); err != nil {
		return err
	}
	p.ID = w.ID
	p.Content = w.Content
	p.Stability = w.Stability
	if w.Condition != nil {
		p.Condition = *w.Condition
	} else {
		p.Condition = Always()
	}
	p.Triggers = w.Triggers
	p.ToolAffinity = w.ToolAffinity
	p.AgentAffinity = w.AgentAffinity
	p.CacheBreakpoint = w.CacheBreakpoint
	return nil
}

// NewPromptChunk builds a Static, Always chunk — the common case.
func NewPromptChunk(id, content string) PromptChunk {
	return PromptChunk{
		ID:        id,
		Content:   content,
		Stability: contextmgr.StabilityStatic,
		Condition: Always(),
	}
}

// WithStability sets the chunk's stability bucket.
func (p PromptChunk) WithStability(s contextmgr.SegmentStability) PromptChunk {
	p.Stability = s
	return p
}

// WithCondition sets the chunk's condition.
func (p PromptChunk) WithCondition(c ChunkCondition) PromptChunk {
	p.Condition = c
	return p
}

// WithTriggers sets the chunk's trigger words.
func (p PromptChunk) WithTriggers(triggers []string) PromptChunk {
	p.Triggers = triggers
	return p
}

// WithToolAffinity sets the chunk's tool affinity gate.
func (p PromptChunk) WithToolAffinity(a ToolAffinity) PromptChunk {
	p.ToolAffinity = &a
	return p
}

// WithAgentAffinity sets the chunk's agent affinity gate.
func (p PromptChunk) WithAgentAffinity(agentType string) PromptChunk {
	p.AgentAffinity = agentType
	return p
}

// WithCacheBreakpoint sets the chunk's cache-breakpoint flag.
func (p PromptChunk) WithCacheBreakpoint(b bool) PromptChunk {
	p.CacheBreakpoint = b
	return p
}

// ============================================================================
// AssemblyContext
// ============================================================================

// capabilityKey is the comparable map key for an (tool, capability) pair,
// mirroring Rust's HashSet<(String, String)>.
type capabilityKey struct {
	Tool       string
	Capability string
}

// AssemblyContext carries the per-assembly inputs the framework populates before
// each assembly. Custom conditions read from it; Features is the escape hatch
// for architect-defined flags.
type AssemblyContext struct {
	SessionID          sporecore.SessionID
	TaskID             sporecore.TaskID
	TurnNumber         uint32
	Mode               promptchunkregistry.Mode
	Phase              sporecore.TaskPhase
	AgentType          string
	ActiveToolNames    map[string]struct{}
	ActiveCapabilities map[capabilityKey]struct{}
	IncomingMessage    *string
	PendingEvents      []sporecore.HookEvent
	Features           map[string]bool
	StorageScope       StorageScope
}

// NewAssemblyContext constructs a minimal context with empty collections and the
// default storage scope.
func NewAssemblyContext(
	sessionID sporecore.SessionID,
	taskID sporecore.TaskID,
	turnNumber uint32,
	mode promptchunkregistry.Mode,
	phase sporecore.TaskPhase,
) AssemblyContext {
	return AssemblyContext{
		SessionID:          sessionID,
		TaskID:             taskID,
		TurnNumber:         turnNumber,
		Mode:               mode,
		Phase:              phase,
		ActiveToolNames:    map[string]struct{}{},
		ActiveCapabilities: map[capabilityKey]struct{}{},
		PendingEvents:      nil,
		Features:           map[string]bool{},
		StorageScope:       DefaultStorageScope,
	}
}

// AddTool marks a tool name active.
func (c *AssemblyContext) AddTool(tool string) {
	if c.ActiveToolNames == nil {
		c.ActiveToolNames = map[string]struct{}{}
	}
	c.ActiveToolNames[tool] = struct{}{}
}

// AddCapability marks a (tool, capability) pair active.
func (c *AssemblyContext) AddCapability(tool, capability string) {
	if c.ActiveCapabilities == nil {
		c.ActiveCapabilities = map[capabilityKey]struct{}{}
	}
	c.ActiveCapabilities[capabilityKey{Tool: tool, Capability: capability}] = struct{}{}
}

func (c *AssemblyContext) hasTool(tool string) bool {
	_, ok := c.ActiveToolNames[tool]
	return ok
}

func (c *AssemblyContext) hasCapability(tool, capability string) bool {
	_, ok := c.ActiveCapabilities[capabilityKey{Tool: tool, Capability: capability}]
	return ok
}

func (c *AssemblyContext) hasEvent(e sporecore.HookEvent) bool {
	for _, pe := range c.PendingEvents {
		if pe == e {
			return true
		}
	}
	return false
}

// --- Serialization ----------------------------------------------------------
//
// The fixtures encode AssemblyContext with snake_case keys, active_tool_names
// as a string array, active_capabilities as an array of [tool, capability]
// pairs, and pending_events as a string array. We map those to/from the
// in-memory set representations.

type assemblyContextWire struct {
	SessionID          sporecore.SessionID      `json:"session_id"`
	TaskID             sporecore.TaskID         `json:"task_id"`
	TurnNumber         uint32                   `json:"turn_number"`
	Mode               promptchunkregistry.Mode `json:"mode"`
	Phase              sporecore.TaskPhase      `json:"phase"`
	AgentType          *string                  `json:"agent_type"`
	ActiveToolNames    []string                 `json:"active_tool_names"`
	ActiveCapabilities [][]string               `json:"active_capabilities"`
	IncomingMessage    *string                  `json:"incoming_message"`
	PendingEvents      []sporecore.HookEvent    `json:"pending_events"`
	Features           map[string]bool          `json:"features"`
	StorageScope       *StorageScope            `json:"storage_scope"`
}

// MarshalJSON emits the cross-language wire form. Set membership is emitted in
// sorted order so the encoding is deterministic.
func (c AssemblyContext) MarshalJSON() ([]byte, error) {
	w := assemblyContextWire{
		SessionID:       c.SessionID,
		TaskID:          c.TaskID,
		TurnNumber:      c.TurnNumber,
		Mode:            c.Mode,
		Phase:           c.Phase,
		IncomingMessage: c.IncomingMessage,
		PendingEvents:   c.PendingEvents,
		Features:        c.Features,
	}
	if c.AgentType != "" {
		at := c.AgentType
		w.AgentType = &at
	}
	tools := make([]string, 0, len(c.ActiveToolNames))
	for t := range c.ActiveToolNames {
		tools = append(tools, t)
	}
	sort.Strings(tools)
	w.ActiveToolNames = tools

	caps := make([][]string, 0, len(c.ActiveCapabilities))
	for k := range c.ActiveCapabilities {
		caps = append(caps, []string{k.Tool, k.Capability})
	}
	sort.Slice(caps, func(i, j int) bool {
		if caps[i][0] != caps[j][0] {
			return caps[i][0] < caps[j][0]
		}
		return caps[i][1] < caps[j][1]
	})
	w.ActiveCapabilities = caps

	scope := c.StorageScope
	if scope != "" {
		w.StorageScope = &scope
	}
	return json.Marshal(w)
}

// UnmarshalJSON parses the cross-language wire form, defaulting absent optional
// fields (#81 discipline: an absent storage_scope defaults to Project).
func (c *AssemblyContext) UnmarshalJSON(data []byte) error {
	var w assemblyContextWire
	if err := json.Unmarshal(data, &w); err != nil {
		return err
	}
	c.SessionID = w.SessionID
	c.TaskID = w.TaskID
	c.TurnNumber = w.TurnNumber
	c.Mode = w.Mode
	c.Phase = w.Phase
	if w.AgentType != nil {
		c.AgentType = *w.AgentType
	} else {
		c.AgentType = ""
	}
	c.ActiveToolNames = map[string]struct{}{}
	for _, t := range w.ActiveToolNames {
		c.ActiveToolNames[t] = struct{}{}
	}
	c.ActiveCapabilities = map[capabilityKey]struct{}{}
	for _, pair := range w.ActiveCapabilities {
		if len(pair) == 2 {
			c.ActiveCapabilities[capabilityKey{Tool: pair[0], Capability: pair[1]}] = struct{}{}
		}
	}
	c.IncomingMessage = w.IncomingMessage
	c.PendingEvents = w.PendingEvents
	if w.Features != nil {
		c.Features = w.Features
	} else {
		c.Features = map[string]bool{}
	}
	if w.StorageScope != nil {
		c.StorageScope = *w.StorageScope
	} else {
		c.StorageScope = DefaultStorageScope
	}
	return nil
}

// ============================================================================
// ChunkProviderError
// ============================================================================

// ChunkProviderError is a load/parse failure raised by a ChunkProvider. Kept
// minimal because the Remote/FileSystem providers are deferred (A6). The Kind
// field discriminates variants for the cross-language wire form.
type ChunkProviderError struct {
	Kind     string `json:"kind"`
	Provider string `json:"provider,omitempty"`
	Detail   string `json:"detail"`
}

const (
	// ChunkProviderErrorLoadFailed — a provider could not load its chunks.
	ChunkProviderErrorLoadFailed = "load_failed"
	// ChunkProviderErrorParseError — a provider's chunk payload failed to parse.
	ChunkProviderErrorParseError = "parse_error"
)

// NewLoadFailedError builds a load_failed ChunkProviderError.
func NewLoadFailedError(provider, detail string) *ChunkProviderError {
	return &ChunkProviderError{Kind: ChunkProviderErrorLoadFailed, Provider: provider, Detail: detail}
}

// NewParseError builds a parse_error ChunkProviderError.
func NewParseError(detail string) *ChunkProviderError {
	return &ChunkProviderError{Kind: ChunkProviderErrorParseError, Detail: detail}
}

// Error implements error.
func (e *ChunkProviderError) Error() string {
	switch e.Kind {
	case ChunkProviderErrorLoadFailed:
		return fmt.Sprintf("chunk load failed from %s: %s", e.Provider, e.Detail)
	case ChunkProviderErrorParseError:
		return fmt.Sprintf("chunk parse error: %s", e.Detail)
	default:
		return fmt.Sprintf("chunk provider error: %s", e.Detail)
	}
}

// ============================================================================
// ChunkProvider
// ============================================================================

// ChunkProvider is the pluggable source of chunks. Load is called at harness
// construction (every request in stateless deployments, once at startup in
// long-lived ones). Invalidate drops cached state so the next Load fetches
// fresh; it is never called mid-session.
type ChunkProvider interface {
	// Load returns all chunks this provider is responsible for.
	Load(ctx context.Context) ([]PromptChunk, error)
	// Invalidate drops any cached state. No-op for providers that don't cache.
	Invalidate()
}

// --- EmbeddedChunkProvider --------------------------------------------------

// EmbeddedChunkProvider serves compile-time / construction-time chunks. The set
// is immutable; Invalidate is a no-op and Load always returns the same chunks.
type EmbeddedChunkProvider struct {
	chunks []PromptChunk
}

// NewEmbeddedChunkProvider builds an embedded provider over the given chunks.
func NewEmbeddedChunkProvider(chunks []PromptChunk) *EmbeddedChunkProvider {
	return &EmbeddedChunkProvider{chunks: chunks}
}

// Load returns the immutable chunk set.
func (p *EmbeddedChunkProvider) Load(_ context.Context) ([]PromptChunk, error) {
	return cloneChunks(p.chunks), nil
}

// Invalidate is a no-op — embedded chunks are immutable constants.
func (p *EmbeddedChunkProvider) Invalidate() {}

// --- InMemoryChunkProvider --------------------------------------------------

// InMemoryChunkProvider serves programmatically registered chunks. Set replaces
// the chunk list; Invalidate is a no-op (clearing would discard programmatic
// registrations, matching the Rust reference).
type InMemoryChunkProvider struct {
	mu     sync.RWMutex
	chunks []PromptChunk
}

// NewInMemoryChunkProvider builds an in-memory provider over the given chunks.
func NewInMemoryChunkProvider(chunks []PromptChunk) *InMemoryChunkProvider {
	return &InMemoryChunkProvider{chunks: chunks}
}

// NewEmptyInMemoryChunkProvider builds an in-memory provider with no chunks.
func NewEmptyInMemoryChunkProvider() *InMemoryChunkProvider {
	return &InMemoryChunkProvider{}
}

// Set replaces the chunk list. The next Load returns the new set.
func (p *InMemoryChunkProvider) Set(chunks []PromptChunk) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.chunks = chunks
}

// Load returns the current chunk set.
func (p *InMemoryChunkProvider) Load(_ context.Context) ([]PromptChunk, error) {
	p.mu.RLock()
	defer p.mu.RUnlock()
	return cloneChunks(p.chunks), nil
}

// Invalidate is a no-op — the architect replaces chunks via Set.
func (p *InMemoryChunkProvider) Invalidate() {}

// --- CompositeChunkProvider -------------------------------------------------

// CompositeChunkProvider merges N providers into one flat list (in add order)
// and propagates Invalidate to every child.
type CompositeChunkProvider struct {
	providers []ChunkProvider
}

// NewCompositeChunkProvider builds an empty composite provider.
func NewCompositeChunkProvider() *CompositeChunkProvider {
	return &CompositeChunkProvider{}
}

// Add appends a child provider and returns the composite for chaining.
func (p *CompositeChunkProvider) Add(provider ChunkProvider) *CompositeChunkProvider {
	p.providers = append(p.providers, provider)
	return p
}

// Load merges every child's chunks in add order.
func (p *CompositeChunkProvider) Load(ctx context.Context) ([]PromptChunk, error) {
	var out []PromptChunk
	for _, child := range p.providers {
		chunks, err := child.Load(ctx)
		if err != nil {
			return nil, err
		}
		out = append(out, chunks...)
	}
	return out, nil
}

// Invalidate propagates to every child.
func (p *CompositeChunkProvider) Invalidate() {
	for _, child := range p.providers {
		child.Invalidate()
	}
}

func cloneChunks(in []PromptChunk) []PromptChunk {
	if in == nil {
		return nil
	}
	out := make([]PromptChunk, len(in))
	copy(out, in)
	return out
}

// Compile-time interface checks.
var (
	_ ChunkProvider = (*EmbeddedChunkProvider)(nil)
	_ ChunkProvider = (*InMemoryChunkProvider)(nil)
	_ ChunkProvider = (*CompositeChunkProvider)(nil)
	_ error         = (*ChunkProviderError)(nil)
)

// ============================================================================
// ContextSourcesBuilder
// ============================================================================

// ContextSourcesBuilder evaluates conditions, buckets chunks by stability,
// derives tool-affinity inclusion, scans triggers, injects pending events, and
// composes a Block-1 ComposedPrompt from the Static bucket. The result feeds
// contextmgr.ContextSources (decision A4).
type ContextSourcesBuilder struct {
	chunks []PromptChunk
}

// AssemblyBuckets is the bucketed outcome of ContextSourcesBuilder.Assemble.
// Each bucket keeps registration order within its stability tier.
type AssemblyBuckets struct {
	Static     []PromptChunk
	PerSession []PromptChunk
	PerTurn    []PromptChunk
}

// NewContextSourcesBuilder builds an empty builder.
func NewContextSourcesBuilder() *ContextSourcesBuilder {
	return &ContextSourcesBuilder{}
}

// NewContextSourcesBuilderWithChunks seeds the builder with chunks (registration
// order is preserved).
func NewContextSourcesBuilderWithChunks(chunks []PromptChunk) *ContextSourcesBuilder {
	return &ContextSourcesBuilder{chunks: chunks}
}

// Register appends a chunk, preserving registration order, and returns the
// builder for chaining.
func (b *ContextSourcesBuilder) Register(chunk PromptChunk) *ContextSourcesBuilder {
	b.chunks = append(b.chunks, chunk)
	return b
}

// Evaluate recursively evaluates a condition against ctx. Rules R1–R9.
func (b *ContextSourcesBuilder) Evaluate(condition ChunkCondition, ctx *AssemblyContext) bool {
	switch condition.kind {
	case KindAlways: // R1
		return true
	case KindWhenMode: // R2
		return ctx.Mode == condition.mode
	case KindWhenToolActive: // R3
		return ctx.hasTool(condition.tool)
	case KindWhenToolCapability: // R4
		return ctx.hasCapability(condition.tool, condition.capability)
	case KindWhenPhase: // R5
		return ctx.Phase == condition.phase
	case KindWhenAgentType: // R5
		return ctx.AgentType != "" && ctx.AgentType == condition.agentType
	case KindWhenFeature: // R5 — present AND true
		v, ok := ctx.Features[condition.feature]
		return ok && v
	case KindOnTrigger: // R6 — substring match; absent message never matches
		if ctx.IncomingMessage == nil {
			return false
		}
		msg := *ctx.IncomingMessage
		for _, w := range condition.words {
			if strings.Contains(msg, w) {
				return true
			}
		}
		return false
	case KindOnEvent: // R7
		return ctx.hasEvent(condition.event)
	case KindAll: // R8
		for _, child := range condition.children {
			if !b.Evaluate(child, ctx) {
				return false
			}
		}
		return true
	case KindAny: // R8
		for _, child := range condition.children {
			if b.Evaluate(child, ctx) {
				return true
			}
		}
		return false
	case KindNot: // R8
		if condition.inner == nil {
			return false
		}
		return !b.Evaluate(*condition.inner, ctx)
	case KindCustom: // R9
		if condition.custom == nil {
			return false
		}
		return condition.custom(ctx)
	default:
		return false
	}
}

// toolAffinityOK reports whether a chunk's tool-affinity gate passes. A chunk
// with no affinity always passes. Rules R12 / R17.
func toolAffinityOK(chunk *PromptChunk, ctx *AssemblyContext) bool {
	if chunk.ToolAffinity == nil {
		return true
	}
	aff := chunk.ToolAffinity
	if !ctx.hasTool(aff.ToolName) {
		return false
	}
	if aff.Capability == "" {
		return true
	}
	return ctx.hasCapability(aff.ToolName, aff.Capability)
}

// agentAffinityOK reports whether a chunk's agent-affinity gate passes. A chunk
// with no agent affinity always passes; otherwise it must match ctx.AgentType.
func agentAffinityOK(chunk *PromptChunk, ctx *AssemblyContext) bool {
	if chunk.AgentAffinity == "" {
		return true
	}
	return ctx.AgentType != "" && ctx.AgentType == chunk.AgentAffinity
}

// triggersMatch reports whether a chunk's triggers match the incoming message.
// An empty trigger list never forces inclusion. Rule R13.
func triggersMatch(chunk *PromptChunk, ctx *AssemblyContext) bool {
	if len(chunk.Triggers) == 0 {
		return false
	}
	if ctx.IncomingMessage == nil {
		return false
	}
	msg := *ctx.IncomingMessage
	for _, t := range chunk.Triggers {
		if strings.Contains(msg, t) {
			return true
		}
	}
	return false
}

// Assemble runs the assembly steps and buckets the included chunks.
// Registration order is preserved within each bucket (R10/R11).
//
// A chunk is included when its condition evaluates true AND its tool-affinity
// AND agent-affinity gates pass. A chunk whose triggers match the incoming
// message is forced into the PerTurn bucket regardless of its declared
// stability (R13). Otherwise bucket assignment follows the chunk's stability.
func (b *ContextSourcesBuilder) Assemble(ctx *AssemblyContext) AssemblyBuckets {
	var buckets AssemblyBuckets

	for i := range b.chunks {
		chunk := b.chunks[i]

		// Gates that apply to EVERY chunk regardless of condition kind.
		if !toolAffinityOK(&chunk, ctx) {
			continue
		}
		if !agentAffinityOK(&chunk, ctx) {
			continue
		}

		conditionOK := b.Evaluate(chunk.Condition, ctx)
		triggerForced := triggersMatch(&chunk, ctx)

		if !conditionOK && !triggerForced {
			continue
		}

		// R13: a trigger match routes the chunk into PerTurn no matter its
		// declared stability. R14 falls out of this too: an OnEvent chunk is
		// only conditionOK when its event is pending, and OnEvent chunks are
		// declared PerTurn by convention.
		if triggerForced {
			buckets.PerTurn = append(buckets.PerTurn, chunk)
			continue
		}

		switch chunk.Stability {
		case contextmgr.StabilityStatic:
			buckets.Static = append(buckets.Static, chunk)
		case contextmgr.StabilityPerSession:
			buckets.PerSession = append(buckets.PerSession, chunk)
		case contextmgr.StabilityPerTurn:
			buckets.PerTurn = append(buckets.PerTurn, chunk)
		}
	}

	return buckets
}

// ComposeBlock1 composes the Static bucket into a #24 ComposedPrompt (Block 1).
// Each Static chunk maps to a #24 PromptChunk in ChunkSlotEnvironment (a
// neutral, non-required slot) with CacheBlockStatic, preserving order. The
// block hashes are recomputed from the mapped chunks so the Block-1 hash is
// stable across identical Static sets (R15).
func (b *ContextSourcesBuilder) ComposeBlock1(buckets AssemblyBuckets) promptchunkregistry.ComposedPrompt {
	chunks := make([]promptchunkregistry.PromptChunk, 0, len(buckets.Static))
	for _, c := range buckets.Static {
		chunks = append(chunks, promptchunkregistry.NewPromptChunk(
			promptchunkregistry.ChunkID(c.ID),
			c.Content,
			promptchunkregistry.ChunkSlotEnvironment,
			promptchunkregistry.CacheBlockStatic,
		))
	}
	composed := promptchunkregistry.ComposedPrompt{Chunks: chunks}
	b1, b2 := composed.RecomputeHashes()
	composed.Block1Hash = b1
	composed.Block2Hash = b2
	composed.Render()
	return composed
}

// BuildContextSources runs the full pipeline: assemble buckets, compose Block 1,
// and produce a contextmgr.ContextSources (decision A4 — PerSession/PerTurn fold
// through the existing composed-prompt / segment machinery downstream; this
// builder supplies the composed Block 1 and the buckets the caller threads into
// the context machinery).
//
// guides, memory, and toolSchemas are passed through verbatim — the builder does
// not synthesize tool description text (decision A5).
func (b *ContextSourcesBuilder) BuildContextSources(
	ctx *AssemblyContext,
	guides []contextmgr.Guide,
	memory []contextmgr.MemoryItem,
	toolSchemas []sporecore.ToolSchema,
) (contextmgr.ContextSources, AssemblyBuckets) {
	buckets := b.Assemble(ctx)
	composed := b.ComposeBlock1(buckets)
	sources := contextmgr.ContextSources{
		Guides:      guides,
		Memory:      memory,
		ToolSchemas: toolSchemas,
		// contextmgr.ContextSources carries a narrowed ComposedPrompt
		// {Rendered, Block1Hash} (the Go #7/#24 wiring); fold the full Block-1
		// rendering and hash into it.
		ComposedPrompt: contextmgr.ComposedPrompt{
			Rendered:   composed.Render(),
			Block1Hash: composed.Block1Hash,
		},
	}
	return sources, buckets
}

// BreakpointIDs returns, in order, the id of every chunk across all buckets that
// declared a cache breakpoint (R16). Exposed for callers wiring the
// PerSession/PerTurn segments into the segment machinery.
func BreakpointIDs(buckets AssemblyBuckets) []string {
	var out []string
	for _, group := range [][]PromptChunk{buckets.Static, buckets.PerSession, buckets.PerTurn} {
		for _, c := range group {
			if c.CacheBreakpoint {
				out = append(out, c.ID)
			}
		}
	}
	return out
}

// ChunksToSegments maps a bucket of chunks into contextmgr.PromptSegments for
// the #7 context machinery (decision A4). Preserves order and carries each
// chunk's cache breakpoint (R16).
func ChunksToSegments(chunks []PromptChunk) []contextmgr.PromptSegment {
	out := make([]contextmgr.PromptSegment, 0, len(chunks))
	for _, c := range chunks {
		out = append(out, contextmgr.PromptSegment{
			Name:            c.ID,
			Content:         c.Content,
			Stability:       c.Stability,
			CacheBreakpoint: c.CacheBreakpoint,
		})
	}
	return out
}

// ============================================================================
// HarnessBuilder integration (R25)
// ============================================================================

// HarnessBuilder accumulates prompt-assembly wiring for a harness. The Go core
// has no central builder type, so this package owns the chunk-provider seam the
// spec calls for: ChunkProvider sets a provider directly, Chunks registers an
// inline list via an InMemoryChunkProvider, and the default is an empty
// InMemoryChunkProvider (R25).
type HarnessBuilder struct {
	provider ChunkProvider
}

// NewHarnessBuilder builds a HarnessBuilder defaulting to an empty
// InMemoryChunkProvider (R25).
func NewHarnessBuilder() *HarnessBuilder {
	return &HarnessBuilder{provider: NewEmptyInMemoryChunkProvider()}
}

// ChunkProvider sets the chunk provider and returns the builder for chaining.
func (h *HarnessBuilder) ChunkProvider(provider ChunkProvider) *HarnessBuilder {
	h.provider = provider
	return h
}

// Chunks registers chunks inline via an InMemoryChunkProvider and returns the
// builder for chaining.
func (h *HarnessBuilder) Chunks(chunks []PromptChunk) *HarnessBuilder {
	h.provider = NewInMemoryChunkProvider(chunks)
	return h
}

// Provider returns the configured chunk provider (never nil).
func (h *HarnessBuilder) Provider() ChunkProvider {
	return h.provider
}
