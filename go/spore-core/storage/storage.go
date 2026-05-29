// Package storage — issue #73: a pluggable, per-domain persistence layer, plus
// the #78 scope + workspace-partitioning extension.
//
// This is the Go port of the Rust reference
// (rust/crates/spore-core/src/storage.rs). It does NOT transliterate Rust; it
// follows the idioms in go/CONVENTIONS.md (consumer-side interfaces,
// context.Context first arg, sentinel + typed errors, defined ID types).
//
// # Scope + workspace partitioning (#78)
//
//   - StorageScope is REUSED from the promptassembly package (its canonical home,
//     decision A2) via a type alias — the Go analog of Rust's
//     `pub use crate::prompt_assembly::StorageScope`. It is NOT redefined here.
//   - MemoryStore is the only domain that gained a scope: AppendMemory and
//     GetMemories take a StorageScope. The other three domains stay scope-flat.
//   - Leaf backends stay SCOPE-DUMB: a FileSystemStorageProvider never learns a
//     WorkspaceID — the user-scope backend is pointed at the already-partitioned
//     root {user_root}/projects/{workspace_id} at construction. InMemory keys
//     memory by (scope, sessionID).
//   - CompositeStorageProvider routes the memory domain per-scope
//     (Memory(scope, store)); unconfigured (memory, scope) pairs fall back to
//     NoOp on Build(). A ScopedMemoryRouter is the memory slot; it owns the
//     cross-scope merge — leaf backends never merge.
//   - StorageProvider.GetMemoriesMerged returns User ∪ Project, newest-first by
//     timestamp, NO dedup, Local excluded — the merge lives in the routing layer.
//   - WorkspaceID + WorkspaceIDFromCanonicalPath is a PURE STRING function (no
//     filesystem access): the cross-language parity anchor pinned by
//     fixtures/storage/workspace_id_derivation.json.
//
// # Known v1 limitation
//
// Memory addressing stays sessionID-keyed for v1. v2 should address
// session-independent / cross-session memory keying — do not introduce it here.
//
// # Domains
//
// Four domain stores, each a Go interface taking context.Context first:
//
//   - SessionStore — pause/resume lifecycle. Stores *sporecore.PausedState
//     keyed by SessionID.
//   - MemoryStore — episodic memory. Append-only log per session;
//     GetMemories(limit) returns the MOST-RECENT N, NEWEST-FIRST (recency).
//   - RunStore — per-run structured state keyed by (SessionID, key). Values are
//     opaque JSON blobs (json.RawMessage); the store never knows the schema.
//   - ObservabilityStore — append-only span storage; no get-by-key, queried by
//     session and time range.
//
// # StorageProvider
//
// A struct of four interface fields (Session, Memory, Run, Observability) with
// accessors and a SingleStorageProvider convenience constructor that places one
// concrete backend (implementing all four interfaces) into all four slots.
//
// # Providers
//
//   - NoOpStorageProvider — silent discard; reads return zero/nil, writes nil.
//     The default for any unconfigured domain.
//   - InMemoryStorageProvider — maps guarded by a sync.Mutex.
//   - FileSystemStorageProvider — disk-backed, atomic write-rename for
//     non-append writes, append for the JSONL logs.
//   - CompositeStorageProvider — a builder routing each domain to its own
//     backend, falling back to NoOp for any unset domain on Build().
//
// # Rules enforced
//
//   - No-op fallback. Unconfigured domains fall back to NoOpStorageProvider;
//     callers never nil-check — they always call the store and the store
//     decides. No-op reads return (nil/zero, false, nil) or ([], nil); writes
//     return nil.
//   - Single-fills-all-slots. SingleStorageProvider places one backend in all
//     four slots.
//   - Composite per-domain routing with NoOp fallback on Build().
//   - Atomic write-rename. FileSystemStorageProvider non-append writes ensure
//     the parent dir, write full bytes to a sibling {target}.tmp, flush + Sync,
//     then os.Rename(tmp, target). No leftover .tmp on success. Layout:
//     {root}/sessions/{id}/state.json     (session, atomic),
//     {root}/sessions/{id}/run/{key}.json (run, atomic),
//     {root}/sessions/{id}/memory.jsonl   (memory, append),
//     {root}/sessions/{id}/trace.jsonl    (observability, append).
//     FlushSession creates a sibling .flushed marker.
//   - GetMemories recency. Reads the JSONL, returns the most-recent limit
//     entries, newest-first.
//   - Last-writer-wins for FS non-append writes via rename; no per-key locking
//     contract — atomic rename is the only durability guarantee.
package storage

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"sort"
	"sync"

	"crypto/sha256"
	"encoding/hex"
	"strings"
	"unicode"

	sporecore "github.com/squirrelsoft-dev/spore-core/go/spore-core"
	"github.com/squirrelsoft-dev/spore-core/go/spore-core/guideregistry"
	"github.com/squirrelsoft-dev/spore-core/go/spore-core/memory"
	"github.com/squirrelsoft-dev/spore-core/go/spore-core/observability"
	"github.com/squirrelsoft-dev/spore-core/go/spore-core/promptassembly"
)

// Re-exported type aliases so storage callers don't have to import every
// sibling package just to use the store signatures. These match the Rust
// imports (PausedState/SessionId from the harness, Timestamp from memory,
// SessionOutcome from guide_registry, SessionMetrics from observability).
type (
	// SessionID re-exported from sporecore.
	SessionID = sporecore.SessionID
	// PausedState re-exported from sporecore.
	PausedState = sporecore.PausedState
	// Timestamp re-exported from memory.
	Timestamp = memory.Timestamp
	// SessionOutcome re-exported from guideregistry.
	SessionOutcome = guideregistry.SessionOutcome
	// SessionMetrics re-exported from observability.
	SessionMetrics = observability.SessionMetrics
)

// StorageScope is REUSED from promptassembly (its canonical home, decision A2),
// re-exported here as a type alias so storage callers can name it without
// importing promptassembly. This is the Go analog of the Rust reference's
// `pub use crate::prompt_assembly::StorageScope` — storage::StorageScope and
// promptassembly.StorageScope resolve to the SAME type. The constants are
// re-exported too so they read naturally at the storage call site.
type StorageScope = promptassembly.StorageScope

// Scope constants, re-exported from promptassembly (same underlying values).
const (
	// StorageScopeUser is the user-global scope.
	StorageScopeUser = promptassembly.StorageScopeUser
	// StorageScopeProject is the per-project scope (the default).
	StorageScopeProject = promptassembly.StorageScopeProject
	// StorageScopeLocal is the machine-local scope.
	StorageScopeLocal = promptassembly.StorageScopeLocal
)

// ============================================================================
// StorageError
// ============================================================================

// ErrNotFound is returned wrapped by NotFoundError; match it with errors.Is.
// It is a sentinel only so callers can branch on "the keyed lookup found
// nothing where a value was required" without depending on the message.
var ErrNotFound = errors.New("storage not found")

// NotFoundError is a typed error for a keyed lookup that found nothing where
// the caller required a value. It carries the domain and key for diagnostics
// and unwraps to ErrNotFound (mirrors the Rust StorageError::NotFound variant).
//
// The reference providers never return NotFoundError for an absent value: a
// missing read returns (zero, false, nil) per Go's comma-ok idiom. The type
// exists so backends with a stricter contract can surface it portably.
type NotFoundError struct {
	Domain string
	Key    string
}

func (e *NotFoundError) Error() string {
	return fmt.Sprintf("storage not found: domain=%s key=%s", e.Domain, e.Key)
}

func (e *NotFoundError) Unwrap() error { return ErrNotFound }

// ============================================================================
// MemoryEntry
// ============================================================================

// MemoryEntry is one episodic memory entry. The JSON tags are byte-identical
// to the Rust reference ({ role, content, timestamp, metadata }) so fixtures
// replay across languages. Metadata defaults to an empty object {}.
type MemoryEntry struct {
	Role      string          `json:"role"`
	Content   string          `json:"content"`
	Timestamp Timestamp       `json:"timestamp"`
	Metadata  json.RawMessage `json:"metadata"`
}

// emptyObject is the canonical empty-object metadata default.
func emptyObject() json.RawMessage { return json.RawMessage("{}") }

// NewMemoryEntry builds an entry with an empty metadata object.
func NewMemoryEntry(role, content string, timestamp Timestamp) MemoryEntry {
	return MemoryEntry{
		Role:      role,
		Content:   content,
		Timestamp: timestamp,
		Metadata:  emptyObject(),
	}
}

// MarshalJSON emits an explicit empty object for nil Metadata so the on-disk
// shape is byte-identical to Rust's serde default.
func (e MemoryEntry) MarshalJSON() ([]byte, error) {
	type alias MemoryEntry
	a := alias(e)
	if len(a.Metadata) == 0 {
		a.Metadata = emptyObject()
	}
	return json.Marshal(a)
}

// UnmarshalJSON fills Metadata with the empty object {} when the field is
// absent or null, mirroring serde's #[serde(default)].
func (e *MemoryEntry) UnmarshalJSON(data []byte) error {
	type alias MemoryEntry
	var a alias
	if err := json.Unmarshal(data, &a); err != nil {
		return err
	}
	if len(a.Metadata) == 0 || string(a.Metadata) == "null" {
		a.Metadata = emptyObject()
	}
	*e = MemoryEntry(a)
	return nil
}

// ============================================================================
// WorkspaceID (#78)
// ============================================================================

// WorkspaceID is a stable identifier for a workspace, derived purely from its
// canonical path. Form: {sanitized_basename}-{8hex}, lowercased. It is the
// cross-language parity anchor — WorkspaceIDFromCanonicalPath is a PURE STRING
// function (it never touches the filesystem) so the pinned fixture
// fixtures/storage/workspace_id_derivation.json is host-independent.
//
// Used at wiring time to partition the user-scope storage root:
// {user_root}/projects/{workspace_id}. Backends never see it. Go initialism
// convention uses ID (uppercase), so the exported name is WorkspaceID.
type WorkspaceID string

// WorkspaceIDFromCanonicalPath derives a WorkspaceID from an already-OS-canonical
// path. The algorithm is pinned, byte-identical across all four languages:
//
//  1. Normalize separators to '/'. On Windows strip a leading drive-letter
//     prefix (e.g. "C:") and convert '\' -> '/'. The input is assumed already
//     OS-canonicalized; this function does NOT re-canonicalize or touch the
//     filesystem.
//  2. Build the canonical path string: forward slashes only, NO trailing slash,
//     UTF-8.
//  3. SHA-256 that string; take the first 8 hex chars (lowercase).
//  4. Basename of the canonical path, lowercased; replace each non-alphanumeric
//     char with '-'; collapse consecutive '-'; strip leading/trailing '-'. An
//     empty basename (root "/") becomes "root".
//  5. Concatenate {sanitized_basename}-{8hex}.
func WorkspaceIDFromCanonicalPath(path string) WorkspaceID {
	canonical := canonicalizePathString(path)

	sum := sha256.Sum256([]byte(canonical))
	hex8 := hex.EncodeToString(sum[:4]) // first 8 hex chars = first 4 bytes

	basename := canonical
	if idx := strings.LastIndex(canonical, "/"); idx >= 0 {
		basename = canonical[idx+1:]
	}
	sanitized := sanitizeBasename(basename)
	if sanitized == "" {
		sanitized = "root"
	}

	return WorkspaceID(sanitized + "-" + hex8)
}

// String returns the underlying derived id string.
func (w WorkspaceID) String() string { return string(w) }

// canonicalizePathString performs steps 1–2 of the derivation: produce the
// canonical path string used for both the hash input and the basename. Forward
// slashes only, no trailing slash (a lone root "/" is preserved).
func canonicalizePathString(path string) string {
	// Normalize Windows backslashes.
	s := strings.ReplaceAll(path, "\\", "/")
	// Strip a leading drive-letter prefix like "C:" (only at the very start).
	if len(s) >= 2 && s[1] == ':' &&
		((s[0] >= 'A' && s[0] <= 'Z') || (s[0] >= 'a' && s[0] <= 'z')) {
		s = s[2:]
	}
	// Strip trailing slashes, but keep a lone root "/".
	for len(s) > 1 && strings.HasSuffix(s, "/") {
		s = s[:len(s)-1]
	}
	return s
}

// sanitizeBasename performs step 4: lowercase, replace each non-alphanumeric
// rune with '-', collapse consecutive '-', strip leading/trailing '-'.
// "Alphanumeric" is ASCII a–z/0–9 after lowercasing (matching the Rust
// is_ascii_alphanumeric check) so the cross-language hash/basename stays
// byte-identical.
func sanitizeBasename(basename string) string {
	lowered := strings.ToLower(basename)
	var b strings.Builder
	b.Grow(len(lowered))
	prevDash := false
	for _, ch := range lowered {
		if ch <= unicode.MaxASCII && (('a' <= ch && ch <= 'z') || ('0' <= ch && ch <= '9')) {
			b.WriteRune(ch)
			prevDash = false
		} else if !prevDash {
			b.WriteByte('-')
			prevDash = true
		}
	}
	return strings.Trim(b.String(), "-")
}

// ============================================================================
// Domain interfaces
// ============================================================================

// SessionStore is the pause/resume lifecycle store. It stores *PausedState
// keyed by SessionID.
type SessionStore interface {
	// GetSession returns the stored state and found=false when absent.
	GetSession(ctx context.Context, id SessionID) (state *PausedState, found bool, err error)
	PutSession(ctx context.Context, id SessionID, state *PausedState) error
	DeleteSession(ctx context.Context, id SessionID) error
	ListSessions(ctx context.Context) ([]SessionID, error)
}

// MemoryStore is the episodic memory store. Append-only log per (scope,
// session) (#78).
//
// A leaf backend is SCOPE-DUMB: it stores under whatever root it was given. The
// scope argument is carried for symmetry and so a single backend could partition
// by scope if it wished (the InMemory backend does), but the v1 wiring routes
// each scope to its own backend via CompositeStorageProvider, so leaf backends
// receive a single scope's traffic. The cross-scope merge
// (StorageProvider.GetMemoriesMerged) lives in the routing layer, never in a
// leaf.
//
// Memory addressing stays sessionID-keyed for v1 (known limitation); v2 should
// address cross-session keying.
type MemoryStore interface {
	AppendMemory(ctx context.Context, scope StorageScope, sessionID SessionID, entry MemoryEntry) error
	// GetMemories returns the most-recent limit entries, newest-first, for scope.
	GetMemories(ctx context.Context, scope StorageScope, sessionID SessionID, limit int) ([]MemoryEntry, error)
}

// RunStore is per-run structured state keyed by (SessionID, key). Values are
// opaque JSON blobs — the store does not know the schema; callers own
// serialization.
type RunStore interface {
	// Get returns the stored value and found=false when absent.
	Get(ctx context.Context, sessionID SessionID, key string) (value json.RawMessage, found bool, err error)
	Put(ctx context.Context, sessionID SessionID, key string, value json.RawMessage) error
	Delete(ctx context.Context, sessionID SessionID, key string) error
	ListKeys(ctx context.Context, sessionID SessionID) ([]string, error)
}

// ObservabilityStore is append-only span storage. Distinct from the other
// three: no get-by-key, queried by session and time range.
type ObservabilityStore interface {
	AppendSpan(ctx context.Context, sessionID SessionID, span json.RawMessage) error
	GetSpans(ctx context.Context, sessionID SessionID) ([]json.RawMessage, error)
	GetSessions(ctx context.Context, since Timestamp, domain *string, outcome *SessionOutcome) ([]SessionMetrics, error)
	FlushSession(ctx context.Context, sessionID SessionID) error
}

// ============================================================================
// StorageProvider
// ============================================================================

// StorageProvider is a composed persistence layer: four independent domain
// stores behind interface values. Built either from a single backend (placed in
// all four slots via SingleStorageProvider) or per-domain via
// CompositeStorageProvider.
type StorageProvider struct {
	session       SessionStore
	memory        MemoryStore
	run           RunStore
	observability ObservabilityStore
}

// NewStorageProvider constructs from four explicit per-domain stores.
func NewStorageProvider(session SessionStore, mem MemoryStore, run RunStore, obs ObservabilityStore) *StorageProvider {
	return &StorageProvider{session: session, memory: mem, run: run, observability: obs}
}

// fullProvider is implemented by a single concrete backend that covers all four
// domains. SingleStorageProvider places it in every slot.
type fullProvider interface {
	SessionStore
	MemoryStore
	RunStore
	ObservabilityStore
}

// SingleStorageProvider places one concrete backend (implementing all four
// domain interfaces) into all four slots.
func SingleStorageProvider(p fullProvider) *StorageProvider {
	return &StorageProvider{session: p, memory: p, run: p, observability: p}
}

// NoOp returns an all-no-op StorageProvider. The default when storage is never
// configured.
func NoOp() *StorageProvider {
	return SingleStorageProvider(NoOpStorageProvider{})
}

// Session returns the session-domain store.
func (p *StorageProvider) Session() SessionStore { return p.session }

// Memory returns the memory-domain store.
func (p *StorageProvider) Memory() MemoryStore { return p.memory }

// GetMemoriesMerged is the cross-scope merged memory read (#78 R6): User ∪
// Project, newest-first by timestamp, NO dedup (identical-content entries are
// all retained). Local is excluded from the merge in v1.
//
// It routes through the memory slot. When the provider was built via
// CompositeStorageProvider that slot is a *ScopedMemoryRouter fanning out to the
// per-scope backends; for SingleStorageProvider / NewStorageProvider the one
// backend serves both scopes (keyed by scope) and merges identically. The merge
// always lives in this routing layer, never in a leaf backend.
func (p *StorageProvider) GetMemoriesMerged(ctx context.Context, sessionID SessionID, limit int) ([]MemoryEntry, error) {
	user, err := p.memory.GetMemories(ctx, StorageScopeUser, sessionID, limit)
	if err != nil {
		return nil, err
	}
	project, err := p.memory.GetMemories(ctx, StorageScopeProject, sessionID, limit)
	if err != nil {
		return nil, err
	}
	combined := append(user, project...)
	return mergeNewestFirst(combined, limit), nil
}

// Run returns the run-domain store.
func (p *StorageProvider) Run() RunStore { return p.run }

// Observability returns the observability-domain store.
func (p *StorageProvider) Observability() ObservabilityStore { return p.observability }

// ============================================================================
// NoOpStorageProvider
// ============================================================================

// NoOpStorageProvider silently discards writes and returns empty reads. It is
// the default for any unconfigured domain. Reads return (nil/zero, false, nil)
// or ([], nil); writes return nil.
type NoOpStorageProvider struct{}

// SessionStore.

func (NoOpStorageProvider) GetSession(context.Context, SessionID) (*PausedState, bool, error) {
	return nil, false, nil
}
func (NoOpStorageProvider) PutSession(context.Context, SessionID, *PausedState) error { return nil }
func (NoOpStorageProvider) DeleteSession(context.Context, SessionID) error            { return nil }
func (NoOpStorageProvider) ListSessions(context.Context) ([]SessionID, error)         { return nil, nil }

// MemoryStore.

func (NoOpStorageProvider) AppendMemory(context.Context, StorageScope, SessionID, MemoryEntry) error {
	return nil
}
func (NoOpStorageProvider) GetMemories(context.Context, StorageScope, SessionID, int) ([]MemoryEntry, error) {
	return nil, nil
}

// RunStore.

func (NoOpStorageProvider) Get(context.Context, SessionID, string) (json.RawMessage, bool, error) {
	return nil, false, nil
}
func (NoOpStorageProvider) Put(context.Context, SessionID, string, json.RawMessage) error { return nil }
func (NoOpStorageProvider) Delete(context.Context, SessionID, string) error               { return nil }
func (NoOpStorageProvider) ListKeys(context.Context, SessionID) ([]string, error) {
	return nil, nil
}

// ObservabilityStore.

func (NoOpStorageProvider) AppendSpan(context.Context, SessionID, json.RawMessage) error { return nil }
func (NoOpStorageProvider) GetSpans(context.Context, SessionID) ([]json.RawMessage, error) {
	return nil, nil
}
func (NoOpStorageProvider) GetSessions(context.Context, Timestamp, *string, *SessionOutcome) ([]SessionMetrics, error) {
	return nil, nil
}
func (NoOpStorageProvider) FlushSession(context.Context, SessionID) error { return nil }

// ============================================================================
// InMemoryStorageProvider
// ============================================================================

// InMemoryStorageProvider is a mutex-guarded in-memory backend used in tests
// and ephemeral runs. The zero value is NOT usable; construct with
// NewInMemoryStorageProvider.
type InMemoryStorageProvider struct {
	mu       sync.Mutex
	sessions map[SessionID]PausedState
	memory   map[memoryKey][]MemoryEntry
	run      map[runKey]json.RawMessage
	spans    map[SessionID][]json.RawMessage
}

type runKey struct {
	session SessionID
	key     string
}

// memoryKey keys episodic memory by (scope, session) (#78): the InMemory backend
// is scope-aware so a single instance handed all scopes still isolates them.
type memoryKey struct {
	scope   StorageScope
	session SessionID
}

// NewInMemoryStorageProvider constructs an empty in-memory backend.
func NewInMemoryStorageProvider() *InMemoryStorageProvider {
	return &InMemoryStorageProvider{
		sessions: make(map[SessionID]PausedState),
		memory:   make(map[memoryKey][]MemoryEntry),
		run:      make(map[runKey]json.RawMessage),
		spans:    make(map[SessionID][]json.RawMessage),
	}
}

// SessionStore.

func (p *InMemoryStorageProvider) GetSession(_ context.Context, id SessionID) (*PausedState, bool, error) {
	p.mu.Lock()
	defer p.mu.Unlock()
	s, ok := p.sessions[id]
	if !ok {
		return nil, false, nil
	}
	cp := s
	return &cp, true, nil
}

func (p *InMemoryStorageProvider) PutSession(_ context.Context, id SessionID, state *PausedState) error {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.sessions[id] = *state
	return nil
}

func (p *InMemoryStorageProvider) DeleteSession(_ context.Context, id SessionID) error {
	p.mu.Lock()
	defer p.mu.Unlock()
	delete(p.sessions, id)
	return nil
}

func (p *InMemoryStorageProvider) ListSessions(_ context.Context) ([]SessionID, error) {
	p.mu.Lock()
	defer p.mu.Unlock()
	out := make([]SessionID, 0, len(p.sessions))
	for k := range p.sessions {
		out = append(out, k)
	}
	sort.Slice(out, func(i, j int) bool { return out[i] < out[j] })
	return out, nil
}

// MemoryStore.

func (p *InMemoryStorageProvider) AppendMemory(_ context.Context, scope StorageScope, sessionID SessionID, entry MemoryEntry) error {
	p.mu.Lock()
	defer p.mu.Unlock()
	k := memoryKey{scope: scope, session: sessionID}
	p.memory[k] = append(p.memory[k], entry)
	return nil
}

func (p *InMemoryStorageProvider) GetMemories(_ context.Context, scope StorageScope, sessionID SessionID, limit int) ([]MemoryEntry, error) {
	p.mu.Lock()
	defer p.mu.Unlock()
	all := p.memory[memoryKey{scope: scope, session: sessionID}]
	return mostRecentNewestFirst(all, limit), nil
}

// RunStore.

func (p *InMemoryStorageProvider) Get(_ context.Context, sessionID SessionID, key string) (json.RawMessage, bool, error) {
	p.mu.Lock()
	defer p.mu.Unlock()
	v, ok := p.run[runKey{sessionID, key}]
	if !ok {
		return nil, false, nil
	}
	return cloneRaw(v), true, nil
}

func (p *InMemoryStorageProvider) Put(_ context.Context, sessionID SessionID, key string, value json.RawMessage) error {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.run[runKey{sessionID, key}] = cloneRaw(value)
	return nil
}

func (p *InMemoryStorageProvider) Delete(_ context.Context, sessionID SessionID, key string) error {
	p.mu.Lock()
	defer p.mu.Unlock()
	delete(p.run, runKey{sessionID, key})
	return nil
}

func (p *InMemoryStorageProvider) ListKeys(_ context.Context, sessionID SessionID) ([]string, error) {
	p.mu.Lock()
	defer p.mu.Unlock()
	var out []string
	for k := range p.run {
		if k.session == sessionID {
			out = append(out, k.key)
		}
	}
	sort.Strings(out)
	return out, nil
}

// ObservabilityStore.

func (p *InMemoryStorageProvider) AppendSpan(_ context.Context, sessionID SessionID, span json.RawMessage) error {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.spans[sessionID] = append(p.spans[sessionID], cloneRaw(span))
	return nil
}

func (p *InMemoryStorageProvider) GetSpans(_ context.Context, sessionID SessionID) ([]json.RawMessage, error) {
	p.mu.Lock()
	defer p.mu.Unlock()
	src := p.spans[sessionID]
	out := make([]json.RawMessage, len(src))
	for i, v := range src {
		out[i] = cloneRaw(v)
	}
	return out, nil
}

// GetSessions returns empty: the in-memory span store does not roll up
// SessionMetrics — that is the ObservabilityProvider's job.
func (p *InMemoryStorageProvider) GetSessions(context.Context, Timestamp, *string, *SessionOutcome) ([]SessionMetrics, error) {
	return nil, nil
}

// FlushSession is a no-op for the in-memory store.
func (p *InMemoryStorageProvider) FlushSession(context.Context, SessionID) error { return nil }

// cloneRaw returns an independent copy of a json.RawMessage so callers can't
// mutate the stored bytes through an aliased slice.
func cloneRaw(v json.RawMessage) json.RawMessage {
	if v == nil {
		return nil
	}
	out := make(json.RawMessage, len(v))
	copy(out, v)
	return out
}

// ============================================================================
// CompositeStorageProvider
// ============================================================================

// CompositeStorageProvider is a builder that routes each domain to its own
// backend, filling any unset domain with NoOpStorageProvider on Build().
//
// Only the memory domain varies by scope (#78): Memory(scope, store) configures
// one StorageScope's backend, and unconfigured (memory, scope) pairs fall back
// to NoOp on Build() via the ScopedMemoryRouter. session, run and observability
// are scope-flat — scope is wiring-only for them.
type CompositeStorageProvider struct {
	session       SessionStore
	memory        map[StorageScope]MemoryStore
	run           RunStore
	observability ObservabilityStore
}

// NewCompositeStorageProvider starts an empty composite builder.
func NewCompositeStorageProvider() *CompositeStorageProvider {
	return &CompositeStorageProvider{memory: make(map[StorageScope]MemoryStore)}
}

// Session sets the session-domain backend and returns the receiver.
func (c *CompositeStorageProvider) Session(store SessionStore) *CompositeStorageProvider {
	c.session = store
	return c
}

// Memory configures the memory backend for one StorageScope and returns the
// receiver. Unconfigured (memory, scope) pairs fall back to NoOpStorageProvider
// on Build() (#78 R7/R11 — Local may be wired to no-op in v1).
func (c *CompositeStorageProvider) Memory(scope StorageScope, store MemoryStore) *CompositeStorageProvider {
	if c.memory == nil {
		c.memory = make(map[StorageScope]MemoryStore)
	}
	c.memory[scope] = store
	return c
}

// Run sets the run-domain backend and returns the receiver.
func (c *CompositeStorageProvider) Run(store RunStore) *CompositeStorageProvider {
	c.run = store
	return c
}

// Observability sets the observability-domain backend and returns the receiver.
func (c *CompositeStorageProvider) Observability(store ObservabilityStore) *CompositeStorageProvider {
	c.observability = store
	return c
}

// Build assembles a *StorageProvider, filling each unset domain — and each unset
// (memory, scope) pair — with a NoOpStorageProvider. The memory slot is always a
// *ScopedMemoryRouter so scoped reads/writes route by scope and
// StorageProvider.GetMemoriesMerged reaches the routing-layer merge.
func (c *CompositeStorageProvider) Build() *StorageProvider {
	noop := NoOpStorageProvider{}
	sp := &StorageProvider{
		session:       c.session,
		memory:        NewScopedMemoryRouter(c.memory),
		run:           c.run,
		observability: c.observability,
	}
	if sp.session == nil {
		sp.session = noop
	}
	if sp.run == nil {
		sp.run = noop
	}
	if sp.observability == nil {
		sp.observability = noop
	}
	return sp
}

// ============================================================================
// ScopedMemoryRouter (#78) — the (memory, scope) routing + merge layer
// ============================================================================

// ScopedMemoryRouter routes MemoryStore traffic to a per-StorageScope backend,
// filling unconfigured scopes with NoOpStorageProvider. It is the memory slot of
// a composite StorageProvider: StorageProvider.Memory() returns this router so a
// caller passing a scope is routed to the right backend, while
// StorageProvider.GetMemoriesMerged reaches the merge in the routing layer. Leaf
// backends stay scope-dumb — they never merge.
type ScopedMemoryRouter struct {
	byScope map[StorageScope]MemoryStore
	noop    MemoryStore
}

// NewScopedMemoryRouter builds a router from a partial scope->backend map; unset
// scopes resolve to a shared no-op. A nil map is treated as empty.
func NewScopedMemoryRouter(byScope map[StorageScope]MemoryStore) *ScopedMemoryRouter {
	cp := make(map[StorageScope]MemoryStore, len(byScope))
	for k, v := range byScope {
		if v != nil {
			cp[k] = v
		}
	}
	return &ScopedMemoryRouter{byScope: cp, noop: NoOpStorageProvider{}}
}

// backend returns the configured backend for scope, or the shared no-op.
func (r *ScopedMemoryRouter) backend(scope StorageScope) MemoryStore {
	if b, ok := r.byScope[scope]; ok {
		return b
	}
	return r.noop
}

// AppendMemory routes to the scope's backend (no-op when unconfigured).
func (r *ScopedMemoryRouter) AppendMemory(ctx context.Context, scope StorageScope, sessionID SessionID, entry MemoryEntry) error {
	return r.backend(scope).AppendMemory(ctx, scope, sessionID, entry)
}

// GetMemories routes to the scope's backend (empty when unconfigured).
func (r *ScopedMemoryRouter) GetMemories(ctx context.Context, scope StorageScope, sessionID SessionID, limit int) ([]MemoryEntry, error) {
	return r.backend(scope).GetMemories(ctx, scope, sessionID, limit)
}

var _ MemoryStore = (*ScopedMemoryRouter)(nil)

// ============================================================================
// Shared helpers
// ============================================================================

// mostRecentNewestFirst returns the most-recent limit items, newest-first,
// given a slice in append (oldest-first) order. A negative limit is treated as
// zero.
func mostRecentNewestFirst[T any](items []T, limit int) []T {
	if limit < 0 {
		limit = 0
	}
	n := len(items)
	if limit < n {
		n = limit
	}
	out := make([]T, 0, n)
	// Walk from the end (newest) backwards.
	for i := len(items) - 1; i >= 0 && len(out) < limit; i-- {
		out = append(out, items[i])
	}
	return out
}

// mergeNewestFirst is the merge step for the cross-scope memory read (#78 R6):
// sort newest-first by timestamp and truncate to limit. NO dedup — identical
// entries are all retained. A STABLE sort preserves the input order among equal
// timestamps, keeping the merge deterministic and byte-identical cross-language.
func mergeNewestFirst(entries []MemoryEntry, limit int) []MemoryEntry {
	sort.SliceStable(entries, func(i, j int) bool {
		return string(entries[i].Timestamp) > string(entries[j].Timestamp)
	})
	if limit < 0 {
		limit = 0
	}
	if limit < len(entries) {
		entries = entries[:limit]
	}
	return entries
}

// ============================================================================
// OTLP endpoint parsing (cross-language ground truth — see fan-out refactor)
// ============================================================================

// ParseOTLPEndpoints parses the comma-separated SPORE_OTLP_ENDPOINT value:
// strings.Split(",") , TrimSpace each segment, drop empty segments. This is the
// single most important cross-language fixture
// (fixtures/storage/otlp_endpoints_parse.json) and MUST be byte-identical in
// every language. It always returns a non-nil slice (possibly empty).
//
// The canonical implementation lives in the observability package (which the
// outbox fan-out also uses); this delegates so there is a single source of
// truth and no import cycle (storage → observability is the only legal
// direction).
func ParseOTLPEndpoints(raw string) []string {
	return observability.ParseOTLPEndpoints(raw)
}

// Compile-time interface checks.
var (
	_ fullProvider = NoOpStorageProvider{}
	_ fullProvider = (*InMemoryStorageProvider)(nil)
	_ fullProvider = (*FileSystemStorageProvider)(nil)
)
