// MarkdownMemoryProvider — a storage.MemoryStore that persists agent memory to a
// single human-readable memory.md file on disk.
//
// # What this demonstrates
//
// The storage seam. The harness is STATELESS — every byte of durable state lives
// behind a storage.StorageProvider. Memory is one of its four domains
// (storage.MemoryStore). This file implements ONLY that domain and composes it
// with storage.NoOpStorageProvider for the other three (session / run /
// observability) via IntoStorageProvider. main.go hands the composed provider's
// Memory() store to the harness builder's Storage seam; the harness then threads
// it into the built-in memory tool's ToolContext on every run. No custom harness
// plumbing — the seam is the whole integration surface.
//
// # The seam
//
// storage.MemoryStore is the swap point. The built-in FileSystemStorageProvider
// persists memory as a JSONL log; this provider persists the SAME MemoryEntry
// values to readable markdown instead. Same interface, same agent, same tool —
// different on-disk shape. Anything implementing storage.MemoryStore slots in
// here.
//
// # On-disk format (round-trips exactly)
//
// Each MemoryEntry is one markdown block. The header line carries the round-trip
// fields (scope, session, timestamp, role); the body is the content:
//
//	## [project] [project-ironwood] 2026-06-02T12:00:00Z — assistant
//
//	Postgres 15 is the system of record.
//
// AppendMemory writes such a block; GetMemories parses them back, filters by
// scope + session, sorts newest-first by timestamp, and takes limit. A
// hand-edited file (extra prose, blank lines, a human-added heading) is
// tolerated: anything that is not a recognized "## [scope] [session] timestamp —
// role" header is treated as body for the preceding entry, and leading prose
// before the first header is ignored.
//
// # Pinned-session-id requirement (read this)
//
// Memory is keyed by SessionID. The memory tool always uses the run's SessionID.
// For Run 2 (recall) to read Run 1's (store) memories, BOTH runs MUST use the
// SAME SessionID — see main.go, which pins SessionID("project-ironwood") rather
// than a generated one. This provider also stores the session id in each header
// so a single memory.md can hold multiple sessions without cross-talk.
package main

import (
	"context"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"

	"github.com/squirrelsoft-dev/spore-core/go/spore-core/storage"
)

// fileHeader is the human-readable preamble written when memory.md is first
// created. It is ignored on read (prose before the first entry header).
const fileHeader = "# Agent Memory\n\n" +
	"Human-readable working memory for this agent. " +
	"Each `##` block below is one remembered entry.\n"

// MarkdownMemoryProvider is a storage.MemoryStore backed by a single
// human-readable memory.md file.
//
// The mutex serializes the read-modify-write of AppendMemory so concurrent
// appends from the harness (which dispatches the memory tool sequentially anyway)
// never interleave a partial write.
type MarkdownMemoryProvider struct {
	path      string
	writeLock sync.Mutex
}

// NewMarkdownMemoryProvider creates a provider over memory.md at path. The file
// is created lazily on the first AppendMemory; a missing file reads as empty.
func NewMarkdownMemoryProvider(path string) *MarkdownMemoryProvider {
	return &MarkdownMemoryProvider{path: path}
}

// IntoStorageProvider composes this provider into a full *storage.StorageProvider:
// the real MemoryStore for the memory domain, NoOpStorageProvider for the other
// three. This is exactly what the example hands to the harness.
func (p *MarkdownMemoryProvider) IntoStorageProvider() *storage.StorageProvider {
	noop := storage.NoOpStorageProvider{}
	return storage.NewStorageProvider(noop, p, noop, noop)
}

// AppendMemory appends one MemoryEntry to memory.md as a markdown block,
// read-modify-write under the write lock. A missing file is created with the
// human-readable preamble first.
func (p *MarkdownMemoryProvider) AppendMemory(_ context.Context, scope storage.StorageScope, sessionID storage.SessionID, entry storage.MemoryEntry) error {
	p.writeLock.Lock()
	defer p.writeLock.Unlock()

	existing, err := os.ReadFile(p.path)
	if err != nil {
		if !errors.Is(err, fs.ErrNotExist) {
			return fmt.Errorf("read memory file %s: %w", p.path, err)
		}
		existing = []byte(fileHeader)
	}

	var b strings.Builder
	b.Write(existing)
	if !strings.HasSuffix(b.String(), "\n") {
		b.WriteByte('\n')
	}
	b.WriteByte('\n')
	b.WriteString(renderBlock(scope, sessionID, entry))

	if dir := filepath.Dir(p.path); dir != "" {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return fmt.Errorf("create memory dir %s: %w", dir, err)
		}
	}
	if err := os.WriteFile(p.path, []byte(b.String()), 0o644); err != nil {
		return fmt.Errorf("write memory file %s: %w", p.path, err)
	}
	return nil
}

// GetMemories parses memory.md, filters by (scope, session), sorts newest-first
// by timestamp, and returns the most-recent limit entries. A missing file reads
// as an empty slice. A negative limit is treated as zero.
func (p *MarkdownMemoryProvider) GetMemories(_ context.Context, scope storage.StorageScope, sessionID storage.SessionID, limit int) ([]storage.MemoryEntry, error) {
	contents, err := os.ReadFile(p.path)
	if err != nil {
		if errors.Is(err, fs.ErrNotExist) {
			return nil, nil
		}
		return nil, fmt.Errorf("read memory file %s: %w", p.path, err)
	}

	var entries []storage.MemoryEntry
	for _, blk := range parseFile(string(contents)) {
		if blk.scope == scope && blk.session == string(sessionID) {
			entries = append(entries, blk.entry)
		}
	}

	// Newest-first by timestamp. RFC-3339 strings sort lexically; a STABLE sort
	// keeps insertion (append) order among equal timestamps.
	sort.SliceStable(entries, func(i, j int) bool {
		return string(entries[i].Timestamp) > string(entries[j].Timestamp)
	})

	if limit < 0 {
		limit = 0
	}
	if limit < len(entries) {
		entries = entries[:limit]
	}
	return entries, nil
}

// GetMemoriesMerged delegates to the single shared storage.MergeMemories helper
// (User ∪ Project, newest-first, no dedup, Local excluded). Go interfaces cannot
// carry default methods, so every MemoryStore implementation delegates here —
// this provider does NOT reimplement the merge.
func (p *MarkdownMemoryProvider) GetMemoriesMerged(ctx context.Context, sessionID storage.SessionID, limit int) ([]storage.MemoryEntry, error) {
	return storage.MergeMemories(ctx, p, sessionID, limit)
}

// scopeToken renders a StorageScope as it is spelled in a header line.
func scopeToken(scope storage.StorageScope) string {
	switch scope {
	case storage.StorageScopeUser:
		return "user"
	case storage.StorageScopeProject:
		return "project"
	case storage.StorageScopeLocal:
		return "local"
	default:
		return string(scope)
	}
}

// parseScopeToken is the inverse of scopeToken; ok=false for an unrecognized
// token so a non-entry "## " line is tolerated rather than swallowed.
func parseScopeToken(s string) (storage.StorageScope, bool) {
	switch s {
	case "user":
		return storage.StorageScopeUser, true
	case "project":
		return storage.StorageScopeProject, true
	case "local":
		return storage.StorageScopeLocal, true
	default:
		return "", false
	}
}

// renderBlock renders one entry as a markdown block: header line, blank line,
// body, trailing newline. The session id is encoded so one file can hold
// multiple sessions.
func renderBlock(scope storage.StorageScope, sessionID storage.SessionID, entry storage.MemoryEntry) string {
	return fmt.Sprintf("## [%s] [%s] %s — %s\n\n%s\n",
		scopeToken(scope),
		string(sessionID),
		string(entry.Timestamp),
		entry.Role,
		strings.TrimRight(entry.Content, "\n"),
	)
}

// parsedBlock is a parsed entry plus the scope + session it was filed under.
type parsedBlock struct {
	scope   storage.StorageScope
	session string
	entry   storage.MemoryEntry
}

// header is a parsed entry header.
type header struct {
	scope   storage.StorageScope
	session string
	ts      string
	role    string
}

// emDash is the header separator: space, U+2014 (em dash), space.
const emDash = " — "

// parseHeader parses a line of the form "## [scope] [session] timestamp — role".
// ok=false for any line that is not a recognized header, so prose and hand-edits
// are tolerated.
func parseHeader(line string) (header, bool) {
	rest, ok := strings.CutPrefix(line, "## ")
	if !ok {
		return header{}, false
	}
	rest, ok = strings.CutPrefix(rest, "[")
	if !ok {
		return header{}, false
	}
	scopeStr, rest, ok := strings.Cut(rest, "] ")
	if !ok {
		return header{}, false
	}
	scope, ok := parseScopeToken(strings.TrimSpace(scopeStr))
	if !ok {
		return header{}, false
	}
	rest, ok = strings.CutPrefix(rest, "[")
	if !ok {
		return header{}, false
	}
	session, rest, ok := strings.Cut(rest, "] ")
	if !ok {
		return header{}, false
	}
	ts, role, ok := strings.Cut(rest, emDash)
	if !ok {
		return header{}, false
	}
	ts = strings.TrimSpace(ts)
	role = strings.TrimSpace(role)
	if ts == "" || role == "" {
		return header{}, false
	}
	return header{scope: scope, session: strings.TrimSpace(session), ts: ts, role: role}, true
}

// parseFile parses the whole file into blocks. Body lines accumulate under the
// most recent header; text before the first header (and any "## " line that is
// not a valid entry header) terminates the current block and is otherwise
// ignored, so hand-edited prose and headings never pollute an entry.
func parseFile(contents string) []parsedBlock {
	var blocks []parsedBlock
	var cur *header
	var body []string

	flush := func() {
		if cur != nil {
			blocks = append(blocks, finishBlock(*cur, body))
		}
		cur, body = nil, nil
	}

	for _, line := range strings.Split(contents, "\n") {
		if h, ok := parseHeader(line); ok {
			flush()
			hc := h
			cur = &hc
		} else if strings.HasPrefix(line, "## ") {
			// A "## " line that is NOT a valid entry header (e.g. a human-added
			// subheading) terminates the current block and is otherwise ignored.
			flush()
		} else if cur != nil {
			body = append(body, line)
		}
		// else: prose before the first header — ignored.
	}
	flush()
	return blocks
}

// finishBlock trims the blank lines bracketing the body and builds the entry.
func finishBlock(h header, body []string) parsedBlock {
	content := strings.TrimSpace(strings.Join(body, "\n"))
	return parsedBlock{
		scope:   h.scope,
		session: h.session,
		entry:   storage.NewMemoryEntry(h.role, content, storage.Timestamp(h.ts)),
	}
}

// Compile-time check: this provider IS a storage.MemoryStore (the swap point).
var _ storage.MemoryStore = (*MarkdownMemoryProvider)(nil)
