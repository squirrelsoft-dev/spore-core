// FileSystemStorageProvider — issue #73 disk-backed storage.
//
// Layout mirrors .spore/:
//   - session → {root}/sessions/{id}/state.json     (atomic write-rename)
//   - run     → {root}/sessions/{id}/run/{key}.json  (atomic write-rename)
//   - memory  → {root}/sessions/{id}/memory.jsonl    (append)
//   - obs     → {root}/sessions/{id}/trace.jsonl     (append)
//
// FlushSession creates a sibling .flushed marker.
//
// Atomic write-rename (byte-identical algorithm across all four languages):
// ensure the parent dir, write full bytes to a sibling {target}.tmp, flush +
// Sync, then os.Rename(tmp, target). On any failure the .tmp is removed so no
// partial sidecar is left behind. Last-writer-wins via rename; no per-key
// locking contract.
package storage

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// FileSystemStorageProvider is a disk-backed StorageProvider rooted at root.
type FileSystemStorageProvider struct {
	root string
}

// NewFileSystemStorageProvider constructs a disk-backed provider rooted at
// root.
func NewFileSystemStorageProvider(root string) *FileSystemStorageProvider {
	return &FileSystemStorageProvider{root: root}
}

// Root returns the provider's root directory.
func (p *FileSystemStorageProvider) Root() string { return p.root }

func (p *FileSystemStorageProvider) sessionDir(id SessionID) string {
	return filepath.Join(p.root, "sessions", string(id))
}
func (p *FileSystemStorageProvider) statePath(id SessionID) string {
	return filepath.Join(p.sessionDir(id), "state.json")
}
func (p *FileSystemStorageProvider) runDir(id SessionID) string {
	return filepath.Join(p.sessionDir(id), "run")
}
func (p *FileSystemStorageProvider) runPath(id SessionID, key string) string {
	return filepath.Join(p.runDir(id), key+".json")
}
func (p *FileSystemStorageProvider) memoryPath(id SessionID) string {
	return filepath.Join(p.sessionDir(id), "memory.jsonl")
}
func (p *FileSystemStorageProvider) tracePath(id SessionID) string {
	return filepath.Join(p.sessionDir(id), "trace.jsonl")
}

// atomicWrite writes bytes to target via a sibling {target}.tmp, fsyncs, then
// renames. On any failure the .tmp is removed so no partial sidecar leaks.
func atomicWrite(target string, b []byte) error {
	if dir := filepath.Dir(target); dir != "" {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return err
		}
	}
	tmp := target + ".tmp"
	if err := func() error {
		f, err := os.Create(tmp)
		if err != nil {
			return err
		}
		// Close once; on the happy path Close happens before Rename.
		if _, err := f.Write(b); err != nil {
			_ = f.Close()
			return err
		}
		if err := f.Sync(); err != nil {
			_ = f.Close()
			return err
		}
		if err := f.Close(); err != nil {
			return err
		}
		return os.Rename(tmp, target)
	}(); err != nil {
		_ = os.Remove(tmp)
		return err
	}
	return nil
}

// appendJSONL appends one JSONL line (the marshaled value plus a trailing \n),
// syncing the handle.
func appendJSONL(path string, b []byte) error {
	if dir := filepath.Dir(path); dir != "" {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return err
		}
	}
	f, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	if err != nil {
		return err
	}
	defer f.Close()
	line := append(append([]byte{}, b...), '\n')
	if _, err := f.Write(line); err != nil {
		return err
	}
	return f.Sync()
}

// readJSONL reads every non-empty JSONL line from path as raw bytes. A missing
// file yields an empty slice (no error).
func readJSONL(path string) ([]json.RawMessage, error) {
	f, err := os.Open(path)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return nil, nil
		}
		return nil, err
	}
	defer f.Close()

	var out []json.RawMessage
	reader := bufio.NewReader(f)
	for {
		line, readErr := reader.ReadString('\n')
		trimmed := strings.TrimSpace(line)
		if trimmed != "" {
			out = append(out, json.RawMessage([]byte(trimmed)))
		}
		if readErr != nil {
			if readErr == io.EOF {
				break
			}
			return nil, readErr
		}
	}
	return out, nil
}

// SessionStore.

func (p *FileSystemStorageProvider) GetSession(_ context.Context, id SessionID) (*PausedState, bool, error) {
	b, err := os.ReadFile(p.statePath(id))
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return nil, false, nil
		}
		return nil, false, err
	}
	var state PausedState
	if err := json.Unmarshal(b, &state); err != nil {
		return nil, false, err
	}
	return &state, true, nil
}

func (p *FileSystemStorageProvider) PutSession(_ context.Context, id SessionID, state *PausedState) error {
	b, err := json.Marshal(state)
	if err != nil {
		return err
	}
	return atomicWrite(p.statePath(id), b)
}

func (p *FileSystemStorageProvider) DeleteSession(_ context.Context, id SessionID) error {
	if err := os.Remove(p.statePath(id)); err != nil && !errors.Is(err, os.ErrNotExist) {
		return err
	}
	return nil
}

func (p *FileSystemStorageProvider) ListSessions(_ context.Context) ([]SessionID, error) {
	entries, err := os.ReadDir(filepath.Join(p.root, "sessions"))
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return nil, nil
		}
		return nil, err
	}
	var out []SessionID
	for _, e := range entries {
		if statePath := filepath.Join(p.root, "sessions", e.Name(), "state.json"); fsExists(statePath) {
			out = append(out, SessionID(e.Name()))
		}
	}
	sort.Slice(out, func(i, j int) bool { return out[i] < out[j] })
	return out, nil
}

// MemoryStore.
//
// The FS backend is SCOPE-DUMB (#78): the user-scope backend is pointed at the
// already-partitioned {user_root}/projects/{workspace_id} at construction, so it
// just writes under whatever root it was given. The scope argument is ignored at
// the leaf — the CompositeStorageProvider's ScopedMemoryRouter is what isolates
// scopes by routing each to its own backend.

func (p *FileSystemStorageProvider) AppendMemory(_ context.Context, _ StorageScope, sessionID SessionID, entry MemoryEntry) error {
	b, err := json.Marshal(entry)
	if err != nil {
		return err
	}
	return appendJSONL(p.memoryPath(sessionID), b)
}

func (p *FileSystemStorageProvider) GetMemories(_ context.Context, _ StorageScope, sessionID SessionID, limit int) ([]MemoryEntry, error) {
	lines, err := readJSONL(p.memoryPath(sessionID))
	if err != nil {
		return nil, err
	}
	entries := make([]MemoryEntry, 0, len(lines))
	for _, l := range lines {
		var e MemoryEntry
		if err := json.Unmarshal(l, &e); err != nil {
			return nil, err
		}
		entries = append(entries, e)
	}
	return mostRecentNewestFirst(entries, limit), nil
}

// GetMemoriesMerged delegates to the shared merge helper (#82 D2). A lone FS
// backend is scope-dumb, so both scopes resolve to the same file; the canonical
// multi-scope wiring routes each scope to its own backend via the
// CompositeStorageProvider's ScopedMemoryRouter.
func (p *FileSystemStorageProvider) GetMemoriesMerged(ctx context.Context, sessionID SessionID, limit int) ([]MemoryEntry, error) {
	return MergeMemories(ctx, p, sessionID, limit)
}

// RunStore.

func (p *FileSystemStorageProvider) Get(_ context.Context, sessionID SessionID, key string) (json.RawMessage, bool, error) {
	b, err := os.ReadFile(p.runPath(sessionID, key))
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return nil, false, nil
		}
		return nil, false, err
	}
	// Normalize through a JSON round-trip so the returned bytes are canonical
	// (parity with a store that decodes then re-encodes); this keeps opaque
	// values comparable across in-memory and filesystem backends.
	var v json.RawMessage
	if err := json.Unmarshal(b, &v); err != nil {
		return nil, false, err
	}
	return v, true, nil
}

func (p *FileSystemStorageProvider) Put(_ context.Context, sessionID SessionID, key string, value json.RawMessage) error {
	return atomicWrite(p.runPath(sessionID, key), []byte(value))
}

func (p *FileSystemStorageProvider) Delete(_ context.Context, sessionID SessionID, key string) error {
	if err := os.Remove(p.runPath(sessionID, key)); err != nil && !errors.Is(err, os.ErrNotExist) {
		return err
	}
	return nil
}

func (p *FileSystemStorageProvider) ListKeys(_ context.Context, sessionID SessionID) ([]string, error) {
	entries, err := os.ReadDir(p.runDir(sessionID))
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return nil, nil
		}
		return nil, err
	}
	var out []string
	for _, e := range entries {
		if key, ok := strings.CutSuffix(e.Name(), ".json"); ok {
			out = append(out, key)
		}
	}
	sort.Strings(out)
	return out, nil
}

// ObservabilityStore.

func (p *FileSystemStorageProvider) AppendSpan(_ context.Context, sessionID SessionID, span json.RawMessage) error {
	return appendJSONL(p.tracePath(sessionID), []byte(span))
}

func (p *FileSystemStorageProvider) GetSpans(_ context.Context, sessionID SessionID) ([]json.RawMessage, error) {
	return readJSONL(p.tracePath(sessionID))
}

// GetSessions returns empty: SessionMetrics roll-up is owned by the
// ObservabilityProvider, not the raw on-disk span store.
func (p *FileSystemStorageProvider) GetSessions(context.Context, Timestamp, *string, *SessionOutcome) ([]SessionMetrics, error) {
	return nil, nil
}

// FlushSession creates the sibling .flushed marker.
func (p *FileSystemStorageProvider) FlushSession(_ context.Context, sessionID SessionID) error {
	dir := p.sessionDir(sessionID)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	f, err := os.Create(filepath.Join(dir, ".flushed"))
	if err != nil {
		return err
	}
	return f.Close()
}

func fsExists(path string) bool {
	_, err := os.Stat(path)
	return err == nil
}
