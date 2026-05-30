// Memory tool (#82, storage seam #78): the scope-aware read/write tool over the
// persisted episodic MemoryStore.
//
// One tool, MemoryTool (NAME = "memory"), dispatched on an `operation`
// discriminator (write, read). It is the agent-facing surface over the scoped
// memory seam shipped in #78.
//
// # Storage seam (#78, opaque marker interface)
//
// ToolContext.MemoryStore is the OPAQUE marker interface
// sporecore.ToolMemoryStore (a documented #78 divergence to dodge an import
// cycle: the root sporecore package cannot name promptassembly.StorageScope, so
// it cannot mirror storage.MemoryStore structurally). This tool reaches the real
// store by asserting MemoryStore back to storage.MemoryStore here in the
// storage-aware tools package — exactly how the existing memProbeTool and the
// ToolRunStore seam are consumed. A nil / non-asserting store is treated as a
// no-op (reads return empty, writes are recoverable no-ops via the no-op store).
//
// # Operations
//
//   - write {scope, role, content, metadata?}: append one MemoryEntry to scope
//     and return the serialized just-written entry as the success content
//     (decision A). metadata defaults to {} (decision C).
//   - read {scope, merged?=false, limit?=50}: return the most-recent `limit`
//     entries newest-first for scope (decision B). With merged=true, return the
//     cross-scope User ∪ Project merge (decision D2) via the store's single
//     GetMemoriesMerged merge.
//
// # Rules enforced
//
//   - R1 write→read roundtrip; R2 write returns the serialized entry; R3 read
//     default limit 50; R4 metadata preserved verbatim, default {}; R5 non-merged
//     scope isolation; R6 merged read via the single merge; R7 Local rejected on
//     BOTH ops with the EXACT message, checked BEFORE any storage access (nothing
//     written); R8 bad params recoverable; R9 storage error recoverable; R10 read
//     does not write.
//
// # Annotations (decision E)
//
// NOT annotated ReadOnly. A read-only tool is run CONCURRENTLY by DispatchAll
// and could race the shared append; like TaskListTool this tool leaves all
// annotations false so the registry dispatches it sequentially.
//
// # Known v1 limitation (#78 Q7)
//
// Memory is SessionID-keyed for v1: the tool always uses ctx's SessionID and
// offers NO cross-session addressing param. v2 should add session-independent
// keying — do not introduce it here.

package tools

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	sporecore "github.com/squirrelsoft-dev/spore-core/go/spore-core"
	"github.com/squirrelsoft-dev/spore-core/go/spore-core/storage"
)

// MemoryToolName is the registered tool name.
const MemoryToolName = "memory"

// LocalScopeRejectedMessage is the EXACT recoverable-error message returned when
// a Local-scoped op is attempted (rejected on both read and write).
const LocalScopeRejectedMessage = "Local scope is not supported by MemoryTool — use User or Project."

// MemoryTool is the single scope-aware read/write tool over episodic memory.
type MemoryTool struct{}

// NewMemoryTool constructs a MemoryTool.
func NewMemoryTool() *MemoryTool { return &MemoryTool{} }

func (*MemoryTool) Name() string                { return MemoryToolName }
func (*MemoryTool) IsSubagentTool() bool        { return false }
func (*MemoryTool) MayProduceLargeOutput() bool { return false }

func (*MemoryTool) Schema() sporecore.RegistryToolSchema {
	// Properties kept sorted/stable for cache stability. `scope` advertises only
	// user/project — `local` is rejected at runtime but intentionally omitted
	// from the advertised enum.
	return sporecore.RegistryToolSchema{
		Name:        MemoryToolName,
		Description: "Read or write scope-aware episodic memory for this session",
		Parameters: json.RawMessage(`{
			"type": "object",
			"properties": {
				"content": {"type": "string"},
				"limit": {"type": "integer"},
				"merged": {"type": "boolean"},
				"metadata": {"type": "object"},
				"operation": {
					"type": "string",
					"enum": ["read", "write"]
				},
				"role": {"type": "string"},
				"scope": {
					"type": "string",
					"enum": ["project", "user"]
				}
			},
			"required": ["operation", "scope"]
		}`),
		// Intentionally NOT ReadOnly: the shared append must dispatch
		// sequentially. See package docs (decision E).
		Annotations: sporecore.ToolAnnotations{},
	}
}

func (t *MemoryTool) Execute(ctx context.Context, call sporecore.ToolCall, _ sporecore.SandboxProvider, toolCtx *sporecore.ToolContext) sporecore.ToolOutput {
	// 1. Parse params (bad input → recoverable).
	var params MemoryToolParams
	if e := parseParams(call, &params); e != nil {
		return e.ToToolOutput()
	}
	if e := validateMemoryParams(&params); e != nil {
		return e.ToToolOutput()
	}

	// 2. Reject Local on BOTH ops BEFORE touching storage (R7 — nothing written).
	if params.Scope == StorageScopeLocal {
		return ExecutionFailed(LocalScopeRejectedMessage, true).ToToolOutput()
	}

	store := memoryStoreFrom(toolCtx)
	sessionID := sporecore.SessionID("")
	if toolCtx != nil {
		sessionID = toolCtx.SessionID
	}

	switch params.Operation {
	case MemoryOperationWrite:
		entry := storage.MemoryEntry{
			Role:      params.Role,
			Content:   params.Content,
			Timestamp: nowTimestamp(),
			Metadata:  json.RawMessage(params.Metadata),
		}
		if err := store.AppendMemory(ctx, params.Scope, sessionID, entry); err != nil {
			return ExecutionFailed(fmt.Sprintf("could not append memory: %s", err), true).ToToolOutput()
		}
		// R2 (decision A): success content = the serialized just-written entry.
		content, err := json.Marshal(entry)
		if err != nil {
			return ExecutionFailed(fmt.Sprintf("could not serialize memory entry: %s", err), true).ToToolOutput()
		}
		return sporecore.ToolOutput{Kind: sporecore.ToolOutputSuccess, Content: string(content)}

	case MemoryOperationRead:
		limit := MemoryDefaultReadLimit
		if params.Limit != nil {
			limit = *params.Limit
		}
		var (
			entries []storage.MemoryEntry
			err     error
		)
		if params.Merged {
			// R6 (decision D2): merged read drives the single merge.
			entries, err = store.GetMemoriesMerged(ctx, sessionID, limit)
		} else {
			// Scoped read (R5 isolation). R10: read never appends.
			entries, err = store.GetMemories(ctx, params.Scope, sessionID, limit)
		}
		if err != nil {
			return ExecutionFailed(fmt.Sprintf("could not read memory: %s", err), true).ToToolOutput()
		}
		if entries == nil {
			entries = []storage.MemoryEntry{}
		}
		content, err := json.Marshal(entries)
		if err != nil {
			return ExecutionFailed(fmt.Sprintf("could not serialize memory entries: %s", err), true).ToToolOutput()
		}
		return sporecore.ToolOutput{Kind: sporecore.ToolOutputSuccess, Content: string(content)}
	}

	// Unreachable: validateMemoryParams rejects any other operation.
	return InvalidParameters(fmt.Sprintf("unknown operation %q", params.Operation)).ToToolOutput()
}

// validateMemoryParams enforces the per-operation field requirements that the
// internally-tagged Rust enum gets for free from serde. An unknown operation, a
// missing required field, or a missing scope is an InvalidParameters error
// (recoverable). Scope value validity (Local) is checked separately in Execute.
func validateMemoryParams(p *MemoryToolParams) *ToolExecutionError {
	if p.Scope == "" {
		return InvalidParameters("missing required field `scope`")
	}
	switch p.Operation {
	case MemoryOperationWrite:
		if p.Role == "" {
			return InvalidParameters("write requires `role`")
		}
		if p.Content == "" {
			return InvalidParameters("write requires `content`")
		}
	case MemoryOperationRead:
		// No additional required fields (merged/limit are optional).
	case "":
		return InvalidParameters("missing required field `operation`")
	default:
		return InvalidParameters(fmt.Sprintf("unknown operation %q", p.Operation))
	}
	return nil
}

// memoryStoreFrom asserts the opaque ToolContext.MemoryStore seam back to the
// real storage.MemoryStore (the documented #78 divergence — same pattern as the
// ToolRunStore seam). A nil context, a nil seam, or a value that does not
// implement storage.MemoryStore resolves to a no-op store so the tool never
// panics: reads return empty, writes are discarded.
func memoryStoreFrom(toolCtx *sporecore.ToolContext) storage.MemoryStore {
	if toolCtx == nil || toolCtx.MemoryStore == nil {
		return storage.NoOpStorageProvider{}
	}
	if ms, ok := toolCtx.MemoryStore.(storage.MemoryStore); ok {
		return ms
	}
	return storage.NoOpStorageProvider{}
}

// nowTimestamp returns the wall-clock timestamp stamped on a freshly-written
// entry: RFC-3339 UTC at seconds precision (YYYY-MM-DDTHH:MM:SSZ), matching the
// Timestamp string shape the fixtures use and the cross-language reference.
func nowTimestamp() storage.Timestamp {
	return storage.Timestamp(time.Now().UTC().Format("2006-01-02T15:04:05Z"))
}

// Compile-time interface check.
var _ sporecore.Tool = (*MemoryTool)(nil)
