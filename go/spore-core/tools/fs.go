// Filesystem tools: ReadFile, WriteFile, ListDir, DeleteFile, MoveFile.

package tools

import (
	"context"
	"encoding/json"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"sort"

	sporecore "github.com/squirrelsoft-dev/spore-core/go/spore-core"
)

// ============================================================================
// ReadFile
// ============================================================================

type ReadFileTool struct{}

func NewReadFileTool() *ReadFileTool { return &ReadFileTool{} }

const ReadFileToolName = "read_file"

func (*ReadFileTool) Name() string                { return ReadFileToolName }
func (*ReadFileTool) IsSubagentTool() bool        { return false }
func (*ReadFileTool) MayProduceLargeOutput() bool { return true }

func (*ReadFileTool) Schema() sporecore.RegistryToolSchema {
	return sporecore.RegistryToolSchema{
		Name:        ReadFileToolName,
		Description: "Read a file's contents",
		Parameters: json.RawMessage(`{
			"type": "object",
			"properties": {"path": {"type": "string"}},
			"required": ["path"]
		}`),
		Annotations: sporecore.ToolAnnotations{ReadOnly: true, Idempotent: true},
	}
}

func (t *ReadFileTool) Execute(ctx context.Context, call sporecore.ToolCall, sandbox sporecore.SandboxProvider) sporecore.ToolOutput {
	var params ReadFileParams
	if e := parseParams(call, &params); e != nil {
		return e.ToToolOutput()
	}
	resolved, v := sandbox.ResolvePath(ctx, params.Path)
	if v != nil {
		return SandboxViolationError(v).ToToolOutput()
	}
	data, err := os.ReadFile(resolved)
	if err != nil {
		return ExecutionFailed(fmt.Sprintf("read failed: %s", err), true).ToToolOutput()
	}
	return finishWithPossibleTruncation(ctx, string(data), call.ID, sandbox)
}

// ============================================================================
// WriteFile
// ============================================================================

type WriteFileTool struct{}

func NewWriteFileTool() *WriteFileTool { return &WriteFileTool{} }

const WriteFileToolName = "write_file"

func (*WriteFileTool) Name() string                { return WriteFileToolName }
func (*WriteFileTool) IsSubagentTool() bool        { return false }
func (*WriteFileTool) MayProduceLargeOutput() bool { return false }

func (*WriteFileTool) Schema() sporecore.RegistryToolSchema {
	return sporecore.RegistryToolSchema{
		Name:        WriteFileToolName,
		Description: "Write content to a file (overwrites by default; set append=true to append)",
		Parameters: json.RawMessage(`{
			"type": "object",
			"properties": {
				"path": {"type": "string"},
				"content": {"type": "string"},
				"append": {"type": "boolean"}
			},
			"required": ["path", "content"]
		}`),
		Annotations: sporecore.ToolAnnotations{Destructive: true},
	}
}

func (t *WriteFileTool) Execute(ctx context.Context, call sporecore.ToolCall, sandbox sporecore.SandboxProvider) sporecore.ToolOutput {
	var params WriteFileParams
	if e := parseParams(call, &params); e != nil {
		return e.ToToolOutput()
	}
	resolved, v := sandbox.ResolvePath(ctx, params.Path)
	if v != nil {
		return SandboxViolationError(v).ToToolOutput()
	}
	var err error
	if params.Append {
		f, openErr := os.OpenFile(resolved, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o644)
		if openErr != nil {
			err = openErr
		} else {
			_, err = f.WriteString(params.Content)
			_ = f.Close()
		}
	} else {
		err = os.WriteFile(resolved, []byte(params.Content), 0o644)
	}
	if err != nil {
		return ExecutionFailed(fmt.Sprintf("write failed: %s", err), true).ToToolOutput()
	}
	return sporecore.ToolOutput{
		Kind:    sporecore.ToolOutputSuccess,
		Content: fmt.Sprintf("wrote %d bytes to %s", len(params.Content), params.Path),
	}
}

// ============================================================================
// ListDir
// ============================================================================

type ListDirTool struct{}

func NewListDirTool() *ListDirTool { return &ListDirTool{} }

const ListDirToolName = "list_dir"

func (*ListDirTool) Name() string                { return ListDirToolName }
func (*ListDirTool) IsSubagentTool() bool        { return false }
func (*ListDirTool) MayProduceLargeOutput() bool { return false }

func (*ListDirTool) Schema() sporecore.RegistryToolSchema {
	return sporecore.RegistryToolSchema{
		Name:        ListDirToolName,
		Description: "List directory entries (optionally recursive)",
		Parameters: json.RawMessage(`{
			"type": "object",
			"properties": {
				"path": {"type": "string"},
				"recursive": {"type": "boolean"}
			},
			"required": ["path"]
		}`),
		Annotations: sporecore.ToolAnnotations{ReadOnly: true},
	}
}

func (t *ListDirTool) Execute(ctx context.Context, call sporecore.ToolCall, sandbox sporecore.SandboxProvider) sporecore.ToolOutput {
	var params ListDirParams
	if e := parseParams(call, &params); e != nil {
		return e.ToToolOutput()
	}
	resolved, v := sandbox.ResolvePath(ctx, params.Path)
	if v != nil {
		return SandboxViolationError(v).ToToolOutput()
	}
	var entries []string
	if params.Recursive {
		err := filepath.WalkDir(resolved, func(path string, d fs.DirEntry, err error) error {
			if err != nil {
				return nil // best effort — skip errors
			}
			entries = append(entries, path)
			return nil
		})
		if err != nil {
			return ExecutionFailed(fmt.Sprintf("walk failed: %s", err), true).ToToolOutput()
		}
	} else {
		ents, err := os.ReadDir(resolved)
		if err != nil {
			return ExecutionFailed(fmt.Sprintf("read_dir failed: %s", err), true).ToToolOutput()
		}
		for _, e := range ents {
			entries = append(entries, filepath.Join(resolved, e.Name()))
		}
	}
	sort.Strings(entries)
	content := ""
	for i, e := range entries {
		if i > 0 {
			content += "\n"
		}
		content += e
	}
	if len(content) > LargeOutputThreshold {
		return finishWithPossibleTruncation(ctx, content, call.ID, sandbox)
	}
	return sporecore.ToolOutput{Kind: sporecore.ToolOutputSuccess, Content: content}
}

// ============================================================================
// DeleteFile
// ============================================================================

type DeleteFileTool struct{}

func NewDeleteFileTool() *DeleteFileTool { return &DeleteFileTool{} }

const DeleteFileToolName = "delete_file"

func (*DeleteFileTool) Name() string                { return DeleteFileToolName }
func (*DeleteFileTool) IsSubagentTool() bool        { return false }
func (*DeleteFileTool) MayProduceLargeOutput() bool { return false }

func (*DeleteFileTool) Schema() sporecore.RegistryToolSchema {
	return sporecore.RegistryToolSchema{
		Name:        DeleteFileToolName,
		Description: "Delete a file",
		Parameters: json.RawMessage(`{
			"type": "object",
			"properties": {"path": {"type": "string"}},
			"required": ["path"]
		}`),
		Annotations: sporecore.ToolAnnotations{Destructive: true},
	}
}

func (t *DeleteFileTool) Execute(ctx context.Context, call sporecore.ToolCall, sandbox sporecore.SandboxProvider) sporecore.ToolOutput {
	var params DeleteFileParams
	if e := parseParams(call, &params); e != nil {
		return e.ToToolOutput()
	}
	resolved, v := sandbox.ResolvePath(ctx, params.Path)
	if v != nil {
		return SandboxViolationError(v).ToToolOutput()
	}
	if err := os.Remove(resolved); err != nil {
		return ExecutionFailed(fmt.Sprintf("delete failed: %s", err), true).ToToolOutput()
	}
	return sporecore.ToolOutput{
		Kind:    sporecore.ToolOutputSuccess,
		Content: fmt.Sprintf("deleted %s", params.Path),
	}
}

// ============================================================================
// MoveFile
// ============================================================================

type MoveFileTool struct{}

func NewMoveFileTool() *MoveFileTool { return &MoveFileTool{} }

const MoveFileToolName = "move_file"

func (*MoveFileTool) Name() string                { return MoveFileToolName }
func (*MoveFileTool) IsSubagentTool() bool        { return false }
func (*MoveFileTool) MayProduceLargeOutput() bool { return false }

func (*MoveFileTool) Schema() sporecore.RegistryToolSchema {
	return sporecore.RegistryToolSchema{
		Name:        MoveFileToolName,
		Description: "Move/rename a file",
		Parameters: json.RawMessage(`{
			"type": "object",
			"properties": {
				"src": {"type": "string"},
				"dst": {"type": "string"}
			},
			"required": ["src", "dst"]
		}`),
		Annotations: sporecore.ToolAnnotations{Destructive: true},
	}
}

func (t *MoveFileTool) Execute(ctx context.Context, call sporecore.ToolCall, sandbox sporecore.SandboxProvider) sporecore.ToolOutput {
	var params MoveFileParams
	if e := parseParams(call, &params); e != nil {
		return e.ToToolOutput()
	}
	src, v := sandbox.ResolvePath(ctx, params.Src)
	if v != nil {
		return SandboxViolationError(v).ToToolOutput()
	}
	dst, v := sandbox.ResolvePath(ctx, params.Dst)
	if v != nil {
		return SandboxViolationError(v).ToToolOutput()
	}
	if err := os.Rename(src, dst); err != nil {
		return ExecutionFailed(fmt.Sprintf("move failed: %s", err), true).ToToolOutput()
	}
	return sporecore.ToolOutput{
		Kind:    sporecore.ToolOutputSuccess,
		Content: fmt.Sprintf("moved %s -> %s", params.Src, params.Dst),
	}
}

// Compile-time interface checks.
var (
	_ sporecore.Tool = (*ReadFileTool)(nil)
	_ sporecore.Tool = (*WriteFileTool)(nil)
	_ sporecore.Tool = (*ListDirTool)(nil)
	_ sporecore.Tool = (*DeleteFileTool)(nil)
	_ sporecore.Tool = (*MoveFileTool)(nil)
)
