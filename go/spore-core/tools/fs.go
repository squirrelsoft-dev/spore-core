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
	"strconv"
	"strings"

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
		Name: ReadFileToolName,
		Description: "Read a file's contents. Optionally read a line range " +
			"(offset is 1-indexed start, length is max lines, 0 = to EOF) " +
			"and/or prefix each line with its number via line_numbers. " +
			"With no optional params the whole file is returned verbatim.",
		Parameters: json.RawMessage(`{
			"type": "object",
			"properties": {
				"path": {"type": "string"},
				"offset": {
					"type": "integer",
					"description": "1-indexed start line (default 1)."
				},
				"length": {
					"type": "integer",
					"description": "Max lines to return; 0 = no limit / read to EOF (default 0)."
				},
				"line_numbers": {
					"type": "boolean",
					"description": "Prefix each returned line with its 1-indexed number (default false)."
				}
			},
			"required": ["path"]
		}`),
		Annotations: sporecore.ToolAnnotations{ReadOnly: true, Idempotent: true},
	}
}

func (t *ReadFileTool) Execute(ctx context.Context, call sporecore.ToolCall, sandbox sporecore.SandboxProvider, _ *sporecore.ToolContext) sporecore.ToolOutput {
	var params ReadFileParams
	if e := parseParams(call, &params); e != nil {
		return e.ToToolOutput()
	}
	resolved, v := sandbox.ResolvePath(ctx, params.Path, sporecore.OperationRead)
	if v != nil {
		return SandboxViolationError(v).ToToolOutput()
	}
	data, err := os.ReadFile(resolved)
	if err != nil {
		return ExecutionFailed(fmt.Sprintf("read failed: %s", err), true).ToToolOutput()
	}
	out, rangeErr := applyReadRange(string(data), &params)
	if rangeErr != "" {
		return sporecore.ToolOutput{Kind: sporecore.ToolOutputError, Message: rangeErr, Recoverable: true}
	}
	return finishWithPossibleTruncation(ctx, out, call.ID, sandbox)
}

// applyReadRange applies the #132 range/line-number transform to a fully-read
// file body. Returns the content to surface, or a non-empty error string for a
// recoverable error. With all params at their defaults the original content is
// returned unchanged (byte-identical to the pre-#132 behavior). Any non-default
// param prepends a "[lines {start}–{end} of {total}]\n" header (U+2013 en-dash).
func applyReadRange(content string, params *ReadFileParams) (out string, errMsg string) {
	isDefault := params.Offset == nil && params.Length == 0 && !params.LineNumbers
	if isDefault {
		return content, ""
	}
	// Offset == nil means "not provided" → default to 1.
	// Offset != nil && *Offset == 0 → recoverable error.
	var offset uint64
	if params.Offset == nil {
		offset = 1
	} else if *params.Offset == 0 {
		return "", "offset must be ≥ 1 (1-indexed)"
	} else {
		offset = *params.Offset
	}
	// Empty file: any params still yield empty content with no header.
	if content == "" {
		return "", ""
	}
	// strings.SplitAfter preserves each line's trailing '\n'; the final line
	// may or may not end in '\n'. This keeps each slice byte-faithful to the
	// source. Filter out a trailing empty string (produced when the file ends
	// with '\n', e.g. "a\nb\n" → ["a\n","b\n",""]).
	rawLines := strings.SplitAfter(content, "\n")
	lines := rawLines
	if len(lines) > 0 && lines[len(lines)-1] == "" {
		lines = lines[:len(lines)-1]
	}
	total := uint64(len(lines))
	if offset > total {
		return "", fmt.Sprintf("offset %d exceeds file length %d", offset, total)
	}
	end := total
	if params.Length != 0 {
		// offset + length - 1, clamped to total (length past EOF is silent).
		candidate := offset + params.Length - 1
		if candidate < total {
			end = candidate
		}
	}
	startIdx := int(offset - 1) // 0-based
	endIdx := int(end)          // exclusive

	var sb strings.Builder
	sb.WriteString(fmt.Sprintf("[lines %d–%d of %d]\n", offset, end, total))
	if params.LineNumbers {
		width := len(strconv.FormatUint(total, 10))
		for i, line := range lines[startIdx:endIdx] {
			n := offset + uint64(i)
			sb.WriteString(fmt.Sprintf("%*d | %s", width, n, line))
		}
	} else {
		for _, line := range lines[startIdx:endIdx] {
			sb.WriteString(line)
		}
	}
	return sb.String(), ""
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

func (t *WriteFileTool) Execute(ctx context.Context, call sporecore.ToolCall, sandbox sporecore.SandboxProvider, _ *sporecore.ToolContext) sporecore.ToolOutput {
	var params WriteFileParams
	if e := parseParams(call, &params); e != nil {
		return e.ToToolOutput()
	}
	resolved, v := sandbox.ResolvePath(ctx, params.Path, sporecore.OperationWrite)
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

func (t *ListDirTool) Execute(ctx context.Context, call sporecore.ToolCall, sandbox sporecore.SandboxProvider, _ *sporecore.ToolContext) sporecore.ToolOutput {
	var params ListDirParams
	if e := parseParams(call, &params); e != nil {
		return e.ToToolOutput()
	}
	resolved, v := sandbox.ResolvePath(ctx, params.Path, sporecore.OperationRead)
	if v != nil {
		return SandboxViolationError(v).ToToolOutput()
	}
	// Emit paths relative to the workspace root so each entry can be fed
	// straight back into read_file/write_file. The sandbox treats every input
	// path as root-relative, so absolute paths would not round-trip (see #93).
	// `resolved` is the absolute path of the listed directory (= root-relative
	// `params.Path`); each entry is under it. Relativize against `resolved`,
	// then re-anchor onto the caller-supplied (root-relative) `params.Path`.
	toRootRelative := func(entryAbsPath string) (string, bool) {
		relToListed, err := filepath.Rel(resolved, entryAbsPath)
		if err != nil {
			return "", false
		}
		// Skip the listed directory itself (WalkDir yields it first as ".").
		if relToListed == "." || relToListed == "" {
			return "", false
		}
		// Re-anchor onto the caller-supplied path, then drop a leading "./" so
		// "."/empty inputs yield bare names. filepath.Clean normalizes away the
		// "." component when params.Path is "." or empty.
		anchored := filepath.Clean(filepath.Join(params.Path, relToListed))
		return filepath.ToSlash(anchored), true
	}
	var entries []string
	if params.Recursive {
		err := filepath.WalkDir(resolved, func(path string, d fs.DirEntry, err error) error {
			if err != nil {
				return nil // best effort — skip errors
			}
			if rel, ok := toRootRelative(path); ok {
				entries = append(entries, rel)
			}
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
			if rel, ok := toRootRelative(filepath.Join(resolved, e.Name())); ok {
				entries = append(entries, rel)
			}
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

func (t *DeleteFileTool) Execute(ctx context.Context, call sporecore.ToolCall, sandbox sporecore.SandboxProvider, _ *sporecore.ToolContext) sporecore.ToolOutput {
	var params DeleteFileParams
	if e := parseParams(call, &params); e != nil {
		return e.ToToolOutput()
	}
	resolved, v := sandbox.ResolvePath(ctx, params.Path, sporecore.OperationWrite)
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

func (t *MoveFileTool) Execute(ctx context.Context, call sporecore.ToolCall, sandbox sporecore.SandboxProvider, _ *sporecore.ToolContext) sporecore.ToolOutput {
	var params MoveFileParams
	if e := parseParams(call, &params); e != nil {
		return e.ToToolOutput()
	}
	src, v := sandbox.ResolvePath(ctx, params.Src, sporecore.OperationWrite)
	if v != nil {
		return SandboxViolationError(v).ToToolOutput()
	}
	dst, v := sandbox.ResolvePath(ctx, params.Dst, sporecore.OperationWrite)
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
