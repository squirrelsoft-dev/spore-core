// Search tools: GrepFiles, FindFiles.

package tools

import (
	"context"
	"encoding/json"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"regexp"
	"sort"

	sporecore "github.com/squirrelsoft-dev/spore-core/go/spore-core"
)

// ============================================================================
// GrepFiles
// ============================================================================

type GrepFilesTool struct{}

func NewGrepFilesTool() *GrepFilesTool { return &GrepFilesTool{} }

const GrepFilesToolName = "grep_files"

func (*GrepFilesTool) Name() string                { return GrepFilesToolName }
func (*GrepFilesTool) IsSubagentTool() bool        { return false }
func (*GrepFilesTool) MayProduceLargeOutput() bool { return true }

func (*GrepFilesTool) Schema() sporecore.RegistryToolSchema {
	return sporecore.RegistryToolSchema{
		Name:        GrepFilesToolName,
		Description: "Search files for a regex pattern",
		Parameters: json.RawMessage(`{
			"type": "object",
			"properties": {
				"pattern": {"type": "string"},
				"path": {"type": "string"},
				"recursive": {"type": "boolean"}
			},
			"required": ["pattern", "path"]
		}`),
		Annotations: sporecore.ToolAnnotations{ReadOnly: true},
	}
}

type grepHit struct {
	path string
	line int
	text string
}

func scanFile(path string, re *regexp.Regexp, out *[]grepHit) {
	data, err := os.ReadFile(path)
	if err != nil {
		return
	}
	lineNo := 0
	for _, line := range splitLines(string(data)) {
		lineNo++
		if re.MatchString(line) {
			*out = append(*out, grepHit{path: path, line: lineNo, text: line})
		}
	}
}

func splitLines(s string) []string {
	var out []string
	start := 0
	for i := 0; i < len(s); i++ {
		if s[i] == '\n' {
			out = append(out, s[start:i])
			start = i + 1
		}
	}
	if start < len(s) {
		out = append(out, s[start:])
	}
	return out
}

func (t *GrepFilesTool) Execute(ctx context.Context, call sporecore.ToolCall, sandbox sporecore.SandboxProvider) sporecore.ToolOutput {
	var params GrepFilesParams
	if e := parseParams(call, &params); e != nil {
		return e.ToToolOutput()
	}
	re, err := regexp.Compile(params.Pattern)
	if err != nil {
		return InvalidParameters(fmt.Sprintf("invalid regex: %s", err)).ToToolOutput()
	}
	root, v := sandbox.ResolvePath(ctx, params.Path, sporecore.OperationRead)
	if v != nil {
		return SandboxViolationError(v).ToToolOutput()
	}
	var hits []grepHit
	info, statErr := os.Stat(root)
	if statErr != nil {
		// Treat as empty result — parity with Rust behaviour.
		return finishWithPossibleTruncation(ctx, "", call.ID, sandbox)
	}
	if params.Recursive {
		_ = filepath.WalkDir(root, func(p string, d fs.DirEntry, e error) error {
			if e != nil || d == nil {
				return nil
			}
			if d.Type().IsRegular() {
				scanFile(p, re, &hits)
			}
			return nil
		})
	} else if info.Mode().IsRegular() {
		scanFile(root, re, &hits)
	} else {
		entries, e := os.ReadDir(root)
		if e == nil {
			for _, ent := range entries {
				p := filepath.Join(root, ent.Name())
				fi, _ := os.Stat(p)
				if fi != nil && fi.Mode().IsRegular() {
					scanFile(p, re, &hits)
				}
			}
		}
	}
	sort.Slice(hits, func(i, j int) bool {
		if hits[i].path != hits[j].path {
			return hits[i].path < hits[j].path
		}
		return hits[i].line < hits[j].line
	})
	body := ""
	for i, h := range hits {
		if i > 0 {
			body += "\n"
		}
		body += fmt.Sprintf("%s:%d:%s", h.path, h.line, h.text)
	}
	return finishWithPossibleTruncation(ctx, body, call.ID, sandbox)
}

// ============================================================================
// FindFiles
// ============================================================================

type FindFilesTool struct{}

func NewFindFilesTool() *FindFilesTool { return &FindFilesTool{} }

const FindFilesToolName = "find_files"

func (*FindFilesTool) Name() string                { return FindFilesToolName }
func (*FindFilesTool) IsSubagentTool() bool        { return false }
func (*FindFilesTool) MayProduceLargeOutput() bool { return true }

func (*FindFilesTool) Schema() sporecore.RegistryToolSchema {
	return sporecore.RegistryToolSchema{
		Name:        FindFilesToolName,
		Description: "Find files matching a glob",
		Parameters: json.RawMessage(`{
			"type": "object",
			"properties": {
				"glob": {"type": "string"},
				"path": {"type": "string"}
			},
			"required": ["glob", "path"]
		}`),
		Annotations: sporecore.ToolAnnotations{ReadOnly: true},
	}
}

func (t *FindFilesTool) Execute(ctx context.Context, call sporecore.ToolCall, sandbox sporecore.SandboxProvider) sporecore.ToolOutput {
	var params FindFilesParams
	if e := parseParams(call, &params); e != nil {
		return e.ToToolOutput()
	}
	root, v := sandbox.ResolvePath(ctx, params.Path, sporecore.OperationRead)
	if v != nil {
		return SandboxViolationError(v).ToToolOutput()
	}
	pattern := filepath.Join(root, params.Glob)
	matches, err := filepath.Glob(pattern)
	if err != nil {
		return InvalidParameters(fmt.Sprintf("invalid glob: %s", err)).ToToolOutput()
	}
	sort.Strings(matches)
	body := ""
	for i, m := range matches {
		if i > 0 {
			body += "\n"
		}
		body += m
	}
	return finishWithPossibleTruncation(ctx, body, call.ID, sandbox)
}

var (
	_ sporecore.Tool = (*GrepFilesTool)(nil)
	_ sporecore.Tool = (*FindFilesTool)(nil)
)
