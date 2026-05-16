// Per-tool input parameter structs (issue #5).
//
// One struct per tool. Each maps 1:1 onto the Rust serde struct in
// rust/crates/spore-core/src/tools/params.rs. Wire-compatible JSON tags use
// snake_case.

package tools

import "encoding/json"

// ----- Filesystem -----

type ReadFileParams struct {
	Path string `json:"path"`
}

type WriteFileParams struct {
	Path    string `json:"path"`
	Content string `json:"content"`
	Append  bool   `json:"append,omitempty"`
}

type ListDirParams struct {
	Path      string `json:"path"`
	Recursive bool   `json:"recursive,omitempty"`
}

type DeleteFileParams struct {
	Path string `json:"path"`
}

type MoveFileParams struct {
	Src string `json:"src"`
	Dst string `json:"dst"`
}

// ----- Exec -----

type BashCommandParams struct {
	Command string   `json:"command"`
	Args    []string `json:"args,omitempty"`
	// Timeout in whole seconds. *uint64 so the absence of the field is
	// distinguishable from 0.
	Timeout *uint64 `json:"timeout,omitempty"`
}

type RunTestsParams struct {
	Command    string  `json:"command"`
	WorkingDir string  `json:"working_dir"`
	Timeout    *uint64 `json:"timeout,omitempty"`
}

// ----- Search -----

type GrepFilesParams struct {
	Pattern   string `json:"pattern"`
	Path      string `json:"path"`
	Recursive bool   `json:"recursive,omitempty"`
}

type FindFilesParams struct {
	Glob string `json:"glob"`
	Path string `json:"path"`
}

// ----- Git -----

type GitLogParams struct {
	N      uint32 `json:"n"`
	Format string `json:"format"`
}

// UnmarshalJSON applies the defaults n=20, format="oneline".
func (p *GitLogParams) UnmarshalJSON(data []byte) error {
	type alias GitLogParams
	a := alias{N: 20, Format: "oneline"}
	if err := json.Unmarshal(data, &a); err != nil {
		return err
	}
	if a.N == 0 {
		a.N = 20
	}
	if a.Format == "" {
		a.Format = "oneline"
	}
	*p = GitLogParams(a)
	return nil
}

type GitDiffParams struct {
	From *string `json:"from,omitempty"`
	To   *string `json:"to,omitempty"`
}

type GitCommitParams struct {
	Message string   `json:"message"`
	Files   []string `json:"files,omitempty"`
}

type GitStatusParams struct{}

// GitResetMode is one of "hard" | "soft" | "mixed".
type GitResetMode string

const (
	GitResetHard  GitResetMode = "hard"
	GitResetSoft  GitResetMode = "soft"
	GitResetMixed GitResetMode = "mixed"
)

type GitResetParams struct {
	Target string       `json:"target"`
	Mode   GitResetMode `json:"mode"`
}

// ----- HTTP -----

type HttpGetParams struct {
	URL     string            `json:"url"`
	Headers map[string]string `json:"headers,omitempty"`
}

type HttpPostParams struct {
	URL     string            `json:"url"`
	Body    json.RawMessage   `json:"body"`
	Headers map[string]string `json:"headers,omitempty"`
}

// ----- Subagent -----

type SubagentParams struct {
	Instruction string `json:"instruction"`
}
