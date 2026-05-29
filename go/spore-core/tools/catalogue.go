// Standard Tool Catalogue (#81): the curated set of tools an architect drops
// into a harness, plus ready-made presets.
//
// Types
//   - StandardTool — a tool implementation bundled with its RegistryToolSchema
//     so the two can never be separated (issue #81, Q2). The HarnessBuilder's
//     Tool() destructures it into a registry Register() call.
//   - StandardTools — a namespace of one constructor per catalogue tool, each
//     returning a StandardTool, plus three presets: ReadonlySet, CodingSet,
//     and FullSet.
//
// Catalogue tools (constructor → registered name)
// Tier 1 (sandbox / stateless):
//   - ReadFile     → read_file     (EXISTING #5 tool)
//   - WriteFile    → write_file     (EXISTING)
//   - EditFile     → edit_file      (NEW)
//   - ListDir      → list_dir       (EXISTING)
//   - GrepFiles    → grep_files     (EXISTING)
//   - Grep         → grep           (NEW, output modes)
//   - FindFiles    → find_files     (EXISTING)
//   - BashCommand  → bash_command   (EXISTING)
//   - SendMessage  → send_message   (NEW)
//   - WebFetch     → web_fetch      (NEW)
//   - WebSearch    → web_search     (NEW)
//
// Tier 2 (storage via *ToolContext):
//   - TodoWrite    → todo_write     (NEW, RunStore key "todo")
//   - TaskList     → task_list      (EXISTING #71)
//
// Tier 3 (escalate / clarify):
//   - EnterPlanMode    → enter_plan_mode    (NEW)
//   - ExitPlanMode     → exit_plan_mode     (NEW)
//   - AskUserQuestion  → ask_user_question  (NEW)
//   - Abort            → abort              (NEW)
//
// Q5 — overlap with the EXISTING #5 catalogue (NO renames)
// The catalogue ships NET-NEW tools ALONGSIDE the existing #5 tools, never
// renaming them. Where a preset needs functionality an existing tool already
// provides, the preset REUSES the existing tool by its existing name:
//   - read_file, write_file, list_dir, find_files, grep_files, bash_command,
//     task_list are the EXISTING tools (their fixtures —
//     fixtures/tools/param_validation.json — stay byte-identical).
//   - edit_file and grep are NEW and live ALONGSIDE write_file / grep_files;
//     they do NOT replace them.
//
// Because StandardToolRegistry.Register is now a last-wins upsert (issue #81,
// Q1), registering a preset and then a custom tool of the same name lets the
// architect override a standard tool.
//
// Q3 — MemoryTool is DEFERRED. It depends on StorageScope (#79) and lands in a
// follow-on issue; it is intentionally NOT part of this catalogue.

package tools

import (
	sporecore "github.com/squirrelsoft-dev/spore-core/go/spore-core"
)

// StandardTool is the catalogue bundle (Tool implementation + RegistryToolSchema).
// It is defined in the root sporecore package (sporecore.StandardTool) so the
// HarnessBuilder can accept it without an import cycle; aliased here for ergonomic
// use from the tools package.
type StandardTool = sporecore.StandardTool

// schemaer is the optional Schema() method every catalogue tool implements.
type schemaer interface {
	Schema() sporecore.RegistryToolSchema
}

// newStandardTool bundles a tool that exposes a Schema() method.
func newStandardTool[T interface {
	sporecore.Tool
	schemaer
}](impl T) StandardTool {
	return StandardTool{Implementation: impl, Schema: impl.Schema()}
}

// StandardTools is the namespace of catalogue-tool constructors and presets.
// Unit type used as a namespace only — never instantiated.
type StandardTools struct{}

// ---- Tier 1 ---------------------------------------------------------------

// ReadFile — EXISTING #5 tool (Q5 overlap: reused, not renamed).
func (StandardTools) ReadFile() StandardTool { return newStandardTool(NewReadFileTool()) }

// WriteFile — EXISTING #5 tool (Q5 overlap: reused, not renamed).
func (StandardTools) WriteFile() StandardTool { return newStandardTool(NewWriteFileTool()) }

// EditFile — NEW unique-match in-place edit (alongside write_file).
func (StandardTools) EditFile() StandardTool { return newStandardTool(NewEditFileTool()) }

// ListDir — EXISTING #5 tool (Q5 overlap: reused, not renamed).
func (StandardTools) ListDir() StandardTool { return newStandardTool(NewListDirTool()) }

// GrepFiles — EXISTING #5 tool (Q5 overlap: reused, not renamed).
func (StandardTools) GrepFiles() StandardTool { return newStandardTool(NewGrepFilesTool()) }

// Grep — NEW regex search with output modes (alongside grep_files).
func (StandardTools) Grep() StandardTool { return newStandardTool(NewGrepTool()) }

// FindFiles — EXISTING #5 tool (Q5 overlap: reused, not renamed).
func (StandardTools) FindFiles() StandardTool { return newStandardTool(NewFindFilesTool()) }

// BashCommand — EXISTING #5 tool (Q5 overlap: reused, not renamed).
func (StandardTools) BashCommand() StandardTool { return newStandardTool(NewBashCommandTool()) }

// SendMessage — NEW; surfaces a UserMessage stream event via the harness loop.
func (StandardTools) SendMessage() StandardTool { return newStandardTool(NewSendMessageTool()) }

// WebFetch — NEW; GET a URL.
func (StandardTools) WebFetch() StandardTool { return newStandardTool(NewWebFetchTool()) }

// WebSearch — NEW; structured search over a configurable HTTP backend. The
// default has NO backend (calls error until one is configured); construct a
// StandardTool over NewWebSearchToolWithEndpoint to wire a real backend.
func (StandardTools) WebSearch() StandardTool { return newStandardTool(NewWebSearchTool()) }

// ---- Tier 2 ---------------------------------------------------------------

// TodoWrite — NEW; persists the todo list via RunStore key "todo".
func (StandardTools) TodoWrite() StandardTool { return newStandardTool(NewTodoWriteTool()) }

// TaskList — EXISTING #71 tool (Q5 overlap: reused, not renamed).
func (StandardTools) TaskList() StandardTool { return newStandardTool(NewTaskListTool()) }

// ---- Tier 3 ---------------------------------------------------------------

// EnterPlanMode — NEW; escalates HarnessSignal.EnterPlanMode.
func (StandardTools) EnterPlanMode() StandardTool { return newStandardTool(NewEnterPlanModeTool()) }

// ExitPlanMode — NEW; escalates HarnessSignal.ExitPlanMode{ Plan }.
func (StandardTools) ExitPlanMode() StandardTool { return newStandardTool(NewExitPlanModeTool()) }

// AskUserQuestion — NEW; returns ToolOutput.AwaitingClarification.
func (StandardTools) AskUserQuestion() StandardTool {
	return newStandardTool(NewAskUserQuestionTool())
}

// Abort — NEW; escalates HarnessSignal.Abort{ Reason }.
func (StandardTools) Abort() StandardTool { return newStandardTool(NewAbortTool()) }

// ---- Presets --------------------------------------------------------------

// ReadonlySet is the read-only investigation set: no mutating or escalating
// tools. Reuses the EXISTING read-only #5 tools by name (Q5 overlap) plus the
// NEW grep.
func (s StandardTools) ReadonlySet() []StandardTool {
	return []StandardTool{
		s.ReadFile(),
		s.ListDir(),
		s.GrepFiles(),
		s.Grep(),
		s.FindFiles(),
		s.WebFetch(),
		s.WebSearch(),
	}
}

// CodingSet is everything in ReadonlySet plus the mutating filesystem tools,
// shell, messaging, and the storage-backed todo/task tools. Reuses EXISTING
// tool names on overlap (Q5).
func (s StandardTools) CodingSet() []StandardTool {
	return []StandardTool{
		s.ReadFile(),
		s.WriteFile(),
		s.EditFile(),
		s.ListDir(),
		s.GrepFiles(),
		s.Grep(),
		s.FindFiles(),
		s.BashCommand(),
		s.SendMessage(),
		s.WebFetch(),
		s.WebSearch(),
		s.TodoWrite(),
		s.TaskList(),
	}
}

// FullSet is the CodingSet plus every Tier-3 control tool (plan / clarify /
// abort).
func (s StandardTools) FullSet() []StandardTool {
	set := s.CodingSet()
	set = append(set,
		s.EnterPlanMode(),
		s.ExitPlanMode(),
		s.AskUserQuestion(),
		s.Abort(),
	)
	return set
}
