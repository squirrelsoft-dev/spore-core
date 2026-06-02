//! Standard Tool Catalogue (#81): the curated set of tools an architect drops
//! into a harness, plus ready-made presets.
//!
//! ## Types
//! - [`StandardTool`] — a tool implementation bundled with its
//!   [`ToolSchema`](crate::tool_registry::ToolSchema) so the two can never be
//!   separated (issue #81, Q2). `HarnessBuilder::tool()` destructures it.
//! - [`StandardTools`] — a namespace of one constructor per catalogue tool,
//!   each returning a [`StandardTool`], plus three presets:
//!   [`readonly_set`](StandardTools::readonly_set),
//!   [`coding_set`](StandardTools::coding_set), and
//!   [`full_set`](StandardTools::full_set).
//!
//! ## Catalogue tools (constructor → registered name)
//! Tier 1 (sandbox / stateless):
//!   - [`read_file`](StandardTools::read_file) → `read_file` (EXISTING #5 tool)
//!   - [`write_file`](StandardTools::write_file) → `write_file` (EXISTING)
//!   - [`edit_file`](StandardTools::edit_file) → `edit_file` (NEW)
//!   - [`list_dir`](StandardTools::list_dir) → `list_dir` (EXISTING)
//!   - [`grep_files`](StandardTools::grep_files) → `grep_files` (EXISTING)
//!   - [`grep`](StandardTools::grep) → `grep` (NEW, output modes)
//!   - [`find_files`](StandardTools::find_files) → `find_files` (EXISTING)
//!   - [`bash_command`](StandardTools::bash_command) → `bash_command` (EXISTING)
//!   - [`send_message`](StandardTools::send_message) → `send_message` (NEW)
//!   - [`web_fetch`](StandardTools::web_fetch) → `web_fetch` (NEW)
//!   - [`web_search`](StandardTools::web_search) → `web_search` (NEW)
//!
//! Tier 2 (storage via `ToolContext`):
//!   - [`todo_write`](StandardTools::todo_write) → `todo_write` (NEW, RunStore key `"todo"`)
//!   - [`task_list`](StandardTools::task_list) → `task_list` (EXISTING #71)
//!   - [`memory`](StandardTools::memory) → `memory` (NEW #82, scope-aware `MemoryStore`)
//!
//! Tier 3 (escalate / clarify):
//!   - [`enter_plan_mode`](StandardTools::enter_plan_mode) → `enter_plan_mode` (NEW)
//!   - [`exit_plan_mode`](StandardTools::exit_plan_mode) → `exit_plan_mode` (NEW)
//!   - [`ask_user_question`](StandardTools::ask_user_question) → `ask_user_question` (NEW)
//!   - [`abort`](StandardTools::abort) → `abort` (NEW)
//!
//! ## Q5 — overlap with the EXISTING #5 catalogue (NO renames)
//! The catalogue deliberately ships NET-NEW tools ALONGSIDE the existing #5
//! tools, never renaming them. Where a preset needs functionality that an
//! existing tool already provides, the preset REUSES the existing tool by its
//! existing name:
//!   - `read_file`, `write_file`, `list_dir`, `find_files`, `grep_files`,
//!     `bash_command` are the EXISTING tools (their fixtures —
//!     `fixtures/tools/param_validation.json` — stay byte-identical).
//!   - `edit_file` and `grep` are NEW and live ALONGSIDE `write_file` /
//!     `grep_files`; they do NOT replace them.
//!
//! Because [`StandardToolRegistry::register`](crate::tool_registry::StandardToolRegistry)
//! is now a last-wins upsert (issue #81, Q1), registering a preset and then a
//! custom tool of the same name lets the architect override a standard tool.
//!
//! ## Q3 — MemoryTool (shipped in #82)
//! `MemoryTool` (`memory`) was deferred from #81 and now ships in #82, on top of
//! the scoped `MemoryStore` seam from #78. It joins the Tier-2 storage tools in
//! `coding_set`/`full_set` alongside `task_list`/`todo_write`.
//!
//! Rules enforced: every constructor pairs the right impl with the right
//! schema; presets reuse existing names on overlap; no `// SPEC QUESTION:`
//! markers remain (all five forks are resolved upstream).

use crate::tool_registry::{Tool, ToolSchema};
use crate::tools::control::{AbortTool, AskUserQuestionTool, EnterPlanModeTool, ExitPlanModeTool};
use crate::tools::edit::EditFileTool;
use crate::tools::exec::BashCommandTool;
use crate::tools::fs::{ListDirTool, ReadFileTool, WriteFileTool};
use crate::tools::memory::MemoryTool;
use crate::tools::message::SendMessageTool;
use crate::tools::search::{FindFilesTool, GrepFilesTool, GrepTool};
use crate::tools::tasklist::TaskListTool;
use crate::tools::todo::TodoWriteTool;
use crate::tools::web::{WebFetchTool, WebSearchTool};

/// A catalogue tool: its [`Tool`] implementation bundled with its
/// [`ToolSchema`] so they can never drift apart (issue #81, Q2).
pub struct StandardTool {
    pub implementation: Box<dyn Tool>,
    pub schema: ToolSchema,
}

impl StandardTool {
    /// Bundle a tool implementation with its schema so it can be registered.
    ///
    /// This is the on-ramp for a **custom sandboxed tool**: implement
    /// [`Tool`](crate::tool_registry::Tool) — whose `execute` receives the
    /// [`SandboxProvider`](crate::harness::SandboxProvider) and
    /// [`ToolContext`](crate::tool_registry::ToolContext) seams — wrap it here,
    /// then add it to a harness with
    /// [`HarnessBuilder::tool`](crate::HarnessBuilder::tool). The builder folds
    /// it into the run loop with sandbox + storage wired in; you do not bridge
    /// the registries yourself.
    pub fn new(implementation: Box<dyn Tool>, schema: ToolSchema) -> Self {
        Self {
            implementation,
            schema,
        }
    }
}

/// Namespace of catalogue-tool constructors and presets. Unit struct used as a
/// namespace only — never instantiated.
pub struct StandardTools;

impl StandardTools {
    // ---- Tier 1 ---------------------------------------------------------

    /// `read_file` — EXISTING #5 tool (Q5 overlap: reused, not renamed).
    pub fn read_file() -> StandardTool {
        StandardTool::new(Box::new(ReadFileTool::new()), ReadFileTool::schema())
    }

    /// `write_file` — EXISTING #5 tool (Q5 overlap: reused, not renamed).
    pub fn write_file() -> StandardTool {
        StandardTool::new(Box::new(WriteFileTool::new()), WriteFileTool::schema())
    }

    /// `edit_file` — NEW unique-match in-place edit (alongside `write_file`).
    pub fn edit_file() -> StandardTool {
        StandardTool::new(Box::new(EditFileTool::new()), EditFileTool::schema())
    }

    /// `list_dir` — EXISTING #5 tool (Q5 overlap: reused, not renamed).
    pub fn list_dir() -> StandardTool {
        StandardTool::new(Box::new(ListDirTool::new()), ListDirTool::schema())
    }

    /// `grep_files` — EXISTING #5 tool (Q5 overlap: reused, not renamed).
    pub fn grep_files() -> StandardTool {
        StandardTool::new(Box::new(GrepFilesTool::new()), GrepFilesTool::schema())
    }

    /// `grep` — NEW regex search with output modes (alongside `grep_files`).
    pub fn grep() -> StandardTool {
        StandardTool::new(Box::new(GrepTool::new()), GrepTool::schema())
    }

    /// `find_files` — EXISTING #5 tool (Q5 overlap: reused, not renamed).
    pub fn find_files() -> StandardTool {
        StandardTool::new(Box::new(FindFilesTool::new()), FindFilesTool::schema())
    }

    /// `bash_command` — EXISTING #5 tool (Q5 overlap: reused, not renamed).
    pub fn bash_command() -> StandardTool {
        StandardTool::new(Box::new(BashCommandTool::new()), BashCommandTool::schema())
    }

    /// `send_message` — NEW; surfaces a `StreamEvent::UserMessage` via the loop.
    pub fn send_message() -> StandardTool {
        StandardTool::new(Box::new(SendMessageTool::new()), SendMessageTool::schema())
    }

    /// `web_fetch` — NEW; GET a URL (reqwest-direct, like `http.rs`).
    pub fn web_fetch() -> StandardTool {
        StandardTool::new(Box::new(WebFetchTool::new()), WebFetchTool::schema())
    }

    /// `web_search` — NEW; structured search over a configurable HTTP backend.
    /// The default has NO backend (calls error until one is configured); use a
    /// custom [`StandardTool`] over [`WebSearchTool::with_endpoint`] to wire a
    /// real backend.
    pub fn web_search() -> StandardTool {
        StandardTool::new(Box::new(WebSearchTool::new()), WebSearchTool::schema())
    }

    /// `web_search` wired to a concrete backend endpoint. The plain
    /// [`web_search`](Self::web_search) preset ships with no backend and errors
    /// on every call until one is configured; use this when you have a search
    /// endpoint (e.g. a Brave/Tavily-compatible URL) to POST the query to.
    pub fn web_search_with_endpoint(endpoint: impl Into<String>) -> StandardTool {
        StandardTool::new(
            Box::new(WebSearchTool::with_endpoint(endpoint)),
            WebSearchTool::schema(),
        )
    }

    // ---- Tier 2 ---------------------------------------------------------

    /// `todo_write` — NEW; persists the todo list via RunStore key `"todo"`.
    pub fn todo_write() -> StandardTool {
        StandardTool::new(Box::new(TodoWriteTool::new()), TodoWriteTool::schema())
    }

    /// `task_list` — EXISTING #71 tool (Q5 overlap: reused, not renamed).
    pub fn task_list() -> StandardTool {
        StandardTool::new(Box::new(TaskListTool::new()), TaskListTool::schema())
    }

    /// `memory` — NEW #82; scope-aware episodic memory read/write via the
    /// `MemoryStore` seam (#78). Registered alongside `task_list`/`todo_write`.
    pub fn memory() -> StandardTool {
        StandardTool::new(Box::new(MemoryTool::new()), MemoryTool::schema())
    }

    // ---- Tier 3 ---------------------------------------------------------

    /// `enter_plan_mode` — NEW; escalates `HarnessSignal::EnterPlanMode`.
    pub fn enter_plan_mode() -> StandardTool {
        StandardTool::new(
            Box::new(EnterPlanModeTool::new()),
            EnterPlanModeTool::schema(),
        )
    }

    /// `exit_plan_mode` — NEW; escalates `HarnessSignal::ExitPlanMode { plan }`.
    pub fn exit_plan_mode() -> StandardTool {
        StandardTool::new(
            Box::new(ExitPlanModeTool::new()),
            ExitPlanModeTool::schema(),
        )
    }

    /// `ask_user_question` — NEW; returns `ToolOutput::AwaitingClarification`.
    pub fn ask_user_question() -> StandardTool {
        StandardTool::new(
            Box::new(AskUserQuestionTool::new()),
            AskUserQuestionTool::schema(),
        )
    }

    /// `abort` — NEW; escalates `HarnessSignal::Abort { reason }`.
    pub fn abort() -> StandardTool {
        StandardTool::new(Box::new(AbortTool::new()), AbortTool::schema())
    }

    // ---- Presets --------------------------------------------------------

    /// Read-only investigation set: no mutating or escalating tools. Reuses the
    /// EXISTING read-only #5 tools by name (Q5 overlap) plus the NEW `grep`.
    pub fn readonly_set() -> Vec<StandardTool> {
        vec![
            Self::read_file(),
            Self::list_dir(),
            Self::grep_files(),
            Self::grep(),
            Self::find_files(),
            Self::web_fetch(),
            Self::web_search(),
        ]
    }

    /// Coding set: everything in [`readonly_set`](Self::readonly_set) plus the
    /// mutating filesystem tools, shell, messaging, and the storage-backed
    /// todo/task tools. Reuses EXISTING tool names on overlap (Q5).
    pub fn coding_set() -> Vec<StandardTool> {
        vec![
            Self::read_file(),
            Self::write_file(),
            Self::edit_file(),
            Self::list_dir(),
            Self::grep_files(),
            Self::grep(),
            Self::find_files(),
            Self::bash_command(),
            Self::send_message(),
            Self::web_fetch(),
            Self::web_search(),
            Self::todo_write(),
            Self::task_list(),
            Self::memory(),
        ]
    }

    /// Full set: the [`coding_set`](Self::coding_set) plus every Tier-3 control
    /// tool (plan / clarify / abort).
    pub fn full_set() -> Vec<StandardTool> {
        let mut set = Self::coding_set();
        set.push(Self::enter_plan_mode());
        set.push(Self::exit_plan_mode());
        set.push(Self::ask_user_question());
        set.push(Self::abort());
        set
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn names(set: &[StandardTool]) -> Vec<String> {
        set.iter().map(|t| t.schema.name.clone()).collect()
    }

    #[test]
    fn every_constructor_pairs_matching_impl_and_schema() {
        let all = StandardTools::full_set();
        for t in &all {
            assert_eq!(
                t.implementation.name(),
                t.schema.name,
                "impl/schema name mismatch for {}",
                t.schema.name
            );
        }
    }

    #[test]
    fn readonly_set_has_no_mutating_or_escalating_tools() {
        let n = names(&StandardTools::readonly_set());
        for forbidden in [
            "write_file",
            "edit_file",
            "bash_command",
            "todo_write",
            "enter_plan_mode",
            "exit_plan_mode",
            "ask_user_question",
            "abort",
        ] {
            assert!(
                !n.contains(&forbidden.to_string()),
                "readonly leaked {forbidden}"
            );
        }
        assert!(n.contains(&"read_file".to_string()));
        assert!(n.contains(&"grep".to_string()));
    }

    #[test]
    fn coding_set_reuses_existing_names_on_overlap() {
        let n = names(&StandardTools::coding_set());
        // Existing #5 names reused, not renamed (Q5).
        for existing in [
            "read_file",
            "write_file",
            "find_files",
            "grep_files",
            "bash_command",
        ] {
            assert!(
                n.contains(&existing.to_string()),
                "missing existing {existing}"
            );
        }
        // New tools alongside.
        assert!(n.contains(&"edit_file".to_string()));
        assert!(n.contains(&"grep".to_string()));
        // No Tier-3 control tools in coding_set.
        assert!(!n.contains(&"abort".to_string()));
    }

    #[test]
    fn full_set_adds_tier3() {
        let n = names(&StandardTools::full_set());
        for tier3 in [
            "enter_plan_mode",
            "exit_plan_mode",
            "ask_user_question",
            "abort",
        ] {
            assert!(n.contains(&tier3.to_string()), "missing tier3 {tier3}");
        }
    }

    #[test]
    fn standard_tool_bundles_impl_and_schema() {
        let t = StandardTools::edit_file();
        assert_eq!(t.implementation.name(), "edit_file");
        assert_eq!(t.schema.name, "edit_file");
        assert!(t.schema.annotations.destructive);
    }

    #[test]
    fn web_search_with_endpoint_is_named_web_search() {
        let t = StandardTools::web_search_with_endpoint("http://localhost:9/search");
        assert_eq!(t.implementation.name(), "web_search");
        assert_eq!(t.schema.name, "web_search");
    }
}
