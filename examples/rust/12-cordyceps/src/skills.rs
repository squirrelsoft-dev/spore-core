//! Architect-side **skills** for the cordyceps coding agent — the
//! [Agent Skills spec](https://agentskills.io/specification), wired example-side.
//!
//! A *skill* is a directory with a `SKILL.md`: YAML frontmatter (`name` +
//! `description`, plus optional `license` / `metadata` / …) followed by a
//! markdown procedure body. This module discovers them and exposes them to the
//! agent with **progressive disclosure**, exactly as the spec describes:
//!
//! 1. **Metadata (~tier 1).** The `name` + `description` of *every* discovered
//!    skill is injected every turn as a compact manifest — cheap, always present,
//!    so the agent knows what exists. (The spec's trigger keywords live inside
//!    `description` — "Use when …"; there is no separate triggers field.)
//! 2. **Instructions (tier 2).** A skill's full `SKILL.md` body is injected only
//!    once it is **active**, and then it stays injected every turn (sticky).
//!
//! A skill becomes active two ways:
//!   * the agent calls the [`load_skill`](load_skill_tool) tool when a request
//!     matches a skill's description (the trigger-word / judgement path); or
//!   * the user loads it from the REPL with `/<name>` (host-driven).
//!
//! Both write the same in-process [`ActiveSkills`] set, which the
//! [`SkillInjectingContextManager`] reads each turn. Because the whole REPL is
//! one process, the active set is a shared `Arc<Mutex<…>>` — no `RunStore`
//! round-trip (the TS/Python/Go #131 versions used run-store because their
//! composed strategy tree crossed per-node seams the active set had to outlive).
//!
//! ## Why this lives in the example, not the harness
//!
//! Issue #9 added the `skill` guide type and the rich context manager knows how
//! to inject skills structurally — but the **live** harness loop calls the
//! pass-through compaction adapter's `assemble`, not the rich one (Known
//! Deviation #8 / issue #115). So today a skill reaches the model only if the
//! example injects it. We do that by wrapping the adapter:
//! [`SkillInjectingContextManager`] delegates every seam method to the inner
//! adapter and only *prepends* the manifest + active bodies in `assemble`. Issue
//! #115 will fold discovery + `load_skill` + sticky injection into the harness
//! itself; until then this is the architect-side shim.

use std::collections::BTreeSet;
use std::fmt::Write as _;
use std::path::Path;
use std::sync::{Arc, Mutex};

use serde_json::json;

use spore_core::harness::{BoxFut, CompactionTurn, SandboxProvider};
use spore_core::{
    AgentContext, Content, HarnessContextManager, HarnessToolResult, Message, RegisteredToolSchema,
    Role, SessionState, StandardTool, Task, Tool, ToolAnnotations, ToolCall, ToolContext,
    ToolOutput,
};

// ============================================================================
// Skill model + discovery
// ============================================================================

/// One parsed skill. `name` is the identity (must match the skill's directory
/// name per the spec); `description` is the one-line manifest entry (where the
/// spec puts trigger keywords); `body` is the markdown procedure injected on load.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SkillEntry {
    pub name: String,
    pub description: String,
    pub body: String,
}

/// The discovered skill catalog — the example's in-process equivalent of the
/// spec's tier-1 metadata index. Entries are sorted by name for stable output.
pub struct SkillCatalog {
    entries: Vec<SkillEntry>,
}

impl SkillCatalog {
    /// Scan, in precedence order (last wins on a name clash):
    ///   1. **bundled** — `<example>/skills/<name>/SKILL.md` (found via
    ///      `CARGO_MANIFEST_DIR`, so it ships with the example regardless of where
    ///      the agent is launched);
    ///   2. **project** — `<workspace>/.spore/skills/<name>/SKILL.md`;
    ///   3. **user** — `~/.spore/skills/<name>/SKILL.md`.
    ///
    /// Discovery happens once, at startup (restart to pick up new files) — the
    /// same lifecycle real skill hosts use.
    pub fn bootstrap(workspace_root: &Path) -> Self {
        let mut entries: Vec<SkillEntry> = Vec::new();
        let mut dirs = vec![
            Path::new(env!("CARGO_MANIFEST_DIR")).join("skills"),
            workspace_root.join(".spore").join("skills"),
        ];
        if let Some(home) = std::env::var_os("HOME") {
            dirs.push(Path::new(&home).join(".spore").join("skills"));
        }
        for dir in &dirs {
            for entry in scan_skill_dir(dir) {
                upsert(&mut entries, entry);
            }
        }
        entries.sort_by(|a, b| a.name.cmp(&b.name));
        Self { entries }
    }

    /// `(name, description, body)` for every skill — handed to the context manager.
    pub fn manifest(&self) -> Vec<SkillEntry> {
        self.entries.clone()
    }

    /// Just the skill names — handed to the `load_skill` tool for validation and
    /// to the REPL for `/<name>` resolution.
    pub fn names(&self) -> Vec<String> {
        self.entries.iter().map(|e| e.name.clone()).collect()
    }

    /// The full entries, for listing (`/skills`).
    pub fn entries(&self) -> &[SkillEntry] {
        &self.entries
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

/// Insert-or-replace by `name`, so a later source overrides an earlier one.
fn upsert(entries: &mut Vec<SkillEntry>, entry: SkillEntry) {
    if let Some(slot) = entries.iter_mut().find(|e| e.name == entry.name) {
        *slot = entry;
    } else {
        entries.push(entry);
    }
}

/// Scan one `skills/` directory: each immediate `<child>/SKILL.md` is a
/// candidate. A missing/unreadable directory or file is skipped silently.
fn scan_skill_dir(dir: &Path) -> Vec<SkillEntry> {
    let mut out = Vec::new();
    let Ok(children) = std::fs::read_dir(dir) else {
        return out;
    };
    for child in children.flatten() {
        let skill_md = child.path().join("SKILL.md");
        if let Ok(text) = std::fs::read_to_string(&skill_md) {
            if let Some(entry) = parse_skill_doc(&text) {
                out.push(entry);
            }
        }
    }
    out
}

/// Parse a `SKILL.md`: optional `---`-delimited YAML frontmatter carrying at
/// least `name` and `description`, then the markdown body. Dependency-free and
/// minimal — enough for the spec's required fields. Optional fields (`license`,
/// `compatibility`, `metadata`, `allowed-tools`) are tolerated and ignored.
/// Returns `None` if there is no usable `name` or the body is empty.
pub fn parse_skill_doc(content: &str) -> Option<SkillEntry> {
    let trimmed = content.trim_start();
    let (front, body) = match trimmed.strip_prefix("---") {
        Some(rest) => match rest.find("\n---") {
            Some(idx) => (
                &rest[..idx],
                rest[idx + 4..].trim_start_matches(['\n', '\r']),
            ),
            None => (rest, ""),
        },
        None => ("", trimmed),
    };

    let name = yaml_scalar(front, "name")?;
    let name = name.trim();
    if name.is_empty() || body.trim().is_empty() {
        return None;
    }
    Some(SkillEntry {
        name: name.to_string(),
        description: yaml_scalar(front, "description")
            .unwrap_or_default()
            .trim()
            .to_string(),
        body: body.to_string(),
    })
}

/// Pull a single top-level `key: value` scalar from a YAML frontmatter block,
/// stripping surrounding quotes. Not a general YAML parser — nested maps (e.g.
/// `metadata:`) are skipped, since their indented children don't match a
/// top-level `key:` line.
fn yaml_scalar(front: &str, key: &str) -> Option<String> {
    for raw in front.lines() {
        let line = raw.trim();
        let Some(after) = line.strip_prefix(key) else {
            continue;
        };
        let Some(value) = after.trim_start().strip_prefix(':') else {
            continue;
        };
        return Some(strip_quotes(value.trim()).to_string());
    }
    None
}

fn strip_quotes(s: &str) -> &str {
    let b = s.as_bytes();
    if b.len() >= 2
        && ((b[0] == b'"' && b[b.len() - 1] == b'"') || (b[0] == b'\'' && b[b.len() - 1] == b'\''))
    {
        &s[1..s.len() - 1]
    } else {
        s
    }
}

// ============================================================================
// Active-skill set + context-manager injection
// ============================================================================

/// The set of skill names whose full bodies are injected every turn. Shared
/// in-process between the `load_skill` tool, the REPL `/<name>` command, and the
/// [`SkillInjectingContextManager`]. `BTreeSet` keeps the injection order stable.
pub type ActiveSkills = Arc<Mutex<BTreeSet<String>>>;

/// A fresh, empty active set.
pub fn new_active_set() -> ActiveSkills {
    Arc::new(Mutex::new(BTreeSet::new()))
}

/// Render the leading injected messages: the manifest (always, when non-empty)
/// plus the full body of each active skill (progressive disclosure). Returned as
/// `User` messages so the harness still inserts the operating system prompt ahead
/// of them at position 0.
fn render_injection(manifest: &[SkillEntry], active: &BTreeSet<String>) -> Vec<Message> {
    if manifest.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::new();

    let mut text = String::from(
        "AVAILABLE SKILLS — reusable procedures you can load on demand. When the \
         user's request matches one's description, call load_skill(name) BEFORE \
         acting, then follow the procedure it injects. Active skills' full bodies \
         are included below and stay in context every turn:\n",
    );
    for e in manifest {
        let _ = writeln!(text, "- {}: {}", e.name, e.description);
    }
    out.push(user_message(text));

    for name in active {
        if let Some(e) = manifest.iter().find(|e| &e.name == name) {
            out.push(user_message(format!(
                "ACTIVE SKILL — {}:\n\n{}",
                e.name, e.body
            )));
        }
    }
    out
}

fn user_message(text: String) -> Message {
    Message {
        role: Role::User,
        content: Content::Text { text },
    }
}

/// A harness-loop context manager that wraps the standard compaction adapter and
/// injects the skill manifest + active skill bodies each turn. EVERY non-`assemble`
/// method delegates verbatim to the inner adapter — forwarding `should_compact` /
/// `append_assistant_message` / the compaction seam is load-bearing: dropping them
/// would silently disable compaction and lose assistant turns.
pub struct SkillInjectingContextManager {
    inner: Arc<dyn HarnessContextManager>,
    active: ActiveSkills,
    manifest: Vec<SkillEntry>,
}

impl SkillInjectingContextManager {
    pub fn new(
        inner: Arc<dyn HarnessContextManager>,
        active: ActiveSkills,
        manifest: Vec<SkillEntry>,
    ) -> Self {
        Self {
            inner,
            active,
            manifest,
        }
    }

    fn injected_messages(&self) -> Vec<Message> {
        let active = self.active.lock().expect("active-skills mutex poisoned");
        render_injection(&self.manifest, &active)
    }
}

impl HarnessContextManager for SkillInjectingContextManager {
    fn assemble<'a>(
        &'a self,
        session: &'a SessionState,
        task: &'a Task,
    ) -> BoxFut<'a, AgentContext> {
        Box::pin(async move {
            let mut context = self.inner.assemble(session, task).await;
            // Build the injection AFTER the await, so the mutex is never held
            // across a suspension point.
            let mut messages = self.injected_messages();
            messages.append(&mut context.messages);
            context.messages = messages;
            context
        })
    }

    fn append_tool_result<'a>(
        &'a self,
        session: &'a mut SessionState,
        result: &'a HarnessToolResult,
    ) -> BoxFut<'a, ()> {
        self.inner.append_tool_result(session, result)
    }

    fn append_assistant_message<'a>(
        &'a self,
        session: &'a mut SessionState,
        message: &'a Message,
    ) -> BoxFut<'a, ()> {
        self.inner.append_assistant_message(session, message)
    }

    fn append_user_message<'a>(
        &'a self,
        session: &'a mut SessionState,
        text: &'a str,
    ) -> BoxFut<'a, ()> {
        self.inner.append_user_message(session, text)
    }

    fn should_compact(&self, session: &SessionState) -> bool {
        self.inner.should_compact(session)
    }

    fn prepare_compaction_turn(&self, session: &SessionState) -> Option<CompactionTurn> {
        self.inner.prepare_compaction_turn(session)
    }

    fn inject_missing_items(&self, context: &mut AgentContext, missing: &[String]) {
        self.inner.inject_missing_items(context, missing)
    }

    fn apply_compaction(&self, session: &mut SessionState, summary: String, dropped: &[Message]) {
        self.inner.apply_compaction(session, summary, dropped)
    }

    fn token_budget_used(&self, session: &SessionState) -> Option<u32> {
        self.inner.token_budget_used(session)
    }
}

// ============================================================================
// load_skill tool
// ============================================================================

/// The registered tool name.
pub const LOAD_SKILL: &str = "load_skill";

/// `load_skill(name)` — activate a skill so its full procedure stays in context
/// for the rest of the session. Holds the shared [`ActiveSkills`] set and the
/// known skill names (to reject unknown ids recoverably), so it is a hand-written
/// [`Tool`] rather than a macro tool (which cannot capture state).
struct LoadSkillTool {
    active: ActiveSkills,
    known: Vec<String>,
}

impl Tool for LoadSkillTool {
    fn name(&self) -> &str {
        LOAD_SKILL
    }

    fn execute<'a>(
        &'a self,
        call: &'a ToolCall,
        _sandbox: &'a (dyn SandboxProvider + 'a),
        _ctx: &'a ToolContext,
    ) -> BoxFut<'a, ToolOutput> {
        Box::pin(async move {
            let name = match call.input.get("name").and_then(|v| v.as_str()) {
                Some(s) if !s.trim().is_empty() => s.trim().to_string(),
                _ => return ToolOutput::error("invalid parameters: `name` (string) is required"),
            };
            if !self.known.iter().any(|n| n == &name) {
                return ToolOutput::error(format!(
                    "unknown skill '{name}'. Choose one listed in AVAILABLE SKILLS."
                ));
            }
            self.active
                .lock()
                .expect("active-skills mutex poisoned")
                .insert(name.clone());
            ToolOutput::success(format!(
                "Loaded skill '{name}' — its full procedure is now in your context. Follow it."
            ))
        })
    }
}

/// Build the `load_skill` [`StandardTool`], closing over the shared active set and
/// the known skill names. Add it to the harness with
/// [`HarnessBuilder::tool`](spore_core::HarnessBuilder::tool).
pub fn load_skill_tool(active: ActiveSkills, known: Vec<String>) -> StandardTool {
    StandardTool::new(
        Box::new(LoadSkillTool { active, known }),
        RegisteredToolSchema {
            name: LOAD_SKILL.into(),
            description: "Activate a skill by name so its full procedure stays in your context \
                          for the rest of the session. Call this BEFORE acting when the user's \
                          request matches a skill in the AVAILABLE SKILLS manifest."
                .into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "The skill's name, exactly as listed in AVAILABLE SKILLS."
                    }
                },
                "required": ["name"]
            }),
            annotations: ToolAnnotations::default(),
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn text_of(m: &Message) -> &str {
        match &m.content {
            Content::Text { text } => text,
            _ => panic!("expected a text message"),
        }
    }

    #[test]
    fn parses_frontmatter_name_description_body() {
        let doc = "---\nname: security-review\ndescription: Review code for security issues.\n---\n\n# Procedure\n\nDo the thing.\n";
        let entry = parse_skill_doc(doc).expect("should parse");
        assert_eq!(entry.name, "security-review");
        assert_eq!(entry.description, "Review code for security issues.");
        assert_eq!(entry.body, "# Procedure\n\nDo the thing.\n");
    }

    #[test]
    fn tolerates_optional_frontmatter_fields() {
        let doc = "---\nname: pdf\ndescription: Handle PDFs.\nlicense: Apache-2.0\nmetadata:\n  author: me\n  version: \"1.0\"\n---\nbody\n";
        let entry = parse_skill_doc(doc).expect("should parse");
        assert_eq!(entry.name, "pdf");
        assert_eq!(entry.description, "Handle PDFs.");
        assert_eq!(entry.body, "body\n");
    }

    #[test]
    fn strips_quotes_from_scalars() {
        let doc = "---\nname: \"q\"\ndescription: 'quoted desc'\n---\nx\n";
        let entry = parse_skill_doc(doc).unwrap();
        assert_eq!(entry.name, "q");
        assert_eq!(entry.description, "quoted desc");
    }

    #[test]
    fn rejects_missing_name_or_empty_body() {
        assert!(parse_skill_doc("---\ndescription: no name\n---\nbody\n").is_none());
        assert!(parse_skill_doc("---\nname: x\ndescription: d\n---\n   \n").is_none());
        assert!(parse_skill_doc("no frontmatter at all").is_none());
    }

    #[test]
    fn injection_is_empty_without_skills() {
        assert!(render_injection(&[], &BTreeSet::new()).is_empty());
    }

    #[test]
    fn injection_has_manifest_then_active_bodies() {
        let manifest = vec![
            SkillEntry {
                name: "audit".into(),
                description: "Audit a module.".into(),
                body: "AUDIT BODY".into(),
            },
            SkillEntry {
                name: "security-review".into(),
                description: "Review security.".into(),
                body: "SEC BODY".into(),
            },
        ];

        // No active skills → manifest only, both skills listed, no bodies.
        let none = render_injection(&manifest, &BTreeSet::new());
        assert_eq!(none.len(), 1);
        let manifest_text = text_of(&none[0]);
        assert!(manifest_text.contains("AVAILABLE SKILLS"));
        assert!(manifest_text.contains("audit: Audit a module."));
        assert!(manifest_text.contains("security-review: Review security."));
        assert!(!manifest_text.contains("SEC BODY"));

        // One active skill → manifest first, then only that skill's body.
        let mut active = BTreeSet::new();
        active.insert("security-review".to_string());
        let msgs = render_injection(&manifest, &active);
        assert_eq!(msgs.len(), 2);
        assert!(text_of(&msgs[0]).contains("AVAILABLE SKILLS"));
        let body = text_of(&msgs[1]);
        assert!(body.contains("ACTIVE SKILL — security-review"));
        assert!(body.contains("SEC BODY"));
        assert!(!body.contains("AUDIT BODY")); // audit not active
    }
}
