//! Architect-side skill loading (zero core-harness change).
//!
//! ## Why this lives in the example, not the harness
//!
//! Issue #9 added `GuideType::Skill` to the `GuideRegistry`, and the rich
//! `StandardContextManager::assemble` knows how to inject skills as a Block-3
//! segment. But the **live** harness loop does not call that rich `assemble` —
//! it calls `StandardCompactionAdapter::assemble`, a pass-through of
//! `session.messages` (see issue #115 / Known Deviation #8). So today skills
//! reach the model only as tool-result text, never as structural injection.
//!
//! This module wires the chain end-to-end **architect-side**, exactly the
//! pattern issue #115 will absorb into the library:
//!
//! 1. A [`SkillCatalog`] scans `.spore/skills/{name}/SKILL.md` (project) then
//!    `~/.spore/skills/{name}/SKILL.md` (user), parses YAML frontmatter
//!    `{name, description}` + markdown body, and `register`s each as a
//!    `Guide::skill(name, body)` in a [`StandardGuideRegistry`]. It also keeps a
//!    manifest side-list of `(name, description)` because [`Guide`] has no
//!    `description` field — the example owns the manifest text.
//! 2. The `load_skill` tool (see `tools::load_skill`) appends a skill id to
//!    `run_store["active_skills"]`.
//! 3. [`SkillInjectingContextManager`] wraps the standard compaction adapter
//!    and, in `assemble`, prepends — **ephemerally**, never into
//!    `session.messages` — (a) the manifest of all skills, and (b) the full body
//!    of every active skill. Everything else delegates verbatim to the inner
//!    adapter.
//!
//! Net effect: the manifest is present every turn (progressive disclosure); a
//! loaded skill's body is re-injected every turn until the session is cleared.
//! Because the active set lives in `run_store` (not the message history), it is
//! compaction-proof.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use spore_core::storage::RunStore;
use spore_core::{
    AgentContext, Content, Guide, GuideRegistry, HarnessContextManager, HarnessToolResult, Message,
    Role, SessionId, SessionState, StandardGuideRegistry, Task,
};

use spore_core::harness::BoxFut;

/// The run-store key under which the `load_skill` tool and the context manager
/// rendezvous on the active-skill id set.
pub const ACTIVE_SKILLS_KEY: &str = "active_skills";

/// One parsed skill: its id (== frontmatter `name`), the one-line description
/// for the manifest, and the markdown body that is injected when active.
#[derive(Debug, Clone)]
pub struct SkillEntry {
    pub name: String,
    pub description: String,
    pub body: String,
}

/// The example's skill catalog: a `StandardGuideRegistry` (the real seam) plus
/// the manifest side-list the example owns (because `Guide` carries no
/// description). Bodies are resolved from the side-list, not re-queried from the
/// registry, so the manifest text and the injected body always agree.
pub struct SkillCatalog {
    registry: Arc<StandardGuideRegistry>,
    manifest: Vec<SkillEntry>,
}

impl SkillCatalog {
    /// Scan the project + user skill directories and register the bundled
    /// `audit` skill so the example is self-contained even with an empty
    /// `.spore/skills/`. Project entries win over user entries; the bundled
    /// `audit` body seeds `.spore/skills/audit/SKILL.md` on first run if absent
    /// (documented in the README) but is also registered directly here so the
    /// example never depends on that seed having been written.
    pub async fn bootstrap(project_root: &Path, bundled_audit: &str) -> Self {
        let registry = Arc::new(StandardGuideRegistry::new());
        let mut manifest: Vec<SkillEntry> = Vec::new();

        // 1. Bundled audit skill — always present, registered first so a
        //    project/user override of the same name supersedes it (last-wins in
        //    the manifest; the registry treats identical content as a no-op).
        if let Some(entry) = parse_skill_doc(bundled_audit) {
            upsert(&mut manifest, entry);
        }

        // 2. Project skills: `.spore/skills/{name}/SKILL.md` relative to cwd.
        for entry in scan_skill_dir(&project_root.join(".spore").join("skills")) {
            upsert(&mut manifest, entry);
        }

        // 3. User skills: `~/.spore/skills/{name}/SKILL.md`.
        if let Some(home) = home_dir() {
            for entry in scan_skill_dir(&home.join(".spore").join("skills")) {
                upsert(&mut manifest, entry);
            }
        }

        // Register every manifest entry as a Skill-type guide. Empty content is
        // rejected by the registry; parse_skill_doc already guarantees a body.
        for entry in &manifest {
            let _ = registry
                .register(Guide::skill(entry.name.clone(), entry.body.clone()))
                .await;
        }

        Self { registry, manifest }
    }

    /// The shared registry — handed to the `load_skill` tool and the context
    /// manager.
    pub fn registry(&self) -> Arc<StandardGuideRegistry> {
        self.registry.clone()
    }

    /// The manifest side-list — handed to the context manager so it can render
    /// `name: description` lines and resolve active bodies.
    pub fn manifest(&self) -> Vec<SkillEntry> {
        self.manifest.clone()
    }
}

/// Insert-or-replace by `name` so later sources override earlier ones.
fn upsert(manifest: &mut Vec<SkillEntry>, entry: SkillEntry) {
    if let Some(slot) = manifest.iter_mut().find(|e| e.name == entry.name) {
        *slot = entry;
    } else {
        manifest.push(entry);
    }
}

/// Scan one `skills/` directory: each `{name}/SKILL.md` is a candidate.
fn scan_skill_dir(dir: &Path) -> Vec<SkillEntry> {
    let mut out = Vec::new();
    let Ok(read) = std::fs::read_dir(dir) else {
        return out;
    };
    for child in read.flatten() {
        let skill_md = child.path().join("SKILL.md");
        if let Ok(content) = std::fs::read_to_string(&skill_md) {
            if let Some(entry) = parse_skill_doc(&content) {
                out.push(entry);
            }
        }
    }
    out
}

/// Parse a `SKILL.md`: a `---`-delimited YAML frontmatter block carrying
/// `name:` and `description:`, followed by the markdown body. Minimal,
/// dependency-free parsing — the example owns this until #115's
/// `FileSystemGuideRegistry` productionizes it. Returns `None` if there is no
/// usable name or the body is empty.
pub fn parse_skill_doc(content: &str) -> Option<SkillEntry> {
    let trimmed = content.trim_start();
    let (name, description, body) = if let Some(rest) = trimmed.strip_prefix("---") {
        // Split the frontmatter block off at the closing `---`.
        let mut parts = rest.splitn(2, "\n---");
        let front = parts.next().unwrap_or("");
        let body = parts.next().unwrap_or("").trim_start_matches('\n');
        let name = yaml_scalar(front, "name");
        let description = yaml_scalar(front, "description").unwrap_or_default();
        (name, description, body.to_string())
    } else {
        (None, String::new(), trimmed.to_string())
    };

    let name = name?;
    if name.trim().is_empty() || body.trim().is_empty() {
        return None;
    }
    Some(SkillEntry {
        name: name.trim().to_string(),
        description: description.trim().to_string(),
        body,
    })
}

/// Pull a single `key: value` scalar out of a YAML frontmatter block. Strips
/// surrounding quotes. Good enough for the `{name, description}` contract; not a
/// general YAML parser.
fn yaml_scalar(front: &str, key: &str) -> Option<String> {
    for line in front.lines() {
        let line = line.trim();
        if let Some(rest) = line.strip_prefix(key) {
            let rest = rest.trim_start();
            if let Some(value) = rest.strip_prefix(':') {
                let value = value.trim().trim_matches('"').trim_matches('\'');
                return Some(value.to_string());
            }
        }
    }
    None
}

/// Best-effort home directory without pulling in a `dirs` dependency.
fn home_dir() -> Option<PathBuf> {
    std::env::var_os("HOME").map(PathBuf::from)
}

/// A [`HarnessContextManager`] that wraps the standard compaction adapter and
/// injects the skill manifest + active skill bodies each turn. ALL non-assemble
/// trait methods delegate verbatim to the inner adapter — only `assemble` is
/// overridden, and even there the base context is produced by the inner adapter
/// first.
pub struct SkillInjectingContextManager {
    inner: Arc<dyn HarnessContextManager>,
    run_store: Arc<dyn RunStore>,
    manifest: Vec<SkillEntry>,
}

impl SkillInjectingContextManager {
    pub fn new(
        inner: Arc<dyn HarnessContextManager>,
        run_store: Arc<dyn RunStore>,
        manifest: Vec<SkillEntry>,
    ) -> Self {
        Self {
            inner,
            run_store,
            manifest,
        }
    }

    /// Read the active-skill id set from `run_store["active_skills"]`. Absent /
    /// malformed ⇒ empty (the manifest is still injected).
    async fn active_skills(&self, session_id: &SessionId) -> Vec<String> {
        match self.run_store.get(session_id, ACTIVE_SKILLS_KEY).await {
            Ok(Some(value)) => serde_json::from_value::<Vec<String>>(value).unwrap_or_default(),
            _ => Vec::new(),
        }
    }

    /// Render the leading injected messages: a manifest segment (always) plus
    /// one body segment per active skill (progressive disclosure). Returned as
    /// `User` messages so the loop still inserts the operating system prompt
    /// ahead of them at position 0.
    fn injected_messages(&self, active: &[String]) -> Vec<Message> {
        let mut out = Vec::new();

        // Manifest: every skill's name + one-line description.
        let mut manifest_text = String::from(
            "AVAILABLE SKILLS (call `load_skill` with a `skill_id` to activate one; its full \
             procedure then stays in context):\n",
        );
        for entry in &self.manifest {
            manifest_text.push_str(&format!("- {}: {}\n", entry.name, entry.description));
        }
        out.push(Message {
            role: Role::User,
            content: Content::Text {
                text: manifest_text,
            },
        });

        // Bodies of active skills, resolved from the manifest side-list.
        for id in active {
            if let Some(entry) = self.manifest.iter().find(|e| &e.name == id) {
                out.push(Message {
                    role: Role::User,
                    content: Content::Text {
                        text: format!("ACTIVE SKILL — {}:\n\n{}", entry.name, entry.body),
                    },
                });
            }
        }
        out
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
            let active = self.active_skills(&task.session_id).await;
            let mut injected = self.injected_messages(&active);
            injected.extend(context.messages);
            context.messages = injected;
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

    fn prepare_compaction_turn(
        &self,
        session: &SessionState,
    ) -> Option<spore_core::harness::CompactionTurn> {
        self.inner.prepare_compaction_turn(session)
    }

    fn inject_missing_items(&self, context: &mut AgentContext, missing: &[String]) {
        self.inner.inject_missing_items(context, missing);
    }

    fn apply_compaction(&self, session: &mut SessionState, summary: String) {
        self.inner.apply_compaction(session, summary);
    }

    fn token_budget_used(&self, session: &SessionState) -> Option<u32> {
        self.inner.token_budget_used(session)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use spore_core::storage::InMemoryStorageProvider;
    use spore_core::StorageProvider;

    /// A minimal pass-through inner CM (the harness `StandardCompactionAdapter`
    /// behaves the same way for `assemble`: a pass-through of session messages).
    struct PassThroughInner;
    impl HarnessContextManager for PassThroughInner {
        fn assemble<'a>(
            &'a self,
            session: &'a SessionState,
            _task: &'a Task,
        ) -> BoxFut<'a, AgentContext> {
            let messages = session.messages.clone();
            Box::pin(async move {
                AgentContext {
                    messages,
                    tools: Vec::new(),
                    params: Default::default(),
                }
            })
        }
        fn append_tool_result<'a>(
            &'a self,
            _session: &'a mut SessionState,
            _result: &'a HarnessToolResult,
        ) -> BoxFut<'a, ()> {
            Box::pin(async {})
        }
        fn append_user_message<'a>(
            &'a self,
            _session: &'a mut SessionState,
            _text: &'a str,
        ) -> BoxFut<'a, ()> {
            Box::pin(async {})
        }
    }

    fn manifest() -> Vec<SkillEntry> {
        vec![
            SkillEntry {
                name: "audit".into(),
                description: "Audit one Rust module for real, actionable defects.".into(),
                body: "GREP-FIRST PROCEDURE BODY".into(),
            },
            SkillEntry {
                name: "other".into(),
                description: "Some other skill.".into(),
                body: "OTHER BODY".into(),
            },
        ]
    }

    fn text_of(context: &AgentContext) -> String {
        context
            .messages
            .iter()
            .filter_map(|m| match &m.content {
                Content::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    #[tokio::test]
    async fn manifest_always_injected_bodies_only_when_active() {
        let storage = Arc::new(StorageProvider::single(Arc::new(
            InMemoryStorageProvider::new(),
        )));
        let cm = SkillInjectingContextManager::new(
            Arc::new(PassThroughInner),
            storage.run().clone(),
            manifest(),
        );

        let session = SessionState::default();
        let task = Task::new(
            "audit a module".to_string(),
            SessionId::new("sess-1"),
            spore_core::LoopStrategy::ReAct { max_iterations: 8 },
        );

        // No active skills yet: manifest present, NO body.
        let ctx = cm.assemble(&session, &task).await;
        let body = text_of(&ctx);
        assert!(
            body.contains("AVAILABLE SKILLS"),
            "manifest must be injected"
        );
        assert!(body.contains("audit: Audit one Rust module"));
        assert!(body.contains("other: Some other skill"));
        assert!(
            !body.contains("GREP-FIRST PROCEDURE BODY"),
            "inactive skill body must NOT be injected"
        );

        // Activate `audit` (as the load_skill tool does) → body appears next turn.
        storage
            .run()
            .put(
                &SessionId::new("sess-1"),
                ACTIVE_SKILLS_KEY,
                serde_json::json!(["audit"]),
            )
            .await
            .unwrap();

        let ctx = cm.assemble(&session, &task).await;
        let body = text_of(&ctx);
        assert!(body.contains("AVAILABLE SKILLS"), "manifest still present");
        assert!(
            body.contains("ACTIVE SKILL — audit"),
            "active skill body must be injected"
        );
        assert!(body.contains("GREP-FIRST PROCEDURE BODY"));
        assert!(
            !body.contains("OTHER BODY"),
            "only the active skill's body is injected"
        );
    }

    #[test]
    fn parses_frontmatter_name_and_description() {
        let doc = "---\nname: audit\ndescription: Audit one module.\n---\n\n# Body\nprocedure";
        let entry = parse_skill_doc(doc).expect("parses");
        assert_eq!(entry.name, "audit");
        assert_eq!(entry.description, "Audit one module.");
        assert!(entry.body.contains("procedure"));
    }

    #[test]
    fn rejects_missing_name_or_empty_body() {
        assert!(parse_skill_doc("---\ndescription: x\n---\nbody").is_none());
        assert!(parse_skill_doc("---\nname: audit\n---\n").is_none());
    }
}
