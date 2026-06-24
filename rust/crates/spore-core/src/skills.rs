//! Issue #115 / SC-26 — **skills** baked into the harness.
//!
//! A *skill* is a directory with a `SKILL.md`: YAML frontmatter (`name` +
//! `description`, plus optional fields) followed by a markdown procedure body.
//! This module discovers them and exposes them to the agent with **progressive
//! disclosure** (the [Agent Skills spec](https://agentskills.io/specification)):
//!
//! 1. **Metadata (tier 1).** The `name` + `description` of every discovered skill
//!    is injected every turn as a compact manifest — cheap, always present, so the
//!    agent knows what exists.
//! 2. **Instructions (tier 2).** A skill's full body is injected only once it is
//!    **active**, and then it stays injected every turn (sticky).
//!
//! Unlike the pre-#115 architect-side shim (which injected skills as ad-hoc User
//! messages via a wrapping context manager), the catalog feeds the rich
//! [`ContextSources`](crate::context::ContextSources) seam: [`SkillCatalog::active_guides`]
//! returns the manifest + active bodies as [`Guide`]s, which the harness places in
//! the structural System block. A skill becomes active when the agent calls the
//! [`load_skill`](SkillCatalog::load_skill_tool) tool (or the host activates it).
//!
//! The active set is a shared `Arc<Mutex<…>>` held inside the [`SkillCatalog`], so
//! the `load_skill` tool and the per-turn `active_guides()` read the SAME set
//! within a harness's lifetime (sticky for the session — the catalog is held as an
//! `Arc<SkillCatalog>` by both the tool and `HarnessConfig`).

use std::collections::BTreeSet;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use serde_json::json;

use crate::guide_registry::Guide;
use crate::harness::{BoxFut, SandboxProvider, ToolOutput};
use crate::model::ToolCall;
use crate::tool_registry::{Tool, ToolAnnotations, ToolContext, ToolSchema as RegisteredToolSchema};
use crate::tools::StandardTool;

/// The registered name of the skill-activation tool.
pub const LOAD_SKILL: &str = "load_skill";

// ============================================================================
// Skill model + discovery
// ============================================================================

/// One parsed skill. `name` is the identity (matches the skill's directory name
/// per the spec); `description` is the one-line manifest entry (where the spec
/// puts trigger keywords); `body` is the markdown procedure injected on load.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SkillEntry {
    pub name: String,
    pub description: String,
    pub body: String,
}

/// Parse a `SKILL.md`: optional `---`-delimited YAML frontmatter carrying at least
/// `name` and `description`, then the markdown body. Dependency-free and minimal —
/// enough for the spec's required fields; optional fields are tolerated and
/// ignored. Returns `None` if there is no usable `name` or the body is empty.
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
/// stripping surrounding quotes. Not a general YAML parser — nested maps are
/// skipped, since their indented children don't match a top-level `key:` line.
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

/// Insert-or-replace by `name`, so a later source overrides an earlier one.
fn upsert(entries: &mut Vec<SkillEntry>, entry: SkillEntry) {
    if let Some(slot) = entries.iter_mut().find(|e| e.name == entry.name) {
        *slot = entry;
    } else {
        entries.push(entry);
    }
}

// ============================================================================
// Catalog
// ============================================================================

/// The discovered skill catalog plus the sticky active-skill set. Held as an
/// `Arc<SkillCatalog>` by both the `load_skill` tool (which activates skills) and
/// the harness (which reads [`active_guides`](Self::active_guides) each turn), so
/// both share the same active set for the catalog's lifetime.
pub struct SkillCatalog {
    entries: Vec<SkillEntry>,
    active: Mutex<BTreeSet<String>>,
}

impl SkillCatalog {
    /// Build a catalog from already-parsed entries (sorted by name for stable
    /// output; later duplicates by name win).
    pub fn from_entries(entries: impl IntoIterator<Item = SkillEntry>) -> Arc<Self> {
        let mut acc: Vec<SkillEntry> = Vec::new();
        for e in entries {
            upsert(&mut acc, e);
        }
        acc.sort_by(|a, b| a.name.cmp(&b.name));
        Arc::new(Self {
            entries: acc,
            active: Mutex::new(BTreeSet::new()),
        })
    }

    /// Discover skills from `extra_dirs` (e.g. a host's bundled `skills/`) plus the
    /// conventional `<workspace>/.spore/skills` and `~/.spore/skills`, in that
    /// precedence order (last wins on a name clash). Discovery happens once.
    pub fn discover(extra_dirs: &[PathBuf], workspace_root: &Path) -> Arc<Self> {
        let mut dirs: Vec<PathBuf> = extra_dirs.to_vec();
        dirs.push(workspace_root.join(".spore").join("skills"));
        if let Some(home) = std::env::var_os("HOME") {
            dirs.push(Path::new(&home).join(".spore").join("skills"));
        }
        let mut entries: Vec<SkillEntry> = Vec::new();
        for dir in &dirs {
            for entry in scan_skill_dir(dir) {
                upsert(&mut entries, entry);
            }
        }
        Self::from_entries(entries)
    }

    /// Skill names, for host-driven activation (`/<name>`) and listing.
    pub fn names(&self) -> Vec<String> {
        self.entries.iter().map(|e| e.name.clone()).collect()
    }

    /// The full entries, for listing.
    pub fn entries(&self) -> &[SkillEntry] {
        &self.entries
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Activate a skill by name so its full body is injected every turn. Returns
    /// `false` for an unknown name (no change).
    pub fn activate(&self, name: &str) -> bool {
        if !self.entries.iter().any(|e| e.name == name) {
            return false;
        }
        self.active
            .lock()
            .expect("active-skills mutex poisoned")
            .insert(name.to_string());
        true
    }

    /// The currently active skill names (sorted).
    pub fn active(&self) -> Vec<String> {
        self.active
            .lock()
            .expect("active-skills mutex poisoned")
            .iter()
            .cloned()
            .collect()
    }

    /// The guides injected this turn (issue #115): a single manifest guide listing
    /// every available skill (tier 1), followed by one guide per active skill
    /// carrying its full body (tier 2). Empty when the catalog has no skills. The
    /// harness appends these to [`ContextSources::guides`](crate::context::ContextSources)
    /// so they reach the model through the structural System block.
    pub fn active_guides(&self) -> Vec<Guide> {
        if self.entries.is_empty() {
            return Vec::new();
        }
        let mut out: Vec<Guide> = Vec::new();

        let mut manifest = String::from(
            "Reusable procedures you can load on demand. When the user's request \
             matches one's description, call load_skill(name) BEFORE acting, then \
             follow the procedure it injects. Active skills' full bodies are \
             included below and stay in context every turn:\n",
        );
        for e in &self.entries {
            let _ = writeln!(manifest, "- {}: {}", e.name, e.description);
        }
        out.push(Guide::skill("AVAILABLE SKILLS", manifest));

        for name in self.active() {
            if let Some(e) = self.entries.iter().find(|e| e.name == name) {
                out.push(Guide::skill(
                    format!("ACTIVE SKILL — {}", e.name),
                    e.body.clone(),
                ));
            }
        }
        out
    }

    /// Build the `load_skill` [`StandardTool`], sharing this catalog's active set.
    /// Add it to the harness with
    /// [`HarnessBuilder::tool`](crate::harness::HarnessBuilder::tool) — or use
    /// [`HarnessBuilder::skills`](crate::harness::HarnessBuilder::skills), which
    /// registers both the catalog and this tool.
    pub fn load_skill_tool(self: &Arc<Self>) -> StandardTool {
        StandardTool::new(
            Box::new(LoadSkillTool {
                catalog: Arc::clone(self),
            }),
            RegisteredToolSchema {
                name: LOAD_SKILL.into(),
                description: "Activate a skill by name so its full procedure stays in your \
                              context for the rest of the session. Call this BEFORE acting when \
                              the user's request matches a skill in the AVAILABLE SKILLS manifest."
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
}

// ============================================================================
// load_skill tool
// ============================================================================

/// `load_skill(name)` — activate a skill so its full procedure stays in context.
/// Holds the shared [`SkillCatalog`] (to mutate the active set and reject unknown
/// ids recoverably), so it is a hand-written [`Tool`] rather than a macro tool.
struct LoadSkillTool {
    catalog: Arc<SkillCatalog>,
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
            if !self.catalog.activate(&name) {
                return ToolOutput::error(format!(
                    "unknown skill '{name}'. Choose one listed in AVAILABLE SKILLS."
                ));
            }
            ToolOutput::success(format!(
                "Loaded skill '{name}' — its full procedure is now in your context. Follow it."
            ))
        })
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_frontmatter_name_description_body() {
        let doc = "---\nname: security-review\ndescription: Review code for security issues.\n---\n\n# Procedure\n\nDo the thing.\n";
        let entry = parse_skill_doc(doc).expect("should parse");
        assert_eq!(entry.name, "security-review");
        assert_eq!(entry.description, "Review code for security issues.");
        assert_eq!(entry.body, "# Procedure\n\nDo the thing.\n");
    }

    #[test]
    fn tolerates_optional_frontmatter_and_strips_quotes() {
        let doc = "---\nname: \"pdf\"\ndescription: 'Handle PDFs.'\nlicense: Apache-2.0\nmetadata:\n  author: me\n---\nbody\n";
        let entry = parse_skill_doc(doc).expect("should parse");
        assert_eq!(entry.name, "pdf");
        assert_eq!(entry.description, "Handle PDFs.");
        assert_eq!(entry.body, "body\n");
    }

    #[test]
    fn rejects_missing_name_or_empty_body() {
        assert!(parse_skill_doc("---\ndescription: no name\n---\nbody\n").is_none());
        assert!(parse_skill_doc("---\nname: x\ndescription: d\n---\n   \n").is_none());
        assert!(parse_skill_doc("no frontmatter at all").is_none());
    }

    #[test]
    fn empty_catalog_yields_no_guides() {
        let cat = SkillCatalog::from_entries([]);
        assert!(cat.active_guides().is_empty());
    }

    #[test]
    fn manifest_guide_then_active_body_guides() {
        let cat = SkillCatalog::from_entries([
            SkillEntry {
                name: "audit".into(),
                description: "Audit a module.".into(),
                body: "AUDIT BODY".into(),
            },
            SkillEntry {
                name: "style".into(),
                description: "Style guide.".into(),
                body: "STYLE BODY".into(),
            },
        ]);
        // Before activation: only the manifest guide.
        let guides = cat.active_guides();
        assert_eq!(guides.len(), 1);
        assert_eq!(guides[0].name, "AVAILABLE SKILLS");
        assert!(guides[0].content.contains("- audit: Audit a module."));
        assert!(guides[0].content.contains("- style: Style guide."));

        // Unknown activation is rejected; known activation is sticky.
        assert!(!cat.activate("nope"));
        assert!(cat.activate("audit"));
        let guides = cat.active_guides();
        assert_eq!(guides.len(), 2);
        assert_eq!(guides[1].name, "ACTIVE SKILL — audit");
        assert_eq!(guides[1].content, "AUDIT BODY");
        // Idempotent.
        assert!(cat.activate("audit"));
        assert_eq!(cat.active_guides().len(), 2);
    }
}
