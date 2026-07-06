//! EditFile tool (#81, net-new Tier-1 sandbox tool).
//!
//! `edit_file` replaces the FIRST and ONLY occurrence of `old_string` with
//! `new_string` in the file at `path`. The match must be UNIQUE:
//! - `old_string` not found      → recoverable [`ToolOutput::Error`].
//! - `old_string` found >1 time  → recoverable [`ToolOutput::Error`].
//!
//! Matching is exact-first with a normalization ladder for model sloppiness
//! (small local models routinely lose trailing whitespace, smart punctuation,
//! and CRLF when echoing file content back):
//! 1. BOM stripped and CRLF canonicalized to LF on BOTH sides before the
//!    exact-substring pass; original line endings and BOM are restored on
//!    write, so a CRLF file stays CRLF.
//! 2. On an exact miss, a whitespace/punctuation-tolerant fallback normalizes
//!    both sides (per-line trailing whitespace stripped; smart quotes,
//!    Unicode dashes, and exotic spaces mapped to ASCII) and, if that finds a
//!    UNIQUE match, maps it back to the original byte span and replaces it.
//! Uniqueness is still required on whichever pass matches — the fallback
//! loosens the comparison, never the contract.
//!
//! This is a net-new tool that does NOT replace `write_file` (issue #81, Q5).
//! Annotated `destructive` (it mutates a file in place).

use serde_json::json;

use crate::harness::{BoxFut, Operation, SandboxProvider, ToolOutput};
use crate::model::ToolCall;
use crate::tools::tool_try;
use crate::tool_registry::{Tool, ToolAnnotations, ToolSchema};
use crate::tools::error::ToolExecutionError;
use crate::tools::params::{parse_params, EditFileParams};

pub struct EditFileTool;

impl EditFileTool {
    pub const NAME: &'static str = "edit_file";
    pub fn new() -> Self {
        Self
    }
    pub fn schema() -> ToolSchema {
        ToolSchema {
            name: Self::NAME.into(),
            description: "Replace the unique occurrence of old_string with new_string in a file"
                .into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old_string": {"type": "string"},
                    "new_string": {"type": "string"},
                },
                "required": ["path", "old_string", "new_string"],
            }),
            annotations: ToolAnnotations {
                destructive: true,
                ..Default::default()
            },
        }
    }
}

impl Default for EditFileTool {
    fn default() -> Self {
        Self::new()
    }
}

impl Tool for EditFileTool {
    fn name(&self) -> &str {
        Self::NAME
    }
    fn execute<'a>(
        &'a self,
        call: &'a ToolCall,
        sandbox: &'a (dyn SandboxProvider + 'a),
        _ctx: &'a crate::tool_registry::ToolContext,
    ) -> BoxFut<'a, ToolOutput> {
        Box::pin(async move {
            let params: EditFileParams = tool_try!(parse_params(call));
            if params.old_string.is_empty() {
                return ToolOutput::Error {
                    message: "old_string must not be empty. To create a new file or fully \
                              overwrite one, use write_file instead."
                        .into(),
                    recoverable: true,
                };
            }
            if params.old_string == params.new_string {
                return ToolOutput::Error {
                    message: "old_string and new_string are identical; nothing would change. \
                              Provide a new_string that differs from old_string."
                        .into(),
                    recoverable: true,
                };
            }
            let resolved = tool_try!(sandbox.resolve_path(&params.path, Operation::Write).await.map_err(ToolExecutionError::SandboxViolation));
            let raw = match tokio::fs::read_to_string(&resolved).await {
                Ok(c) => c,
                Err(e) => {
                    return ToolOutput::Error {
                        message: format!("read failed: {e}"),
                        recoverable: true,
                    };
                }
            };
            // Canonicalize: strip BOM, CRLF → LF on both sides. Restored on write.
            let had_bom = raw.starts_with('\u{FEFF}');
            let stripped = raw.strip_prefix('\u{FEFF}').unwrap_or(&raw);
            let had_crlf = stripped.contains("\r\n");
            let content = stripped.replace("\r\n", "\n");
            let old = params.old_string.replace("\r\n", "\n");
            let new = params.new_string.replace("\r\n", "\n");

            let count = content.matches(&old).count();
            let (mut updated, note) = if count == 1 {
                (content.replacen(&old, &new, 1), "")
            } else if count > 1 {
                return ToolOutput::Error {
                    message: format!(
                        "old_string is not unique in {} ({count} occurrences); include more \
                         surrounding lines so it matches exactly one place",
                        params.path
                    ),
                    recoverable: true,
                };
            } else {
                // Exact miss → whitespace/punctuation-tolerant fallback.
                match fuzzy_find(&content, &old) {
                    FuzzyOutcome::Unique { start, end } => {
                        let mut u = String::with_capacity(content.len() + new.len());
                        u.push_str(&content[..start]);
                        u.push_str(&new);
                        u.push_str(&content[end..]);
                        (u, " (old_string matched via whitespace-tolerant fallback)")
                    }
                    FuzzyOutcome::NotFound => {
                        return ToolOutput::Error {
                            message: format!(
                                "old_string not found in {} (even with whitespace-tolerant \
                                 matching). It must match the file content exactly, including \
                                 indentation and line breaks. Re-read the file and copy the \
                                 text verbatim.",
                                params.path
                            ),
                            recoverable: true,
                        };
                    }
                    FuzzyOutcome::Ambiguous { count } => {
                        return ToolOutput::Error {
                            message: format!(
                                "old_string is not unique in {} ({count} occurrences after \
                                 whitespace-tolerant matching); include more surrounding lines \
                                 so it matches exactly one place",
                                params.path
                            ),
                            recoverable: true,
                        };
                    }
                }
            };
            if had_crlf {
                updated = updated.replace('\n', "\r\n");
            }
            if had_bom {
                updated.insert(0, '\u{FEFF}');
            }
            match tokio::fs::write(&resolved, updated.as_bytes()).await {
                Ok(_) => ToolOutput::Success {
                    content: format!("edited {}{note}", params.path),
                    truncated: false,
                },
                Err(e) => ToolOutput::Error {
                    message: format!("write failed: {e}"),
                    recoverable: true,
                },
            }
        })
    }
}

// ============================================================================
// Whitespace/punctuation-tolerant matching
// ============================================================================

enum FuzzyOutcome {
    Unique { start: usize, end: usize },
    NotFound,
    Ambiguous { count: usize },
}

/// Find `needle` in `content` after normalizing BOTH sides (see
/// [`normalize_for_fuzzy`]). A unique normalized match is mapped back to the
/// byte span of the ORIGINAL content so the replacement never rewrites text
/// the model didn't target.
fn fuzzy_find(content: &str, needle: &str) -> FuzzyOutcome {
    let (norm_content, map) = normalize_for_fuzzy(content);
    let (norm_needle, _) = normalize_for_fuzzy(needle);
    if norm_needle.is_empty() {
        return FuzzyOutcome::NotFound;
    }
    let count = norm_content.matches(&norm_needle).count();
    match count {
        0 => FuzzyOutcome::NotFound,
        1 => {
            let j = norm_content
                .find(&norm_needle)
                .expect("count==1 guarantees a match");
            FuzzyOutcome::Unique {
                start: map[j],
                end: map[j + norm_needle.len()],
            }
        }
        n => FuzzyOutcome::Ambiguous { count: n },
    }
}

/// Normalize text for tolerant matching, keeping a byte map back to the
/// source: `map[i]` is the source byte offset of the character that produced
/// normalized byte `i`, with one sentinel entry holding `source.len()`.
///
/// Normalizations (the ways models most often corrupt copied text):
/// - per-line trailing whitespace dropped (also at EOF)
/// - smart quotes → `'` / `"`
/// - Unicode dashes (en/em/figure/horizontal-bar/minus) → `-`
/// - exotic spaces (NBSP, thin space, ideographic space, …) → ` `
fn normalize_for_fuzzy(source: &str) -> (String, Vec<usize>) {
    fn push_mapped(norm: &mut String, map: &mut Vec<usize>, ch: char, off: usize) {
        norm.push(ch);
        for _ in 0..ch.len_utf8() {
            map.push(off);
        }
    }
    let mut norm = String::with_capacity(source.len());
    let mut map: Vec<usize> = Vec::with_capacity(source.len() + 1);
    // Run of spaces/tabs not yet known to be line-trailing.
    let mut pending_ws: Vec<(char, usize)> = Vec::new();
    for (off, ch) in source.char_indices() {
        let mapped = match ch {
            '\u{2018}' | '\u{2019}' | '\u{201A}' | '\u{201B}' => '\'',
            '\u{201C}' | '\u{201D}' | '\u{201E}' | '\u{201F}' => '"',
            '\u{2012}' | '\u{2013}' | '\u{2014}' | '\u{2015}' | '\u{2212}' => '-',
            '\u{00A0}' | '\u{1680}' | '\u{2000}'..='\u{200A}' | '\u{202F}' | '\u{205F}'
            | '\u{3000}' => ' ',
            other => other,
        };
        if mapped == ' ' || mapped == '\t' {
            pending_ws.push((mapped, off));
        } else if mapped == '\n' {
            pending_ws.clear(); // whitespace ran to end-of-line: trailing, drop it
            push_mapped(&mut norm, &mut map, '\n', off);
        } else {
            for (wch, woff) in pending_ws.drain(..) {
                push_mapped(&mut norm, &mut map, wch, woff);
            }
            push_mapped(&mut norm, &mut map, mapped, off);
        }
    }
    // Whitespace at EOF is trailing too — dropped with pending_ws.
    map.push(source.len());
    (norm, map)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tool_registry::mock::{test_ctx, AllowAllSandbox};
    use serde_json::json;
    use tempfile::TempDir;

    fn call(input: serde_json::Value) -> ToolCall {
        ToolCall {
            id: "c1".into(),
            name: EditFileTool::NAME.into(),
            input,
        }
    }

    #[tokio::test]
    async fn edit_replaces_unique_occurrence() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("a.txt");
        tokio::fs::write(&path, "hello world\n").await.unwrap();
        let sb = AllowAllSandbox;
        let r = EditFileTool::new()
            .execute(
                &call(json!({
                    "path": path.to_str().unwrap(),
                    "old_string": "world",
                    "new_string": "there",
                })),
                &sb,
                &test_ctx(),
            )
            .await;
        assert!(matches!(r, ToolOutput::Success { .. }));
        assert_eq!(
            tokio::fs::read_to_string(&path).await.unwrap(),
            "hello there\n"
        );
    }

    #[tokio::test]
    async fn edit_not_found_is_recoverable_error() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("a.txt");
        tokio::fs::write(&path, "hello\n").await.unwrap();
        let sb = AllowAllSandbox;
        let r = EditFileTool::new()
            .execute(
                &call(json!({
                    "path": path.to_str().unwrap(),
                    "old_string": "absent",
                    "new_string": "x",
                })),
                &sb,
                &test_ctx(),
            )
            .await;
        match r {
            ToolOutput::Error {
                recoverable,
                message,
            } => {
                assert!(recoverable);
                assert!(message.contains("not found"));
            }
            other => panic!("{other:?}"),
        }
    }

    #[tokio::test]
    async fn edit_non_unique_is_recoverable_error() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("a.txt");
        tokio::fs::write(&path, "x x x\n").await.unwrap();
        let sb = AllowAllSandbox;
        let r = EditFileTool::new()
            .execute(
                &call(json!({
                    "path": path.to_str().unwrap(),
                    "old_string": "x",
                    "new_string": "y",
                })),
                &sb,
                &test_ctx(),
            )
            .await;
        match r {
            ToolOutput::Error {
                recoverable,
                message,
            } => {
                assert!(recoverable);
                assert!(message.contains("not unique"));
            }
            other => panic!("{other:?}"),
        }
    }

    #[tokio::test]
    async fn edit_missing_file_is_recoverable_error() {
        let sb = AllowAllSandbox;
        let r = EditFileTool::new()
            .execute(
                &call(json!({
                    "path": "/no/such/file",
                    "old_string": "a",
                    "new_string": "b",
                })),
                &sb,
                &test_ctx(),
            )
            .await;
        match r {
            ToolOutput::Error { recoverable, .. } => assert!(recoverable),
            other => panic!("{other:?}"),
        }
    }

    #[tokio::test]
    async fn edit_bad_params_is_recoverable_error() {
        let sb = AllowAllSandbox;
        let r = EditFileTool::new()
            .execute(&call(json!({"path": "/x"})), &sb, &test_ctx())
            .await;
        match r {
            ToolOutput::Error { recoverable, .. } => assert!(recoverable),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn schema_is_destructive() {
        let s = EditFileTool::schema();
        assert!(s.annotations.destructive);
        assert!(!s.annotations.read_only);
    }

    // ---------------- normalization ladder ----------------

    async fn run_edit(path: &std::path::Path, old: &str, new: &str) -> ToolOutput {
        EditFileTool::new()
            .execute(
                &call(json!({
                    "path": path.to_str().unwrap(),
                    "old_string": old,
                    "new_string": new,
                })),
                &AllowAllSandbox,
                &test_ctx(),
            )
            .await
    }

    /// A CRLF file must accept an LF old_string (models rarely echo \r\n) and
    /// stay CRLF after the edit.
    #[tokio::test]
    async fn edit_crlf_file_with_lf_old_string_preserves_crlf() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("a.txt");
        tokio::fs::write(&path, "one\r\ntwo\r\nthree\r\n").await.unwrap();
        let r = run_edit(&path, "one\ntwo\n", "uno\ndos\n").await;
        assert!(matches!(r, ToolOutput::Success { .. }), "got {r:?}");
        assert_eq!(
            tokio::fs::read_to_string(&path).await.unwrap(),
            "uno\r\ndos\r\nthree\r\n"
        );
    }

    /// Trailing whitespace in the FILE that the model didn't echo must not
    /// break the match; the fallback replaces the original (trailing-ws) span.
    #[tokio::test]
    async fn edit_tolerates_trailing_whitespace_in_file() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("a.txt");
        tokio::fs::write(&path, "fn main() {   \n    body();\t\n}\n")
            .await
            .unwrap();
        let r = run_edit(&path, "fn main() {\n    body();\n}", "fn main() {\n    improved();\n}")
            .await;
        match &r {
            ToolOutput::Success { content, .. } => {
                assert!(content.contains("whitespace-tolerant"), "got {content:?}")
            }
            other => panic!("{other:?}"),
        }
        assert_eq!(
            tokio::fs::read_to_string(&path).await.unwrap(),
            "fn main() {\n    improved();\n}\n"
        );
    }

    /// Smart quotes / unicode dashes in the file, ASCII in old_string.
    #[tokio::test]
    async fn edit_tolerates_smart_punctuation() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("a.md");
        tokio::fs::write(&path, "It\u{2019}s a \u{201C}test\u{201D} \u{2014} really\n")
            .await
            .unwrap();
        let r = run_edit(&path, "It's a \"test\" - really", "plain now").await;
        assert!(matches!(r, ToolOutput::Success { .. }), "got {r:?}");
        assert_eq!(
            tokio::fs::read_to_string(&path).await.unwrap(),
            "plain now\n"
        );
    }

    /// The fallback loosens comparison, not the uniqueness contract.
    #[tokio::test]
    async fn edit_fuzzy_ambiguous_is_recoverable_error() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("a.txt");
        // "line()\n" is in the file only with trailing whitespace before the
        // newline, so the exact pass misses and the fallback sees TWO matches.
        tokio::fs::write(&path, "line()  \nline()\t\n").await.unwrap();
        let r = run_edit(&path, "line()\n", "other()\n").await;
        match r {
            ToolOutput::Error { message, recoverable } => {
                assert!(recoverable);
                assert!(message.contains("not unique"), "got {message:?}");
                assert!(message.contains("whitespace-tolerant"), "got {message:?}");
            }
            other => panic!("{other:?}"),
        }
    }

    #[tokio::test]
    async fn edit_preserves_bom() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("a.txt");
        tokio::fs::write(&path, "\u{FEFF}hello world\n").await.unwrap();
        let r = run_edit(&path, "world", "there").await;
        assert!(matches!(r, ToolOutput::Success { .. }), "got {r:?}");
        assert_eq!(
            tokio::fs::read_to_string(&path).await.unwrap(),
            "\u{FEFF}hello there\n"
        );
    }

    #[tokio::test]
    async fn edit_empty_old_string_is_corrective_error() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("a.txt");
        tokio::fs::write(&path, "hello\n").await.unwrap();
        let r = run_edit(&path, "", "x").await;
        match r {
            ToolOutput::Error { message, recoverable } => {
                assert!(recoverable);
                assert!(message.contains("write_file"), "got {message:?}");
            }
            other => panic!("{other:?}"),
        }
    }

    #[tokio::test]
    async fn edit_identical_strings_is_corrective_error() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("a.txt");
        tokio::fs::write(&path, "hello\n").await.unwrap();
        let r = run_edit(&path, "hello", "hello").await;
        match r {
            ToolOutput::Error { message, recoverable } => {
                assert!(recoverable);
                assert!(message.contains("identical"), "got {message:?}");
            }
            other => panic!("{other:?}"),
        }
    }

    /// Exact matches must NOT go through the fallback (no surprise span edits)
    /// and must not carry the fallback note.
    #[tokio::test]
    async fn edit_exact_match_has_no_fallback_note() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("a.txt");
        tokio::fs::write(&path, "alpha beta\n").await.unwrap();
        let r = run_edit(&path, "beta", "gamma").await;
        match r {
            ToolOutput::Success { content, .. } => {
                assert!(!content.contains("fallback"), "got {content:?}")
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn normalize_for_fuzzy_maps_spans_back_to_source() {
        let src = "foo  \nbar";
        let (norm, map) = normalize_for_fuzzy(src);
        assert_eq!(norm, "foo\nbar");
        // Normalized byte 4 ('b') came from source offset 6.
        assert_eq!(map[4], 6);
        assert_eq!(*map.last().unwrap(), src.len());
    }
}
