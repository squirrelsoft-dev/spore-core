# Sandbox Violation Fixtures

Cross-language scenarios for `SandboxProvider` path resolution (issue #6).
Each implementation loads every JSON file here and asserts that the
canonical sandbox produces the same violation `kind` (or success).

## Schema

```jsonc
{
  "name": "escape_via_dotdot",
  // Each scenario is rooted at a freshly-created temp directory; the
  // string `workspace_root` is descriptive only and not used for path
  // construction. Implementations MUST create a TempDir, populate it
  // using `filesystem`, and then exercise `raw_path` against that root.
  "workspace_root": "/work",
  "raw_path": "../etc/passwd",
  "filesystem": {
    // Map of relative-to-temp-root path -> { "dir": true } or
    // { "file": "<content>" }. Created before resolve_path is called.
    "src": { "dir": true },
    "src/lib.rs": { "file": "// hello" }
  },
  "allowed_paths": [],          // relative-to-root strings
  "denied_paths": [],           // relative-to-root strings
  "denied_extensions": [],
  "read_only": false,
  "max_file_size": 0,
  "operation": "read",          // read | write | execute
  "expected": {
    "kind": "path_escape"       // matches SandboxViolation #[serde(tag="kind")]
    // For success cases: { "kind": "ok" }
  }
}
```

## Why the workspace_root field is descriptive only

Hard-coded absolute paths like `/work` are not creatable cross-platform in
CI. Each language test creates a `TempDir`, treats it as the workspace
root, materializes `filesystem` underneath, and resolves `raw_path`
against it. The fixture's literal `workspace_root` value documents intent
but is not consumed.

## Adding a scenario

1. Pick a single `SandboxViolation` variant (or `ok`).
2. Author the JSON file.
3. Run `cargo test -p spore-core sandbox_violation_fixtures` in Rust, plus
   the equivalent suite in TypeScript / Python / Go.
4. Every language must reach the same verdict.
