# Tool fixtures (issue #5)

Shared, language-agnostic scenarios for the standard tool catalogue.
All four language implementations replay these and must produce identical
outcome categories (success vs. recoverable parameter error vs. truncation
behaviour). They MUST NOT assert on language-specific error messages.

## Files

### `param_validation.json`
Array of `{ tool, input, expected }`.

- `tool`: one of `read_file`, `write_file`, `list_dir`, `delete_file`,
  `move_file`, `grep_files`, `find_files`, `git_status`, `git_log`.
- `input`: the JSON `ToolCall.input` to feed the tool.
- `expected`: `"ok"` if the call should reach execution (regardless of
  whether the underlying op then errors on a missing file), or
  `"invalid_parameters"` if parameter parsing should reject it as a
  recoverable error.

### `output_truncation.json`
Array of `{ content_length, head_tokens, tail_tokens, expects_truncated }`.

Probes `SandboxProvider::handle_large_output` against synthetic ASCII bodies.
`expects_truncated` is `true` iff the returned summary differs from the
input — implementations are free to choose their own truncation marker.

### `subagent_scenarios.json`
Array of `{ name, child_run_result, parent_call_id, expected }` describing
how `SubagentTool` should map scripted child `RunResult` shapes onto its
own `ToolOutput`. `child_run_result.kind` is `success` | `failure` |
`waiting_for_human`; `expected.kind` is `success` | `error` |
`waiting_for_human` with `parent_tool_call_id` echoed back in the latter.
