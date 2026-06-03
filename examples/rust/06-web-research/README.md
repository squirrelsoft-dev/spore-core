# 06 — Web research (external API tools)

The first example whose tools reach **outside the process** to a third-party
HTTP service. The agent is asked a question that needs current information,
searches the web with `web_search`, synthesizes a **cited** answer, and writes it
to `answer.md`.

The thesis is the same one 04 and 05 keep proving, taken one step further: **the
harness doesn't change — only the tool set does.** An external API is just a
tool. It drops into the exact same `conversational(model)` builder, the same
`ReAct` loop, and the same `WorkspaceScopedSandbox` you saw in 04.

## The contrast with 04

|          | 04 — filesystem-agent                         | 06 — web-research                                            |
| -------- | --------------------------------------------- | ----------------------------------------------------------- |
| Builder  | `conversational(model)`                       | `conversational(model)` *(same)*                            |
| Loop     | `ReAct`                                       | `ReAct` *(same)*                                            |
| Sandbox  | `WorkspaceScopedSandbox` over `sample-files/` | `WorkspaceScopedSandbox` over `workspace/` *(same pattern)* |
| Output   | stream-printed `think` / `act` / `obs`        | stream-printed `think` / `act` / `obs` *(same)*             |
| Tools    | `.tools(StandardTools::coding_set())`         | SearXNG-configured `web_search` (GET) + `write_file` + `read_file` |
| Side eff | writes `SUMMARY.md`                            | writes `answer.md`                                          |

The only substantive change is the tool registration. 04 used the local
filesystem `coding_set()`; 06 swaps in an **external** search tool and keeps two
file tools so the agent can persist its answer:

```rust
// SearXNG GET config: `?q=<query>` appended; `format=json` rides on the endpoint.
let web_search_tool = WebSearchTool::with_config(WebSearchConfig {
    endpoint,
    method: SearchMethod::Get,
    query_param: "q".into(),
    auth_headers: Vec::new(),
    body_auth_params: Vec::new(),
})
.expect("SearXNG config is valid (no auth env vars)");
let web_search = StandardTool::new(Box::new(web_search_tool), WebSearchTool::schema());

let harness = HarnessBuilder::conversational(model)
    .sandbox(Arc::new(sandbox))                                   // same as 04
    .tool(web_search)                                             // ← external API (GET)
    .tool(StandardTools::write_file())                            // ← writes answer.md
    .tool(StandardTools::read_file())
    .system_prompt(SYSTEM_PROMPT)
    .build();
```

`.tool()` and `.tools()` push into the same tool registry (last-wins upsert by
name), so an external tool and the catalogue file tools compose in one builder
chain with no special handling.

### The agent writes the answer, not `main`

Just like 04, the file write happens **inside the ReAct loop** via the catalogue
`write_file` tool, backed by the `WorkspaceScopedSandbox`. `main.rs` never writes
`answer.md` itself — the agent does, and the sandbox keeps it from escaping
`workspace/`. Different tools, same harness, same sandbox.

## Watching the loop

The search queries and result snippets show up in the stream because
`web_search` dispatches through the harness like any other catalogue tool:

```
think  · turn 1
    act    → web_search({"query":"recommended way to install Rust on macOS"})
    obs → {"results":[{"title":"Install Rust","url":"https://rustup.rs", …
    act    → write_file({"path":"answer.md","content":"# Installing Rust on macOS …"})
think  · turn 2

answer (2 turn(s)): I searched for current install guidance and wrote answer.md …

answer.md now exists on disk: …/workspace/answer.md
```

## The search backend (SearXNG)

There is **no live web-search backend in spore-core**. The endpoint is injected,
so you must supply one. The example reads it from `SPORE_WEB_SEARCH_ENDPOINT` and
exits if it is unset. The tool issues `GET <endpoint>?q=<query>` and hands the
response body back to the agent verbatim — the `format=json` selector rides on
the endpoint URL (`.../search?format=json`) and the query is appended as
`&q=<query>`. reqwest's `.query()` **preserves** the existing query string, so
both `format=json` and `q=...` reach SearXNG.

### SearXNG setup

By default SearXNG only serves HTML. Enable the JSON format in `settings.yml`:

```yaml
search:
  formats:
    - html
    - json
```

Restart SearXNG, then point the example at it:

```sh
export SPORE_WEB_SEARCH_ENDPOINT="http://localhost:8888/search?format=json"
```

**Raw Brave / Tavily** would use a custom auth header (`X-Subscription-Token` /
`Authorization`). That **is** now supported — the `auth_headers` field on
`WebSearchConfig`, added in
[issue #108](https://github.com/squirrelsoft-dev/spore-core/issues/108), attaches
a secret resolved from an env var on every request. This example simply targets
keyless SearXNG and configures no auth.

## A note on the model

This example uses Ollama (`llama3.2`), the same local model as examples 01–05, so
it runs with no hosted-model account. Synthesis quality scales with the model: a
**larger hosted model will produce noticeably better, better-cited answers**. The
harness is model-agnostic — swap the model interface and change nothing else.

### Native vs. structured tool calls (`--structured`)

By default the example uses **native** Ollama tool calling: `write_file` gets a
real typed schema and the model is steered straight to search → write → answer.
This is what tool-capable models (including hosted `*-cloud` models such as
`gemma4:31b-cloud`) want.

Small local models (e.g. `llama3.2`) sometimes leak `<|python_tag|>` or emit
malformed JSON on the native channel. Pass `--structured` to switch on
schema-constrained decoding (`ModelParams { structured_tool_calls: true, .. }`),
which forces one clean JSON tool call per turn:

```sh
cargo run -- --structured                       # llama3.2 with constrained decoding
cargo run -- --model gemma4:31b-cloud           # capable model, native tool calls
```

> Note: structured mode exposes an always-available `final` envelope. Weaker
> instruction-followers can take that exit prematurely (emitting an empty answer
> without ever calling `write_file`). If you see an empty `answer (N turn(s)):`
> and no `answer.md`, drop `--structured` and use native tool calling.

## Prerequisites

```sh
ollama serve &
ollama pull llama3.2
# A SearXNG JSON endpoint (enable the `json` format first — see above):
export SPORE_WEB_SEARCH_ENDPOINT="http://localhost:8888/search?format=json"
```

See `.env.example` for all the variables.

## Run

```sh
cd examples/rust/06-web-research
cargo run
cargo run -- --prompt "What are the current options for running WebAssembly outside the browser? Cite sources and write answer.md."
```

`SPORE_OLLAMA_MODEL` / `SPORE_OLLAMA_BASE_URL` override the model id and endpoint,
and `--model` overrides the id for a single run — same as 04 / 05.

`answer.md` is written into `workspace/` and is `.gitignore`d so re-runs stay
clean.
