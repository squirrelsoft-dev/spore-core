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
| Tools    | `.tools(StandardTools.coding_set())`          | `web_search` (GET config) + `write_file` + `read_file`      |
| Side eff | writes `SUMMARY.md`                           | writes `answer.md`                                          |

The only substantive change is the tool registration. 04 used the local
filesystem `coding_set()`; 06 swaps in an **external** search tool and keeps two
file tools so the agent can persist its answer:

```python
# SearXNG's JSON API is GET /search?q=<query>&format=json. #108 added
# WebSearchConfig so we can drive the tool with GET + the `q` query param.
web_search_config = WebSearchConfig(
    endpoint=endpoint,                # e.g. http://localhost:8888/search?format=json
    method=SearchMethod.GET,
    query_param="q",
    auth_headers=[],
    body_auth_params=[],
)
web_search = StandardTool(
    WebSearchTool.with_config(web_search_config), WebSearchTool.schema()
)

harness = (
    HarnessBuilder.conversational(model)
    .sandbox(sandbox)                  # same as 04
    .tool(web_search)                  # ← external API (GET ?q=<query>)
    .tool(StandardTools.write_file())  # ← writes answer.md
    .tool(StandardTools.read_file())
    .system_prompt(SYSTEM_PROMPT)
    .build()
)
```

`.tool()` and `.tools()` push into the same tool registry (last-wins upsert by
name), so an external tool and the catalogue file tools compose in one builder
chain with no special handling.

### The agent writes the answer, not `main`

Just like 04, the file write happens **inside the ReAct loop** via the catalogue
`write_file` tool, backed by the `WorkspaceScopedSandbox`. `main.py` never writes
`answer.md` itself — the agent does, and the sandbox keeps it from escaping
`workspace/`. Different tools, same harness, same sandbox. (04 = `coding_set`;
06 = `web_search` + `write_file` / `read_file`.)

## Watching the loop

The search queries and result snippets show up in the stream because
`web_search` dispatches through the harness like any other catalogue tool:

```
think  · turn 1
    act    → web_search({"query": "recommended way to install Rust on macOS"})
    obs → {"results":[{"title":"Install Rust","url":"https://rustup.rs", …
    act    → write_file({"path": "answer.md", "content": "# Installing Rust on macOS …"})
think  · turn 2

answer (2 turn(s)): I searched for current install guidance and wrote answer.md …

answer.md now exists on disk: …/workspace/answer.md
```

## The search backend

There is **no live web-search backend in spore-core**. The endpoint is injected,
so you must supply one. The example reads it from `SPORE_WEB_SEARCH_ENDPOINT` and
exits if it is unset. `web_search` issues `GET <endpoint>?q=<query>` and hands the
response body back to the agent verbatim.

This example targets a self-hosted **SearXNG** instance, whose JSON API is
`GET /search?q=<query>&format=json`. You put the `?format=json` on the endpoint
URL itself; the GET path **preserves** that existing query string and appends
`q=<query>`. The configurable GET method + query param this relies on were added
in [#108](https://github.com/squirrelsoft-dev/spore-core/issues/108) (now
resolved) — the same `WebSearchConfig` also carries `auth_headers` /
`body_auth_params` for Brave (`X-Subscription-Token`) and Tavily (in-body
`api_key`), which this example leaves empty.

### SearXNG setup

1. Enable the JSON output format in your SearXNG `settings.yml`:

   ```yaml
   search:
     formats:
       - html
       - json
   ```

2. Restart SearXNG so the change takes effect.
3. Point the example at it (note the `?format=json`):

   ```sh
   export SPORE_WEB_SEARCH_ENDPOINT="http://localhost:8888/search?format=json"
   ```

## A note on the model

This example uses Ollama (`llama3.2`), the same local model as examples 01–05, so
it runs with no hosted-model account. Synthesis quality scales with the model: a
**larger hosted model will produce noticeably better, better-cited answers**. The
harness is model-agnostic — swap the model interface and change nothing else.

### Tool-calling mode: native by default, `--structured` to opt in

By default this example uses **native Ollama tool calling** — the real typed
tool schema — which works well for tool-capable / cloud models like
`gemma4:31b-cloud`. Pass `--structured` to opt into schema-constrained decoding
via `HarnessBuilder.model_params(ModelParams(structured_tool_calls=True))`, which
helps **small local models** (e.g. `llama3.2`) emit one clean tool call per turn
(no interleaved reasoning) instead of malformed JSON:

```sh
uv run main.py                # native tool calling (default)
uv run main.py --structured   # constrained decoding for small local models
```

One caveat for structured mode: it exposes an always-available `final` envelope
whose content is optional, so a **capable** model can emit `{"tool":"final"}`
prematurely and hand back an EMPTY answer without ever calling `write_file`. If
you run with `--structured` and see an empty answer and no `answer.md` on disk,
drop the flag and use the native default.

## Prerequisites

```sh
ollama serve &
ollama pull llama3.2
# A SearXNG JSON endpoint (see "SearXNG setup" above):
export SPORE_WEB_SEARCH_ENDPOINT="http://localhost:8888/search?format=json"
```

Plus [`uv`](https://docs.astral.sh/uv/). See `.env.example` for all the
variables.

## Run

```sh
cd examples/python/06-web-research
uv run main.py                # native tool calling (default)
uv run main.py --structured   # constrained decoding for small local models
uv run main.py --prompt "What are the current options for running WebAssembly outside the browser? Cite sources and write answer.md."
```

`SPORE_OLLAMA_MODEL` / `SPORE_OLLAMA_BASE_URL` override the model id and endpoint,
and `--model` overrides the id for a single run — same as 04 / 05.

`answer.md` is written into `workspace/` and is `.gitignore`d so re-runs stay
clean.
