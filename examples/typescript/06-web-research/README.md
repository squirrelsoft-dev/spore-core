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

|          | 04 — filesystem-agent                         | 06 — web-research                                           |
| -------- | --------------------------------------------- | ----------------------------------------------------------- |
| Builder  | `conversational(model)`                       | `conversational(model)` _(same)_                            |
| Loop     | ReAct                                         | ReAct _(same)_                                              |
| Sandbox  | `WorkspaceScopedSandbox` over `sample-files/` | `WorkspaceScopedSandbox` over `workspace/` _(same pattern)_ |
| Output   | stream-printed `think` / `act` / `obs`        | stream-printed `think` / `act` / `obs` _(same)_             |
| Tools    | `.tools(StandardTools.codingSet())`           | `WebSearchTool.withConfig(..)` + `writeFile` + `readFile`   |
| Side eff | writes `SUMMARY.md`                           | writes `answer.md`                                          |

The only substantive change is the tool registration. 04 used the local
filesystem `codingSet()`; 06 swaps in an **external** search tool and keeps two
file tools so the agent can persist its answer:

```ts
const harness = HarnessBuilder.conversational(model)
  .sandbox(sandbox) // same as 04
  .tool({
    // GET <endpoint>?q=<query>; the endpoint's `?format=json` is preserved.
    implementation: WebSearchTool.withConfig({
      endpoint,
      method: "GET",
      queryParam: "q",
      authHeaders: [],
      bodyAuthParams: [],
    }),
    schema: WebSearchTool.schema(),
  }) // ← external API
  .tool(StandardTools.writeFile()) // ← writes answer.md
  .tool(StandardTools.readFile())
  .systemPrompt(SYSTEM_PROMPT)
  .build();
```

`.tool()` pushes into the same tool registry (last-wins upsert by name), so an
external tool and the catalogue file tools compose in one builder chain with no
special handling.

### The agent writes the answer, not `main`

Just like 04, the file write happens **inside the ReAct loop** via the catalogue
`write_file` tool, backed by the `WorkspaceScopedSandbox`. `main.ts` never writes
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

## The search backend (SearXNG, GET + JSON)

There is **no live web-search backend in spore-core**. The endpoint is injected,
so you must supply one. The example reads it from `SPORE_WEB_SEARCH_ENDPOINT` and
exits if it is unset. `web_search` issues `GET <endpoint>?q=<query>` and hands the
response body back to the agent verbatim.

This example targets a self-hosted **SearXNG** JSON API. The endpoint already
carries `format=json`; the GET path **preserves** that query string and appends
the `q` param, so the backend receives `GET /search?format=json&q=<query>`.

`web_search` is configured via `WebSearchTool.withConfig` (core
[issue #108](https://github.com/squirrelsoft-dev/spore-core/issues/108), now
implemented) with `method: "GET"` and `queryParam: "q"`. Auth-bearing backends
(Brave's `X-Subscription-Token`, Tavily's in-body `api_key`) are also supported
through the `authHeaders` / `bodyAuthParams` config fields; this example uses no
auth.

### SearXNG setup

1. Enable the JSON output format in your SearXNG `settings.yml`:

   ```yaml
   search:
     formats:
       - html
       - json
   ```

2. Restart SearXNG so the new format takes effect.

3. Point the example at it:

   ```sh
   export SPORE_WEB_SEARCH_ENDPOINT="http://localhost:8888/search?format=json"
   ```

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
schema-constrained decoding
(`modelParams({ structured_tool_calls: true, stop_sequences: [] })`), which
forces one clean JSON tool call per turn:

```sh
pnpm start -- --structured                # llama3.2 with constrained decoding
pnpm start -- --model gemma4:31b-cloud    # capable model, native tool calls
```

> Note: structured mode exposes an always-available `final` envelope. Weaker
> instruction-followers can take that exit prematurely (emitting an empty answer
> without ever calling `write_file`). If you see an empty `answer (N turn(s)):`
> and no `answer.md`, drop `--structured` and use native tool calling.

## Prerequisites

```sh
ollama serve &
ollama pull llama3.2
# A SearXNG JSON API (see "SearXNG setup" above):
export SPORE_WEB_SEARCH_ENDPOINT="http://localhost:8888/search?format=json"
```

See `.env.example` for all the variables.

## Run

```sh
cd examples/typescript/06-web-research
pnpm install
pnpm start
pnpm start -- --prompt "What are the current options for running WebAssembly outside the browser? Cite sources and write answer.md."
```

`SPORE_OLLAMA_MODEL` / `SPORE_OLLAMA_BASE_URL` override the model id and endpoint,
and `--model` overrides the id for a single run — same as 04 / 05.

`answer.md` is written into `workspace/` and is `.gitignore`d so re-runs stay
clean.
