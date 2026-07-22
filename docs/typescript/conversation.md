# Conversation — TypeScript

> TypeScript code for the [conversation guide](../guides/conversation.md). Full runnable program:
> [`examples/typescript/02-conversational-repl`](../../examples/typescript/02-conversational-repl).

## Thread the state yourself

Build the harness once, reuse it for every turn, and carry the post-run `SessionState` forward:

```ts
import {
  HarnessBuilder,
  OllamaModelInterface,
  SessionId,
  newTask,
  reactPerLoop,
  runResultSessionState,
  type RunResult,
  type SessionState,
} from "@spore/core";

const harness = HarnessBuilder.conversational(
  new OllamaModelInterface("llama3.2"),
).build();

// One session id for the whole conversation; one running state we thread back in.
const sessionId = SessionId.generate();
let state: SessionState = { messages: [], extras: {} };

while (true) {
  const line = await readUserLine();            // your input source; break on EOF / "/exit"
  if (line === "") continue;

  const task = newTask(line, sessionId, reactPerLoop(4));
  // Thread the running state in. The harness appends `line` as the new user
  // message before calling the model.
  const result = await harness.run({ task, session_state: state });

  if (result.kind === "success") {
    console.log(`bot> ${result.output}`);
    // Carry the post-run state forward — lossless, no reconstruction.
    // It already holds this turn's user + assistant (+ tool) messages.
    state = runResultSessionState(result);
  } else {
    console.error(`bot> [did not succeed: ${result.kind}]`);
  }
}
```

The key line is `state = runResultSessionState(result);` — read the post-run `SessionState` off
the `RunResult` and reuse it. `state.messages.length` tells you how much history has accumulated.

## Let the harness persist it

Wire a `StorageProvider` and enable auto-persist; then reuse the **id** instead of threading state:

```ts
import {
  HarnessBuilder,
  OllamaModelInterface,
  InMemoryStorageProvider,
  StorageProvider,
  SessionId,
  newTask,
  reactPerLoop,
} from "@spore/core";

const storage = StorageProvider.single(new InMemoryStorageProvider());
const harness = HarnessBuilder.conversational(
  new OllamaModelInterface("llama3.2"),
)
  .storage(storage)
  .autoPersistSessions(true)
  .build();

// Each turn: same `sessionId`, no `session_state`. The harness loads the
// prior state and saves the new one automatically.
const sessionId = SessionId.generate();
const task = newTask(line, sessionId, reactPerLoop(4));
await harness.run({ task });
```

Swap `InMemoryStorageProvider` for `FileSystemStorageProvider` (or a composed provider) to
survive restarts — see [storage-seam](../reference/storage-seam.md).

## Notes

- **`session_state`** seeds a run with prior history. Omit it (and rely on
  auto-persist) for the hands-free path.
- **An empty `SessionState`** is `{ messages: [], extras: {} }`. A success `RunResult`'s
  `session_state` is the post-run conversation, carrying tool-call and tool-result messages too —
  which is why it round-trips a tool-using agent losslessly. Read it via `runResultSessionState`
  (it defaults absence to an empty state, the TS analogue of Rust's `#[serde(default)]`).
- **Reuse one `SessionId`** for the whole conversation; that id is the thread identity and the
  auto-persist key.

## Run it

```sh
cd examples/typescript/02-conversational-repl
pnpm install
pnpm start        # then chat; /exit or Ctrl-D to quit
```