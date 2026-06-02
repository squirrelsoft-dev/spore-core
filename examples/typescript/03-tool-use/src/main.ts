/**
 * spore-core example 03 — ReAct with local tools.
 *
 * The agent now *acts*: it thinks, calls a tool, observes the result, and loops
 * until it can answer. The tools here are deliberately trivial — `calculator`,
 * `get_current_time`, `reverse_string` — because the star of this example is the
 * **Think -> Act -> Observe** loop, not the tools.
 *
 * ## What it shows
 *
 * - Implementing the harness-loop `ToolRegistry` directly: `schemas()` advertises
 *   the tools to the model, `dispatch()` runs them. No filesystem, no sandbox
 *   needed — these are pure functions, so the `conversational(model)` defaults
 *   (incl. `NullSandbox`) are fine; we only override the tool registry.
 * - The loop itself: the program prints each turn (Think) and each tool call +
 *   result (Act / Observe), so you can watch the agent work.
 *
 * ## Run it
 *
 * ```sh
 * ollama serve &
 * ollama pull llama3.2
 * pnpm install
 * pnpm start
 * pnpm start -- --prompt "reverse the word 'mycelium' and multiply 6 by 7"
 * ```
 */

import {
  HarnessBuilder,
  OLLAMA_DEFAULT_BASE_URL,
  OllamaModelInterface,
  SessionId,
  newTask,
  toolOutput,
  type HarnessStreamEvent,
  type ToolCall,
  type ToolOutput,
  type ToolRegistry,
  type ToolSchema,
} from "@spore/core";

/**
 * Three trivial, pure-compute tools, exposed through the harness-loop tool
 * registry. `schemas()` is what the model sees; `dispatch()` is what runs.
 */
class LocalTools implements ToolRegistry {
  schemas(): ToolSchema[] {
    return [
      schema(
        "calculator",
        "Compute a binary arithmetic operation. 'op' is one of + - * /.",
        {
          a: { type: "number" },
          b: { type: "number" },
          op: { type: "string", enum: ["+", "-", "*", "/"] },
        },
        ["a", "b", "op"],
      ),
      schema(
        "get_current_time",
        "Return the current time of day as HH:MM:SS UTC. Takes no arguments.",
        {},
        [],
      ),
      schema("reverse_string", "Reverse the characters in a string.", { text: { type: "string" } }, [
        "text",
      ]),
    ];
  }

  // No tool here halts the loop; an unknown tool simply yields a recoverable error.
  isAlwaysHalt(_toolName: string): boolean {
    return false;
  }

  async dispatch(call: ToolCall): Promise<ToolOutput> {
    const input = (call.input ?? {}) as Record<string, unknown>;
    const argsJson = JSON.stringify(input);
    let result: { ok: true; content: string } | { ok: false; message: string };
    switch (call.name) {
      case "calculator":
        result = calculator(input);
        break;
      case "get_current_time":
        result = { ok: true, content: currentTime() };
        break;
      case "reverse_string":
        result = reverseString(input);
        break;
      default:
        result = { ok: false, message: `unknown tool: ${call.name}` };
        break;
    }
    // Print the Act + Observe step so the loop is visible.
    if (result.ok) {
      console.log(`    act    -> ${call.name}(${argsJson}) = ${result.content}`);
      return toolOutput.success(result.content);
    }
    console.log(`    act    -> ${call.name}(${argsJson}) failed: ${result.message}`);
    return toolOutput.error(result.message);
  }
}

type ToolResult = { ok: true; content: string } | { ok: false; message: string };

function calculator(input: Record<string, unknown>): ToolResult {
  const a = num(input, "a");
  if (!a.ok) return a;
  const b = num(input, "b");
  if (!b.ok) return b;
  const op = typeof input.op === "string" ? input.op : null;
  if (op == null) return { ok: false, message: "missing string 'op'" };
  let value: number;
  switch (op) {
    case "+":
      value = a.value + b.value;
      break;
    case "-":
      value = a.value - b.value;
      break;
    case "*":
      value = a.value * b.value;
      break;
    case "/":
      if (b.value === 0) return { ok: false, message: "division by zero" };
      value = a.value / b.value;
      break;
    default:
      return { ok: false, message: `unknown op '${op}' (use + - * /)` };
  }
  return { ok: true, content: String(value) };
}

function num(
  input: Record<string, unknown>,
  key: string,
): { ok: true; value: number } | { ok: false; message: string } {
  const raw = input[key];
  if (raw === undefined) return { ok: false, message: `missing number '${key}'` };
  // Models often pass numbers as JSON strings ("144"); accept either.
  if (typeof raw === "number" && Number.isFinite(raw)) return { ok: true, value: raw };
  if (typeof raw === "string") {
    const parsed = Number.parseFloat(raw.trim());
    if (Number.isFinite(parsed)) return { ok: true, value: parsed };
  }
  return { ok: false, message: `'${key}' is not a number: ${JSON.stringify(raw)}` };
}

function currentTime(): string {
  const secs = Math.floor(Date.now() / 1000);
  const pad = (n: number) => String(n).padStart(2, "0");
  return `${pad(Math.floor(secs / 3600) % 24)}:${pad(Math.floor(secs / 60) % 60)}:${pad(secs % 60)} UTC`;
}

function reverseString(input: Record<string, unknown>): ToolResult {
  const text = input.text;
  if (typeof text !== "string") return { ok: false, message: "missing string 'text'" };
  return { ok: true, content: [...text].reverse().join("") };
}

function schema(
  name: string,
  description: string,
  properties: Record<string, unknown>,
  required: string[],
): ToolSchema {
  return {
    name,
    description,
    input_schema: { type: "object", properties, required },
  };
}

async function main(): Promise<void> {
  const args = process.argv.slice(2);
  const modelId = argValue(args, "--model") ?? process.env.SPORE_OLLAMA_MODEL ?? "llama3.2";
  const baseUrl = process.env.SPORE_OLLAMA_BASE_URL ?? OLLAMA_DEFAULT_BASE_URL;
  const prompt =
    argValue(args, "--prompt") ??
    "Use your tools to answer: what is 144 divided by 12, what is the current time, " +
      "and what is 'harness' reversed?";

  // Same conversational harness as 01/02 — we only swap in our tool registry.
  const model = OllamaModelInterface.withBaseUrl(modelId, baseUrl);
  const harness = HarnessBuilder.conversational(model).toolRegistry(new LocalTools()).build();

  const task = newTask(prompt, SessionId.generate(), { kind: "re_act", max_iterations: 6 });

  console.log(`model  : ${modelId}`);
  console.log(`prompt : ${prompt}\n`);

  // Print each turn so the "Think" steps are visible alongside the tool calls.
  const onStream = (event: HarnessStreamEvent): void => {
    if (event.kind === "turn_start") {
      console.log(`think  · turn ${event.turn}`);
    }
  };

  const result = await harness.run({ task, on_stream: onStream });
  if (result.kind === "success") {
    console.log(`\nanswer (${result.turns} turn(s)): ${result.output}`);
    return;
  }
  console.error(`\nrun did not succeed: ${JSON.stringify(result)}`);
  process.exit(1);
}

function argValue(args: string[], flag: string): string | undefined {
  const i = args.indexOf(flag);
  return i >= 0 ? args[i + 1] : undefined;
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
