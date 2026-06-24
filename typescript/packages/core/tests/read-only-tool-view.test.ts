/**
 * ReadOnlyToolView — the SelfVerifying eval-phase read-only tool filter
 * (spore-core SC-30, issue #153).
 *
 * Mirrors the Rust unit test
 * (`read_only_tool_view_filters_to_readonly_allowlist`): an inner registry
 * advertising read + write + exec tools is wrapped in a view restricted to the
 * read-only allow-list. The view advertises only the intersection, dispatches
 * allow-listed calls through to the inner registry, and blocks the rest with a
 * recoverable error WITHOUT reaching the inner registry.
 */

import { describe, expect, it } from "vitest";

import {
  toolRegistry,
  type ToolCall,
  type ToolOutput,
  type ToolRegistry,
  type ToolSchema,
} from "../src/index.js";

const { READONLY_EVAL_TOOL_NAMES, ReadOnlyToolView } = toolRegistry;

// Inner registry advertising read + write + exec tools; records every dispatch.
class InnerRegistry implements ToolRegistry {
  readonly dispatched: string[] = [];

  async dispatch(call: ToolCall): Promise<ToolOutput> {
    this.dispatched.push(call.name);
    return { kind: "success", content: "ok", truncated: false };
  }

  isAlwaysHalt(_toolName: string): boolean {
    return false;
  }

  schemas(): ToolSchema[] {
    return ["read_file", "write_file", "bash"].map((name) => ({
      name,
      description: "",
      input_schema: { type: "object" },
    }));
  }
}

function call(id: string, name: string): ToolCall {
  return { id, name, input: {} };
}

describe("ReadOnlyToolView (SC-30, #153)", () => {
  it("advertises only the read-only intersection of the inner catalogue", () => {
    const inner = new InnerRegistry();
    const view = new ReadOnlyToolView(inner, READONLY_EVAL_TOOL_NAMES);

    const names = view.schemas().map((s) => s.name);
    expect(names).toContain("read_file");
    expect(names).not.toContain("write_file");
    expect(names).not.toContain("bash");
  });

  it("dispatches an allow-listed tool through to the inner registry", async () => {
    const inner = new InnerRegistry();
    const view = new ReadOnlyToolView(inner, READONLY_EVAL_TOOL_NAMES);

    const out = await view.dispatch(call("1", "read_file"));
    expect(out.kind).toBe("success");
    expect(inner.dispatched).toEqual(["read_file"]);
  });

  it("blocks a non-allow-listed tool with a recoverable error, never reaching inner", async () => {
    const inner = new InnerRegistry();
    const view = new ReadOnlyToolView(inner, READONLY_EVAL_TOOL_NAMES);

    const out = await view.dispatch(call("2", "write_file"));
    expect(out.kind).toBe("error");
    if (out.kind === "error") {
      expect(out.recoverable).toBe(true);
      expect(out.message).toContain("read-only evaluate phase");
    }
    // The blocked call never reached the inner registry.
    expect(inner.dispatched).toEqual([]);
  });

  it("delegates isAlwaysHalt to the inner registry", () => {
    const inner = new InnerRegistry();
    const view = new ReadOnlyToolView(inner, READONLY_EVAL_TOOL_NAMES);
    expect(view.isAlwaysHalt("read_file")).toBe(false);
  });
});
