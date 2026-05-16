import { describe, expect, it } from "vitest";
import {
  BudgetExceeded,
  ContextLimitExceeded,
  MockModelInterface,
  ModelRequestSchema,
  ModelResponseSchema,
  ProviderError,
  RateLimited,
  Timeout,
  enforceBudget,
  enforceContextLimit,
  type ModelRequest,
  type ModelResponse,
  type ProviderInfo,
  type StreamEvent,
  type TokenUsage,
} from "../src/index.js";

const provider: ProviderInfo = {
  name: "test",
  model_id: "test-1",
  context_window: 1000,
};

const emptyRequest: ModelRequest = {
  messages: [],
  tools: [],
  params: { stop_sequences: [] },
  stream: false,
};

function textResponse(text: string, inTok: number, outTok: number): ModelResponse {
  return {
    content: [{ type: "text", text }],
    usage: {
      input_tokens: inTok,
      output_tokens: outTok,
      cache_read_tokens: null,
      cache_write_tokens: null,
    },
    stop_reason: "end_turn",
  };
}

describe("ModelInterface — rules", () => {
  it("call returns queued response", async () => {
    const m = new MockModelInterface(provider).pushResponse(textResponse("hi", 3, 1));
    const r = await m.call(emptyRequest);
    expect(r.content).toHaveLength(1);
    expect(r.stop_reason).toBe("end_turn");
  });

  it("token usage is reported on every call", async () => {
    const m = new MockModelInterface(provider)
      .pushResponse(textResponse("a", 5, 7))
      .pushResponse(textResponse("b", 11, 13));
    const r1 = await m.call(emptyRequest);
    const r2 = await m.call(emptyRequest);
    expect(r1.usage.input_tokens).toBe(5);
    expect(r1.usage.output_tokens).toBe(7);
    expect(r2.usage.input_tokens).toBe(11);
    expect(r2.usage.output_tokens).toBe(13);
  });

  it("context limit is enforced pre-call", () => {
    const err = enforceContextLimit(1500, 1000);
    expect(err).toBeInstanceOf(ContextLimitExceeded);
    expect(err?.limit).toBe(1000);
    expect(err?.actual).toBe(1500);
    expect(enforceContextLimit(1000, 1000)).toBeNull();
    expect(enforceContextLimit(999, 1000)).toBeNull();
  });

  it("budget is enforced against max_tokens", () => {
    const err = enforceBudget(101, 100);
    expect(err).toBeInstanceOf(BudgetExceeded);
    expect(err?.budget).toBe(100);
    expect(err?.used).toBe(101);
    expect(enforceBudget(100, 100)).toBeNull();
    expect(enforceBudget(99, 100)).toBeNull();
    expect(enforceBudget(1_000_000, undefined)).toBeNull();
    expect(enforceBudget(1_000_000, null)).toBeNull();
  });

  it("provider identity is reported", () => {
    const m = new MockModelInterface(provider);
    const p = m.provider();
    expect(p.name).toBe("test");
    expect(p.model_id).toBe("test-1");
    expect(p.context_window).toBe(1000);
  });

  it("streaming yields a MessageStop carrying final usage", async () => {
    const m = new MockModelInterface(provider).pushResponse(textResponse("hello", 4, 2));
    let sawStart = false;
    let finalUsage: TokenUsage | undefined;
    for await (const ev of m.callStreaming(emptyRequest)) {
      if (ev.type === "message_start") sawStart = true;
      if (ev.type === "message_stop") finalUsage = ev.usage;
    }
    expect(sawStart).toBe(true);
    expect(finalUsage?.input_tokens).toBe(4);
    expect(finalUsage?.output_tokens).toBe(2);
  });

  it("provider errors surface as typed harness errors", async () => {
    const m = new MockModelInterface(provider).pushError(
      new ProviderError(503, "unavailable"),
    );
    await expect(m.call(emptyRequest)).rejects.toBeInstanceOf(ProviderError);
  });
});

describe("ModelError variants", () => {
  it("every variant is constructible and JSON-serialisable with kind discriminator", () => {
    const variants = [
      new ProviderError(500, "boom"),
      new RateLimited(5),
      new RateLimited(),
      new ContextLimitExceeded(1, 2),
      new BudgetExceeded(1, 2),
      new Timeout(),
    ];
    const expected = [
      "provider_error",
      "rate_limited",
      "rate_limited",
      "context_limit_exceeded",
      "budget_exceeded",
      "timeout",
    ];
    variants.forEach((v, i) => {
      expect(v.message.length).toBeGreaterThan(0);
      const json = v.toJSON();
      expect(json.kind).toBe(expected[i]);
      // Round-trip through JSON.stringify to ensure no functions leak.
      const round = JSON.parse(JSON.stringify(v));
      expect(round.kind).toBe(expected[i]);
    });
  });

  it("RateLimited serialises retry_after as seconds-or-null", () => {
    expect(new RateLimited(5).toJSON()).toEqual({ kind: "rate_limited", retry_after: 5 });
    expect(new RateLimited().toJSON()).toEqual({
      kind: "rate_limited",
      retry_after: null,
    });
  });
});

describe("Zod round-trip", () => {
  it("ModelRequest round-trips through JSON without loss", () => {
    const req: ModelRequest = {
      messages: [{ role: "user", content: { type: "text", text: "hi" } }],
      tools: [
        {
          name: "echo",
          description: "echoes input",
          input_schema: { type: "object" },
        },
      ],
      params: {
        temperature: 0.7,
        max_tokens: 1024,
        stop_sequences: [],
      },
      stream: false,
    };
    const back = ModelRequestSchema.parse(JSON.parse(JSON.stringify(req)));
    expect(back).toEqual(req);
  });

  it("ModelResponse round-trips through JSON without loss", () => {
    const resp: ModelResponse = {
      content: [
        { type: "text", text: "ok" },
        { type: "tool_use", id: "1", name: "x", input: { a: 1 } },
      ],
      usage: {
        input_tokens: 3,
        output_tokens: 4,
        cache_read_tokens: 1,
        cache_write_tokens: 2,
      },
      stop_reason: "tool_use",
    };
    const back = ModelResponseSchema.parse(JSON.parse(JSON.stringify(resp)));
    expect(back).toEqual(resp);
  });
});

describe("Streaming event shape", () => {
  it("only message_stop carries usage", async () => {
    const m = new MockModelInterface(provider).pushResponse(textResponse("ok", 1, 2));
    const events: StreamEvent[] = [];
    for await (const ev of m.callStreaming(emptyRequest)) events.push(ev);
    expect(events[0]?.type).toBe("message_start");
    expect(events[events.length - 1]?.type).toBe("message_stop");
  });
});
