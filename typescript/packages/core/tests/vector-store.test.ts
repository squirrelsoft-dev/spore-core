import { describe, expect, it } from "vitest";

import { InMemoryVectorStore } from "../src/vector-store/standard.js";
import {
  MetadataFilter,
  VectorStoreErrorException,
  dimensionMismatch,
  vectorRecord,
} from "../src/vector-store/types.js";

function rec(id: string, vector: number[], kind: string) {
  return vectorRecord(id, vector, { kind });
}

describe("InMemoryVectorStore", () => {
  it("add/search orders by similarity (highest first)", async () => {
    const store = new InMemoryVectorStore(3);
    await store.add([
      rec("a", [1, 0, 0], "chunk"),
      rec("b", [0, 1, 0], "chunk"),
      rec("c", [0.9, 0.1, 0], "chunk"),
    ]);
    const hits = await store.search([1, 0, 0], 2);
    expect(hits.length).toBe(2);
    expect(hits[0]!.id).toBe("a");
    expect(hits[1]!.id).toBe("c");
  });

  it("filter scopes search (applied as part of search, not post-truncation)", async () => {
    const store = new InMemoryVectorStore(2);
    await store.add([
      rec("a", [1, 0], "chunk"),
      rec("b", [1, 0], "summary"),
    ]);
    const hits = await store.search([1, 0], 10, MetadataFilter.new().eq("kind", "summary"));
    expect(hits.length).toBe(1);
    expect(hits[0]!.id).toBe("b");
  });

  it("upsert by id does not duplicate", async () => {
    const store = new InMemoryVectorStore(2);
    await store.add([rec("a", [1, 0], "chunk")]);
    await store.add([rec("a", [0, 1], "chunk")]);
    expect(await store.len()).toBe(1);
  });

  it("delete returns count and removes matching records", async () => {
    const store = new InMemoryVectorStore(2);
    await store.add([
      rec("a", [1, 0], "chunk"),
      rec("b", [0, 1], "summary"),
    ]);
    const removed = await store.delete(MetadataFilter.new().eq("kind", "chunk"));
    expect(removed).toBe(1);
    expect(await store.len()).toBe(1);
  });

  it("empty filter delete = full clear", async () => {
    const store = new InMemoryVectorStore(2);
    await store.add([rec("a", [1, 0], "chunk"), rec("b", [0, 1], "summary")]);
    const removed = await store.delete(MetadataFilter.new());
    expect(removed).toBe(2);
    expect(await store.isEmpty()).toBe(true);
  });

  it("dimension mismatch rejected with typed Error carrying structured detail", async () => {
    const store = new InMemoryVectorStore(3);
    const err = await store.add([rec("a", [1, 0], "chunk")]).catch((e) => e);
    expect(err).toBeInstanceOf(VectorStoreErrorException);
    expect((err as VectorStoreErrorException).detail).toEqual(dimensionMismatch(3, 2));
  });

  it("metadata filters compare arrays/objects structurally", async () => {
    const store = new InMemoryVectorStore(2);
    await store.add([
      vectorRecord("a", [1, 0], { kind: "chunk", tags: ["a", "b"], scope: { project: "p" } }),
      vectorRecord("b", [1, 0], { kind: "chunk", tags: ["b"], scope: { project: "p" } }),
    ]);
    const hits = await store.search(
      [1, 0],
      10,
      MetadataFilter.new().eq("tags", ["a", "b"]).eq("scope", { project: "p" }),
    );
    expect(hits.map((h) => h.id)).toEqual(["a"]);
  });

  it("per-instance isolation: two stores do not share rows", async () => {
    const s1 = new InMemoryVectorStore(2);
    const s2 = new InMemoryVectorStore(2);
    await s1.add([rec("a", [1, 0], "chunk")]);
    expect(await s2.len()).toBe(0);
    const hits = await s2.search([1, 0], 10);
    expect(hits).toEqual([]);
  });

  it("len/isEmpty", async () => {
    const store = new InMemoryVectorStore(2);
    expect(await store.isEmpty()).toBe(true);
    await store.add([rec("a", [1, 0], "chunk")]);
    expect(await store.isEmpty()).toBe(false);
    expect(await store.len()).toBe(1);
  });
});