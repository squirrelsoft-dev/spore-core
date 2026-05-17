/**
 * StandardGuideRegistry — in-memory reference implementation of
 * {@link GuideRegistry} (spore-core issue #9).
 *
 * Mirrors `rust/crates/spore-core/src/guide_registry.rs#StandardGuideRegistry`
 * — same rules, same fixture outcomes. Suitable for tests and short-lived
 * processes; production deployments use a durable backing store.
 *
 * Conflict heuristic: two guides conflict when they share a domain and have
 * Jaccard token overlap >= `CONFLICT_THRESHOLD` (0.6) but non-identical content.
 *
 * Time math: RFC 3339 strings are parsed via pure integer arithmetic so the
 * provider has no external dependency on a date library; the window cutoff is
 * compared lexicographically against `recorded_at`. Tests pin `now` via
 * {@link StandardGuideRegistry.setNow} for determinism.
 */

import {
  type Guide,
  type GuideConflict,
  GuideId,
  type GuideQuery,
  type GuideRegistry,
  GuideRegistryError,
  type GuideStatus,
  type GuideUsageRecord,
  type ImprovementSignal,
  type PendingReason,
  Timestamp,
} from "./types.js";

const CONFLICT_THRESHOLD = 0.6;

// ============================================================================
// Helpers
// ============================================================================

function validateContent(s: string): void {
  if (s.trim().length === 0) {
    throw GuideRegistryError.validationFailed("content must not be empty");
  }
}

function tokenize(s: string): string[] {
  return s
    .toLowerCase()
    .split(/[^a-z0-9]+/u)
    .filter((t) => t.length > 0);
}

function jaccard(a: string, b: string): number {
  const ta = new Set(tokenize(a));
  const tb = new Set(tokenize(b));
  if (ta.size === 0 && tb.size === 0) return 0;
  let inter = 0;
  for (const t of ta) if (tb.has(t)) inter += 1;
  const union = ta.size + tb.size - inter;
  return union === 0 ? 0 : inter / union;
}

function enforceMetaAgentPending(g: Guide): void {
  if (g.source.kind !== "meta_agent_proposed") return;
  g.status = {
    kind: "pending_review",
    reason: { kind: "automated_proposal" },
    since: g.source.proposed_at,
  };
}

// ── Time math (pure integer; no external date lib) ─────────────────────────

function daysFromCivil(y: number, m: number, d: number): number {
  const yy = m <= 2 ? y - 1 : y;
  const era = Math.floor(yy / 400);
  const yoe = yy - era * 400;
  const monthAdj = m > 2 ? m - 3 : m + 9;
  const doy = Math.floor((153 * monthAdj + 2) / 5) + d - 1;
  const doe = yoe * 365 + Math.floor(yoe / 4) - Math.floor(yoe / 100) + doy;
  return era * 146097 + doe - 719468;
}

function civilFromTotalSecs(totalSecs: number): [number, number, number, number, number, number] {
  const days = Math.floor(totalSecs / 86400);
  const rem = totalSecs - days * 86400;
  const hh = Math.floor(rem / 3600);
  const mm = Math.floor((rem - hh * 3600) / 60);
  const ss = rem - hh * 3600 - mm * 60;
  const z = days + 719468;
  const era = Math.floor(z / 146097);
  const doe = z - era * 146097;
  const yoe = Math.floor(
    (doe - Math.floor(doe / 1460) + Math.floor(doe / 36524) - Math.floor(doe / 146096)) / 365,
  );
  let y = yoe + era * 400;
  const doy = doe - (365 * yoe + Math.floor(yoe / 4) - Math.floor(yoe / 100));
  const mp = Math.floor((5 * doy + 2) / 153);
  const d = doy - Math.floor((153 * mp + 2) / 5) + 1;
  const m = mp < 10 ? mp + 3 : mp - 9;
  if (m <= 2) y += 1;
  return [y, m, d, hh, mm, ss];
}

function parseRfc3339Secs(s: string): number | null {
  const trimmed = s.endsWith("Z") ? s.slice(0, -1) : s;
  const tIdx = trimmed.indexOf("T");
  if (tIdx < 0) return null;
  const date = trimmed.slice(0, tIdx);
  const time = trimmed.slice(tIdx + 1).split(".")[0]!;
  const dParts = date.split("-");
  if (dParts.length !== 3) return null;
  const y = Number(dParts[0]);
  const mo = Number(dParts[1]);
  const d = Number(dParts[2]);
  const tParts = time.split(":");
  if (tParts.length !== 3) return null;
  const hh = Number(tParts[0]);
  const mm = Number(tParts[1]);
  const ss = Number(tParts[2]);
  if (
    !Number.isFinite(y) ||
    !Number.isFinite(mo) ||
    !Number.isFinite(d) ||
    !Number.isFinite(hh) ||
    !Number.isFinite(mm) ||
    !Number.isFinite(ss)
  ) {
    return null;
  }
  return daysFromCivil(y, mo, d) * 86400 + hh * 3600 + mm * 60 + ss;
}

function pad(n: number, width: number): string {
  return n.toString().padStart(width, "0");
}

function formatRfc3339Secs(secs: number): string {
  const [y, mo, d, hh, mm, ss] = civilFromTotalSecs(secs);
  return `${pad(y, 4)}-${pad(mo, 2)}-${pad(d, 2)}T${pad(hh, 2)}:${pad(mm, 2)}:${pad(ss, 2)}Z`;
}

function rfc3339Subtract(now: Timestamp, windowSecs: number): Timestamp | null {
  const secs = parseRfc3339Secs(now.value);
  if (secs === null) return null;
  return new Timestamp(formatRfc3339Secs(secs - windowSecs));
}

function nowRfc3339(): Timestamp {
  const secs = Math.floor(Date.now() / 1000);
  return new Timestamp(formatRfc3339Secs(secs));
}

function inWindow(t: Timestamp, cutoff: Timestamp | null): boolean {
  if (cutoff === null) return true;
  return t.value >= cutoff.value;
}

// ── Cloning (defensive copies on the boundaries) ───────────────────────────

function cloneGuide(g: Guide): Guide {
  return {
    id: new GuideId(g.id.value),
    name: g.name,
    content: g.content,
    guide_type: g.guide_type,
    domain: g.domain ?? null,
    source: cloneSource(g.source),
    status: cloneStatus(g.status),
    created_at: g.created_at,
    last_used: g.last_used ?? null,
    version: g.version,
  };
}

function cloneSource(s: Guide["source"]): Guide["source"] {
  switch (s.kind) {
    case "manual":
      return { kind: "manual" };
    case "session_generated":
      return { kind: "session_generated", session_id: s.session_id };
    case "trace_distilled":
      return { kind: "trace_distilled", session_ids: [...s.session_ids] };
    case "meta_agent_proposed":
      return { kind: "meta_agent_proposed", proposed_at: s.proposed_at };
  }
}

function cloneStatus(s: GuideStatus): GuideStatus {
  switch (s.kind) {
    case "active":
      return { kind: "active" };
    case "pending_review":
      return { kind: "pending_review", reason: clonePendingReason(s.reason), since: s.since };
    case "deprecated":
      return { kind: "deprecated", reason: s.reason, at: s.at };
    case "stale":
      return { kind: "stale", last_used: s.last_used };
  }
}

function clonePendingReason(r: PendingReason): PendingReason {
  switch (r.kind) {
    case "automated_proposal":
      return { kind: "automated_proposal" };
    case "performance_degradation":
      return { kind: "performance_degradation", failure_rate_delta: r.failure_rate_delta };
    case "conflict_detected":
      return {
        kind: "conflict_detected",
        conflicts_with: r.conflicts_with.map((g) => new GuideId(g.value)),
      };
    case "manual_flag":
      return { kind: "manual_flag", note: r.note };
  }
}

function cloneUsage(r: GuideUsageRecord): GuideUsageRecord {
  return {
    guide_id: new GuideId(r.guide_id.value),
    session_id: r.session_id,
    task_domain: r.task_domain ?? null,
    outcome:
      r.outcome.kind === "failure"
        ? { kind: "failure", reason: r.outcome.reason }
        : { kind: r.outcome.kind },
    recorded_at: r.recorded_at,
  };
}

// ============================================================================
// StandardGuideRegistry
// ============================================================================

export class StandardGuideRegistry implements GuideRegistry {
  private readonly guides = new Map<string, Guide>();
  private readonly usage: GuideUsageRecord[] = [];
  private nowOverride: Timestamp | null = null;

  /** Test helper: pin `now` for deterministic window math. */
  setNow(ts: Timestamp): void {
    this.nowOverride = ts;
  }

  private currentNow(): Timestamp {
    return this.nowOverride ?? nowRfc3339();
  }

  async register(guide: Guide): Promise<GuideId> {
    validateContent(guide.content);
    const working = cloneGuide(guide);
    enforceMetaAgentPending(working);

    const conflicts = await this.checkConflicts(working.content, working.domain ?? null);
    const first = conflicts[0];
    if (first) {
      throw GuideRegistryError.conflictDetected({
        guide_a: new GuideId(working.id.value),
        guide_b: first.guide_b,
        reason: first.reason,
      });
    }

    this.guides.set(working.id.value, working);
    return new GuideId(working.id.value);
  }

  async select(query: GuideQuery): Promise<Guide[]> {
    const wantDomain = query.domain ?? null;
    const types = query.guide_types;
    const scored: Array<[number, Guide]> = [];
    for (const g of this.guides.values()) {
      if (g.status.kind !== "active") continue;
      const haveDomain = g.domain ?? null;
      if (wantDomain !== null) {
        if (haveDomain === null) continue;
        if (haveDomain !== wantDomain) continue;
      }
      if (types.length > 0 && !types.includes(g.guide_type)) continue;
      const score = jaccard(query.task_instruction, g.content);
      scored.push([score, cloneGuide(g)]);
    }
    scored.sort((a, b) => b[0] - a[0]);
    return scored.map(([, g]) => g);
  }

  async recordUsage(record: GuideUsageRecord): Promise<void> {
    const existing = this.guides.get(record.guide_id.value);
    if (!existing) {
      throw GuideRegistryError.notFound(new GuideId(record.guide_id.value));
    }
    existing.last_used = record.recorded_at;
    this.usage.push(cloneUsage(record));
  }

  async usageHistory(id: GuideId): Promise<GuideUsageRecord[]> {
    if (!this.guides.has(id.value)) {
      throw GuideRegistryError.notFound(new GuideId(id.value));
    }
    return this.usage.filter((r) => r.guide_id.value === id.value).map(cloneUsage);
  }

  async deprecate(id: GuideId, reason: string): Promise<void> {
    const g = this.guides.get(id.value);
    if (!g) throw GuideRegistryError.notFound(new GuideId(id.value));
    g.status = { kind: "deprecated", reason, at: this.currentNow() };
  }

  async markPendingReview(id: GuideId, reason: PendingReason): Promise<void> {
    const g = this.guides.get(id.value);
    if (!g) throw GuideRegistryError.notFound(new GuideId(id.value));
    g.status = {
      kind: "pending_review",
      reason: clonePendingReason(reason),
      since: this.currentNow(),
    };
  }

  async promoteToActive(id: GuideId): Promise<void> {
    const g = this.guides.get(id.value);
    if (!g) throw GuideRegistryError.notFound(new GuideId(id.value));
    if (g.status.kind !== "pending_review") {
      throw GuideRegistryError.validationFailed("promote_to_active requires PendingReview status");
    }
    g.status = { kind: "active" };
  }

  async analyzePerformance(
    windowSecs: number,
    minFailureRateDelta: number,
    minPatternOccurrences: number,
  ): Promise<ImprovementSignal[]> {
    const now = this.currentNow();
    const cutoff = rfc3339Subtract(now, windowSecs);

    const inWin = this.usage.filter((r) => inWindow(r.recorded_at, cutoff));
    const signals: ImprovementSignal[] = [];

    const totalFailures = inWin.filter((r) => r.outcome.kind === "failure").length;
    const totalRecords = inWin.length;

    for (const [gidStr, guide] of this.guides) {
      // Conflict-derived signal.
      if (
        guide.status.kind === "pending_review" &&
        guide.status.reason.kind === "conflict_detected"
      ) {
        const other = guide.status.reason.conflicts_with[0];
        if (other) {
          signals.push({
            kind: "conflict_resolution_needed",
            conflict: {
              guide_a: new GuideId(gidStr),
              guide_b: new GuideId(other.value),
              reason: "pending-review conflict",
            },
          });
        }
      }

      const withGuide = inWin.filter((r) => r.guide_id.value === gidStr);
      if (withGuide.length === 0) continue;
      const withFail = withGuide.filter((r) => r.outcome.kind === "failure").length;
      const withRate = withFail / withGuide.length;
      const withoutCount = Math.max(0, totalRecords - withGuide.length);
      const baseline = withoutCount === 0 ? 0 : (totalFailures - withFail) / withoutCount;
      if (withRate - baseline >= minFailureRateDelta) {
        signals.push({
          kind: "guide_deprecation_recommended",
          guide_id: new GuideId(gidStr),
          reason: `failure-rate delta ${(withRate - baseline).toFixed(2)} (with=${withRate.toFixed(2)} vs baseline=${baseline.toFixed(2)})`,
        });
      }
    }

    // Pattern detection across failures.
    const patterns = new Map<string, GuideUsageRecord["session_id"][]>();
    for (const r of inWin) {
      if (r.outcome.kind !== "failure") continue;
      const list = patterns.get(r.outcome.reason) ?? [];
      list.push(r.session_id);
      patterns.set(r.outcome.reason, list);
    }
    for (const [pattern, sessionIds] of patterns) {
      if (sessionIds.length >= minPatternOccurrences) {
        signals.push({
          kind: "skill_generation_needed",
          pattern,
          session_ids: sessionIds,
        });
      }
    }

    return signals;
  }

  async checkConflicts(
    content: string,
    domain: string | null | undefined,
  ): Promise<GuideConflict[]> {
    const wantDomain = domain ?? null;
    const out: GuideConflict[] = [];
    for (const g of this.guides.values()) {
      if (g.status.kind !== "active") continue;
      const haveDomain = g.domain ?? null;
      if (haveDomain !== wantDomain) continue;
      if (g.content === content) continue;
      const score = jaccard(g.content, content);
      if (score >= CONFLICT_THRESHOLD) {
        out.push({
          guide_a: new GuideId("<new>"),
          guide_b: new GuideId(g.id.value),
          reason: `Jaccard overlap ${score.toFixed(2)} >= ${CONFLICT_THRESHOLD}`,
        });
      }
    }
    return out;
  }

  // ── Test-only inspection helpers ───────────────────────────────────────
  /** @internal — test helper, returns a clone of the current guide. */
  _peekGuide(id: GuideId): Guide | undefined {
    const g = this.guides.get(id.value);
    return g ? cloneGuide(g) : undefined;
  }
}
