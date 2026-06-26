# Hill Climbing Loop — Correctness Fix Plan

**Status:** ready to implement · **Scope:** Rust reference first, then TS/Py/Go parity issue
**Owner seam:** `HillClimbing` loop strategy in the harness
**Date:** 2026-06-25

This plan is self-contained. It assumes no prior conversation context. It cites
exact file/line locations as of this writing (line numbers may drift — grep the
named symbols if they don't match).

---

## 1. Problem: the loop does not climb

`HillClimbing` today is structurally incapable of behaving like local search,
because it has **no durable pointer to the accepted candidate**. The only thing
that survives across iterations is a bare score (`current_best: f64`); the
"candidate" is purely the mutable shared working tree, and the harness never
commits it.

### Evidence (Rust reference)

- `current_best` is an `f64` score, not a candidate/SHA — `rust/crates/spore-core/src/harness.rs:11429` (`hill_iteration` signature) and the `run` loop var seeded at `harness.rs:3236`.
- The harness **never commits** an accepted candidate — `hill_climbing_commit_hash()` returns `None` unconditionally: `harness.rs:10486` (comment: *"The harness never commits… v1 has no per-keep commit"*). Every ledger row therefore records `commit_hash = ""`.
- Revert is hardcoded `git reset --hard HEAD` — `harness.rs:10464` (`hill_climbing_revert`). Because HEAD never advances, **HEAD is always the original starting commit.**
- Each iteration re-seeds a fresh `SessionState::default()` with only the original instruction and forces `continue_task_list = false` — `harness.rs:3346-3357`. No score and no failure detail are fed back to the agent.

### The concrete bug (keep → discard loses accepted progress)

With `revert_on_no_improvement = true`:

1. iter 1 edits → tree = `start+Δ1`, score S1 > S0 → **kept** (no commit, no revert). Tree = `start+Δ1`, `current_best = S1`.
2. iter 2 edits on top → tree = `start+Δ1+Δ2`, score S2 ≤ S1 → **discarded** → `git reset --hard HEAD` → tree snaps back to **`start`** (HEAD == start), *not* to `start+Δ1`.

The accepted `Δ1` is destroyed. `current_best` still reports `S1`, but the
artifact that produced it is gone. Every later iteration climbs from `start`
while being measured against `S1` — usually unbeatable in one fresh-session
turn, so the loop stagnates and exits.

With `revert_on_no_improvement = false` it is also wrong, differently: discarded
neighbors are never undone, so the tree accumulates rejected changes and the next
neighbor branches from a base already judged worse. `current` (the tree) drifts
through rejected states; the score is just a high-water mark detached from it.

**Net:** today's loop is blind best-of-N independent sampling with a destructive
revert. The only case it handles correctly is the degenerate one (first
iteration, or a monotonic run with zero discards-after-keeps) — which is likely
why existing tests pass.

---

## 2. Root cause, stated as one missing primitive

There is **no durable per-accepted-candidate reference**. Adding one — a real
`current_sha` / `best_sha` backed by a commit per accepted candidate — is
load-bearing for *two* separate correctness problems:

1. **Revert correctness.** "Revert a bad neighbor" must mean `reset --hard
   <current_sha>` (the last *accepted* state), not `reset --hard HEAD` (the
   original start).
2. **Scoring stability (LLM judges).** The mitigation for judge noise is to
   re-score current and neighbor under identical conditions per comparison. That
   is impossible today because current's *artifact* isn't retained — only a stale
   number. The same SHA primitive enables it. This is not hypothetical: the ex10
   `ReadmeQualityEvaluator` is an LLM judge, so this loop is already exposed to
   accept-on-noise.

Fix the primitive first. It is small relative to the worktree refactor and it
fixes the actual bug.

---

## 3. In scope vs. deferred

**IN SCOPE (this change — the correctness core):**

- Durable candidate pointers: commit each accepted candidate; track `current_sha`
  and `best_sha` alongside `current_best`/`best_score`.
- Revert targets `current_sha`, not `HEAD`.
- Ledger rows record the real accepted SHA in `commit_hash` (use the reserved
  `VcsProvider` seam).
- Explicit winner application at loop exit (`reset --hard best_sha`).
- A **non-degenerate regression test**: keep-then-discard, asserting the tree
  returns to the *kept* state, not the start.

**DEFERRED (separate issues — do NOT expand this change to cover them):**

- **Worktree-per-neighbor isolation / parallel / beam.** Not free here: the
  `SandboxProvider` is built around a single `workspace_root` and the loop
  recurses `self.inner.run(cx)` on one shared `cx`. Per-neighbor worktrees means
  per-neighbor sandbox plumbing. The correctness fix needs none of it. Do
  worktrees later, only when parallel/beam or true side-effect isolation is the
  goal.
- **Fresh sandbox per evaluation.** Today one workspace + one sandbox env is
  reused across iterations, so temp files / bound ports / scratch DBs / caches
  leak across iterations and can corrupt scores. Real gap, separate change.
- **LLM-judge re-score-pairwise stability.** Unblocked by this change's SHA
  primitive, but implement as a follow-up (re-evaluate `current_sha`'s artifact
  and the neighbor in the same comparison rather than trusting a stored score).
- **Feedback-informed neighbors.** Passing score + failure detail into the
  iteration prompt is a genuine design decision (trades classical blind-restart
  diversity for faster but more local climbing). Out of scope here.
- **Target-threshold termination** ("stop when score ≥ target"). Today only
  plateau (`max_stagnation`) and budget exist. Small, additive, separate.

---

## 4. Design

### 4.1 State (in `HillClimbingConfig::run`, `harness.rs:3183-3448`)

Add candidate pointers next to the existing score scalars:

```
start_sha     = head_sha()              // rev-parse HEAD at baseline
current_sha   = start_sha               // where the next neighbor branches from
best_sha      = start_sha               // best artifact ever accepted
current_best  = baseline_score          // EXISTS today
best_score    = current_best            // NEW — track separately from day one
```

Keep `current` and `best` as **two distinct pointers even though they are always
equal under strict greedy.** They diverge the moment annealing/random-restart is
added; recording both now avoids a later ledger refactor. Same for
`current_best` vs `best_score`.

### 4.2 Per-iteration decision (in `hill_iteration`, `harness.rs:11420`)

`hill_iteration` already owns the revert call, so localize the git side effects
there. Change its contract to take and return the candidate SHA:

- **On keep** (`should_keep` true): commit the working tree → `nbr_sha`. Set
  `current_sha = nbr_sha`. If the score beats `best_score`, set
  `best_sha = nbr_sha`, `best_score = score`. Record the row with
  `commit_hash = nbr_sha`, status `Kept`.
- **On discard** (or `MetricError`): `reset --hard <current_sha>` (NOT `HEAD`).
  `current_sha`/`best_sha` unchanged. Record row, status `Discarded`/error.

Suggested signature change: return `(new_best_score, new_current_sha,
non_improvement)` (or thread a small struct) instead of today's `(f64, bool)`.

### 4.3 Revert (in `hill_climbing_revert`, `harness.rs:10464`)

Take a target SHA argument. Default path (no `VcsProvider`): `git reset --hard
<sha>` through the sandbox — same mechanism as today, different target. With a
`VcsProvider`, delegate (`vcs.reset_to(sha)` — see 4.5).

### 4.4 Commit + head resolution (replaces `hill_climbing_commit_hash`, `harness.rs:10486`)

Default path commits through the sandbox, consistent with SC-14 / issue #60
("the harness NEVER spawns git directly" — it goes through the sandbox or the
provider):

- `head_sha()` → `git rev-parse HEAD` through the sandbox, parse stdout.
- `commit_accepted()` → `git add -A && git commit -m "hill-climb iter N"` through
  the sandbox, then `head_sha()`. Returns the new SHA.

Best-effort error handling stays as today (swallow + continue), but a failed
*commit* should be logged/recorded — a silently-uncommitted keep reintroduces the
bug. Prefer recording the keep as an error row over silently advancing
`current_best` without a SHA.

### 4.5 `VcsProvider` seam (trait at `harness.rs:5140`)

Extend the trait so non-git workspaces stay correct (the seam was explicitly
reserved for this — see comment at `harness.rs:10481`). Add, with git-through-
sandbox defaults:

- `head(&self) -> Result<String, VcsError>`
- `commit_checkpoint(&self, label) -> Result<String /*sha*/, VcsError>`
- `reset_to(&self, sha) -> Result<(), VcsError>`

Keep existing `log` / `status` / `revert`. The default no-op provider must not
leave the loop broken — i.e. when no provider is wired, the harness uses its own
git-through-sandbox implementation (4.3 / 4.4), it does **not** fall through to a
no-op.

### 4.6 Winner application (loop exit, all return paths in `run`)

On strict greedy `current_sha == best_sha` always and the tree already holds best
(a trailing discard reverts to `current_sha == best_sha`). Still add an explicit
final `reset --hard best_sha` before returning success, so the contract holds
once annealing diverges the two pointers later.

Also surface `best_sha` in the partial JSON returned on budget/stagnation/escalation
exits — `hill_climbing_partial_json(current_best)` at `harness.rs:3364` should
include the best SHA so the caller knows which commit holds the best artifact.

### 4.7 Ledger

`ResultsEntry` already exists (`metric.rs`) and is written to
`.spore/results/{task_id}.tsv`. The schema change is small: `commit_hash` becomes
populated (it's currently always `""`). Confirm the TSV column already exists
(it does — `hill_write_tsv`), so this is data, not schema. Per-iteration row
stays one line: `(iter, commit_hash, metric_value, direction, status, duration,
description)`.

### 4.8 Target loop shape (reference)

```
current_sha = best_sha = head_sha()
current_best = best_score = score(baseline)          # baseline eval, EXISTS
ledger.append(iter=0, sha=current_sha, score, status=Kept)

for iter in 1..:
    if budget_exceeded(): return Failure(BudgetExceeded, partial=best)
    # fresh session re-seeded with instruction (EXISTS), branch base = current_sha
    run(self.inner)                                  # agent proposes a change
    score = evaluate()                               # EXISTS

    if should_keep(score, current_best, dir, min_delta):   # EXISTS, metric.rs:133
        nbr_sha = commit_accepted()                  # NEW
        current_sha, current_best = nbr_sha, score
        if beats(score, best_score):
            best_sha, best_score = nbr_sha, score
        ledger.append(iter, nbr_sha, score, Kept); stagnation = 0
    else:
        if revert_on_no_improvement: reset_hard(current_sha)   # CHANGED target
        ledger.append(iter, head_sha(), score, Discarded); stagnation += 1

    if stagnation >= max_stagnation: break           # EXISTS

reset_hard(best_sha)                                 # NEW explicit winner apply
```

Two lines make it *climbing*: branch base = `current_sha`, and `current_sha =
nbr_sha` on accept.

---

## 5. Regression test (the load-bearing deliverable)

The current tests almost certainly only exercise the degenerate path. Add a test
that fails on `main` and passes after the fix:

**Scenario — keep then discard:**

1. Seed a git workspace; baseline score S0.
2. iter 1: stub the agent to make edit Δ1; stub the evaluator to return S1 > S0
   → expect **Kept**, a new commit, `current_sha` advances.
3. iter 2: stub the agent to make edit Δ2 (on top of Δ1); stub the evaluator to
   return S2 ≤ S1 → expect **Discarded** + revert.
4. **Assert: the working tree equals the iter-1 (kept) state — Δ1 present, Δ2
   gone.** On `main` this fails (tree resets to the original start, Δ1 lost).
5. Assert `current_best == S1` AND the tree actually reproduces S1 (score the
   reverted tree, expect S1) — i.e. score and artifact have not diverged.

Use a real in-memory/temp git workspace, not a no-op store — per the
`no-op store silently disables durable behavior` lesson, a no-op VCS path would
assert the opposite while passing. Wire a real git-backed sandbox.

Add a second test for `revert_on_no_improvement = false`: assert that a discarded
neighbor's changes do **not** become the base of the next iteration (next
neighbor branches from `current_sha`, i.e. the last kept commit), if that
behavior is adopted — see Decision D3.

---

## 6. Decisions (pinned, with rationale)

- **D1 — Commit through the sandbox in the default path (no `VcsProvider`
  required).** Recommended. The harness already shells `git reset` through the
  sandbox for revert; shelling `git add/commit/rev-parse` is consistent with
  SC-14 ("never spawn git directly; go through sandbox or provider"). Alternative
  (require a `VcsProvider` for commits) leaves the default path broken — rejected.

- **D2 — Two pointers (`current`/`best`) and two scores from day one**, even
  though equal under greedy. Cheap now; avoids a ledger refactor when annealing/
  restart lands.

- **D3 — `revert_on_no_improvement` semantics.** After this fix, `true` =
  `reset --hard current_sha`. Open sub-question: when `false`, should the next
  neighbor still branch from `current_sha` (correct hill climbing) or from the
  drifted tree (today's behavior)? **Recommend: always branch from
  `current_sha`** — i.e. the flag controls only whether the workspace is
  physically reset, not which base the climb uses. Without worktrees, "branch
  from current_sha but don't reset the tree" isn't physically meaningful on a
  single shared tree, so in practice `false` may have to imply a soft reset too.
  Flag for the implementer: if keeping a meaningful `false` mode on a shared tree
  is awkward, prefer making the reset unconditional and documenting that
  `revert_on_no_improvement = false` is only honored under worktree isolation
  (deferred). Confirm with the owner before removing the flag's effect.

- **D4 — Dirty start state.** Capture `head_sha()` at baseline. Decide whether a
  workspace that starts with uncommitted changes is (a) committed as the baseline
  candidate, or (b) rejected as misconfigured. **Recommend (a)**: commit the
  starting tree as the baseline so `current_sha`/reverts are always well-defined.

- **D5 — Commit failure handling.** A keep whose commit fails must not silently
  advance `current_best` without a SHA (that reintroduces the bug). Record it as
  an error row and treat as non-improvement.

---

## 7. Files to touch (Rust reference)

- `rust/crates/spore-core/src/harness.rs`
  - `HillClimbingConfig` struct — `~583-599` (no field change expected unless D3
    alters the flag).
  - `HillClimbingConfig::run` — `3183-3448` (add pointer state, thread SHAs,
    winner apply, partial JSON).
  - `hill_baseline` — `11338` (capture `start_sha`; optionally commit baseline per D4).
  - `hill_iteration` — `11420` (commit-on-keep, reset-to-current_sha-on-discard,
    return SHA; today returns `(f64, bool)`).
  - `hill_climbing_revert` — `10464` (take target SHA).
  - `hill_climbing_commit_hash` — `10486` (replace with real commit/head helpers).
  - `VcsProvider` trait — `5140` (add `head` / `commit_checkpoint` / `reset_to`,
    with git-through-sandbox defaults).
  - `hill_climbing_partial_json` — near `3364` (include `best_sha`).
- `rust/crates/spore-core/src/metric.rs`
  - `should_keep` — `133-144` (no change; it's already correct strict-greedy).
  - `ResultsEntry` — confirm `commit_hash` field + TSV column (exists; now populated).
- Live example: `examples/.../10-*` hill-climbing example — verify end-to-end
  that the kept artifact survives a subsequent discard.

---

## 8. Cross-language sequencing

Per the established workflow (`cross-language-fix-sequencing`): **land + verify
Rust first, including the live example, THEN file a single GitHub parity issue
for TS/Py/Go.** Do not port all four at once.

Equivalent locations for the parity issue:

- **TypeScript:** `typescript/packages/core/src/harness/standard.ts` — `hill_baseline`/`hill_iteration` around `~1718-1860`.
- **Python:** `python/packages/spore_core/src/spore_core/harness.py` — `hill_iteration` around `~9067-9157`.
- **Go:** `go/spore-core/hill_climbing.go` — the HillClimbing strategy driver (whole file).

Per `consumer-friction-port-patterns`: copy any model-/file-facing strings (commit
messages, TSV values) byte-identical across languages; watch for import-layering
and each language's git-invocation idiom through its sandbox seam.

---

## 9. Acceptance criteria

- [ ] `current_sha`/`best_sha` exist and are committed per accepted candidate.
- [ ] Revert targets `current_sha`, never the original `HEAD`.
- [ ] Ledger rows carry the real accepted SHA (no longer always `""`).
- [ ] Winner explicitly applied (`reset --hard best_sha`) at exit; partial JSON
      includes `best_sha`.
- [ ] Keep-then-discard regression test added and passing (tree returns to the
      kept state, and re-scoring the reverted tree reproduces the kept score).
- [ ] No worktree, fresh-sandbox, feedback, re-score, or threshold work crept in
      (those are deferred — see §3).
- [ ] Rust live example verified; TS/Py/Go parity issue filed (not implemented in
      this change).
