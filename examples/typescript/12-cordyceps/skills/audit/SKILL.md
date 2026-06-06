---
name: audit
description: Audit one Rust module for real, actionable defects.
---

# Audit procedure

You audit exactly ONE Rust module. Stay inside it. Be specific. Never read a
whole file — grep first, then read only the narrow line ranges you need.

## Discipline

1. **Grep first.** Search the module for risk patterns before reading anything:
   - `unwrap(`, `expect(`, `panic!`, `unreachable!`, `todo!`, `unimplemented!`
   - `unsafe`
   - `as ` casts that can truncate/overflow (`as u8`, `as i32`, `as usize`)
   - `clone()` in hot/loop paths, needless allocations
   - `.lock().unwrap()` (poison/panic), blocking calls in async fns
   - integer arithmetic that can overflow, slicing/indexing that can panic
   - swallowed errors (`let _ =`, `.ok();`, `if let Err(_) = `)
   - TODO/FIXME/HACK comments admitting known gaps
2. **Read narrow.** For each grep hit that looks real, `read_file` only the
   surrounding lines (e.g. a 10–20 line window). Confirm it is a genuine defect
   in context, not a false positive.
3. **Escalate when unsure** (the consult ladder):
   - `research_best_practices` — when you are unsure whether a pattern is the
     idiomatic Rust way or a real smell. Pass a focused `question`.
   - `consult_advisor` — when you are stuck on whether a finding is real, or how
     to rank its severity/priority. Reserve for genuine uncertainty.
   - If a consult comes back "budget exhausted", proceed on your own judgement.
4. **Keep findings real.** Each finding must name a concrete file, line, and a
   specific, actionable problem. No boilerplate ("consider adding tests"), no
   style nits, no speculation. If the module is clean, return an empty array.

## Output schema (HARD)

Your FINAL answer must be a JSON array (and nothing else) of objects:

```json
[
  {
    "file": "relative/path/to/file.rs",
    "line": 42,
    "severity": "low" | "medium" | "high" | "critical",
    "description": "Specific, actionable description of the real defect."
  }
]
```

- `file`: path relative to the repo root.
- `line`: the 1-based line number of the defect.
- `severity`: one of `low`, `medium`, `high`, `critical`.
- `description`: what is wrong and why it matters — concrete, not generic.

Read-only: you never modify files. One module only. No whole-file reads.
