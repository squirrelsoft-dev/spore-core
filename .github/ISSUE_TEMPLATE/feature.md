---
name: Feature / Component implementation
about: Track implementation of a harness component across all four languages
title: "[component-name]: <short description>"
labels: feature
---

## Spec issue(s)

<!--
Link to the canonical component issue (one of #1–#13 or follow-ons like
#23–#26) and the section of /docs/harness-engineering-concepts.md that
this feature implements.
-->

- Spec issue: #
- Spec section: `docs/harness-engineering-concepts.md` —

## Summary

<!-- One paragraph: what is being built and why. Link the spec, do not
restate it. -->

## Acceptance criteria

<!-- Concrete, testable. The reviewer must be able to verify each. -->

- [ ]
- [ ]
- [ ]

## Shared fixtures

<!--
Which fixtures from /fixtures/ does this feature consume or extend?
If new fixtures are required, list them here. Every new fixture must
ship with replay tests in ALL FOUR languages — see /fixtures/README.md.
-->

- Existing fixtures used:
- New fixtures added:

## Implementation checklist

Track the four parallel implementations. All four boxes must be checked
before this issue closes — public API and behaviour stay aligned across
languages.

- [ ] **Rust** — `/rust/` (see `rust/CONVENTIONS.md`)
- [ ] **TypeScript** — `/typescript/` (see `typescript/CONVENTIONS.md`)
- [ ] **Python** — `/python/` (see `python/CONVENTIONS.md`)
- [ ] **Go** — `/go/` (see `go/CONVENTIONS.md`)

## Consistency notes

<!--
Call out anywhere the four implementations are NOT expected to be
byte-identical — usually idiomatic differences (e.g. Go's `error` return
vs Rust's `Result`, Python's snake_case fields). Everything else MUST
match: public type shapes, field names on the wire, fixture outcomes,
error categorisation (always-halt vs recoverable).
-->

## Testing

<!--
How to verify the four implementations agree:

1. Same fixture(s) from /fixtures/ load in all four languages.
2. Replay tests assert the same outcome in each language.
3. CI matrix (.github/workflows/{rust,typescript,python,go}.yml) is green.

If a fixture fails in one language but passes in the others, fix the
implementation — never the fixture.
-->

- [ ] Unit tests added in all four languages
- [ ] Shared-fixture replay tests added in all four languages
- [ ] CI matrix green for all four languages
- [ ] Public type shapes match across languages (field names, variants)
