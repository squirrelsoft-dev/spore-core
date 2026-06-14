package sporecore

import (
	"encoding/json"
	"testing"
)

// ============================================================================
// Output-schema validator unit tests (issue #139).
//
// Mirrors the Rust reference unit tests (output_schema.rs). Same frozen
// literals, same first-match-wins evaluation order, same determinism rules
// (sorted property iteration, sorted enum rendering, the integer SEMANTIC
// check: 42.0 passes, 42.5 fails). These literals are HASH-LOAD-BEARING — the
// four ports must reproduce these exact bytes.
// ============================================================================

func sch(s string) json.RawMessage { return json.RawMessage(s) }

// ── Feedback + error literals (byte-exact, HASH-LOAD-BEARING) ───────────────

func TestFeedbackMessageExactBytes(t *testing.T) {
	got := feedbackMessage("X.")
	want := "Your previous response did not match the required output schema. X. Reply with only a JSON value that satisfies the schema."
	if got != want {
		t.Fatalf("feedback:\n got %q\nwant %q", got, want)
	}
}

func TestErrorLiteralsExactBytes(t *testing.T) {
	if errNotJSON != "The response was not valid JSON." {
		t.Fatalf("errNotJSON = %q", errNotJSON)
	}
	if got, want := errRootType("object", "array"), `Expected type "object" but found "array".`; got != want {
		t.Fatalf("errRootType = %q, want %q", got, want)
	}
	if got, want := errMissingRequired("name"), `Missing required property "name".`; got != want {
		t.Fatalf("errMissingRequired = %q, want %q", got, want)
	}
	if got, want := errPropertyType("age", "integer", "string"), `Property "age" should be type "integer" but found "string".`; got != want {
		t.Fatalf("errPropertyType = %q, want %q", got, want)
	}
	if got, want := errPropertyEnum("status", `["error","ok"]`, `"maybe"`), `Property "status" must be one of ["error","ok"] but found "maybe".`; got != want {
		t.Fatalf("errPropertyEnum = %q, want %q", got, want)
	}
}

// ── Step 1: parse ───────────────────────────────────────────────────────────

func TestStep1NotJSON(t *testing.T) {
	if got := validateOutput("not json at all", sch(`{"type":"object"}`)); got != errNotJSON {
		t.Fatalf("got %q, want errNotJSON", got)
	}
}

// ── Step 2: root type ───────────────────────────────────────────────────────

func TestStep2RootTypeMismatch(t *testing.T) {
	if got, want := validateOutput("[1,2,3]", sch(`{"type":"object"}`)), errRootType("object", "array"); got != want {
		t.Fatalf("got %q, want %q", got, want)
	}
}

func TestStep2RootTypeMatch(t *testing.T) {
	if got := validateOutput("[1,2,3]", sch(`{"type":"array"}`)); got != "" {
		t.Fatalf("got %q, want valid", got)
	}
}

// ── Step 3: required (array order) ──────────────────────────────────────────

func TestStep3MissingRequiredFirstInArrayOrder(t *testing.T) {
	// required order is [b, a]; both absent → the FIRST in array order (b) is
	// reported, NOT the lexicographically-first (a).
	if got, want := validateOutput("{}", sch(`{"type":"object","required":["b","a"]}`)), errMissingRequired("b"); got != want {
		t.Fatalf("got %q, want %q", got, want)
	}
}

func TestStep3RequiredPresent(t *testing.T) {
	if got := validateOutput(`{"a":1}`, sch(`{"type":"object","required":["a"]}`)); got != "" {
		t.Fatalf("got %q, want valid", got)
	}
}

// ── Step 4: present-property type (sorted key order) ────────────────────────

func TestStep4PropertyTypeMismatchSortedOrder(t *testing.T) {
	// Both age and zip are wrong (number schema, string value). Sorted key order
	// ⇒ age reported first (NOT JSON insertion order, which Go map randomizes).
	schema := sch(`{"type":"object","properties":{"zip":{"type":"number"},"age":{"type":"number"}}}`)
	if got, want := validateOutput(`{"age":"x","zip":"y"}`, schema), errPropertyType("age", "number", "string"); got != want {
		t.Fatalf("got %q, want %q", got, want)
	}
}

func TestStep4IntegerAcceptsWholeNumber42_0(t *testing.T) {
	// 42.0 has no fractional part → passes integer. 42 (no decimal) also passes.
	schema := sch(`{"type":"object","properties":{"n":{"type":"integer"}}}`)
	if got := validateOutput(`{"n":42.0}`, schema); got != "" {
		t.Fatalf("42.0 should pass integer, got %q", got)
	}
	if got := validateOutput(`{"n":42}`, schema); got != "" {
		t.Fatalf("42 should pass integer, got %q", got)
	}
}

func TestStep4IntegerRejectsFractional42_5(t *testing.T) {
	// 42.5 has a fractional part → fails integer (reported actual type is number).
	schema := sch(`{"type":"object","properties":{"n":{"type":"integer"}}}`)
	if got, want := validateOutput(`{"n":42.5}`, schema), errPropertyType("n", "integer", "number"); got != want {
		t.Fatalf("got %q, want %q", got, want)
	}
}

func TestStep4NumberAcceptsFractional(t *testing.T) {
	schema := sch(`{"type":"object","properties":{"n":{"type":"number"}}}`)
	if got := validateOutput(`{"n":42.5}`, schema); got != "" {
		t.Fatalf("42.5 should pass number, got %q", got)
	}
}

// ── Step 5: present-property enum (sorted enum rendering) ───────────────────

func TestStep5EnumViolationRendersSortedEnum(t *testing.T) {
	// Author order is ["ok","error"]; the message renders SORTED (["error","ok"])
	// for determinism. Membership itself is order-free.
	schema := sch(`{"type":"object","properties":{"status":{"type":"string","enum":["ok","error"]}}}`)
	want := errPropertyEnum("status", `["error","ok"]`, `"maybe"`)
	if got := validateOutput(`{"status":"maybe"}`, schema); got != want {
		t.Fatalf("got %q, want %q", got, want)
	}
}

func TestStep5EnumMemberPassesRegardlessOfOrder(t *testing.T) {
	schema := sch(`{"type":"object","properties":{"status":{"type":"string","enum":["ok","error"]}}}`)
	if got := validateOutput(`{"status":"ok"}`, schema); got != "" {
		t.Fatalf(`"ok" should pass, got %q`, got)
	}
	if got := validateOutput(`{"status":"error"}`, schema); got != "" {
		t.Fatalf(`"error" should pass, got %q`, got)
	}
}

// ── Step 6: valid ───────────────────────────────────────────────────────────

func TestStep6FullObjectValid(t *testing.T) {
	schema := sch(`{"type":"object","required":["status","count"],"properties":{"status":{"type":"string","enum":["ok","error"]},"count":{"type":"integer"}}}`)
	if got := validateOutput(`{"status":"ok","count":3}`, schema); got != "" {
		t.Fatalf("got %q, want valid", got)
	}
}

// ── Evaluation order: earlier rule wins ─────────────────────────────────────

func TestOrderRootTypeBeatsRequired(t *testing.T) {
	// A non-object value with a required list: step 2 (root type) fires before
	// step 3 (required).
	schema := sch(`{"type":"object","required":["a"]}`)
	if got, want := validateOutput(`"a string"`, schema), errRootType("object", "string"); got != want {
		t.Fatalf("got %q, want %q", got, want)
	}
}

func TestOrderRequiredBeatsPropertyType(t *testing.T) {
	// a absent (required) AND b present with wrong type: step 3 wins.
	schema := sch(`{"type":"object","required":["a"],"properties":{"b":{"type":"number"}}}`)
	if got, want := validateOutput(`{"b":"x"}`, schema), errMissingRequired("a"); got != want {
		t.Fatalf("got %q, want %q", got, want)
	}
}

func TestOrderPropertyTypeBeatsEnum(t *testing.T) {
	// s present with wrong type AND would also violate enum: step 4 wins.
	schema := sch(`{"type":"object","properties":{"s":{"type":"string","enum":["ok"]}}}`)
	if got, want := validateOutput(`{"s":123}`, schema), errPropertyType("s", "string", "number"); got != want {
		t.Fatalf("got %q, want %q", got, want)
	}
}

// ── canonicalizeSchema: key-sorted compact JSON (delivery + reporting) ──────

func TestCanonicalizeSchemaKeySorted(t *testing.T) {
	// Author order differs; canonical form is key-sorted compact JSON — the exact
	// bytes the directive seed embeds and the violation reports, pinned so the
	// four ports match.
	in := sch(`{"required":["status","count"],"type":"object","properties":{"status":{"enum":["ok","error"],"type":"string"},"count":{"type":"integer"}}}`)
	want := `{"properties":{"count":{"type":"integer"},"status":{"enum":["ok","error"],"type":"string"}},"required":["status","count"],"type":"object"}`
	if got := canonicalizeSchema(in); got != want {
		t.Fatalf("canonicalizeSchema:\n got %s\nwant %s", got, want)
	}
}
