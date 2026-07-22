package sporecore

import (
	"encoding/json"
	"testing"
)

func schemaWithProps(props json.RawMessage) *ToolSchema {
	return &ToolSchema{
		Name:        "t",
		Description: "t",
		InputSchema: json.RawMessage(`{"type":"object","properties":` + string(props) + `}`),
	}
}

func TestCoerceRepairsQuotedKeyAgainstSchema(t *testing.T) {
	s := schemaWithProps(json.RawMessage(`{"content":{"type":"string"}}`))
	input := json.RawMessage(`{"\"content\"":"Adding slugify to util.rs."}`)
	out, ok := CoerceToolArgs(input, s)
	if !ok {
		t.Fatal("expected repair")
	}
	var got map[string]any
	_ = json.Unmarshal(out, &got)
	if got["content"] != "Adding slugify to util.rs." {
		t.Fatalf("out = %s", out)
	}
	if _, present := got[`"content"`]; present {
		t.Fatal("quoted key should be gone")
	}
}

func TestCoerceRepairsQuotedKeyWithNoSchema(t *testing.T) {
	input := json.RawMessage(`{"'path'":"src/util.rs"}`)
	out, ok := CoerceToolArgs(input, nil)
	if !ok {
		t.Fatal("expected repair")
	}
	var got map[string]any
	_ = json.Unmarshal(out, &got)
	if got["path"] != "src/util.rs" {
		t.Fatalf("out = %s", out)
	}
}

func TestCoerceRepairsQuotedKeyAndCoercesValueTogether(t *testing.T) {
	s := schemaWithProps(json.RawMessage(`{"tasks":{"type":"array"}}`))
	input := json.RawMessage(`{"\"tasks\"":"[\"a\",\"b\"]"}`)
	out, ok := CoerceToolArgs(input, s)
	if !ok {
		t.Fatal("expected repair")
	}
	var got map[string]any
	_ = json.Unmarshal(out, &got)
	arr, _ := got["tasks"].([]any)
	if len(arr) != 2 || arr[0] != "a" || arr[1] != "b" {
		t.Fatalf("out = %s", out)
	}
}

func TestCoerceDoesNotInventKeyMissingSchema(t *testing.T) {
	s := schemaWithProps(json.RawMessage(`{"content":{"type":"string"}}`))
	input := json.RawMessage(`{"\"bogus\"":"x"}`)
	if _, ok := CoerceToolArgs(input, s); ok {
		t.Fatal("should not repair a key whose stripped form isn't a property")
	}
}

func TestCoerceFalseStringToBool(t *testing.T) {
	s := schemaWithProps(json.RawMessage(`{"append":{"type":"boolean"}}`))
	out, ok := CoerceToolArgs(json.RawMessage(`{"append":"false"}`), s)
	if !ok {
		t.Fatal("expected repair")
	}
	if string(out) != `{"append":false}` {
		t.Fatalf("out = %s", out)
	}
}

func TestCoerceTrueStringToBoolCaseInsensitive(t *testing.T) {
	s := schemaWithProps(json.RawMessage(`{"append":{"type":"boolean"}}`))
	out, ok := CoerceToolArgs(json.RawMessage(`{"append":"TRUE"}`), s)
	if !ok {
		t.Fatal("expected repair")
	}
	if string(out) != `{"append":true}` {
		t.Fatalf("out = %s", out)
	}
}

func TestCoerceNumericStringToInteger(t *testing.T) {
	s := schemaWithProps(json.RawMessage(`{"n":{"type":"integer"}}`))
	out, ok := CoerceToolArgs(json.RawMessage(`{"n":"42"}`), s)
	if !ok {
		t.Fatal("expected repair")
	}
	if string(out) != `{"n":42}` {
		t.Fatalf("out = %s", out)
	}
}

func TestCoerceFloatStringToNumber(t *testing.T) {
	s := schemaWithProps(json.RawMessage(`{"x":{"type":"number"}}`))
	out, ok := CoerceToolArgs(json.RawMessage(`{"x":"2.5"}`), s)
	if !ok {
		t.Fatal("expected repair")
	}
	if string(out) != `{"x":2.5}` {
		t.Fatalf("out = %s", out)
	}
}

func TestCoerceJSONArrayStringToArray(t *testing.T) {
	s := schemaWithProps(json.RawMessage(`{"args":{"type":"array"}}`))
	out, ok := CoerceToolArgs(json.RawMessage(`{"args":"[\"a\",\"b\"]"}`), s)
	if !ok {
		t.Fatal("expected repair")
	}
	var got map[string]any
	_ = json.Unmarshal(out, &got)
	arr, _ := got["args"].([]any)
	if len(arr) != 2 {
		t.Fatalf("out = %s", out)
	}
}

func TestCoerceAlreadyValidBoolLeftUnchanged(t *testing.T) {
	s := schemaWithProps(json.RawMessage(`{"append":{"type":"boolean"}}`))
	if _, ok := CoerceToolArgs(json.RawMessage(`{"append":false}`), s); ok {
		t.Fatal("already-valid value should not be touched")
	}
}

func TestCoerceNonCoercibleStringLeftAlone(t *testing.T) {
	s := schemaWithProps(json.RawMessage(`{"path":{"type":"string"}}`))
	if _, ok := CoerceToolArgs(json.RawMessage(`{"path":"/etc/hosts"}`), s); ok {
		t.Fatal("a non-coercible string should not be touched")
	}
}

func TestCoerceStringFieldDoesNotConsumeBoolTokenWhenSchemaSaysString(t *testing.T) {
	s := schemaWithProps(json.RawMessage(`{"q":{"type":"string"}}`))
	if _, ok := CoerceToolArgs(json.RawMessage(`{"q":"false"}`), s); ok {
		t.Fatal(`"false" is a legitimate string when schema says string`)
	}
}

func TestCoerceHeuristicWithoutSchema(t *testing.T) {
	out, ok := CoerceToolArgs(json.RawMessage(`{"a":"true","b":"7","c":"hello"}`), nil)
	if !ok {
		t.Fatal("expected repair")
	}
	var got map[string]any
	_ = json.Unmarshal(out, &got)
	if got["a"] != true || got["b"] != float64(7) || got["c"] != "hello" {
		t.Fatalf("out = %s", out)
	}
}

func TestCoerceNonObjectInputReturnsFalse(t *testing.T) {
	if _, ok := CoerceToolArgs(json.RawMessage(`"scalar"`), nil); ok {
		t.Fatal("scalar input should not be repaired")
	}
	if _, ok := CoerceToolArgs(json.RawMessage(`["a"]`), nil); ok {
		t.Fatal("array input should not be repaired")
	}
}

func TestStandardRepairProducesRepairedCall(t *testing.T) {
	repair := StandardToolCallRepair{}
	call := ToolCall{ID: "c1", Name: "write", Input: json.RawMessage(`{"append":"false"}`)}
	s := schemaWithProps(json.RawMessage(`{"append":{"type":"boolean"}}`))
	repaired, ok := repair.Repair(call, "invalid parameters", s)
	if !ok {
		t.Fatal("expected repair")
	}
	if repaired.ID != "c1" || repaired.Name != "write" {
		t.Fatalf("repaired = %+v", repaired)
	}
	if string(repaired.Input) != `{"append":false}` {
		t.Fatalf("repaired input = %s", repaired.Input)
	}
}

func TestStandardRepairGivesUpWhenNothingToFix(t *testing.T) {
	repair := StandardToolCallRepair{}
	call := ToolCall{ID: "c1", Name: "write", Input: json.RawMessage(`{"append":false}`)}
	s := schemaWithProps(json.RawMessage(`{"append":{"type":"boolean"}}`))
	if _, ok := repair.Repair(call, "boom", s); ok {
		t.Fatal("should give up when nothing to fix")
	}
}

func TestIsParamErrorClassification(t *testing.T) {
	for _, m := range []string{
		"invalid parameters: bad input",
		"invalid type foo",
		"missing required field: path",
		"missing field bar",
		"schema validation failed",
		"unknown field x",
		"unknown variant y",
	} {
		if !IsParamError(m) {
			t.Errorf("IsParamError(%q) = false, want true", m)
		}
	}
	for _, m := range []string{
		"command exited with code 1",
		"file not found",
		"connection refused",
	} {
		if IsParamError(m) {
			t.Errorf("IsParamError(%q) = true, want false", m)
		}
	}
}

func TestEffectiveMaxRepairAttemptsDefault(t *testing.T) {
	c := HarnessConfig{}
	if got := c.EffectiveMaxRepairAttempts(); got != 1 {
		t.Fatalf("default = %d, want 1", got)
	}
	zero := uint32(0)
	c.MaxRepairAttempts = &zero
	if got := c.EffectiveMaxRepairAttempts(); got != 0 {
		t.Fatalf("explicit 0 = %d, want 0", got)
	}
	two := uint32(2)
	c.MaxRepairAttempts = &two
	if got := c.EffectiveMaxRepairAttempts(); got != 2 {
		t.Fatalf("explicit 2 = %d, want 2", got)
	}
}
