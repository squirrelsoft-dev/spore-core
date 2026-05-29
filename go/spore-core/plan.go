// Plan phase / plan artifact — PlanExecute, phase 1 of 2 (issue #70).
//
// This file owns the *capture* half of the PlanExecute plan phase: turning a
// planner model's FinalResponse text into a structured PlanArtifact. The
// *phase driver* itself (runPlanPhase / runPlanExecute) lives on
// StandardHarness in harness.go because it needs the harness's turn machinery;
// this file supplies the deterministic, total text→artifact step and the phase
// error type.
//
// Public surface:
//   - PlanArtifact — defined in hooks.go; the existing, serializable contract
//     ({ tasks []string, rationale string }) that is the payload of the
//     OnPlanCreated hook (issue #69). This issue REUSES it rather than defining
//     a competing type. It is the contract consumed by #72 / #59.
//   - PlanPhaseError — error type for the plan phase.
//   - CapturePlanArtifact — the model-text → PlanArtifact capture function.
//     Deterministic and total: never panics; malformed input yields a
//     PlanPhaseError with kind PlanErrorUnparseablePlan.
//
// Rules enforced (the phase driver in harness.go enforces R1–R8, R10–R11):
//   - R9  CapturePlanArtifact is deterministic and total.
//
// Resolved spec decisions (issue #70 — all four FINAL, mirror the Rust
// reference byte-for-byte):
//   - Q1 (model routing): HarnessConfig.PlannerAgent plus its builder setter.
//     When the strategy is PlanExecute and PlannerAgent is set, the plan turn
//     runs on it; otherwise it runs on the default Agent. PlanModel stays as
//     DESCRIPTIVE metadata only — there is no ModelConfig→agent factory.
//   - Q2 (HITL): the plan phase ALWAYS runs to completion. It fires
//     OnPlanCreated synchronously (the hook may rewrite the artifact via the
//     *PlanArtifact pointer); the stored artifact reflects any mutation. No
//     pause, no WaitingForHuman path.
//   - Q3 (capture grammar): JSON-in-response. Trim ASCII whitespace; strip a
//     single leading ```/```json fence line and a single trailing ``` fence if
//     present; parse a JSON object with `tasks` (required array of strings,
//     kept verbatim, may be empty) and `rationale` (optional string, default
//     ""). Any failure → PlanPhaseError{Kind: PlanErrorUnparseablePlan}.
//   - Q4 (terminal RunResult): after producing, firing OnPlanCreated, and
//     storing the artifact, the PlanExecute arm HALTS with the distinct
//     HaltExecutePhaseNotImplemented reason — separate from the generic
//     HaltStrategyNotYetImplemented the other strategies still use.

package sporecore

import (
	"encoding/json"
	"fmt"
)

// PlanExecuteExtrasKey is the key under which the produced PlanArtifact is
// stored in SessionState.Extras (serialized JSON). Stable across all four
// languages.
const PlanExecuteExtrasKey = "plan_execute"

// PlanPhaseErrorKind discriminates PlanPhaseError variants. Tag values match
// the Rust enum (#[serde(tag = "kind", rename_all = "snake_case")]).
type PlanPhaseErrorKind string

const (
	// PlanErrorUnparseablePlan: the planner's response text could not be parsed
	// into a PlanArtifact under the Q3 grammar (not valid JSON, not a JSON
	// object, or `tasks` absent / not an array / containing a non-string
	// element).
	PlanErrorUnparseablePlan PlanPhaseErrorKind = "unparseable_plan"
	// PlanErrorPlanningTurnFailed: the plan turn errored or did not produce a
	// FinalResponse (e.g. the planner requested a tool call — R2).
	PlanErrorPlanningTurnFailed PlanPhaseErrorKind = "planning_turn_failed"
)

// PlanPhaseError is the error type raised by the plan phase. It carries a kind
// tag plus a human-readable message, matching the Rust tagged enum on the
// wire: {"kind":"unparseable_plan","message":"..."}.
type PlanPhaseError struct {
	Kind    PlanPhaseErrorKind `json:"kind"`
	Message string             `json:"message"`
}

// Error implements the error interface.
func (e *PlanPhaseError) Error() string {
	switch e.Kind {
	case PlanErrorUnparseablePlan:
		return fmt.Sprintf("unparseable plan: %s", e.Message)
	case PlanErrorPlanningTurnFailed:
		return fmt.Sprintf("planning turn failed: %s", e.Message)
	default:
		return fmt.Sprintf("plan phase error (%s): %s", e.Kind, e.Message)
	}
}

// newUnparseablePlan builds an UnparseablePlan error.
func newUnparseablePlan(message string) *PlanPhaseError {
	return &PlanPhaseError{Kind: PlanErrorUnparseablePlan, Message: message}
}

// CapturePlanArtifact captures a PlanArtifact from a planner's FinalResponse
// text.
//
// This is the canonical Q3 grammar — it MUST be byte-identical across all four
// languages, so it is kept simple and total:
//
//  1. Trim leading/trailing ASCII whitespace.
//  2. If the trimmed text begins with a triple-backtick fence, strip a single
//     leading fence line (the opening ``` plus any language tag up to and
//     including the first newline) and a single trailing ``` fence, then trim
//     again.
//  3. Parse the result as a JSON object with `tasks` (required array of JSON
//     strings, kept verbatim; an empty array is allowed) and `rationale`
//     (optional string, default "").
//
// Any deviation → *PlanPhaseError{Kind: PlanErrorUnparseablePlan}. Never panics.
func CapturePlanArtifact(finalText string) (PlanArtifact, error) {
	trimmed := trimASCIIWS(finalText)
	body := stripCodeFence(trimmed)

	// Decode into a generic value so we can enforce the object/array/string
	// shape exactly (and reject non-string task elements verbatim).
	var value any
	if err := json.Unmarshal([]byte(body), &value); err != nil {
		return PlanArtifact{}, newUnparseablePlan(fmt.Sprintf("invalid JSON: %s", err))
	}

	obj, ok := value.(map[string]any)
	if !ok {
		return PlanArtifact{}, newUnparseablePlan("top-level JSON value is not an object")
	}

	tasksValue, ok := obj["tasks"]
	if !ok {
		return PlanArtifact{}, newUnparseablePlan("missing required field `tasks`")
	}
	tasksArray, ok := tasksValue.([]any)
	if !ok {
		return PlanArtifact{}, newUnparseablePlan("field `tasks` is not an array")
	}

	tasks := make([]string, 0, len(tasksArray))
	for i, element := range tasksArray {
		s, ok := element.(string)
		if !ok {
			return PlanArtifact{}, newUnparseablePlan(fmt.Sprintf("element %d of `tasks` is not a string", i))
		}
		// Verbatim — do NOT trim or filter.
		tasks = append(tasks, s)
	}

	// `rationale` is optional; default "". If present it must be a string.
	rationale := ""
	if rv, present := obj["rationale"]; present {
		s, ok := rv.(string)
		if !ok {
			return PlanArtifact{}, newUnparseablePlan("field `rationale` is not a string")
		}
		rationale = s
	}

	return PlanArtifact{Tasks: tasks, Rationale: rationale}, nil
}

// isASCIIWS reports whether c is one of the ASCII whitespace bytes the Q3
// grammar trims. Matches ' ', '\t', '\n', '\r', and the form-feed /
// vertical-tab — kept to the ASCII set so trimming is byte-identical
// cross-language.
func isASCIIWS(c byte) bool {
	switch c {
	case ' ', '\t', '\n', '\r', '\v', '\f':
		return true
	default:
		return false
	}
}

// trimASCIIWS trims leading and trailing ASCII whitespace (the isASCIIWS set).
func trimASCIIWS(s string) string {
	start := 0
	for start < len(s) && isASCIIWS(s[start]) {
		start++
	}
	end := len(s)
	for end > start && isASCIIWS(s[end-1]) {
		end--
	}
	return s[start:end]
}

// trimEndASCIIWS trims only trailing ASCII whitespace.
func trimEndASCIIWS(s string) string {
	end := len(s)
	for end > 0 && isASCIIWS(s[end-1]) {
		end--
	}
	return s[:end]
}

// stripCodeFence strips a single leading ```/```json fence line and a single
// trailing ``` fence, if the (already-trimmed) input opens with a triple-
// backtick fence. Returns the inner body, re-trimmed. If the input does not
// open with a fence it is returned unchanged.
func stripCodeFence(trimmed string) string {
	const fence = "```"
	if len(trimmed) < len(fence) || trimmed[:len(fence)] != fence {
		return trimmed
	}
	afterOpen := trimmed[len(fence):]

	// Drop the rest of the opening fence line (the optional language tag) up to
	// and including the first newline. A fence with no newline at all has no
	// body to parse; let JSON parsing reject it downstream.
	bodyStart := afterOpen
	for i := 0; i < len(afterOpen); i++ {
		if afterOpen[i] == '\n' {
			bodyStart = afterOpen[i+1:]
			break
		}
	}

	// Strip a single trailing closing fence if present, then re-trim.
	body := trimEndASCIIWS(bodyStart)
	if len(body) >= len(fence) && body[len(body)-len(fence):] == fence {
		body = body[:len(body)-len(fence)]
	} else {
		body = bodyStart
	}

	return trimASCIIWS(body)
}
