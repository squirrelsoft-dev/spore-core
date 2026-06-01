//go:build dangerous

// Dangerous-feature surface for package sporecore (issue #34).
//
// This file is compiled ONLY when the `dangerous` build tag is set:
//
//	go build -tags dangerous ./...
//	go test  -tags dangerous ./...
//
// It defines the two named safety footguns that must not be reachable in a
// default build:
//
//   - ModeYolo            — full autonomy, no approval gates (SwitchMode target)
//   - IsolationNone       — no sandbox path enforcement
//
// In a default build this file is excluded, so neither identifier exists and
// naming either one is a compile error rather than a runtime warning. Wire tags
// stay "yolo" and "none" so values round-trip byte-for-byte with the other
// language implementations under their own dangerous gates. Intended for
// benchmarking and local development only — do not enable in production.
package sporecore

import "log"

// ModeYolo — full autonomy, no approval gates. Gated behind the `dangerous`
// build tag (issue #34); absent from the default build. Wire tag "yolo".
const ModeYolo Mode = "yolo"

// IsolationNone — no path enforcement. Gated behind the `dangerous` build tag
// (issue #34); absent from the default build. The safe-by-default isolation
// mode is IsolationWorkspaceScoped.
type IsolationNone struct{}

func (IsolationNone) sealedIsolationMode() {}
func (IsolationNone) Kind() string         { return "none" }

func init() {
	warnIfDangerousIsolation = func(mode IsolationMode) {
		if _, isNone := mode.(IsolationNone); isNone {
			log.Printf("spore-core: WorkspaceScopedSandbox constructed with IsolationNone — " +
				"trusted-dev use only; do not enable silently in production")
		}
	}
}
