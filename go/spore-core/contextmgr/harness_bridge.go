// Harness compaction-loop bridge (issue #46).
//
// The root sporecore package defines the compaction-loop seam in root-package
// terms (sporecore.CompactionVerifier, sporecore.CompactionTurn) so its loop
// needs no import of this package (the import edge runs contextmgr ->
// sporecore; the reverse is a cycle). This file supplies the adapters that
// project this package's rich compaction/verification types onto that seam.
//
// keyTermVerifierAdapter wraps the standard KeyTermVerifier as a
// sporecore.CompactionVerifier: it type-asserts the opaque bridge payloads
// (CompactionTurn.PreserveHints / VerificationState) back to this package's
// concrete types, exactly mirroring how the observability HarnessObserver
// adapter bridges spans without an import cycle.

package contextmgr

import (
	sporecore "github.com/squirrelsoft-dev/spore-core/go/spore-core"
)

// keyTermVerifierAdapter adapts KeyTermVerifier to sporecore.CompactionVerifier.
type keyTermVerifierAdapter struct {
	inner KeyTermVerifier
}

// NewKeyTermVerifier returns the standard post-compaction verifier wired for
// the harness loop seam (issue #46). It is the builder default. The adapter
// type-asserts the bridge's opaque PreserveHints (CompactionPreserveHints) and
// VerificationState (*SessionState / SessionState); a payload of an unexpected
// type degrades to an empty hint set / nil state, so verification never panics
// — it simply finds no required terms and passes.
func NewKeyTermVerifier() sporecore.CompactionVerifier {
	return keyTermVerifierAdapter{inner: KeyTermVerifier{}}
}

// Verify implements sporecore.CompactionVerifier.
func (a keyTermVerifierAdapter) Verify(summary string, turn *sporecore.CompactionTurn) sporecore.CompactionVerificationResult {
	var hints CompactionPreserveHints
	if h, ok := turn.PreserveHints.(CompactionPreserveHints); ok {
		hints = h
	}

	var state *SessionState
	switch s := turn.VerificationState.(type) {
	case *SessionState:
		state = s
	case SessionState:
		state = &s
	}

	res := a.inner.Verify(summary, hints, state)
	missing := res.MissingItems
	if missing == nil {
		missing = []string{}
	}
	return sporecore.CompactionVerificationResult{
		Passed:       res.Passed,
		MissingItems: missing,
	}
}
