//go:build dangerous

// Dangerous-feature surface for promptchunkregistry (issue #34).
//
// ModeYolo (full autonomy, no approval gates) is a named safety footgun. It is
// compiled ONLY when the `dangerous` build tag is set:
//
//	go build -tags dangerous ./...
//	go test  -tags dangerous ./...
//
// In a default build this file is excluded, so the ModeYolo identifier does not
// exist and any attempt to name it is a compile error rather than a runtime
// warning. The wire tag stays "yolo" so a yolo-tagged value round-trips
// byte-for-byte with the other language implementations under their own
// dangerous gates. Intended for benchmarking and local development only — do
// not enable in production deployments.
package promptchunkregistry

// ModeYolo — full autonomy, no approval gates. Gated behind the `dangerous`
// build tag (issue #34); absent from the default build.
const ModeYolo Mode = "yolo"

func init() {
	dangerousModePromptChunk = func(m Mode) (ChunkID, string, bool) {
		if m == ModeYolo {
			return "mode-yolo", "Mode: Yolo. Full autonomy. No approval gates.", true
		}
		return "", "", false
	}
	dangerousModeApprovalPolicy = func(m Mode) (ApprovalPolicy, bool) {
		if m == ModeYolo {
			return ApprovalPolicyNone, true
		}
		return "", false
	}
	dangerousModeChunks = func() []PromptChunk {
		return []PromptChunk{ModeYolo.PromptChunk()}
	}
}
