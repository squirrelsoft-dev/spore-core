package termination

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	sporecore "github.com/squirrelsoft-dev/spore-core/go/spore-core"
)

// sqlFixtureExpected mirrors fixtures/completion_check/sql_result.json:
// {"kind":"complete"} or {"kind":"incomplete","contains":"..."}.
type sqlFixtureExpected struct {
	Kind     string `json:"kind"`
	Contains string `json:"contains"`
}

type sqlFixtureCase struct {
	Name            string              `json:"name"`
	SQLToolName     string              `json:"sql_tool_name"`
	ExpectedColumns []string            `json:"expected_columns"`
	MinRows         *int                `json:"min_rows"`
	Messages        []sporecore.Message `json:"messages"`
	Expected        sqlFixtureExpected  `json:"expected"`
}

type sqlFixtureSuite struct {
	Cases []sqlFixtureCase `json:"cases"`
}

// TestSqlResultCheckFixtureReplay loads the shared
// fixtures/completion_check/sql_result.json suite and asserts the Go
// SqlResultCheck produces the same Complete / Incomplete(contains=...)
// classification as the Rust reference.
func TestSqlResultCheckFixtureReplay(t *testing.T) {
	wd, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	// go/spore-core/termination → ../../../fixtures/completion_check/sql_result.json
	path := filepath.Join(wd, "..", "..", "..", "fixtures", "completion_check", "sql_result.json")
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	var suite sqlFixtureSuite
	if err := json.Unmarshal(raw, &suite); err != nil {
		t.Fatalf("parse fixture: %v", err)
	}
	if len(suite.Cases) == 0 {
		t.Fatalf("fixture has no cases")
	}
	for _, c := range suite.Cases {
		c := c
		t.Run(c.Name, func(t *testing.T) {
			state := sporecore.SessionState{Messages: c.Messages}
			snap := NewSessionStateSnapshot(
				sporecore.SessionID("fix"),
				sporecore.TaskID("fix"),
				state,
			)
			check := NewSqlResultCheck().WithToolName(c.SQLToolName)
			if c.ExpectedColumns != nil {
				check = check.WithExpectedColumns(c.ExpectedColumns)
			}
			if c.MinRows != nil {
				check = check.WithMinRows(*c.MinRows)
			}
			reason, complete, err := check.Check(context.Background(), &snap)
			if err != nil {
				t.Fatalf("case %q: unexpected error: %v", c.Name, err)
			}
			switch c.Expected.Kind {
			case "complete":
				if !complete || reason != "" {
					t.Fatalf("case %q: expected complete, got (%q, %v)", c.Name, reason, complete)
				}
			case "incomplete":
				if complete {
					t.Fatalf("case %q: expected incomplete, got complete", c.Name)
				}
				if !strings.Contains(reason, c.Expected.Contains) {
					t.Fatalf("case %q: expected reason to contain %q, got %q", c.Name, c.Expected.Contains, reason)
				}
			default:
				t.Fatalf("case %q: unknown expected kind %q", c.Name, c.Expected.Kind)
			}
		})
	}
}
