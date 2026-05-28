// Command spore-eval is the EvalHarness CLI, mirroring the spore-e2e-agent
// pattern.
//
// Subcommands:
//
//	run <suite.json> [--candidate ID ...] [--n N] [--json]
//	    Load a suite and print/serialize its shape. Candidate harness configs
//	    are wired programmatically via EvalHarness in the MVP; this command
//	    validates the suite and reports its shape.
//	promote <suite.json> <task_id>
//	    Manually promote a challenge task to regression, bump suite_version,
//	    rewrite the JSON in place (Rule 31). Auto-promotion is deferred.
package main

import (
	"encoding/json"
	"fmt"
	"os"

	"github.com/squirrelsoft-dev/spore-core/go/spore-eval/manifest"
)

func main() {
	args := os.Args[1:]
	if len(args) == 0 {
		printUsage()
		fmt.Fprintln(os.Stderr, "error: no subcommand given")
		os.Exit(2)
	}
	var err error
	switch args[0] {
	case "run":
		err = cmdRun(args[1:])
	case "promote":
		err = cmdPromote(args[1:])
	case "-h", "--help", "help":
		printUsage()
		return
	default:
		printUsage()
		err = fmt.Errorf("unknown subcommand: %s", args[0])
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
}

func printUsage() {
	fmt.Fprint(os.Stderr,
		"spore-eval — EvalHarness CLI\n\n"+
			"USAGE:\n"+
			"  spore-eval run <suite.json> [--candidate ID ...] [--n N] [--json]\n"+
			"  spore-eval promote <suite.json> <task_id>\n")
}

func cmdRun(args []string) error {
	var suitePath string
	var candidates []string
	n := uint32(3)
	jsonOut := false
	for i := 0; i < len(args); i++ {
		switch args[i] {
		case "--candidate":
			i++
			if i >= len(args) {
				return fmt.Errorf("--candidate needs an ID")
			}
			candidates = append(candidates, args[i])
		case "--n":
			i++
			if i >= len(args) {
				return fmt.Errorf("--n needs a number")
			}
			var v uint32
			if _, e := fmt.Sscanf(args[i], "%d", &v); e != nil {
				return fmt.Errorf("--n must be a u32: %w", e)
			}
			n = v
		case "--json":
			jsonOut = true
		default:
			if len(args[i]) >= 2 && args[i][:2] == "--" {
				return fmt.Errorf("unexpected argument: %s", args[i])
			}
			if suitePath == "" {
				suitePath = args[i]
			} else {
				return fmt.Errorf("unexpected argument: %s", args[i])
			}
		}
	}
	if suitePath == "" {
		return fmt.Errorf("missing <suite.json>")
	}
	suite, err := manifest.LoadSuitePath(suitePath)
	if err != nil {
		return fmt.Errorf("loading suite %s: %w", suitePath, err)
	}
	if candidates == nil {
		candidates = []string{}
	}
	if jsonOut {
		summary := map[string]any{
			"suite_version":     suite.SuiteVersion,
			"regression":        len(suite.Regression),
			"challenge":         len(suite.Challenge),
			"canary":            len(suite.Canary),
			"n_runs_per_config": n,
			"candidates":        candidates,
			"note":              "candidate harness configs are wired programmatically; this CLI validates the suite and reports its shape",
		}
		b, _ := json.MarshalIndent(summary, "", "  ")
		fmt.Println(string(b))
		return nil
	}
	fmt.Printf("loaded suite v%d — regression=%d, challenge=%d, canary=%d (n=%d)\n",
		suite.SuiteVersion, len(suite.Regression), len(suite.Challenge), len(suite.Canary), n)
	return nil
}

func cmdPromote(args []string) error {
	if len(args) < 1 {
		return fmt.Errorf("missing <suite.json>")
	}
	if len(args) < 2 {
		return fmt.Errorf("missing <task_id>")
	}
	suitePath, taskID := args[0], args[1]
	suite, err := manifest.LoadSuitePath(suitePath)
	if err != nil {
		return fmt.Errorf("loading suite %s: %w", suitePath, err)
	}
	before := suite.SuiteVersion
	if err := manifest.PromoteChallengeTask(suite, taskID); err != nil {
		return fmt.Errorf("promoting %s: %w", taskID, err)
	}
	out, err := manifest.SuiteToJSON(suite)
	if err != nil {
		return err
	}
	if err := os.WriteFile(suitePath, []byte(out), 0o644); err != nil {
		return fmt.Errorf("writing %s: %w", suitePath, err)
	}
	fmt.Printf("promoted %s: suite_version %d -> %d (challenge -> regression)\n", taskID, before, suite.SuiteVersion)
	return nil
}
