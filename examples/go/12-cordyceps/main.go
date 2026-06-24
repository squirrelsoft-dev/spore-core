// spore-core example 12 — cordyceps: the capstone of the Composable Execution
// refactor (#117–#131).
//
// # The thesis: you describe a strategy as DATA — a composed LoopStrategy tree —
// wire its string handles to concrete collaborators in an ExecutionRegistry, and
// the harness runs the whole nested machine under one shared budget / usage /
// observability context.
//
// The motivating composition is:
//
//	Ralph[ PlanExecute[ ReAct, SelfVerifying[ ReAct ] ] ]
//	│       │             │      │             │
//	│       │             │      │             └─ worker: audits ONE module
//	│       │             │      └─ Default-FAIL evaluator (single read-only turn)
//	│       │             └─ plan: explores the repo, builds a blocker-aware DAG
//	│       └─ plan→ready-set: walks the DAG in dependency order, self-verifying each task
//	└─ continuation wrapper: resets the window, resumes from durable progress
//
// # What changed vs. the pre-#131 example (HONEST note)
//
// The old depth-1 example used a hand-built SubagentTool orchestrator with a
// per-node consult mediator (#114) and an architect-side load_skill tool (#115).
// The declarative tree has NO SubagentTool seam, so:
//
//   - the #114 consult ladder is PRESERVED, with its mediation seam moved. The
//     worker still calls research_best_practices / consult_advisor, which lower to
//     a Consult ToolOutput. With no SubagentTool to mediate, the worker-leaf
//     consult propagates all the way up to a top-level RunResult.Consult, and the
//     HOST run loop mediates it — routing by `kind` to a helper harness with a
//     per-kind budget + overflow policy (research → web_search, budget 5,
//     SoftFail; advice → cloud advisor, budget 3, EscalateToHuman). Identical
//     #114 semantics, host-owned budgets.
//   - the audit skill is KEPT, now baked into the harness (#115 / SC-26):
//     HarnessBuilder.Skills takes a sporecore.SkillCatalog (discovered from the
//     bundled skills/ dir + .spore/skills), registers the load_skill tool, and
//     injects the manifest + each ACTIVE skill's body STRUCTURALLY via the
//     ContextSources seam — no example-side SkillInjectingContextManager shim.
//     The example pre-activates `audit` at startup so the audit procedure reaches
//     the model structurally every turn, compaction-proof, with no load_skill
//     round-trip required (the model may still call load_skill for others).
//
// # The tree is DATA
//
// We do NOT hand-build the LoopStrategy. We //go:embed the canonical fixture
// fixtures/strategy/cordyceps_tree.json and deserialize it — so this example
// proves the canonical fixture deserializes and runs.
//
// # Run it
//
//	ollama serve &
//	ollama pull gemma4:e4b
//	export SPORE_WEB_SEARCH_ENDPOINT="http://localhost:8888/search?format=json"
//	go run .
package main

import (
	"bufio"
	"context"
	_ "embed"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	sporecore "github.com/squirrelsoft-dev/spore-core/go/spore-core"
	"github.com/squirrelsoft-dev/spore-core/go/spore-core/observability"
	"github.com/squirrelsoft-dev/spore-core/go/spore-core/ollama"
	"github.com/squirrelsoft-dev/spore-core/go/spore-core/storage"
	"github.com/squirrelsoft-dev/spore-core/go/spore-core/tools"
	"github.com/squirrelsoft-dev/spore-core/go/spore-core/verifier"
)

// cordycepsTreeJSON is the canonical composed-strategy fixture, embedded so the
// example proves the ground-truth tree deserializes (and runs) verbatim — never
// hand-built. Go's //go:embed is the idiomatic equivalent of Rust's include_str!.
//
//go:embed cordyceps_tree.json
var cordycepsTreeJSON []byte

// execEvaluatorKey is the verifier registry key the SelfVerifying node's
// evaluator handle resolves to.
const execEvaluatorKey = "exec-evaluator"

// defaultAuditPrompt is the pre-filled audit prompt (press enter to accept).
const defaultAuditPrompt = "Audit this repository for Rust defects. Discover the crates and their " +
	"modules, audit each module for real, actionable defects, and write a markdown report of the " +
	"most important findings to `workspace/findings.md`."

const execSystemPrompt = "You are a cordyceps execution machine. Your strategy is composed " +
	"declaratively: a Ralph continuation wrapper drives a PlanExecute, whose plan phase explores " +
	"the repo and builds a blocker-aware task graph via `task_list`, and whose execute phase walks " +
	"that graph as a ready-set — auditing one module per ready task, each result self-verified by a " +
	"read-only evaluator (Default-FAIL: only an explicit PASS clears a task).\n\n" +
	"You are already scoped to the repository root (READ-ONLY). Use `.` for the root and paths " +
	"relative to it (e.g. `rust/crates`); never prefix a path with the repository's own folder " +
	"name. The audit is read-only — you have no write tool; never attempt to modify source files.\n\n" +
	"Follow the ACTIVE `audit` skill's procedure and output schema exactly: grep first, read " +
	"narrow, and return findings as a JSON array of {file, line, severity, description}.\n\n" +
	"PLAN phase: explore the repo with `list_dir`/`grep`, then build a blocker-aware task graph " +
	"with `task_list` (one task per module; add dependencies where one audit should wait on " +
	"another). RALPH wrapper: resume from durable `task_list` progress after each context-window " +
	"reset and keep going until every task is done."

const researchPrompt = "You are a research worker. A peer agent needs factual, current information " +
	"on a Rust best-practice or language question. Use `web_search` to find the answer. Issue " +
	"focused queries, read the results, and return a concise cited answer in plain text. Do not " +
	"answer from memory alone — always search first."

const advisorPrompt = "You are a senior Rust advisor. A worker has escalated a candidate finding " +
	"to you because they need a judgment call. Use `read_file` and `grep` to examine the specific " +
	"code in question. Then make a decision: is this a real defect, what is the severity " +
	"(low / medium / high / critical), and why. Be decisive. State your verdict in one sentence, " +
	"your reasoning in two. Do not hedge."

// planSchema — the task-graph contract the plan phase's ReAct emits.
func planSchema() json.RawMessage {
	return json.RawMessage(`{
  "type": "object",
  "properties": {
    "tasks": {
      "type": "array",
      "description": "Ordered task-graph entries; each names a module to audit.",
      "items": {
        "type": "object",
        "properties": {
          "module": { "type": "string", "description": "Module path to audit." },
          "blockers": {
            "type": "array",
            "items": { "type": "integer" },
            "description": "1-based ids of tasks this one waits on."
          }
        },
        "required": ["module"]
      }
    },
    "rationale": { "type": "string" }
  },
  "required": ["tasks"]
}`)
}

// workerSchema — the per-module finding contract the worker ReAct emits.
func workerSchema() json.RawMessage {
	return json.RawMessage(`{
  "type": "array",
  "description": "Findings for ONE module.",
  "items": {
    "type": "object",
    "properties": {
      "file": { "type": "string", "description": "Path relative to the repo root." },
      "line": { "type": "integer", "description": "1-based line of the defect." },
      "severity": { "enum": ["low", "medium", "high", "critical"] },
      "description": { "type": "string", "description": "Concrete, actionable defect." }
    },
    "required": ["file", "line", "severity", "description"]
  }
}`)
}

// planTools is the plan-tools catalogue: explore + author the task graph (read-only).
func planTools() []sporecore.StandardTool {
	return []sporecore.StandardTool{
		tools.StandardTools{}.ListDir(),
		tools.StandardTools{}.Grep(),
		tools.StandardTools{}.TaskList(),
	}
}

// execTools is the exec-tools catalogue: read-only audit + the #114 consult
// ladder. The two consult tools lower to a Consult ToolOutput, which the host run
// loop mediates (the seam moved off SubagentTool).
func execTools() []sporecore.StandardTool {
	return []sporecore.StandardTool{
		tools.StandardTools{}.ReadFile(),
		tools.StandardTools{}.Grep(),
		researchBestPracticesTool(),
		consultAdvisorTool(),
	}
}

// buildWebSearch builds the SearXNG-backed web_search catalogue tool (identical
// to 06/11).
func buildWebSearch(endpoint string) (sporecore.StandardTool, error) {
	tool, err := tools.NewWebSearchToolFromConfig(tools.WebSearchConfig{
		Endpoint:       endpoint,
		Method:         tools.SearchMethodGet,
		QueryParam:     "q",
		AuthHeaders:    []tools.AuthHeader{},
		BodyAuthParams: []tools.BodyAuthParam{},
	})
	if err != nil {
		return sporecore.StandardTool{}, fmt.Errorf("failed to construct web_search tool: %w", err)
	}
	return tools.StandardTool{Implementation: tool, Schema: tool.Schema()}, nil
}

// buildResearchHarness builds the research handler harness (web_search only) — the
// kind=research consult handler. Run host-side on a ConsultRequest.
func buildResearchHarness(model, baseURL, endpoint string) (sporecore.Harness, error) {
	webSearch, err := buildWebSearch(endpoint)
	if err != nil {
		return nil, err
	}
	mi := ollama.WithBaseURL(model, baseURL)
	return observability.ConversationalBuilder(mi).
		Tool(webSearch).
		SystemPrompt(researchPrompt).
		Build(), nil
}

// buildAdvisorHarness builds the advisor handler harness (cloud model, read_file +
// grep) — the kind=advice consult handler. Rides the same Ollama endpoint via
// WithBaseURL; only the model id differs (heterogeneous models).
func buildAdvisorHarness(model, baseURL string, repoSandbox sporecore.SandboxProvider) sporecore.Harness {
	mi := ollama.WithBaseURL(model, baseURL)
	return observability.ConversationalBuilder(mi).
		Sandbox(repoSandbox).
		Tool(tools.StandardTools{}.ReadFile()).
		Tool(tools.StandardTools{}.Grep()).
		SystemPrompt(advisorPrompt).
		Build()
}

// buildConsultHandlers builds the HOST-owned kind → {handler, budget, overflow}
// map (#114). The composed tree has no SubagentTool, so the host run loop holds
// these entries and mediates each RunResult.Consult against them — the per-kind
// budget lives for the whole run (see mediateConsult).
func buildConsultHandlers(research, advisor sporecore.Harness) map[string]sporecore.ConsultHandlerEntry {
	return map[string]sporecore.ConsultHandlerEntry{
		kindResearch: {
			Handler:  research,
			Budget:   5,
			Overflow: sporecore.ConsultOverflowPolicy{Kind: sporecore.ConsultOverflowSoftFail},
		},
		kindAdvice: {
			Handler:  advisor,
			Budget:   3,
			Overflow: sporecore.ConsultOverflowPolicy{Kind: sporecore.ConsultOverflowEscalateToHuman},
		},
	}
}

// modelAgent builds a model agent (sporecore.Agent) over the local Ollama model.
func modelAgent(id, model, baseURL string) sporecore.Agent {
	mi := ollama.WithBaseURL(model, baseURL)
	return sporecore.NewModelAgent(sporecore.AgentID(id), mi)
}

// execEvaluator is the Default-FAIL self-verification evaluator registered under
// exec-evaluator. A single read-only turn (MaxIterations = 1); the neither-pattern
// => Failed contract is built into EvaluatorResponseVerifier.
func execEvaluator() sporecore.Verifier {
	v, err := verifier.NewEvaluatorResponseVerifier(`(?i)\bPASS\b`, `(?i)\bFAIL\b`, 1)
	if err != nil {
		panic(fmt.Sprintf("evaluator regexes are valid: %v", err))
	}
	return verifier.AsHarnessVerifier(v)
}

// buildRegistry assembles the ExecutionRegistry the cordyceps tree's handles
// resolve against: agents planner/executor/ralph-agent, toolsets
// plan-tools/exec-tools, schemas plan-schema/worker-schema, and the exec-evaluator
// verifier. The handle STRINGS are ground truth from the fixture; this is the
// host-side wiring of those strings to collaborators.
//
// The toolset HANDLES must resolve for Validate(). Per-node toolset scoping is now
// RESOLVED (Issue 2): each leaf carrying a non-empty toolset handle dispatches its
// OWN tools, wired via the builder's .ToolsetTools("plan-tools", ...) /
// .ToolsetTools("exec-tools", ...) per-key catalogues. These registry slots are
// presence-only — validation entries the standalone registry's Validate() contract
// needs, NEVER dispatched (dispatch goes through the builder's per-key catalogues).
// They are kept so this registry stays self-consistent on its own; an explicit slot
// wins over the harness's auto-fill (fill-only). The real tools live on the builder.
func buildRegistry(model, baseURL string) sporecore.ExecutionRegistry {
	return sporecore.NewExecutionRegistryBuilder().
		Agent("planner", modelAgent("planner", model, baseURL)).
		Agent("executor", modelAgent("executor", model, baseURL)).
		Agent("ralph-agent", modelAgent("ralph-agent", model, baseURL)).
		Toolset("plan-tools", sporecore.NewStandardToolRegistry()).
		Toolset("exec-tools", sporecore.NewStandardToolRegistry()).
		Schema("plan-schema", planSchema()).
		Schema("worker-schema", workerSchema()).
		Verifier(execEvaluatorKey, execEvaluator()).
		Build()
}

// buildTask builds the cordyceps Task: the composed tree deserialized from the
// fixture, under a generous global backstop so the per-node PerLoop{12} worker
// bound fires first.
func buildTask(prompt string, session sporecore.SessionID) (sporecore.Task, error) {
	var tree sporecore.LoopStrategy
	if err := json.Unmarshal(cordycepsTreeJSON, &tree); err != nil {
		return sporecore.Task{}, fmt.Errorf("cordyceps_tree.json deserializes: %w", err)
	}
	maxTurns := uint32(64)
	return sporecore.NewTask(prompt, session, tree).
		WithBudget(sporecore.BudgetLimits{MaxTurns: &maxTurns}), nil
}

func main() {
	if err := run(); err != nil {
		fmt.Fprintf(os.Stderr, "\n%s\n", err)
		os.Exit(1)
	}
}

func run() error {
	ctx := context.Background()

	model := flagOrEnv("--model", "SPORE_OLLAMA_MODEL", "gemma4:e4b")
	// The #114 advisor consult handler runs a heterogeneous (cloud) model.
	advisorModel := flagOrEnv("--advisor-model", "SPORE_ADVISOR_MODEL", "minimax-m3:cloud")
	baseURL := os.Getenv("SPORE_OLLAMA_BASE_URL")
	if baseURL == "" {
		baseURL = ollama.DefaultBaseURL
	}

	// The research consult handler needs a SearXNG JSON endpoint — fail fast (like
	// examples 06/11) so a missing backend is a startup error, not a mid-run
	// surprise.
	endpoint := strings.TrimSpace(flagValue("--search-url"))
	if endpoint == "" {
		endpoint = strings.TrimSpace(os.Getenv("SPORE_WEB_SEARCH_ENDPOINT"))
	}
	if endpoint == "" {
		fmt.Fprintln(os.Stderr, "SPORE_WEB_SEARCH_ENDPOINT is not set.\n"+
			"Set it to a SearXNG JSON endpoint, e.g. "+
			"http://localhost:8888/search?format=json. See .env.example and the README.")
		os.Exit(2)
	}

	repoRoot, err := os.Getwd()
	if err != nil {
		return fmt.Errorf("failed to resolve repo root: %w", err)
	}
	if abs, absErr := filepath.Abs(repoRoot); absErr == nil {
		repoRoot = abs
	}
	workspaceRoot := filepath.Join(repoRoot, "workspace")
	if err := os.MkdirAll(workspaceRoot, 0o755); err != nil {
		return fmt.Errorf("failed to create workspace dir: %w", err)
	}

	// #142: a STABLE project id derived from the canonicalized repo root (decision
	// 5 — derived from the workspace root, not the per-window session). It keys the
	// DURABLE task_list / plan / Ralph checkpoint so they survive Ralph window
	// resets AND process restarts. ProjectIDFromPath canonicalizes first (symlinks,
	// macOS case); fall back to the pure derivation over the abs path if the repo
	// root cannot be canonicalized.
	projectID, pidErr := storage.ProjectIDFromPath(repoRoot)
	if pidErr != nil {
		projectID = storage.ProjectIDFromCanonicalPath(repoRoot)
	}
	// The CENTRAL durable root, à la Claude Code: ~/.spore/projects/<project_id>/
	// (decision 1). Falls back to a .spore dir under the repo if no home dir.
	sporeRoot := filepath.Join(repoRoot, ".spore", "projects", projectID.String())
	if home, homeErr := os.UserHomeDir(); homeErr == nil && home != "" {
		sporeRoot = filepath.Join(home, ".spore", "projects", projectID.String())
	}
	if err := os.MkdirAll(sporeRoot, 0o755); err != nil {
		return fmt.Errorf("failed to create durable root: %w", err)
	}

	// AC5: the fully-bounded tree's worst-case per-window turn count is computable
	// BEFORE the run. Ralph[PlanExecute[ReAct{4}, SelfVerifying[ReAct{12}]]] =
	// 4 + (12 + 1) = 17. An Unlimited anywhere would collapse this to (_, false).
	var treePreview sporecore.LoopStrategy
	if err := json.Unmarshal(cordycepsTreeJSON, &treePreview); err != nil {
		return fmt.Errorf("cordyceps_tree.json deserializes: %w", err)
	}
	maxSteps, bounded := treePreview.MaxSteps()
	maxStepsStr := "Unlimited (no static bound)"
	if bounded {
		maxStepsStr = fmt.Sprintf("%d", maxSteps)
	}

	fmt.Printf("model        : %s\n", model)
	fmt.Printf("advisor model: %s\n", advisorModel)
	fmt.Printf("search       : %s\n", endpoint)
	fmt.Printf("repo root    : %s\n", repoRoot)
	fmt.Printf("project id   : %s\n", projectID)
	fmt.Printf("durable root : %s\n", sporeRoot)
	fmt.Printf("strategy     : Ralph[PlanExecute[ReAct, SelfVerifying[ReAct]]] (from fixture)\n")
	fmt.Printf("max_steps    : %s  (per-window worst case; Unlimited anywhere => none)\n", maxStepsStr)
	fmt.Printf("consults     : research(web_search, budget 5, soft-fail), advice(advisor, budget 3, escalate)\n\n")

	reader := bufio.NewScanner(os.Stdin)
	for {
		prompt, ok := readAuditPrompt(reader)
		if !ok {
			break
		}

		session := sporecore.NewSessionID()
		// #142: route the DURABLE run domain to a FileSystemStorageProvider under
		// the central root (atomic write-rename) so the task_list / plan / Ralph
		// checkpoint persist across context-window resets AND process restarts.
		// Session and observability share the same durable root; memory stays
		// project-scoped. Keyed by the stable project namespace, not the per-window
		// session id.
		durable := storage.NewFileSystemStorageProvider(sporeRoot)
		store := storage.NewCompositeStorageProvider().
			Run(durable).
			Memory(storage.StorageScopeProject, durable).
			Build()
		runStore := store.Run()

		// Read-only repo sandbox: the audit never writes source files.
		sandbox, err := sporecore.NewWorkspaceScopedSandbox(sporecore.WorkspaceConfig{Root: repoRoot, ReadOnly: true})
		if err != nil {
			return fmt.Errorf("failed to create repo sandbox: %w", err)
		}
		// The advisor handler gets its own read-only view of the repo (read_file +
		// grep) so it can inspect the code the worker is asking about.
		advisorSandbox, err := sporecore.NewWorkspaceScopedSandbox(sporecore.WorkspaceConfig{Root: repoRoot, ReadOnly: true})
		if err != nil {
			return fmt.Errorf("failed to create advisor sandbox: %w", err)
		}

		// The HOST-owned consult ladder (#114). The seam moved off SubagentTool:
		// the host loop holds these handlers + per-kind budgets for the whole run.
		researchHandler, err := buildResearchHarness(model, baseURL, endpoint)
		if err != nil {
			return err
		}
		advisorHandler := buildAdvisorHarness(advisorModel, baseURL, advisorSandbox)
		consultHandlers := buildConsultHandlers(researchHandler, advisorHandler)

		// Discover skills (bundled skills/ dir + .spore/skills + ~/.spore/skills),
		// then pre-activate `audit` so its body rides the structural ContextSources
		// seam every turn (no load_skill round-trip in the composed tree). #115 /
		// SC-26: HarnessBuilder.Skills (below) registers the catalog AND the
		// load_skill tool, and the harness injects the manifest + each ACTIVE
		// skill's body STRUCTURALLY — no example-side context-manager shim.
		catalog := sporecore.Discover([]string{filepath.Join(repoRoot, "skills")}, repoRoot)
		if !catalog.Activate("audit") {
			return fmt.Errorf("bundled audit skill not discovered under %s", filepath.Join(repoRoot, "skills"))
		}

		// The harness's own model drives the Ralph wrapper; the per-node agents come
		// from the registry. Compaction/summarization uses this model too. Issue 2
		// (per-node toolset scoping): the real tools are wired PER TOOLSET, not onto
		// one global catalogue. Each leaf carrying a non-empty toolset handle
		// dispatches ONLY its own catalogue — the planner (plan-tools) cannot reach
		// exec-only tools and the executor (exec-tools) cannot reach plan-only tools.
		registry := buildRegistry(model, baseURL)
		harnessModel := ollama.WithBaseURL(model, baseURL)
		cfg := observability.ConversationalBuilder(harnessModel).
			Sandbox(sandbox).
			Storage(runStore, nil).
			// #142: pin the stable project id so durable artifacts (the task_list,
			// plan, and Ralph checkpoint) key by it — surviving the per-window
			// NewSessionID() above and process restarts — instead of the ephemeral
			// session id. The namespace is the project id projected onto the
			// SessionID axis (the builder cannot import storage directly).
			ProjectID(projectID.Namespace()).
			Skills(catalog).
			SystemPrompt(execSystemPrompt).
			ToolsetTools("plan-tools", planTools()...).
			ToolsetTools("exec-tools", execTools()...).
			BuildConfig()
		// The composed tree is wired declaratively: the registry resolves the
		// node handles, and SurfaceToHuman makes a runaway node pause to the HITL
		// REPL (AC6) rather than aborting the whole run.
		cfg.Registry = registry
		cfg.EscalationMode = sporecore.SurfaceToHumanEscalation()
		harness := sporecore.NewStandardHarness(cfg)

		task, err := buildTask(prompt, session)
		if err != nil {
			return err
		}

		// Per-RUN consult counts (host-owned, #114): how many consults of each kind
		// have already been mediated. Persists across every pause/resume of THIS
		// audit so the per-kind budget bounds the whole run, not one turn.
		consultCounts := map[string]uint32{}
		result := harness.Run(ctx, sporecore.NewHarnessRunOptions(task))
		for {
			switch result.Kind {
			case sporecore.RunSuccess:
				fmt.Printf("\ndone (%d turn(s)): %s\n", result.Turns, truncate(result.Output, 400))
				findings := filepath.Join(workspaceRoot, "findings.md")
				if _, statErr := os.Stat(findings); statErr == nil {
					fmt.Printf("\nfindings.md written: %s\n", findings)
				}
			case sporecore.RunFailure:
				fmt.Fprintf(os.Stderr, "\nfailed after %d turn(s): %+v\n", result.Turns, result.Reason)
			case sporecore.RunConsult:
				// A worker leaf consult propagated up through the composed tree (no
				// SubagentTool to absorb it). The host mediates.
				if result.State == nil || result.ConsultRequest == nil {
					fmt.Fprintln(os.Stderr, "\nconsult arrived without a resumable state; aborting.")
				} else {
					result = mediateConsult(ctx, harness, reader, consultHandlers, consultCounts, *result.ConsultRequest, *result.State)
					continue
				}
			case sporecore.RunWaitingForHuman:
				if result.State == nil {
					fmt.Fprintln(os.Stderr, "\nescalation arrived without a resumable state; aborting.")
				} else {
					result = handleHumanEscalation(ctx, harness, reader, *result.State, result.Request)
					continue
				}
			default:
				fmt.Fprintf(os.Stderr, "\nrun ended unexpectedly: %s %+v\n", result.Kind, result.Reason)
			}
			break
		}
	}

	fmt.Println("\nbye.")
	return nil
}

// mediateConsult mediates one worker leaf consult HOST-SIDE (#114, seam relocated
// off SubagentTool). Routes by kind, enforces the per-kind budget held in counts
// for the whole run, runs the handler harness as a direct child, and resumes the
// paused composed tree with the answer — or applies the overflow policy (SoftFail
// resumes with BudgetExhausted; EscalateToHuman surfaces the advisor ladder to the
// operator). Identical to the old SubagentTool mediate_consult, only the owner
// moved.
func mediateConsult(
	ctx context.Context,
	harness sporecore.Harness,
	reader *bufio.Scanner,
	handlers map[string]sporecore.ConsultHandlerEntry,
	counts map[string]uint32,
	request sporecore.ConsultRequest,
	state sporecore.PausedState,
) sporecore.RunResult {
	// No handler for this kind => resume the worker without help (loud, not a
	// silent stall).
	entry, ok := handlers[request.Kind]
	if !ok {
		fmt.Fprintf(os.Stderr, "\n(no consult handler for kind %q; worker proceeds)\n", request.Kind)
		return harness.ResumeConsult(ctx, state, sporecore.NewConsultBudgetExhausted(
			fmt.Sprintf("no consult handler for kind %q; proceed without further help", request.Kind)), nil)
	}

	// Per-kind budget: counts[kind] is how many consults of this kind were already
	// mediated this run. The handler runs while used < budget; the (budget+1)th
	// consult overflows.
	if counts[request.Kind] >= entry.Budget {
		switch entry.Overflow.Kind {
		case sporecore.ConsultOverflowSoftFail:
			fmt.Printf("\n(consult budget for %q exhausted — worker finishes with what it has)\n", request.Kind)
			return harness.ResumeConsult(ctx, state, sporecore.NewConsultBudgetExhausted(
				fmt.Sprintf("consult budget for kind %q exhausted; proceed without further help", request.Kind)), nil)
		default: // ConsultOverflowEscalateToHuman
			return handleConsultOverflow(ctx, harness, reader, entry, request, state)
		}
	}

	// Run the handler harness as a direct child (depth-1, never under the worker)
	// on the consult rendered to text, then resume with its answer.
	counts[request.Kind]++
	fmt.Printf("\n┌─ consult (%s) → %d of %d budget\n", request.Kind, counts[request.Kind], entry.Budget)
	answer := runConsultHandler(ctx, entry.Handler, request)
	fmt.Printf("└─ consult answer: %s\n", truncate(answer, 200))
	return harness.ResumeConsult(ctx, state, sporecore.NewConsultAnswer(answer), nil)
}

// runConsultHandler runs a consult handler harness on the rendered request and
// returns its answer text (or its failure text, so a handler that does not
// cleanly complete never stalls the worker).
func runConsultHandler(ctx context.Context, handler sporecore.Harness, request sporecore.ConsultRequest) string {
	task := sporecore.NewTask(
		renderConsultInstruction(request),
		sporecore.NewSessionID(),
		sporecore.ReActStrategy(16),
	)
	r := handler.Run(ctx, sporecore.NewHarnessRunOptions(task))
	if r.Kind == sporecore.RunSuccess {
		return r.Output
	}
	return fmt.Sprintf("consult handler did not complete cleanly: %s %+v", r.Kind, r.Reason)
}

// renderConsultInstruction renders a ConsultRequest to the handler's instruction
// text (#114).
func renderConsultInstruction(request sporecore.ConsultRequest) string {
	return fmt.Sprintf(
		"A worker agent is requesting help (kind: %s).\n\nSituation: %s\n\nAttempts so far: %d\n\nQuestion: %s",
		request.Kind, request.Situation, request.Attempts, request.Question)
}

// handleConsultOverflow handles the advice consult overflowing its budget under
// EscalateToHuman: present the #114 three-choice ladder to the operator and resume
// the worker with the decision. Preserves the original ladder semantics host-side.
func handleConsultOverflow(
	ctx context.Context,
	harness sporecore.Harness,
	reader *bufio.Scanner,
	entry sporecore.ConsultHandlerEntry,
	request sporecore.ConsultRequest,
	state sporecore.PausedState,
) sporecore.RunResult {
	fmt.Println("\n╔═ HUMAN ESCALATION (advisor budget exhausted) ═")
	fmt.Printf("║ situation: %s\n", truncate(request.Situation, 200))
	fmt.Printf("║ question : %s\n", truncate(request.Question, 200))
	fmt.Println("║ [1] run the advisor once more (host-side)")
	fmt.Println("║ [2] abort this consult — worker proceeds without help")
	fmt.Println("║ [3] type a free-form answer yourself")
	fmt.Println("╚═════════════════════════════════════════════════")

	switch strings.TrimSpace(promptLine(reader, "> ")) {
	case "2":
		return harness.ResumeConsult(ctx, state, sporecore.NewConsultBudgetExhausted(
			"advisor budget exhausted; proceed without further help"), nil)
	case "3":
		text := promptLine(reader, "answer> ")
		return harness.ResumeConsult(ctx, state, sporecore.NewConsultAnswer(text), nil)
	default:
		// Default ([1] or empty): run the advisor handler once more host-side and
		// inject its answer — a bounded escape hatch past the per-kind budget.
		fmt.Println("(running advisor for one more turn…)")
		answer := runConsultHandler(ctx, entry.Handler, request)
		fmt.Printf("advisor: %s\n", truncate(answer, 300))
		return harness.ResumeConsult(ctx, state, sporecore.NewConsultAnswer(answer), nil)
	}
}

// handleHumanEscalation presents a BudgetExhausted pause and resumes with the
// operator's choice. The composed tree surfaces a runaway node here under
// SurfaceToHuman; we offer its available_actions and resume by re-resolving
// handles (no reconfiguration).
func handleHumanEscalation(
	ctx context.Context,
	harness sporecore.Harness,
	reader *bufio.Scanner,
	state sporecore.PausedState,
	request *sporecore.HumanRequest,
) sporecore.RunResult {
	if request == nil || request.Kind != sporecore.HumanReqBudgetExhausted {
		// The composed tree only escalates via BudgetExhausted; anything else is
		// unexpected — halt cleanly.
		fmt.Fprintf(os.Stderr, "\nunexpected human request: %+v\n", request)
		return harness.Resume(ctx, state, sporecore.HumanResponse{Kind: sporecore.HumanRespHalt}, nil)
	}

	fmt.Printf("\n╔═ BUDGET ESCALATION (%s) ═══════════════════\n", request.Phase)
	actions := request.AvailableActions
	for i, a := range actions {
		fmt.Printf("║ [%d] %s\n", i+1, describeAction(a))
	}
	fmt.Println("╚═════════════════════════════════════════════════")

	choice := strings.TrimSpace(promptLine(reader, "> "))
	idx := 0
	if n, err := parseChoice(choice); err == nil && n >= 1 {
		idx = n - 1
	}
	// Default to a small budget bump so an empty line keeps the run going.
	action := sporecore.EscalationAction{Kind: sporecore.EscalationContinueWithBudget, Steps: 12}
	if idx >= 0 && idx < len(actions) {
		action = actions[idx]
	}
	fmt.Printf("(resuming with %s)\n", describeAction(action))
	return harness.Resume(ctx, state, sporecore.HumanResponse{Kind: sporecore.HumanRespEscalate, Action: action}, nil)
}

func describeAction(a sporecore.EscalationAction) string {
	switch a.Kind {
	case sporecore.EscalationContinueWithBudget:
		return fmt.Sprintf("continue with +%d steps", a.Steps)
	case sporecore.EscalationSkip:
		return "skip this task"
	default:
		return "fail this node"
	}
}

func parseChoice(s string) (int, error) {
	var n int
	_, err := fmt.Sscanf(s, "%d", &n)
	return n, err
}

// readAuditPrompt reads one audit prompt from the REPL. (prompt, true) to run (an
// empty line => the default verbatim); ("", false) on EOF (Ctrl-D), which quits.
func readAuditPrompt(reader *bufio.Scanner) (string, bool) {
	fmt.Println("Default audit prompt (press enter to accept, type your own, or Ctrl-D to quit):")
	fmt.Printf("  %s\n", defaultAuditPrompt)
	fmt.Print("audit> ")
	if !reader.Scan() {
		return "", false
	}
	line := strings.TrimRight(reader.Text(), "\r\n")
	if strings.TrimSpace(line) == "" {
		return defaultAuditPrompt, true
	}
	return line, true
}

// promptLine prints a prompt and reads one line from the scanner.
func promptLine(reader *bufio.Scanner, prompt string) string {
	fmt.Print(prompt)
	if !reader.Scan() {
		return ""
	}
	return strings.TrimRight(reader.Text(), "\r\n")
}

// flagValue returns the value following the given flag in os.Args, or "".
func flagValue(flag string) string {
	args := os.Args[1:]
	for i, a := range args {
		if a == flag && i+1 < len(args) {
			return args[i+1]
		}
	}
	return ""
}

// flagOrEnv resolves a value from a CLI flag, then an env var, then a default.
func flagOrEnv(flag, env, def string) string {
	if v := flagValue(flag); v != "" {
		return v
	}
	if v := os.Getenv(env); v != "" {
		return v
	}
	return def
}

// truncate keeps boundary lines readable.
func truncate(s string, max int) string {
	s = strings.ReplaceAll(s, "\n", " ")
	r := []rune(s)
	if len(r) <= max {
		return s
	}
	return string(r[:max]) + "…"
}
