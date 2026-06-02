//! End-to-end scenario assembly (issue #57).
//!
//! Reusable wiring shared by the `e2e_agent` example binary AND the hermetic
//! integration tests, so a live run (`OllamaModelInterface` + `ModelAgent`) and
//! an offline run (`MockAgent` + `ScriptedToolRegistry`) drive the *same* code
//! path. The four `build_scenario_sN` builders are generic over
//! `Arc<dyn Agent>` + `Arc<dyn HarnessToolRegistry>`, so the only difference
//! between live and mock mode is which agent/registry you inject.
//!
//! Everything in this module is on the **live** path and therefore does NOT
//! require the `test-utils` feature.
//!
//! ## Architectural gaps closed here (see issue #57)
//!
//! - [`RealToolRegistry`] bridges the two `ToolRegistry` traits: the harness
//!   loop calls [`crate::harness::ToolRegistry::dispatch`] (no sandbox arg),
//!   while the real tools live behind
//!   [`crate::tool_registry::ToolRegistry::dispatch`] (takes a sandbox). The
//!   bridge owns the inner [`StandardToolRegistry`] + an
//!   `Arc<dyn SandboxProvider>` and forwards, mapping `DispatchError` onto a
//!   recoverable [`ToolOutput::Error`].
//! - [`SchemaInjectingContextManager`] decorates any harness context manager so
//!   `assemble().tools` is populated from the registry's tool schemas (sorted
//!   by name for cache stability). Without it the compaction adapter surfaces
//!   no tools and the model can never emit a tool call.
//! - [`FailingTool`] always returns a *recoverable* error so the harness
//!   appends it as a tool result and the agent can adapt (scenario S4).
//! - [`CompleteOnFinalResponse`] is a non-test termination policy: it lets the
//!   loop succeed as soon as the agent produces a final response.

use std::sync::Arc;

use crate::agent::{Agent, Context as AgentContext};
use crate::cache_provider::CacheProvider;
use crate::compaction_adapter::{seed_rich_state, HarnessContextManagerExt};
use crate::context::{CompactionConfig, SessionState as RichSessionState, StandardContextManager};
use crate::harness::{
    BoxFut, BudgetSnapshot, ContextManager as HarnessContextManager, HarnessBuilder,
    SandboxProvider, SessionState as HarnessState, StandardHarness, Task, TerminationDecision,
    TerminationPolicy, ToolOutput, ToolRegistry as HarnessToolRegistry, ToolResult,
};
use crate::model::ModelInterface;
use crate::model::{Content, Message, Role, ToolCall, ToolSchema};
use crate::tool_registry::{
    StandardToolRegistry, Tool, ToolAnnotations, ToolContext, ToolRegistry,
    ToolSchema as RegistrySchema,
};
use crate::tools::exec::{BashCommandTool, ExecTool};
use crate::tools::fs::{ListDirTool, ReadFileTool, WriteFileTool};
use crate::tools::tasklist::TaskListTool;

// ============================================================================
// RealToolRegistry — bridge between the two ToolRegistry traits
// ============================================================================
//
// The bridge graduated into `tool_registry` (the blessed surface) so callers
// don't import it from a scenarios-flavored module (#91). Re-exported here for
// back-compat with existing scenario/example wiring.
pub use crate::tool_registry::RealToolRegistry;

// ============================================================================
// SchemaInjectingContextManager — fills assemble().tools from the registry
// ============================================================================

/// Operational system prompt for the live agent. The compaction adapter's
/// `assemble` produces a context with **no system prompt** (it has no
/// `ContextSources` to render one), so without this the model receives only the
/// task as a user message and no guidance on how to behave. The three rules
/// target the failure modes observed with small local models: describing
/// actions instead of taking them, passing stringified arguments, and declaring
/// success without checking the result.
pub const AGENT_SYSTEM_PROMPT: &str = "\
You are an autonomous agent that completes tasks by calling the provided tools. \
Follow these rules:

1. ACT, DON'T DESCRIBE. To make something happen, call the appropriate tool. \
Writing a shell command, code snippet, or file contents into your text reply \
does NOT run it — only a real tool call has any effect. When a task asks you to \
produce a file or a result, call the tool that performs the action and let the \
tool do the work; never paste the command, code, or expression you *would* run \
as if it were the finished result.

2. USE CORRECTLY-TYPED ARGUMENTS. Pass tool arguments as typed JSON: booleans \
as true/false (not \"true\"), numbers as 12 (not \"12\"), lists as [\"a\"] (not \
\"[\\\"a\\\"]\"). Quoted-string scalars where a bool/number/array is expected \
will be rejected.

3. VERIFY BEFORE FINISHING. Before replying DONE, confirm your work actually \
satisfies the request. If you wrote a file, read it back with read_file and \
check its contents are exactly what was asked. If they do not match, fix it and \
verify again. Only reply DONE once you have verified the result is correct.";

/// Decorates a harness [`HarnessContextManager`], delegating every seam method
/// to the inner manager but injecting the registry's tool schemas into
/// `assemble().tools` and prepending [`AGENT_SYSTEM_PROMPT`]. The compaction
/// adapter's `assemble` returns an empty tool list and no system prompt, so
/// without this decorator the model never sees any tools (and can never emit a
/// tool call) nor any operational guidance in live mode.
pub struct SchemaInjectingContextManager {
    inner: Arc<dyn HarnessContextManager>,
    tools: Vec<ToolSchema>,
}

impl SchemaInjectingContextManager {
    /// Wrap `inner`, injecting `tools` (sorted by name) into every assembled
    /// context.
    pub fn new(inner: Arc<dyn HarnessContextManager>, mut tools: Vec<ToolSchema>) -> Self {
        tools.sort_by(|a, b| a.name.cmp(&b.name));
        Self { inner, tools }
    }
}

impl HarnessContextManager for SchemaInjectingContextManager {
    fn assemble<'a>(
        &'a self,
        session: &'a HarnessState,
        task: &'a Task,
    ) -> BoxFut<'a, AgentContext> {
        let tools = self.tools.clone();
        Box::pin(async move {
            let mut ctx = self.inner.assemble(session, task).await;
            ctx.tools = tools;
            // Prepend the operational system prompt. The adapter's assemble
            // yields none, so the model would otherwise get no guidance. Guard
            // against duplicates so a resumed/seeded session that already leads
            // with a System message isn't given two.
            let has_system = ctx
                .messages
                .first()
                .is_some_and(|m| matches!(m.role, Role::System));
            if !has_system {
                ctx.messages.insert(
                    0,
                    Message {
                        role: Role::System,
                        content: Content::Text {
                            text: AGENT_SYSTEM_PROMPT.to_string(),
                        },
                    },
                );
            }
            ctx
        })
    }

    fn append_tool_result<'a>(
        &'a self,
        session: &'a mut HarnessState,
        result: &'a ToolResult,
    ) -> BoxFut<'a, ()> {
        self.inner.append_tool_result(session, result)
    }

    fn append_assistant_message<'a>(
        &'a self,
        session: &'a mut HarnessState,
        message: &'a Message,
    ) -> BoxFut<'a, ()> {
        self.inner.append_assistant_message(session, message)
    }

    fn append_user_message<'a>(
        &'a self,
        session: &'a mut HarnessState,
        text: &'a str,
    ) -> BoxFut<'a, ()> {
        self.inner.append_user_message(session, text)
    }

    fn should_compact(&self, session: &HarnessState) -> bool {
        self.inner.should_compact(session)
    }

    fn prepare_compaction_turn(
        &self,
        session: &HarnessState,
    ) -> Option<crate::harness::CompactionTurn> {
        self.inner.prepare_compaction_turn(session)
    }

    fn inject_missing_items(&self, context: &mut AgentContext, missing: &[String]) {
        self.inner.inject_missing_items(context, missing)
    }

    fn apply_compaction(&self, session: &mut HarnessState, summary: String) {
        self.inner.apply_compaction(session, summary)
    }

    fn token_budget_used(&self, session: &HarnessState) -> Option<u32> {
        self.inner.token_budget_used(session)
    }
}

// ============================================================================
// FailingTool — deliberately-failing recoverable tool (S4)
// ============================================================================

/// A tool that always fails with a *recoverable* error. Used by scenario S4 to
/// prove the loop surfaces a tool error to the agent and lets it adapt rather
/// than crashing or hanging. Must NOT be `is_always_halt`.
pub struct FailingTool;

impl FailingTool {
    pub const NAME: &'static str = "flaky_op";

    pub fn new() -> Self {
        Self
    }

    /// Registry schema for the failing tool.
    pub fn schema() -> RegistrySchema {
        RegistrySchema {
            name: Self::NAME.into(),
            description: "A flaky operation that fails the first time it is called".into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": { "reason": { "type": "string" } },
            }),
            annotations: ToolAnnotations {
                idempotent: true,
                ..Default::default()
            },
        }
    }
}

impl Default for FailingTool {
    fn default() -> Self {
        Self::new()
    }
}

impl Tool for FailingTool {
    fn name(&self) -> &str {
        Self::NAME
    }

    fn execute<'a>(
        &'a self,
        _call: &'a ToolCall,
        _sandbox: &'a (dyn SandboxProvider + 'a),
        _ctx: &'a ToolContext,
    ) -> BoxFut<'a, ToolOutput> {
        Box::pin(async move {
            ToolOutput::Error {
                message: "flaky_op is unavailable right now; try a different approach".into(),
                recoverable: true,
            }
        })
    }
}

// ============================================================================
// CompleteOnFinalResponse — non-test termination policy
// ============================================================================

/// Termination policy that lets the loop complete as soon as the agent
/// produces a final response (always `Continue`, which the harness interprets
/// as "accept the final response and succeed"). Available on the live path
/// without the `test-utils` feature.
pub struct CompleteOnFinalResponse;

impl TerminationPolicy for CompleteOnFinalResponse {
    fn evaluate<'a>(
        &'a self,
        _session: &'a HarnessState,
        _budget_used: &'a BudgetSnapshot,
    ) -> BoxFut<'a, TerminationDecision> {
        Box::pin(async { TerminationDecision::Continue })
    }
}

// ============================================================================
// Real tool registry construction
// ============================================================================

/// Build a [`StandardToolRegistry`] for `scenario`. The base catalog is always
/// `read_file`, `write_file`, `list_dir`, `exec`, and [`FailingTool`]
/// (`flaky_op`). The real shell tool `bash_command` is added ONLY for
/// [`ScenarioId::S5`] — S1/S2 measure reasoning + act-don't-describe, and a
/// live model handed a shell could shortcut S1 with `cat … | tr … > …` without
/// demonstrating the intended behavior. `exec` is safe everywhere because it
/// cannot pipe or redirect.
pub fn build_real_tool_registry(scenario: ScenarioId) -> Arc<StandardToolRegistry> {
    let registry = StandardToolRegistry::new();
    // Registration errors here are programming errors (duplicate/invalid
    // schema) — surface them loudly via expect rather than silently.
    registry
        .register(Box::new(ReadFileTool::new()), ReadFileTool::schema())
        .expect("register read_file");
    registry
        .register(Box::new(WriteFileTool::new()), WriteFileTool::schema())
        .expect("register write_file");
    registry
        .register(Box::new(ListDirTool::new()), ListDirTool::schema())
        .expect("register list_dir");
    registry
        .register(Box::new(ExecTool::new()), ExecTool::schema())
        .expect("register exec");
    registry
        .register(Box::new(FailingTool::new()), FailingTool::schema())
        .expect("register flaky_op");
    registry
        .register(Box::new(TaskListTool::new()), TaskListTool::schema())
        .expect("register task_list");
    if matches!(scenario, ScenarioId::S5) {
        registry
            .register(Box::new(BashCommandTool::new()), BashCommandTool::schema())
            .expect("register bash_command");
    }
    Arc::new(registry)
}

// ============================================================================
// Rich context-manager assembly (live compaction)
// ============================================================================

/// Build a real compaction-capable context manager: a
/// [`StandardContextManager`] wrapped in the [`StandardCompactionAdapter`]
/// (`into_harness_adapter`). Generic over the model so live mode passes the
/// Ollama model and tests pass a stub.
pub fn build_rich_context_manager<M: ModelInterface + 'static>(
    model: Arc<M>,
    cache: Arc<dyn CacheProvider>,
    config: CompactionConfig,
) -> Arc<dyn HarnessContextManager> {
    Arc::new(StandardContextManager::new(model, cache, config)).into_harness_adapter()
}

/// Seed a harness [`HarnessState`] with rich compaction state for the S3
/// scenario: a small window, a budget near the threshold, and a history longer
/// than `preserve_recent_n` so compaction fires mid-run. The session can then
/// compact, continue, and compact again (healthy multi-compaction) because the
/// token-accounting fix decrements the budget on each compaction.
pub fn seed_compaction_state(
    session: &mut HarnessState,
    task_instruction: &str,
    session_id: crate::harness::SessionId,
    task_id: crate::harness::TaskId,
    window_limit: u32,
    token_budget_used: u32,
    history_len: usize,
) {
    use crate::model::{Content, Message, Role};
    let mut rich = RichSessionState::new(session_id, task_id, task_instruction);
    rich.window_limit = window_limit;
    rich.token_budget_used = token_budget_used;
    rich.message_history = (0..history_len)
        .map(|i| Message {
            role: if i % 2 == 0 {
                Role::User
            } else {
                Role::Assistant
            },
            content: Content::Text {
                text: format!(
                    "history message {i}: progress notes on the payment service deploy with \
                     enough content to carry a meaningful token estimate for reclamation"
                ),
            },
        })
        .collect();
    seed_rich_state(session, &rich);
}

// ============================================================================
// Scenario builders (generic over agent + tool registry)
// ============================================================================

/// The scenario id, parsed from the CLI arg `s1`..`s5`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScenarioId {
    S1,
    S2,
    S3,
    S4,
    S5,
}

impl ScenarioId {
    /// Parse `s1`..`s5` (case-insensitive).
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_lowercase().as_str() {
            "s1" => Some(Self::S1),
            "s2" => Some(Self::S2),
            "s3" => Some(Self::S3),
            "s4" => Some(Self::S4),
            "s5" => Some(Self::S5),
            _ => None,
        }
    }

    /// The default prompt that drives this scenario.
    pub fn prompt(self) -> &'static str {
        match self {
            Self::S1 => {
                "Complete this task step by step, using the provided tools:\n\
                 1. Call read_file to read the contents of input.txt. Use the \
                 exact text it returns — do not invent or substitute any text.\n\
                 2. Take that exact text and rewrite it with every lowercase \
                 letter changed to its capital form, keeping all other \
                 characters, spaces, and punctuation the same.\n\
                 3. Call write_file with path 'output.txt' and content set to \
                 the uppercased text from step 2 — the literal capital letters \
                 themselves. The content must be the transformed words from \
                 input.txt, NOT a shell command, NOT a $(...) expression, and \
                 NOT any code.\n\
                 4. Call read_file on output.txt and check its contents equal \
                 the uppercased text from step 2.\n\
                 Reply DONE only once output.txt contains input.txt's contents \
                 in all capital letters."
            }
            Self::S2 => {
                "Create a file notes.md containing a TODO list with one item: 'set up the \
                 project'. Use write_file. Reply DONE when written."
            }
            Self::S3 => {
                "Summarize the long conversation so far and continue working on the deploy of \
                 the payment service. Reply DONE when finished."
            }
            Self::S4 => {
                "Call the flaky_op tool. If it fails, do not give up: write a file \
                 recovered.txt explaining that flaky_op failed and how you adapted, using \
                 write_file. Reply DONE when finished."
            }
            Self::S5 => {
                "Transform input.txt into output.txt with every lowercase letter \
                 uppercased, using the shell.\n\
                 1. Call bash_command with a real shell pipeline that reads \
                 input.txt, uppercases it, and writes output.txt — e.g. \
                 `cat input.txt | tr a-z A-Z > output.txt`. This is exactly what \
                 the bash_command tool is for: it runs your script via /bin/sh \
                 -c, so pipes (|) and redirects (>) work.\n\
                 2. Call read_file on output.txt and check its contents are \
                 input.txt's text in all capital letters.\n\
                 Reply DONE only once output.txt contains the uppercased text."
            }
        }
    }
}

/// Assemble a [`StandardHarness`] for the given scenario from injected
/// components. Generic over the agent and tool registry so live mode
/// (`OllamaModelInterface`/`ModelAgent` + [`RealToolRegistry`]) and mock mode
/// (`MockAgent` + `ScriptedToolRegistry`) share one code path.
///
/// `tool_schemas` are injected into every assembled context (sorted by name)
/// via [`SchemaInjectingContextManager`]. Pass the registry's schemas in live
/// mode, or an empty vec in mock mode where the scripted agent does not need
/// them.
///
/// `observability` is injected directly when present — the example passes a
/// durable-outbox provider; hermetic tests pass an in-memory provider to assert
/// spans. `None` runs with no observability.
#[allow(clippy::too_many_arguments)]
pub fn build_scenario(
    _scenario: ScenarioId,
    agent: Arc<dyn Agent>,
    tools: Arc<dyn HarnessToolRegistry>,
    sandbox: Arc<dyn SandboxProvider>,
    context_manager: Arc<dyn HarnessContextManager>,
    termination_policy: Arc<dyn TerminationPolicy>,
    tool_schemas: Vec<ToolSchema>,
    observability: Option<Arc<dyn crate::harness::ObservabilityProvider>>,
) -> StandardHarness {
    let context_manager: Arc<dyn HarnessContextManager> = Arc::new(
        SchemaInjectingContextManager::new(context_manager, tool_schemas),
    );

    let builder = HarnessBuilder::new(agent, tools, sandbox, context_manager, termination_policy)
        // Honor SPORE_TRACE_CONTENT / SPORE_TRACE_CONTENT_MAX_LEN (#64) so a live
        // run can capture gen_ai.* conversation + tool content for Phoenix.
        // Defaults OFF when the env var is unset.
        .content_capture(crate::ContentCaptureConfig::from_env())
        // Tool-call repair: weak models (e.g. llama3.2) emit stringified args
        // ("false" for a bool, a string for a sequence); deterministic coercion
        // repairs and re-dispatches so the agent recovers instead of looping.
        .tool_call_repair(Arc::new(crate::tool_call_repair::StandardToolCallRepair));
    let builder = match observability {
        Some(obs) => builder.observability(obs),
        None => builder,
    };
    builder.build()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::harness::SessionId;
    use crate::storage::{InMemoryStorageProvider, MemoryStore, RunStore};
    use crate::tool_registry::mock::AllowAllSandbox;

    /// A fresh in-memory run store for bridge tests.
    fn test_run_store() -> Arc<dyn RunStore> {
        Arc::new(InMemoryStorageProvider::new())
    }

    /// A fresh in-memory memory store for bridge tests (#78).
    fn test_memory_store() -> Arc<dyn MemoryStore> {
        Arc::new(InMemoryStorageProvider::new())
    }

    #[test]
    fn scenario_id_parses() {
        assert_eq!(ScenarioId::parse("s1"), Some(ScenarioId::S1));
        assert_eq!(ScenarioId::parse("S4"), Some(ScenarioId::S4));
        assert_eq!(ScenarioId::parse("s5"), Some(ScenarioId::S5));
        assert_eq!(ScenarioId::parse("nope"), None);
    }

    fn schema_names(scenario: ScenarioId) -> Vec<String> {
        let reg = build_real_tool_registry(scenario);
        let bridge = RealToolRegistry::new(
            reg,
            Arc::new(AllowAllSandbox),
            SessionId::new("schema-test"),
            test_run_store(),
            test_memory_store(),
        );
        bridge
            .model_schemas()
            .iter()
            .map(|s| s.name.clone())
            .collect()
    }

    #[test]
    fn real_registry_exposes_sorted_schemas() {
        let names = schema_names(ScenarioId::S1);
        let mut sorted = names.clone();
        sorted.sort();
        assert_eq!(names, sorted, "schemas must be sorted by name");
        assert!(names.contains(&"flaky_op".to_string()));
        assert!(names.contains(&"read_file".to_string()));
    }

    #[test]
    fn s1_registry_has_exec_not_bash_command() {
        let names = schema_names(ScenarioId::S1);
        assert!(names.contains(&"exec".to_string()));
        assert!(!names.contains(&"bash_command".to_string()));
    }

    #[test]
    fn s2_registry_lacks_bash_command() {
        let names = schema_names(ScenarioId::S2);
        assert!(names.contains(&"exec".to_string()));
        assert!(!names.contains(&"bash_command".to_string()));
    }

    #[test]
    fn s5_registry_has_bash_command() {
        let names = schema_names(ScenarioId::S5);
        assert!(names.contains(&"bash_command".to_string()));
        assert!(names.contains(&"exec".to_string()));
    }

    #[tokio::test]
    async fn failing_tool_returns_recoverable_error() {
        let bridge = RealToolRegistry::new(
            build_real_tool_registry(ScenarioId::S4),
            Arc::new(AllowAllSandbox),
            SessionId::new("s4-test"),
            test_run_store(),
            test_memory_store(),
        );
        let out = bridge
            .dispatch(ToolCall {
                id: "c1".into(),
                name: FailingTool::NAME.into(),
                input: serde_json::json!({}),
            })
            .await;
        match out {
            ToolOutput::Error { recoverable, .. } => assert!(recoverable),
            other => panic!("expected recoverable error, got {other:?}"),
        }
        assert!(!bridge.is_always_halt(FailingTool::NAME));
    }
}
