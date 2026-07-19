// ============================================================================
// Test-only stubs of the sibling traits (cover ReAct paths in the unit tests
// before the canonical impls land in #4–#13).
// ============================================================================

use super::*;
use std::sync::Mutex;

pub struct NoopContextManager;
impl ContextManager for NoopContextManager {
    fn assemble<'a>(
        &'a self,
        session: &'a SessionState,
        _task: &'a Task,
        _sources: &'a ContextSources,
    ) -> BoxFut<'a, Context> {
        let messages = session.messages.clone();
        Box::pin(async move {
            Context {
                messages,
                tools: vec![],
                params: crate::model::ModelParams::default(),
            }
        })
    }
    fn append_tool_result<'a>(
        &'a self,
        session: &'a mut SessionState,
        result: &'a ToolResult,
    ) -> BoxFut<'a, ()> {
        Box::pin(async move {
            let text = match &result.output {
                ToolOutput::Success { content, .. } => content.clone(),
                ToolOutput::Error { message, .. } => format!("[error] {message}"),
                ToolOutput::SandboxViolation { violation } => {
                    format!("[error] sandbox violation: {violation}")
                }
                ToolOutput::WaitingForHuman { .. } => "[waiting]".into(),
                ToolOutput::Escalate { .. } => "[escalate]".into(),
                ToolOutput::AwaitingClarification { .. } => "[clarification]".into(),
                ToolOutput::Consult { .. } => "[consult]".into(),
            };
            // Mirrors the real context managers: results are structural and name
            // the call they answer. A double that flattened this to text would
            // let a test pass while production shipped unlinked tool messages.
            session.messages.push(Message {
                role: crate::model::Role::Tool,
                content: crate::model::Content::ToolResult(crate::model::ToolResult {
                    tool_use_id: result.call_id.clone(),
                    content: text,
                    is_error: crate::compaction_adapter::tool_result_is_error(&result.output),
                }),
            });
        })
    }
    fn append_assistant_message<'a>(
        &'a self,
        session: &'a mut SessionState,
        message: &'a Message,
    ) -> BoxFut<'a, ()> {
        let message = message.clone();
        Box::pin(async move {
            session.messages.push(message);
        })
    }
    fn append_user_message<'a>(
        &'a self,
        session: &'a mut SessionState,
        text: &'a str,
    ) -> BoxFut<'a, ()> {
        Box::pin(async move {
            session.messages.push(Message {
                role: crate::model::Role::User,
                content: crate::model::Content::Text { text: text.into() },
            });
        })
    }
}

pub struct AllowAllSandbox;
impl SandboxProvider for AllowAllSandbox {
    fn validate<'a>(&'a self, _call: &'a ToolCall) -> BoxFut<'a, Result<(), SandboxViolation>> {
        Box::pin(async { Ok(()) })
    }
}

/// Deterministic [`VcsProvider`] double for tests and fixture replay (issue
/// #58 v2). It returns pre-loaded strings VERBATIM with no process spawning,
/// so multi-context-window Ralph continuation can be exercised hermetically.
/// [`log`](VcsProvider::log) ignores its [`VcsLogArgs`] and yields
/// `log_output`; [`status`](VcsProvider::status) yields `status_output`.
pub struct FixtureVcsProvider {
    log_output: String,
    status_output: String,
}

impl FixtureVcsProvider {
    /// Construct a fixture provider returning `log_output` from `log` and
    /// `status_output` from `status`.
    pub fn new(log_output: impl Into<String>, status_output: impl Into<String>) -> Self {
        Self {
            log_output: log_output.into(),
            status_output: status_output.into(),
        }
    }
}

impl VcsProvider for FixtureVcsProvider {
    fn log<'a>(&'a self, _args: &'a VcsLogArgs) -> BoxFut<'a, Result<String, VcsError>> {
        Box::pin(async move { Ok(self.log_output.clone()) })
    }

    fn status<'a>(&'a self) -> BoxFut<'a, Result<String, VcsError>> {
        Box::pin(async move { Ok(self.status_output.clone()) })
    }
}

pub struct ScriptedSandbox {
    outcomes: Mutex<std::collections::VecDeque<Result<(), SandboxViolation>>>,
}
impl Default for ScriptedSandbox {
    fn default() -> Self {
        Self::new()
    }
}
impl ScriptedSandbox {
    pub fn new() -> Self {
        Self {
            outcomes: Mutex::new(Default::default()),
        }
    }
    pub fn push(&self, r: Result<(), SandboxViolation>) -> &Self {
        self.outcomes.lock().unwrap().push_back(r);
        self
    }
}
impl SandboxProvider for ScriptedSandbox {
    fn validate<'a>(&'a self, _call: &'a ToolCall) -> BoxFut<'a, Result<(), SandboxViolation>> {
        let r = self.outcomes.lock().unwrap().pop_front().unwrap_or(Ok(()));
        Box::pin(async move { r })
    }
}

pub struct ScriptedToolRegistry {
    outputs: Mutex<std::collections::VecDeque<ToolOutput>>,
    pub call_count: std::sync::atomic::AtomicUsize,
    always_halt: Mutex<Vec<String>>,
    /// Optional advertised tool schemas (issue #137: lets the error-loop
    /// breaker's `enrich_tool_error` render a real parameter schema). Empty
    /// by default — preserving every existing caller's behavior.
    schemas: Mutex<Vec<ToolSchema>>,
    /// When non-empty, `dispatch` returns the SAME recoverable error for
    /// EVERY call regardless of the queued `outputs` (issue #137 — models a
    /// tool that always rejects a malformed call, e.g. `add_task` missing a
    /// required `description`). Used by the tool-error-loop replay/tests.
    sticky_error: Mutex<Option<String>>,
}
impl Default for ScriptedToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}
impl ScriptedToolRegistry {
    pub fn new() -> Self {
        Self {
            outputs: Mutex::new(Default::default()),
            call_count: Default::default(),
            always_halt: Mutex::new(vec![]),
            schemas: Mutex::new(vec![]),
            sticky_error: Mutex::new(None),
        }
    }
    pub fn push(&self, o: ToolOutput) -> &Self {
        self.outputs.lock().unwrap().push_back(o);
        self
    }
    pub fn mark_always_halt(&self, name: &str) {
        self.always_halt.lock().unwrap().push(name.into());
    }
    /// Advertise a tool schema (issue #137). Returns `&Self` for chaining.
    pub fn with_schema(&self, schema: ToolSchema) -> &Self {
        self.schemas.lock().unwrap().push(schema);
        self
    }
    /// Make every dispatch return the same recoverable error message,
    /// regardless of arguments (issue #137 error-loop scenario).
    pub fn always_recoverable_error(&self, message: &str) -> &Self {
        *self.sticky_error.lock().unwrap() = Some(message.into());
        self
    }
}
impl ToolRegistry for ScriptedToolRegistry {
    fn dispatch<'a>(&'a self, _call: ToolCall) -> BoxFut<'a, ToolOutput> {
        self.call_count
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        if let Some(message) = self.sticky_error.lock().unwrap().clone() {
            return Box::pin(async move {
                ToolOutput::Error {
                    message,
                    recoverable: true,
                }
            });
        }
        let out = self
            .outputs
            .lock()
            .unwrap()
            .pop_front()
            .unwrap_or(ToolOutput::Success {
                content: "ok".into(),
                truncated: false,
            });
        Box::pin(async move { out })
    }
    fn is_always_halt(&self, tool_name: &str) -> bool {
        self.always_halt
            .lock()
            .unwrap()
            .iter()
            .any(|n| n == tool_name)
    }
    fn schemas(&self) -> Vec<ToolSchema> {
        self.schemas.lock().unwrap().clone()
    }
}

/// Tool registry that models a single tool taking one `flag: bool`
/// parameter. Dispatch deserializes `flag` strictly: a real JSON bool
/// succeeds, anything else (e.g. the string `"false"`) returns a
/// recoverable error — exactly the weak-model failure tool-call repair
/// targets. Exposes the tool's schema via [`ToolRegistry::schemas`] so the
/// repair provider can read expected types.
pub struct BoolToolRegistry {
    pub call_count: std::sync::atomic::AtomicUsize,
}
impl Default for BoolToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}
impl BoolToolRegistry {
    pub fn new() -> Self {
        Self {
            call_count: Default::default(),
        }
    }
}
impl ToolRegistry for BoolToolRegistry {
    fn dispatch<'a>(&'a self, call: ToolCall) -> BoxFut<'a, ToolOutput> {
        self.call_count
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let out = match call.input.get("flag") {
            Some(serde_json::Value::Bool(b)) => ToolOutput::Success {
                content: format!("flag={b}"),
                truncated: false,
            },
            other => ToolOutput::Error {
                message: format!("invalid parameters: expected bool `flag`, got {other:?}"),
                recoverable: true,
            },
        };
        Box::pin(async move { out })
    }
    fn schemas(&self) -> Vec<ToolSchema> {
        vec![ToolSchema {
            name: "set_flag".into(),
            description: "set a boolean flag".into(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": { "flag": { "type": "boolean" } },
                "required": ["flag"],
            }),
        }]
    }
}

pub struct AlwaysContinuePolicy;
impl TerminationPolicy for AlwaysContinuePolicy {
    fn evaluate<'a>(
        &'a self,
        _session: &'a SessionState,
        _budget_used: &'a BudgetSnapshot,
    ) -> BoxFut<'a, TerminationDecision> {
        Box::pin(async { TerminationDecision::Continue })
    }
}

pub struct ScriptedTerminationPolicy {
    decisions: Mutex<std::collections::VecDeque<TerminationDecision>>,
}
impl Default for ScriptedTerminationPolicy {
    fn default() -> Self {
        Self::new()
    }
}
impl ScriptedTerminationPolicy {
    pub fn new() -> Self {
        Self {
            decisions: Mutex::new(Default::default()),
        }
    }
    pub fn push(&self, d: TerminationDecision) -> &Self {
        self.decisions.lock().unwrap().push_back(d);
        self
    }
}
impl TerminationPolicy for ScriptedTerminationPolicy {
    fn evaluate<'a>(
        &'a self,
        _session: &'a SessionState,
        _budget_used: &'a BudgetSnapshot,
    ) -> BoxFut<'a, TerminationDecision> {
        let d = self
            .decisions
            .lock()
            .unwrap()
            .pop_front()
            .unwrap_or(TerminationDecision::Continue);
        Box::pin(async move { d })
    }
}

/// Scripted [`MiddlewareChain`] test double (rich surface, issue #11). Decisions
/// are queued per hook via [`push`](ScriptedMiddleware::push); each `fire_*`
/// method pops the front entry if it targets that hook, else returns `Continue`.
/// Unlike the `StandardMiddlewareChain`, scripted decisions are returned RAW (no
/// `validate_decision`), so a test can exercise the harness's defensive handling
/// of an out-of-place decision.
pub struct ScriptedMiddleware {
    decisions: Mutex<std::collections::VecDeque<(HookPoint, MiddlewareDecision)>>,
}
impl Default for ScriptedMiddleware {
    fn default() -> Self {
        Self::new()
    }
}
impl ScriptedMiddleware {
    pub fn new() -> Self {
        Self {
            decisions: Mutex::new(Default::default()),
        }
    }
    pub fn push(&self, h: HookPoint, d: MiddlewareDecision) -> &Self {
        self.decisions.lock().unwrap().push_back((h, d));
        self
    }
    /// Pop and return the scripted decision for `hook` if it is at the front of
    /// the queue, else `Continue`.
    fn next_for(&self, hook: HookPoint) -> MiddlewareDecision {
        let mut q = self.decisions.lock().unwrap();
        match q.front() {
            Some(front) if front.0 == hook => q.pop_front().unwrap().1,
            _ => MiddlewareDecision::Continue,
        }
    }
}
impl MiddlewareChain for ScriptedMiddleware {
    fn register(
        &self,
        _middleware: Box<dyn crate::middleware::Middleware>,
    ) -> Result<(), crate::middleware::MiddlewareError> {
        // The double scripts decisions directly; registration is a no-op.
        Ok(())
    }

    fn fire_before_session<'a>(
        &'a self,
        _task: &'a Task,
        _session_id: &'a SessionId,
    ) -> BoxFut<'a, MiddlewareDecision> {
        let d = self.next_for(HookPoint::BeforeSession);
        Box::pin(async move { d })
    }

    fn fire_before_turn<'a>(
        &'a self,
        _session: &'a mut SessionState,
        _turn_number: u32,
    ) -> BoxFut<'a, MiddlewareDecision> {
        let d = self.next_for(HookPoint::BeforeTurn);
        Box::pin(async move { d })
    }

    fn fire_before_tool<'a>(
        &'a self,
        _calls: &'a mut Vec<ToolCall>,
        _turn_number: u32,
    ) -> BoxFut<'a, MiddlewareDecision> {
        let d = self.next_for(HookPoint::BeforeTool);
        Box::pin(async move { d })
    }

    fn fire_after_tool<'a>(
        &'a self,
        _calls: &'a [ToolCall],
        _results: &'a mut Vec<ToolResult>,
    ) -> BoxFut<'a, MiddlewareDecision> {
        let d = self.next_for(HookPoint::AfterTool);
        Box::pin(async move { d })
    }

    fn fire_before_completion<'a>(
        &'a self,
        _response: &'a str,
        _turn_number: u32,
        _state: &'a SessionState,
    ) -> BoxFut<'a, MiddlewareDecision> {
        let d = self.next_for(HookPoint::BeforeCompletion);
        Box::pin(async move { d })
    }

    fn fire_after_session<'a>(
        &'a self,
        _result: &'a RunResult,
        _session_id: &'a SessionId,
    ) -> BoxFut<'a, ()> {
        // After hooks ignore the decision; drain a scripted AfterSession entry.
        let _ = self.next_for(HookPoint::AfterSession);
        Box::pin(async {})
    }
}
