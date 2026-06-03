//! Web tools (#81, net-new Tier-1 tools): `web_fetch`, `web_search`.
//!
//! Both follow the `http.rs` reqwest-direct pattern (a shared
//! [`reqwest::Client`]). They are `open_world` (external effects) and
//! `read_only`.
//!
//! - [`WebFetchTool`] (`web_fetch`) — GET a URL, return the body text.
//! - [`WebSearchTool`] (`web_search`) — send the query to a configurable search
//!   endpoint and return the structured JSON results verbatim. There is no live
//!   web-search backend in spore-core; the endpoint is injected at construction
//!   so tests drive it against a mock HTTP server (NEVER the live network). The
//!   default endpoint is empty, which yields a recoverable error until a real
//!   backend is configured.
//!
//! ## `web_search` configurability (#108)
//!
//! [`WebSearchTool::with_config`] accepts a [`WebSearchConfig`] so the tool can
//! talk to real search APIs (Brave, Tavily, …) that need GET-style param
//! encoding and/or auth secrets. The original [`WebSearchTool::new`] and
//! [`WebSearchTool::with_endpoint`] constructors and their behavior are
//! **frozen and unchanged**: `with_endpoint` still POSTs `{"query": <q>}` as a
//! JSON body and returns the response body verbatim.
//!
//! ### New public types
//! - [`SearchMethod`] — `Get` | `Post` (default `Post`). NOT a bool.
//! - [`WebSearchConfig`] — endpoint + method + `query_param` + two auth lists.
//! - [`WebSearchConfigError`] — construction-time error from `with_config`.
//!
//! ### `WebSearchConfig` fields
//! - `endpoint: String` — the search backend URL.
//! - `method: SearchMethod` — `Get` or `Post` (default `Post`).
//! - `query_param: String` — the field/param name the query is keyed under
//!   (default `"query"`; Brave uses `"q"`).
//! - `auth_headers: Vec<(String, String)>` — `(header_name, env_var)` pairs.
//!   Each env var is resolved at construction time and attached as an HTTP
//!   header (by `header_name`) on every request, for both GET and POST.
//! - `body_auth_params: Vec<(String, String)>` — `(field_name, env_var)` pairs.
//!   Each env var is resolved at construction time and injected as a secret
//!   request parameter. For POST it goes into the JSON body alongside the query
//!   (e.g. Tavily's `{"api_key": ..., "query": ...}`). For GET it is appended to
//!   the URL query string alongside the query param.
//!
//! ### `with_config` behavior / rules enforced (#108 locked decisions)
//! 1. Auth surface is BOTH header auth (`auth_headers`) and in-body/secret
//!    params (`body_auth_params`).
//! 2. `method` is an enum (`SearchMethod`), default `Post`. `query_param` is
//!    configurable, default `"query"`. GET URL-encodes the query into the query
//!    string under `query_param`; POST keys it in the JSON body under
//!    `query_param` (as today).
//! 3. Response normalization is OUT OF SCOPE — the response body is returned
//!    verbatim through the existing truncation path for both methods.
//! 4. Env sourcing mirrors `AnthropicModelInterface::from_env`: the caller
//!    supplies the env-var NAME; the value is resolved at CONSTRUCTION time; an
//!    unset OR empty env var is a construction error
//!    ([`WebSearchConfigError`]) — a request is NEVER sent with a missing or
//!    empty secret. Resolved secrets are held only in a private, non-serializable
//!    struct whose `Debug` impl redacts them.

use std::sync::OnceLock;

use serde::{Deserialize, Serialize};
use serde_json::json;
use thiserror::Error;

use crate::harness::{BoxFut, SandboxProvider, ToolOutput};
use crate::model::ToolCall;
use crate::tool_registry::{Tool, ToolAnnotations, ToolSchema};
use crate::tools::finish_with_possible_truncation;
use crate::tools::params::{parse_params, WebFetchParams, WebSearchParams};

fn shared_client() -> &'static reqwest::Client {
    static CLIENT: OnceLock<reqwest::Client> = OnceLock::new();
    CLIENT.get_or_init(reqwest::Client::new)
}

// ============================================================================
// WebFetch
// ============================================================================

pub struct WebFetchTool;

impl WebFetchTool {
    pub const NAME: &'static str = "web_fetch";
    pub fn new() -> Self {
        Self
    }
    pub fn schema() -> ToolSchema {
        ToolSchema {
            name: Self::NAME.into(),
            description: "Fetch the contents of a URL".into(),
            parameters: json!({
                "type": "object",
                "properties": {"url": {"type": "string"}},
                "required": ["url"],
            }),
            annotations: ToolAnnotations {
                read_only: true,
                open_world: true,
                ..Default::default()
            },
        }
    }
}

impl Default for WebFetchTool {
    fn default() -> Self {
        Self::new()
    }
}

impl Tool for WebFetchTool {
    fn name(&self) -> &str {
        Self::NAME
    }
    fn may_produce_large_output(&self) -> bool {
        true
    }
    fn execute<'a>(
        &'a self,
        call: &'a ToolCall,
        sandbox: &'a (dyn SandboxProvider + 'a),
        _ctx: &'a crate::tool_registry::ToolContext,
    ) -> BoxFut<'a, ToolOutput> {
        Box::pin(async move {
            let params: WebFetchParams = match parse_params(call) {
                Ok(p) => p,
                Err(e) => return e.into(),
            };
            match shared_client().get(&params.url).send().await {
                Ok(resp) => match resp.text().await {
                    Ok(body) => finish_with_possible_truncation(body, &call.id, sandbox).await,
                    Err(e) => ToolOutput::Error {
                        message: format!("web fetch body read failed: {e}"),
                        recoverable: true,
                    },
                },
                Err(e) => ToolOutput::Error {
                    message: format!("web fetch failed: {e}"),
                    recoverable: true,
                },
            }
        })
    }
}

// ============================================================================
// WebSearch
// ============================================================================

/// HTTP method used to dispatch a web search. An enum (NOT a bool) so the wire
/// shape is explicit and extensible.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SearchMethod {
    /// URL-encode the query (and any body-auth params) into the query string.
    Get,
    /// JSON body keyed by `query_param` (and any body-auth params). Default.
    #[default]
    Post,
}

/// Construction-time configuration for [`WebSearchTool::with_config`].
///
/// Env-var NAMES (not values) are stored here; values are resolved when
/// `with_config` is called. See the module docs for per-field semantics and the
/// rules enforced.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WebSearchConfig {
    /// Search backend URL.
    pub endpoint: String,
    /// GET or POST (default POST).
    #[serde(default)]
    pub method: SearchMethod,
    /// `(header_name, env_var)` pairs — env value attached as an HTTP header.
    #[serde(default)]
    pub auth_headers: Vec<(String, String)>,
    /// Field/param name the query is keyed under (default `"query"`).
    #[serde(default = "default_query_param")]
    pub query_param: String,
    /// `(body_field_name, env_var)` pairs — env value injected into the request
    /// body (POST) or query string (GET) as a secret param.
    #[serde(default)]
    pub body_auth_params: Vec<(String, String)>,
}

fn default_query_param() -> String {
    "query".to_string()
}

impl WebSearchConfig {
    /// Convenience constructor with POST + `"query"` + no auth — equivalent to
    /// the frozen [`WebSearchTool::with_endpoint`] behavior.
    pub fn new(endpoint: impl Into<String>) -> Self {
        Self {
            endpoint: endpoint.into(),
            method: SearchMethod::Post,
            auth_headers: Vec::new(),
            query_param: default_query_param(),
            body_auth_params: Vec::new(),
        }
    }
}

/// Error returned by [`WebSearchTool::with_config`] when a referenced env var is
/// unset or empty. Mirrors the `from_env` precedent: a request is never sent
/// with a missing/empty secret.
#[derive(Debug, Clone, PartialEq, Eq, Error, Serialize, Deserialize)]
#[serde(tag = "kind")]
#[non_exhaustive]
pub enum WebSearchConfigError {
    #[error("env var `{env_var}` not set")]
    EnvVarNotSet { env_var: String },
    #[error("env var `{env_var}` is empty")]
    EnvVarEmpty { env_var: String },
}

/// Resolved (header_name, secret_value) and (field_name, secret_value) pairs.
/// Private + non-serializable; `Debug` redacts the secret values so they never
/// leak into logs or traces.
#[derive(Clone)]
struct ResolvedBackend {
    endpoint: String,
    method: SearchMethod,
    query_param: String,
    auth_headers: Vec<(String, String)>,
    body_auth_params: Vec<(String, String)>,
}

impl std::fmt::Debug for ResolvedBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let redacted_headers: Vec<(&str, &str)> = self
            .auth_headers
            .iter()
            .map(|(name, _)| (name.as_str(), "<redacted>"))
            .collect();
        let redacted_body: Vec<(&str, &str)> = self
            .body_auth_params
            .iter()
            .map(|(name, _)| (name.as_str(), "<redacted>"))
            .collect();
        f.debug_struct("ResolvedBackend")
            .field("endpoint", &self.endpoint)
            .field("method", &self.method)
            .field("query_param", &self.query_param)
            .field("auth_headers", &redacted_headers)
            .field("body_auth_params", &redacted_body)
            .finish()
    }
}

/// Resolve an env var by NAME at construction time. Unset or empty → error.
fn resolve_env(env_var: &str) -> Result<String, WebSearchConfigError> {
    let value = std::env::var(env_var).map_err(|_| WebSearchConfigError::EnvVarNotSet {
        env_var: env_var.to_string(),
    })?;
    if value.trim().is_empty() {
        return Err(WebSearchConfigError::EnvVarEmpty {
            env_var: env_var.to_string(),
        });
    }
    Ok(value)
}

/// Web search tool. The search backend endpoint is injected so tests run
/// against a mock HTTP server. With no endpoint configured (the default), every
/// call is a recoverable error.
#[derive(Debug)]
pub struct WebSearchTool {
    backend: Option<ResolvedBackend>,
}

impl WebSearchTool {
    pub const NAME: &'static str = "web_search";

    /// Construct with no backend configured (calls error until one is set).
    pub fn new() -> Self {
        Self { backend: None }
    }

    /// Construct with a search endpoint (the query is POSTed to it as JSON
    /// `{ "query": ... }`; the response body is returned verbatim).
    ///
    /// FROZEN behavior — kept byte-for-byte compatible with the original tool.
    pub fn with_endpoint(endpoint: impl Into<String>) -> Self {
        Self {
            backend: Some(ResolvedBackend {
                endpoint: endpoint.into(),
                method: SearchMethod::Post,
                query_param: default_query_param(),
                auth_headers: Vec::new(),
                body_auth_params: Vec::new(),
            }),
        }
    }

    /// Construct from a [`WebSearchConfig`], resolving every referenced env var
    /// at construction time. Returns [`WebSearchConfigError`] if any auth env
    /// var is unset or empty — no request is ever attempted in that case.
    pub fn with_config(config: WebSearchConfig) -> Result<Self, WebSearchConfigError> {
        let auth_headers = config
            .auth_headers
            .iter()
            .map(|(name, env_var)| resolve_env(env_var).map(|v| (name.clone(), v)))
            .collect::<Result<Vec<_>, _>>()?;
        let body_auth_params = config
            .body_auth_params
            .iter()
            .map(|(name, env_var)| resolve_env(env_var).map(|v| (name.clone(), v)))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            backend: Some(ResolvedBackend {
                endpoint: config.endpoint,
                method: config.method,
                query_param: config.query_param,
                auth_headers,
                body_auth_params,
            }),
        })
    }

    pub fn schema() -> ToolSchema {
        ToolSchema {
            name: Self::NAME.into(),
            description: "Search the web and return structured results".into(),
            parameters: json!({
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            }),
            annotations: ToolAnnotations {
                read_only: true,
                open_world: true,
                ..Default::default()
            },
        }
    }
}

impl Default for WebSearchTool {
    fn default() -> Self {
        Self::new()
    }
}

impl Tool for WebSearchTool {
    fn name(&self) -> &str {
        Self::NAME
    }
    fn may_produce_large_output(&self) -> bool {
        true
    }
    fn execute<'a>(
        &'a self,
        call: &'a ToolCall,
        sandbox: &'a (dyn SandboxProvider + 'a),
        _ctx: &'a crate::tool_registry::ToolContext,
    ) -> BoxFut<'a, ToolOutput> {
        Box::pin(async move {
            let params: WebSearchParams = match parse_params(call) {
                Ok(p) => p,
                Err(e) => return e.into(),
            };
            let Some(backend) = self.backend.as_ref() else {
                return ToolOutput::Error {
                    message: "web_search backend not configured".into(),
                    recoverable: true,
                };
            };
            let mut req = match backend.method {
                SearchMethod::Get => {
                    // Query + body-auth params are URL-encoded into the query
                    // string; reqwest's `query` encodes spaces, `&`, etc.
                    let mut query: Vec<(&str, &str)> =
                        vec![(backend.query_param.as_str(), params.query.as_str())];
                    for (field, value) in &backend.body_auth_params {
                        query.push((field.as_str(), value.as_str()));
                    }
                    shared_client().get(&backend.endpoint).query(&query)
                }
                SearchMethod::Post => {
                    // Query + body-auth params go into the JSON body (Tavily
                    // shape: {"api_key": ..., "query": ...}).
                    let mut body = serde_json::Map::new();
                    body.insert(
                        backend.query_param.clone(),
                        serde_json::Value::String(params.query.clone()),
                    );
                    for (field, value) in &backend.body_auth_params {
                        body.insert(field.clone(), serde_json::Value::String(value.clone()));
                    }
                    shared_client()
                        .post(&backend.endpoint)
                        .json(&serde_json::Value::Object(body))
                }
            };
            for (name, value) in &backend.auth_headers {
                req = req.header(name.as_str(), value.as_str());
            }
            match req.send().await {
                Ok(resp) => match resp.text().await {
                    Ok(body) => finish_with_possible_truncation(body, &call.id, sandbox).await,
                    Err(e) => ToolOutput::Error {
                        message: format!("web search body read failed: {e}"),
                        recoverable: true,
                    },
                },
                Err(e) => ToolOutput::Error {
                    message: format!("web search failed: {e}"),
                    recoverable: true,
                },
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tool_registry::mock::{test_ctx, AllowAllSandbox};
    use serde_json::json;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn call(name: &str, input: serde_json::Value) -> ToolCall {
        ToolCall {
            id: "c1".into(),
            name: name.into(),
            input,
        }
    }

    #[tokio::test]
    async fn web_fetch_returns_body() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/page"))
            .respond_with(ResponseTemplate::new(200).set_body_string("page-body"))
            .mount(&server)
            .await;
        let sb = AllowAllSandbox;
        let url = format!("{}/page", server.uri());
        let r = WebFetchTool::new()
            .execute(&call("web_fetch", json!({"url": url})), &sb, &test_ctx())
            .await;
        match r {
            ToolOutput::Success { content, .. } => assert_eq!(content, "page-body"),
            other => panic!("{other:?}"),
        }
    }

    #[tokio::test]
    async fn web_search_returns_structured_results_from_mock() {
        let server = MockServer::start().await;
        let results = r#"{"results":[{"title":"t","url":"u"}]}"#;
        Mock::given(method("POST"))
            .and(path("/search"))
            .respond_with(ResponseTemplate::new(200).set_body_string(results))
            .mount(&server)
            .await;
        let sb = AllowAllSandbox;
        let endpoint = format!("{}/search", server.uri());
        let r = WebSearchTool::with_endpoint(endpoint)
            .execute(
                &call("web_search", json!({"query": "rust"})),
                &sb,
                &test_ctx(),
            )
            .await;
        match r {
            ToolOutput::Success { content, .. } => assert_eq!(content, results),
            other => panic!("{other:?}"),
        }
    }

    #[tokio::test]
    async fn web_search_without_backend_is_recoverable_error() {
        let sb = AllowAllSandbox;
        let r = WebSearchTool::new()
            .execute(&call("web_search", json!({"query": "x"})), &sb, &test_ctx())
            .await;
        match r {
            ToolOutput::Error { recoverable, .. } => assert!(recoverable),
            other => panic!("{other:?}"),
        }
    }

    // ── #108: GET / auth headers / in-body auth / env resolution ─────────────
    //
    // Env vars: each test uses a UNIQUE env-var name so parallel test execution
    // never races on a shared name. We set/remove only that test's own name.

    fn success_body(content: &str) -> String {
        content.to_string()
    }

    #[tokio::test]
    async fn get_method_url_encodes_query_into_query_string() {
        use wiremock::matchers::query_param;
        let server = MockServer::start().await;
        // Brave-style: GET with the query under `q`, special chars encoded.
        Mock::given(method("GET"))
            .and(path("/search"))
            .and(query_param("q", "rust & go"))
            .respond_with(ResponseTemplate::new(200).set_body_string("get-results"))
            .mount(&server)
            .await;
        let sb = AllowAllSandbox;
        let cfg = WebSearchConfig {
            endpoint: format!("{}/search", server.uri()),
            method: SearchMethod::Get,
            auth_headers: vec![],
            query_param: "q".into(),
            body_auth_params: vec![],
        };
        let tool = WebSearchTool::with_config(cfg).unwrap();
        let r = tool
            .execute(
                &call("web_search", json!({"query": "rust & go"})),
                &sb,
                &test_ctx(),
            )
            .await;
        match r {
            ToolOutput::Success { content, .. } => assert_eq!(content, "get-results"),
            other => panic!("{other:?}"),
        }
    }

    #[tokio::test]
    async fn auth_header_attached_on_get() {
        use wiremock::matchers::header;
        let env_name = "__SPORE_TEST_BRAVE_KEY_GET__";
        std::env::set_var(env_name, "brave-secret");
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/search"))
            .and(header("X-Subscription-Token", "brave-secret"))
            .respond_with(ResponseTemplate::new(200).set_body_string("ok"))
            .mount(&server)
            .await;
        let sb = AllowAllSandbox;
        let cfg = WebSearchConfig {
            endpoint: format!("{}/search", server.uri()),
            method: SearchMethod::Get,
            auth_headers: vec![("X-Subscription-Token".into(), env_name.into())],
            query_param: "q".into(),
            body_auth_params: vec![],
        };
        let tool = WebSearchTool::with_config(cfg).unwrap();
        let r = tool
            .execute(&call("web_search", json!({"query": "x"})), &sb, &test_ctx())
            .await;
        std::env::remove_var(env_name);
        assert!(matches!(r, ToolOutput::Success { .. }), "{r:?}");
    }

    #[tokio::test]
    async fn auth_header_attached_on_post() {
        use wiremock::matchers::header;
        let env_name = "__SPORE_TEST_BRAVE_KEY_POST__";
        std::env::set_var(env_name, "brave-secret");
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/search"))
            .and(header("X-Subscription-Token", "brave-secret"))
            .respond_with(ResponseTemplate::new(200).set_body_string("ok"))
            .mount(&server)
            .await;
        let sb = AllowAllSandbox;
        let cfg = WebSearchConfig {
            endpoint: format!("{}/search", server.uri()),
            method: SearchMethod::Post,
            auth_headers: vec![("X-Subscription-Token".into(), env_name.into())],
            query_param: "query".into(),
            body_auth_params: vec![],
        };
        let tool = WebSearchTool::with_config(cfg).unwrap();
        let r = tool
            .execute(&call("web_search", json!({"query": "x"})), &sb, &test_ctx())
            .await;
        std::env::remove_var(env_name);
        assert!(matches!(r, ToolOutput::Success { .. }), "{r:?}");
    }

    #[tokio::test]
    async fn multiple_auth_headers_all_attached() {
        use wiremock::matchers::header;
        let a = "__SPORE_TEST_MULTI_A__";
        let b = "__SPORE_TEST_MULTI_B__";
        std::env::set_var(a, "aaa");
        std::env::set_var(b, "bbb");
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/search"))
            .and(header("X-A", "aaa"))
            .and(header("X-B", "bbb"))
            .respond_with(ResponseTemplate::new(200).set_body_string("ok"))
            .mount(&server)
            .await;
        let sb = AllowAllSandbox;
        let cfg = WebSearchConfig {
            endpoint: format!("{}/search", server.uri()),
            method: SearchMethod::Post,
            auth_headers: vec![("X-A".into(), a.into()), ("X-B".into(), b.into())],
            query_param: "query".into(),
            body_auth_params: vec![],
        };
        let tool = WebSearchTool::with_config(cfg).unwrap();
        let r = tool
            .execute(&call("web_search", json!({"query": "x"})), &sb, &test_ctx())
            .await;
        std::env::remove_var(a);
        std::env::remove_var(b);
        assert!(matches!(r, ToolOutput::Success { .. }), "{r:?}");
    }

    #[tokio::test]
    async fn in_body_auth_param_tavily_shape() {
        use wiremock::matchers::body_json;
        let env_name = "__SPORE_TEST_TAVILY_KEY__";
        std::env::set_var(env_name, "tav-secret");
        let server = MockServer::start().await;
        // Tavily POST body: {"api_key": <secret>, "query": <q>}.
        Mock::given(method("POST"))
            .and(path("/search"))
            .and(body_json(json!({"api_key": "tav-secret", "query": "rust"})))
            .respond_with(ResponseTemplate::new(200).set_body_string("tavily-results"))
            .mount(&server)
            .await;
        let sb = AllowAllSandbox;
        let cfg = WebSearchConfig {
            endpoint: format!("{}/search", server.uri()),
            method: SearchMethod::Post,
            auth_headers: vec![],
            query_param: "query".into(),
            body_auth_params: vec![("api_key".into(), env_name.into())],
        };
        let tool = WebSearchTool::with_config(cfg).unwrap();
        let r = tool
            .execute(
                &call("web_search", json!({"query": "rust"})),
                &sb,
                &test_ctx(),
            )
            .await;
        std::env::remove_var(env_name);
        match r {
            ToolOutput::Success { content, .. } => assert_eq!(content, "tavily-results"),
            other => panic!("{other:?}"),
        }
    }

    #[tokio::test]
    async fn missing_env_var_is_construction_error_no_request() {
        let env_name = "__SPORE_TEST_WEB_MISSING__";
        std::env::remove_var(env_name);
        // No mock server / endpoint: if a request were attempted it would fail
        // loudly, but we assert construction errors out first.
        let cfg = WebSearchConfig {
            endpoint: "http://127.0.0.1:1/never".into(),
            method: SearchMethod::Post,
            auth_headers: vec![("X-Key".into(), env_name.into())],
            query_param: "query".into(),
            body_auth_params: vec![],
        };
        let err = WebSearchTool::with_config(cfg).unwrap_err();
        assert!(
            matches!(err, WebSearchConfigError::EnvVarNotSet { ref env_var } if env_var == env_name),
            "{err:?}"
        );
    }

    #[tokio::test]
    async fn empty_env_var_is_construction_error() {
        let env_name = "__SPORE_TEST_WEB_EMPTY__";
        std::env::set_var(env_name, "   ");
        let cfg = WebSearchConfig {
            endpoint: "http://127.0.0.1:1/never".into(),
            method: SearchMethod::Post,
            auth_headers: vec![],
            query_param: "query".into(),
            body_auth_params: vec![("api_key".into(), env_name.into())],
        };
        let err = WebSearchTool::with_config(cfg).unwrap_err();
        std::env::remove_var(env_name);
        assert!(
            matches!(err, WebSearchConfigError::EnvVarEmpty { ref env_var } if env_var == env_name),
            "{err:?}"
        );
    }

    #[tokio::test]
    async fn no_auth_post_carries_only_content_type() {
        use wiremock::matchers::header_exists;
        let server = MockServer::start().await;
        // Content-Type present (reqwest .json sets it); assert success with no
        // auth configured. We can't easily assert absence of arbitrary headers,
        // so we verify the request still matches a Content-Type-only expectation.
        Mock::given(method("POST"))
            .and(path("/search"))
            .and(header_exists("content-type"))
            .respond_with(ResponseTemplate::new(200).set_body_string("ok"))
            .mount(&server)
            .await;
        let sb = AllowAllSandbox;
        let cfg = WebSearchConfig {
            endpoint: format!("{}/search", server.uri()),
            method: SearchMethod::Post,
            auth_headers: vec![],
            query_param: "query".into(),
            body_auth_params: vec![],
        };
        let tool = WebSearchTool::with_config(cfg).unwrap();
        let r = tool
            .execute(&call("web_search", json!({"query": "x"})), &sb, &test_ctx())
            .await;
        assert!(matches!(r, ToolOutput::Success { .. }), "{r:?}");
    }

    #[tokio::test]
    async fn post_default_query_shape_unchanged_via_config() {
        use wiremock::matchers::body_json;
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/search"))
            .and(body_json(json!({"query": "rust"})))
            .respond_with(ResponseTemplate::new(200).set_body_string(success_body("res")))
            .mount(&server)
            .await;
        let sb = AllowAllSandbox;
        let tool =
            WebSearchTool::with_config(WebSearchConfig::new(format!("{}/search", server.uri())))
                .unwrap();
        let r = tool
            .execute(
                &call("web_search", json!({"query": "rust"})),
                &sb,
                &test_ctx(),
            )
            .await;
        match r {
            ToolOutput::Success { content, .. } => assert_eq!(content, "res"),
            other => panic!("{other:?}"),
        }
    }

    #[tokio::test]
    async fn get_returns_body_verbatim() {
        let server = MockServer::start().await;
        let raw = r#"{"web":{"results":[{"title":"t"}]}}"#;
        Mock::given(method("GET"))
            .and(path("/search"))
            .respond_with(ResponseTemplate::new(200).set_body_string(raw))
            .mount(&server)
            .await;
        let sb = AllowAllSandbox;
        let cfg = WebSearchConfig {
            endpoint: format!("{}/search", server.uri()),
            method: SearchMethod::Get,
            auth_headers: vec![],
            query_param: "q".into(),
            body_auth_params: vec![],
        };
        let tool = WebSearchTool::with_config(cfg).unwrap();
        let r = tool
            .execute(&call("web_search", json!({"query": "t"})), &sb, &test_ctx())
            .await;
        match r {
            ToolOutput::Success { content, .. } => assert_eq!(content, raw),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn search_method_default_is_post() {
        assert_eq!(SearchMethod::default(), SearchMethod::Post);
    }

    #[tokio::test]
    async fn web_fetch_bad_url_is_recoverable() {
        let sb = AllowAllSandbox;
        let r = WebFetchTool::new()
            .execute(
                &call("web_fetch", json!({"url": "not-a-url://////"})),
                &sb,
                &test_ctx(),
            )
            .await;
        match r {
            ToolOutput::Error { recoverable, .. } => assert!(recoverable),
            other => panic!("{other:?}"),
        }
    }
}
