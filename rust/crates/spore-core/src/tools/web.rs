//! Web tools (#81, net-new Tier-1 tools): `web_fetch`, `web_search`.
//!
//! Both follow the `http.rs` reqwest-direct pattern (a shared
//! [`reqwest::Client`]). They are `open_world` (external effects) and
//! `read_only`.
//!
//! - [`WebFetchTool`] (`web_fetch`) — GET a URL, return the body text.
//! - [`WebSearchTool`] (`web_search`) — POST the query to a configurable search
//!   endpoint and return the structured JSON results verbatim. There is no live
//!   web-search backend in spore-core; the endpoint is injected at construction
//!   so tests drive it against a mock HTTP server (NEVER the live network). The
//!   default endpoint is empty, which yields a recoverable error until a real
//!   backend is configured.

use std::sync::OnceLock;

use serde_json::json;

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

/// Web search tool. The search backend endpoint is injected so tests run
/// against a mock HTTP server. With no endpoint configured (the default), every
/// call is a recoverable error.
pub struct WebSearchTool {
    endpoint: Option<String>,
}

impl WebSearchTool {
    pub const NAME: &'static str = "web_search";

    /// Construct with no backend configured (calls error until one is set).
    pub fn new() -> Self {
        Self { endpoint: None }
    }

    /// Construct with a search endpoint (the query is POSTed to it as JSON
    /// `{ "query": ... }`; the response body is returned verbatim).
    pub fn with_endpoint(endpoint: impl Into<String>) -> Self {
        Self {
            endpoint: Some(endpoint.into()),
        }
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
            let Some(endpoint) = self.endpoint.as_ref() else {
                return ToolOutput::Error {
                    message: "web_search backend not configured".into(),
                    recoverable: true,
                };
            };
            let req = shared_client()
                .post(endpoint)
                .json(&json!({"query": params.query}));
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
