//! HTTP tools: HttpGet, HttpPost.

use std::sync::OnceLock;

use serde_json::{json, Value};

use crate::harness::{BoxFut, SandboxProvider, ToolOutput};
use crate::model::ToolCall;
use crate::tool_registry::{Tool, ToolAnnotations, ToolSchema};
use crate::tools::finish_with_possible_truncation;
use crate::tools::params::{parse_params, HttpGetParams, HttpPostParams};

fn shared_client() -> &'static reqwest::Client {
    static CLIENT: OnceLock<reqwest::Client> = OnceLock::new();
    CLIENT.get_or_init(reqwest::Client::new)
}

fn apply_headers(
    mut req: reqwest::RequestBuilder,
    headers: Option<&serde_json::Map<String, Value>>,
) -> reqwest::RequestBuilder {
    if let Some(map) = headers {
        for (k, v) in map {
            if let Some(s) = v.as_str() {
                req = req.header(k, s);
            }
        }
    }
    req
}

// ============================================================================
// HttpGet
// ============================================================================

pub struct HttpGetTool;

impl HttpGetTool {
    pub const NAME: &'static str = "http_get";
    pub fn new() -> Self {
        Self
    }
    pub fn schema() -> ToolSchema {
        ToolSchema {
            name: Self::NAME.into(),
            description: "Perform an HTTP GET".into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                    "headers": {"type": "object"},
                },
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

impl Default for HttpGetTool {
    fn default() -> Self {
        Self::new()
    }
}

impl Tool for HttpGetTool {
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
            let params: HttpGetParams = match parse_params(call) {
                Ok(p) => p,
                Err(e) => return e.into(),
            };
            let req = shared_client().get(&params.url);
            let req = apply_headers(req, params.headers.as_ref());
            match req.send().await {
                Ok(resp) => match resp.text().await {
                    Ok(body) => finish_with_possible_truncation(body, &call.id, sandbox).await,
                    Err(e) => ToolOutput::Error {
                        message: format!("http body read failed: {e}"),
                        recoverable: true,
                    },
                },
                Err(e) => ToolOutput::Error {
                    message: format!("http get failed: {e}"),
                    recoverable: true,
                },
            }
        })
    }
}

// ============================================================================
// HttpPost
// ============================================================================

pub struct HttpPostTool;

impl HttpPostTool {
    pub const NAME: &'static str = "http_post";
    pub fn new() -> Self {
        Self
    }
    pub fn schema() -> ToolSchema {
        ToolSchema {
            name: Self::NAME.into(),
            description: "Perform an HTTP POST with a JSON body".into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                    "body": {},
                    "headers": {"type": "object"},
                },
                "required": ["url", "body"],
            }),
            annotations: ToolAnnotations {
                open_world: true,
                ..Default::default()
            },
        }
    }
}

impl Default for HttpPostTool {
    fn default() -> Self {
        Self::new()
    }
}

impl Tool for HttpPostTool {
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
            let params: HttpPostParams = match parse_params(call) {
                Ok(p) => p,
                Err(e) => return e.into(),
            };
            let has_content_type = params
                .headers
                .as_ref()
                .map(|m| m.keys().any(|k| k.eq_ignore_ascii_case("content-type")))
                .unwrap_or(false);
            let mut req = shared_client().post(&params.url).json(&params.body);
            if !has_content_type {
                req = req.header("content-type", "application/json");
            }
            req = apply_headers(req, params.headers.as_ref());
            match req.send().await {
                Ok(resp) => match resp.text().await {
                    Ok(body) => finish_with_possible_truncation(body, &call.id, sandbox).await,
                    Err(e) => ToolOutput::Error {
                        message: format!("http body read failed: {e}"),
                        recoverable: true,
                    },
                },
                Err(e) => ToolOutput::Error {
                    message: format!("http post failed: {e}"),
                    recoverable: true,
                },
            }
        })
    }
}

// ============================================================================
// Tests
// ============================================================================

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
    async fn http_get_returns_body() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/hello"))
            .respond_with(ResponseTemplate::new(200).set_body_string("world"))
            .mount(&server)
            .await;
        let sb = AllowAllSandbox;
        let url = format!("{}/hello", server.uri());
        let r = HttpGetTool::new()
            .execute(&call("http_get", json!({"url": url})), &sb, &test_ctx())
            .await;
        match r {
            ToolOutput::Success { content, .. } => assert_eq!(content, "world"),
            other => panic!("{other:?}"),
        }
    }

    #[tokio::test]
    async fn http_post_sends_json_body() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/echo"))
            .respond_with(ResponseTemplate::new(200).set_body_string("ok"))
            .mount(&server)
            .await;
        let sb = AllowAllSandbox;
        let url = format!("{}/echo", server.uri());
        let r = HttpPostTool::new()
            .execute(
                &call("http_post", json!({"url": url, "body": {"x": 1}})),
                &sb,
                &test_ctx(),
            )
            .await;
        match r {
            ToolOutput::Success { content, .. } => assert_eq!(content, "ok"),
            other => panic!("{other:?}"),
        }
    }

    #[tokio::test]
    async fn http_get_invalid_url_is_recoverable() {
        let sb = AllowAllSandbox;
        let r = HttpGetTool::new()
            .execute(
                &call("http_get", json!({"url": "not-a-url://////"})),
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
