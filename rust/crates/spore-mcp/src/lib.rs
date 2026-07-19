//! `spore-mcp` — an MCP (Model Context Protocol) client that surfaces a remote
//! MCP server's tools as native spore-core [`Tool`]s.
//!
//! spore-core's tool layer already anticipates this: every tool is a `Tool`
//! impl, the registry advertises one [`ToolNamespace`] per logical group, and
//! its own docs note "for an MCP future, one server maps naturally to one
//! namespace." This crate fills that future in. It quarantines the `rmcp`
//! dependency (the official Rust MCP SDK) behind the existing trait — spore-core
//! itself gains nothing new.
//!
//! ## Shape
//! - [`McpServerConfig`] / [`McpTransport`] — how to reach a server (stdio child
//!   process, or streamable-HTTP with custom headers).
//! - [`McpManager`] — connect-on-first-use, cache, reconnect-if-closed,
//!   [`probe`](McpManager::probe), [`disconnect`](McpManager::disconnect).
//! - [`McpManager::tools_for`] returns one [`StandardTool`] per remote tool,
//!   ready for [`HarnessBuilder::tool`](spore_core::HarnessBuilder)/`tools`.
//!
//! ```no_run
//! # async fn demo() -> Result<(), spore_mcp::McpError> {
//! use spore_mcp::{McpManager, McpServerConfig};
//!
//! let manager = McpManager::new();
//! let config = McpServerConfig::stdio("files", "mcp-server-filesystem")
//!     .arg("/some/dir");
//! let tools = manager.tools_for(&config).await?;
//! // for st in tools { harness_builder = harness_builder.tool(st); }
//! # let _ = tools; Ok(())
//! # }
//! ```

use std::collections::HashMap;

use rmcp::model::{CallToolRequestParams, CallToolResult, RawContent, Tool as McpTool};
use rmcp::service::{Peer, RoleClient, RunningService};
use rmcp::transport::streamable_http_client::StreamableHttpClientTransportConfig;
use rmcp::transport::{StreamableHttpClientTransport, TokioChildProcess};
use rmcp::ServiceExt;
use thiserror::Error;

use spore_core::model::ToolCall;
use spore_core::tool_registry::{Tool, ToolAnnotations, ToolSchema};
use spore_core::{BoxFut, SandboxProvider, StandardTool, ToolContext, ToolNamespace, ToolOutput};

// ============================================================================
// Errors
// ============================================================================

#[derive(Debug, Error)]
pub enum McpError {
    /// Failed to reach / handshake with the server (spawn failure, bad URL,
    /// transport drop).
    #[error("MCP connection error: {message}")]
    Connection { message: String },

    /// The server returned a protocol-level error (`tools/list`, `tools/call`).
    #[error("MCP protocol error: {message}")]
    Protocol { message: String },
}

// ============================================================================
// Server configuration
// ============================================================================

/// How to reach an MCP server.
#[derive(Debug, Clone)]
pub enum McpTransport {
    /// Spawn a child process and speak MCP over its stdio.
    Stdio {
        command: String,
        args: Vec<String>,
        env: Vec<(String, String)>,
    },
    /// Connect to a streamable-HTTP MCP endpoint with optional custom headers
    /// (e.g. `Authorization`).
    Http {
        url: String,
        headers: Vec<(String, String)>,
    },
}

/// A named MCP server: a stable `name` (the cache key / namespace handle) plus
/// its [`McpTransport`].
#[derive(Debug, Clone)]
pub struct McpServerConfig {
    pub name: String,
    pub transport: McpTransport,
}

impl McpServerConfig {
    /// A stdio server launched by running `command`.
    pub fn stdio(name: impl Into<String>, command: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            transport: McpTransport::Stdio {
                command: command.into(),
                args: Vec::new(),
                env: Vec::new(),
            },
        }
    }

    /// A streamable-HTTP server at `url`.
    pub fn http(name: impl Into<String>, url: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            transport: McpTransport::Http {
                url: url.into(),
                headers: Vec::new(),
            },
        }
    }

    /// Append a stdio command argument (ignored for HTTP transports).
    pub fn arg(mut self, arg: impl Into<String>) -> Self {
        if let McpTransport::Stdio { args, .. } = &mut self.transport {
            args.push(arg.into());
        }
        self
    }

    /// Append several stdio command arguments.
    pub fn args<I, S>(mut self, new_args: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        if let McpTransport::Stdio { args, .. } = &mut self.transport {
            args.extend(new_args.into_iter().map(Into::into));
        }
        self
    }

    /// Set a stdio environment variable.
    pub fn env(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        if let McpTransport::Stdio { env, .. } = &mut self.transport {
            env.push((key.into(), value.into()));
        }
        self
    }

    /// Set an HTTP request header (ignored for stdio transports).
    pub fn header(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        if let McpTransport::Http { headers, .. } = &mut self.transport {
            headers.push((key.into(), value.into()));
        }
        self
    }
}

// ============================================================================
// Connection
// ============================================================================

/// Open a fresh connection to the server, returning the live `RunningService`
/// whose `peer()` issues `tools/list` and `tools/call`.
async fn connect(config: &McpServerConfig) -> Result<RunningService<RoleClient, ()>, McpError> {
    match &config.transport {
        McpTransport::Stdio { command, args, env } => {
            let mut cmd = tokio::process::Command::new(command);
            cmd.args(args);
            for (k, v) in env {
                cmd.env(k, v);
            }
            let transport = TokioChildProcess::new(cmd).map_err(|e| McpError::Connection {
                message: format!("spawning `{command}` failed: {e}"),
            })?;
            ().serve(transport).await.map_err(|e| McpError::Connection {
                message: format!("handshake with `{}` failed: {e}", config.name),
            })
        }
        McpTransport::Http { url, headers } => {
            let mut cfg = StreamableHttpClientTransportConfig::with_uri(url.clone());
            if !headers.is_empty() {
                let mut map: HashMap<http::HeaderName, http::HeaderValue> = HashMap::new();
                for (k, v) in headers {
                    let name = http::HeaderName::from_bytes(k.as_bytes()).map_err(|e| {
                        McpError::Connection {
                            message: format!("invalid header name {k:?}: {e}"),
                        }
                    })?;
                    let value =
                        http::HeaderValue::from_str(v).map_err(|e| McpError::Connection {
                            message: format!("invalid header value for {k:?}: {e}"),
                        })?;
                    map.insert(name, value);
                }
                cfg.custom_headers = map;
            }
            let transport = StreamableHttpClientTransport::from_config(cfg);
            ().serve(transport).await.map_err(|e| McpError::Connection {
                message: format!("connecting to `{}` failed: {e}", config.name),
            })
        }
    }
}

// ============================================================================
// Manager
// ============================================================================

/// Caches one live connection per server `name`, connecting on first use and
/// transparently reconnecting if a cached connection has closed. Cheap to share
/// behind an `Arc`.
#[derive(Default)]
pub struct McpManager {
    connections: tokio::sync::Mutex<HashMap<String, RunningService<RoleClient, ()>>>,
}

impl McpManager {
    pub fn new() -> Self {
        Self::default()
    }

    /// Return a `peer` for `config`, (re)connecting if there is no live cached
    /// connection.
    async fn peer_for(&self, config: &McpServerConfig) -> Result<Peer<RoleClient>, McpError> {
        let mut conns = self.connections.lock().await;
        let stale = match conns.get(&config.name) {
            Some(svc) => svc.is_transport_closed(),
            None => true,
        };
        if stale {
            if let Some(old) = conns.remove(&config.name) {
                let _ = old.cancel().await;
            }
            let svc = connect(config).await?;
            conns.insert(config.name.clone(), svc);
        }
        // Present by construction above.
        Ok(conns
            .get(&config.name)
            .expect("connection just inserted")
            .peer()
            .clone())
    }

    /// List the server's tools and adapt each into a [`StandardTool`] ready to
    /// register on a harness. Connects (and caches) on first use.
    pub async fn tools_for(&self, config: &McpServerConfig) -> Result<Vec<StandardTool>, McpError> {
        let peer = self.peer_for(config).await?;
        let tools = list_tools(&peer).await?;
        Ok(tools
            .into_iter()
            .map(|t| build_standard_tool(t, peer.clone()))
            .collect())
    }

    /// Like [`tools_for`](Self::tools_for) but also returns a [`ToolNamespace`]
    /// grouping the server's tools under one header — the "one server → one
    /// namespace" pattern for keeping a large server's tools deferred behind a
    /// [`ToolSearchRegistry`](spore_core::ToolSearchRegistry).
    pub async fn tools_for_namespaced(
        &self,
        config: &McpServerConfig,
    ) -> Result<(Vec<StandardTool>, ToolNamespace), McpError> {
        let tools = self.tools_for(config).await?;
        let namespace = ToolNamespace::new(
            config.name.clone(),
            format!("Tools from the `{}` MCP server", config.name),
            tools.iter().map(|t| t.schema.name.clone()),
        );
        Ok((tools, namespace))
    }

    /// One-shot connect → `tools/list` → disconnect, returning the tool names.
    /// For a "test connection" affordance; does not touch the cache.
    pub async fn probe(&self, config: &McpServerConfig) -> Result<Vec<String>, McpError> {
        let svc = connect(config).await?;
        let result = list_tools(svc.peer()).await;
        let _ = svc.cancel().await;
        Ok(result?.into_iter().map(|t| t.name.to_string()).collect())
    }

    /// Drop and cancel the cached connection for `name` (kills a stdio child).
    /// No-op if not connected.
    pub async fn disconnect(&self, name: &str) {
        let mut conns = self.connections.lock().await;
        if let Some(svc) = conns.remove(name) {
            let _ = svc.cancel().await;
        }
    }
}

async fn list_tools(peer: &Peer<RoleClient>) -> Result<Vec<McpTool>, McpError> {
    peer.list_all_tools().await.map_err(|e| McpError::Protocol {
        message: format!("tools/list failed: {e}"),
    })
}

// ============================================================================
// Adapter (rmcp tool → spore_core::Tool)
// ============================================================================

fn build_standard_tool(tool: McpTool, peer: Peer<RoleClient>) -> StandardTool {
    let schema = tool_to_schema(&tool);
    let adapter = McpToolAdapter {
        tool_name: tool.name.to_string(),
        peer,
    };
    StandardTool::new(Box::new(adapter), schema)
}

/// Project an `rmcp` tool description onto spore-core's registry schema.
fn tool_to_schema(tool: &McpTool) -> ToolSchema {
    ToolSchema {
        name: tool.name.to_string(),
        description: tool
            .description
            .as_ref()
            .map(|d| d.to_string())
            .unwrap_or_default(),
        parameters: serde_json::Value::Object((*tool.input_schema).clone()),
        annotations: annotations_from(tool.annotations.as_ref()),
    }
}

/// Map MCP's tool-annotation *hints* onto spore-core's [`ToolAnnotations`].
/// MCP's documented defaults are `read_only=false`, `open_world=true`.
fn annotations_from(a: Option<&rmcp::model::ToolAnnotations>) -> ToolAnnotations {
    match a {
        Some(a) => ToolAnnotations {
            read_only: a.read_only_hint.unwrap_or(false),
            destructive: a.destructive_hint.unwrap_or(false),
            idempotent: a.idempotent_hint.unwrap_or(false),
            open_world: a.open_world_hint.unwrap_or(true),
        },
        None => ToolAnnotations {
            open_world: true,
            ..Default::default()
        },
    }
}

struct McpToolAdapter {
    tool_name: String,
    peer: Peer<RoleClient>,
}

impl Tool for McpToolAdapter {
    fn name(&self) -> &str {
        &self.tool_name
    }

    fn may_produce_large_output(&self) -> bool {
        // A remote tool can return arbitrarily large content; let the harness
        // route it through `handle_large_output`.
        true
    }

    fn execute<'a>(
        &'a self,
        call: &'a ToolCall,
        _sandbox: &'a (dyn SandboxProvider + 'a),
        _ctx: &'a ToolContext,
    ) -> BoxFut<'a, ToolOutput> {
        Box::pin(async move {
            let arguments = match &call.input {
                serde_json::Value::Object(map) => Some(map.clone()),
                // MCP `tools/call` arguments are an object (or absent).
                _ => None,
            };
            // `CallToolRequestParams` is `#[non_exhaustive]`, so build it through
            // its `Deserialize` impl rather than a struct literal — version-robust.
            let params: CallToolRequestParams = match serde_json::from_value(serde_json::json!({
                "name": self.tool_name,
                "arguments": arguments,
            })) {
                Ok(params) => params,
                Err(e) => {
                    return ToolOutput::error(format!(
                        "could not build MCP call for `{}`: {e}",
                        self.tool_name
                    ))
                }
            };
            match self.peer.call_tool(params).await {
                Ok(result) => result_to_output(result),
                Err(e) => {
                    ToolOutput::error(format!("MCP tool `{}` call failed: {e}", self.tool_name))
                }
            }
        })
    }
}

/// Convert a tool-call result into a [`ToolOutput`]. A server-flagged
/// `is_error` becomes a RECOVERABLE [`ToolOutput::error`] — the model sees the
/// failure text and can adapt, exactly like a local tool's recoverable error.
fn result_to_output(result: CallToolResult) -> ToolOutput {
    let is_error = result.is_error.unwrap_or(false);
    let text = extract_text(&result);
    if is_error {
        ToolOutput::error(text)
    } else {
        ToolOutput::success(text)
    }
}

/// Flatten an MCP result's content blocks into a single string: text blocks
/// joined by newlines, falling back to the structured-content JSON when there
/// is no text (e.g. a tool that returns only structured output).
fn extract_text(result: &CallToolResult) -> String {
    let mut parts = Vec::new();
    for content in result.content.iter() {
        if let RawContent::Text(t) = &content.raw {
            parts.push(t.text.clone());
        }
    }
    if parts.is_empty() {
        if let Some(structured) = &result.structured_content {
            return structured.to_string();
        }
    }
    parts.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    use rmcp::model::{Content, Tool as McpTool};
    use std::sync::Arc;

    #[test]
    fn config_builders() {
        let stdio = McpServerConfig::stdio("files", "server-fs")
            .arg("/data")
            .env("TOKEN", "x");
        match &stdio.transport {
            McpTransport::Stdio { command, args, env } => {
                assert_eq!(command, "server-fs");
                assert_eq!(args, &["/data"]);
                assert_eq!(env, &[("TOKEN".to_string(), "x".to_string())]);
            }
            _ => panic!("expected stdio"),
        }

        let http = McpServerConfig::http("remote", "https://mcp.example/v1")
            .header("Authorization", "Bearer t");
        match &http.transport {
            McpTransport::Http { url, headers } => {
                assert_eq!(url, "https://mcp.example/v1");
                assert_eq!(
                    headers,
                    &[("Authorization".to_string(), "Bearer t".to_string())]
                );
            }
            _ => panic!("expected http"),
        }
    }

    #[test]
    fn schema_projection() {
        let input_schema = Arc::new(
            serde_json::json!({
                "type": "object",
                "properties": { "q": { "type": "string" } }
            })
            .as_object()
            .unwrap()
            .clone(),
        );
        let tool = McpTool::new("search", "Search the web", input_schema);
        let schema = tool_to_schema(&tool);
        assert_eq!(schema.name, "search");
        assert_eq!(schema.description, "Search the web");
        assert_eq!(schema.parameters["properties"]["q"]["type"], "string");
        // No annotations on the wire → spore default (open-world external tool).
        assert!(schema.annotations.open_world);
        assert!(!schema.annotations.read_only);
    }

    #[test]
    fn annotations_default_when_absent() {
        let a = annotations_from(None);
        assert!(a.open_world);
        assert!(!a.read_only);
        assert!(!a.destructive);
        assert!(!a.idempotent);
    }

    #[test]
    fn success_result_maps_to_success_output() {
        let result = CallToolResult::success(vec![Content::text("hello"), Content::text("world")]);
        match result_to_output(result) {
            ToolOutput::Success { content, .. } => assert_eq!(content, "hello\nworld"),
            other => panic!("expected Success, got {other:?}"),
        }
    }

    #[test]
    fn error_result_maps_to_recoverable_error() {
        let result = CallToolResult::error(vec![Content::text("boom")]);
        match result_to_output(result) {
            ToolOutput::Error {
                message,
                recoverable,
            } => {
                assert_eq!(message, "boom");
                assert!(recoverable);
            }
            other => panic!("expected Error, got {other:?}"),
        }
    }

    #[test]
    fn manager_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<McpManager>();
    }
}
