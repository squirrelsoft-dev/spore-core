//! `EmbeddingModel` — boundary between the RAG layer and an embedding provider.
//!
//! Mirrors [`ModelInterface`](crate::model::ModelInterface): object-safe via
//! [`BoxFut`](crate::harness::BoxFut), a blanket `Arc` impl so an
//! `Arc<dyn EmbeddingModel>` is itself an `EmbeddingModel`, and provider
//! implementations that reuse the existing `reqwest` client and shared retry
//! plumbing. The trait deals ONLY in text → vector; storing and searching the
//! resulting vectors is [`VectorStore`](crate::vector_store::VectorStore).
//!
//! Two providers ship in-core because they add no new dependency — they reuse
//! the existing `reqwest` client (and, for OpenAI, the shared
//! [`provider_http`](crate::provider_http) retry loop):
//!   - [`OllamaEmbeddingModel`] — POST `/api/embed` (captures the vector the
//!     chat client's `count_tokens` path deliberately discards).
//!   - [`OpenAIEmbeddingModel`] — POST `${base_url}/embeddings`.
//!
//! Both report a fixed [`dimensions`](EmbeddingModel::dimensions). Ollama does
//! not advertise vector length out-of-band, so a caller that does not know it
//! statically can discover it once via
//! [`OllamaEmbeddingModel::probe_dimensions`] and pin it with `with_dimensions`
//! before sizing a vector store.

use std::sync::Arc;
use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::harness::BoxFut;
use crate::model::ModelError;

// ============================================================================
// Trait + identity
// ============================================================================

/// Identity of an embedding provider/model, for routing and diagnostics.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EmbeddingInfo {
    /// Provider name, e.g. `"ollama"` / `"openai"`.
    pub provider: String,
    /// Provider model id, e.g. `"nomic-embed-text"` / `"text-embedding-3-small"`.
    pub model_id: String,
    /// Length of the vectors this model produces (`0` when not yet known).
    pub dimensions: usize,
}

/// Turns text into embedding vectors. Object-safe (`BoxFut`, not RPITIT) so a
/// `Box<dyn EmbeddingModel>` / `Arc<dyn EmbeddingModel>` can be injected
/// wherever an embedder is expected.
pub trait EmbeddingModel: Send + Sync {
    /// Embed a batch of inputs, returning one vector per input IN INPUT ORDER.
    /// An empty `inputs` returns an empty `Vec` without contacting the provider.
    fn embed<'a>(&'a self, inputs: Vec<String>) -> BoxFut<'a, Result<Vec<Vec<f32>>, ModelError>>;

    /// Convenience: embed a single input. Defaults to a one-element
    /// [`embed`](Self::embed) call.
    fn embed_one<'a>(&'a self, input: String) -> BoxFut<'a, Result<Vec<f32>, ModelError>> {
        Box::pin(async move {
            let mut vectors = self.embed(vec![input]).await?;
            vectors.pop().ok_or_else(|| ModelError::ProviderError {
                code: 0,
                message: "embedding provider returned no vectors".into(),
            })
        })
    }

    /// Vector length this model produces. `0` means "not yet known" — see the
    /// module header on discovering it for Ollama.
    fn dimensions(&self) -> usize;

    /// Provider identity for tracing/routing.
    fn info(&self) -> EmbeddingInfo;
}

/// Blanket impl so an `Arc`-wrapped embedder is itself an `EmbeddingModel`
/// (mirrors the `ModelInterface` blanket impl). `?Sized` covers the `dyn` case.
impl<T: EmbeddingModel + ?Sized> EmbeddingModel for Arc<T> {
    fn embed<'a>(&'a self, inputs: Vec<String>) -> BoxFut<'a, Result<Vec<Vec<f32>>, ModelError>> {
        (**self).embed(inputs)
    }

    fn dimensions(&self) -> usize {
        (**self).dimensions()
    }

    fn info(&self) -> EmbeddingInfo {
        (**self).info()
    }
}

/// Cosine similarity of two equal-length vectors in `[-1, 1]` (higher = closer).
/// Returns `0.0` when either vector has zero magnitude or lengths differ — a
/// degenerate input is treated as "no similarity" rather than `NaN`/panic. Used
/// by in-memory vector stores and as the metric most providers' vectors are
/// tuned for.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }
    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a.sqrt() * norm_b.sqrt())
}

// ============================================================================
// Ollama
// ============================================================================

/// `EmbeddingModel` over a local Ollama server's `/api/embed` endpoint.
///
/// No API key; default `base_url` is `http://localhost:11434`. Like the chat
/// [`OllamaModelInterface`](crate::ollama::OllamaModelInterface) this fails fast
/// on connection errors (no retry) with a "not running" hint. Ollama does not
/// report vector length except in the embedding itself, so `dimensions` starts
/// at `0` (unknown) — set it with [`with_dimensions`](Self::with_dimensions) or
/// discover it with [`probe_dimensions`](Self::probe_dimensions).
pub struct OllamaEmbeddingModel {
    model_id: String,
    base_url: String,
    dimensions: usize,
    timeout: Duration,
    http_client: Arc<reqwest::Client>,
}

#[derive(Serialize)]
struct OllamaEmbedRequest<'a> {
    model: &'a str,
    input: &'a [String],
}

#[derive(Deserialize)]
struct OllamaEmbedResponse {
    #[serde(default)]
    embeddings: Vec<Vec<f32>>,
}

impl OllamaEmbeddingModel {
    pub const DEFAULT_BASE_URL: &'static str = "http://localhost:11434";
    pub const DEFAULT_TIMEOUT: Duration = Duration::from_secs(120);

    pub fn new(model_id: impl Into<String>) -> Self {
        Self {
            model_id: model_id.into(),
            base_url: Self::DEFAULT_BASE_URL.into(),
            dimensions: 0,
            timeout: Self::DEFAULT_TIMEOUT,
            http_client: Arc::new(reqwest::Client::new()),
        }
    }

    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }

    /// Pin the vector length (so a [`VectorStore`](crate::vector_store::VectorStore)
    /// can be sized without a probe round-trip).
    pub fn with_dimensions(mut self, dimensions: usize) -> Self {
        self.dimensions = dimensions;
        self
    }

    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    pub fn with_http_client(mut self, client: Arc<reqwest::Client>) -> Self {
        self.http_client = client;
        self
    }

    /// Discover the model's vector length by embedding a probe string. Returns
    /// the length; does NOT mutate `self` — pin it with `with_dimensions(n)`.
    pub async fn probe_dimensions(&self) -> Result<usize, ModelError> {
        let vector = self.embed_one("dimension probe".into()).await?;
        Ok(vector.len())
    }
}

impl std::fmt::Debug for OllamaEmbeddingModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OllamaEmbeddingModel")
            .field("model_id", &self.model_id)
            .field("base_url", &self.base_url)
            .field("dimensions", &self.dimensions)
            .field("timeout", &self.timeout)
            .finish()
    }
}

impl EmbeddingModel for OllamaEmbeddingModel {
    fn embed<'a>(&'a self, inputs: Vec<String>) -> BoxFut<'a, Result<Vec<Vec<f32>>, ModelError>> {
        Box::pin(async move {
            if inputs.is_empty() {
                return Ok(Vec::new());
            }
            let url = format!("{}/api/embed", self.base_url);
            let body = OllamaEmbedRequest {
                model: &self.model_id,
                input: &inputs,
            };
            let resp = self
                .http_client
                .post(&url)
                .header("content-type", "application/json")
                .timeout(self.timeout)
                .json(&body)
                .send()
                .await
                .map_err(|e| {
                    if e.is_timeout() {
                        ModelError::Timeout
                    } else if e.is_connect() {
                        ModelError::ProviderError {
                            code: 0,
                            message: format!("Ollama not running at {}", self.base_url),
                        }
                    } else {
                        ModelError::Transport {
                            message: format!("HTTP transport error: {e}"),
                        }
                    }
                })?;
            if !resp.status().is_success() {
                let code = resp.status().as_u16();
                let body_text = resp.text().await.unwrap_or_default();
                return Err(ModelError::ProviderError {
                    code,
                    message: body_text.chars().take(500).collect(),
                });
            }
            let parsed: OllamaEmbedResponse =
                resp.json().await.map_err(|e| ModelError::ProviderError {
                    code: 0,
                    message: format!("decode embed response failed: {e}"),
                })?;
            Ok(parsed.embeddings)
        })
    }

    fn dimensions(&self) -> usize {
        self.dimensions
    }

    fn info(&self) -> EmbeddingInfo {
        EmbeddingInfo {
            provider: "ollama".into(),
            model_id: self.model_id.clone(),
            dimensions: self.dimensions,
        }
    }
}

// ============================================================================
// OpenAI
// ============================================================================

/// `EmbeddingModel` over `${base_url}/embeddings` (OpenAI + OpenAI-compatible
/// servers). Reuses the shared [`provider_http`](crate::provider_http) retry
/// loop and `{error:{message}}` envelope, exactly like the chat
/// [`OpenAIModelInterface`](crate::openai::OpenAIModelInterface).
///
/// `dimensions` defaults from the model id for the known `text-embedding-*`
/// families; [`with_dimensions`](Self::with_dimensions) overrides it AND, for
/// the v3 families that accept the truncation parameter, asks the API for that
/// length. The id-prefix gate keeps older models (e.g. `ada-002`, which rejects
/// the parameter) working.
pub struct OpenAIEmbeddingModel {
    api_key: String,
    model_id: String,
    base_url: String,
    dimensions: usize,
    timeout: Duration,
    max_retries: u32,
    http_client: Arc<reqwest::Client>,
}

#[derive(Serialize)]
struct OpenAIEmbedRequest<'a> {
    model: &'a str,
    input: &'a [String],
    #[serde(skip_serializing_if = "Option::is_none")]
    dimensions: Option<usize>,
}

#[derive(Deserialize)]
struct OpenAIEmbedResponse {
    data: Vec<OpenAIEmbedDatum>,
}

#[derive(Deserialize)]
struct OpenAIEmbedDatum {
    embedding: Vec<f32>,
    #[serde(default)]
    index: usize,
}

impl OpenAIEmbeddingModel {
    pub const DEFAULT_BASE_URL: &'static str = "https://api.openai.com/v1";
    pub const DEFAULT_TIMEOUT: Duration = Duration::from_secs(120);
    pub const DEFAULT_MAX_RETRIES: u32 = 3;

    pub fn new(api_key: impl Into<String>, model_id: impl Into<String>) -> Self {
        let model_id = model_id.into();
        let dimensions = Self::default_dimensions(&model_id);
        Self {
            api_key: api_key.into(),
            model_id,
            base_url: Self::DEFAULT_BASE_URL.into(),
            dimensions,
            timeout: Self::DEFAULT_TIMEOUT,
            max_retries: Self::DEFAULT_MAX_RETRIES,
            http_client: Arc::new(reqwest::Client::new()),
        }
    }

    pub fn from_env(env_var: &str, model_id: impl Into<String>) -> Result<Self, ModelError> {
        Ok(Self::new(
            crate::provider_http::key_from_env(env_var)?,
            model_id,
        ))
    }

    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }

    /// Pin the vector length. For the `text-embedding-3-*` families this is also
    /// sent as the request `dimensions` parameter (the API truncates to it);
    /// older models that reject the parameter ignore this on the wire.
    pub fn with_dimensions(mut self, dimensions: usize) -> Self {
        self.dimensions = dimensions;
        self
    }

    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    pub fn with_max_retries(mut self, n: u32) -> Self {
        self.max_retries = n;
        self
    }

    pub fn with_http_client(mut self, client: Arc<reqwest::Client>) -> Self {
        self.http_client = client;
        self
    }

    /// Vector length for the known `text-embedding-*` families; `0` otherwise.
    fn default_dimensions(model_id: &str) -> usize {
        match model_id {
            "text-embedding-3-large" => 3072,
            "text-embedding-3-small" | "text-embedding-ada-002" => 1536,
            _ => 0,
        }
    }

    /// Whether this model accepts the request `dimensions` truncation parameter.
    fn supports_dimensions_param(&self) -> bool {
        self.model_id.starts_with("text-embedding-3")
    }
}

impl std::fmt::Debug for OpenAIEmbeddingModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OpenAIEmbeddingModel")
            .field("api_key", &"<redacted>")
            .field("model_id", &self.model_id)
            .field("base_url", &self.base_url)
            .field("dimensions", &self.dimensions)
            .field("timeout", &self.timeout)
            .field("max_retries", &self.max_retries)
            .finish()
    }
}

impl EmbeddingModel for OpenAIEmbeddingModel {
    fn embed<'a>(&'a self, inputs: Vec<String>) -> BoxFut<'a, Result<Vec<Vec<f32>>, ModelError>> {
        Box::pin(async move {
            if inputs.is_empty() {
                return Ok(Vec::new());
            }
            let url = format!("{}/embeddings", self.base_url);
            let dimensions = if self.dimensions > 0 && self.supports_dimensions_param() {
                Some(self.dimensions)
            } else {
                None
            };
            let body = OpenAIEmbedRequest {
                model: &self.model_id,
                input: &inputs,
                dimensions,
            };
            let resp = crate::provider_http::send_with_retry(
                || {
                    self.http_client
                        .post(&url)
                        .bearer_auth(&self.api_key)
                        .json(&body)
                },
                self.max_retries,
                self.timeout,
                crate::provider_http::is_retryable_status,
                |resp| async move {
                    let parts = crate::provider_http::error_parts(resp).await;
                    ModelError::ProviderError {
                        code: parts.code,
                        message: parts.message,
                    }
                },
            )
            .await?;
            let mut parsed: OpenAIEmbedResponse =
                resp.json().await.map_err(|e| ModelError::ProviderError {
                    code: 0,
                    message: format!("decode embeddings response failed: {e}"),
                })?;
            // The API returns data in request order, but `index` is authoritative
            // — sort by it so a non-conforming compatible server can't scramble
            // the input↔vector correspondence.
            parsed.data.sort_by_key(|d| d.index);
            Ok(parsed.data.into_iter().map(|d| d.embedding).collect())
        })
    }

    fn dimensions(&self) -> usize {
        self.dimensions
    }

    fn info(&self) -> EmbeddingInfo {
        EmbeddingInfo {
            provider: "openai".into(),
            model_id: self.model_id.clone(),
            dimensions: self.dimensions,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wiremock::matchers::{header, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    #[test]
    fn cosine_basics() {
        let a = [1.0, 0.0, 0.0];
        let b = [1.0, 0.0, 0.0];
        let c = [0.0, 1.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 1e-6);
        assert!(cosine_similarity(&a, &c).abs() < 1e-6);
        // Degenerate inputs are 0.0, never NaN.
        assert_eq!(cosine_similarity(&a, &[0.0, 0.0, 0.0]), 0.0);
        assert_eq!(cosine_similarity(&a, &[1.0, 2.0]), 0.0);
    }

    #[test]
    fn object_safe() {
        let _embedder: Box<dyn EmbeddingModel> =
            Box::new(OllamaEmbeddingModel::new("nomic-embed-text"));
        let _arc: Arc<dyn EmbeddingModel> =
            Arc::new(OpenAIEmbeddingModel::new("k", "text-embedding-3-small"));
        assert_eq!(_arc.dimensions(), 1536);
    }

    #[tokio::test]
    async fn ollama_embed_round_trip() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/api/embed"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "embeddings": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
            })))
            .mount(&server)
            .await;

        let model = OllamaEmbeddingModel::new("nomic-embed-text").with_base_url(server.uri());
        let vectors = model
            .embed(vec!["hello".into(), "world".into()])
            .await
            .unwrap();
        assert_eq!(vectors, vec![vec![0.1, 0.2, 0.3], vec![0.4, 0.5, 0.6]]);
    }

    #[tokio::test]
    async fn ollama_empty_input_skips_call() {
        // No mock mounted: an empty batch must not contact the server.
        let model =
            OllamaEmbeddingModel::new("nomic-embed-text").with_base_url("http://127.0.0.1:1");
        assert_eq!(model.embed(Vec::new()).await.unwrap().len(), 0);
    }

    #[tokio::test]
    async fn openai_embed_orders_by_index_and_authenticates() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/embeddings"))
            .and(header("authorization", "Bearer test-key"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "data": [
                    { "embedding": [0.9, 0.9], "index": 1 },
                    { "embedding": [0.1, 0.1], "index": 0 }
                ]
            })))
            .mount(&server)
            .await;

        let model = OpenAIEmbeddingModel::new("test-key", "text-embedding-3-small")
            .with_base_url(server.uri());
        let vectors = model.embed(vec!["a".into(), "b".into()]).await.unwrap();
        // Reordered by `index`, so input "a" (index 0) maps to [0.1, 0.1].
        assert_eq!(vectors, vec![vec![0.1, 0.1], vec![0.9, 0.9]]);
    }

    #[tokio::test]
    async fn openai_maps_error_envelope() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/embeddings"))
            .respond_with(ResponseTemplate::new(400).set_body_json(serde_json::json!({
                "error": { "message": "bad model" }
            })))
            .mount(&server)
            .await;

        let model = OpenAIEmbeddingModel::new("k", "text-embedding-3-small")
            .with_base_url(server.uri())
            .with_max_retries(0);
        let err = model.embed(vec!["x".into()]).await.unwrap_err();
        match err {
            ModelError::ProviderError { code, message } => {
                assert_eq!(code, 400);
                assert_eq!(message, "bad model");
            }
            other => panic!("expected ProviderError, got {other:?}"),
        }
    }
}
