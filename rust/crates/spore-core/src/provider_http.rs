//! Shared HTTP plumbing for the hosted model providers (`anthropic`, `openai`).
//!
//! These two providers speak the same transport shape — JSON request body,
//! retry/backoff on transient statuses, a `{ "error": { "message": … } }` error
//! envelope — and previously each carried a byte-for-byte copy of the retry
//! loop, the backoff function, the status-error mapper, the body encoder, and
//! the `from_env` key reader. This module is the single owner of all of that;
//! the per-provider files keep only what genuinely differs (their wire types,
//! their headers/endpoints, and the small status→`ModelError` match).
//!
//! `ollama` deliberately does NOT route through here: it uses NDJSON, a plain
//! `{ "error": "…" }` envelope, and model-availability probing, so its
//! transport stays self-contained.

use std::future::Future;
use std::time::Duration;

use reqwest::header::HeaderMap;
use serde::Serialize;

use crate::model::ModelError;

/// The base set of retryable HTTP status codes shared by the hosted providers.
/// (Anthropic additionally retries `529`; it composes this with `|| code == 529`.)
pub(crate) fn is_retryable_status(code: u16) -> bool {
    matches!(code, 408 | 425 | 429 | 500 | 502 | 503 | 504)
}

/// Exponential backoff with a 30s ceiling: 0.5s, 1s, 2s, 4s, … Pure function.
pub(crate) fn backoff_delay(attempt: u32) -> Duration {
    let base_ms: u64 = 500u64.saturating_mul(1u64 << attempt.min(6));
    Duration::from_millis(base_ms.min(30_000))
}

/// Parse a `retry-after` header (whole seconds) into a `Duration`.
pub(crate) fn retry_after(headers: &HeaderMap) -> Option<Duration> {
    headers
        .get("retry-after")
        .and_then(|h| h.to_str().ok())
        .and_then(|s| s.parse::<u64>().ok())
        .map(Duration::from_secs)
}

/// Serialize a request body to JSON, mapping a failure to a `ProviderError`
/// with the canonical `"request encode failed: …"` message.
pub(crate) fn encode_body<T: Serialize>(body: &T) -> Result<String, ModelError> {
    serde_json::to_string(body).map_err(|e| ModelError::ProviderError {
        code: 0,
        message: format!("request encode failed: {e}"),
    })
}

/// Read an API key from `env_var`, rejecting an unset or empty value.
pub(crate) fn key_from_env(env_var: &str) -> Result<String, ModelError> {
    let key = std::env::var(env_var).map_err(|_| ModelError::ProviderError {
        code: 0,
        message: format!("env var `{env_var}` not set"),
    })?;
    if key.trim().is_empty() {
        return Err(ModelError::ProviderError {
            code: 0,
            message: format!("env var `{env_var}` is empty"),
        });
    }
    Ok(key)
}

/// The hosted providers return errors as `{ "error": { "message": "…" } }`.
/// One shared deserialization shape for both.
#[derive(serde::Deserialize)]
struct ProviderErrorBody {
    #[serde(default)]
    error: Option<ProviderErrorInner>,
}

#[derive(serde::Deserialize)]
struct ProviderErrorInner {
    #[serde(default)]
    message: Option<String>,
}

/// Status code + `retry-after` + a human message extracted from a non-success
/// response, ready for a provider's small status→`ModelError` match.
pub(crate) struct ErrorParts {
    pub code: u16,
    pub retry_after: Option<Duration>,
    pub message: String,
}

/// Consume a non-success response into [`ErrorParts`]: pulls the message from
/// the typed `{error:{message}}` body, falling back to the first 500 chars of
/// the raw body.
pub(crate) async fn error_parts(resp: reqwest::Response) -> ErrorParts {
    let code = resp.status().as_u16();
    let retry_after = retry_after(resp.headers());
    let body_text = resp.text().await.unwrap_or_default();
    let message = serde_json::from_str::<ProviderErrorBody>(&body_text)
        .ok()
        .and_then(|b| b.error.and_then(|e| e.message))
        .unwrap_or_else(|| body_text.chars().take(500).collect());
    ErrorParts {
        code,
        retry_after,
        message,
    }
}

/// Retrying HTTP sender shared by the hosted providers. `request_builder` is
/// re-invoked each attempt (so a consumed body can be rebuilt); `retryable`
/// decides per-status whether to retry; `map_error` converts a terminal
/// non-retryable response into a `ModelError`.
pub(crate) async fn send_with_retry<RetFn, ErrFn, ErrFut>(
    request_builder: impl Fn() -> reqwest::RequestBuilder,
    max_retries: u32,
    timeout: Duration,
    retryable: RetFn,
    map_error: ErrFn,
) -> Result<reqwest::Response, ModelError>
where
    RetFn: Fn(u16) -> bool,
    ErrFn: Fn(reqwest::Response) -> ErrFut,
    ErrFut: Future<Output = ModelError>,
{
    let mut attempt: u32 = 0;
    loop {
        let req = request_builder().timeout(timeout);
        match req.send().await {
            Ok(resp) => {
                let status = resp.status();
                if status.is_success() {
                    return Ok(resp);
                }
                let code = status.as_u16();
                if retryable(code) && attempt < max_retries {
                    let delay = retry_after(resp.headers()).unwrap_or_else(|| backoff_delay(attempt));
                    tokio::time::sleep(delay).await;
                    attempt += 1;
                    continue;
                }
                return Err(map_error(resp).await);
            }
            Err(e) if e.is_timeout() && attempt < max_retries => {
                tokio::time::sleep(backoff_delay(attempt)).await;
                attempt += 1;
                continue;
            }
            Err(e) if e.is_timeout() => return Err(ModelError::Timeout),
            Err(e) => {
                return Err(ModelError::Transport {
                    message: format!("HTTP transport error: {e}"),
                })
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backoff_grows_then_caps() {
        let d0 = backoff_delay(0);
        let d3 = backoff_delay(3);
        let dmax = backoff_delay(20);
        assert!(d3 > d0);
        assert!(dmax <= Duration::from_millis(30_000));
    }

    #[test]
    fn retryable_status_set_excludes_529() {
        // 529 is Anthropic-specific; the shared base set must NOT include it.
        assert!(is_retryable_status(429));
        assert!(is_retryable_status(503));
        assert!(!is_retryable_status(529));
        assert!(!is_retryable_status(400));
    }
}
