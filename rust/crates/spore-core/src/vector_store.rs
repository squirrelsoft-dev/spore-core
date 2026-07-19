//! `VectorStore` — store and similarity-search embedding vectors.
//!
//! The companion to [`EmbeddingModel`](crate::embedding::EmbeddingModel): the
//! embedder turns text into vectors, the store persists them with arbitrary
//! JSON metadata and answers top-k nearest-neighbour queries. The trait is
//! deliberately vector-only — "embed this text, then store/search it" is a
//! composition the caller (or a higher-level RAG crate) performs, keeping the
//! storage backend independent of any particular embedding provider.
//!
//! Object-safe via [`BoxFut`](crate::harness::BoxFut), with a blanket `Arc`
//! impl, so an `Arc<dyn VectorStore>` is itself a `VectorStore`. The reference
//! backend that ships separately is `spore-sqlvec` (sqlite-vec); an in-memory
//! cosine implementation lives in this module's tests as the trait's executable
//! contract.

use std::sync::Arc;

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::harness::BoxFut;

// ============================================================================
// Records, hits, filters
// ============================================================================

/// A vector to store: a caller-chosen `id`, the `vector` itself, and arbitrary
/// JSON `metadata` (a JSON object by convention — e.g. the source text, a stack
/// id, a `kind` discriminator). Re-adding an existing `id` is an upsert.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VectorRecord {
    pub id: String,
    pub vector: Vec<f32>,
    #[serde(default)]
    pub metadata: serde_json::Value,
}

impl VectorRecord {
    pub fn new(id: impl Into<String>, vector: Vec<f32>, metadata: serde_json::Value) -> Self {
        Self {
            id: id.into(),
            vector,
            metadata,
        }
    }
}

/// One search result: the stored `id`, its `score` (HIGHER = more similar —
/// backends normalise their native distance into this convention), and the
/// record's `metadata`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VectorHit {
    pub id: String,
    pub score: f32,
    #[serde(default)]
    pub metadata: serde_json::Value,
}

/// An AND-ed set of equality predicates over top-level metadata keys. Kept
/// intentionally small: it expresses the common "scope this search to
/// `kind = "chunk"` AND `stack_id = "x"`" need. Disjunctions are expressed by
/// running multiple searches and merging — exactly how multi-stage RAG retrieval
/// already composes results.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct MetadataFilter {
    /// `(key, value)` pairs; a record matches when every key equals its value.
    pub equals: Vec<(String, serde_json::Value)>,
}

impl MetadataFilter {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add an equality predicate (builder style).
    pub fn eq(mut self, key: impl Into<String>, value: impl Into<serde_json::Value>) -> Self {
        self.equals.push((key.into(), value.into()));
        self
    }

    pub fn is_empty(&self) -> bool {
        self.equals.is_empty()
    }

    /// Whether a record's `metadata` satisfies every predicate. Backends that
    /// cannot push a predicate down to storage can apply it with this.
    pub fn matches(&self, metadata: &serde_json::Value) -> bool {
        self.equals.iter().all(|(k, v)| metadata.get(k) == Some(v))
    }
}

// ============================================================================
// Errors
// ============================================================================

#[derive(Debug, Error, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
#[non_exhaustive]
pub enum VectorStoreError {
    /// A record (or query) vector length did not match the store's configured
    /// dimensionality.
    #[error("vector dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    /// Any backend/transport failure (SQL error, I/O, serialization).
    #[error("vector store backend error: {message}")]
    Backend { message: String },
}

// ============================================================================
// Trait
// ============================================================================

/// Store and similarity-search embedding vectors. Object-safe via `BoxFut`.
pub trait VectorStore: Send + Sync {
    /// Insert or upsert records (matched by `id`). Implementations should reject
    /// a record whose vector length differs from the store's dimensionality with
    /// [`VectorStoreError::DimensionMismatch`].
    fn add<'a>(&'a self, records: Vec<VectorRecord>) -> BoxFut<'a, Result<(), VectorStoreError>>;

    /// Return up to `top_k` nearest records to `query`, highest `score` first.
    /// When `filter` is `Some`, only records whose metadata satisfies it are
    /// considered (the filter is applied as part of the search, not as a
    /// post-truncation drop).
    fn search<'a>(
        &'a self,
        query: Vec<f32>,
        top_k: usize,
        filter: Option<MetadataFilter>,
    ) -> BoxFut<'a, Result<Vec<VectorHit>, VectorStoreError>>;

    /// Delete every record whose metadata matches `filter`. Returns the count
    /// removed. An empty filter matches all records (a full clear).
    fn delete<'a>(&'a self, filter: MetadataFilter) -> BoxFut<'a, Result<u64, VectorStoreError>>;

    /// Number of stored vectors (diagnostics / tests).
    fn len<'a>(&'a self) -> BoxFut<'a, Result<u64, VectorStoreError>>;

    /// Whether the store holds no vectors. Defaults to `len() == 0`.
    fn is_empty<'a>(&'a self) -> BoxFut<'a, Result<bool, VectorStoreError>> {
        Box::pin(async move { Ok(self.len().await? == 0) })
    }
}

/// Blanket impl so an `Arc`-wrapped store is itself a `VectorStore`.
impl<T: VectorStore + ?Sized> VectorStore for Arc<T> {
    fn add<'a>(&'a self, records: Vec<VectorRecord>) -> BoxFut<'a, Result<(), VectorStoreError>> {
        (**self).add(records)
    }

    fn search<'a>(
        &'a self,
        query: Vec<f32>,
        top_k: usize,
        filter: Option<MetadataFilter>,
    ) -> BoxFut<'a, Result<Vec<VectorHit>, VectorStoreError>> {
        (**self).search(query, top_k, filter)
    }

    fn delete<'a>(&'a self, filter: MetadataFilter) -> BoxFut<'a, Result<u64, VectorStoreError>> {
        (**self).delete(filter)
    }

    fn len<'a>(&'a self) -> BoxFut<'a, Result<u64, VectorStoreError>> {
        (**self).len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embedding::cosine_similarity;
    use std::sync::Mutex;

    /// Reference in-memory `VectorStore`: the trait's executable contract.
    /// Mirrors what `spore-sqlvec` must do, using cosine over a `Vec`.
    struct InMemoryVectorStore {
        dimensions: usize,
        rows: Mutex<Vec<VectorRecord>>,
    }

    impl InMemoryVectorStore {
        fn new(dimensions: usize) -> Self {
            Self {
                dimensions,
                rows: Mutex::new(Vec::new()),
            }
        }
    }

    impl VectorStore for InMemoryVectorStore {
        fn add<'a>(
            &'a self,
            records: Vec<VectorRecord>,
        ) -> BoxFut<'a, Result<(), VectorStoreError>> {
            Box::pin(async move {
                let mut rows = self.rows.lock().unwrap();
                for record in records {
                    if record.vector.len() != self.dimensions {
                        return Err(VectorStoreError::DimensionMismatch {
                            expected: self.dimensions,
                            actual: record.vector.len(),
                        });
                    }
                    // Upsert by id.
                    if let Some(existing) = rows.iter_mut().find(|r| r.id == record.id) {
                        *existing = record;
                    } else {
                        rows.push(record);
                    }
                }
                Ok(())
            })
        }

        fn search<'a>(
            &'a self,
            query: Vec<f32>,
            top_k: usize,
            filter: Option<MetadataFilter>,
        ) -> BoxFut<'a, Result<Vec<VectorHit>, VectorStoreError>> {
            Box::pin(async move {
                let rows = self.rows.lock().unwrap();
                let mut hits: Vec<VectorHit> = rows
                    .iter()
                    .filter(|r| match &filter {
                        Some(f) => f.matches(&r.metadata),
                        None => true,
                    })
                    .map(|r| VectorHit {
                        id: r.id.clone(),
                        score: cosine_similarity(&query, &r.vector),
                        metadata: r.metadata.clone(),
                    })
                    .collect();
                hits.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
                hits.truncate(top_k);
                Ok(hits)
            })
        }

        fn delete<'a>(
            &'a self,
            filter: MetadataFilter,
        ) -> BoxFut<'a, Result<u64, VectorStoreError>> {
            Box::pin(async move {
                let mut rows = self.rows.lock().unwrap();
                let before = rows.len();
                rows.retain(|r| !filter.matches(&r.metadata));
                Ok((before - rows.len()) as u64)
            })
        }

        fn len<'a>(&'a self) -> BoxFut<'a, Result<u64, VectorStoreError>> {
            Box::pin(async move { Ok(self.rows.lock().unwrap().len() as u64) })
        }
    }

    fn rec(id: &str, vector: Vec<f32>, kind: &str) -> VectorRecord {
        VectorRecord::new(id, vector, serde_json::json!({ "kind": kind }))
    }

    #[tokio::test]
    async fn add_search_orders_by_similarity() {
        let store = InMemoryVectorStore::new(3);
        store
            .add(vec![
                rec("a", vec![1.0, 0.0, 0.0], "chunk"),
                rec("b", vec![0.0, 1.0, 0.0], "chunk"),
                rec("c", vec![0.9, 0.1, 0.0], "chunk"),
            ])
            .await
            .unwrap();
        let hits = store.search(vec![1.0, 0.0, 0.0], 2, None).await.unwrap();
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].id, "a");
        assert_eq!(hits[1].id, "c");
    }

    #[tokio::test]
    async fn filter_scopes_search() {
        let store = InMemoryVectorStore::new(2);
        store
            .add(vec![
                rec("a", vec![1.0, 0.0], "chunk"),
                rec("b", vec![1.0, 0.0], "summary"),
            ])
            .await
            .unwrap();
        let filter = MetadataFilter::new().eq("kind", "summary");
        let hits = store
            .search(vec![1.0, 0.0], 10, Some(filter))
            .await
            .unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].id, "b");
    }

    #[tokio::test]
    async fn upsert_and_delete_and_len() {
        let store = InMemoryVectorStore::new(2);
        store
            .add(vec![rec("a", vec![1.0, 0.0], "chunk")])
            .await
            .unwrap();
        // Re-adding the same id upserts rather than duplicating.
        store
            .add(vec![rec("a", vec![0.0, 1.0], "chunk")])
            .await
            .unwrap();
        assert_eq!(store.len().await.unwrap(), 1);

        store
            .add(vec![rec("b", vec![0.0, 1.0], "summary")])
            .await
            .unwrap();
        let removed = store
            .delete(MetadataFilter::new().eq("kind", "chunk"))
            .await
            .unwrap();
        assert_eq!(removed, 1);
        assert_eq!(store.len().await.unwrap(), 1);
    }

    #[tokio::test]
    async fn dimension_mismatch_rejected() {
        let store = InMemoryVectorStore::new(3);
        let err = store
            .add(vec![rec("a", vec![1.0, 0.0], "chunk")])
            .await
            .unwrap_err();
        assert_eq!(
            err,
            VectorStoreError::DimensionMismatch {
                expected: 3,
                actual: 2
            }
        );
    }

    #[test]
    fn object_safe() {
        let _store: Box<dyn VectorStore> = Box::new(InMemoryVectorStore::new(4));
    }
}
