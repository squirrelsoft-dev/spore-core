//! `spore-sqlvec` — a [`VectorStore`](spore_core::VectorStore) backed by
//! [sqlite-vec](https://github.com/asg017/sqlite-vec).
//!
//! This is the concrete, dependency-heavy half of spore-core's vector layer:
//! spore-core defines the `VectorStore` / `EmbeddingModel` traits with zero new
//! dependencies; this crate quarantines `rusqlite` + `sqlite-vec` (a bundled C
//! extension) behind that trait. Vectors live in a `vec0` virtual table; each
//! record carries a `TEXT` primary-key id, its embedding, an opaque JSON
//! `metadata` blob (auxiliary column), and zero or more declared *indexed* keys
//! promoted to real `vec0` metadata columns so equality filters push down into
//! the KNN query rather than dropping results after truncation.
//!
//! ```no_run
//! # async fn demo() -> Result<(), spore_core::VectorStoreError> {
//! use spore_core::{VectorStore, VectorRecord, MetadataFilter};
//! use spore_sqlvec::SqliteVecStore;
//!
//! let store = SqliteVecStore::builder(3)
//!     .table("chunks")
//!     .indexed_key("kind")
//!     .open_in_memory()
//!     .await?;
//!
//! store.add(vec![VectorRecord::new(
//!     "doc-1",
//!     vec![0.1, 0.2, 0.3],
//!     serde_json::json!({ "kind": "chunk", "text": "hello" }),
//! )]).await?;
//!
//! let hits = store
//!     .search(vec![0.1, 0.2, 0.3], 5, Some(MetadataFilter::new().eq("kind", "chunk")))
//!     .await?;
//! # let _ = hits; Ok(())
//! # }
//! ```

use std::sync::Once;

use spore_core::vector_store::{
    MetadataFilter, VectorHit, VectorRecord, VectorStore, VectorStoreError,
};
use spore_core::BoxFut;

use rusqlite::types::Value as SqlValue;

/// Register the `sqlite-vec` extension as a SQLite auto-extension, once per
/// process. `vec0` virtual tables are unusable until this runs, and it must run
/// before any connection that touches them is opened — auto-extensions apply to
/// every subsequent `sqlite3_open`.
fn ensure_vec_extension() {
    static INIT: Once = Once::new();
    INIT.call_once(|| unsafe {
        // The init fn is an opaque C symbol; the auto-extension API wants the
        // generic SQLite entrypoint signature. This transmute between two
        // `extern "C"` fn pointers is the documented `sqlite-vec` registration.
        #[allow(clippy::missing_transmute_annotations)]
        rusqlite::ffi::sqlite3_auto_extension(Some(std::mem::transmute(
            sqlite_vec::sqlite3_vec_init as *const (),
        )));
    });
}

fn backend(e: impl std::fmt::Display) -> VectorStoreError {
    VectorStoreError::Backend {
        message: e.to_string(),
    }
}

/// A SQLite identifier must be a bare `[A-Za-z_][A-Za-z0-9_]*` — we interpolate
/// table and column names directly into DDL/queries (they cannot be bound), so
/// reject anything that could break out of an identifier position.
fn validate_ident(kind: &str, ident: &str) -> Result<(), VectorStoreError> {
    let ok = !ident.is_empty()
        && ident
            .chars()
            .enumerate()
            .all(|(i, c)| c == '_' || c.is_ascii_alphanumeric() && !(i == 0 && c.is_ascii_digit()));
    if ok {
        Ok(())
    } else {
        Err(VectorStoreError::Backend {
            message: format!("invalid {kind} identifier: {ident:?}"),
        })
    }
}

/// Little-endian float32 byte encoding — the BLOB form `sqlite-vec` reads as a
/// `float[N]` vector. Unambiguous across crate versions (no JSON parsing).
fn vector_blob(v: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(v.len() * 4);
    for f in v {
        bytes.extend_from_slice(&f.to_le_bytes());
    }
    bytes
}

/// Coerce a JSON metadata value to the TEXT stored in an indexed `vec0` column
/// (and compared against in a filter). `null`/absent → SQL `NULL`; strings pass
/// through verbatim; everything else uses its compact JSON form.
fn indexed_value(value: Option<&serde_json::Value>) -> SqlValue {
    match value {
        None | Some(serde_json::Value::Null) => SqlValue::Null,
        Some(serde_json::Value::String(s)) => SqlValue::Text(s.clone()),
        Some(other) => SqlValue::Text(other.to_string()),
    }
}

// ============================================================================
// Builder
// ============================================================================

/// Builder for a [`SqliteVecStore`]. Created via [`SqliteVecStore::builder`].
pub struct SqliteVecStoreBuilder {
    table: String,
    dimensions: usize,
    indexed_keys: Vec<String>,
}

impl SqliteVecStoreBuilder {
    /// Override the `vec0` table name (default `"vectors"`).
    pub fn table(mut self, table: impl Into<String>) -> Self {
        self.table = table.into();
        self
    }

    /// Promote a metadata key to an indexed `vec0` column so equality filters on
    /// it push down into the KNN query. A [`search`](VectorStore::search) filter
    /// may only reference indexed keys.
    pub fn indexed_key(mut self, key: impl Into<String>) -> Self {
        self.indexed_keys.push(key.into());
        self
    }

    /// Promote several metadata keys at once. See [`indexed_key`](Self::indexed_key).
    pub fn indexed_keys<I, S>(mut self, keys: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.indexed_keys.extend(keys.into_iter().map(Into::into));
        self
    }

    /// Open (or create) the store at `path`.
    pub async fn open(
        self,
        path: impl AsRef<std::path::Path>,
    ) -> Result<SqliteVecStore, VectorStoreError> {
        ensure_vec_extension();
        let conn = tokio_rusqlite::Connection::open(path)
            .await
            .map_err(backend)?;
        self.finish(conn).await
    }

    /// Open an in-memory store (private to this connection; lost on drop).
    pub async fn open_in_memory(self) -> Result<SqliteVecStore, VectorStoreError> {
        ensure_vec_extension();
        let conn = tokio_rusqlite::Connection::open_in_memory()
            .await
            .map_err(backend)?;
        self.finish(conn).await
    }

    async fn finish(
        self,
        conn: tokio_rusqlite::Connection,
    ) -> Result<SqliteVecStore, VectorStoreError> {
        validate_ident("table", &self.table)?;
        for key in &self.indexed_keys {
            validate_ident("indexed key", key)?;
        }
        let ddl = self.create_table_sql();
        conn.call(move |c| {
            c.execute_batch(&ddl)?;
            Ok(())
        })
        .await
        .map_err(backend)?;
        Ok(SqliteVecStore {
            conn,
            table: self.table,
            dimensions: self.dimensions,
            indexed_keys: self.indexed_keys,
        })
    }

    fn create_table_sql(&self) -> String {
        // Column order: PK, indexed metadata columns, the vector column, then the
        // opaque auxiliary metadata blob (`+` prefix = stored-but-unindexed).
        let mut cols = vec!["id TEXT PRIMARY KEY".to_string()];
        for key in &self.indexed_keys {
            cols.push(format!("{key} TEXT"));
        }
        cols.push(format!(
            "embedding FLOAT[{}] distance_metric=cosine",
            self.dimensions
        ));
        cols.push("+metadata TEXT".to_string());
        format!(
            "CREATE VIRTUAL TABLE IF NOT EXISTS {} USING vec0(\n  {}\n)",
            self.table,
            cols.join(",\n  ")
        )
    }
}

// ============================================================================
// Store
// ============================================================================

/// A [`VectorStore`] over a `sqlite-vec` `vec0` table. Cheap to clone — the
/// underlying [`tokio_rusqlite::Connection`] is a handle to one background
/// worker thread.
#[derive(Clone)]
pub struct SqliteVecStore {
    conn: tokio_rusqlite::Connection,
    table: String,
    dimensions: usize,
    indexed_keys: Vec<String>,
}

impl SqliteVecStore {
    /// Start building a store whose vectors have `dimensions` components.
    pub fn builder(dimensions: usize) -> SqliteVecStoreBuilder {
        SqliteVecStoreBuilder {
            table: "vectors".to_string(),
            dimensions,
            indexed_keys: Vec::new(),
        }
    }

    /// The configured vector dimensionality.
    pub fn dimensions(&self) -> usize {
        self.dimensions
    }

    fn check_dims(&self, len: usize) -> Result<(), VectorStoreError> {
        if len == self.dimensions {
            Ok(())
        } else {
            Err(VectorStoreError::DimensionMismatch {
                expected: self.dimensions,
                actual: len,
            })
        }
    }

    /// Reject a filter that references a key not declared via `indexed_key`, so
    /// filtering is always exact (pushed to SQL) rather than a silent
    /// post-truncation drop.
    fn validate_filter(&self, filter: &MetadataFilter) -> Result<(), VectorStoreError> {
        for (key, _) in &filter.equals {
            if !self.indexed_keys.iter().any(|k| k == key) {
                return Err(VectorStoreError::Backend {
                    message: format!(
                        "filter key {key:?} is not indexed; declare it via SqliteVecStore::builder(..).indexed_key({key:?})"
                    ),
                });
            }
        }
        Ok(())
    }
}

impl VectorStore for SqliteVecStore {
    fn add<'a>(&'a self, records: Vec<VectorRecord>) -> BoxFut<'a, Result<(), VectorStoreError>> {
        Box::pin(async move {
            for r in &records {
                self.check_dims(r.vector.len())?;
            }
            if records.is_empty() {
                return Ok(());
            }
            let table = self.table.clone();
            let indexed_keys = self.indexed_keys.clone();
            self.conn
                .call(move |c| {
                    let tx = c.transaction()?;
                    // Column list: id, <indexed keys...>, embedding, metadata.
                    let mut col_names = vec!["id".to_string()];
                    col_names.extend(indexed_keys.iter().cloned());
                    col_names.push("embedding".to_string());
                    col_names.push("metadata".to_string());
                    let placeholders = vec!["?"; col_names.len()].join(", ");
                    let insert_sql = format!(
                        "INSERT INTO {table}({}) VALUES ({placeholders})",
                        col_names.join(", ")
                    );
                    let delete_sql = format!("DELETE FROM {table} WHERE id = ?1");
                    for r in records {
                        // Upsert: vec0 has no ON CONFLICT, so delete-then-insert.
                        tx.execute(&delete_sql, [&r.id])?;
                        let mut params: Vec<SqlValue> = Vec::with_capacity(col_names.len());
                        params.push(SqlValue::Text(r.id.clone()));
                        for key in &indexed_keys {
                            params.push(indexed_value(r.metadata.get(key)));
                        }
                        params.push(SqlValue::Blob(vector_blob(&r.vector)));
                        params.push(SqlValue::Text(r.metadata.to_string()));
                        tx.execute(&insert_sql, rusqlite::params_from_iter(params))?;
                    }
                    tx.commit()?;
                    Ok(())
                })
                .await
                .map_err(backend)
        })
    }

    fn search<'a>(
        &'a self,
        query: Vec<f32>,
        top_k: usize,
        filter: Option<MetadataFilter>,
    ) -> BoxFut<'a, Result<Vec<VectorHit>, VectorStoreError>> {
        Box::pin(async move {
            self.check_dims(query.len())?;
            if let Some(f) = &filter {
                self.validate_filter(f)?;
            }
            let table = self.table.clone();
            self.conn
                .call(move |c| {
                    let mut params: Vec<SqlValue> = vec![SqlValue::Blob(vector_blob(&query))];
                    let mut wheres = vec!["embedding MATCH ?1".to_string()];
                    if let Some(f) = &filter {
                        for (key, value) in &f.equals {
                            params.push(indexed_value(Some(value)));
                            wheres.push(format!("{key} = ?{}", params.len()));
                        }
                    }
                    params.push(SqlValue::Integer(top_k as i64));
                    let limit_idx = params.len();
                    let sql = format!(
                        "SELECT id, metadata, distance FROM {table} WHERE {} ORDER BY distance LIMIT ?{limit_idx}",
                        wheres.join(" AND ")
                    );
                    let mut stmt = c.prepare(&sql)?;
                    let rows = stmt
                        .query_map(rusqlite::params_from_iter(params), |row| {
                            let id: String = row.get(0)?;
                            let metadata: String = row.get(1)?;
                            let distance: f64 = row.get(2)?;
                            Ok((id, metadata, distance))
                        })?
                        .collect::<Result<Vec<_>, _>>()?;
                    let hits = rows
                        .into_iter()
                        .map(|(id, metadata, distance)| VectorHit {
                            id,
                            // cosine distance ∈ [0, 2] → similarity ∈ [-1, 1].
                            score: 1.0 - distance as f32,
                            metadata: serde_json::from_str(&metadata)
                                .unwrap_or(serde_json::Value::Null),
                        })
                        .collect();
                    Ok(hits)
                })
                .await
                .map_err(backend)
        })
    }

    fn delete<'a>(&'a self, filter: MetadataFilter) -> BoxFut<'a, Result<u64, VectorStoreError>> {
        Box::pin(async move {
            self.validate_filter(&filter)?;
            let table = self.table.clone();
            self.conn
                .call(move |c| {
                    let tx = c.transaction()?;
                    // Collect matching ids (metadata-only scan, no KNN), then
                    // delete by primary key — the form vec0 reliably supports.
                    let mut params: Vec<SqlValue> = Vec::new();
                    let select_sql = if filter.is_empty() {
                        format!("SELECT id FROM {table}")
                    } else {
                        let mut wheres = Vec::new();
                        for (key, value) in &filter.equals {
                            params.push(indexed_value(Some(value)));
                            wheres.push(format!("{key} = ?{}", params.len()));
                        }
                        format!("SELECT id FROM {table} WHERE {}", wheres.join(" AND "))
                    };
                    let ids: Vec<String> = {
                        let mut stmt = tx.prepare(&select_sql)?;
                        let rows = stmt
                            .query_map(rusqlite::params_from_iter(params), |row| {
                                row.get::<_, String>(0)
                            })?
                            .collect::<Result<Vec<_>, _>>()?;
                        rows
                    };
                    let delete_sql = format!("DELETE FROM {table} WHERE id = ?1");
                    for id in &ids {
                        tx.execute(&delete_sql, [id])?;
                    }
                    tx.commit()?;
                    Ok(ids.len() as u64)
                })
                .await
                .map_err(backend)
        })
    }

    fn len<'a>(&'a self) -> BoxFut<'a, Result<u64, VectorStoreError>> {
        Box::pin(async move {
            let table = self.table.clone();
            self.conn
                .call(move |c| {
                    let n: i64 =
                        c.query_row(&format!("SELECT count(*) FROM {table}"), [], |row| {
                            row.get(0)
                        })?;
                    Ok(n as u64)
                })
                .await
                .map_err(backend)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use spore_core::vector_store::MetadataFilter;

    fn rec(id: &str, v: Vec<f32>, kind: &str, text: &str) -> VectorRecord {
        VectorRecord::new(id, v, serde_json::json!({ "kind": kind, "text": text }))
    }

    async fn store() -> SqliteVecStore {
        SqliteVecStore::builder(3)
            .table("chunks")
            .indexed_key("kind")
            .open_in_memory()
            .await
            .unwrap()
    }

    #[tokio::test]
    async fn add_search_round_trip_orders_by_similarity() {
        let store = store().await;
        store
            .add(vec![
                rec("a", vec![1.0, 0.0, 0.0], "chunk", "alpha"),
                rec("b", vec![0.0, 1.0, 0.0], "chunk", "beta"),
                rec("c", vec![0.92, 0.08, 0.0], "chunk", "gamma"),
            ])
            .await
            .unwrap();
        assert_eq!(store.len().await.unwrap(), 3);

        let hits = store.search(vec![1.0, 0.0, 0.0], 2, None).await.unwrap();
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].id, "a");
        assert_eq!(hits[1].id, "c");
        // Identical vectors → cosine similarity ≈ 1.0.
        assert!(
            (hits[0].score - 1.0).abs() < 1e-4,
            "score was {}",
            hits[0].score
        );
        // Metadata round-trips.
        assert_eq!(hits[0].metadata.get("text").unwrap(), "alpha");
    }

    #[tokio::test]
    async fn filter_pushes_down_and_does_not_truncate_drop() {
        let store = store().await;
        // The nearest vector to the query is a "summary"; with a chunk-only
        // filter and top_k=1 it must NOT be returned (proves the filter applies
        // inside the KNN, not after a top-1 truncation).
        store
            .add(vec![
                rec("near-summary", vec![1.0, 0.0, 0.0], "summary", "s"),
                rec("far-chunk", vec![0.0, 1.0, 0.0], "chunk", "c"),
            ])
            .await
            .unwrap();
        let hits = store
            .search(
                vec![1.0, 0.0, 0.0],
                1,
                Some(MetadataFilter::new().eq("kind", "chunk")),
            )
            .await
            .unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].id, "far-chunk");
    }

    #[tokio::test]
    async fn upsert_replaces_in_place() {
        let store = store().await;
        store
            .add(vec![rec("a", vec![1.0, 0.0, 0.0], "chunk", "v1")])
            .await
            .unwrap();
        store
            .add(vec![rec("a", vec![0.0, 1.0, 0.0], "chunk", "v2")])
            .await
            .unwrap();
        assert_eq!(store.len().await.unwrap(), 1);
        let hits = store.search(vec![0.0, 1.0, 0.0], 1, None).await.unwrap();
        assert_eq!(hits[0].metadata.get("text").unwrap(), "v2");
    }

    #[tokio::test]
    async fn delete_by_filter_and_clear() {
        let store = store().await;
        store
            .add(vec![
                rec("a", vec![1.0, 0.0, 0.0], "chunk", "a"),
                rec("b", vec![0.0, 1.0, 0.0], "summary", "b"),
                rec("c", vec![0.0, 0.0, 1.0], "chunk", "c"),
            ])
            .await
            .unwrap();
        let removed = store
            .delete(MetadataFilter::new().eq("kind", "chunk"))
            .await
            .unwrap();
        assert_eq!(removed, 2);
        assert_eq!(store.len().await.unwrap(), 1);

        // Empty filter clears all.
        let removed = store.delete(MetadataFilter::new()).await.unwrap();
        assert_eq!(removed, 1);
        assert_eq!(store.len().await.unwrap(), 0);
    }

    #[tokio::test]
    async fn dimension_mismatch_rejected() {
        let store = store().await;
        let err = store
            .add(vec![VectorRecord::new(
                "a",
                vec![1.0, 0.0],
                serde_json::json!({}),
            )])
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

    #[tokio::test]
    async fn undeclared_filter_key_is_rejected() {
        let store = store().await;
        let err = store
            .search(
                vec![1.0, 0.0, 0.0],
                5,
                Some(MetadataFilter::new().eq("stack", "x")),
            )
            .await
            .unwrap_err();
        match err {
            VectorStoreError::Backend { message } => assert!(message.contains("not indexed")),
            other => panic!("expected Backend, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn is_object_safe_vector_store() {
        let store = store().await;
        let dynamic: std::sync::Arc<dyn VectorStore> = std::sync::Arc::new(store);
        dynamic
            .add(vec![rec("a", vec![1.0, 0.0, 0.0], "chunk", "x")])
            .await
            .unwrap();
        assert_eq!(dynamic.len().await.unwrap(), 1);
    }
}
