#![cfg(feature = "write-postgres")]
//! `DELETE` and `UPDATE` on the single-catalog Postgres writer, against a lake in a
//! Postgres schema of its own. Each statement is one snapshot that supersedes the
//! delete file of every data file it touches, readable by us and by DuckDB.

use std::sync::Arc;

use arrow::array::{Array, Int64Array, RecordBatch, StringArray, UInt64Array};
use arrow::datatypes::{DataType, Field, Schema};
use datafusion::prelude::*;
use object_store::local::LocalFileSystem;
use sqlx::postgres::PgPool;
use tempfile::TempDir;
use testcontainers::ContainerAsync;
use testcontainers::runners::AsyncRunner;
use testcontainers_modules::postgres::Postgres;

use datafusion_ducklake::metadata_writer::{DeleteFileEntry, DeleteFileInfo, MetadataWriter};
use datafusion_ducklake::{
    DuckLakeCatalog, DuckLakeError, DuckLakeTableWriter, LakePool, PostgresMetadataProvider,
    PostgresSingleCatalogMetadataWriter, register_ducklake_functions,
};

const LAKE: &str = "lake";

async fn database() -> anyhow::Result<(PgPool, ContainerAsync<Postgres>)> {
    let container = Postgres::default().start().await?;
    let port = container.get_host_port_ipv4(5432).await?;
    let pool = LakePool::connect(
        &format!("postgresql://postgres:postgres@127.0.0.1:{port}/postgres"),
        4,
    )
    .await?;
    Ok((pool, container))
}

/// The writer for the lake in [`LAKE`], its data under `data`.
async fn writer(pool: &PgPool, data: &TempDir) -> Arc<PostgresSingleCatalogMetadataWriter> {
    let writer =
        PostgresSingleCatalogMetadataWriter::from_lake(LakePool::in_schema(pool.clone(), LAKE));
    let path = format!("{}/", data.path().display());
    let writer = tokio::task::spawn_blocking(move || {
        writer.initialize_schema().unwrap();
        writer.set_data_path(&path).unwrap();
        writer
    })
    .await
    .unwrap();
    Arc::new(writer)
}

/// Writes `main.t (id, name)` with ids 1 to 4, as one data file.
async fn seed(pool: &PgPool, data: &TempDir) {
    let batch = RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("name", DataType::Utf8, true),
        ])),
        vec![
            Arc::new(Int64Array::from_iter_values(1..=4)),
            Arc::new(StringArray::from(vec!["a", "b", "c", "d"])),
        ],
    )
    .unwrap();
    DuckLakeTableWriter::new(writer(pool, data).await, Arc::new(LocalFileSystem::new()))
        .unwrap()
        .write_table("main", "t", &[batch])
        .await
        .unwrap();
}

fn provider(pool: &PgPool) -> PostgresMetadataProvider {
    PostgresMetadataProvider::from_lake(LakePool::in_schema(pool.clone(), LAKE))
}

/// A session that can write the lake as `lake`, with the DuckLake table functions.
async fn session(pool: &PgPool, data: &TempDir) -> SessionContext {
    let catalog =
        DuckLakeCatalog::with_writer(Arc::new(provider(pool)), writer(pool, data).await).unwrap();
    let ctx = SessionContext::new();
    ctx.register_catalog("lake", Arc::new(catalog));
    register_ducklake_functions(&ctx, Arc::new(provider(pool)));
    ctx
}

/// The rows affected by a `DELETE` or `UPDATE`.
async fn affected(ctx: &SessionContext, sql: &str) -> u64 {
    let batches = ctx.sql(sql).await.unwrap().collect().await.unwrap();
    batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<UInt64Array>()
        .unwrap()
        .value(0)
}

/// `(id, name)` rows of `sql`, whose first two columns they are.
async fn rows(ctx: &SessionContext, sql: &str) -> Vec<(i64, String)> {
    let batches = ctx.sql(sql).await.unwrap().collect().await.unwrap();
    batches
        .iter()
        .flat_map(|batch| {
            let id = batch
                .column(0)
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap()
                .clone();
            let name = arrow::compute::cast(batch.column(1), &DataType::Utf8).unwrap();
            let name = name.as_any().downcast_ref::<StringArray>().unwrap().clone();
            (0..batch.num_rows())
                .map(move |i| (id.value(i), name.value(i).to_string()))
                .collect::<Vec<_>>()
        })
        .collect()
}

async fn head(pool: &PgPool) -> i64 {
    sqlx::query_scalar("SELECT MAX(snapshot_id) FROM lake.ducklake_snapshot")
        .fetch_one(pool)
        .await
        .unwrap()
}

/// `(delete_count, end_snapshot)` of every delete file, oldest first.
async fn delete_files(pool: &PgPool) -> Vec<(i64, Option<i64>)> {
    sqlx::query_as(
        "SELECT delete_count, end_snapshot FROM lake.ducklake_delete_file
         ORDER BY delete_file_id",
    )
    .fetch_all(pool)
    .await
    .unwrap()
}

#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn delete_and_update_are_one_snapshot_each() {
    let (pool, _container) = database().await.unwrap();
    let data = TempDir::new().unwrap();
    seed(&pool, &data).await;
    let seeded = head(&pool).await;
    let ctx = session(&pool, &data).await;

    assert_eq!(
        affected(&ctx, "DELETE FROM lake.main.t WHERE id = 1").await,
        1
    );
    let deleted = head(&pool).await;
    assert_eq!(deleted, seeded + 1);
    assert_eq!(delete_files(&pool).await, [(1, None)]);

    assert_eq!(
        affected(&ctx, "UPDATE lake.main.t SET name = 'B' WHERE id = 2").await,
        1
    );
    let updated = head(&pool).await;
    assert_eq!(updated, deleted + 1);
    // The update's delete file carries both deleted positions and supersedes the
    // first one in the same snapshot as the new row version.
    assert_eq!(delete_files(&pool).await, [(1, Some(updated)), (2, None)]);

    assert_eq!(
        rows(&ctx, "SELECT id, name FROM lake.main.t ORDER BY id").await,
        [(2, "B".into()), (3, "c".into()), (4, "d".into())]
    );
    let changes = format!(
        "SELECT id, change_type FROM ducklake_table_changes('main.t', {updated}, {updated})
         ORDER BY change_type"
    );
    assert_eq!(
        rows(&ctx, &changes).await,
        [(2, "update_postimage".into()), (2, "update_preimage".into())]
    );
    let before =
        format!("SELECT id, name FROM ducklake_table_at('main', 't', {seeded}) ORDER BY id");
    assert_eq!(
        rows(&ctx, &before).await,
        [(1, "a".into()), (2, "b".into()), (3, "c".into()), (4, "d".into())]
    );
}

#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn a_delete_read_against_a_superseded_delete_file_conflicts() {
    let (pool, _container) = database().await.unwrap();
    let data = TempDir::new().unwrap();
    seed(&pool, &data).await;
    let base = head(&pool).await;
    let ctx = session(&pool, &data).await;
    // Another writer deletes from the file after `base`.
    affected(&ctx, "DELETE FROM lake.main.t WHERE id = 1").await;

    let (table_id, data_file_id): (i64, i64) =
        sqlx::query_as("SELECT table_id, data_file_id FROM lake.ducklake_data_file")
            .fetch_one(&pool)
            .await
            .unwrap();
    let writer = writer(&pool, &data).await;
    let stale = DeleteFileEntry {
        data_file_id,
        expected_prev_delete_file: None,
        delete: DeleteFileInfo::new("never-written.parquet", 1, 1),
    };
    let committed = tokio::task::spawn_blocking(move || {
        writer.commit_positional_deletes(table_id, "main", "t", base, &[stale])
    })
    .await
    .unwrap();

    assert!(
        matches!(committed, Err(DuckLakeError::Conflict(_))),
        "{committed:?}"
    );
    assert_eq!(
        head(&pool).await,
        base + 1,
        "the refused delete left a snapshot"
    );
    assert_eq!(delete_files(&pool).await, [(1, None)]);
}

#[cfg(feature = "metadata-duckdb")]
#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn duckdb_reads_our_deletes_and_updates() {
    let (pool, container) = database().await.unwrap();
    let port = container.get_host_port_ipv4(5432).await.unwrap();
    let data = TempDir::new().unwrap();
    seed(&pool, &data).await;
    let ctx = session(&pool, &data).await;
    affected(&ctx, "DELETE FROM lake.main.t WHERE id = 1").await;
    affected(&ctx, "UPDATE lake.main.t SET name = 'B' WHERE id = 2").await;

    crate::common::ensure_ducklake_installed();
    let duckdb = duckdb::Connection::open_in_memory().unwrap();
    duckdb
        .execute_batch("INSTALL postgres; LOAD postgres; LOAD ducklake;")
        .unwrap();
    let dsn = format!("host=127.0.0.1 port={port} dbname=postgres user=postgres password=postgres");
    duckdb
        .execute_batch(&format!(
            "ATTACH 'ducklake:postgres:{dsn}' AS ours (METADATA_SCHEMA '{LAKE}', READ_ONLY)"
        ))
        .unwrap();
    let mut read = duckdb
        .prepare("SELECT id, name FROM ours.main.t ORDER BY id")
        .unwrap();
    let read: Vec<(i64, String)> = read
        .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))
        .unwrap()
        .collect::<Result<_, _>>()
        .unwrap();
    assert_eq!(read, [(2, "B".into()), (3, "c".into()), (4, "d".into())]);
}
