#![cfg(feature = "write-postgres")]
//! `DELETE` and `UPDATE` on the single-catalog Postgres writer, against a lake in a
//! Postgres schema of its own. Each statement is one snapshot that supersedes the
//! delete file of every data file it touches, readable by us and by DuckDB.

use std::sync::Arc;

use arrow::array::{Array, Int64Array, RecordBatch, StringArray, StringViewArray, UInt64Array};
use arrow::datatypes::{DataType, Field, Schema};
use datafusion::prelude::*;
use object_store::local::LocalFileSystem;
use sqlx::postgres::PgPool;
use tempfile::TempDir;
use testcontainers::ContainerAsync;
use testcontainers::runners::AsyncRunner;
use testcontainers_modules::postgres::Postgres;

use datafusion::catalog::CatalogProvider;
use datafusion::common::DFSchema;
use datafusion_ducklake::metadata_writer::{
    DeleteFileEntry, DeleteFileInfo, MetadataWriter, WriteMode,
};
use datafusion_ducklake::{
    DuckLakeCatalog, DuckLakeError, DuckLakeTable, DuckLakeTableWriter, LakePool,
    PostgresMetadataProvider, PostgresSingleCatalogMetadataWriter, TableWriteOptions,
    register_ducklake_functions,
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

/// `main.t` as the lake has it now, read only.
async fn table(pool: &PgPool) -> Arc<dyn datafusion::catalog::TableProvider> {
    let provider = provider(pool);
    let catalog = tokio::task::spawn_blocking(move || DuckLakeCatalog::new(provider).unwrap())
        .await
        .unwrap();
    catalog
        .schema("main")
        .unwrap()
        .table("t")
        .await
        .unwrap()
        .unwrap()
}

#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn a_keyed_write_reads_the_live_rows_it_names_and_commits_once() {
    let (pool, _container) = database().await.unwrap();
    let data = TempDir::new().unwrap();
    seed(&pool, &data).await;
    let ctx = session(&pool, &data).await;
    affected(&ctx, "DELETE FROM lake.main.t WHERE id = 3").await;
    let base = head(&pool).await;

    let provider = table(&pool).await;
    let table = provider.downcast_ref::<DuckLakeTable>().unwrap();
    assert_eq!(table.snapshot(), base);
    let schema = DFSchema::try_from(table.data_schema().as_ref().clone()).unwrap();
    let predicate = ctx
        .state()
        .create_physical_expr(
            col("id").in_list(vec![lit(2_i64), lit(3_i64), lit(9_i64)], false),
            &schema,
        )
        .unwrap();
    let matched = table.matching_rows(&ctx.state(), &predicate).await.unwrap();

    // Row 3 is deleted and 9 was never written, so only row 2, at position 1.
    let [file] = matched.as_slice() else {
        panic!("{matched:?}");
    };
    assert_eq!(file.positions, [1]);
    assert_eq!(file.deleted, [2].into());
    let ids = file
        .rows
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    assert_eq!(ids.values().as_ref(), [2]);

    // Replace row 2 and add row 9 in one snapshot, against the snapshot read.
    let writer =
        DuckLakeTableWriter::new(writer(&pool, &data).await, Arc::new(LocalFileSystem::new()))
            .unwrap();
    let batch = |ids: Vec<i64>, names: Vec<&str>| {
        RecordBatch::try_new(
            table.data_schema(),
            vec![Arc::new(Int64Array::from(ids)), Arc::new(StringViewArray::from(names))],
        )
        .unwrap()
    };
    let delete = writer
        .write_delete_file("main", "t", &file.data_file_path, &[1, 2])
        .await
        .unwrap();
    let entry = DeleteFileEntry {
        data_file_id: file.data_file_id,
        expected_prev_delete_file: file.delete_file_id,
        delete,
    };
    let at_base = TableWriteOptions::new().with_expected_base_snapshot_id(base);
    let mut transaction = writer.transaction().with_options(&at_base);
    transaction
        .stage_write_with_deletes(
            "main",
            "t",
            &table.data_schema(),
            WriteMode::Append,
            &[batch(vec![2, 9], vec!["B", "i"])],
            &[entry],
            &[],
        )
        .await
        .unwrap();
    let committed = transaction.commit().await.unwrap();
    assert_eq!(committed[0].snapshot_id, base + 1);
    assert_eq!(
        rows(&ctx, "SELECT id, name FROM lake.main.t ORDER BY id").await,
        [(1, "a".into()), (2, "B".into()), (4, "d".into()), (9, "i".into())]
    );

    // A second write read at the same snapshot finds the table moved on.
    let mut stale = writer.transaction().with_options(&at_base);
    stale
        .stage_write(
            "main",
            "t",
            &table.data_schema(),
            WriteMode::Append,
            &[batch(vec![10], vec!["j"])],
        )
        .await
        .unwrap();
    let refused = stale.commit().await;
    assert!(
        matches!(refused, Err(DuckLakeError::Conflict(_))),
        "{refused:?}"
    );
    assert_eq!(head(&pool).await, base + 1);
}

/// Every file under `dir`, however deep.
fn files_under(dir: &std::path::Path) -> usize {
    std::fs::read_dir(dir)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .map(|path| {
            if path.is_dir() {
                files_under(&path)
            } else {
                1
            }
        })
        .sum()
}

#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn a_delete_file_handed_to_a_stage_that_fails_is_removed() {
    let (pool, _container) = database().await.unwrap();
    let data = TempDir::new().unwrap();
    seed(&pool, &data).await;
    let files = files_under(data.path());
    let writer =
        DuckLakeTableWriter::new(writer(&pool, &data).await, Arc::new(LocalFileSystem::new()))
            .unwrap();
    let data_file: (i64, String) =
        sqlx::query_as("SELECT data_file_id, path FROM lake.ducklake_data_file")
            .fetch_one(&pool)
            .await
            .unwrap();
    let delete = writer
        .write_delete_file("main", "t", &data_file.1, &[0])
        .await
        .unwrap();
    assert_eq!(files_under(data.path()), files + 1);

    // A null id, which the table does not allow, fails the stage.
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, true),
        Field::new("name", DataType::Utf8View, true),
    ]));
    let rows = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int64Array::from(vec![None::<i64>])),
            Arc::new(StringViewArray::from(vec!["x"])),
        ],
    )
    .unwrap();
    let entry = DeleteFileEntry {
        data_file_id: data_file.0,
        expected_prev_delete_file: None,
        delete,
    };
    let mut transaction = writer.transaction();
    let staged = transaction
        .stage_write_with_deletes(
            "main",
            "t",
            &schema,
            WriteMode::Append,
            &[rows],
            &[entry],
            &[],
        )
        .await;
    assert!(staged.is_err(), "{staged:?}");
    assert_eq!(files_under(data.path()), files, "the delete file was left");
}

#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn a_table_lock_holds_the_tables_other_writers_off() {
    let (pool, _container) = database().await.unwrap();
    let lake = LakePool::in_schema(pool.clone(), LAKE);
    let other_lake = LakePool::in_schema(pool.clone(), "other");
    let wait = std::time::Duration::from_millis(300);

    let held = lake.lock_table("main.t").await.unwrap();
    // Another table, and the same table in another lake, are not held.
    lake.lock_table("main.u")
        .await
        .unwrap()
        .release()
        .await
        .unwrap();
    other_lake
        .lock_table("main.t")
        .await
        .unwrap()
        .release()
        .await
        .unwrap();
    assert!(
        tokio::time::timeout(wait, lake.lock_table("main.t"))
            .await
            .is_err(),
        "a second holder took a held lock"
    );

    held.release().await.unwrap();
    tokio::time::timeout(wait, lake.lock_table("main.t"))
        .await
        .expect("the released lock is free")
        .unwrap();

    // A dropped lock is released as well.
    drop(lake.lock_table("main.v").await.unwrap());
    tokio::time::timeout(std::time::Duration::from_secs(5), lake.lock_table("main.v"))
        .await
        .expect("the dropped lock is free")
        .unwrap();
}

#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn a_login_that_may_only_change_rows_commits_to_a_lake() {
    let (pool, container) = database().await.unwrap();
    let data = TempDir::new().unwrap();
    seed(&pool, &data).await;
    // A login with the lake's rows and sequences, but no right to create anything.
    sqlx::raw_sql(
        "CREATE ROLE rows_only LOGIN PASSWORD 'rows_only';
         GRANT USAGE ON SCHEMA lake TO rows_only;
         GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA lake TO rows_only;
         GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA lake TO rows_only;",
    )
    .execute(&pool)
    .await
    .unwrap();
    let port = container.get_host_port_ipv4(5432).await.unwrap();
    let rows_only = LakePool::connect(
        &format!("postgresql://rows_only:rows_only@127.0.0.1:{port}/postgres"),
        4,
    )
    .await
    .unwrap();
    let writer = Arc::new(PostgresSingleCatalogMetadataWriter::from_lake(
        LakePool::in_schema(rows_only.clone(), LAKE),
    ));
    let catalog = DuckLakeCatalog::with_writer(
        Arc::new(PostgresMetadataProvider::from_lake(LakePool::in_schema(
            rows_only, LAKE,
        ))),
        writer.clone(),
    )
    .unwrap();
    let ctx = SessionContext::new();
    ctx.register_catalog("lake", Arc::new(catalog));

    assert_eq!(
        affected(&ctx, "DELETE FROM lake.main.t WHERE id = 1").await,
        1
    );
    let batch = RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("name", DataType::Utf8View, true),
        ])),
        vec![Arc::new(Int64Array::from(vec![5])), Arc::new(StringViewArray::from(vec!["e"]))],
    )
    .unwrap();
    let writer = DuckLakeTableWriter::new(writer, Arc::new(LocalFileSystem::new())).unwrap();
    let mut transaction = writer.transaction();
    transaction
        .stage_write("main", "t", &batch.schema(), WriteMode::Append, &[batch])
        .await
        .unwrap();
    transaction.commit().await.unwrap();
    assert_eq!(
        rows(&ctx, "SELECT id, name FROM lake.main.t ORDER BY id").await,
        [(2, "b".into()), (3, "c".into()), (4, "d".into()), (5, "e".into())]
    );
}

#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn a_lakes_data_path_is_none_until_it_is_created() {
    let (pool, _container) = database().await.unwrap();
    let lake = PostgresSingleCatalogMetadataWriter::from_lake(LakePool::in_schema(
        pool.clone(),
        "not_yet",
    ));
    assert_eq!(lake.data_path().await.unwrap(), None);
    let lake = tokio::task::spawn_blocking(move || {
        lake.initialize_schema().unwrap();
        lake
    })
    .await
    .unwrap();
    assert_eq!(
        lake.data_path().await.unwrap(),
        None,
        "created, but given no path"
    );
    let lake = tokio::task::spawn_blocking(move || {
        lake.set_data_path("s3://bucket/lake/").unwrap();
        lake
    })
    .await
    .unwrap();
    assert_eq!(
        lake.data_path().await.unwrap().as_deref(),
        Some("s3://bucket/lake/")
    );
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
