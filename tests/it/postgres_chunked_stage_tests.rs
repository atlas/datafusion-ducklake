#![cfg(feature = "write-postgres")]
//! One table staged a chunk at a time in one `DuckLakeWriteTransaction`, on
//! the single-catalog Postgres writer: successive staging calls for a table
//! join one stage, which commits as one snapshot.

use std::sync::Arc;

use arrow::array::{Int64Array, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use datafusion::prelude::*;
use object_store::local::LocalFileSystem;
use sqlx::postgres::PgPool;
use tempfile::TempDir;
use testcontainers::ContainerAsync;
use testcontainers::runners::AsyncRunner;
use testcontainers_modules::postgres::Postgres;

use datafusion_ducklake::metadata_writer::{DeleteFileEntry, MetadataWriter, WriteMode};
use datafusion_ducklake::{
    DuckLakeCatalog, DuckLakeError, DuckLakeTableWriter, DuckLakeWriteOptions, LakePool,
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

/// The writer for the lake in [`LAKE`], its data under `data`, inlining no rows
/// unless a test asks.
async fn writer(pool: &PgPool, data: &TempDir) -> Arc<PostgresSingleCatalogMetadataWriter> {
    let writer =
        PostgresSingleCatalogMetadataWriter::from_lake(LakePool::in_schema(pool.clone(), LAKE));
    let path = format!("{}/", data.path().display());
    let writer = tokio::task::spawn_blocking(move || {
        writer.initialize_schema().unwrap();
        writer.set_data_path(&path).unwrap();
        writer
            .set_global_setting("data_inlining_row_limit", "0")
            .unwrap();
        writer
    })
    .await
    .unwrap();
    Arc::new(writer)
}

async fn table_writer(pool: &PgPool, data: &TempDir) -> DuckLakeTableWriter {
    DuckLakeTableWriter::new(writer(pool, data).await, Arc::new(LocalFileSystem::new())).unwrap()
}

fn provider(pool: &PgPool) -> PostgresMetadataProvider {
    PostgresMetadataProvider::from_lake(LakePool::in_schema(pool.clone(), LAKE))
}

async fn session(pool: &PgPool, data: &TempDir) -> SessionContext {
    let catalog =
        DuckLakeCatalog::with_writer(Arc::new(provider(pool)), writer(pool, data).await).unwrap();
    let ctx = SessionContext::new();
    ctx.register_catalog("lake", Arc::new(catalog));
    register_ducklake_functions(&ctx, Arc::new(provider(pool)));
    ctx
}

fn schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("name", DataType::Utf8, true),
    ]))
}

/// Rows `ids`, each named after its id.
fn batch(ids: std::ops::RangeInclusive<i64>) -> RecordBatch {
    let names: Vec<String> = ids.clone().map(|id| format!("n{id}")).collect();
    RecordBatch::try_new(
        schema(),
        vec![Arc::new(Int64Array::from_iter_values(ids)), Arc::new(StringArray::from(names))],
    )
    .unwrap()
}

/// Ids of `sql`, whose first column they are.
async fn ids(ctx: &SessionContext, sql: &str) -> Vec<i64> {
    let batches = ctx.sql(sql).await.unwrap().collect().await.unwrap();
    batches
        .iter()
        .flat_map(|batch| {
            batch
                .column(0)
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap()
                .values()
                .to_vec()
        })
        .collect()
}

async fn head(pool: &PgPool) -> i64 {
    sqlx::query_scalar("SELECT MAX(snapshot_id) FROM lake.ducklake_snapshot")
        .fetch_one(pool)
        .await
        .unwrap()
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

/// `(data_file_id, path)` of `table`'s live data files, oldest first.
async fn data_files(pool: &PgPool, table: &str) -> Vec<(i64, String)> {
    sqlx::query_as(
        "SELECT f.data_file_id, f.path FROM lake.ducklake_data_file f
         JOIN lake.ducklake_table t USING (table_id)
         WHERE t.table_name = $1 AND t.end_snapshot IS NULL AND f.end_snapshot IS NULL
         ORDER BY f.data_file_id",
    )
    .bind(table)
    .fetch_all(pool)
    .await
    .unwrap()
}

/// Writes `main.<table>` with one data file of `ids`.
async fn seed(writer: &DuckLakeTableWriter, table: &str, ids: std::ops::RangeInclusive<i64>) {
    writer
        .append_table("main", table, &[batch(ids)])
        .await
        .unwrap();
}

async fn stage(
    transaction: &mut datafusion_ducklake::DuckLakeWriteTransaction<'_>,
    table: &str,
    ids: std::ops::RangeInclusive<i64>,
) {
    transaction
        .stage_write("main", table, &schema(), WriteMode::Append, &[batch(ids)])
        .await
        .unwrap();
}

/// A delete file for `data_file` deleting `positions`, expecting none live.
async fn delete_entry(
    writer: &DuckLakeTableWriter,
    data_file: &(i64, String),
    positions: &[i64],
) -> DeleteFileEntry {
    let delete = writer
        .write_delete_file("main", "t", &data_file.1, positions)
        .await
        .unwrap();
    DeleteFileEntry {
        data_file_id: data_file.0,
        expected_prev_delete_file: None,
        delete,
    }
}

#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn chunks_of_one_table_commit_as_one_snapshot() {
    let (pool, _container) = database().await.unwrap();
    let data = TempDir::new().unwrap();
    let writer = table_writer(&pool, &data).await;
    // The schema, created by another table's first write.
    seed(&writer, "other", 1..=1).await;
    let ctx = session(&pool, &data).await;
    ctx.sql("CREATE TABLE lake.main.t (id BIGINT NOT NULL, name VARCHAR)")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let created = head(&pool).await;

    let mut transaction = writer.transaction();
    stage(&mut transaction, "t", 1..=2).await;
    stage(&mut transaction, "t", 3..=5).await;
    stage(&mut transaction, "t", 6..=9).await;
    let results = transaction.commit().await.unwrap();

    assert_eq!(results.len(), 1, "{results:?}");
    assert_eq!(results[0].records_written, 9);
    assert_eq!(results[0].files_written, 3);
    assert_eq!(results[0].snapshot_id, created + 1);
    assert_eq!(head(&pool).await, created + 1, "exactly one snapshot");

    let at = |snapshot: i64| {
        format!("SELECT id FROM ducklake_table_at('main', 't', {snapshot}) ORDER BY id")
    };
    assert_eq!(
        ids(&ctx, &at(created + 1)).await,
        (1..=9).collect::<Vec<_>>()
    );
    assert!(ids(&ctx, &at(created)).await.is_empty());

    // Row ids are handed out across the chunks' files in staging order.
    let files: Vec<(i64, i64, i64)> = sqlx::query_as(
        "SELECT f.row_id_start, f.record_count, f.begin_snapshot FROM lake.ducklake_data_file f
         JOIN lake.ducklake_table t USING (table_id)
         WHERE t.table_name = 't' ORDER BY f.data_file_id",
    )
    .fetch_all(&pool)
    .await
    .unwrap();
    assert_eq!(
        files,
        [(0, 2, created + 1), (2, 3, created + 1), (5, 4, created + 1)]
    );
}

#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn rows_and_deletes_in_chunks_commit_together() {
    let (pool, _container) = database().await.unwrap();
    let data = TempDir::new().unwrap();
    let writer = table_writer(&pool, &data).await;
    seed(&writer, "t", 1..=4).await;
    seed(&writer, "t", 5..=8).await;
    let [first, second] = data_files(&pool, "t").await.try_into().unwrap();
    let base = head(&pool).await;

    let at_base = TableWriteOptions::new().with_expected_base_snapshot_id(base);
    let mut transaction = writer.transaction().with_options(&at_base);
    stage(&mut transaction, "t", 10..=11).await;
    // Ids 2 and 3 of the first file.
    let entry = delete_entry(&writer, &first, &[1, 2]).await;
    transaction
        .stage_write_with_deletes(
            "main",
            "t",
            &schema(),
            WriteMode::Append,
            &[batch(12..=13)],
            &[entry],
            &[],
        )
        .await
        .unwrap();
    // Id 8 of the second.
    let entry = delete_entry(&writer, &second, &[3]).await;
    transaction
        .stage_deletes("main", "t", &schema(), &[entry], &[])
        .unwrap();
    let results = transaction.commit().await.unwrap();

    assert_eq!(results.len(), 1, "{results:?}");
    assert_eq!(results[0].records_written, 4);
    assert_eq!(head(&pool).await, base + 1);
    let ctx = session(&pool, &data).await;
    assert_eq!(
        ids(&ctx, "SELECT id FROM lake.main.t ORDER BY id").await,
        [1, 4, 5, 6, 7, 10, 11, 12, 13]
    );
    let delete_snapshots: Vec<i64> =
        sqlx::query_scalar("SELECT begin_snapshot FROM lake.ducklake_delete_file")
            .fetch_all(&pool)
            .await
            .unwrap();
    assert_eq!(delete_snapshots, [base + 1, base + 1]);
}

#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn a_second_delete_file_for_a_data_file_is_refused() {
    let (pool, _container) = database().await.unwrap();
    let data = TempDir::new().unwrap();
    let writer = table_writer(&pool, &data).await;
    seed(&writer, "t", 1..=4).await;
    let [file] = data_files(&pool, "t").await.try_into().unwrap();
    let seeded = files_under(data.path());
    let base = head(&pool).await;

    let mut transaction = writer.transaction();
    stage(&mut transaction, "t", 10..=11).await;
    let entry = delete_entry(&writer, &file, &[0]).await;
    transaction
        .stage_deletes("main", "t", &schema(), &[entry], &[])
        .unwrap();
    assert_eq!(files_under(data.path()), seeded + 2);

    // A joining call's own files are removed when it is refused.
    let again = delete_entry(&writer, &file, &[0, 1]).await;
    let refused = transaction
        .stage_write_with_deletes(
            "main",
            "t",
            &schema(),
            WriteMode::Append,
            &[batch(12..=12)],
            &[again],
            &[],
        )
        .await;
    assert!(
        matches!(refused, Err(DuckLakeError::InvalidConfig(_))),
        "{refused:?}"
    );
    assert_eq!(files_under(data.path()), seeded + 2);

    // As are those of a refused synchronous call, with the staged ones, when
    // the transaction aborts.
    let again = delete_entry(&writer, &file, &[0, 1]).await;
    let refused = transaction.stage_deletes("main", "t", &schema(), &[again], &[]);
    assert!(
        matches!(refused, Err(DuckLakeError::InvalidConfig(_))),
        "{refused:?}"
    );
    transaction.abort().await.unwrap();
    assert_eq!(files_under(data.path()), seeded);
    assert_eq!(head(&pool).await, base);
}

#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn a_failed_commit_removes_every_chunks_files() {
    let (pool, _container) = database().await.unwrap();
    let data = TempDir::new().unwrap();
    let writer = table_writer(&pool, &data).await;
    seed(&writer, "t", 1..=4).await;
    let [file] = data_files(&pool, "t").await.try_into().unwrap();
    let base = head(&pool).await;

    let at_base = TableWriteOptions::new().with_expected_base_snapshot_id(base);
    let mut transaction = writer.transaction().with_options(&at_base);
    stage(&mut transaction, "t", 10..=11).await;
    stage(&mut transaction, "t", 12..=14).await;
    let entry = delete_entry(&writer, &file, &[0]).await;
    transaction
        .stage_deletes("main", "t", &schema(), &[entry], &[])
        .unwrap();

    // Another commit moves the table past the base.
    seed(&writer, "t", 20..=20).await;
    let committed = files_under(data.path());

    let refused = transaction.commit().await;
    assert!(
        matches!(refused, Err(DuckLakeError::Conflict(_))),
        "{refused:?}"
    );
    assert_eq!(files_under(data.path()), committed - 3);
    assert_eq!(head(&pool).await, base + 1);
}

#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn a_chunk_that_inlines_next_to_one_that_does_not_is_refused() {
    // The single-catalog writer never inlines; the multicatalog one does.
    let (pool, _container) = database().await.unwrap();
    datafusion_ducklake::initialize_multicatalog_schema(&pool)
        .await
        .unwrap();
    let catalog = datafusion_ducklake::MulticatalogManager::new(pool.clone())
        .create_catalog("inlining")
        .await
        .unwrap();
    let data = TempDir::new().unwrap();
    let metadata = datafusion_ducklake::PostgresMetadataWriter::with_pool(pool.clone(), catalog)
        .await
        .unwrap();
    metadata
        .set_data_path(&format!("{}/", data.path().display()))
        .unwrap();
    let head = || async {
        sqlx::query_scalar::<_, i64>("SELECT MAX(snapshot_id) FROM ducklake_snapshot")
            .fetch_one(&pool)
            .await
            .unwrap()
    };
    let writer = DuckLakeTableWriter::new(Arc::new(metadata), Arc::new(LocalFileSystem::new()))
        .unwrap()
        .with_options(&DuckLakeWriteOptions::default().with_data_inlining_row_limit(3));
    // Above the limit, so the table has a data file.
    seed(&writer, "t", 1..=4).await;
    let seeded = files_under(data.path());
    let base = head().await;

    let mut transaction = writer.transaction();
    stage(&mut transaction, "t", 10..=11).await;
    assert_eq!(files_under(data.path()), seeded, "two rows are inlined");
    let refused = transaction
        .stage_write("main", "t", &schema(), WriteMode::Append, &[batch(12..=16)])
        .await;
    assert!(
        matches!(refused, Err(DuckLakeError::InvalidConfig(_))),
        "{refused:?}"
    );
    assert_eq!(files_under(data.path()), seeded, "the refused chunk's file");

    // Nor do files take inlined rows after them.
    let mut transaction = writer.transaction();
    stage(&mut transaction, "t", 12..=16).await;
    let refused = transaction
        .stage_write("main", "t", &schema(), WriteMode::Append, &[batch(10..=11)])
        .await;
    assert!(
        matches!(refused, Err(DuckLakeError::InvalidConfig(_))),
        "{refused:?}"
    );
    transaction.abort().await.unwrap();
    assert_eq!(files_under(data.path()), seeded);
    assert_eq!(head().await, base);
}

#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn chunks_of_one_table_leave_another_tables_stage_alone() {
    let (pool, _container) = database().await.unwrap();
    let data = TempDir::new().unwrap();
    let writer = table_writer(&pool, &data).await;
    seed(&writer, "a", 1..=1).await;
    seed(&writer, "b", 1..=1).await;
    let base = head(&pool).await;

    let mut transaction = writer.transaction();
    stage(&mut transaction, "a", 10..=11).await;
    stage(&mut transaction, "b", 10..=10).await;
    stage(&mut transaction, "a", 12..=14).await;
    let results = transaction.commit().await.unwrap();

    assert_eq!(results.len(), 2, "{results:?}");
    assert_eq!(results[0].records_written, 5);
    assert_eq!(results[1].records_written, 1);
    assert_eq!(results[0].snapshot_id, base + 1);
    assert_eq!(results[1].snapshot_id, base + 1);
    assert_eq!(head(&pool).await, base + 1);
    let ctx = session(&pool, &data).await;
    assert_eq!(
        ids(&ctx, "SELECT id FROM lake.main.a ORDER BY id").await,
        [1, 10, 11, 12, 13, 14]
    );
    assert_eq!(
        ids(&ctx, "SELECT id FROM lake.main.b ORDER BY id").await,
        [1, 10]
    );
}
