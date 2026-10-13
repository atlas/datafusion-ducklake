#![cfg(feature = "write-postgres")]
//! Data inlining on the single-catalog Postgres writer: a write below the
//! table's inlining limit becomes rows in the catalog, a larger one a Parquet
//! file, and a Replace or DELETE ends inlined rows instead of removing them.

use std::sync::Arc;

use arrow::array::{Int64Array, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use datafusion::prelude::*;
use object_store::local::LocalFileSystem;
use sqlx::postgres::PgPool;
use tempfile::TempDir;
use testcontainers::ContainerAsync;
use testcontainers::runners::AsyncRunner;
use testcontainers_modules::postgres::Postgres;

use datafusion_ducklake::metadata_writer::MetadataWriter;
use datafusion_ducklake::{
    DuckLakeCatalog, DuckLakeTableWriter, DuckLakeWriteOptions, LakePool, MetadataProvider,
    PostgresMetadataProvider, PostgresSingleCatalogMetadataWriter, register_ducklake_functions,
};

const LAKE: &str = "lake";

struct Lake {
    pool: PgPool,
    data: TempDir,
    container: ContainerAsync<Postgres>,
}

impl Lake {
    async fn start() -> Self {
        let container = Postgres::default().start().await.unwrap();
        let port = container.get_host_port_ipv4(5432).await.unwrap();
        let pool = LakePool::connect(
            &format!("postgresql://postgres:postgres@127.0.0.1:{port}/postgres"),
            4,
        )
        .await
        .unwrap();
        Self {
            pool,
            data: TempDir::new().unwrap(),
            container,
        }
    }

    async fn writer(&self) -> Arc<PostgresSingleCatalogMetadataWriter> {
        let writer = PostgresSingleCatalogMetadataWriter::from_lake(LakePool::in_schema(
            self.pool.clone(),
            LAKE,
        ));
        let path = format!("{}/", self.data.path().display());
        let writer = tokio::task::spawn_blocking(move || {
            writer.initialize_schema().unwrap();
            writer.set_data_path(&path).unwrap();
            writer
        })
        .await
        .unwrap();
        Arc::new(writer)
    }

    /// A table writer that inlines writes of up to `limit` rows.
    async fn table_writer(&self, limit: usize) -> DuckLakeTableWriter {
        DuckLakeTableWriter::new(self.writer().await, Arc::new(LocalFileSystem::new()))
            .unwrap()
            .with_options(&DuckLakeWriteOptions::default().with_data_inlining_row_limit(limit))
    }

    fn provider(&self) -> Arc<dyn MetadataProvider> {
        Arc::new(PostgresMetadataProvider::from_lake(LakePool::in_schema(
            self.pool.clone(),
            LAKE,
        )))
    }

    /// A session over the lake as `lake`, at `snapshot` or the latest.
    async fn session(&self, snapshot: Option<i64>) -> SessionContext {
        let provider = self.provider();
        let catalog = match snapshot {
            Some(snapshot) => {
                let provider = provider.clone();
                tokio::task::spawn_blocking(move || {
                    DuckLakeCatalog::with_snapshot(provider, snapshot).unwrap()
                })
                .await
                .unwrap()
            },
            None => DuckLakeCatalog::with_writer(provider.clone(), self.writer().await).unwrap(),
        };
        let ctx = SessionContext::new();
        ctx.register_catalog("lake", Arc::new(catalog));
        register_ducklake_functions(&ctx, provider);
        ctx
    }

    async fn rows(&self, snapshot: Option<i64>) -> Vec<(i64, String)> {
        let batches = self
            .session(snapshot)
            .await
            .sql("SELECT id, name FROM lake.main.t ORDER BY id")
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
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

    async fn head(&self) -> i64 {
        sqlx::query_scalar("SELECT MAX(snapshot_id) FROM lake.ducklake_snapshot")
            .fetch_one(&self.pool)
            .await
            .unwrap()
    }

    async fn changes(&self, snapshot: i64) -> String {
        sqlx::query_scalar(
            "SELECT changes_made FROM lake.ducklake_snapshot_changes WHERE snapshot_id = $1",
        )
        .bind(snapshot)
        .fetch_one(&self.pool)
        .await
        .unwrap()
    }

    async fn data_files(&self) -> i64 {
        sqlx::query_scalar("SELECT COUNT(*) FROM lake.ducklake_data_file")
            .fetch_one(&self.pool)
            .await
            .unwrap()
    }

    /// `(id, end_snapshot)` of every inlined row of `main.t`, by row id.
    async fn inlined(&self) -> Vec<(i64, Option<i64>)> {
        let table: String = sqlx::query_scalar(
            "SELECT table_name FROM lake.ducklake_inlined_data_tables ORDER BY schema_version DESC",
        )
        .fetch_one(&self.pool)
        .await
        .unwrap();
        sqlx::query_as(sqlx::AssertSqlSafe(format!(
            "SELECT id, end_snapshot FROM lake.\"{table}\" ORDER BY row_id"
        )))
        .fetch_all(&self.pool)
        .await
        .unwrap()
    }
}

fn batch(ids: &[i64]) -> RecordBatch {
    RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("name", DataType::Utf8, true),
        ])),
        vec![
            Arc::new(Int64Array::from(ids.to_vec())),
            Arc::new(StringArray::from_iter_values(
                ids.iter().map(|id| format!("n{id}")),
            )),
        ],
    )
    .unwrap()
}

fn named(ids: &[i64]) -> Vec<(i64, String)> {
    ids.iter().map(|id| (*id, format!("n{id}"))).collect()
}

#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn a_write_below_the_limit_is_inlined_and_a_larger_one_is_a_file() {
    let lake = Lake::start().await;
    let writer = lake.table_writer(3).await;

    let small = writer
        .write_table("main", "t", &[batch(&[1, 2])])
        .await
        .unwrap();
    assert_eq!(small.files_written, 0);
    assert_eq!(lake.data_files().await, 0);
    let inlined_at = lake.head().await;
    assert!(
        lake.changes(inlined_at)
            .await
            .contains(&format!("inlined_insert:{}", small.table_id)),
        "{}",
        lake.changes(inlined_at).await
    );

    let large = writer
        .append_table("main", "t", &[batch(&[3, 4, 5, 6])])
        .await
        .unwrap();
    assert_eq!(large.files_written, 1);
    assert_eq!(lake.rows(None).await, named(&[1, 2, 3, 4, 5, 6]));
    assert_eq!(lake.rows(Some(inlined_at)).await, named(&[1, 2]));
}

#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn a_replace_ends_the_inlined_rows_it_replaces() {
    let lake = Lake::start().await;
    let writer = lake.table_writer(3).await;
    writer
        .write_table("main", "t", &[batch(&[1, 2])])
        .await
        .unwrap();
    let before = lake.head().await;

    // Inlined over inlined, then a file over that.
    writer
        .write_table("main", "t", &[batch(&[3])])
        .await
        .unwrap();
    let replaced_inline = lake.head().await;
    assert_eq!(lake.rows(None).await, named(&[3]));
    assert_eq!(
        lake.inlined().await,
        [(1, Some(replaced_inline)), (2, Some(replaced_inline)), (3, None)]
    );
    assert!(
        lake.changes(replaced_inline)
            .await
            .contains("inlined_delete:"),
        "{}",
        lake.changes(replaced_inline).await
    );

    writer
        .write_table("main", "t", &[batch(&[4, 5, 6, 7])])
        .await
        .unwrap();
    assert_eq!(lake.rows(None).await, named(&[4, 5, 6, 7]));
    assert_eq!(lake.rows(Some(before)).await, named(&[1, 2]));
    assert_eq!(lake.rows(Some(replaced_inline)).await, named(&[3]));
}

#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn sql_inserts_follow_the_tables_inlining_setting_and_deletes_end_rows() {
    let lake = Lake::start().await;
    let created = lake
        .table_writer(0)
        .await
        .write_table("main", "t", &[batch(&[1, 2, 3, 4])])
        .await
        .unwrap();
    let writer = lake.writer().await;
    let table_id = created.table_id;
    tokio::task::spawn_blocking(move || {
        writer
            .set_table_setting(table_id, "data_inlining_row_limit", "10")
            .unwrap()
    })
    .await
    .unwrap();

    let ctx = lake.session(None).await;
    ctx.sql("INSERT INTO lake.main.t VALUES (5, 'n5'), (6, 'n6')")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    assert_eq!(lake.data_files().await, 1);
    let inserted = lake.head().await;

    lake.session(None)
        .await
        .sql("DELETE FROM lake.main.t WHERE id IN (2, 5)")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let deleted = lake.head().await;
    assert_eq!(lake.inlined().await, [(5, Some(deleted)), (6, None)]);
    assert_eq!(lake.rows(None).await, named(&[1, 3, 4, 6]));
    assert_eq!(lake.rows(Some(inserted)).await, named(&[1, 2, 3, 4, 5, 6]));
}

#[cfg(feature = "metadata-duckdb")]
#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn duckdb_reads_our_inlined_rows() {
    let lake = Lake::start().await;
    let writer = lake.table_writer(3).await;
    writer
        .write_table("main", "t", &[batch(&[1, 2, 3, 4])])
        .await
        .unwrap();
    writer
        .append_table("main", "t", &[batch(&[5, 6])])
        .await
        .unwrap();

    let port = lake.container.get_host_port_ipv4(5432).await.unwrap();
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
    assert_eq!(read, named(&[1, 2, 3, 4, 5, 6]));
}
