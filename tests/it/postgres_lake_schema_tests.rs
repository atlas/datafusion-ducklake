#![cfg(feature = "write-postgres")]
//! Many single-catalog lakes in one Postgres database, each in its own schema,
//! sharing one connection pool through [`LakePool`].

use std::sync::Arc;
use std::sync::mpsc;
use std::time::Duration;

use arrow::array::Int64Array;
use datafusion::prelude::*;
use sqlx::postgres::{PgPool, PgPoolOptions};
use tempfile::TempDir;
use testcontainers::ContainerAsync;
use testcontainers::runners::AsyncRunner;
use testcontainers_modules::postgres::Postgres;

use datafusion_ducklake::metadata_writer::MetadataWriter;
use datafusion_ducklake::{
    DuckLakeCatalog, LakePool, PostgresMetadataProvider, PostgresSingleCatalogMetadataWriter,
};

async fn database() -> anyhow::Result<(PgPool, ContainerAsync<Postgres>)> {
    let container = Postgres::default().start().await?;
    let port = container.get_host_port_ipv4(5432).await?;
    let pool = PgPoolOptions::new()
        .max_connections(4)
        .connect(&format!(
            "postgresql://postgres:postgres@127.0.0.1:{port}/postgres"
        ))
        .await?;
    Ok((pool, container))
}

/// Creates a lake with a `main` schema in Postgres schema `schema`.
fn lake(pool: &PgPool, schema: &str, data: &TempDir) -> PostgresSingleCatalogMetadataWriter {
    let writer =
        PostgresSingleCatalogMetadataWriter::from_lake(LakePool::in_schema(pool.clone(), schema));
    writer.initialize_schema().unwrap();
    let path = data.path().join(schema);
    std::fs::create_dir_all(&path).unwrap();
    writer.set_data_path(path.to_str().unwrap()).unwrap();
    let snapshot = writer.create_snapshot().unwrap();
    writer.get_or_create_schema("main", None, snapshot).unwrap();
    writer
}

async fn sql(pool: &PgPool, schema: &str, sql: &str) -> Vec<arrow::record_batch::RecordBatch> {
    let lake = LakePool::in_schema(pool.clone(), schema);
    let catalog = DuckLakeCatalog::with_writer(
        Arc::new(PostgresMetadataProvider::from_lake(lake.clone())),
        Arc::new(PostgresSingleCatalogMetadataWriter::from_lake(lake)),
    )
    .unwrap();
    let ctx = SessionContext::new();
    ctx.register_catalog("lake", Arc::new(catalog));
    ctx.sql(sql).await.unwrap().collect().await.unwrap()
}

async fn count(pool: &PgPool, schema: &str) -> i64 {
    let batches = sql(pool, schema, "SELECT count(*) FROM lake.main.t").await;
    batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap()
        .value(0)
}

#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn lakes_sharing_a_pool_see_only_their_own_tables() {
    let (pool, _container) = database().await.unwrap();
    let data = TempDir::new().unwrap();
    lake(&pool, "tenant a", &data);
    lake(&pool, "tenant_b", &data);

    sql(&pool, "tenant a", "CREATE TABLE lake.main.t (id BIGINT)").await;
    sql(
        &pool,
        "tenant_b",
        "CREATE TABLE lake.main.t (id BIGINT, name VARCHAR)",
    )
    .await;
    sql(&pool, "tenant a", "INSERT INTO lake.main.t VALUES (1)").await;
    sql(
        &pool,
        "tenant_b",
        "INSERT INTO lake.main.t VALUES (1, 'x'), (2, 'y'), (3, 'z')",
    )
    .await;

    assert_eq!(count(&pool, "tenant a").await, 1);
    assert_eq!(count(&pool, "tenant_b").await, 3);

    let outside: i64 = sqlx::query_scalar(
        "SELECT count(*) FROM information_schema.tables
         WHERE table_name LIKE 'ducklake%' AND table_schema NOT IN ('tenant a', 'tenant_b')",
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    assert_eq!(
        outside, 0,
        "a lake's catalog tables belong in its own schema"
    );
}

#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn a_commit_lock_does_not_block_another_lake() {
    let (pool, _container) = database().await.unwrap();
    let data = TempDir::new().unwrap();
    let a = lake(&pool, "tenant_a", &data);
    let b = lake(&pool, "tenant_b", &data);

    let (held, holding) = mpsc::channel();
    let (release, released) = mpsc::channel::<()>();
    let holder = std::thread::spawn(move || {
        a.with_commit_lock(
            "main.t",
            Box::new(move || {
                held.send(()).unwrap();
                let _ = released.recv_timeout(Duration::from_secs(30));
                Ok(())
            }),
        )
    });
    holding.recv_timeout(Duration::from_secs(30)).unwrap();

    let (done, finished) = mpsc::channel();
    std::thread::spawn(move || {
        let result = b.with_commit_lock("main.t", Box::new(|| Ok(())));
        let _ = done.send(result);
    });
    let other = finished.recv_timeout(Duration::from_secs(5));
    release.send(()).unwrap();
    holder.join().unwrap().unwrap();
    other
        .expect("the same table identity in another lake waited for this lake's lock")
        .unwrap();
}
