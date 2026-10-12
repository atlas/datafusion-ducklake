#![cfg(feature = "write-postgres")]
//! Geometry columns written and read the way DuckDB's DuckLake extension does: catalog
//! type `geometry`, WKB in a Parquet `GEOMETRY` column, and a bounding box with the
//! shapes present in `extra_stats`.

use std::sync::Arc;

use arrow::array::{Array, BinaryArray, Int64Array, RecordBatch};
use arrow::datatypes::{DataType, Field, Schema};
use datafusion::prelude::*;
use object_store::local::LocalFileSystem;
use sqlx::postgres::PgPool;
use tempfile::TempDir;
use testcontainers::ContainerAsync;
use testcontainers::runners::AsyncRunner;
use testcontainers_modules::postgres::Postgres;

use datafusion_ducklake::geometry::{GeometryStats, geometry_field};
use datafusion_ducklake::metadata_writer::MetadataWriter;
use datafusion_ducklake::{
    DuckLakeCatalog, DuckLakeTableWriter, LakePool, PostgresMetadataProvider,
    PostgresSingleCatalogMetadataWriter,
};

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

/// WKB for `POINT (x y)`.
fn point(x: f64, y: f64) -> Vec<u8> {
    let mut wkb = vec![1, 1, 0, 0, 0];
    wkb.extend(x.to_le_bytes());
    wkb.extend(y.to_le_bytes());
    wkb
}

/// WKB for `LINESTRING Z` through `points`.
fn linestring_z(points: &[(f64, f64, f64)]) -> Vec<u8> {
    let mut wkb = vec![1];
    wkb.extend(1002u32.to_le_bytes());
    wkb.extend((points.len() as u32).to_le_bytes());
    for (x, y, z) in points {
        wkb.extend(x.to_le_bytes());
        wkb.extend(y.to_le_bytes());
        wkb.extend(z.to_le_bytes());
    }
    wkb
}

fn shapes() -> Vec<Option<Vec<u8>>> {
    vec![Some(point(10.75, 59.5)), None, Some(linestring_z(&[(-3.0, 1.0, 5.0), (2.0, 70.25, 8.0)]))]
}

/// Writes `main.t (id, g)` into the lake in Postgres schema `schema`, with `g` the
/// field given.
async fn write(pool: &PgPool, schema: &str, data: &TempDir, g: Field) -> anyhow::Result<()> {
    let writer =
        PostgresSingleCatalogMetadataWriter::from_lake(LakePool::in_schema(pool.clone(), schema));
    let path = data.path().join(schema);
    std::fs::create_dir_all(&path)?;
    let path = format!("{}/", path.display());
    let writer = tokio::task::spawn_blocking(move || -> anyhow::Result<_> {
        writer.initialize_schema()?;
        writer.set_data_path(&path)?;
        Ok(writer)
    })
    .await??;
    let shapes = shapes();
    let batch = RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            g,
        ])),
        vec![
            Arc::new(Int64Array::from_iter_values(1..=3)),
            Arc::new(BinaryArray::from_iter(shapes.iter().map(Option::as_deref))),
        ],
    )?;
    DuckLakeTableWriter::new(Arc::new(writer), Arc::new(LocalFileSystem::new()))?
        .write_table("main", "t", &[batch])
        .await?;
    Ok(())
}

/// The lake in `schema`, registered as `lake`.
fn context(pool: &PgPool, schema: &str) -> SessionContext {
    let provider = PostgresMetadataProvider::from_lake(LakePool::in_schema(pool.clone(), schema));
    let ctx = SessionContext::new();
    ctx.register_catalog("lake", Arc::new(DuckLakeCatalog::new(provider).unwrap()));
    ctx
}

/// `g` from `main.t`, ordered by id, and the field it came back as.
async fn read_geometry(ctx: &SessionContext) -> (Field, Vec<Option<Vec<u8>>>) {
    let batches = ctx
        .sql("SELECT g FROM lake.main.t ORDER BY id")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let field = batches[0].schema().field(0).clone();
    let values = batches
        .iter()
        .flat_map(|batch| {
            let column = arrow::compute::cast(batch.column(0), &DataType::Binary).unwrap();
            let column = column.as_any().downcast_ref::<BinaryArray>().unwrap();
            (0..column.len())
                .map(|i| column.is_valid(i).then(|| column.value(i).to_vec()))
                .collect::<Vec<_>>()
        })
        .collect();
    (field, values)
}

async fn catalog_stats(pool: &PgPool, schema: &str) -> (String, Option<String>, Option<String>) {
    let mut conn = pool.acquire().await.unwrap();
    sqlx::query("SELECT set_config('search_path', $1, false)")
        .bind(format!("\"{schema}\""))
        .execute(&mut *conn)
        .await
        .unwrap();
    sqlx::query_as(
        "SELECT c.column_type, f.extra_stats, t.extra_stats
         FROM ducklake_column c
         JOIN ducklake_file_column_stats f USING (table_id, column_id)
         JOIN ducklake_table_column_stats t USING (table_id, column_id)
         WHERE c.column_name = 'g' AND c.end_snapshot IS NULL",
    )
    .fetch_one(&mut *conn)
    .await
    .unwrap()
}

fn assert_bounds(extra_stats: &str) {
    let stats = GeometryStats::parse(extra_stats).unwrap();
    let b = stats.bbox;
    assert_eq!(
        (b.xmin, b.xmax, b.ymin, b.ymax, b.zmin, b.zmax, b.mmin),
        (-3.0, 10.75, 1.0, 70.25, Some(5.0), Some(8.0), None)
    );
    assert_eq!(
        stats.types.into_iter().collect::<Vec<_>>(),
        ["linestring_z", "point"]
    );
}

#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn wkb_geometry_is_stored_as_geometry_with_its_bounds() {
    let (pool, _container) = database().await.unwrap();
    let data = TempDir::new().unwrap();
    write(&pool, "lake", &data, geometry_field("g", true))
        .await
        .unwrap();

    let (column_type, file_stats, table_stats) = catalog_stats(&pool, "lake").await;
    assert_eq!(column_type, "geometry");
    assert_bounds(&file_stats.unwrap());
    assert_bounds(&table_stats.unwrap());

    let (field, values) = read_geometry(&context(&pool, "lake")).await;
    assert_eq!(field.extension_type_name(), Some("geoarrow.wkb"));
    assert_eq!(values, shapes());
}

#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn a_crs_on_the_input_is_left_out_of_the_lake() {
    let (pool, _container) = database().await.unwrap();
    let data = TempDir::new().unwrap();
    let mut g = geometry_field("g", true);
    let mut metadata = g.metadata().clone();
    metadata.insert(
        "ARROW:extension:metadata".to_string(),
        r#"{"crs":"EPSG:25833"}"#.to_string(),
    );
    g.set_metadata(metadata);
    write(&pool, "lake", &data, g).await.unwrap();

    // Reads take the catalog's field, so look at what the file itself says.
    let file = parquet_files(data.path()).pop().unwrap();
    let schema = parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder::try_new(
        std::fs::File::open(file).unwrap(),
    )
    .unwrap()
    .schema()
    .clone();
    let g = schema.field_with_name("g").unwrap();
    assert_eq!(g.extension_type_name(), Some("geoarrow.wkb"));
    assert_eq!(
        g.metadata()
            .get("ARROW:extension:metadata")
            .map(String::as_str),
        Some("{}")
    );
}

fn parquet_files(dir: &std::path::Path) -> Vec<std::path::PathBuf> {
    let mut files = Vec::new();
    for entry in std::fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        if path.is_dir() {
            files.extend(parquet_files(&path));
        } else if path.extension().is_some_and(|e| e == "parquet") {
            files.push(path);
        }
    }
    files
}

#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn plain_binary_stays_a_blob() {
    let (pool, _container) = database().await.unwrap();
    let data = TempDir::new().unwrap();
    write(
        &pool,
        "lake",
        &data,
        Field::new("g", DataType::Binary, true),
    )
    .await
    .unwrap();

    let (column_type, file_stats, _) = catalog_stats(&pool, "lake").await;
    assert_eq!(column_type, "blob");
    assert_eq!(file_stats, None);
    let (field, _) = read_geometry(&context(&pool, "lake")).await;
    assert_eq!(field.extension_type_name(), None);
}

#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn other_geoarrow_encodings_are_refused() {
    let (pool, _container) = database().await.unwrap();
    let data = TempDir::new().unwrap();
    let wkt = Field::new("g", DataType::Binary, true).with_metadata(
        [(
            "ARROW:extension:name".to_string(),
            "geoarrow.wkt".to_string(),
        )]
        .into(),
    );
    let error = write(&pool, "lake", &data, wkt).await.unwrap_err();
    assert!(
        error.to_string().contains("geoarrow.wkb"),
        "the error should say which encoding to use: {error}"
    );
}

/// DuckDB attaches a lake we wrote and sees `GEOMETRY`, and we read a lake DuckDB
/// wrote as `geoarrow.wkb` with its bounds.
#[cfg(feature = "metadata-duckdb")]
#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn duckdb_and_we_read_each_others_geometry() {
    let (pool, container) = database().await.unwrap();
    let port = container.get_host_port_ipv4(5432).await.unwrap();
    let data = TempDir::new().unwrap();
    write(&pool, "ours", &data, geometry_field("g", true))
        .await
        .unwrap();

    crate::common::ensure_ducklake_installed();
    let duckdb = duckdb::Connection::open_in_memory().unwrap();
    duckdb
        .execute_batch("INSTALL postgres; LOAD postgres; LOAD ducklake;")
        .unwrap();
    let dsn = format!("host=127.0.0.1 port={port} dbname=postgres user=postgres password=postgres");
    duckdb
        .execute_batch(&format!(
            "ATTACH 'ducklake:postgres:{dsn}' AS ours (METADATA_SCHEMA 'ours', READ_ONLY)"
        ))
        .unwrap();
    let mut read = duckdb
        .prepare("SELECT typeof(g), g::VARCHAR FROM ours.main.t ORDER BY id")
        .unwrap();
    let rows: Vec<(String, Option<String>)> = read
        .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))
        .unwrap()
        .collect::<Result<_, _>>()
        .unwrap();
    assert_eq!(
        rows,
        [
            (
                "GEOMETRY".to_string(),
                Some("POINT (10.75 59.5)".to_string())
            ),
            ("GEOMETRY".to_string(), None),
            (
                "GEOMETRY".to_string(),
                Some("LINESTRING Z (-3 1 5, 2 70.25 8)".to_string())
            ),
        ]
    );

    let theirs = data.path().join("theirs");
    duckdb
        .execute_batch(&format!(
            "ATTACH 'ducklake:postgres:{dsn}' AS theirs
                 (METADATA_SCHEMA 'theirs', DATA_PATH '{}/', DATA_INLINING_ROW_LIMIT 0);
             CREATE TABLE theirs.main.t (id BIGINT, g GEOMETRY);
             INSERT INTO theirs.main.t VALUES
                 (1, 'POINT (10.75 59.5)'::GEOMETRY),
                 (2, NULL),
                 (3, 'LINESTRING Z (-3 1 5, 2 70.25 8)'::GEOMETRY);
             DETACH theirs;",
            theirs.display()
        ))
        .unwrap();
    let (column_type, file_stats, _) = catalog_stats(&pool, "theirs").await;
    assert_eq!(column_type, "geometry");
    assert_bounds(&file_stats.unwrap());
    let (field, values) = read_geometry(&context(&pool, "theirs")).await;
    assert_eq!(field.extension_type_name(), Some("geoarrow.wkb"));
    assert_eq!(values, shapes());
}
