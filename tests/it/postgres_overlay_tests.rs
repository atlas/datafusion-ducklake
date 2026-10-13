#![cfg(feature = "write-postgres")]
//! Overlays read from a lake in a Postgres schema of its own: the underlay as
//! it was at the base, less what the overlay hides, plus the overlay's rows.

use std::sync::Arc;

use arrow::array::{Int64Array, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use datafusion::catalog::{CatalogProvider, TableProvider};
use datafusion::prelude::*;
use object_store::local::LocalFileSystem;
use sqlx::postgres::PgPool;
use tempfile::TempDir;
use testcontainers::ContainerAsync;
use testcontainers::runners::AsyncRunner;
use testcontainers_modules::postgres::Postgres;

use datafusion_ducklake::metadata_writer::MetadataWriter;
use datafusion_ducklake::overlay::{
    DuckLakeOverlayTable, OverlayBase, OverlayBaseProvider, create_overlay, flush_overlay,
    plan_rebase, rebase_overlay,
};
use datafusion_ducklake::overlay_postgres::{initialize_overlay_schema, load_overlay_base};
use datafusion_ducklake::{
    DuckLakeCatalog, DuckLakeTable, DuckLakeTableWriter, LakePool, MetadataProvider,
    PostgresMetadataProvider, PostgresSingleCatalogMetadataWriter, register_ducklake_functions,
};

const LAKE: &str = "lake";

struct Lake {
    pool: PgPool,
    data: TempDir,
    _container: ContainerAsync<Postgres>,
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
        let lake = Self {
            pool,
            data: TempDir::new().unwrap(),
            _container: container,
        };
        let writer = lake.writer().await;
        drop(writer);
        initialize_overlay_schema(&lake.lake_pool()).await.unwrap();
        lake
    }

    fn lake_pool(&self) -> LakePool {
        LakePool::in_schema(self.pool.clone(), LAKE)
    }

    async fn writer(&self) -> Arc<PostgresSingleCatalogMetadataWriter> {
        let writer = PostgresSingleCatalogMetadataWriter::from_lake(self.lake_pool());
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

    fn table_writer(
        &self,
        writer: Arc<PostgresSingleCatalogMetadataWriter>,
    ) -> DuckLakeTableWriter {
        DuckLakeTableWriter::new(writer, Arc::new(LocalFileSystem::new())).unwrap()
    }

    /// Writes `main.<table> (id, name)` as one data file.
    async fn write(&self, table: &str, ids: &[i64], names: &[&str]) {
        let batch = RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new("id", DataType::Int64, false),
                Field::new("name", DataType::Utf8, true),
            ])),
            vec![
                Arc::new(Int64Array::from(ids.to_vec())),
                Arc::new(StringArray::from(names.to_vec())),
            ],
        )
        .unwrap();
        self.table_writer(self.writer().await)
            .write_table("main", table, &[batch])
            .await
            .unwrap();
    }

    fn provider(&self) -> Arc<dyn MetadataProvider> {
        Arc::new(PostgresMetadataProvider::from_lake(self.lake_pool()))
    }

    async fn sql(&self, sql: &str) {
        let catalog = DuckLakeCatalog::with_writer(self.provider(), self.writer().await).unwrap();
        let ctx = SessionContext::new();
        ctx.register_catalog("lake", Arc::new(catalog));
        register_ducklake_functions(&ctx, self.provider());
        ctx.sql(sql).await.unwrap().collect().await.unwrap();
    }

    async fn head(&self) -> i64 {
        sqlx::query_scalar("SELECT MAX(snapshot_id) FROM lake.ducklake_snapshot")
            .fetch_one(&self.pool)
            .await
            .unwrap()
    }

    async fn table_id(&self, name: &str) -> i64 {
        sqlx::query_scalar(
            "SELECT table_id FROM lake.ducklake_table WHERE table_name = $1 AND end_snapshot IS NULL",
        )
        .bind(name)
        .fetch_one(&self.pool)
        .await
        .unwrap()
    }

    /// The table's one data file live at `snapshot`: its id and path.
    async fn data_file(&self, table_id: i64, snapshot: i64) -> (i64, String) {
        sqlx::query_as(
            "SELECT data_file_id, path FROM lake.ducklake_data_file
             WHERE table_id = $1 AND begin_snapshot <= $2
               AND (end_snapshot IS NULL OR $2 < end_snapshot)",
        )
        .bind(table_id)
        .bind(snapshot)
        .fetch_one(&self.pool)
        .await
        .unwrap()
    }

    /// Records `overlay` as an overlay of `underlay` at `base`, from `begin`.
    async fn make_overlay(&self, overlay: i64, underlay: i64, base: i64, begin: i64) {
        sqlx::query(
            "INSERT INTO lake.ducklake_overlay
                 (overlay_table_id, underlay_table_id, base_snapshot, begin_snapshot)
             VALUES ($1, $2, $3, $4)",
        )
        .bind(overlay)
        .bind(underlay)
        .bind(base)
        .bind(begin)
        .execute(&self.pool)
        .await
        .unwrap();
    }

    async fn hide(
        &self,
        overlay: i64,
        data_file: i64,
        position: i64,
        begin: i64,
        end: Option<i64>,
    ) {
        sqlx::query(
            "INSERT INTO lake.ducklake_overlay_inlined_delete
                 (overlay_table_id, data_file_id, row_id, begin_snapshot, end_snapshot)
             VALUES ($1, $2, $3, $4, $5)",
        )
        .bind(overlay)
        .bind(data_file)
        .bind(position)
        .bind(begin)
        .bind(end)
        .execute(&self.pool)
        .await
        .unwrap();
    }

    /// A table as a plain catalog at `snapshot` sees it.
    async fn table_at(
        &self,
        provider: Arc<dyn MetadataProvider>,
        snapshot: i64,
        name: &str,
    ) -> Arc<DuckLakeTable> {
        let catalog = tokio::task::spawn_blocking(move || {
            DuckLakeCatalog::with_snapshot(provider, snapshot).unwrap()
        })
        .await
        .unwrap();
        let table: Arc<dyn TableProvider> = catalog
            .schema("main")
            .unwrap()
            .table(name)
            .await
            .unwrap()
            .unwrap();
        let table = (table.as_ref() as &dyn std::any::Any)
            .downcast_ref::<DuckLakeTable>()
            .unwrap();
        Arc::new(table.clone())
    }

    /// The overlay table `name` read at `snapshot`.
    async fn overlay(&self, name: &str, snapshot: i64) -> DuckLakeOverlayTable {
        let id = self.table_id(name).await;
        let base = load_overlay_base(&self.lake_pool(), id, snapshot)
            .await
            .unwrap()
            .expect("an overlay");
        let base_snapshot = base.base_snapshot;
        let underlay_name: String =
            sqlx::query_scalar("SELECT table_name FROM lake.ducklake_table WHERE table_id = $1")
                .bind(base.underlay_table_id)
                .fetch_one(&self.pool)
                .await
                .unwrap();
        let base = Arc::new(base);
        let base_provider = Arc::new(OverlayBaseProvider::new(self.provider(), base.clone()));
        let underlay = self
            .table_at(base_provider, base_snapshot, &underlay_name)
            .await;
        let overlay = self.table_at(self.provider(), snapshot, name).await;
        DuckLakeOverlayTable::try_new(base, underlay, overlay).unwrap()
    }
}

impl Lake {
    /// The overlay `name` as it is now, writable.
    async fn writable_overlay(&self, name: &str) -> DuckLakeOverlayTable {
        let provider = self.provider();
        let writer = self.writer().await;
        let catalog = DuckLakeCatalog::with_writer(provider.clone(), writer).unwrap();
        let table: Arc<dyn TableProvider> = catalog
            .schema("main")
            .unwrap()
            .table(name)
            .await
            .unwrap()
            .unwrap();
        let overlay = (table.as_ref() as &dyn std::any::Any)
            .downcast_ref::<DuckLakeTable>()
            .unwrap()
            .clone();
        let base = load_overlay_base(
            &self.lake_pool(),
            self.table_id(name).await,
            overlay.snapshot(),
        )
        .await
        .unwrap()
        .expect("an overlay");
        let (base_snapshot, underlay_id) = (base.base_snapshot, base.underlay_table_id);
        let underlay_name: String =
            sqlx::query_scalar("SELECT table_name FROM lake.ducklake_table WHERE table_id = $1")
                .bind(underlay_id)
                .fetch_one(&self.pool)
                .await
                .unwrap();
        let base = Arc::new(base);
        let base_provider = Arc::new(OverlayBaseProvider::new(provider, base.clone()));
        let underlay = self
            .table_at(base_provider, base_snapshot, &underlay_name)
            .await;
        DuckLakeOverlayTable::try_new(base, underlay, Arc::new(overlay)).unwrap()
    }

    /// Runs one DML statement on the overlay `name` as `ov`; returns its count.
    async fn change(&self, name: &str, sql: &str) -> datafusion::error::Result<u64> {
        let ctx = SessionContext::new();
        ctx.register_table("ov", Arc::new(self.writable_overlay(name).await))
            .unwrap();
        let batches = ctx.sql(sql).await?.collect().await?;
        Ok(batches[0]
            .column(0)
            .as_any()
            .downcast_ref::<arrow::array::UInt64Array>()
            .unwrap()
            .value(0))
    }

    /// Creates the overlay `name` of `main.t` as it is now.
    async fn create(&self, name: &str) -> (i64, i64) {
        let underlay = self.table_at(self.provider(), self.head().await, "t").await;
        let writer = self.writer().await;
        let (snapshot, table_id) =
            create_overlay(&self.table_writer(writer.clone()), "main", name, &underlay)
                .await
                .unwrap();
        tokio::task::spawn_blocking(move || {
            writer
                .set_table_setting(table_id, "data_inlining_row_limit", "10")
                .unwrap()
        })
        .await
        .unwrap();
        (snapshot, table_id)
    }

    /// Flushes the overlay `o` as it is now.
    async fn flush(&self) -> Option<i64> {
        let table = self.writable_overlay("o").await;
        let writer = self.writer().await;
        let ctx = SessionContext::new();
        flush_overlay(&table, &self.table_writer(writer), "main", &ctx.state())
            .await
            .unwrap()
    }

    /// Plans rebasing the overlay `name` onto the underlay's head, and
    /// returns the plan's conflicting ids, sorted.
    async fn conflicts(&self, name: &str) -> Vec<i64> {
        let (plan, _, _) = self.plan(name).await;
        let mut ids = plan
            .conflicts
            .iter()
            .map(|key| match key {
                datafusion::common::ScalarValue::Int64(Some(id)) => *id,
                other => panic!("an id key, not {other:?}"),
            })
            .collect::<Vec<_>>();
        ids.sort_unstable();
        ids
    }

    async fn plan(
        &self,
        name: &str,
    ) -> (
        datafusion_ducklake::overlay::OverlayRebase,
        DuckLakeOverlayTable,
        Arc<DuckLakeTable>,
    ) {
        let table = self.writable_overlay(name).await;
        let base = table.base().base_snapshot;
        let old = self.table_at(self.provider(), base, "t").await;
        let new = self.table_at(self.provider(), self.head().await, "t").await;
        let ctx = SessionContext::new();
        let plan = plan_rebase(&table, &old, &new, "id", &ctx.state())
            .await
            .unwrap();
        (plan, table, new)
    }

    /// Rebases the overlay `name` onto the underlay's head, taking the
    /// underlay's version of `take_underlay`.
    async fn rebase(&self, name: &str, take_underlay: &[i64]) -> datafusion_ducklake::Result<i64> {
        let (plan, table, new) = self.plan(name).await;
        self.apply(&table, &plan, &new, take_underlay).await
    }

    async fn apply(
        &self,
        table: &DuckLakeOverlayTable,
        plan: &datafusion_ducklake::overlay::OverlayRebase,
        new: &DuckLakeTable,
        take_underlay: &[i64],
    ) -> datafusion_ducklake::Result<i64> {
        let take = take_underlay
            .iter()
            .map(|id| datafusion::common::ScalarValue::Int64(Some(*id)))
            .collect();
        let ctx = SessionContext::new();
        rebase_overlay(
            table,
            plan,
            new,
            &take,
            &self.table_writer(self.writer().await),
            &ctx.state(),
        )
        .await
    }

    async fn count(&self, sql: &str) -> i64 {
        sqlx::query_scalar(sqlx::AssertSqlSafe(sql.to_string()))
            .fetch_one(&self.pool)
            .await
            .unwrap()
    }
}

async fn rows(table: impl TableProvider + 'static) -> Vec<(i64, String)> {
    let ctx = SessionContext::new();
    ctx.register_table("t", Arc::new(table)).unwrap();
    let batches = ctx
        .sql("SELECT id, name FROM t ORDER BY id")
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

fn expected(rows: &[(i64, &str)]) -> Vec<(i64, String)> {
    rows.iter()
        .map(|(id, name)| (*id, name.to_string()))
        .collect()
}

#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn an_overlay_reads_its_base_less_what_it_hides_plus_its_own_rows() {
    let lake = Lake::start().await;
    lake.write("t", &[1, 2, 3, 4], &["a", "b", "c", "d"]).await;
    let underlay = lake.table_id("t").await;
    let base = lake.head().await;
    let (file, _) = lake.data_file(underlay, base).await;

    // The overlay changes record 2 and adds record 5.
    lake.write("o", &[2, 5], &["b2", "e"]).await;
    let overlay = lake.table_id("o").await;
    let created = lake.head().await;
    lake.make_overlay(overlay, underlay, base, created).await;
    lake.hide(overlay, file, 1, created, None).await;

    // The underlay moves on after the base: the overlay does not see it.
    lake.sql("DELETE FROM lake.main.t WHERE id = 3").await;
    lake.sql("INSERT INTO lake.main.t VALUES (6, 'f')").await;

    assert_eq!(
        rows(lake.overlay("o", lake.head().await).await).await,
        expected(&[(1, "a"), (2, "b2"), (3, "c"), (4, "d"), (5, "e")])
    );
}

#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn an_overlay_delete_file_takes_the_place_of_the_underlays() {
    let lake = Lake::start().await;
    lake.write("t", &[1, 2, 3, 4], &["a", "b", "c", "d"]).await;
    let underlay = lake.table_id("t").await;
    // At the base the underlay has its own delete file, hiding record 4.
    lake.sql("DELETE FROM lake.main.t WHERE id = 4").await;
    let base = lake.head().await;
    let (file, path) = lake.data_file(underlay, base).await;

    lake.write("o", &[9], &["i"]).await;
    let overlay = lake.table_id("o").await;
    let created = lake.head().await;
    lake.make_overlay(overlay, underlay, base, created).await;

    // The overlay hides record 1. Its delete file holds the underlay's
    // deletes at the base as well, so record 4 stays hidden.
    let written = lake
        .table_writer(lake.writer().await)
        .write_delete_file("main", "t", &path, &[0, 3])
        .await
        .unwrap();
    sqlx::query(
        "INSERT INTO lake.ducklake_overlay_delete_file
             (overlay_table_id, data_file_id, path, path_is_relative, file_size_bytes,
              footer_size, delete_count, begin_snapshot)
         VALUES ($1, $2, $3, $4, $5, $6, $7, $8)",
    )
    .bind(overlay)
    .bind(file)
    .bind(&written.path)
    .bind(written.path_is_relative)
    .bind(written.file_size_bytes)
    .bind(written.footer_size)
    .bind(written.delete_count)
    .bind(created)
    .execute(&lake.pool)
    .await
    .unwrap();

    assert_eq!(
        rows(lake.overlay("o", created).await).await,
        expected(&[(2, "b"), (3, "c"), (9, "i")])
    );
}

#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn an_overlay_reads_as_of_its_own_earlier_snapshots() {
    let lake = Lake::start().await;
    lake.write("t", &[1, 2], &["a", "b"]).await;
    let underlay = lake.table_id("t").await;
    let base = lake.head().await;
    let (file, _) = lake.data_file(underlay, base).await;

    lake.write("o", &[3], &["c"]).await;
    let overlay = lake.table_id("o").await;
    let first = lake.head().await;
    lake.make_overlay(overlay, underlay, base, first).await;

    // A later overlay snapshot hides record 1; one after that restores it.
    lake.sql("INSERT INTO lake.main.o VALUES (4, 'd')").await;
    let second = lake.head().await;
    lake.sql("INSERT INTO lake.main.o VALUES (5, 'e')").await;
    let third = lake.head().await;
    lake.hide(overlay, file, 0, second, Some(third)).await;

    assert_eq!(
        rows(lake.overlay("o", first).await).await,
        expected(&[(1, "a"), (2, "b"), (3, "c")])
    );
    assert_eq!(
        rows(lake.overlay("o", second).await).await,
        expected(&[(2, "b"), (3, "c"), (4, "d")])
    );
    assert_eq!(
        rows(lake.overlay("o", third).await).await,
        expected(&[(1, "a"), (2, "b"), (3, "c"), (4, "d"), (5, "e")])
    );
}

#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn a_table_that_is_not_an_overlay_has_no_base() {
    let lake = Lake::start().await;
    lake.write("t", &[1], &["a"]).await;
    let table = lake.table_id("t").await;
    let head = lake.head().await;
    assert!(
        load_overlay_base(&lake.lake_pool(), table, head)
            .await
            .unwrap()
            .is_none()
    );
}

#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn an_overlay_refuses_columns_that_differ_from_its_underlays() {
    let lake = Lake::start().await;
    lake.write("t", &[1], &["a"]).await;
    let head = lake.head().await;
    let batch = RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, false)])),
        vec![Arc::new(Int64Array::from(vec![2]))],
    )
    .unwrap();
    lake.table_writer(lake.writer().await)
        .write_table("main", "o", &[batch])
        .await
        .unwrap();
    let underlay = lake.table_at(lake.provider(), head, "t").await;
    let overlay = lake.table_at(lake.provider(), lake.head().await, "o").await;
    let base = |table_id, base_snapshot| {
        Arc::new(OverlayBase {
            underlay_table_id: table_id,
            base_snapshot,
            inlined_deletes: Vec::new(),
            inlined_row_deletes: Default::default(),
            delete_files: std::collections::HashMap::new(),
        })
    };
    let t = lake.table_id("t").await;
    assert!(
        DuckLakeOverlayTable::try_new(base(t, head), underlay.clone(), overlay.clone()).is_err(),
        "the columns differ"
    );
    assert!(
        DuckLakeOverlayTable::try_new(base(t, head - 1), underlay, overlay).is_err(),
        "the underlay is not read at the base"
    );
}

/// An overlay's base is read on the catalog runtime, as every other use of a
/// lake's pool is. A connection it opened on a single-threaded caller's runtime
/// would be driven only while that thread awaits, so a synchronous catalog call
/// on that thread that is handed the connection would wait forever.
#[test]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
fn an_overlay_base_is_read_on_the_catalog_runtime() {
    let setup = tokio::runtime::Runtime::new().unwrap();
    let lake = setup.block_on(Lake::start());
    setup.block_on(lake.write("t", &[1], &["a"]));
    let table = setup.block_on(lake.table_id("t"));
    let port = setup
        .block_on(lake._container.get_host_port_ipv4(5432))
        .unwrap();
    let url = format!("postgresql://postgres:postgres@127.0.0.1:{port}/postgres");

    let (done, finished) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let caller = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        caller.block_on(async {
            // Two connections, one of them open.
            let shared = LakePool::connect(&url, 2).await.unwrap();
            let pool = LakePool::in_schema(shared.clone(), LAKE);
            let provider = PostgresMetadataProvider::from_lake(pool.clone());
            let head = provider.get_current_snapshot().unwrap();
            // With the open one taken, reading the base opens the second.
            let held = shared.acquire().await.unwrap();
            assert!(
                load_overlay_base(&pool, table, head)
                    .await
                    .unwrap()
                    .is_none()
            );
            // Then a synchronous call on this thread is handed the second.
            let read = provider.get_current_snapshot();
            drop(held);
            done.send(read.map(|_| ()).map_err(|e| e.to_string()))
                .unwrap();
        });
    });
    let read = finished.recv_timeout(std::time::Duration::from_secs(60));
    // Removed on a runtime, before any assertion can unwind past it.
    setup.block_on(async move { drop(lake) });
    read.expect("the catalog call finishes")
        .expect("the catalog call reads");
}

#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn an_overlay_takes_inserts_updates_and_deletes_without_touching_its_underlay() {
    let lake = Lake::start().await;
    lake.write("t", &[1, 2, 3, 4], &["a", "b", "c", "d"]).await;
    let underlay_files = lake
        .count("SELECT COUNT(*) FROM lake.ducklake_data_file")
        .await;
    let (created, overlay) = lake.create("o").await;
    let underlay = lake.table_id("t").await;
    assert_eq!(
        lake.count(&format!(
            "SELECT COUNT(*) FROM lake.ducklake_overlay WHERE overlay_table_id = {overlay}
               AND underlay_table_id = {underlay} AND begin_snapshot = {created}"
        ))
        .await,
        1,
        "the table and its overlay row begin in one snapshot"
    );

    assert_eq!(
        lake.change("o", "UPDATE ov SET name = 'B' WHERE id = 2")
            .await
            .unwrap(),
        1
    );
    assert_eq!(
        lake.change("o", "DELETE FROM ov WHERE id = 3")
            .await
            .unwrap(),
        1
    );
    assert_eq!(
        lake.change("o", "INSERT INTO ov VALUES (5, 'e')")
            .await
            .unwrap(),
        1
    );
    // Rows the overlay already holds change in the overlay table only.
    assert_eq!(
        lake.change("o", "UPDATE ov SET name = 'BB' WHERE id = 2")
            .await
            .unwrap(),
        1
    );
    assert_eq!(
        lake.change("o", "DELETE FROM ov WHERE id = 5")
            .await
            .unwrap(),
        1
    );
    // Nothing matches twice.
    assert_eq!(
        lake.change("o", "DELETE FROM ov WHERE id = 3")
            .await
            .unwrap(),
        0
    );

    assert_eq!(
        rows(lake.overlay("o", lake.head().await).await).await,
        expected(&[(1, "a"), (2, "BB"), (4, "d")])
    );
    assert_eq!(
        lake.count(&format!(
            "SELECT COUNT(*) FROM lake.ducklake_overlay_inlined_delete
             WHERE overlay_table_id = {overlay} AND end_snapshot IS NULL"
        ))
        .await,
        2,
        "two underlay rows hidden, each once"
    );
    assert_eq!(
        lake.count("SELECT COUNT(*) FROM lake.ducklake_data_file")
            .await,
        underlay_files,
        "small overlay edits are inlined"
    );
    // The underlay is untouched.
    let head = lake.head().await;
    let plain = lake.table_at(lake.provider(), head, "t").await;
    assert_eq!(
        rows(DuckLakeTable::clone(&plain)).await,
        expected(&[(1, "a"), (2, "b"), (3, "c"), (4, "d")])
    );
}

#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn an_update_reaching_both_parts_commits_one_snapshot() {
    let lake = Lake::start().await;
    lake.write("t", &[1, 2, 3], &["a", "b", "c"]).await;
    lake.create("o").await;
    lake.change("o", "INSERT INTO ov VALUES (9, 'i')")
        .await
        .unwrap();
    let before = lake.head().await;

    assert_eq!(
        lake.change("o", "UPDATE ov SET name = 'x' WHERE id IN (1, 9)")
            .await
            .unwrap(),
        2
    );
    assert_eq!(lake.head().await, before + 1);
    assert_eq!(
        rows(lake.overlay("o", lake.head().await).await).await,
        expected(&[(1, "x"), (2, "b"), (3, "c"), (9, "x")])
    );
    assert_eq!(
        rows(lake.overlay("o", before).await).await,
        expected(&[(1, "a"), (2, "b"), (3, "c"), (9, "i")])
    );
}

#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn an_overlay_writes_against_its_base_whatever_its_underlay_did_since() {
    let lake = Lake::start().await;
    lake.write("t", &[1, 2], &["a", "b"]).await;
    lake.create("o").await;
    lake.sql("DELETE FROM lake.main.t WHERE id = 1").await;
    lake.sql("INSERT INTO lake.main.t VALUES (3, 'c')").await;

    assert_eq!(
        lake.change("o", "UPDATE ov SET name = 'z' WHERE id = 1")
            .await
            .unwrap(),
        1
    );
    assert_eq!(
        lake.change("o", "DELETE FROM ov WHERE id = 3")
            .await
            .unwrap(),
        0
    );
    assert_eq!(
        rows(lake.overlay("o", lake.head().await).await).await,
        expected(&[(1, "z"), (2, "b")])
    );
}

#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn a_write_that_read_a_stale_overlay_conflicts() {
    let lake = Lake::start().await;
    lake.write("t", &[1, 2], &["a", "b"]).await;
    lake.create("o").await;
    let stale = lake.writable_overlay("o").await;
    lake.change("o", "DELETE FROM ov WHERE id = 1")
        .await
        .unwrap();
    let before = lake.head().await;

    let ctx = SessionContext::new();
    ctx.register_table("ov", Arc::new(stale)).unwrap();
    let error = match ctx.sql("DELETE FROM ov WHERE id = 2").await {
        Ok(frame) => frame.collect().await.expect_err("a stale write conflicts"),
        Err(error) => error,
    };
    assert!(
        error.to_string().contains("changed after snapshot"),
        "{error}"
    );
    assert_eq!(lake.head().await, before, "a conflict commits nothing");
    assert_eq!(
        rows(lake.overlay("o", before).await).await,
        expected(&[(2, "b")])
    );
}

#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn a_write_conflicts_with_a_change_to_the_overlay_table_alone() {
    let lake = Lake::start().await;
    lake.write("t", &[1, 2], &["a", "b"]).await;
    lake.create("o").await;
    // Only hides underlay rows, so stages nothing in the overlay table.
    let stale = lake.writable_overlay("o").await;
    lake.change("o", "INSERT INTO ov VALUES (3, 'c')")
        .await
        .unwrap();
    let before = lake.head().await;

    let ctx = SessionContext::new();
    ctx.register_table("ov", Arc::new(stale)).unwrap();
    let error = match ctx.sql("DELETE FROM ov WHERE id = 1").await {
        Ok(frame) => frame.collect().await.expect_err("a stale write conflicts"),
        Err(error) => error,
    };
    assert!(
        error.to_string().to_lowercase().contains("conflict"),
        "{error}"
    );
    assert_eq!(lake.head().await, before, "a conflict commits nothing");
}

#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn a_flush_folds_what_an_overlay_holds_inline_into_files() {
    let lake = Lake::start().await;
    lake.write("t", &[1, 2, 3, 4], &["a", "b", "c", "d"]).await;
    // At the base the underlay itself has deleted record 4.
    lake.sql("DELETE FROM lake.main.t WHERE id = 4").await;
    let (_, overlay) = lake.create("o").await;
    lake.change("o", "UPDATE ov SET name = 'B' WHERE id = 2")
        .await
        .unwrap();
    lake.change("o", "INSERT INTO ov VALUES (5, 'e')")
        .await
        .unwrap();
    let before = lake.head().await;
    let expected_rows = expected(&[(1, "a"), (2, "B"), (3, "c"), (5, "e")]);

    assert!(lake.flush().await.is_some());
    let live_inline = format!(
        "SELECT COUNT(*) FROM lake.ducklake_overlay_inlined_delete
         WHERE overlay_table_id = {overlay} AND end_snapshot IS NULL"
    );
    let live_files = format!(
        "SELECT COUNT(*) FROM lake.ducklake_overlay_delete_file
         WHERE overlay_table_id = {overlay} AND end_snapshot IS NULL"
    );
    assert_eq!(lake.count(&live_inline).await, 0);
    assert_eq!(lake.count(&live_files).await, 1);
    assert_eq!(
        rows(lake.overlay("o", lake.head().await).await).await,
        expected_rows
    );
    assert_eq!(rows(lake.overlay("o", before).await).await, expected_rows);
    assert!(lake.flush().await.is_none(), "nothing left to flush");

    // A later hide in the same file is folded with the earlier ones, and the
    // underlay's own delete at the base stays.
    lake.change("o", "DELETE FROM ov WHERE id = 3")
        .await
        .unwrap();
    lake.change("o", "DELETE FROM ov WHERE id = 2")
        .await
        .unwrap();
    assert!(lake.flush().await.is_some());
    assert_eq!(lake.count(&live_inline).await, 0);
    assert_eq!(lake.count(&live_files).await, 1);
    assert_eq!(
        rows(lake.overlay("o", lake.head().await).await).await,
        expected(&[(1, "a"), (5, "e")])
    );
    // Writes after a flush see what it folded.
    assert_eq!(
        lake.change("o", "DELETE FROM ov WHERE id IN (2, 3, 4)")
            .await
            .unwrap(),
        0
    );
    assert_eq!(
        lake.change("o", "UPDATE ov SET name = 'A' WHERE id = 1")
            .await
            .unwrap(),
        1
    );
    assert_eq!(
        rows(lake.overlay("o", lake.head().await).await).await,
        expected(&[(1, "A"), (5, "e")])
    );
}

#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn a_rebase_without_conflicts_brings_in_the_underlays_changes() {
    let lake = Lake::start().await;
    lake.write("t", &[1, 2, 3, 4], &["a", "b", "c", "d"]).await;
    let (_, overlay) = lake.create("o").await;
    lake.change("o", "UPDATE ov SET name = 'B' WHERE id = 2")
        .await
        .unwrap();
    lake.change("o", "DELETE FROM ov WHERE id = 3")
        .await
        .unwrap();
    lake.change("o", "INSERT INTO ov VALUES (5, 'e')")
        .await
        .unwrap();
    lake.sql("UPDATE lake.main.t SET name = 'A' WHERE id = 1")
        .await;
    lake.sql("INSERT INTO lake.main.t VALUES (6, 'f')").await;
    lake.sql("DELETE FROM lake.main.t WHERE id = 4").await;
    let underlay_head = lake.head().await;

    assert_eq!(lake.conflicts("o").await, Vec::<i64>::new());
    let rebased = lake.rebase("o", &[]).await.unwrap();

    assert_eq!(
        rows(lake.overlay("o", rebased).await).await,
        expected(&[(1, "A"), (2, "B"), (5, "e"), (6, "f")])
    );
    assert_eq!(
        lake.count(&format!(
            "SELECT base_snapshot FROM lake.ducklake_overlay
             WHERE overlay_table_id = {overlay} AND end_snapshot IS NULL"
        ))
        .await,
        underlay_head
    );
    // Writes after the rebase act on the new base.
    lake.change("o", "UPDATE ov SET name = 'F' WHERE id = 6")
        .await
        .unwrap();
    assert_eq!(
        rows(lake.overlay("o", lake.head().await).await).await,
        expected(&[(1, "A"), (2, "B"), (5, "e"), (6, "F")])
    );
}

#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn a_rebase_resolves_each_conflict_one_way_or_the_other() {
    let lake = Lake::start().await;
    lake.write("t", &[1, 2, 3], &["a", "b", "c"]).await;
    lake.create("o").await;
    lake.change("o", "UPDATE ov SET name = 'mine' WHERE id = 1")
        .await
        .unwrap();
    lake.change("o", "DELETE FROM ov WHERE id = 2")
        .await
        .unwrap();
    lake.change("o", "INSERT INTO ov VALUES (7, 'mine')")
        .await
        .unwrap();
    lake.sql("UPDATE lake.main.t SET name = 'theirs' WHERE id IN (1, 2)")
        .await;
    lake.sql("INSERT INTO lake.main.t VALUES (7, 'theirs')")
        .await;

    assert_eq!(lake.conflicts("o").await, vec![1, 2, 7]);
    // Keep the overlay's 1 and 7; take the underlay's 2.
    let rebased = lake.rebase("o", &[2]).await.unwrap();
    assert_eq!(
        rows(lake.overlay("o", rebased).await).await,
        expected(&[(1, "mine"), (2, "theirs"), (3, "c"), (7, "mine")])
    );
    assert_eq!(lake.conflicts("o").await, Vec::<i64>::new(), "settled");

    // Taking the underlay's version drops the overlay's row for it.
    lake.sql("UPDATE lake.main.t SET name = 'again' WHERE id = 7")
        .await;
    assert_eq!(lake.conflicts("o").await, vec![7]);
    let rebased = lake.rebase("o", &[7]).await.unwrap();
    assert_eq!(
        rows(lake.overlay("o", rebased).await).await,
        expected(&[(1, "mine"), (2, "theirs"), (3, "c"), (7, "again")])
    );
}

#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn a_rebase_after_a_flush_and_a_stale_plan() {
    let lake = Lake::start().await;
    lake.write("t", &[1, 2, 3], &["a", "b", "c"]).await;
    lake.create("o").await;
    lake.change("o", "DELETE FROM ov WHERE id = 2")
        .await
        .unwrap();
    lake.flush().await.unwrap();
    lake.sql("DELETE FROM lake.main.t WHERE id = 3").await;

    // A plan the overlay moved past is refused.
    let (plan, table, new) = lake.plan("o").await;
    lake.change("o", "INSERT INTO ov VALUES (4, 'd')")
        .await
        .unwrap();
    let before = lake.head().await;
    let error = lake.apply(&table, &plan, &new, &[]).await.unwrap_err();
    assert!(
        matches!(error, datafusion_ducklake::DuckLakeError::Conflict(_)),
        "{error}"
    );
    assert_eq!(lake.head().await, before);

    // Replanned, the flushed delete is carried to the new base.
    let rebased = lake.rebase("o", &[]).await.unwrap();
    assert_eq!(
        rows(lake.overlay("o", rebased).await).await,
        expected(&[(1, "a"), (4, "d")])
    );
}
