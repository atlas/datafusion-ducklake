//! Overlays: a writable table layered over another table at a pinned snapshot.
//!
//! An overlay reads its *underlay* as it was at the overlay's *base* snapshot,
//! hides the underlay rows the overlay deleted or replaced, and adds the
//! overlay's own rows. The overlay's own rows are an ordinary DuckLake table
//! in the same lake, with its own files, inlined rows and deletes. The only
//! state an overlay adds to the lake is what it hides in the underlay's files:
//!
//! - positions in underlay data files, laid out as DuckLake's inlined
//!   deletions (`data_file_id`, `row_id`);
//! - per underlay data file, at most one overlay delete file. It holds the
//!   underlay's own deletes at the base as well as the overlay's, so it takes
//!   the place of the underlay's delete file for that data file.
//!
//! [`OverlayBaseProvider`] presents the underlay at its base with those deletes
//! applied, through the ordinary [`DuckLakeTable`] scan. [`DuckLakeOverlayTable`]
//! unions that scan with the overlay table's.

use crate::inlined_filter::{InlinedDataScan, InlinedFilter};
use crate::metadata_provider::{
    ColumnTag, ColumnWithTable, DataFileChange, DeleteFileChange, DuckLakeFileData,
    DuckLakeFileMetadata, DuckLakeInlinedData, DuckLakeInlinedDelete, DuckLakeNameMapping,
    DuckLakeStatistics, DuckLakeTableColumn, DuckLakeTableField, DuckLakeTableFile, DuckLakeTag,
    FileWithTable, MetadataProvider, ObjectTag, SchemaMetadata, SnapshotChangeMetadata,
    SnapshotMetadata, TableMetadata, TableWithSchema, TagTarget, ViewMetadata, ViewWithSchema,
};
use crate::table::DuckLakeTable;
use crate::{DuckLakeError, Result};
use arrow::datatypes::SchemaRef;
use async_trait::async_trait;
use datafusion::catalog::{Session, TableProvider};
use datafusion::error::{DataFusionError, Result as DataFusionResult};
use datafusion::logical_expr::{Expr, TableProviderFilterPushDown, TableType};
use datafusion::physical_plan::ExecutionPlan;
use datafusion::physical_plan::union::UnionExec;
use std::collections::HashMap;
use std::sync::Arc;

/// An overlay delete file: the delete file an overlay reads for one underlay
/// data file in place of the underlay's own.
#[derive(Debug, Clone)]
pub struct OverlayDeleteFile {
    /// The overlay delete file's own id, in the overlay's catalog table.
    pub delete_file_id: i64,
    /// Every deleted position in the data file: the underlay's deletes at the
    /// base and the overlay's.
    pub file: DuckLakeFileData,
    pub delete_count: i64,
}

/// What an overlay hides in its underlay, as of one overlay snapshot.
#[derive(Debug, Clone)]
pub struct OverlayBase {
    pub underlay_table_id: i64,
    /// The underlay snapshot the overlay reads.
    pub base_snapshot: i64,
    /// Positions hidden in underlay data files, beyond the overlay delete files.
    pub inlined_deletes: Vec<DuckLakeInlinedDelete>,
    /// Overlay delete files, by the underlay data file they apply to.
    pub delete_files: HashMap<i64, OverlayDeleteFile>,
}

/// A provider that answers for the underlay at its base with the overlay's
/// deletes applied, and passes every other call through unchanged.
///
/// It serves one [`DuckLakeTable`] built at the base snapshot, and only for
/// reads: that table must not be given a writer.
#[derive(Debug)]
pub struct OverlayBaseProvider {
    inner: Arc<dyn MetadataProvider>,
    base: Arc<OverlayBase>,
}

impl OverlayBaseProvider {
    pub fn new(inner: Arc<dyn MetadataProvider>, base: Arc<OverlayBase>) -> Self {
        Self {
            inner,
            base,
        }
    }

    fn applies(&self, table_id: i64, snapshot_id: i64) -> Result<bool> {
        if table_id != self.base.underlay_table_id {
            return Ok(false);
        }
        if snapshot_id != self.base.base_snapshot {
            return Err(DuckLakeError::InvalidSnapshot(format!(
                "an overlay reads its underlay at snapshot {}, not {snapshot_id}",
                self.base.base_snapshot
            )));
        }
        Ok(true)
    }

    fn replace_delete_file(&self, file: &mut DuckLakeTableFile) {
        if let Some(overlay) = self.base.delete_files.get(&file.data_file_id) {
            file.delete_file_id = Some(overlay.delete_file_id);
            file.delete_file = Some(overlay.file.clone());
            file.delete_count = Some(overlay.delete_count);
        }
    }
}

/// The underlay with an overlay's deletes is never the catalog's current table
/// generation, so its catalog row counts, column statistics and delete counts
/// must never answer for it. [`DuckLakeTable`] uses those only when the
/// current snapshot equals its own, which no base snapshot equals.
const NEVER_CURRENT: i64 = i64::MAX;

impl MetadataProvider for OverlayBaseProvider {
    fn get_current_snapshot(&self) -> Result<i64> {
        Ok(NEVER_CURRENT)
    }

    fn get_data_path(&self) -> Result<String> {
        self.inner.get_data_path()
    }

    fn get_metadata_settings(
        &self,
        schema_id: Option<i64>,
        table_id: Option<i64>,
    ) -> Result<HashMap<String, String>> {
        self.inner.get_metadata_settings(schema_id, table_id)
    }

    fn list_snapshots(&self) -> Result<Vec<SnapshotMetadata>> {
        self.inner.list_snapshots()
    }

    fn list_snapshot_changes(&self) -> Result<Vec<SnapshotChangeMetadata>> {
        self.inner.list_snapshot_changes()
    }

    fn find_snapshot_by_commit_extra_info(&self, needle: &str) -> Result<Option<i64>> {
        self.inner.find_snapshot_by_commit_extra_info(needle)
    }

    fn list_schemas(&self, snapshot_id: i64) -> Result<Vec<SchemaMetadata>> {
        self.inner.list_schemas(snapshot_id)
    }

    fn list_tables(&self, schema_id: i64, snapshot_id: i64) -> Result<Vec<TableMetadata>> {
        self.inner.list_tables(schema_id, snapshot_id)
    }

    fn list_views(&self, schema_id: i64, snapshot_id: i64) -> Result<Vec<ViewMetadata>> {
        self.inner.list_views(schema_id, snapshot_id)
    }

    fn get_table_structure(
        &self,
        table_id: i64,
        snapshot_id: i64,
    ) -> Result<Vec<DuckLakeTableColumn>> {
        self.inner.get_table_structure(table_id, snapshot_id)
    }

    fn get_table_fields(&self, table_id: i64, snapshot_id: i64) -> Result<Vec<DuckLakeTableField>> {
        self.inner.get_table_fields(table_id, snapshot_id)
    }

    fn get_name_mapping(&self, mapping_id: i64) -> Result<DuckLakeNameMapping> {
        self.inner.get_name_mapping(mapping_id)
    }

    fn get_table_files_for_select(
        &self,
        table_id: i64,
        snapshot_id: i64,
    ) -> Result<Vec<DuckLakeTableFile>> {
        let mut files = self
            .inner
            .get_table_files_for_select(table_id, snapshot_id)?;
        if self.applies(table_id, snapshot_id)? {
            files
                .iter_mut()
                .for_each(|file| self.replace_delete_file(file));
        }
        Ok(files)
    }

    fn get_partition_spec(
        &self,
        table_id: i64,
        snapshot_id: i64,
    ) -> Result<Option<crate::partition::PartitionSpec>> {
        self.inner.get_partition_spec(table_id, snapshot_id)
    }

    fn get_sort_spec(
        &self,
        table_id: i64,
        snapshot_id: i64,
    ) -> Result<Option<crate::sort::SortSpec>> {
        self.inner.get_sort_spec(table_id, snapshot_id)
    }

    fn get_table_statistics(&self, table_id: i64, snapshot_id: i64) -> Result<DuckLakeStatistics> {
        self.inner.get_table_statistics(table_id, snapshot_id)
    }

    fn get_table_summary_statistics(
        &self,
        table_id: i64,
        snapshot_id: i64,
    ) -> Result<DuckLakeStatistics> {
        self.inner
            .get_table_summary_statistics(table_id, snapshot_id)
    }

    fn get_table_file_metadata_page(
        &self,
        table_id: i64,
        snapshot_id: i64,
        after_data_file_id: Option<i64>,
        limit: usize,
    ) -> Result<Vec<DuckLakeFileMetadata>> {
        let mut page = self.inner.get_table_file_metadata_page(
            table_id,
            snapshot_id,
            after_data_file_id,
            limit,
        )?;
        if self.applies(table_id, snapshot_id)? {
            page.iter_mut()
                .for_each(|metadata| self.replace_delete_file(&mut metadata.file));
        }
        Ok(page)
    }

    fn get_table_file_metadata_page_filtered(
        &self,
        table_id: i64,
        snapshot_id: i64,
        after_data_file_id: Option<i64>,
        limit: usize,
        filter: Option<&crate::stats_filter::StatsFilter>,
    ) -> Result<Vec<DuckLakeFileMetadata>> {
        let mut page = self.inner.get_table_file_metadata_page_filtered(
            table_id,
            snapshot_id,
            after_data_file_id,
            limit,
            filter,
        )?;
        if self.applies(table_id, snapshot_id)? {
            page.iter_mut()
                .for_each(|metadata| self.replace_delete_file(&mut metadata.file));
        }
        Ok(page)
    }

    fn get_inlined_data(
        &self,
        table_id: i64,
        snapshot_id: i64,
        columns: &[DuckLakeTableColumn],
    ) -> Result<Vec<arrow::record_batch::RecordBatch>> {
        self.inner.get_inlined_data(table_id, snapshot_id, columns)
    }

    fn scan_inlined_data(
        &self,
        table_id: i64,
        snapshot_id: i64,
        columns: &[DuckLakeTableColumn],
        filter: Option<&InlinedFilter>,
    ) -> Result<InlinedDataScan> {
        self.inner
            .scan_inlined_data(table_id, snapshot_id, columns, filter)
    }

    fn get_inlined_deletes(
        &self,
        table_id: i64,
        snapshot_id: i64,
    ) -> Result<Vec<DuckLakeInlinedDelete>> {
        let mut deletes = self.inner.get_inlined_deletes(table_id, snapshot_id)?;
        if self.applies(table_id, snapshot_id)? {
            deletes.extend_from_slice(&self.base.inlined_deletes);
        }
        Ok(deletes)
    }

    fn get_inlined_data_with_row_ids(
        &self,
        table_id: i64,
        snapshot_id: i64,
        columns: &[DuckLakeTableColumn],
    ) -> Result<Vec<DuckLakeInlinedData>> {
        self.inner
            .get_inlined_data_with_row_ids(table_id, snapshot_id, columns)
    }

    fn get_table_row_count(&self, table_id: i64, snapshot_id: i64) -> Result<u64> {
        if self.applies(table_id, snapshot_id)? {
            return Err(DuckLakeError::Unsupported(
                "an overlay's underlay has no catalog row count".into(),
            ));
        }
        self.inner.get_table_row_count(table_id, snapshot_id)
    }

    fn get_schema_by_name(&self, name: &str, snapshot_id: i64) -> Result<Option<SchemaMetadata>> {
        self.inner.get_schema_by_name(name, snapshot_id)
    }

    fn get_table_by_name(
        &self,
        schema_id: i64,
        name: &str,
        snapshot_id: i64,
    ) -> Result<Option<TableMetadata>> {
        self.inner.get_table_by_name(schema_id, name, snapshot_id)
    }

    fn get_view_by_name(
        &self,
        schema_id: i64,
        name: &str,
        snapshot_id: i64,
    ) -> Result<Option<ViewMetadata>> {
        self.inner.get_view_by_name(schema_id, name, snapshot_id)
    }

    fn table_exists(&self, schema_id: i64, name: &str, snapshot_id: i64) -> Result<bool> {
        self.inner.table_exists(schema_id, name, snapshot_id)
    }

    fn get_tags(&self, target: TagTarget, snapshot_id: i64) -> Result<Vec<DuckLakeTag>> {
        self.inner.get_tags(target, snapshot_id)
    }

    fn get_view_id_by_name(
        &self,
        schema_id: i64,
        name: &str,
        snapshot_id: i64,
    ) -> Result<Option<i64>> {
        self.inner.get_view_id_by_name(schema_id, name, snapshot_id)
    }

    fn list_all_tables(&self, snapshot_id: i64) -> Result<Vec<TableWithSchema>> {
        self.inner.list_all_tables(snapshot_id)
    }

    fn list_all_views(&self, snapshot_id: i64) -> Result<Vec<ViewWithSchema>> {
        self.inner.list_all_views(snapshot_id)
    }

    fn list_all_columns(&self, snapshot_id: i64) -> Result<Vec<ColumnWithTable>> {
        self.inner.list_all_columns(snapshot_id)
    }

    fn list_all_object_tags(&self, snapshot_id: i64) -> Result<Vec<ObjectTag>> {
        self.inner.list_all_object_tags(snapshot_id)
    }

    fn list_all_column_tags(&self, snapshot_id: i64) -> Result<Vec<ColumnTag>> {
        self.inner.list_all_column_tags(snapshot_id)
    }

    fn list_all_files(&self, snapshot_id: i64) -> Result<Vec<FileWithTable>> {
        self.inner.list_all_files(snapshot_id)
    }

    fn get_data_files_added_between_snapshots(
        &self,
        table_id: i64,
        start_snapshot: i64,
        end_snapshot: i64,
    ) -> Result<Vec<DataFileChange>> {
        self.inner
            .get_data_files_added_between_snapshots(table_id, start_snapshot, end_snapshot)
    }

    fn get_delete_files_added_between_snapshots(
        &self,
        table_id: i64,
        start_snapshot: i64,
        end_snapshot: i64,
    ) -> Result<Vec<DeleteFileChange>> {
        self.inner
            .get_delete_files_added_between_snapshots(table_id, start_snapshot, end_snapshot)
    }
}

/// An overlay as a table: its underlay at the base with the overlay's deletes,
/// and the overlay's own rows.
#[derive(Debug)]
pub struct DuckLakeOverlayTable {
    underlay: Arc<DuckLakeTable>,
    overlay: Arc<DuckLakeTable>,
}

impl DuckLakeOverlayTable {
    /// `underlay` must be built at the base snapshot over an
    /// [`OverlayBaseProvider`], and `overlay` is the overlay's own table at the
    /// snapshot being read. Both are read-only here and must have the same
    /// columns, in the same order, with the same types and nullability.
    pub fn try_new(underlay: Arc<DuckLakeTable>, overlay: Arc<DuckLakeTable>) -> Result<Self> {
        let (left, right) = (underlay.schema(), overlay.schema());
        let same = left.fields().len() == right.fields().len()
            && left.fields().iter().zip(right.fields()).all(|(a, b)| {
                a.name() == b.name()
                    && a.data_type() == b.data_type()
                    && a.is_nullable() == b.is_nullable()
            });
        if !same {
            return Err(DuckLakeError::InvalidConfig(
                "an overlay's table must have its underlay's columns".into(),
            ));
        }
        Ok(Self {
            underlay,
            overlay,
        })
    }
}

#[async_trait]
impl TableProvider for DuckLakeOverlayTable {
    fn schema(&self) -> SchemaRef {
        self.underlay.schema()
    }

    fn table_type(&self) -> TableType {
        TableType::Base
    }

    fn supports_filters_pushdown(
        &self,
        filters: &[&Expr],
    ) -> DataFusionResult<Vec<TableProviderFilterPushDown>> {
        // Both scans apply deletes after reading, so DataFusion must reapply
        // every filter, as for a single table (see `DuckLakeTable`).
        Ok(filters
            .iter()
            .map(|_| TableProviderFilterPushDown::Inexact)
            .collect())
    }

    async fn scan(
        &self,
        state: &dyn Session,
        projection: Option<&Vec<usize>>,
        filters: &[Expr],
        limit: Option<usize>,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        // A limit is safe on each side: a union of two inputs, each cut at
        // `limit`, still holds at least `limit` rows when the overlay does.
        let underlay = self
            .underlay
            .scan(state, projection, filters, limit)
            .await?;
        let overlay = self.overlay.scan(state, projection, filters, limit).await?;
        if underlay.schema() != overlay.schema() {
            return Err(DataFusionError::Internal(
                "an overlay's scans disagree on their schema".into(),
            ));
        }
        UnionExec::try_new(vec![underlay, overlay])
    }
}
