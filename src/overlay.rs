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
//! - the underlay's inlined rows it hides, by row id, as DuckLake ends an
//!   inlined row by its row id;
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
    /// Row ids of the underlay's inlined rows the overlay hides.
    pub inlined_row_deletes: std::collections::HashSet<i64>,
    /// Overlay delete files, by the underlay data file they apply to.
    pub delete_files: HashMap<i64, OverlayDeleteFile>,
}

/// What one overlay write changes in what the overlay hides, committed with the
/// overlay table's own changes in one snapshot by
/// [`MetadataWriter::commit_overlay`](crate::metadata_writer::MetadataWriter::commit_overlay).
#[derive(Debug, Clone, Default)]
pub struct OverlayCommit {
    /// The overlay's table: a staged write's table id, or a committed table's.
    pub overlay_table_id: i64,
    /// Make the table an overlay of `(underlay_table_id, base_snapshot)` in
    /// this snapshot. The table must not be an overlay already.
    pub create: Option<(i64, i64)>,
    /// Merge the overlay into its underlay and close it (see [`merge_overlay`]).
    pub merge: bool,
    /// Move the base to this underlay snapshot, no earlier than the current
    /// one. Everything hidden at the old base stops being hidden; `hide` then
    /// names positions at the new base.
    pub rebase: Option<i64>,
    /// Underlay positions the overlay starts to hide. Each must be in an
    /// underlay data file live at the base, and not hidden already.
    pub hide: Vec<DuckLakeInlinedDelete>,
    /// Row ids of underlay inlined rows the overlay starts to hide. Each must
    /// be an inlined row of the underlay visible at the base, and not hidden
    /// already.
    pub hide_inlined: Vec<i64>,
    /// Overlay delete files that fold every position the overlay hides inline
    /// in their data files; those inline positions end.
    pub fold: Vec<OverlayFold>,
}

/// A new overlay delete file for one underlay data file, from a flush.
#[derive(Debug, Clone)]
pub struct OverlayFold {
    /// The underlay data file it applies to.
    pub data_file_id: i64,
    /// The live overlay delete file it supersedes, or none. The commit
    /// conflicts if the live one is another.
    pub expected_prev_delete_file: Option<i64>,
    /// Every deleted position of the data file: the underlay's deletes at the
    /// base, the previous overlay delete file's and the folded ones.
    pub file: DuckLakeFileData,
    pub delete_count: i64,
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

    /// Whether a read of `table_id`'s inlined rows at `snapshot_id` must leave
    /// out rows the overlay hides.
    fn hides_inlined_rows(&self, table_id: i64, snapshot_id: i64) -> Result<bool> {
        Ok(self.applies(table_id, snapshot_id)? && !self.base.inlined_row_deletes.is_empty())
    }

    /// Which of `row_ids` the overlay does not hide.
    fn unhidden(&self, row_ids: &[i64]) -> arrow::array::BooleanArray {
        row_ids
            .iter()
            .map(|row_id| Some(!self.base.inlined_row_deletes.contains(row_id)))
            .collect()
    }

    /// The underlay's inlined rows visible at the base that the overlay does
    /// not hide, read as a scan reads them.
    fn visible_inlined_rows(
        &self,
        table_id: i64,
        snapshot_id: i64,
        columns: &[DuckLakeTableColumn],
        filter: Option<&InlinedFilter>,
    ) -> Result<Vec<arrow::record_batch::RecordBatch>> {
        self.read_inlined_rows(
            table_id,
            columns,
            crate::metadata_provider::InlinedRowSelection::VisibleAt(snapshot_id),
            filter,
        )
        .map(|rows| rows.into_iter().map(|rows| rows.batch).collect())
    }

    fn replace_delete_file(&self, file: &mut DuckLakeTableFile) {
        if let Some(overlay) = self.base.delete_files.get(&file.data_file_id) {
            file.delete_file_id = Some(overlay.delete_file_id);
            file.delete_file = Some(overlay.file.clone());
            file.delete_count = Some(overlay.delete_count);
        }
    }
}

/// The values of `values` that `keep` keeps.
fn kept<T: Clone>(values: &[T], keep: &arrow::array::BooleanArray) -> Vec<T> {
    values
        .iter()
        .zip(keep.iter())
        .filter(|(_, keep)| keep.unwrap_or(false))
        .map(|(value, _)| value.clone())
        .collect()
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

    fn find_snapshot_committed_with(
        &self,
        extra_info: &str,
        after: Option<i64>,
    ) -> Result<Option<i64>> {
        self.inner.find_snapshot_committed_with(extra_info, after)
    }

    fn get_table_head(&self, table_id: i64, snapshot_id: i64) -> Result<Option<i64>> {
        self.inner.get_table_head(table_id, snapshot_id)
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
        if !self.hides_inlined_rows(table_id, snapshot_id)? {
            return self.inner.get_inlined_data(table_id, snapshot_id, columns);
        }
        self.visible_inlined_rows(table_id, snapshot_id, columns, None)
    }

    fn scan_inlined_data(
        &self,
        table_id: i64,
        snapshot_id: i64,
        columns: &[DuckLakeTableColumn],
        filter: Option<&InlinedFilter>,
    ) -> Result<InlinedDataScan> {
        if !self.hides_inlined_rows(table_id, snapshot_id)? {
            return self
                .inner
                .scan_inlined_data(table_id, snapshot_id, columns, filter);
        }
        Ok(InlinedDataScan::from_batches(self.visible_inlined_rows(
            table_id,
            snapshot_id,
            columns,
            filter,
        )?))
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
        let data = self
            .inner
            .get_inlined_data_with_row_ids(table_id, snapshot_id, columns)?;
        if !self.hides_inlined_rows(table_id, snapshot_id)? {
            return Ok(data);
        }
        data.into_iter()
            .map(|data| {
                let keep = self.unhidden(&data.row_ids);
                Ok(DuckLakeInlinedData {
                    row_ids: kept(&data.row_ids, &keep),
                    begin_snapshots: kept(&data.begin_snapshots, &keep),
                    batch: arrow::compute::filter_record_batch(&data.batch, &keep)?,
                    table_name: data.table_name,
                })
            })
            .collect()
    }

    fn read_inlined_rows(
        &self,
        table_id: i64,
        columns: &[DuckLakeTableColumn],
        selection: crate::metadata_provider::InlinedRowSelection,
        filter: Option<&InlinedFilter>,
    ) -> Result<Vec<crate::metadata_provider::DuckLakeInlinedRows>> {
        let hides = match selection {
            crate::metadata_provider::InlinedRowSelection::VisibleAt(snapshot_id) => {
                self.hides_inlined_rows(table_id, snapshot_id)?
            },
            _ => false,
        };
        let rows = self
            .inner
            .read_inlined_rows(table_id, columns, selection, filter)?;
        if !hides {
            return Ok(rows);
        }
        rows.into_iter()
            .map(|rows| {
                let keep = self.unhidden(&rows.row_ids);
                Ok(crate::metadata_provider::DuckLakeInlinedRows {
                    row_ids: kept(&rows.row_ids, &keep),
                    snapshots: kept(&rows.snapshots, &keep),
                    end_snapshots: kept(&rows.end_snapshots, &keep),
                    batch: arrow::compute::filter_record_batch(&rows.batch, &keep)?,
                    table_name: rows.table_name,
                })
            })
            .collect()
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
    base: Arc<OverlayBase>,
    underlay: Arc<DuckLakeTable>,
    overlay: Arc<DuckLakeTable>,
}

impl DuckLakeOverlayTable {
    /// `base` is the overlay as of the overlay table's snapshot, `underlay`
    /// must be built at the base snapshot over an [`OverlayBaseProvider`] of
    /// `base`, and `overlay` is the overlay's own table at that snapshot. Both are read-only here and must have the same
    /// columns, in the same order, with the same types and nullability.
    pub fn try_new(
        base: Arc<OverlayBase>,
        underlay: Arc<DuckLakeTable>,
        overlay: Arc<DuckLakeTable>,
    ) -> Result<Self> {
        if underlay.table_id() != base.underlay_table_id
            || underlay.snapshot() != base.base_snapshot
        {
            return Err(DuckLakeError::InvalidConfig(format!(
                "an overlay's underlay is table {} at snapshot {}",
                base.underlay_table_id, base.base_snapshot
            )));
        }
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
            base,
            underlay,
            overlay,
        })
    }

    /// What the overlay hides, as of the overlay table's snapshot.
    pub fn base(&self) -> &Arc<OverlayBase> {
        &self.base
    }

    /// The underlay at the base, as the overlay reads it.
    pub(crate) fn underlay(&self) -> &DuckLakeTable {
        &self.underlay
    }

    /// The overlay's own table.
    pub(crate) fn overlay(&self) -> &DuckLakeTable {
        &self.overlay
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

    /// Adds the rows to the overlay table.
    #[cfg(feature = "write")]
    async fn insert_into(
        &self,
        state: &dyn Session,
        input: Arc<dyn ExecutionPlan>,
        insert_op: datafusion::logical_expr::dml::InsertOp,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        if insert_op != datafusion::logical_expr::dml::InsertOp::Append {
            return Err(DataFusionError::NotImplemented(
                "an overlay only takes appending inserts".to_string(),
            ));
        }
        self.overlay.insert_into(state, input, insert_op).await
    }

    /// Hides matching underlay rows and deletes matching overlay rows.
    #[cfg(feature = "write")]
    async fn delete_from(
        &self,
        state: &dyn Session,
        filters: Vec<Expr>,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        self.statement(
            state,
            filters,
            Vec::new(),
            crate::overlay_write::StatementKind::Delete,
        )
    }

    /// Hides matching underlay rows and writes their new versions to the
    /// overlay table, and updates matching overlay rows.
    #[cfg(feature = "write")]
    async fn update(
        &self,
        state: &dyn Session,
        assignments: Vec<(String, Expr)>,
        filters: Vec<Expr>,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        self.statement(
            state,
            filters,
            assignments,
            crate::overlay_write::StatementKind::Update,
        )
    }
}

#[cfg(feature = "write")]
impl DuckLakeOverlayTable {
    /// Plans a `DELETE` or `UPDATE`; [`crate::overlay_write`] runs it.
    fn statement(
        &self,
        state: &dyn Session,
        filters: Vec<Expr>,
        assignments: Vec<(String, Expr)>,
        kind: crate::overlay_write::StatementKind,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        use datafusion::common::DFSchema;
        use datafusion::logical_expr::utils::conjunction;

        let writer = self.overlay.writer().cloned().ok_or_else(|| {
            DataFusionError::Plan(
                "the overlay table is read-only; open it with a writer".to_string(),
            )
        })?;
        let schema_name = self
            .overlay
            .schema_name()
            .map(str::to_string)
            .ok_or_else(|| {
                DataFusionError::Internal("schema name not set for a writable table".to_string())
            })?;
        let physical_schema = self.overlay.physical_schema();
        let df_schema = DFSchema::try_from(physical_schema.as_ref().clone())?;
        let predicate = conjunction(filters)
            .map(|expr| state.create_physical_expr(expr, &df_schema))
            .transpose()?;
        let assignments = assignments
            .into_iter()
            .map(|(column, expr)| {
                let index = physical_schema.index_of(&column).map_err(|_| {
                    DataFusionError::Plan(format!(
                        "UPDATE assignment targets unknown column '{column}'"
                    ))
                })?;
                Ok((index, state.create_physical_expr(expr, &df_schema)?))
            })
            .collect::<DataFusionResult<Vec<_>>>()?;
        let session_state = state
            .as_any()
            .downcast_ref::<datafusion::execution::SessionState>()
            .ok_or_else(|| {
                DataFusionError::Internal("an overlay write needs a SessionState".to_string())
            })?
            .clone();
        Ok(Arc::new(crate::overlay_write::OverlayWriteExec::new(
            crate::overlay_write::OverlayStatement {
                underlay: Arc::clone(&self.underlay),
                overlay: Arc::clone(&self.overlay),
                overlay_table_id: self.overlay.table_id(),
                writer,
                schema_name,
                predicate,
                assignments,
                kind,
            },
            session_state,
        )))
    }
}

/// Creates `schema_name.table_name` with `underlay`'s columns and makes it an
/// overlay of `underlay` at the snapshot `underlay` was read at, in one
/// snapshot. Returns that snapshot and the overlay table's id.
///
/// The overlay table's columns take the underlay's column ids, which DuckLake
/// scopes to a table, so the overlay table's files are files of the underlay
/// too and a merge registers them as they are.
#[cfg(feature = "write")]
pub async fn create_overlay(
    table_writer: &crate::table_writer::DuckLakeTableWriter,
    schema_name: &str,
    table_name: &str,
    underlay: &DuckLakeTable,
) -> Result<(i64, i64)> {
    let mut transaction = table_writer.transaction();
    let paths = column_paths(underlay.fields());
    let staged = transaction.stage_table_with_column_ids(
        schema_name,
        table_name,
        underlay.data_schema().as_ref(),
        |path| paths.get(path).copied(),
    )?;
    let (snapshot_id, tables) = transaction
        .commit_overlay(&OverlayCommit {
            overlay_table_id: staged,
            create: Some((underlay.table_id(), underlay.snapshot())),
            rebase: None,
            merge: false,
            hide: Vec::new(),
            hide_inlined: Vec::new(),
            fold: Vec::new(),
        })
        .await?;
    let table = tables.first().ok_or_else(|| {
        DuckLakeError::Internal("an overlay's creation committed no table".to_string())
    })?;
    Ok((snapshot_id, table.table_id))
}

/// Flushes an overlay: the overlay table is flushed as any table is, its
/// inlined rows written to Parquet and its inlined deletions of file rows
/// merged into delete files; then every underlay data file the overlay hides
/// rows of inline gets an overlay delete file holding all its deleted
/// positions, superseding the previous one. Each step takes a snapshot of its
/// own, and neither changes what the overlay reads. Returns the last snapshot,
/// or `None` when there was nothing to flush.
///
/// `table` must be writable and read at the overlay's head; each commit
/// conflicts if the overlay changed since. `underlay_schema` names the
/// underlay's schema, under which the new delete files are written beside the
/// underlay's own. Files written for a commit that fails are left for orphan
/// cleanup, as an `UPDATE`'s are.
#[cfg(feature = "write")]
pub async fn flush_overlay(
    table: &DuckLakeOverlayTable,
    table_writer: &crate::table_writer::DuckLakeTableWriter,
    underlay_schema: &str,
    state: &dyn Session,
) -> Result<Option<i64>> {
    use std::collections::BTreeSet;

    let overlay_schema = table.overlay.schema_name().ok_or_else(|| {
        DuckLakeError::InvalidConfig("the overlay table is read-only".to_string())
    })?;
    let hidden_files = table
        .base
        .inlined_deletes
        .iter()
        .map(|delete| delete.data_file_id)
        .collect::<BTreeSet<_>>();
    let flushed = table_writer
        .flush_table_inlined_data(
            Arc::clone(table.overlay.provider()),
            overlay_schema,
            table.overlay.table_name(),
            table.overlay.snapshot(),
        )
        .await?
        .map(|flushed| flushed.snapshot_id);
    if hidden_files.is_empty() {
        return Ok(flushed);
    }

    let options = crate::table_writer::TableWriteOptions::new()
        .with_expected_base_snapshot_id(flushed.unwrap_or(table.overlay.snapshot()));
    let transaction = table_writer.transaction().with_options(&options);

    let deleted_inline = table
        .underlay
        .inlined_deletes_by_file()
        .map_err(|error| DuckLakeError::Internal(error.to_string()))?;
    let mut fold = Vec::with_capacity(hidden_files.len());
    for file in table.underlay.files()? {
        if !hidden_files.contains(&file.data_file_id) {
            continue;
        }
        let mut positions = match &file.delete_file {
            Some(delete_file) => table
                .underlay
                .read_delete_file_positions(state, delete_file)
                .await
                .map_err(|error| DuckLakeError::Internal(error.to_string()))?,
            None => Default::default(),
        };
        if let Some(inline) = deleted_inline.get(&file.data_file_id) {
            positions.extend(inline);
        }
        let mut positions = positions.into_iter().collect::<Vec<_>>();
        positions.sort_unstable();
        let written = table_writer
            .write_delete_file(
                underlay_schema,
                table.underlay.table_name(),
                &file.file.path,
                &positions,
            )
            .await?;
        let mut data = DuckLakeFileData::new(
            written.path,
            written.path_is_relative,
            written.file_size_bytes,
        );
        data.footer_size = written.footer_size;
        fold.push(OverlayFold {
            data_file_id: file.data_file_id,
            expected_prev_delete_file: table
                .base
                .delete_files
                .get(&file.data_file_id)
                .map(|previous| previous.delete_file_id),
            file: data,
            delete_count: written.delete_count,
        });
    }

    let (snapshot_id, _) = transaction
        .commit_overlay(&OverlayCommit {
            overlay_table_id: table.overlay.table_id(),
            create: None,
            rebase: None,
            merge: false,
            hide: Vec::new(),
            hide_inlined: Vec::new(),
            fold,
        })
        .await?;
    Ok(Some(snapshot_id))
}

#[cfg(feature = "write")]
pub use crate::overlay_rebase::{OverlayRebase, plan_rebase, rebase_overlay};

/// Each column's path, its name and its parents' from the top-level column
/// down, to its id.
#[cfg(feature = "write")]
fn column_paths(
    fields: &[crate::metadata_provider::DuckLakeTableField],
) -> HashMap<Vec<String>, i64> {
    let by_id = fields
        .iter()
        .map(|field| (field.column_id, field))
        .collect::<HashMap<_, _>>();
    fields
        .iter()
        .map(|field| {
            let mut path = vec![field.column_name.clone()];
            let mut parent = field.parent_column;
            while let Some(id) = parent {
                let Some(node) = by_id.get(&id) else {
                    break;
                };
                path.push(node.column_name.clone());
                parent = node.parent_column;
            }
            path.reverse();
            (path, field.column_id)
        })
        .collect()
}

/// Merges `table` into its underlay in one snapshot and closes the overlay.
///
/// The underlay's delete files for the data files the overlay hides rows of
/// become the overlay's delete files, which already hold the underlay's
/// deletes at the base, and the overlay table's live files, each with its
/// delete file, become files of the underlay. No data is copied.
///
/// The overlay must be flushed ([`flush_overlay`]), its base must be the
/// underlay's head (rebase first, [`rebase_overlay`]), and `table` must be
/// read at the overlay's head: the commit conflicts if the underlay changed
/// since the base or the overlay since `table` was read. The overlay stays
/// readable at earlier snapshots. Returns the snapshot.
#[cfg(feature = "write")]
pub async fn merge_overlay(
    table: &DuckLakeOverlayTable,
    table_writer: &crate::table_writer::DuckLakeTableWriter,
) -> Result<i64> {
    let inlined = table.overlay.inlined_data_with_row_ids()?;
    let deleted_inline = table
        .overlay
        .inlined_deletes_by_file()
        .map_err(|error| DuckLakeError::Internal(error.to_string()))?;
    if !table.base.inlined_deletes.is_empty()
        || inlined.iter().any(|data| data.batch.num_rows() > 0)
        || deleted_inline
            .values()
            .any(|positions| !positions.is_empty())
    {
        return Err(DuckLakeError::InvalidConfig(
            "flush an overlay before merging it".to_string(),
        ));
    }
    let options = crate::table_writer::TableWriteOptions::new()
        .with_expected_base_snapshot_id(table.overlay.snapshot());
    let (snapshot_id, _) = table_writer
        .transaction()
        .with_options(&options)
        .commit_overlay(&OverlayCommit {
            overlay_table_id: table.overlay.table_id(),
            create: None,
            rebase: None,
            merge: true,
            hide: Vec::new(),
            hide_inlined: Vec::new(),
            fold: Vec::new(),
        })
        .await?;
    Ok(snapshot_id)
}
