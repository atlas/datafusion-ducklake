//! Writes to an overlay (see [`crate::overlay`]).
//!
//! `INSERT` adds rows to the overlay table. `DELETE` and `UPDATE` reach rows in
//! both parts of the overlay: rows of the overlay table are deleted or updated
//! as in any table, and rows of the underlay at the base are hidden, with an
//! `UPDATE` writing their new versions into the overlay table. Each statement
//! commits once, through [`MetadataWriter::commit_overlay`], conditional on
//! nothing in the overlay having changed since the statement read it.
//!
//! Like `DELETE` and `UPDATE` on a table, the work runs at execute time, so
//! planning (`EXPLAIN`) changes nothing.

use std::collections::HashSet;
use std::fmt;
use std::sync::Arc;

use arrow::array::{ArrayRef, RecordBatch, UInt64Array};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use datafusion::catalog::Session;
use datafusion::common::tree_node::TreeNodeRecursion;
use datafusion::error::{DataFusionError, Result as DataFusionResult};
use datafusion::execution::{SendableRecordBatchStream, SessionState, TaskContext};
use datafusion::physical_expr::expressions::lit;
use datafusion::physical_expr::{EquivalenceProperties, Partitioning, PhysicalExpr};
use datafusion::physical_plan::apply_expression_roots;
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties};
use futures::stream;

use crate::delete_exec::{
    CollectedDeletes, DeletionTarget, FileDeletions, LatestDeleteFiles, collect_deletes,
};
use crate::metadata_provider::DuckLakeInlinedDelete;
use crate::metadata_writer::{MetadataWriter, StagedTableData, WriteMode};
use crate::overlay::OverlayCommit;
use crate::table::{DuckLakeTable, FileUpdateOutput, UpdateSourceScan};
use crate::table_writer::{DuckLakeTableWriter, TableWriteOptions, validate_not_null_batches};

/// An overlay statement: what it changes and in which tables.
#[derive(Clone)]
pub(crate) struct OverlayStatement {
    /// The underlay at the base, read through
    /// [`OverlayBaseProvider`](crate::overlay::OverlayBaseProvider).
    pub(crate) underlay: Arc<DuckLakeTable>,
    /// The overlay table at the snapshot the statement reads.
    pub(crate) overlay: Arc<DuckLakeTable>,
    pub(crate) overlay_table_id: i64,
    pub(crate) writer: Arc<dyn MetadataWriter>,
    pub(crate) schema_name: String,
    /// The `WHERE` predicate over the physical columns, or all rows.
    pub(crate) predicate: Option<Arc<dyn PhysicalExpr>>,
    /// `(physical column, new value)` for an `UPDATE`; empty for a `DELETE`.
    pub(crate) assignments: Vec<(usize, Arc<dyn PhysicalExpr>)>,
    pub(crate) kind: StatementKind,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum StatementKind {
    Delete,
    Update,
}

impl OverlayStatement {
    fn predicate(&self) -> Arc<dyn PhysicalExpr> {
        self.predicate.clone().unwrap_or_else(|| lit(true))
    }

    /// Run the statement and return how many rows it changed.
    async fn run(&self, state: &SessionState, context: Arc<TaskContext>) -> DataFusionResult<u64> {
        let object_store = context
            .runtime_env()
            .object_store(self.overlay.object_store_url().as_ref())?;
        let table_writer = DuckLakeTableWriter::new(Arc::clone(&self.writer), object_store)
            .map(|writer| writer.with_options(self.overlay.write_options()))
            .map_err(external)?;
        let overlay_name = self.overlay.table_name().to_string();
        let physical_schema = self.overlay.physical_schema();
        let read_at = self.overlay.snapshot();
        let inline_limit = table_writer.data_inlining_row_limit();
        let mut options = TableWriteOptions::new();
        options.expected_base_snapshot_id = Some(read_at);
        let mut transaction = table_writer.transaction().with_options(&options);
        let latest = LatestDeleteFiles {
            table: &self.overlay,
            state,
        };
        transaction.set_delete_file_positions(&latest);

        let (hide, hide_inlined, changed, data, deletions, inlined_rows) = match self.kind {
            StatementKind::Delete => {
                let (hide, hide_inlined) = self.underlay_matches(state).await?;
                let CollectedDeletes {
                    entries,
                    mut deletions,
                    inlined_rows,
                    count,
                } = collect_deletes(
                    &self.overlay,
                    state,
                    &self.predicate(),
                    &table_writer,
                    &self.schema_name,
                    &overlay_name,
                    &self.overlay.files().map_err(external)?,
                    &self.overlay.inlined_data_with_row_ids().map_err(external)?,
                    Some(inline_limit),
                    read_at,
                )
                .await?;
                deletions.positional.extend(entries);
                let changed = (hide.len() + hide_inlined.len()) as u64 + count;
                (
                    hide,
                    hide_inlined,
                    changed,
                    StagedTableData::None,
                    deletions,
                    inlined_rows,
                )
            },
            StatementKind::Update => {
                // The underlay's matching rows: hidden, and their new versions
                // written to the overlay table as rows of its own.
                let mut versions = Vec::new();
                let mut hide = Vec::new();
                let mut changed = 0;
                for (scan, out) in self.updated_files(&self.underlay, state, &context).await? {
                    changed += out.matched_count as u64;
                    versions.extend(out.updated_batches);
                    hide.extend(out.matched_positions.into_iter().map(|row_id| {
                        DuckLakeInlinedDelete {
                            data_file_id: scan.data_file_id,
                            row_id,
                        }
                    }));
                }
                // And its inlined rows, hidden by row id.
                let mut hide_inlined = Vec::new();
                for rows in &self
                    .underlay
                    .inlined_data_with_row_ids()
                    .map_err(external)?
                {
                    if let Some((batch, superseded)) = self.underlay.apply_update_to_inlined(
                        rows,
                        self.predicate.as_ref(),
                        &self.assignments,
                    )? {
                        changed += batch.num_rows() as u64;
                        versions.push(batch);
                        hide_inlined.extend(superseded.into_iter().map(|row| row.row_id));
                    }
                }
                // The overlay table's own rows, updated as an UPDATE of any
                // table deletes and rewrites them.
                let mut deletions = FileDeletions::default();
                for (scan, out) in self.updated_files(&self.overlay, state, &context).await? {
                    changed += out.matched_count as u64;
                    versions.extend(out.updated_batches);
                    let target = DeletionTarget {
                        data_file_id: scan.data_file_id,
                        data_file_path: &scan.source_path,
                        record_count: scan.record_count,
                        delete_file: scan
                            .delete_file_id
                            .map(|id| (id, &scan.existing_parquet_deletes)),
                    };
                    deletions
                        .record(
                            &self.overlay,
                            &table_writer,
                            &self.schema_name,
                            &overlay_name,
                            &target,
                            out.matched_positions,
                            inline_limit,
                            read_at,
                        )
                        .await?;
                }
                let mut inlined_rows = Vec::new();
                for rows in &self.overlay.inlined_data_with_row_ids().map_err(external)? {
                    if let Some((batch, superseded)) = self.overlay.apply_update_to_inlined(
                        rows,
                        self.predicate.as_ref(),
                        &self.assignments,
                    )? {
                        changed += batch.num_rows() as u64;
                        versions.push(batch);
                        inlined_rows.extend(superseded);
                    }
                }
                validate_not_null_batches(physical_schema.as_ref(), &versions).map_err(external)?;
                // Every new version is a new row of the overlay table: the
                // underlay's row ids are not the overlay table's to keep.
                let versions = versions
                    .iter()
                    .map(|batch| {
                        RecordBatch::try_new(
                            Arc::clone(&physical_schema),
                            batch.columns()[..batch.num_columns() - 1].to_vec(),
                        )
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let rows = versions.iter().map(RecordBatch::num_rows).sum::<usize>();
                let data = if rows == 0 {
                    StagedTableData::None
                } else if rows <= inline_limit
                    && table_writer
                        .should_inline(
                            &self.schema_name,
                            &overlay_name,
                            rows,
                            physical_schema.as_ref(),
                            &versions,
                        )
                        .map_err(external)?
                {
                    StagedTableData::Inlined(versions)
                } else {
                    let mut session = table_writer
                        .begin_write(
                            &self.schema_name,
                            &overlay_name,
                            physical_schema.as_ref(),
                            WriteMode::Append,
                        )
                        .map(|session| session.with_base_snapshot_id(read_at))
                        .map_err(external)?;
                    for batch in &versions {
                        session.write_batch(batch).map_err(external)?;
                    }
                    StagedTableData::Files(session.finish_files().await.map_err(external)?)
                };
                (hide, hide_inlined, changed, data, deletions, inlined_rows)
            },
        };

        if changed == 0 {
            transaction.abort().await.map_err(external)?;
            return Ok(0);
        }
        if !matches!(data, StagedTableData::None)
            || !inlined_rows.is_empty()
            || !deletions.positional.is_empty()
            || deletions.needs_staged_commit()
        {
            transaction
                .stage_row_mutation(
                    &self.schema_name,
                    &overlay_name,
                    self.overlay.table_id(),
                    self.overlay.column_ids(),
                    physical_schema.as_ref(),
                    read_at,
                    data,
                    None,
                    deletions.positional,
                    deletions.rewritten,
                    inlined_rows,
                    deletions.inlined,
                    deletions.dropped,
                )
                .await
                .map_err(external)?;
        }
        if changed == 0 {
            transaction.abort().await.map_err(external)?;
            return Ok(0);
        }
        transaction
            .commit_overlay(&OverlayCommit {
                overlay_table_id: self.overlay_table_id,
                create: None,
                hide,
                hide_inlined,
            })
            .await
            .map_err(external)?;
        Ok(changed)
    }

    /// The underlay positions at the base that the predicate matches and that
    /// are neither deleted in the underlay nor hidden by the overlay, and the
    /// row ids of the underlay's inlined rows it matches that the overlay does
    /// not hide (the base provider leaves those out).
    async fn underlay_matches(
        &self,
        state: &SessionState,
    ) -> DataFusionResult<(Vec<DuckLakeInlinedDelete>, Vec<i64>)> {
        let state: &dyn Session = state;
        let predicate = self.predicate();
        // The base provider gives each file the delete file the overlay reads
        // and adds the overlay's hidden positions to its inlined deletes.
        let hidden = self.underlay.inlined_deletes_by_file()?;
        let mut matches = Vec::new();
        for file in self.underlay.files().map_err(external)? {
            let matched = self
                .underlay
                .resolve_positions(state, &file.file, Arc::clone(&predicate))
                .await?;
            if matched.is_empty() {
                continue;
            }
            let mut deleted = match &file.delete_file {
                Some(delete_file) => {
                    self.underlay
                        .read_delete_file_positions(state, delete_file)
                        .await?
                },
                None => HashSet::new(),
            };
            if let Some(hidden) = hidden.get(&file.data_file_id) {
                deleted.extend(hidden);
            }
            let mut positions = matched.difference(&deleted).copied().collect::<Vec<_>>();
            positions.sort_unstable();
            matches.extend(positions.into_iter().map(|row_id| DuckLakeInlinedDelete {
                data_file_id: file.data_file_id,
                row_id,
            }));
        }
        let inlined = self.underlay.matching_inlined_row_ids(&predicate)?;
        Ok((matches, inlined))
    }

    /// Each of `table`'s data files with rows the statement updates, and
    /// those rows' new versions (`[physical columns..., rowid]`) and positions.
    async fn updated_files(
        &self,
        table: &DuckLakeTable,
        state: &SessionState,
        context: &Arc<TaskContext>,
    ) -> DataFusionResult<Vec<(UpdateSourceScan, FileUpdateOutput)>> {
        let state: &dyn Session = state;
        let inlined = table.inlined_deletes_by_file()?;
        let mut updated = Vec::new();
        for file in table.files().map_err(external)? {
            let scan = table
                .build_update_scan_filtered(
                    state,
                    &file,
                    inlined.get(&file.data_file_id),
                    self.predicate.as_ref(),
                )
                .await?;
            let batches =
                datafusion::physical_plan::collect(Arc::clone(&scan.scan), Arc::clone(context))
                    .await?;
            let out = table.apply_update_to_batches(
                &scan,
                &batches,
                self.predicate.as_ref(),
                &self.assignments,
            )?;
            if out.matched_count > 0 {
                updated.push((scan, out));
            }
        }
        Ok(updated)
    }
}

fn external(error: crate::DuckLakeError) -> DataFusionError {
    DataFusionError::External(Box::new(error))
}

fn count_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![Field::new(
        "count",
        DataType::UInt64,
        false,
    )]))
}

/// Runs an [`OverlayStatement`] at execute time and yields its row count.
pub(crate) struct OverlayWriteExec {
    statement: OverlayStatement,
    /// DataFusion's `TaskContext` cannot plan the reads a statement makes, so
    /// the planning session is kept for execute time, as `DELETE` keeps it.
    session_state: SessionState,
    properties: Arc<PlanProperties>,
}

impl OverlayWriteExec {
    pub(crate) fn new(statement: OverlayStatement, session_state: SessionState) -> Self {
        let properties = Arc::new(PlanProperties::new(
            EquivalenceProperties::new(count_schema()),
            Partitioning::UnknownPartitioning(1),
            datafusion::physical_plan::execution_plan::EmissionType::Final,
            datafusion::physical_plan::execution_plan::Boundedness::Bounded,
        ));
        Self {
            statement,
            session_state,
            properties,
        }
    }
}

impl fmt::Debug for OverlayWriteExec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OverlayWriteExec")
            .field("kind", &self.statement.kind)
            .field("overlay_table_id", &self.statement.overlay_table_id)
            .finish_non_exhaustive()
    }
}

impl DisplayAs for OverlayWriteExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "OverlayWriteExec: {:?} overlay={}",
            self.statement.kind, self.statement.overlay_table_id
        )
    }
}

impl ExecutionPlan for OverlayWriteExec {
    fn apply_expressions(
        &self,
        f: &mut dyn FnMut(&Arc<dyn PhysicalExpr>) -> DataFusionResult<TreeNodeRecursion>,
    ) -> DataFusionResult<TreeNodeRecursion> {
        apply_expression_roots(
            self.statement
                .predicate
                .iter()
                .chain(self.statement.assignments.iter().map(|(_, expr)| expr)),
            f,
        )
    }

    fn name(&self) -> &str {
        "OverlayWriteExec"
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.properties
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        Vec::new()
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        if children.is_empty() {
            Ok(self)
        } else {
            Err(DataFusionError::Internal(
                "OverlayWriteExec has no children".to_string(),
            ))
        }
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> DataFusionResult<SendableRecordBatchStream> {
        if partition != 0 {
            return Err(DataFusionError::Internal(format!(
                "OverlayWriteExec only supports partition 0, got {partition}"
            )));
        }
        let statement = self.statement.clone();
        let state = self.session_state.clone();
        let stream = stream::once(async move {
            let changed = statement.run(&state, context).await?;
            let count: ArrayRef = Arc::new(UInt64Array::from(vec![changed]));
            Ok(RecordBatch::try_new(count_schema(), vec![count])?)
        });
        Ok(Box::pin(RecordBatchStreamAdapter::new(
            count_schema(),
            stream,
        )))
    }
}
