//! Rebasing an overlay onto a newer snapshot of its underlay (see
//! [`crate::overlay`]).
//!
//! An overlay touches a set of records by key: those whose underlay rows it
//! hides at its base, and those its own table holds. The underlay changed a
//! record between two snapshots when the record's row differs between them,
//! by value, so a compaction that moved rows changes nothing. A record both
//! touched is a conflict, which the caller resolves by keeping the overlay's
//! version or taking the underlay's.
//!
//! A rebase then commits one rule at the new base: the overlay hides exactly
//! the underlay rows whose key it touches, less the keys resolved to the
//! underlay, whose rows in the overlay table it deletes instead. That covers
//! rows left where they were, rows a compaction moved, records both sides
//! updated and records both sides inserted. Every hidden position starts
//! inline at the new base; a later flush folds them.
//!
//! Both steps read whole tables into memory (row values by key), so they suit
//! tables that fit; reading by key per file is later work.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use arrow::array::RecordBatch;
use arrow::row::{RowConverter, SortField};
use datafusion::catalog::{Session, TableProvider};
use datafusion::common::{DFSchema, ScalarValue};
use datafusion::error::DataFusionError;
use datafusion::logical_expr::{col, lit};
use datafusion::physical_expr::PhysicalExpr;

use crate::delete_exec::{CollectedDeletes, LatestDeleteFiles, collect_deletes};
use crate::metadata_provider::DuckLakeInlinedDelete;
use crate::metadata_writer::StagedTableData;
use crate::overlay::{DuckLakeOverlayTable, OverlayCommit};
use crate::table::DuckLakeTable;
use crate::table_writer::{DuckLakeTableWriter, TableWriteOptions};
use crate::{DuckLakeError, Result};

/// A planned rebase: the snapshot it moves to and the records both sides
/// changed. Apply it with [`rebase_overlay`].
#[derive(Debug, Clone)]
pub struct OverlayRebase {
    /// The underlay snapshot the rebase moves the base to.
    pub new_base: i64,
    /// Keys of records both the overlay and the underlay changed since the
    /// base, in no particular order.
    pub conflicts: Vec<ScalarValue>,
    key_column: String,
    touched: HashSet<ScalarValue>,
    overlay_snapshot: i64,
}

/// Plans rebasing `table` from its base onto `new_base`.
///
/// `old_base` and `new_base` are the underlay read plainly (not through the
/// overlay) at the overlay's base and at the target snapshot. `key_column`
/// names the column that identifies a record in both the underlay and the
/// overlay table; its values must be unique in each.
pub async fn plan_rebase(
    table: &DuckLakeOverlayTable,
    old_base: &DuckLakeTable,
    new_base: &DuckLakeTable,
    key_column: &str,
    state: &dyn Session,
) -> Result<OverlayRebase> {
    let base = table.base();
    let underlay_id = base.underlay_table_id;
    if old_base.table_id() != underlay_id
        || old_base.snapshot() != base.base_snapshot
        || new_base.table_id() != underlay_id
    {
        return Err(DuckLakeError::InvalidConfig(format!(
            "a rebase reads the underlay {underlay_id} at its base {} and at the target",
            base.base_snapshot
        )));
    }
    if new_base.snapshot() < base.base_snapshot {
        return Err(DuckLakeError::InvalidConfig(format!(
            "an overlay cannot rebase back from snapshot {} to {}",
            base.base_snapshot,
            new_base.snapshot()
        )));
    }

    let before = rows_by_key(old_base, key_column, state).await?;
    let visible = keys(table.underlay(), key_column, state).await?;
    let own = keys(table.overlay(), key_column, state).await?;
    let mut touched = before
        .keys()
        .filter(|key| !visible.contains(*key))
        .cloned()
        .collect::<HashSet<_>>();
    touched.extend(own);

    let after = rows_by_key(new_base, key_column, state).await?;
    let conflicts = touched
        .iter()
        .filter(|key| before.get(*key) != after.get(*key))
        .cloned()
        .collect();
    Ok(OverlayRebase {
        new_base: new_base.snapshot(),
        conflicts,
        key_column: key_column.to_string(),
        touched,
        overlay_snapshot: table.overlay().snapshot(),
    })
}

/// Applies `plan` in one snapshot: the overlay's base becomes
/// [`OverlayRebase::new_base`], what it hid at the old base stops being hidden,
/// and it hides the underlay rows at the new base whose key it touches, except
/// `take_underlay`'s, whose rows in the overlay table it deletes instead.
///
/// `table` must be the writable overlay `plan` was made from and `new_base`
/// the underlay at the plan's target. The commit conflicts if the overlay
/// changed since the plan read it. Returns the snapshot.
pub async fn rebase_overlay(
    table: &DuckLakeOverlayTable,
    plan: &OverlayRebase,
    new_base: &DuckLakeTable,
    take_underlay: &HashSet<ScalarValue>,
    table_writer: &DuckLakeTableWriter,
    state: &dyn Session,
) -> Result<i64> {
    if new_base.snapshot() != plan.new_base
        || new_base.table_id() != table.base().underlay_table_id
        || table.overlay().snapshot() != plan.overlay_snapshot
    {
        return Err(DuckLakeError::InvalidConfig(
            "a rebase applies to the overlay and target its plan read".to_string(),
        ));
    }
    let overlay = table.overlay();
    let overlay_schema = overlay.schema_name().ok_or_else(|| {
        DuckLakeError::InvalidConfig("the overlay table is read-only".to_string())
    })?;

    let hidden = plan
        .touched
        .iter()
        .filter(|key| !take_underlay.contains(*key))
        .cloned()
        .collect::<Vec<_>>();
    let mut hide = Vec::new();
    let mut hide_inlined = Vec::new();
    if let Some(predicate) = key_in(new_base, &plan.key_column, &hidden, state)? {
        for file in new_base
            .matching_file_rows(state, &predicate)
            .await
            .map_err(datafusion_error)?
        {
            hide.extend(file.positions.iter().map(|row_id| DuckLakeInlinedDelete {
                data_file_id: file.data_file_id,
                row_id: *row_id,
            }));
        }
        hide_inlined = new_base
            .matching_inlined_row_ids(&predicate)
            .map_err(datafusion_error)?;
    }

    let options = TableWriteOptions::new().with_expected_base_snapshot_id(overlay.snapshot());
    let mut transaction = table_writer.transaction().with_options(&options);
    let latest = LatestDeleteFiles {
        table: overlay,
        state,
    };
    transaction.set_delete_file_positions(&latest);
    let dropped = take_underlay.iter().cloned().collect::<Vec<_>>();
    if let Some(predicate) = key_in(overlay, &plan.key_column, &dropped, state)? {
        let CollectedDeletes {
            entries,
            mut deletions,
            inlined_rows,
            ..
        } = collect_deletes(
            overlay,
            state,
            &predicate,
            table_writer,
            overlay_schema,
            overlay.table_name(),
            &overlay.files()?,
            &overlay.inlined_data_with_row_ids()?,
            Some(table_writer.data_inlining_row_limit()),
            overlay.snapshot(),
        )
        .await
        .map_err(datafusion_error)?;
        deletions.positional.extend(entries);
        transaction
            .stage_row_mutation(
                overlay_schema,
                overlay.table_name(),
                overlay.table_id(),
                overlay.column_ids(),
                overlay.physical_schema().as_ref(),
                overlay.snapshot(),
                StagedTableData::None,
                None,
                deletions.positional,
                deletions.rewritten,
                inlined_rows,
                deletions.inlined,
                deletions.dropped,
            )
            .await?;
    }
    let (snapshot_id, _) = transaction
        .commit_overlay(&OverlayCommit {
            overlay_table_id: overlay.table_id(),
            create: None,
            rebase: Some(plan.new_base),
            hide,
            hide_inlined,
            fold: Vec::new(),
        })
        .await?;
    Ok(snapshot_id)
}

/// `key_column IN (keys)` over `table`'s data columns, or `None` for no keys.
fn key_in(
    table: &DuckLakeTable,
    key_column: &str,
    keys: &[ScalarValue],
    state: &dyn Session,
) -> Result<Option<Arc<dyn PhysicalExpr>>> {
    if keys.is_empty() {
        return Ok(None);
    }
    let schema = table.data_schema();
    let df_schema = DFSchema::try_from(schema.as_ref().clone()).map_err(datafusion_error)?;
    let expr = col(key_column).in_list(keys.iter().cloned().map(lit).collect(), false);
    Ok(Some(
        state
            .create_physical_expr(expr, &df_schema)
            .map_err(datafusion_error)?,
    ))
}

/// Every row of `table`, read as a scan reads it.
async fn read_all(table: &DuckLakeTable, state: &dyn Session) -> Result<Vec<RecordBatch>> {
    let plan = table
        .scan(state, None, &[], None)
        .await
        .map_err(datafusion_error)?;
    datafusion::physical_plan::collect(plan, state.task_ctx())
        .await
        .map_err(datafusion_error)
}

/// The keys of every row `table` reads.
async fn keys(
    table: &DuckLakeTable,
    key_column: &str,
    state: &dyn Session,
) -> Result<HashSet<ScalarValue>> {
    let index = key_index(table, key_column)?;
    let mut keys = HashSet::new();
    for batch in read_all(table, state).await? {
        for row in 0..batch.num_rows() {
            keys.insert(
                ScalarValue::try_from_array(batch.column(index), row).map_err(datafusion_error)?,
            );
        }
    }
    Ok(keys)
}

/// Each row of `table` by key, as its encoded values, so equal rows compare
/// equal whichever file holds them.
async fn rows_by_key(
    table: &DuckLakeTable,
    key_column: &str,
    state: &dyn Session,
) -> Result<HashMap<ScalarValue, Vec<u8>>> {
    let index = key_index(table, key_column)?;
    let schema = table.data_schema();
    let converter = RowConverter::new(
        schema
            .fields()
            .iter()
            .map(|field| SortField::new(field.data_type().clone()))
            .collect(),
    )
    .map_err(|error| DuckLakeError::Internal(error.to_string()))?;
    let mut rows = HashMap::new();
    for batch in read_all(table, state).await? {
        let encoded = converter
            .convert_columns(batch.columns())
            .map_err(|error| DuckLakeError::Internal(error.to_string()))?;
        for row in 0..batch.num_rows() {
            let key =
                ScalarValue::try_from_array(batch.column(index), row).map_err(datafusion_error)?;
            if rows
                .insert(key, encoded.row(row).as_ref().to_vec())
                .is_some()
            {
                return Err(DuckLakeError::InvalidConfig(format!(
                    "key column '{key_column}' is not unique"
                )));
            }
        }
    }
    Ok(rows)
}

fn key_index(table: &DuckLakeTable, key_column: &str) -> Result<usize> {
    table
        .data_schema()
        .index_of(key_column)
        .map_err(|_| DuckLakeError::InvalidConfig(format!("no key column '{key_column}'")))
}

/// A DataFusion error as ours, unwrapping one of ours that it carries (a
/// write's conflict, say) so its kind survives.
fn datafusion_error(error: DataFusionError) -> DuckLakeError {
    match error {
        DataFusionError::External(inner) => match inner.downcast::<DuckLakeError>() {
            Ok(ours) => *ours,
            Err(other) => DuckLakeError::DataFusion(DataFusionError::External(other)),
        },
        other => DuckLakeError::DataFusion(other),
    }
}
