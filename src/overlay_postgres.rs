//! Overlay state in a PostgreSQL catalog (see [`crate::overlay`]).
//!
//! Four tables beside the lake's `ducklake_*` tables, each versioned by lake
//! snapshot like the catalog's own, so an overlay can be read as of any
//! snapshot that still exists:
//!
//! - `ducklake_overlay`: which table an overlay layers over, and at which base.
//!   A rebase ends one row and begins another.
//! - `ducklake_overlay_inlined_delete`: positions the overlay hides in
//!   underlay data files, as DuckLake's inlined deletions are laid out.
//! - `ducklake_overlay_inlined_row_delete`: the underlay's inlined rows the
//!   overlay hides, by row id, which DuckLake's own inlined rows are ended by.
//! - `ducklake_overlay_delete_file`: at most one live overlay delete file per
//!   underlay data file, holding the underlay's deletes at the base and the
//!   overlay's. Paths resolve against the underlay table's path.

use crate::Result;
use crate::metadata_pool_postgres::LakePool;
use crate::metadata_provider::{DuckLakeFileData, DuckLakeInlinedDelete};
use crate::overlay::{OverlayBase, OverlayDeleteFile};
use sqlx::Row;
use std::collections::HashMap;

/// Creates the overlay tables in the lake's schema if they are missing.
pub const SQL_CREATE_OVERLAY_TABLES: &str = "
CREATE TABLE IF NOT EXISTS ducklake_overlay (
    overlay_table_id BIGINT NOT NULL,
    underlay_table_id BIGINT NOT NULL,
    base_snapshot BIGINT NOT NULL,
    begin_snapshot BIGINT NOT NULL,
    end_snapshot BIGINT,
    PRIMARY KEY (overlay_table_id, begin_snapshot),
    CHECK (overlay_table_id <> underlay_table_id),
    CHECK (end_snapshot IS NULL OR end_snapshot > begin_snapshot)
);
CREATE UNIQUE INDEX IF NOT EXISTS ducklake_overlay_live
    ON ducklake_overlay (overlay_table_id) WHERE end_snapshot IS NULL;
CREATE TABLE IF NOT EXISTS ducklake_overlay_inlined_delete (
    overlay_table_id BIGINT NOT NULL,
    data_file_id BIGINT NOT NULL,
    row_id BIGINT NOT NULL,
    begin_snapshot BIGINT NOT NULL,
    end_snapshot BIGINT,
    PRIMARY KEY (overlay_table_id, data_file_id, row_id, begin_snapshot),
    CHECK (row_id >= 0),
    CHECK (end_snapshot IS NULL OR end_snapshot > begin_snapshot)
);
CREATE TABLE IF NOT EXISTS ducklake_overlay_inlined_row_delete (
    overlay_table_id BIGINT NOT NULL,
    row_id BIGINT NOT NULL,
    begin_snapshot BIGINT NOT NULL,
    end_snapshot BIGINT,
    PRIMARY KEY (overlay_table_id, row_id, begin_snapshot),
    CHECK (row_id >= 0),
    CHECK (end_snapshot IS NULL OR end_snapshot > begin_snapshot)
);
CREATE TABLE IF NOT EXISTS ducklake_overlay_delete_file (
    delete_file_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    overlay_table_id BIGINT NOT NULL,
    data_file_id BIGINT NOT NULL,
    path VARCHAR NOT NULL,
    path_is_relative BOOLEAN NOT NULL,
    file_size_bytes BIGINT NOT NULL,
    footer_size BIGINT,
    delete_count BIGINT NOT NULL,
    begin_snapshot BIGINT NOT NULL,
    end_snapshot BIGINT,
    CHECK (delete_count > 0),
    CHECK (end_snapshot IS NULL OR end_snapshot > begin_snapshot)
);
CREATE UNIQUE INDEX IF NOT EXISTS ducklake_overlay_delete_file_live
    ON ducklake_overlay_delete_file (overlay_table_id, data_file_id) WHERE end_snapshot IS NULL;
";

/// Creates the overlay tables in `pool`'s lake.
pub async fn initialize_overlay_schema(pool: &LakePool) -> Result<()> {
    let pool = pool.clone();
    // On the catalog runtime, like every other use of the pool, so a connection
    // it opens never belongs to the caller's runtime.
    crate::metadata_provider::connect_on_catalog_runtime(async move {
        sqlx::raw_sql(SQL_CREATE_OVERLAY_TABLES)
            .execute(&pool)
            .await?;
        Ok(())
    })
    .await
}

/// The overlay `overlay_table_id` as of `snapshot_id`, or `None` if that table
/// was not an overlay then.
///
/// It reads on the catalog runtime, like every other use of the pool, so a
/// connection it opens never belongs to the caller's runtime.
pub async fn load_overlay_base(
    pool: &LakePool,
    overlay_table_id: i64,
    snapshot_id: i64,
) -> Result<Option<OverlayBase>> {
    let pool = pool.clone();
    crate::metadata_provider::connect_on_catalog_runtime(async move {
        load_overlay_base_on(&pool, overlay_table_id, snapshot_id).await
    })
    .await
}

async fn load_overlay_base_on(
    pool: &LakePool,
    overlay_table_id: i64,
    snapshot_id: i64,
) -> Result<Option<OverlayBase>> {
    // The reads need no shared transaction: at a committed snapshot, what is
    // visible never changes. Later commits only add rows that begin after it
    // and end rows after it.
    let Some(overlay) = sqlx::query(
        "SELECT underlay_table_id, base_snapshot FROM ducklake_overlay
         WHERE overlay_table_id = $1
           AND begin_snapshot <= $2 AND (end_snapshot IS NULL OR $2 < end_snapshot)",
    )
    .bind(overlay_table_id)
    .bind(snapshot_id)
    .fetch_optional(pool)
    .await?
    else {
        return Ok(None);
    };

    let inlined_deletes = sqlx::query(
        "SELECT data_file_id, row_id FROM ducklake_overlay_inlined_delete
         WHERE overlay_table_id = $1
           AND begin_snapshot <= $2 AND (end_snapshot IS NULL OR $2 < end_snapshot)
         ORDER BY data_file_id, row_id",
    )
    .bind(overlay_table_id)
    .bind(snapshot_id)
    .fetch_all(pool)
    .await?
    .into_iter()
    .map(|row| {
        Ok(DuckLakeInlinedDelete {
            data_file_id: row.try_get(0)?,
            row_id: row.try_get(1)?,
        })
    })
    .collect::<Result<Vec<_>>>()?;

    let inlined_row_deletes = sqlx::query_scalar(
        "SELECT row_id FROM ducklake_overlay_inlined_row_delete
         WHERE overlay_table_id = $1
           AND begin_snapshot <= $2 AND (end_snapshot IS NULL OR $2 < end_snapshot)",
    )
    .bind(overlay_table_id)
    .bind(snapshot_id)
    .fetch_all(pool)
    .await?
    .into_iter()
    .collect();

    let mut delete_files = HashMap::new();
    for row in sqlx::query(
        "SELECT delete_file_id, data_file_id, path, path_is_relative, file_size_bytes,
                footer_size, delete_count
         FROM ducklake_overlay_delete_file
         WHERE overlay_table_id = $1
           AND begin_snapshot <= $2 AND (end_snapshot IS NULL OR $2 < end_snapshot)",
    )
    .bind(overlay_table_id)
    .bind(snapshot_id)
    .fetch_all(pool)
    .await?
    {
        let mut file = DuckLakeFileData::new(row.try_get(2)?, row.try_get(3)?, row.try_get(4)?);
        file.footer_size = row.try_get(5)?;
        let data_file_id: i64 = row.try_get(1)?;
        delete_files.insert(
            data_file_id,
            OverlayDeleteFile {
                delete_file_id: row.try_get(0)?,
                file,
                delete_count: row.try_get(6)?,
            },
        );
    }

    Ok(Some(OverlayBase {
        underlay_table_id: overlay.try_get(0)?,
        base_snapshot: overlay.try_get(1)?,
        inlined_deletes,
        inlined_row_deletes,
        delete_files,
    }))
}

/// Applies `commit` to the overlay tables at `snapshot_id`, inside the writer's
/// transaction. `overlay_table_id` is the committed id of the overlay's table.
#[cfg(feature = "write-postgres")]
pub(crate) async fn apply_overlay_at_snapshot(
    tx: &mut sqlx::Transaction<'_, sqlx::Postgres>,
    snapshot_id: i64,
    overlay_table_id: i64,
    commit: &crate::overlay::OverlayCommit,
    expected_base_snapshot_id: Option<i64>,
) -> Result<()> {
    use crate::DuckLakeError;

    if let Some((underlay_table_id, base_snapshot)) = commit.create {
        let existing: Option<i64> = sqlx::query_scalar(
            "SELECT overlay_table_id FROM ducklake_overlay
             WHERE overlay_table_id = $1 AND end_snapshot IS NULL",
        )
        .bind(overlay_table_id)
        .fetch_optional(&mut **tx)
        .await?;
        if existing.is_some() {
            return Err(DuckLakeError::InvalidConfig(format!(
                "table {overlay_table_id} is already an overlay"
            )));
        }
        let underlay_live: Option<i64> = sqlx::query_scalar(
            "SELECT table_id FROM ducklake_table
             WHERE table_id = $1 AND begin_snapshot <= $2
               AND (end_snapshot IS NULL OR $2 < end_snapshot)",
        )
        .bind(underlay_table_id)
        .bind(base_snapshot)
        .fetch_optional(&mut **tx)
        .await?;
        if underlay_live.is_none() {
            return Err(DuckLakeError::TableNotFound(format!(
                "underlay {underlay_table_id} at snapshot {base_snapshot}"
            )));
        }
        sqlx::query(
            "INSERT INTO ducklake_overlay
                 (overlay_table_id, underlay_table_id, base_snapshot, begin_snapshot)
             VALUES ($1, $2, $3, $4)",
        )
        .bind(overlay_table_id)
        .bind(underlay_table_id)
        .bind(base_snapshot)
        .bind(snapshot_id)
        .execute(&mut **tx)
        .await?;
    }

    // Lock the overlay's row so concurrent writes to the same overlay commit
    // one after another, each seeing what the other hid.
    let Some(row) = sqlx::query(
        "SELECT underlay_table_id, base_snapshot FROM ducklake_overlay
         WHERE overlay_table_id = $1 AND end_snapshot IS NULL
         FOR UPDATE",
    )
    .bind(overlay_table_id)
    .fetch_optional(&mut **tx)
    .await?
    else {
        return Err(DuckLakeError::InvalidConfig(format!(
            "table {overlay_table_id} is not an overlay"
        )));
    };
    let underlay_table_id: i64 = row.try_get(0)?;
    let mut base_snapshot: i64 = row.try_get(1)?;

    if let Some(expected) = expected_base_snapshot_id {
        let changed: bool = sqlx::query_scalar(
            "SELECT EXISTS(SELECT 1 FROM ducklake_overlay WHERE overlay_table_id = $1
                 AND (begin_snapshot > $2 OR end_snapshot > $2))
             OR EXISTS(SELECT 1 FROM ducklake_overlay_inlined_delete WHERE overlay_table_id = $1
                 AND (begin_snapshot > $2 OR end_snapshot > $2))
             OR EXISTS(SELECT 1 FROM ducklake_overlay_inlined_row_delete
                 WHERE overlay_table_id = $1 AND (begin_snapshot > $2 OR end_snapshot > $2))
             OR EXISTS(SELECT 1 FROM ducklake_overlay_delete_file WHERE overlay_table_id = $1
                 AND (begin_snapshot > $2 OR end_snapshot > $2))",
        )
        .bind(overlay_table_id)
        .bind(expected)
        .fetch_one(&mut **tx)
        .await?;
        if changed {
            return Err(DuckLakeError::Conflict(format!(
                "overlay {overlay_table_id} changed after snapshot {expected}; \
                 retry the write against the new state"
            )));
        }
    }

    if commit.merge {
        if commit.rebase.is_some()
            || !commit.hide.is_empty()
            || !commit.hide_inlined.is_empty()
            || !commit.fold.is_empty()
        {
            return Err(DuckLakeError::InvalidConfig(
                "a merge changes nothing else in the overlay".to_string(),
            ));
        }
        if expected_base_snapshot_id.is_none() {
            return Err(DuckLakeError::InvalidConfig(
                "a merge must name the overlay snapshot it read".to_string(),
            ));
        }
        return merge_at_snapshot(
            tx,
            snapshot_id,
            overlay_table_id,
            underlay_table_id,
            base_snapshot,
        )
        .await;
    }

    if let Some(new_base) = commit.rebase {
        rebase_at_snapshot(
            tx,
            snapshot_id,
            overlay_table_id,
            underlay_table_id,
            base_snapshot,
            new_base,
        )
        .await?;
        base_snapshot = new_base;
    }

    hide_inlined_rows(
        tx,
        snapshot_id,
        overlay_table_id,
        underlay_table_id,
        base_snapshot,
        &commit.hide_inlined,
    )
    .await?;
    for fold in &commit.fold {
        apply_fold(
            tx,
            snapshot_id,
            overlay_table_id,
            underlay_table_id,
            base_snapshot,
            fold,
        )
        .await?;
    }

    if commit.hide.is_empty() {
        return Ok(());
    }
    let (files, rows): (Vec<i64>, Vec<i64>) = commit
        .hide
        .iter()
        .map(|delete| (delete.data_file_id, delete.row_id))
        .unzip();
    // Every position must be in an underlay file live at the base, in range,
    // listed once, and not hidden already.
    let valid: i64 = sqlx::query_scalar(
        "SELECT COUNT(DISTINCT (h.f, h.r)) FROM unnest($1::BIGINT[], $2::BIGINT[]) AS h(f, r)
         JOIN ducklake_data_file d ON d.data_file_id = h.f
         WHERE d.table_id = $3 AND d.begin_snapshot <= $4
           AND (d.end_snapshot IS NULL OR $4 < d.end_snapshot)
           AND h.r >= 0 AND h.r < d.record_count
           AND NOT EXISTS (SELECT 1 FROM ducklake_overlay_inlined_delete o
                WHERE o.overlay_table_id = $5 AND o.data_file_id = h.f AND o.row_id = h.r
                  AND o.end_snapshot IS NULL)",
    )
    .bind(&files)
    .bind(&rows)
    .bind(underlay_table_id)
    .bind(base_snapshot)
    .bind(overlay_table_id)
    .fetch_one(&mut **tx)
    .await?;
    if usize::try_from(valid).ok() != Some(commit.hide.len()) {
        return Err(DuckLakeError::Conflict(format!(
            "overlay {overlay_table_id} cannot hide these rows: one is not in its underlay at \
             the base, is listed twice or is hidden already"
        )));
    }
    sqlx::query(
        "INSERT INTO ducklake_overlay_inlined_delete
             (overlay_table_id, data_file_id, row_id, begin_snapshot)
         SELECT $1, f, r, $2 FROM unnest($3::BIGINT[], $4::BIGINT[]) AS h(f, r)",
    )
    .bind(overlay_table_id)
    .bind(snapshot_id)
    .bind(&files)
    .bind(&rows)
    .execute(&mut **tx)
    .await?;
    Ok(())
}

/// Starts hiding the underlay's inlined rows `row_ids`. Each must be an
/// inlined row of the underlay visible at the base, listed once, and not
/// hidden already.
#[cfg(feature = "write-postgres")]
async fn hide_inlined_rows(
    tx: &mut sqlx::Transaction<'_, sqlx::Postgres>,
    snapshot_id: i64,
    overlay_table_id: i64,
    underlay_table_id: i64,
    base_snapshot: i64,
    row_ids: &[i64],
) -> Result<()> {
    use crate::DuckLakeError;
    use crate::metadata_writer_postgres::quote_ident;

    if row_ids.is_empty() {
        return Ok(());
    }
    let mut valid = 0_i64;
    for (table, system) in
        crate::metadata_writer_postgres::inlined_tables(tx, underlay_table_id).await?
    {
        let sql = format!(
            "SELECT COUNT(DISTINCT i.{row_id}) FROM {table} i
             WHERE i.{row_id} = ANY($1) AND i.{begin} <= $2
               AND (i.{end} IS NULL OR $2 < i.{end})
               AND NOT EXISTS (SELECT 1 FROM ducklake_overlay_inlined_row_delete o
                    WHERE o.overlay_table_id = $3 AND o.row_id = i.{row_id}
                      AND o.end_snapshot IS NULL)",
            table = quote_ident(&table),
            row_id = system.row_id,
            begin = system.begin_snapshot,
            end = system.end_snapshot,
        );
        valid += sqlx::query_scalar::<_, i64>(sqlx::AssertSqlSafe(sql))
            .bind(row_ids)
            .bind(base_snapshot)
            .bind(overlay_table_id)
            .fetch_one(&mut **tx)
            .await?;
    }
    if usize::try_from(valid).ok() != Some(row_ids.len()) {
        return Err(DuckLakeError::Conflict(format!(
            "overlay {overlay_table_id} cannot hide these inlined rows: one is not in its \
             underlay at the base, is listed twice or is hidden already"
        )));
    }
    sqlx::query(
        "INSERT INTO ducklake_overlay_inlined_row_delete
             (overlay_table_id, row_id, begin_snapshot)
         SELECT $1, r, $2 FROM unnest($3::BIGINT[]) AS h(r)",
    )
    .bind(overlay_table_id)
    .bind(snapshot_id)
    .bind(row_ids)
    .execute(&mut **tx)
    .await?;
    Ok(())
}

/// Supersedes the overlay's delete file for one underlay data file and ends
/// the inline positions it folds.
#[cfg(feature = "write-postgres")]
async fn apply_fold(
    tx: &mut sqlx::Transaction<'_, sqlx::Postgres>,
    snapshot_id: i64,
    overlay_table_id: i64,
    underlay_table_id: i64,
    base_snapshot: i64,
    fold: &crate::overlay::OverlayFold,
) -> Result<()> {
    use crate::DuckLakeError;

    let in_base: bool = sqlx::query_scalar(
        "SELECT EXISTS(SELECT 1 FROM ducklake_data_file
             WHERE data_file_id = $1 AND table_id = $2 AND begin_snapshot <= $3
               AND (end_snapshot IS NULL OR $3 < end_snapshot))",
    )
    .bind(fold.data_file_id)
    .bind(underlay_table_id)
    .bind(base_snapshot)
    .fetch_one(&mut **tx)
    .await?;
    if !in_base {
        return Err(DuckLakeError::Conflict(format!(
            "data file {} is not in overlay {overlay_table_id}'s underlay at the base",
            fold.data_file_id
        )));
    }
    let live: Option<i64> = sqlx::query_scalar(
        "SELECT delete_file_id FROM ducklake_overlay_delete_file
         WHERE overlay_table_id = $1 AND data_file_id = $2 AND end_snapshot IS NULL",
    )
    .bind(overlay_table_id)
    .bind(fold.data_file_id)
    .fetch_optional(&mut **tx)
    .await?;
    if live != fold.expected_prev_delete_file {
        return Err(DuckLakeError::Conflict(format!(
            "overlay {overlay_table_id}'s delete file for data file {} changed from {:?} to {live:?}",
            fold.data_file_id, fold.expected_prev_delete_file
        )));
    }
    if let Some(previous) = live {
        sqlx::query(
            "UPDATE ducklake_overlay_delete_file SET end_snapshot = $1 WHERE delete_file_id = $2",
        )
        .bind(snapshot_id)
        .bind(previous)
        .execute(&mut **tx)
        .await?;
    }
    sqlx::query(
        "INSERT INTO ducklake_overlay_delete_file
             (overlay_table_id, data_file_id, path, path_is_relative, file_size_bytes,
              footer_size, delete_count, begin_snapshot)
         VALUES ($1, $2, $3, $4, $5, $6, $7, $8)",
    )
    .bind(overlay_table_id)
    .bind(fold.data_file_id)
    .bind(&fold.file.path)
    .bind(fold.file.path_is_relative)
    .bind(fold.file.file_size_bytes)
    .bind(fold.file.footer_size)
    .bind(fold.delete_count)
    .bind(snapshot_id)
    .execute(&mut **tx)
    .await?;
    sqlx::query(
        "UPDATE ducklake_overlay_inlined_delete SET end_snapshot = $1
         WHERE overlay_table_id = $2 AND data_file_id = $3 AND end_snapshot IS NULL",
    )
    .bind(snapshot_id)
    .bind(overlay_table_id)
    .bind(fold.data_file_id)
    .execute(&mut **tx)
    .await?;
    Ok(())
}

/// Moves the overlay's base to `new_base` at `snapshot_id`: ends its base row
/// and begins one at the new base, and ends everything it hid at the old one.
#[cfg(feature = "write-postgres")]
async fn rebase_at_snapshot(
    tx: &mut sqlx::Transaction<'_, sqlx::Postgres>,
    snapshot_id: i64,
    overlay_table_id: i64,
    underlay_table_id: i64,
    base_snapshot: i64,
    new_base: i64,
) -> Result<()> {
    use crate::DuckLakeError;

    if new_base < base_snapshot {
        return Err(DuckLakeError::InvalidConfig(format!(
            "overlay {overlay_table_id} cannot rebase back from snapshot {base_snapshot} \
             to {new_base}"
        )));
    }
    let underlay_live: bool = sqlx::query_scalar(
        "SELECT EXISTS(SELECT 1 FROM ducklake_table
             WHERE table_id = $1 AND begin_snapshot <= $2
               AND (end_snapshot IS NULL OR $2 < end_snapshot))",
    )
    .bind(underlay_table_id)
    .bind(new_base)
    .fetch_one(&mut **tx)
    .await?;
    if !underlay_live {
        return Err(DuckLakeError::TableNotFound(format!(
            "underlay {underlay_table_id} at snapshot {new_base}"
        )));
    }
    for statement in [
        "UPDATE ducklake_overlay SET end_snapshot = $1
         WHERE overlay_table_id = $2 AND end_snapshot IS NULL",
        "UPDATE ducklake_overlay_inlined_delete SET end_snapshot = $1
         WHERE overlay_table_id = $2 AND end_snapshot IS NULL",
        "UPDATE ducklake_overlay_inlined_row_delete SET end_snapshot = $1
         WHERE overlay_table_id = $2 AND end_snapshot IS NULL",
        "UPDATE ducklake_overlay_delete_file SET end_snapshot = $1
         WHERE overlay_table_id = $2 AND end_snapshot IS NULL",
    ] {
        sqlx::query(statement)
            .bind(snapshot_id)
            .bind(overlay_table_id)
            .execute(&mut **tx)
            .await?;
    }
    sqlx::query(
        "INSERT INTO ducklake_overlay
             (overlay_table_id, underlay_table_id, base_snapshot, begin_snapshot)
         VALUES ($1, $2, $3, $4)",
    )
    .bind(overlay_table_id)
    .bind(underlay_table_id)
    .bind(new_base)
    .bind(snapshot_id)
    .execute(&mut **tx)
    .await?;
    Ok(())
}

/// Whether the table's inlined rows or inlined deletions of its file rows
/// changed after `snapshot`.
#[cfg(feature = "write-postgres")]
async fn inlined_changed_since(
    tx: &mut sqlx::Transaction<'_, sqlx::Postgres>,
    table_id: i64,
    snapshot: i64,
) -> Result<bool> {
    use crate::metadata_writer_postgres::quote_ident;

    if sqlx::query_scalar::<_, bool>(
        "SELECT to_regclass('ducklake_inlined_data_tables') IS NOT NULL",
    )
    .fetch_one(&mut **tx)
    .await?
    {
        for (table, system) in crate::metadata_writer_postgres::inlined_tables(tx, table_id).await?
        {
            let sql = format!(
                "SELECT EXISTS(SELECT 1 FROM {} WHERE {begin} > $1 OR {end} > $1)",
                quote_ident(&table),
                begin = system.begin_snapshot,
                end = system.end_snapshot,
            );
            if sqlx::query_scalar::<_, bool>(sqlx::AssertSqlSafe(sql))
                .bind(snapshot)
                .fetch_one(&mut **tx)
                .await?
            {
                return Ok(true);
            }
        }
    }
    let deletes = crate::metadata_provider::inlined_delete_table_name(table_id)?;
    if !sqlx::query_scalar::<_, bool>("SELECT to_regclass($1) IS NOT NULL")
        .bind(&deletes)
        .fetch_one(&mut **tx)
        .await?
    {
        return Ok(false);
    }
    let sql = format!(
        "SELECT EXISTS(SELECT 1 FROM {} WHERE begin_snapshot > $1)",
        quote_ident(&deletes)
    );
    Ok(sqlx::query_scalar::<_, bool>(sqlx::AssertSqlSafe(sql))
        .bind(snapshot)
        .fetch_one(&mut **tx)
        .await?)
}

/// Merges the overlay into its underlay at `snapshot_id` and closes it. The
/// overlay's delete files become the underlay's, and the overlay table's live
/// files, each with its delete file, are registered as the underlay's by
/// absolute path, with new row ids.
#[cfg(feature = "write-postgres")]
async fn merge_at_snapshot(
    tx: &mut sqlx::Transaction<'_, sqlx::Postgres>,
    snapshot_id: i64,
    overlay_table_id: i64,
    underlay_table_id: i64,
    base_snapshot: i64,
) -> Result<()> {
    use crate::DuckLakeError;
    use crate::path_resolver::resolve_path;

    let changed: bool = sqlx::query_scalar(
        "SELECT EXISTS(SELECT 1 FROM ducklake_data_file WHERE table_id = $1
             AND (begin_snapshot > $2 OR end_snapshot > $2))
         OR EXISTS(SELECT 1 FROM ducklake_delete_file WHERE table_id = $1
             AND (begin_snapshot > $2 OR end_snapshot > $2))
         OR EXISTS(SELECT 1 FROM ducklake_column WHERE table_id = $1
             AND (begin_snapshot > $2 OR end_snapshot > $2))
         OR EXISTS(SELECT 1 FROM ducklake_table WHERE table_id = $1
             AND (begin_snapshot > $2 OR end_snapshot IS NOT NULL))",
    )
    .bind(underlay_table_id)
    .bind(base_snapshot)
    .fetch_one(&mut **tx)
    .await?;
    if changed || inlined_changed_since(tx, underlay_table_id, base_snapshot).await? {
        return Err(DuckLakeError::Conflict(format!(
            "underlay {underlay_table_id} changed after overlay {overlay_table_id}'s base \
             {base_snapshot}; rebase the overlay first"
        )));
    }

    let inline_hides: bool = sqlx::query_scalar(
        "SELECT EXISTS(SELECT 1 FROM ducklake_overlay_inlined_delete
             WHERE overlay_table_id = $1 AND end_snapshot IS NULL)",
    )
    .bind(overlay_table_id)
    .fetch_one(&mut **tx)
    .await?;
    let inline_tables: Vec<String> = sqlx::query_scalar(
        "SELECT table_name FROM ducklake_inlined_data_tables WHERE table_id = $1",
    )
    .bind(overlay_table_id)
    .fetch_all(&mut **tx)
    .await?;
    let mut inline_rows = false;
    for name in inline_tables {
        let sql = format!(
            "SELECT EXISTS(SELECT 1 FROM {} WHERE end_snapshot IS NULL)",
            crate::metadata_writer_postgres::quote_ident(&name)
        );
        inline_rows |= sqlx::query_scalar::<_, bool>(sqlx::AssertSqlSafe(sql))
            .fetch_one(&mut **tx)
            .await?;
    }
    let deleted_inline = crate::metadata_provider::inlined_delete_table_name(overlay_table_id)?;
    if sqlx::query_scalar::<_, bool>("SELECT to_regclass($1) IS NOT NULL")
        .bind(&deleted_inline)
        .fetch_one(&mut **tx)
        .await?
    {
        let sql = format!(
            "SELECT EXISTS(SELECT 1 FROM {})",
            crate::metadata_writer_postgres::quote_ident(&deleted_inline)
        );
        inline_rows |= sqlx::query_scalar::<_, bool>(sqlx::AssertSqlSafe(sql))
            .fetch_one(&mut **tx)
            .await?;
    }
    if inline_hides || inline_rows {
        return Err(DuckLakeError::InvalidConfig(format!(
            "overlay {overlay_table_id} holds rows inline; flush it before merging"
        )));
    }

    // The overlay's delete files hold the underlay's deletes at the base, which
    // are its deletes now, so they take the place of the underlay's.
    let delete_files = sqlx::query(
        "SELECT data_file_id, path, path_is_relative, file_size_bytes, footer_size, delete_count
         FROM ducklake_overlay_delete_file
         WHERE overlay_table_id = $1 AND end_snapshot IS NULL",
    )
    .bind(overlay_table_id)
    .fetch_all(&mut **tx)
    .await?;
    for row in &delete_files {
        let data_file_id: i64 = row.try_get(0)?;
        sqlx::query(
            "UPDATE ducklake_delete_file SET end_snapshot = $1
             WHERE data_file_id = $2 AND end_snapshot IS NULL",
        )
        .bind(snapshot_id)
        .bind(data_file_id)
        .execute(&mut **tx)
        .await?;
        sqlx::query(
            "INSERT INTO ducklake_delete_file
                 (data_file_id, table_id, path, path_is_relative, file_size_bytes, footer_size,
                  delete_count, begin_snapshot)
             VALUES ($1, $2, $3, $4, $5, $6, $7, $8)",
        )
        .bind(data_file_id)
        .bind(underlay_table_id)
        .bind(row.try_get::<String, _>(1)?)
        .bind(row.try_get::<bool, _>(2)?)
        .bind(row.try_get::<i64, _>(3)?)
        .bind(row.try_get::<Option<i64>, _>(4)?)
        .bind(row.try_get::<i64, _>(5)?)
        .bind(snapshot_id)
        .execute(&mut **tx)
        .await?;
    }

    // The overlay table's files, which carry the underlay's column ids.
    let data_path: String = sqlx::query_scalar(
        "SELECT value FROM ducklake_metadata
         WHERE key = 'data_path' AND (scope IS NULL OR scope = '')",
    )
    .fetch_one(&mut **tx)
    .await?;
    let location = sqlx::query(
        "SELECT s.path, s.path_is_relative, t.path, t.path_is_relative
         FROM ducklake_table t JOIN ducklake_schema s ON s.schema_id = t.schema_id
         WHERE t.table_id = $1 AND t.end_snapshot IS NULL",
    )
    .bind(overlay_table_id)
    .fetch_one(&mut **tx)
    .await?;
    let schema_path = resolve_path(
        &data_path,
        &location.try_get::<String, _>(0)?,
        location.try_get(1)?,
    )?;
    let table_path = resolve_path(
        &schema_path,
        &location.try_get::<String, _>(2)?,
        location.try_get(3)?,
    )?;

    let files = sqlx::query(
        "SELECT d.data_file_id, d.path, d.path_is_relative, d.file_size_bytes, d.footer_size,
                d.record_count, d.row_id_start, d.partition_id, d.encryption_key, d.mapping_id,
                del.path, del.path_is_relative, del.file_size_bytes, del.footer_size,
                del.delete_count
         FROM ducklake_data_file d
         LEFT JOIN ducklake_delete_file del
           ON del.data_file_id = d.data_file_id AND del.end_snapshot IS NULL
         WHERE d.table_id = $1 AND d.end_snapshot IS NULL
         ORDER BY d.data_file_id",
    )
    .bind(overlay_table_id)
    .fetch_all(&mut **tx)
    .await?;
    sqlx::query(
        "INSERT INTO ducklake_table_stats (table_id, record_count, next_row_id, file_size_bytes)
         VALUES ($1, 0, 0, 0) ON CONFLICT (table_id) DO NOTHING",
    )
    .bind(underlay_table_id)
    .execute(&mut **tx)
    .await?;
    let mut next_row_id: i64 = sqlx::query_scalar(
        "SELECT next_row_id FROM ducklake_table_stats WHERE table_id = $1 FOR UPDATE",
    )
    .bind(underlay_table_id)
    .fetch_one(&mut **tx)
    .await?;
    let (mut records, mut bytes) = (0_i64, 0_i64);
    let mut file_stats = Vec::with_capacity(files.len());
    for file in &files {
        let old_id: i64 = file.try_get(0)?;
        let record_count: i64 = file.try_get::<Option<i64>, _>(5)?.unwrap_or(0);
        if file.try_get::<Option<i64>, _>(6)?.is_none()
            || file.try_get::<Option<i64>, _>(7)?.is_some()
            || file.try_get::<Option<String>, _>(8)?.is_some()
            || file.try_get::<Option<i64>, _>(9)?.is_some()
        {
            return Err(DuckLakeError::Unsupported(format!(
                "overlay file {old_id} embeds row ids, is partitioned, encrypted or mapped, \
                 and cannot be merged as it is"
            )));
        }
        let path = resolve_path(
            &table_path,
            &file.try_get::<String, _>(1)?,
            file.try_get(2)?,
        )?;
        let size: i64 = file.try_get(3)?;
        let new_id: i64 = sqlx::query_scalar(
            "INSERT INTO ducklake_data_file
                 (table_id, path, path_is_relative, file_size_bytes, footer_size, record_count,
                  row_id_start, begin_snapshot)
             VALUES ($1, $2, FALSE, $3, $4, $5, $6, $7) RETURNING data_file_id",
        )
        .bind(underlay_table_id)
        .bind(&path)
        .bind(size)
        .bind(file.try_get::<Option<i64>, _>(4)?)
        .bind(record_count)
        .bind(next_row_id)
        .bind(snapshot_id)
        .fetch_one(&mut **tx)
        .await?;
        sqlx::query(
            "INSERT INTO ducklake_file_column_stats
                 (data_file_id, table_id, column_id, column_size_bytes, value_count, null_count,
                  min_value, max_value, contains_nan, extra_stats)
             SELECT $1, $2, column_id, column_size_bytes, value_count, null_count,
                    min_value, max_value, contains_nan, extra_stats
             FROM ducklake_file_column_stats WHERE data_file_id = $3",
        )
        .bind(new_id)
        .bind(underlay_table_id)
        .bind(old_id)
        .execute(&mut **tx)
        .await?;
        file_stats.push(
            sqlx::query_as::<
                _,
                (
                    i64,
                    Option<String>,
                    Option<String>,
                    Option<i64>,
                    Option<bool>,
                    Option<i64>,
                    Option<String>,
                ),
            >(
                "SELECT column_id, min_value, max_value, null_count, contains_nan, value_count,
                        extra_stats
                 FROM ducklake_file_column_stats WHERE data_file_id = $1",
            )
            .bind(old_id)
            .fetch_all(&mut **tx)
            .await?
            .into_iter()
            .map(
                |(
                    column_id,
                    min_value,
                    max_value,
                    null_count,
                    contains_nan,
                    value_count,
                    extra_stats,
                )| {
                    crate::stats_encode::FileColumnStat {
                        column_id,
                        min_value,
                        max_value,
                        null_count,
                        contains_nan,
                        value_count,
                        extra_stats,
                    }
                },
            )
            .collect::<Vec<_>>(),
        );
        if let Some(delete_path) = file.try_get::<Option<String>, _>(10)? {
            let delete_path = resolve_path(&table_path, &delete_path, file.try_get(11)?)?;
            sqlx::query(
                "INSERT INTO ducklake_delete_file
                     (data_file_id, table_id, path, path_is_relative, file_size_bytes,
                      footer_size, delete_count, begin_snapshot)
                 VALUES ($1, $2, $3, FALSE, $4, $5, $6, $7)",
            )
            .bind(new_id)
            .bind(underlay_table_id)
            .bind(&delete_path)
            .bind(file.try_get::<i64, _>(12)?)
            .bind(file.try_get::<Option<i64>, _>(13)?)
            .bind(file.try_get::<Option<i64>, _>(14)?)
            .bind(snapshot_id)
            .execute(&mut **tx)
            .await?;
        }
        next_row_id += record_count;
        records += record_count;
        bytes += size;
    }
    sqlx::query(
        "UPDATE ducklake_table_stats
         SET next_row_id = $1, record_count = record_count + $2,
             file_size_bytes = file_size_bytes + $3
         WHERE table_id = $4",
    )
    .bind(next_row_id)
    .bind(records)
    .bind(bytes)
    .bind(underlay_table_id)
    .execute(&mut **tx)
    .await?;
    // The merged files' column statistics roll into the underlay's, as any
    // file a commit registers does.
    let fields = crate::metadata_writer_postgres::live_stat_fields(underlay_table_id, tx).await?;
    crate::metadata_writer_postgres::merge_file_table_column_stats(
        tx,
        underlay_table_id,
        &fields,
        &file_stats,
    )
    .await?;

    // The underlay's inlined rows the overlay hides end in the underlay, as a
    // DELETE of them would end them.
    let hidden: Vec<i64> = sqlx::query_scalar(
        "SELECT row_id FROM ducklake_overlay_inlined_row_delete
         WHERE overlay_table_id = $1 AND end_snapshot IS NULL",
    )
    .bind(overlay_table_id)
    .fetch_all(&mut **tx)
    .await?;
    if !hidden.is_empty() {
        let mut ended = 0_u64;
        for (table, system) in
            crate::metadata_writer_postgres::inlined_tables(tx, underlay_table_id).await?
        {
            let sql = format!(
                "UPDATE {} SET {end} = $1 WHERE {row_id} = ANY($2) AND {end} IS NULL",
                crate::metadata_writer_postgres::quote_ident(&table),
                end = system.end_snapshot,
                row_id = system.row_id,
            );
            ended += sqlx::query(sqlx::AssertSqlSafe(sql))
                .bind(snapshot_id)
                .bind(&hidden)
                .execute(&mut **tx)
                .await?
                .rows_affected();
        }
        if usize::try_from(ended).ok() != Some(hidden.len()) {
            return Err(DuckLakeError::Conflict(format!(
                "underlay {underlay_table_id} no longer holds every inlined row overlay \
                 {overlay_table_id} hides; rebase the overlay first"
            )));
        }
    }

    for statement in [
        "UPDATE ducklake_overlay SET end_snapshot = $1
         WHERE overlay_table_id = $2 AND end_snapshot IS NULL",
        "UPDATE ducklake_overlay_inlined_row_delete SET end_snapshot = $1
         WHERE overlay_table_id = $2 AND end_snapshot IS NULL",
        "UPDATE ducklake_overlay_delete_file SET end_snapshot = $1
         WHERE overlay_table_id = $2 AND end_snapshot IS NULL",
    ] {
        sqlx::query(statement)
            .bind(snapshot_id)
            .bind(overlay_table_id)
            .execute(&mut **tx)
            .await?;
    }
    Ok(())
}
