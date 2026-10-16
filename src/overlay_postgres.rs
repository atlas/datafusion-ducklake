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
    let base_snapshot: i64 = row.try_get(1)?;

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

    hide_inlined_rows(
        tx,
        snapshot_id,
        overlay_table_id,
        underlay_table_id,
        base_snapshot,
        &commit.hide_inlined,
    )
    .await?;
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
