//! Overlay state in a PostgreSQL catalog (see [`crate::overlay`]).
//!
//! Three tables beside the lake's `ducklake_*` tables, each versioned by lake
//! snapshot like the catalog's own, so an overlay can be read as of any
//! snapshot that still exists:
//!
//! - `ducklake_overlay`: which table an overlay layers over, and at which base.
//!   A rebase ends one row and begins another.
//! - `ducklake_overlay_inlined_delete`: positions the overlay hides in
//!   underlay data files, as DuckLake's inlined deletions are laid out.
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
        delete_files,
    }))
}
