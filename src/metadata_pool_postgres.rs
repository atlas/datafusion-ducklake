//! A Postgres pool shared by many lakes, each in its own schema.
//!
//! The catalog SQL names its tables unqualified, so which lake a statement
//! reaches is decided by the connection's `search_path`. A [`LakePool`] sets it
//! to the lake's schema every time it hands out a connection, so one pool can
//! serve any number of lakes without a pool per lake. DuckDB attaches the same
//! layout with `METADATA_SCHEMA`.

use futures::future::BoxFuture;
use futures::stream::{self, BoxStream, StreamExt, TryStreamExt};
use sqlx::pool::PoolConnection;
use sqlx::postgres::{PgPool, PgQueryResult, PgRow, PgStatement, PgTypeInfo};
use sqlx::{Either, Execute, Executor, Postgres, Transaction};

/// A connection pool and the schema holding one lake's catalog tables.
///
/// Without a schema the connection's own `search_path` is used, as before; a
/// pool shared with lakes that have one must not be used that way, because a
/// connection keeps the last lake's `search_path` when it goes back.
#[derive(Debug, Clone)]
pub struct LakePool {
    pool: PgPool,
    search_path: Option<String>,
}

impl LakePool {
    /// The lake whose tables resolve through each connection's `search_path`.
    pub fn new(pool: PgPool) -> Self {
        Self {
            pool,
            search_path: None,
        }
    }

    /// The lake whose tables are in `schema`.
    pub fn in_schema(pool: PgPool, schema: &str) -> Self {
        Self {
            pool,
            search_path: Some(quote(schema)),
        }
    }

    /// Opens a pool of at most `max_connections` to a database holding lakes, for
    /// [`LakePool::in_schema`] to share between them. Like
    /// [`crate::PostgresMetadataProvider::new`], it opens the pool on the runtime this
    /// crate drives catalog I/O on, so a synchronous catalog call never waits on a
    /// connection registered with the caller's runtime.
    pub async fn connect(url: &str, max_connections: u32) -> crate::Result<PgPool> {
        let url = url.to_string();
        Ok(
            crate::metadata_provider::connect_on_catalog_runtime(async move {
                sqlx::postgres::PgPoolOptions::new()
                    .max_connections(max_connections)
                    .connect(&url)
                    .await
            })
            .await?,
        )
    }

    /// The pool, shared with every other lake on it.
    pub fn pool(&self) -> &PgPool {
        &self.pool
    }

    /// The schema's quoted name, when the lake has one.
    pub fn schema(&self) -> Option<&str> {
        self.search_path.as_deref()
    }

    /// A connection whose unqualified names reach this lake.
    pub async fn acquire(&self) -> Result<PoolConnection<Postgres>, sqlx::Error> {
        let mut connection = self.pool.acquire().await?;
        if let Some(path) = &self.search_path {
            sqlx::query("SELECT set_config('search_path', $1, false)")
                .bind(path)
                .execute(&mut *connection)
                .await?;
        }
        Ok(connection)
    }

    /// A transaction whose unqualified names reach this lake.
    pub async fn begin(&self) -> Result<Transaction<'static, Postgres>, sqlx::Error> {
        let mut transaction = self.pool.begin().await?;
        if let Some(path) = &self.search_path {
            sqlx::query("SELECT set_config('search_path', $1, true)")
                .bind(path)
                .execute(&mut *transaction)
                .await?;
        }
        Ok(transaction)
    }
}

/// Something a catalog transaction can be begun on: a pool for one lake, or
/// one whose connections' own `search_path` decides.
pub(crate) trait Begin: Sync {
    fn begin_catalog(&self) -> BoxFuture<'_, Result<Transaction<'static, Postgres>, sqlx::Error>>;
}

impl Begin for PgPool {
    fn begin_catalog(&self) -> BoxFuture<'_, Result<Transaction<'static, Postgres>, sqlx::Error>> {
        Box::pin(self.begin())
    }
}

impl Begin for LakePool {
    fn begin_catalog(&self) -> BoxFuture<'_, Result<Transaction<'static, Postgres>, sqlx::Error>> {
        Box::pin(self.begin())
    }
}

fn quote(schema: &str) -> String {
    format!("\"{}\"", schema.replace('"', "\"\""))
}

impl<'p> Executor<'p> for &'_ LakePool {
    type Database = Postgres;

    fn fetch_many<'e, 'q: 'e, E>(
        self,
        query: E,
    ) -> BoxStream<'e, Result<Either<PgQueryResult, PgRow>, sqlx::Error>>
    where
        E: 'q + Execute<'q, Postgres>,
    {
        let lake = self.clone();
        // Collected because the stream would otherwise borrow the connection it
        // owns; catalog queries are small and `fetch_all` collects anyway.
        stream::once(async move {
            let mut connection = lake.acquire().await?;
            connection.fetch_many(query).try_collect::<Vec<_>>().await
        })
        .map(|rows| match rows {
            Ok(rows) => stream::iter(rows.into_iter().map(Ok)).boxed(),
            Err(error) => stream::once(async move { Err(error) }).boxed(),
        })
        .flatten()
        .boxed()
    }

    fn fetch_optional<'e, 'q: 'e, E>(
        self,
        query: E,
    ) -> BoxFuture<'e, Result<Option<PgRow>, sqlx::Error>>
    where
        E: 'q + Execute<'q, Postgres>,
    {
        let lake = self.clone();
        Box::pin(async move { lake.acquire().await?.fetch_optional(query).await })
    }

    fn prepare_with<'e>(
        self,
        sql: sqlx::SqlStr,
        parameters: &'e [PgTypeInfo],
    ) -> BoxFuture<'e, Result<PgStatement, sqlx::Error>>
    where
        'p: 'e,
    {
        let lake = self.clone();
        Box::pin(async move { lake.acquire().await?.prepare_with(sql, parameters).await })
    }

    fn describe<'e>(
        self,
        sql: sqlx::SqlStr,
    ) -> BoxFuture<'e, Result<sqlx::Describe<Postgres>, sqlx::Error>> {
        let lake = self.clone();
        Box::pin(async move { lake.acquire().await?.describe(sql).await })
    }
}
