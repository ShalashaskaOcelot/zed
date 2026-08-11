use anyhow::Result;
use db::{
    query,
    sqlez::{
        bindable::{Bind, Column, StaticColumnCount},
        domain::Domain,
        statement::Statement,
    },
    sqlez_macros::sql,
};
use fs::MTime;
use std::path::PathBuf;
use workspace::{ItemId, WorkspaceDb, WorkspaceId};

/// Persisted form of a notebook item for workspace session restore, following
/// `SerializedEditor`: `contents` holds the notebook's nbformat JSON whenever it
/// has UNSAVED changes (whether or not it is file-backed), so hot exit restores
/// them, and is `None` for a clean notebook, which is simply reopened from
/// `abs_path`. `mtime` is what the file had when we last read or wrote it, so a
/// change made while Zed was closed can be told apart from our own last save.
#[derive(Clone, Debug, PartialEq, Default)]
pub(crate) struct SerializedNotebook {
    pub(crate) abs_path: Option<PathBuf>,
    pub(crate) contents: Option<String>,
    pub(crate) mtime: Option<MTime>,
}

impl StaticColumnCount for SerializedNotebook {
    fn column_count() -> usize {
        4
    }
}

impl Bind for SerializedNotebook {
    fn bind(&self, statement: &Statement, start_index: i32) -> Result<i32> {
        let start_index = statement.bind(&self.abs_path, start_index)?;
        let start_index = statement.bind(&self.contents, start_index)?;
        // Split across two columns because sqlez has no `MTime` binding and the
        // value has to survive as (seconds, nanos) exactly — the same shape
        // `SerializedEditor` uses.
        let start_index = match self
            .mtime
            .and_then(|mtime| mtime.to_seconds_and_nanos_for_persistence())
        {
            Some((seconds, nanos)) => {
                let start_index = statement.bind(&(seconds as i64), start_index)?;
                statement.bind(&(nanos as i32), start_index)?
            }
            None => {
                let start_index = statement.bind::<Option<i64>>(&None, start_index)?;
                statement.bind::<Option<i32>>(&None, start_index)?
            }
        };
        Ok(start_index)
    }
}

impl Column for SerializedNotebook {
    fn column(statement: &mut Statement, start_index: i32) -> Result<(Self, i32)> {
        let (abs_path, start_index): (Option<PathBuf>, i32) =
            Column::column(statement, start_index)?;
        let (contents, start_index): (Option<String>, i32) =
            Column::column(statement, start_index)?;
        let (mtime_seconds, start_index): (Option<i64>, i32) =
            Column::column(statement, start_index)?;
        let (mtime_nanos, start_index): (Option<i32>, i32) =
            Column::column(statement, start_index)?;
        let mtime = mtime_seconds
            .zip(mtime_nanos)
            .map(|(seconds, nanos)| MTime::from_seconds_and_nanos(seconds as u64, nanos as u32));
        Ok((
            Self {
                abs_path,
                contents,
                mtime,
            },
            start_index,
        ))
    }
}

pub(crate) struct NotebookDb(db::sqlez::thread_safe_connection::ThreadSafeConnection);

impl Domain for NotebookDb {
    const NAME: &str = stringify!(NotebookDb);

    const MIGRATIONS: &[&str] = &[
        sql!(
            CREATE TABLE notebook_editors(
                item_id INTEGER NOT NULL,
                workspace_id INTEGER NOT NULL,
                path BLOB,
                contents TEXT,
                PRIMARY KEY(item_id, workspace_id),
                FOREIGN KEY(workspace_id) REFERENCES workspaces(workspace_id)
                ON DELETE CASCADE
                ON UPDATE CASCADE
            ) STRICT;
        ),
        sql!(
            ALTER TABLE notebook_editors ADD COLUMN mtime_seconds INTEGER DEFAULT NULL;
            ALTER TABLE notebook_editors ADD COLUMN mtime_nanos INTEGER DEFAULT NULL;
        ),
    ];
}

db::static_connection!(NotebookDb, [WorkspaceDb]);

impl NotebookDb {
    query! {
        pub fn get_serialized_notebook(item_id: ItemId, workspace_id: WorkspaceId) -> Result<Option<SerializedNotebook>> {
            SELECT path, contents, mtime_seconds, mtime_nanos FROM notebook_editors
            WHERE item_id = ? AND workspace_id = ?
        }
    }

    query! {
        pub async fn save_serialized_notebook(item_id: ItemId, workspace_id: WorkspaceId, serialized_notebook: SerializedNotebook) -> Result<()> {
            INSERT INTO notebook_editors
                (item_id, workspace_id, path, contents, mtime_seconds, mtime_nanos)
            VALUES
                (?1, ?2, ?3, ?4, ?5, ?6)
            ON CONFLICT DO UPDATE SET
                item_id = ?1,
                workspace_id = ?2,
                path = ?3,
                contents = ?4,
                mtime_seconds = ?5,
                mtime_nanos = ?6
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The mtime is split across two columns by hand, so the column indices in
    /// `Bind`/`Column` and in the two queries all have to agree. A round trip is
    /// the cheapest way to keep them honest — an off-by-one silently returns the
    /// wrong field rather than failing to compile.
    #[gpui::test]
    async fn test_save_and_get_serialized_notebook(cx: &mut gpui::TestAppContext) {
        let db = cx.update(|cx| WorkspaceDb::global(cx));
        let workspace_id = db.next_id().await.unwrap();
        let notebook_db = cx.update(|cx| NotebookDb::global(cx));

        // A clean saved notebook: path only, nothing to restore.
        let serialized = SerializedNotebook {
            abs_path: Some(PathBuf::from("analysis.ipynb")),
            contents: None,
            mtime: None,
        };
        notebook_db
            .save_serialized_notebook(1234, workspace_id, serialized.clone())
            .await
            .unwrap();
        assert_eq!(
            notebook_db
                .get_serialized_notebook(1234, workspace_id)
                .unwrap()
                .unwrap(),
            serialized
        );

        // A saved notebook with unsaved changes: path, contents AND the mtime
        // the file had when we last agreed with it.
        let serialized = SerializedNotebook {
            abs_path: Some(PathBuf::from("analysis.ipynb")),
            contents: Some(r#"{"cells":[]}"#.to_owned()),
            mtime: Some(MTime::from_seconds_and_nanos(100, 42)),
        };
        notebook_db
            .save_serialized_notebook(1234, workspace_id, serialized.clone())
            .await
            .unwrap();
        assert_eq!(
            notebook_db
                .get_serialized_notebook(1234, workspace_id)
                .unwrap()
                .unwrap(),
            serialized
        );

        // An untitled notebook: contents only. Overwriting the row above also
        // checks that the mtime columns are cleared rather than left behind.
        let serialized = SerializedNotebook {
            abs_path: None,
            contents: Some(r#"{"cells":[]}"#.to_owned()),
            mtime: None,
        };
        notebook_db
            .save_serialized_notebook(1234, workspace_id, serialized.clone())
            .await
            .unwrap();
        assert_eq!(
            notebook_db
                .get_serialized_notebook(1234, workspace_id)
                .unwrap()
                .unwrap(),
            serialized
        );
    }
}
