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
use std::path::PathBuf;
use workspace::{ItemId, WorkspaceDb, WorkspaceId};

/// Persisted form of a notebook item for workspace session restore. A SAVED
/// notebook stores only its `abs_path` and is reopened by path (reloaded from
/// disk); an UNTITLED notebook stores its nbformat JSON in `contents` and comes
/// back as an untitled item with its cells intact.
#[derive(Clone, Debug, PartialEq, Default)]
pub(crate) struct SerializedNotebook {
    pub(crate) abs_path: Option<PathBuf>,
    pub(crate) contents: Option<String>,
}

impl StaticColumnCount for SerializedNotebook {
    fn column_count() -> usize {
        2
    }
}

impl Bind for SerializedNotebook {
    fn bind(&self, statement: &Statement, start_index: i32) -> Result<i32> {
        let start_index = statement.bind(&self.abs_path, start_index)?;
        let start_index = statement.bind(&self.contents, start_index)?;
        Ok(start_index)
    }
}

impl Column for SerializedNotebook {
    fn column(statement: &mut Statement, start_index: i32) -> Result<(Self, i32)> {
        let (abs_path, start_index): (Option<PathBuf>, i32) =
            Column::column(statement, start_index)?;
        let (contents, start_index): (Option<String>, i32) =
            Column::column(statement, start_index)?;
        Ok((Self { abs_path, contents }, start_index))
    }
}

pub(crate) struct NotebookDb(db::sqlez::thread_safe_connection::ThreadSafeConnection);

impl Domain for NotebookDb {
    const NAME: &str = stringify!(NotebookDb);

    const MIGRATIONS: &[&str] = &[sql!(
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
    )];
}

db::static_connection!(NotebookDb, [WorkspaceDb]);

impl NotebookDb {
    query! {
        pub fn get_serialized_notebook(item_id: ItemId, workspace_id: WorkspaceId) -> Result<Option<SerializedNotebook>> {
            SELECT path, contents FROM notebook_editors
            WHERE item_id = ? AND workspace_id = ?
        }
    }

    query! {
        pub async fn save_serialized_notebook(item_id: ItemId, workspace_id: WorkspaceId, serialized_notebook: SerializedNotebook) -> Result<()> {
            INSERT INTO notebook_editors
                (item_id, workspace_id, path, contents)
            VALUES
                (?1, ?2, ?3, ?4)
            ON CONFLICT DO UPDATE SET
                item_id = ?1,
                workspace_id = ?2,
                path = ?3,
                contents = ?4
        }
    }
}
