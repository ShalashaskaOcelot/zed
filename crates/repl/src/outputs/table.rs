//! # Table Output for REPL
//!
//! This module provides functionality to render tabular data in Zed's REPL output.
//!
//! It supports the [Frictionless Data Table Schema](https://specs.frictionlessdata.io/table-schema/)
//! for data interchange, implemented by Pandas in Python and Polars for Deno.
//!
//! # Python Example
//!
//! Tables can be created and displayed in two main ways:
//!
//! 1. Using raw JSON data conforming to the Tabular Data Resource specification.
//! 2. Using Pandas DataFrames (in Python kernels).
//!
//! ## Raw JSON Method
//!
//! To create a table using raw JSON, you need to provide a JSON object that conforms
//! to the Tabular Data Resource specification. Here's an example:
//!
//! ```json
//! {
//!     "schema": {
//!         "fields": [
//!             {"name": "id", "type": "integer"},
//!             {"name": "name", "type": "string"},
//!             {"name": "age", "type": "integer"}
//!         ]
//!     },
//!     "data": [
//!         {"id": 1, "name": "Alice", "age": 30},
//!         {"id": 2, "name": "Bob", "age": 28},
//!         {"id": 3, "name": "Charlie", "age": 35}
//!     ]
//! }
//! ```
//!
//! ## Pandas Method
//!
//! To create a table using Pandas in a Python kernel, you can use the following steps:
//!
//! ```python
//! import pandas as pd
//!
//! # Enable table schema output
//! pd.set_option('display.html.table_schema', True)
//!
//! # Create a DataFrame
//! df = pd.DataFrame({
//!     'id': [1, 2, 3],
//!     'name': ['Alice', 'Bob', 'Charlie'],
//!     'age': [30, 28, 35]
//! })
//!
//! # Display the DataFrame
//! display(df)
//! ```
use gpui::{AnyElement, ClipboardItem, FontWeight, TextRun};
use runtimelib::datatable::{FieldType, TableSchema, TableSchemaField};
use runtimelib::media::datatable::TabularDataResource;
use serde_json::Value;
use settings::Settings;
use theme_settings::ThemeSettings;
use ui::{IntoElement, Styled, div, prelude::*, v_flex};
use util::markdown::MarkdownEscaped;

use crate::outputs::OutputContent;

/// TableView renders a static table inline in a buffer.
///
/// It uses the <https://specs.frictionlessdata.io/tabular-data-resource/>
/// specification for data interchange.
pub struct TableView {
    pub table: TabularDataResource,
    pub widths: Vec<Pixels>,
    cached_clipboard_content: ClipboardItem,
}

fn cell_content(row: &Value, field: &str) -> String {
    match row.get(field) {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Number(n)) => n.to_string(),
        Some(Value::Bool(b)) => b.to_string(),
        Some(Value::Array(arr)) => format!("{:?}", arr),
        Some(Value::Object(obj)) => format!("{:?}", obj),
        Some(Value::Null) | None => String::new(),
    }
}

// Declare constant for the padding multiple on the line height
const TABLE_Y_PADDING_MULTIPLE: f32 = 0.5;

/// Cap on rendered rows so a huge DataFrame doesn't create tens of thousands
/// of elements. The clipboard content (Copy Output) still contains all rows.
const MAX_RENDERED_ROWS: usize = 300;

/// Try to interpret markdown (produced by `html_to_markdown` from an HTML
/// output) as a single table, so DataFrame-style HTML outputs (e.g. pandas'
/// default `text/html` repr) can be rendered with the native `TableView` grid
/// instead of as markdown text. Returns `None` when the markdown is not
/// essentially just one table — callers should fall back to markdown
/// rendering.
pub fn table_from_markdown(markdown: &str) -> Option<TabularDataResource> {
    let mut table_rows: Vec<Vec<String>> = Vec::new();

    for line in markdown.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        if trimmed.starts_with('|') {
            let cells: Vec<String> = trimmed
                .trim_matches('|')
                .split('|')
                .map(|cell| cell.trim().to_string())
                .collect();
            // Skip markdown separator rows (| --- | --- |).
            let is_separator = cells
                .iter()
                .all(|cell| !cell.is_empty() && cell.trim_matches(':').chars().all(|c| c == '-'));
            if !is_separator {
                table_rows.push(cells);
            }
        } else {
            // Tolerate pandas' trailing "N rows × M columns" summary line;
            // anything else means this output is more than a table.
            let is_shape_summary = trimmed.contains("rows") && trimmed.contains("columns");
            if !is_shape_summary {
                return None;
            }
        }
    }

    let (header, data_rows) = table_rows.split_first()?;
    if header.is_empty() {
        return None;
    }

    // Column names must be unique to key the row objects (pandas' index column
    // has an empty header; duplicates get a numeric suffix).
    let mut names: Vec<String> = Vec::with_capacity(header.len());
    for (index, name) in header.iter().enumerate() {
        let base = if name.is_empty() {
            // Most likely the DataFrame index column. The field name is also
            // the header label, so use (unique) whitespace to display blank.
            " ".repeat(index + 1)
        } else {
            name.clone()
        };
        let mut candidate = base.clone();
        let mut suffix = 2;
        while names.contains(&candidate) {
            candidate = format!("{base} ({suffix})");
            suffix += 1;
        }
        names.push(candidate);
    }

    // A column is numeric (right-aligned) when every non-empty cell parses as
    // a number, ignoring pandas' "..." truncation markers.
    let fields = names
        .iter()
        .enumerate()
        .map(|(column, name)| {
            let mut any_value = false;
            let numeric = data_rows.iter().all(|row| {
                let value = row.get(column).map(String::as_str).unwrap_or("");
                if value.is_empty() || value == "..." || value == "…" {
                    return true;
                }
                any_value = true;
                value.replace(',', "").parse::<f64>().is_ok()
            });
            TableSchemaField {
                name: name.clone(),
                field_type: if numeric && any_value {
                    FieldType::Number
                } else {
                    FieldType::String
                },
                ..Default::default()
            }
        })
        .collect();

    let data = data_rows
        .iter()
        .map(|row| {
            let mut object = serde_json::Map::new();
            for (column, name) in names.iter().enumerate() {
                let value = row.get(column).cloned().unwrap_or_default();
                object.insert(name.clone(), Value::String(value));
            }
            Value::Object(object)
        })
        .collect();

    Some(TabularDataResource {
        schema: TableSchema {
            fields,
            ..Default::default()
        },
        data: Some(data),
        ..Default::default()
    })
}

impl TableView {
    pub fn new(table: &TabularDataResource, window: &mut Window, cx: &mut App) -> Self {
        let mut widths = Vec::with_capacity(table.schema.fields.len());

        let text_system = window.text_system();
        let text_style = window.text_style();
        let text_font = ThemeSettings::get_global(cx).buffer_font.clone();
        let font_size = ThemeSettings::get_global(cx).buffer_font_size(cx);
        let mut runs = [TextRun {
            len: 0,
            font: text_font,
            color: text_style.color,
            ..Default::default()
        }];

        for field in table.schema.fields.iter() {
            runs[0].len = field.name.len();
            let mut width = text_system
                .layout_line(&field.name, font_size, &runs, None)
                .width;

            let Some(data) = table.data.as_ref() else {
                widths.push(width);
                continue;
            };

            for row in data {
                let content = cell_content(row, &field.name);
                runs[0].len = content.len();
                let cell_width = window
                    .text_system()
                    .layout_line(&content, font_size, &runs, None)
                    .width;

                width = width.max(cell_width)
            }

            widths.push(width)
        }

        let cached_clipboard_content = Self::create_clipboard_content(table);

        Self {
            table: table.clone(),
            widths,
            cached_clipboard_content: ClipboardItem::new_string(cached_clipboard_content),
        }
    }

    fn create_clipboard_content(table: &TabularDataResource) -> String {
        let data = match table.data.as_ref() {
            Some(data) => data,
            None => &Vec::new(),
        };
        let schema = table.schema.clone();

        let mut markdown = format!(
            "| {} |\n",
            table
                .schema
                .fields
                .iter()
                .map(|field| field.name.clone())
                .collect::<Vec<_>>()
                .join(" | ")
        );

        markdown.push_str("|---");
        for _ in 1..table.schema.fields.len() {
            markdown.push_str("|---");
        }
        markdown.push_str("|\n");

        let body = data
            .iter()
            .map(|record: &Value| {
                let row_content = schema
                    .fields
                    .iter()
                    .map(|field| MarkdownEscaped(&cell_content(record, &field.name)).to_string())
                    .collect::<Vec<_>>();

                row_content.join(" | ")
            })
            .collect::<Vec<String>>();

        for row in body {
            markdown.push_str(&format!("| {} |\n", row));
        }

        markdown
    }

    pub fn render_row(
        &self,
        schema: &TableSchema,
        is_header: bool,
        striped: bool,
        row: &Value,
        window: &mut Window,
        cx: &mut App,
    ) -> AnyElement {
        let theme = cx.theme();

        let line_height = window.line_height();

        let row_cells = schema
            .fields
            .iter()
            .zip(self.widths.iter())
            .map(|(field, width)| {
                let container = match field.field_type {
                    runtimelib::datatable::FieldType::String => div(),

                    runtimelib::datatable::FieldType::Number
                    | runtimelib::datatable::FieldType::Integer
                    | runtimelib::datatable::FieldType::Date
                    | runtimelib::datatable::FieldType::Time
                    | runtimelib::datatable::FieldType::Datetime
                    | runtimelib::datatable::FieldType::Year
                    | runtimelib::datatable::FieldType::Duration
                    | runtimelib::datatable::FieldType::Yearmonth => v_flex().items_end(),

                    _ => div(),
                };

                let is_null = !is_header && matches!(row.get(&field.name), Some(Value::Null) | None);
                let value = cell_content(row, &field.name);

                let cell = container
                    .min_w(*width + px(22.))
                    .w(*width + px(22.))
                    .px_2()
                    .py((TABLE_Y_PADDING_MULTIPLE / 2.0) * line_height);

                if is_header {
                    cell.font_weight(FontWeight::SEMIBOLD).child(value)
                } else if is_null {
                    // Render missing values as a dimmed placeholder instead of
                    // an empty cell.
                    cell.text_color(theme.colors().text_muted).child("—")
                } else {
                    cell.child(value)
                }
            })
            .collect::<Vec<_>>();

        let mut total_width = px(0.);
        for width in self.widths.iter() {
            // Width fudge factor: border + 2 (heading), padding
            total_width += *width + px(22.);
        }

        let row_element = h_flex().w(total_width).children(row_cells);

        if is_header {
            row_element
                .bg(theme.colors().element_background)
                .border_b_1()
                .border_color(theme.colors().border)
        } else {
            row_element
                .border_b_1()
                .border_color(theme.colors().border_variant)
                .when(striped, |this| {
                    this.bg(theme.colors().element_background.opacity(0.35))
                })
        }
        .into_any_element()
    }
}

impl Render for TableView {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let data = match &self.table.data {
            Some(data) => data,
            None => return div().into_any_element(),
        };

        let mut headings = serde_json::Map::new();
        for field in &self.table.schema.fields {
            headings.insert(field.name.clone(), Value::String(field.name.clone()));
        }
        let header = self.render_row(
            &self.table.schema,
            true,
            false,
            &Value::Object(headings),
            window,
            cx,
        );

        let row_count = data.len();
        let body: Vec<AnyElement> = data
            .iter()
            .take(MAX_RENDERED_ROWS)
            .enumerate()
            .map(|(index, row)| {
                self.render_row(&self.table.schema, false, index % 2 == 1, row, window, cx)
            })
            .collect();

        v_flex()
            .id("table")
            .overflow_x_scroll()
            .w_full()
            .child(
                v_flex()
                    .rounded_md()
                    .border_1()
                    .border_color(cx.theme().colors().border)
                    .overflow_hidden()
                    .child(header)
                    .children(body),
            )
            .when(row_count > MAX_RENDERED_ROWS, |this| {
                this.child(
                    div()
                        .px_2()
                        .py_1()
                        .text_xs()
                        .text_color(cx.theme().colors().text_muted)
                        .child(format!(
                            "Showing first {MAX_RENDERED_ROWS} of {row_count} rows"
                        )),
                )
            })
            .into_any_element()
    }
}

impl OutputContent for TableView {
    fn clipboard_content(&self, _window: &Window, _cx: &App) -> Option<ClipboardItem> {
        Some(self.cached_clipboard_content.clone())
    }

    fn has_clipboard_content(&self, _window: &Window, _cx: &App) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The shape produced by html_to_markdown for a pandas DataFrame repr:
    /// an index column with an empty header, and a trailing shape summary.
    #[test]
    fn test_table_from_markdown_pandas_dataframe() {
        let markdown = "\
|  | name | age |
| --- | --- | --- |
| 0 | Alice | 30 |
| 1 | Bob | 28 |
5 rows × 2 columns";

        let table = table_from_markdown(markdown).expect("should parse as a table");
        assert_eq!(table.schema.fields.len(), 3);
        // Index column header displays blank but is a unique field name.
        assert!(table.schema.fields[0].name.trim().is_empty());
        assert_eq!(table.schema.fields[1].name, "name");
        // The index and age columns are numeric (right-aligned); name is not.
        assert_eq!(table.schema.fields[0].field_type, FieldType::Number);
        assert_eq!(table.schema.fields[1].field_type, FieldType::String);
        assert_eq!(table.schema.fields[2].field_type, FieldType::Number);

        let data = table.data.expect("table has data");
        assert_eq!(data.len(), 2);
        assert_eq!(data[0].get("name"), Some(&Value::String("Alice".into())));
    }

    #[test]
    fn test_table_from_markdown_truncation_markers_stay_numeric() {
        let markdown = "\
| a |
| --- |
| 1 |
| ... |
| 3 |";
        let table = table_from_markdown(markdown).expect("should parse");
        assert_eq!(table.schema.fields[0].field_type, FieldType::Number);
    }

    #[test]
    fn test_table_from_markdown_rejects_mixed_content() {
        let markdown = "\
# A heading

| a | b |
| --- | --- |
| 1 | 2 |";
        assert!(
            table_from_markdown(markdown).is_none(),
            "markdown with non-table content should fall back to markdown rendering"
        );
    }

    #[test]
    fn test_table_from_markdown_duplicate_headers() {
        let markdown = "\
| x | x |
| --- | --- |
| 1 | 2 |";
        let table = table_from_markdown(markdown).expect("should parse");
        assert_eq!(table.schema.fields[0].name, "x");
        assert_ne!(table.schema.fields[1].name, "x");
        let data = table.data.expect("has data");
        assert_eq!(data[0].as_object().map(|obj| obj.len()), Some(2));
    }
}
