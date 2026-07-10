"""Formatting helpers for financial record context."""

from __future__ import annotations

from src.models import ConvFinQARecord, TableValue


def format_table(table: dict[str, dict[str, TableValue]]) -> str:
    """Format the column-oriented dataset table as Markdown."""
    columns = list(table.keys())
    row_labels = _ordered_row_labels(table)

    header = ["metric", *columns]
    rows = [header, ["---", *["---"] * len(columns)]]

    for row_label in row_labels:
        row = [row_label]
        for column in columns:
            value = table[column].get(row_label, "")
            row.append(_format_value(value))
        rows.append(row)

    return "\n".join(_format_markdown_row(row) for row in rows)


def format_record_context(record: ConvFinQARecord) -> str:
    """Format a record into plain context for the LLM prompt."""
    return "\n\n".join(
        [
            f"Record ID: {record.id}",
            "Pre-table text:\n" + record.doc.pre_text,
            "Table:\n" + format_table(record.doc.table),
            "Post-table text:\n" + record.doc.post_text,
        ],
    )


def _ordered_row_labels(table: dict[str, dict[str, TableValue]]) -> list[str]:
    row_labels: list[str] = []
    seen: set[str] = set()
    for column_values in table.values():
        for row_label in column_values:
            if row_label not in seen:
                seen.add(row_label)
                row_labels.append(row_label)
    return row_labels


def _format_value(value: TableValue | str) -> str:
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def _format_markdown_row(values: list[str]) -> str:
    escaped_values = [value.replace("|", "\\|") for value in values]
    return "| " + " | ".join(escaped_values) + " |"
