"""Convert FinQA tables to markdown and linearized text formats."""

from typing import Optional


def table_to_markdown(table: list[list[str]]) -> str:
    """Convert a list-of-lists table to markdown format.

    Args:
        table: List of rows, where each row is a list of cell strings.
               First row is treated as the header.

    Returns:
        Markdown-formatted table string.
    """
    if not table or not table[0]:
        return ""

    # Clean cells
    cleaned = []
    for row in table:
        cleaned.append([cell.strip() if cell else "" for cell in row])

    # Determine column widths for alignment
    num_cols = max(len(row) for row in cleaned)

    # Pad rows to uniform width
    for row in cleaned:
        while len(row) < num_cols:
            row.append("")

    # Build markdown
    lines = []

    # Header
    header = cleaned[0]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * num_cols) + " |")

    # Data rows
    for row in cleaned[1:]:
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


def table_to_linearized(table: list[list[str]]) -> str:
    """Convert a table to linearized natural language text.

    This format is better for embedding similarity matching because
    it converts structured data into natural language sentences.

    Example output:
        "The revenue of 2008 is $920M; The revenue of 2007 is $825M;"

    Args:
        table: List of rows (header first).

    Returns:
        Linearized text representation.
    """
    if not table or len(table) < 2:
        return ""

    header = [cell.strip() if cell else "" for cell in table[0]]
    sentences = []

    for row in table[1:]:
        cells = [cell.strip() if cell else "" for cell in row]
        row_label = cells[0] if cells else ""

        for col_idx in range(1, min(len(header), len(cells))):
            col_name = header[col_idx]
            value = cells[col_idx]
            if value and col_name:
                sentences.append(f"The {col_name} of {row_label} is {value}")

    return "; ".join(sentences) + "." if sentences else ""


def extract_cell(
    table: list[list[str]],
    row_query: str,
    col_query: str
) -> Optional[str]:
    """Extract a specific cell value from the table.

    Args:
        table: List of rows (header first).
        row_query: Substring to match in the first column (row header).
        col_query: Substring to match in the header row.

    Returns:
        Cell value string, or None if not found.
    """
    if not table or len(table) < 2:
        return None

    header = [cell.strip().lower() for cell in table[0]]

    # Find matching column
    col_idx = None
    for idx, col in enumerate(header):
        if col_query.lower() in col:
            col_idx = idx
            break

    if col_idx is None:
        return None

    # Find matching row
    row_query_lower = row_query.lower()
    for row in table[1:]:
        if row and row_query_lower in row[0].strip().lower():
            if col_idx < len(row):
                return row[col_idx].strip()

    return None
