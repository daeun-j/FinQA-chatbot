"""Tool for extracting specific values from financial tables."""

from langchain_core.tools import tool
from src.data_processing.table_parser import extract_cell


@tool
def lookup_table_value(row_name: str, column_name: str, table_json: str) -> str:
    """Look up a specific cell value from a financial table.

    Args:
        row_name: Name/label of the row (searches first column).
        column_name: Name of the column header.
        table_json: JSON string of the table (list of lists).

    Returns:
        The cell value, or an error message if not found.
    """
    import json
    try:
        table = json.loads(table_json)
    except json.JSONDecodeError:
        return "Error: Could not parse table JSON."

    result = extract_cell(table, row_name, column_name)
    if result is None:
        return f"Not found: row containing '{row_name}', column containing '{column_name}'"
    return f"Value: {result}"
