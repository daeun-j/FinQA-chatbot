"""Deterministic calculator implementing the FinQA DSL.

The FinQA DSL supports operations:
  add(a, b), subtract(a, b), multiply(a, b), divide(a, b),
  exp(a, b), greater(a, b),
  table_sum(column), table_average(column), table_max(column), table_min(column)

References:
  #0, #1, ... refer to results of previous steps
  const_X refers to constant X (e.g., const_100 = 100, const_m1 = -1)
"""

import re
from typing import Optional
try:
    from langchain_core.tools import tool
except ImportError:
    def tool(func): return func


# Parse constants: const_1, const_100, const_1000, const_m1, etc.
CONST_PATTERN = re.compile(r"const_m?(\d+)")


def parse_number(token: str, variables: dict[str, float]) -> Optional[float]:
    """Parse a token as a number, variable reference, or constant.

    Args:
        token: String like "920", "#0", "const_100", etc.
        variables: Dict of variable name -> value (e.g., {"#0": 825.0}).

    Returns:
        Float value, or None if parsing fails.
    """
    token = token.strip()

    # Variable reference: #0, #1, ...
    if token.startswith("#"):
        return variables.get(token)

    # Constant: const_100, const_m1, etc.
    if token.startswith("const_"):
        if token == "const_m1":
            return -1.0
        match = CONST_PATTERN.match(token)
        if match:
            return float(match.group(1))
        return None

    # Try parsing as a plain number
    # Handle percentages and commas
    cleaned = token.replace(",", "").replace("%", "")
    try:
        return float(cleaned)
    except ValueError:
        return None


def execute_operation(
    op: str, args: list[str], variables: dict[str, float],
    table: Optional[list[list[str]]] = None,
) -> Optional[float]:
    """Execute a single FinQA DSL operation.

    Args:
        op: Operation name (add, subtract, multiply, divide, etc.)
        args: List of argument strings.
        variables: Current variable bindings.
        table: Optional table for table_* operations.

    Returns:
        Result as float, or None on error.
    """
    if op in ("add", "subtract", "multiply", "divide", "exp", "greater"):
        if len(args) != 2:
            return None
        a = parse_number(args[0], variables)
        b = parse_number(args[1], variables)
        if a is None or b is None:
            return None

        if op == "add":
            return a + b
        elif op == "subtract":
            return a - b
        elif op == "multiply":
            return a * b
        elif op == "divide":
            if b == 0:
                return None
            return a / b
        elif op == "exp":
            return a ** b
        elif op == "greater":
            return 1.0 if a > b else 0.0

    # Table aggregation operations
    if op in ("table_sum", "table_average", "table_max", "table_min"):
        if table is None or len(args) != 1:
            return None
        return _table_aggregate(op, args[0], table)

    return None


def _table_aggregate(
    op: str, col_query: str, table: list[list[str]]
) -> Optional[float]:
    """Perform aggregation over a table column.

    Args:
        op: One of table_sum, table_average, table_max, table_min.
        col_query: Column name or index to aggregate.
        table: Table as list of lists.

    Returns:
        Aggregated result, or None on error.
    """
    if not table or len(table) < 2:
        return None

    header = [cell.strip().lower() for cell in table[0]]

    # Find column by name or index
    col_idx = None
    try:
        col_idx = int(col_query)
    except ValueError:
        col_query_lower = col_query.strip().lower()
        for idx, col in enumerate(header):
            if col_query_lower in col:
                col_idx = idx
                break

    if col_idx is None or col_idx >= len(header):
        return None

    # Extract numeric values from the column
    values = []
    for row in table[1:]:
        if col_idx < len(row):
            cell = row[col_idx].strip().replace(",", "").replace("%", "")
            # Strip parentheses (accounting negative notation)
            if cell.startswith("(") and cell.endswith(")"):
                cell = "-" + cell[1:-1]
            try:
                values.append(float(cell))
            except ValueError:
                continue

    if not values:
        return None

    if op == "table_sum":
        return sum(values)
    elif op == "table_average":
        return sum(values) / len(values)
    elif op == "table_max":
        return max(values)
    elif op == "table_min":
        return min(values)

    return None


# Pattern to match: operation(arg1, arg2)
STEP_PATTERN = re.compile(r"(\w+)\(([^)]*)\)")


def execute_program(
    program: str,
    table: Optional[list[list[str]]] = None,
) -> tuple[Optional[float], list[dict]]:
    """Execute a full FinQA DSL program (sequence of operations).

    Args:
        program: Program string like "subtract(920, 95), divide(#0, 95)"
        table: Optional table for table_* operations.

    Returns:
        Tuple of (final_result, execution_trace).
        execution_trace is a list of dicts with step details.
    """
    variables: dict[str, float] = {}
    trace: list[dict] = []

    # Split into individual steps
    steps = STEP_PATTERN.findall(program)

    for step_idx, (op, args_str) in enumerate(steps):
        # Parse arguments (comma-separated)
        args = [a.strip() for a in args_str.split(",") if a.strip()]

        result = execute_operation(op, args, variables, table)

        var_name = f"#{step_idx}"
        step_info = {
            "step": step_idx,
            "operation": op,
            "args": args,
            "result": result,
            "variable": var_name,
        }
        trace.append(step_info)

        if result is not None:
            variables[var_name] = result
        else:
            # Execution failed at this step
            return None, trace

    # Final result is the last step's result
    if trace:
        return trace[-1]["result"], trace
    return None, trace


@tool
def calculate(expression: str) -> str:
    """Execute a mathematical expression using the FinQA DSL.

    Supported operations:
    - add(a, b): addition
    - subtract(a, b): subtraction
    - multiply(a, b): multiplication
    - divide(a, b): division
    - exp(a, b): exponentiation
    - greater(a, b): returns 1 if a > b, else 0

    Use #0, #1, ... to reference results of previous steps.
    Use const_100, const_1000, etc. for constants.

    Example: "subtract(920, 825), divide(#0, 825), multiply(#0, const_100)"
    This computes: (920-825)/825 * 100 = 11.52%

    Args:
        expression: A FinQA DSL program string.

    Returns:
        String describing the computation result.
    """
    result, trace = execute_program(expression)

    if result is None:
        return f"Error: Could not execute '{expression}'. Trace: {trace}"

    # Format trace for the LLM to see
    trace_str = "\n".join(
        f"  {s['variable']} = {s['operation']}({', '.join(s['args'])}) = {s['result']}"
        for s in trace
    )
    return f"Result: {result}\n\nExecution trace:\n{trace_str}"
