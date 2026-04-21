"""Tests for the FinQA DSL calculator."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tools.calculator import execute_program, parse_number, execute_operation


def test_parse_number():
    variables = {"#0": 95.0, "#1": 10.0}
    assert parse_number("920", variables) == 920.0
    assert parse_number("#0", variables) == 95.0
    assert parse_number("const_100", variables) == 100.0
    assert parse_number("const_m1", variables) == -1.0
    assert parse_number("const_1000", variables) == 1000.0
    assert parse_number("1,234", variables) == 1234.0
    print("[PASS] test_parse_number")


def test_single_operations():
    variables = {}
    assert execute_operation("add", ["10", "20"], variables) == 30.0
    assert execute_operation("subtract", ["920", "825"], variables) == 95.0
    assert execute_operation("multiply", ["10", "5"], variables) == 50.0
    assert execute_operation("divide", ["100", "4"], variables) == 25.0
    assert execute_operation("greater", ["10", "5"], variables) == 1.0
    assert execute_operation("greater", ["5", "10"], variables) == 0.0
    print("[PASS] test_single_operations")


def test_multi_step_program():
    # (920 - 825) / 825 * 100 = 11.515...
    # #0 = subtract(920,825) = 95
    # #1 = divide(#0, 825) = 0.11515...
    # #2 = multiply(#1, const_100) = 11.515...
    program = "subtract(920, 825), divide(#0, 825), multiply(#1, const_100)"
    result, trace = execute_program(program)
    assert result is not None
    assert abs(result - 11.515151515) < 0.01
    assert len(trace) == 3
    print(f"[PASS] test_multi_step_program: result={result:.4f}")


def test_percentage_change():
    program = "subtract(450, 380), divide(#0, 380), multiply(#1, const_100)"
    result, trace = execute_program(program)
    assert result is not None
    expected = (450 - 380) / 380 * 100  # 18.42%
    assert abs(result - expected) < 0.01
    print(f"[PASS] test_percentage_change: result={result:.4f}")


def test_division_by_zero():
    result, trace = execute_program("divide(100, 0)")
    assert result is None
    print("[PASS] test_division_by_zero")


def test_table_sum():
    table = [
        ["segment", "revenue"],
        ["north america", "3200"],
        ["europe", "1500"],
        ["asia", "534"],
    ]
    result, trace = execute_program("table_sum(1)", table=table)
    assert result == 5234.0
    print(f"[PASS] test_table_sum: result={result}")


def test_constants():
    program = "divide(500, const_100)"
    result, _ = execute_program(program)
    assert result == 5.0
    print("[PASS] test_constants")


def test_simple_divide():
    program = "divide(450, 100)"
    result, _ = execute_program(program)
    assert result == 4.5
    print("[PASS] test_simple_divide")


if __name__ == "__main__":
    test_parse_number()
    test_single_operations()
    test_multi_step_program()
    test_percentage_change()
    test_division_by_zero()
    test_table_sum()
    test_constants()
    test_simple_divide()
    print("\nAll calculator tests passed!")
