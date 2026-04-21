"""Tests for table parser."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_processing.table_parser import table_to_markdown, table_to_linearized, extract_cell


def test_table_to_markdown():
    table = [
        ["", "2008", "2007", "2006"],
        ["revenue", "$920", "$825", "$780"],
        ["operating income", "$150", "$130", "$110"],
    ]
    md = table_to_markdown(table)
    assert "| revenue | $920 | $825 | $780 |" in md
    assert "| --- |" in md
    print("[PASS] test_table_to_markdown")


def test_table_to_linearized():
    table = [
        ["", "2008", "2007"],
        ["revenue", "$920", "$825"],
        ["cost", "$500", "$450"],
    ]
    lin = table_to_linearized(table)
    assert "The 2008 of revenue is $920" in lin
    assert "The 2007 of cost is $450" in lin
    print("[PASS] test_table_to_linearized")


def test_extract_cell():
    table = [
        ["item", "2008", "2007"],
        ["revenue", "920", "825"],
        ["cost", "500", "450"],
    ]
    assert extract_cell(table, "revenue", "2008") == "920"
    assert extract_cell(table, "cost", "2007") == "450"
    assert extract_cell(table, "nonexistent", "2008") is None
    print("[PASS] test_extract_cell")


def test_empty_table():
    assert table_to_markdown([]) == ""
    assert table_to_linearized([]) == ""
    assert extract_cell([], "a", "b") is None
    print("[PASS] test_empty_table")


if __name__ == "__main__":
    test_table_to_markdown()
    test_table_to_linearized()
    test_extract_cell()
    test_empty_table()
    print("\nAll table parser tests passed!")
