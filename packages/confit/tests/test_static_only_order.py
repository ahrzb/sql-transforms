"""The static-tables-only carve-out's second edge: a tie-producing ORDER BY.

goal.md exclusion: whole-relation-shapes carves static-tables-only queries out
of the whole-relation refusals -- the query is evaluated once by DuckDB at
build and frozen -- and then applies one rule to the carve-out: **what a
whole-relation construct selects is frozen only when it is a function of the
query.** A row limit is not, and refuses (test_arrow_schema_api.py). Neither
is an ORDER BY that leaves two rows tied: the query says nothing about which
of them comes first, so freezing whichever sequence this build's DuckDB run
produced would let two builds of the same function disagree.

So the ORDER BY question is asked of the RESULT, not of the clause: an
ORDER BY serves when its keys separate every row, and refuses when they do
not. The keys are what ties are measured on, so two rows that tie on the
keys refuse even where they carry equal values everywhere else -- "a tie" is
a statement about the sort keys, which is the reading that needs no argument
about which duplicate row is which.

Zero-row and one-row results have no ties. NULL and NaN are ordinary
tie-capable values: DuckDB gives each of them a place in the sort order and
groups them with their own kind, and so does this check.
"""

from __future__ import annotations

import pyarrow as pa
import pytest
from confit import DuckDBInferFn, compare

ROW = pa.schema([pa.field("k", pa.int64(), nullable=False)])

# g is unique; v ties x with y and leaves z alone.
TIES = pa.table({"g": ["x", "y", "z"], "v": pa.array([1, 1, 2], pa.int64())})
# The same tie, with the tied rows split apart in scan order.
APART = pa.table({"g": ["x", "y", "z"], "v": pa.array([1, 2, 1], pa.int64())})
# Nothing ties, on any single column.
UNIQ = pa.table({"g": ["x", "y", "z"], "v": pa.array([3, 1, 2], pa.int64())})

TIE_MSG = (
    "unsupported: tie-producing ORDER BY on a static-tables-only query -- "
    "which of the tied rows comes first depends on scan order, not the query"
)


def build(sql, table=TIES, name="s"):
    return DuckDBInferFn(sql, row_tables={"__THIS__": ROW}, static_tables={name: table})


def refuses(sql, table=TIES):
    with pytest.raises(ValueError) as e:
        build(sql, table)
    return str(e.value)


# ------------------------------------------------------------- still serves --


def test_a_sort_key_that_separates_every_row_serves_as_before(oracle):
    sql = "SELECT g AS o, v AS t FROM s ORDER BY v"
    fn = build(sql, UNIQ)
    assert fn.backend == "constant"
    oracle.load("s", UNIQ)
    compare.assert_rows(
        fn.infer_rows([]), compare.rows(oracle.answer(sql)), ordered=True
    )
    # and the frozen order is the one the ORDER BY asked for
    assert [r["t"] for r in fn.infer_rows([])] == [1, 2, 3]


def test_a_second_key_that_breaks_the_tie_serves():
    fn = build("SELECT g AS o, v AS t FROM s ORDER BY v, g")
    assert fn.backend == "constant"
    assert [r["o"] for r in fn.infer_rows([])] == ["x", "y", "z"]


def test_a_zero_row_result_has_no_ties():
    fn = build("SELECT g AS o FROM s WHERE v > 99 ORDER BY g")
    assert fn.backend == "constant"
    assert fn.infer_rows([]) == []


def test_a_one_row_result_has_no_ties():
    fn = build("SELECT max(v) AS o FROM s ORDER BY o")
    assert fn.backend == "constant"
    assert fn.infer_rows([]) == [{"o": 2}]


def test_an_expression_key_that_separates_every_row_serves():
    # The key is not in the output at all: it is computed where the ORDER BY
    # computes it, over the query's own input.
    fn = build("SELECT g AS o FROM s ORDER BY v * -1", UNIQ)
    assert fn.backend == "constant"
    assert [r["o"] for r in fn.infer_rows([])] == ["x", "z", "y"]


def test_a_query_without_an_order_by_is_untouched():
    fn = build("SELECT g AS o, v AS t FROM s")
    assert fn.backend == "constant"
    assert len(fn.infer_rows([])) == 3


# ------------------------------------------------------------------ refuses --


def test_a_tie_refuses_by_name():
    assert refuses("SELECT g AS o, v AS t FROM s ORDER BY v") == TIE_MSG


def test_a_tie_between_rows_apart_in_scan_order_refuses():
    assert refuses("SELECT g AS o, v AS t FROM s ORDER BY v", APART) == TIE_MSG


def test_descending_ties_the_same_rows():
    assert refuses("SELECT g AS o, v AS t FROM s ORDER BY v DESC") == TIE_MSG


def test_a_tie_on_every_key_of_a_multi_key_sort_refuses():
    assert refuses("SELECT g AS o, v AS t FROM s ORDER BY v, v + 1") == TIE_MSG


def test_an_expression_key_can_tie():
    # abs() collapses -1 and 1, which the column itself keeps apart.
    signed = pa.table({"g": ["x", "y", "z"], "v": pa.array([-1, 1, 2], pa.int64())})
    assert refuses("SELECT g AS o FROM s ORDER BY abs(v)", signed) == TIE_MSG


def test_an_alias_key_can_tie():
    # ORDER BY resolves `t` to the OUTPUT alias, not to any column of s.
    assert refuses("SELECT g AS o, v + 0 AS t FROM s ORDER BY t") == TIE_MSG


def test_a_positional_key_can_tie():
    assert refuses("SELECT g AS o, v AS t FROM s ORDER BY 2") == TIE_MSG


def test_the_goal_documents_own_grouped_example_refuses():
    # goal.md exclusion: whole-relation-shapes, the executed example: the
    # groups x and y both total 1, and the query does not order them.
    assert refuses("SELECT g AS o, sum(v) AS t FROM s GROUP BY g ORDER BY t") == TIE_MSG


def test_null_is_a_tie_capable_value():
    nulls = pa.table({"g": ["x", "y", "z"], "v": pa.array([None, None, 2], pa.int64())})
    assert refuses("SELECT g AS o, v AS t FROM s ORDER BY v", nulls) == TIE_MSG


def test_nan_is_a_tie_capable_value():
    nan = float("nan")
    nans = pa.table({"g": ["x", "y", "z"], "v": pa.array([nan, nan, 2.0])})
    assert refuses("SELECT g AS o, v AS t FROM s ORDER BY v", nans) == TIE_MSG


def test_rows_that_tie_on_the_key_refuse_even_when_they_are_equal_rows():
    # The check is on the SORT KEYS. These two output rows are identical, so
    # no build could disagree with another about them -- but "tie" names the
    # keys, and the conservative reading is the one that needs no argument
    # about which of two equal rows is which.
    assert refuses("SELECT v AS o FROM s ORDER BY v") == TIE_MSG
