"""The static-tables-only carve-out's second edge: an order the query does
not fix.

goal.md exclusion: whole-relation-shapes carves static-tables-only queries out
of the whole-relation refusals -- the query is evaluated once by DuckDB at
build and frozen -- and then applies one rule to the carve-out: **what a
whole-relation construct selects is frozen only when it is a function of the
query.** Three readings follow from it, and all three are measured (the
measurement is in the commit that introduced this file: one static table of
60k rows fed through a tying GROUP BY, the same statement under five DuckDB
settings a build machine picks for itself -- default, threads=1/2/8,
preserve_insertion_order=false):

1. A top-level ORDER BY that leaves two rows tied does not fix their order:
   the frozen SEQUENCE came out five different ways. REFUSES.
2. An ORDER BY that is NOT at the top orders nothing that anyone is promised.
   Row order on this path is not part of the contract at all -- the
   differential compares static-only results as an unordered multiset
   (fuzz/oracle.py compare_mode, `constant-unordered`) and the oracle spec
   says so (docs/oracle/03-nondeterminism.md). Measured, an inner ORDER BY
   moved the sequence and left the SET identical under all five settings. So
   it changes nothing that is promised, and it SERVES.
3. Selection BY POSITION anywhere in the statement does move the set, and no
   ORDER BY rescues it: a LIMIT in a derived table answered four ways, in a
   CTE five, DISTINCT ON five, QUALIFY over row_number() five, row_number()
   over tied keys five, and USING SAMPLE differently on all twelve of twelve
   fresh connections. All REFUSE.

The ORDER BY question is therefore asked of the RESULT, not of the clause: an
ORDER BY serves when its keys separate every row, and refuses when they do
not. Ties are read off the KEYS, so two rows that tie on the keys refuse even
where they carry equal values everywhere else -- "a tie" is a statement about
the sort keys, which is the reading that needs no argument about which
duplicate row is which.

Zero-row and one-row results have no ties. NULL and NaN are ordinary
tie-capable values: DuckDB gives each of them a place in the sort order and
groups them with their own kind, and so does this check.

The shapes above are read off DuckDB's OWN parse of the statement
(`json_serialize_sql`), because the carve-out exists FOR the DuckDB dialect
that this crate's parser cannot read. Every refusal below is therefore also a
pin on that reading: if DuckDB's serialization changes, these go red rather
than going quiet.
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
OUTSIDE_MSG = (
    "unsupported: a sort key its projection cannot carry on a "
    "static-tables-only query -- a tie among its rows could not be ruled out, "
    "and which of two tied rows comes first depends on scan order, not the "
    "query"
)


def build(sql, table=TIES, name="s"):
    return DuckDBInferFn(sql, row_tables={"__THIS__": ROW}, static_tables={name: table})


def refuses(sql, table=TIES):
    with pytest.raises(ValueError) as e:
        build(sql, table)
    return " ".join(str(e.value).split())


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


def test_a_query_without_an_order_by_is_untouched():
    fn = build("SELECT g AS o, v AS t FROM s")
    assert fn.backend == "constant"
    assert len(fn.infer_rows([])) == 3


def test_an_order_by_below_the_top_serves_because_order_is_not_promised():
    # Reading 2 in the module docstring: row order on the constant path is not
    # part of the contract, and an inner ORDER BY changes nothing else. Its
    # keys tie here on purpose -- a tie that orders nothing is not a tie
    # anybody was promised the absence of.
    fn = build("SELECT o, t FROM (SELECT g AS o, v AS t FROM s ORDER BY v) q")
    assert fn.backend == "constant"
    assert sorted(r["o"] for r in fn.infer_rows([])) == ["x", "y", "z"]


def test_a_repeated_output_name_is_addressed_by_position_not_by_name():
    # Two output columns called x, and the key is the second. Keys are read
    # off the frozen result POSITIONALLY, so the name being taken twice costs
    # nothing: position 2 is g, which separates every row, and it serves.
    fn = build("SELECT v AS x, g AS x FROM s ORDER BY 2")
    assert fn.backend == "constant"
    assert len(fn.infer_rows([])) == 3


def test_a_repeated_output_name_that_ties_still_refuses():
    # The same shape with the key on the column that ties: reading position 2
    # as "the column called x" would have measured the first one instead.
    assert refuses("SELECT g AS x, v AS x FROM s ORDER BY 2") == TIE_MSG


def test_a_window_over_the_whole_partition_serves():
    # `OVER ()` with no order and no row-position function is an aggregate
    # over the whole partition: a set, so scan order cannot reach it.
    fn = build("SELECT g AS o, max(v) OVER () AS m FROM s ORDER BY g")
    assert fn.backend == "constant"
    assert [r["m"] for r in fn.infer_rows([])] == [2, 2, 2]


def test_plain_distinct_serves():
    # DISTINCT collapses a set; it does not pick a row out of a group.
    fn = build("SELECT DISTINCT v AS o FROM s ORDER BY o")
    assert fn.backend == "constant"
    assert [r["o"] for r in fn.infer_rows([])] == [1, 2]


# --------------------------------------------------- refuses: a tie in order --


def test_a_tie_refuses_by_name():
    assert refuses("SELECT g AS o, v AS t FROM s ORDER BY v") == TIE_MSG


def test_a_tie_between_rows_apart_in_scan_order_refuses():
    assert refuses("SELECT g AS o, v AS t FROM s ORDER BY v", APART) == TIE_MSG


def test_descending_ties_the_same_rows():
    assert refuses("SELECT g AS o, v AS t FROM s ORDER BY v DESC") == TIE_MSG


def test_a_tie_on_every_key_of_a_multi_key_sort_refuses():
    assert refuses("SELECT g AS o, v AS t FROM s ORDER BY v, v + 0") == TIE_MSG


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


# -------------------------------------------------------- ORDER BY ALL, read --


def test_order_by_all_is_every_output_column():
    # ALL sorts by every output column, so a tie under it is a repeated
    # output ROW. These three rows are distinct, and the query serves.
    fn = build("SELECT g AS o, v AS t FROM s ORDER BY ALL")
    assert fn.backend == "constant"
    assert [r["o"] for r in fn.infer_rows([])] == ["x", "y", "z"]


def test_order_by_all_over_a_repeated_row_refuses():
    dup = pa.table({"g": ["x", "x", "z"], "v": pa.array([1, 1, 2], pa.int64())})
    assert refuses("SELECT g AS o, v AS t FROM s ORDER BY ALL", dup) == TIE_MSG


def test_order_by_quoted_all_is_the_column_named_all():
    # `ORDER BY "all"` names a COLUMN. Here that column separates every row,
    # and reading it as sort-by-everything would answer about the wrong keys.
    fn = build('SELECT v AS "all", g AS o FROM s ORDER BY "all"', UNIQ)
    assert fn.backend == "constant"
    assert [r["all"] for r in fn.infer_rows([])] == [1, 2, 3]


def test_bare_all_that_is_also_an_output_column_is_still_every_column():
    # DuckDB reads a bare ALL as sort-by-every-output-column even when an
    # output column is called `all` -- measured. Every row is distinct on the
    # whole tuple, so it serves.
    fn = build('SELECT v AS "all", g AS o FROM s ORDER BY ALL')
    assert fn.backend == "constant"
    assert sorted(r["o"] for r in fn.infer_rows([])) == ["x", "y", "z"]


def test_a_column_named_all_that_ties_refuses_under_its_own_quoting():
    assert refuses('SELECT v AS "all", g AS o FROM s ORDER BY "all"') == TIE_MSG


# ------------------------------------ a key the frozen result does not carry --
#
# `ORDER BY v` where the output calls it `t`, `ORDER BY abs(v)`, `ORDER BY
# s.v`: none of these is an output column, so the key is added to the
# QUERY'S OWN projection -- by rewriting DuckDB's parse of the statement and
# asking DuckDB to print it back -- and measured beside the rows it orders.


def test_a_key_the_output_does_not_carry_is_computed_where_it_belongs():
    fn = build("SELECT g AS o FROM s ORDER BY abs(v)", UNIQ)
    assert fn.backend == "constant"
    assert [r["o"] for r in fn.infer_rows([])] == ["y", "z", "x"]


def test_a_key_the_output_does_not_carry_can_tie():
    # abs() collapses -1 and 1, which the column itself keeps apart.
    signed = pa.table({"g": ["x", "y", "z"], "v": pa.array([-1, 1, 2], pa.int64())})
    assert refuses("SELECT g AS o FROM s ORDER BY abs(v)", signed) == TIE_MSG


def test_a_qualified_key_is_computed_where_it_belongs():
    fn = build("SELECT g AS o FROM s ORDER BY s.v", UNIQ)
    assert fn.backend == "constant"
    assert [r["o"] for r in fn.infer_rows([])] == ["y", "z", "x"]


def test_a_mixed_output_and_computed_key_pair_is_measured_on_both():
    # One key off the frozen result, one added to the projection. The first
    # ties every row; the second separates them, so it serves.
    fn = build("SELECT g AS o, 1 AS c FROM s ORDER BY c, v", UNIQ)
    assert fn.backend == "constant"
    assert [r["o"] for r in fn.infer_rows([])] == ["y", "z", "x"]


def test_a_key_the_projection_cannot_carry_refuses():
    # An added column changes what DISTINCT collapses, and with it the rows
    # whose keys would be compared -- so the key cannot be measured there,
    # and an unruled-out tie is refused like a found one.
    assert refuses("SELECT DISTINCT g AS o FROM s ORDER BY abs(v)", UNIQ) == OUTSIDE_MSG


def test_a_set_operation_sorted_by_position_is_still_measured():
    # A set operation has no projection to grow, but a key that IS one of its
    # output columns never needs one.
    sql = "SELECT g AS o FROM s UNION ALL SELECT g AS o FROM s ORDER BY 1"
    assert refuses(sql, UNIQ) == TIE_MSG


# ------------------------------------------- refuses: selection by position --


def test_a_limit_in_a_derived_table_refuses():
    msg = refuses("SELECT o FROM (SELECT g AS o FROM s LIMIT 2) q")
    assert msg.startswith("unsupported: row limit (LIMIT/OFFSET)"), msg


def test_a_limit_in_a_cte_refuses():
    msg = refuses("WITH c AS (SELECT g AS o FROM s LIMIT 2) SELECT o FROM c")
    assert msg.startswith("unsupported: row limit (LIMIT/OFFSET)"), msg


def test_using_sample_refuses():
    msg = refuses("SELECT g AS o FROM s USING SAMPLE 2 ROWS")
    assert msg.startswith("unsupported: row limit (USING SAMPLE)"), msg


def test_distinct_on_refuses():
    msg = refuses("SELECT DISTINCT ON (v) g AS o, v AS t FROM s")
    assert msg.startswith("unsupported: DISTINCT ON on a static-tables-only"), msg


def test_qualify_refuses():
    msg = refuses("SELECT g AS o FROM s QUALIFY row_number() OVER (PARTITION BY v) = 1")
    assert msg.startswith("unsupported: QUALIFY on a static-tables-only"), msg


def test_a_row_position_window_function_refuses():
    msg = refuses("SELECT g AS o, row_number() OVER (ORDER BY v) AS r FROM s")
    assert msg.startswith("unsupported: a row-position window function"), msg


def test_lag_is_a_row_position_window_function():
    msg = refuses("SELECT g AS o, lag(v) OVER (ORDER BY v) AS r FROM s")
    assert msg.startswith("unsupported: a row-position window function"), msg


# ------------------------------------------------------ the DuckDB dialect --


def test_a_tie_under_from_first_syntax_refuses():
    # The carve-out exists FOR syntax this crate's parser cannot read. The
    # FROM-first form is one; a reading that could not see it would let a tie
    # through in exactly the place the carve-out is for.
    assert refuses("FROM s SELECT g AS o, v AS t ORDER BY t") == TIE_MSG


def test_a_unique_key_under_from_first_syntax_still_serves():
    fn = build("FROM s SELECT g AS o, v AS t ORDER BY t", UNIQ)
    assert fn.backend == "constant"
    assert [r["t"] for r in fn.infer_rows([])] == [1, 2, 3]


def test_a_statement_duckdb_will_not_expose_refuses_rather_than_serving():
    # PIVOT executes but does not serialize (measured), so its shape cannot be
    # inspected. It carries an ORDER BY, and an order that cannot be read is
    # an order whose ties cannot be ruled out -- so it refuses BY NAME instead
    # of quietly freezing whichever sequence this build produced.
    msg = refuses("PIVOT s ON g USING sum(v) GROUP BY v ORDER BY ALL")
    assert msg.startswith(
        "unsupported: ORDER BY in a statement DuckDB would not expose"
    ), msg


def test_select_top_is_still_a_row_limit_even_though_duckdb_rejects_it():
    # `SELECT TOP n` is not DuckDB syntax at all, so nothing can be read off
    # it; the token reading names it anyway rather than falling silent.
    msg = refuses("SELECT TOP 1 g AS o FROM s")
    assert msg.startswith("unsupported: row limit (SELECT TOP)"), msg
