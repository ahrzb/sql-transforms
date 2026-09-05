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
    # groups x and y both answer 1, and the query does not order them.
    assert refuses("SELECT g AS o, min(v) AS t FROM s GROUP BY g ORDER BY t") == TIE_MSG


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


def test_select_top_is_still_named_as_a_row_limit():
    # `SELECT TOP n` is not DuckDB syntax at all, so DuckDB never runs it and
    # this query is never a static-tables-only one: the refusal that reaches
    # the caller is the row path's, which names the same clause without
    # claiming a path the query never took.
    msg = refuses("SELECT TOP 1 g AS o FROM s")
    assert msg == "unsupported: SELECT TOP (row limit)", msg


# ------------------------------------------- a draw is not part of the query --


def test_a_volatile_function_as_a_sort_key_refuses():
    # The tie probe re-evaluates its keys, so a fresh draw never collides with
    # the one that was frozen: eight builds of this statement over the same
    # table produced four different sequences. The draw is not in the query,
    # so the function is refused by name instead of measured.
    msg = refuses("SELECT g AS o, v AS t FROM s ORDER BY random()")
    assert msg == (
        "unsupported: the non-deterministic function random() on a "
        "static-tables-only query -- its value is drawn when the query runs, "
        "not fixed by the query"
    ), msg


def test_a_volatile_function_anywhere_refuses_not_only_in_the_order_by():
    msg = refuses("SELECT gen_random_uuid() AS o FROM s")
    assert msg.startswith(
        "unsupported: the non-deterministic function gen_random_uuid() on a"
    ), msg


def test_a_clock_function_refuses_because_two_builds_read_different_clocks():
    # DuckDB calls now() CONSISTENT_WITHIN_QUERY: one value per query, a
    # different one per build. Freezing it makes the answer a function of when
    # it was built.
    msg = refuses("SELECT now() AS o FROM s")
    assert msg.startswith("unsupported: the non-deterministic function now() on a"), msg


def test_the_bare_clock_keywords_refuse_too():
    # `current_timestamp` is a keyword, not a call: DuckDB's parse shows it as
    # a bare column reference, so the reading has to know the spelling.
    for kw in ("current_timestamp", "current_date", "localtimestamp"):
        msg = refuses(f"SELECT {kw} AS o FROM s")
        assert msg.startswith(
            f"unsupported: the non-deterministic function {kw}() on a"
        ), (kw, msg)


def test_a_consistent_function_still_serves():
    # The rule reads DuckDB's own stability column, so a CONSISTENT function
    # is a function of its arguments and stays on the frozen path.
    fn = build("SELECT upper(g) AS o, abs(v) AS t FROM s ORDER BY o")
    assert fn.backend == "constant"
    assert [r["o"] for r in fn.infer_rows([])] == ["X", "Y", "Z"]


def test_a_static_column_named_like_a_volatile_function_still_serves():
    # The bare-name reading covers the clock keywords only, so an ordinary
    # column called `uuid` or `today` is not mistaken for the function of that
    # name -- which is also how DuckDB binds it, the column winning.
    tbl = pa.table({"uuid": ["a", "b", "c"], "today": pa.array([1, 2, 3], pa.int64())})
    fn = build("SELECT uuid AS o, today AS t FROM s ORDER BY t", tbl)
    assert fn.backend == "constant"
    assert [r["o"] for r in fn.infer_rows([])] == ["a", "b", "c"]


# -------------------------------------- an aggregate that follows the scan --

AGG_MSG_HEAD = "unsupported: order-sensitive aggregate "


def test_selection_aggregates_refuse_by_name():
    # first/last/any_value/arg_max pick ONE row out of a group and the query
    # says nothing about which: measured over a 60k static table fed through a
    # tying GROUP BY, first() answered four ways and arg_max five.
    for expr, name in [
        ("first(g)", "first"),
        ("last(g)", "last"),
        ("any_value(g)", "any_value"),
        ("arbitrary(g)", "arbitrary"),
        ("arg_max(g, v)", "arg_max"),
        ("arg_min(g, v)", "arg_min"),
        ("mode(v)", "mode"),
    ]:
        msg = refuses(f"SELECT {expr} AS o FROM s")
        assert msg.startswith(AGG_MSG_HEAD + name + " on a"), (expr, msg)


def test_sequence_aggregates_refuse_by_name():
    # list/string_agg build an ORDERED container out of an unordered input.
    for expr, name in [
        ("list(g)", "list"),
        ("array_agg(g)", "array_agg"),
        ("string_agg(g, ',')", "string_agg"),
        ("group_concat(g)", "group_concat"),
    ]:
        msg = refuses(f"SELECT {expr} AS o FROM s")
        assert msg.startswith(AGG_MSG_HEAD + name + " on a"), (expr, msg)


def test_a_floating_point_accumulator_refuses_because_association_is_not_a_law():
    # DuckDB declares sum and avg ORDER_DEPENDENT, and their DOUBLE overloads
    # earn it: over 200k rows arriving in a hash order the settings move, avg
    # and sum each answered six ways.
    for name in ("sum", "avg", "stddev", "product"):
        msg = refuses(f"SELECT {name}(v) AS o FROM s")
        assert msg.startswith(AGG_MSG_HEAD + name + " on a"), (name, msg)


def test_an_aggregates_own_order_by_does_not_rescue_it():
    # The carve-out DuckDB's own optimizer applies -- an internal ORDER BY
    # over keys that separate every row -- is not read here: measuring
    # tie-freeness per GROUP is a probe this reading does not build, so the
    # family refuses whole and the message says the ORDER BY was not read.
    msg = refuses("SELECT list(g ORDER BY v) AS o FROM s")
    assert msg == (
        "unsupported: order-sensitive aggregate list on a static-tables-only "
        "query -- its answer follows scan order, and an ORDER BY inside the "
        "aggregate is not read as a fix"
    ), msg


def test_the_aggregates_duckdb_declares_order_free_still_serve():
    # count/min/max/median/bool_and are the names DuckDB itself flags
    # NOT_ORDER_DEPENDENT for every overload; measured, they answered one way
    # under all six settings.
    fn = build(
        "SELECT count(*) AS a, min(v) AS b, max(v) AS c, median(v) AS d, "
        "bool_and(v > 0) AS e FROM s"
    )
    assert fn.backend == "constant"
    assert fn.infer_rows([]) == [{"a": 3, "b": 1, "c": 2, "d": 1.0, "e": True}]


def test_a_grouped_count_still_serves():
    fn = build("SELECT g AS o, count(*) AS n FROM s GROUP BY g ORDER BY o")
    assert fn.backend == "constant"
    assert [r["n"] for r in fn.infer_rows([])] == [1, 1, 1]


# ----------------------------------------------- a window frame over rows --


def test_a_row_based_frame_refuses():
    # A ROWS frame counts NEIGHBOURS, so which rows are in it is the scan
    # order: over a hash-ordered input the same running sum answered six ways
    # while its RANGE spelling answered one.
    msg = refuses(
        "SELECT g AS o, max(v) OVER (ORDER BY v ROWS BETWEEN 1 PRECEDING AND "
        "CURRENT ROW) AS w FROM s",
        UNIQ,
    )
    assert msg.startswith("unsupported: a row-based window frame"), msg


def test_a_row_based_frame_without_an_order_by_refuses_too():
    msg = refuses(
        "SELECT g AS o, max(v) OVER (ROWS BETWEEN UNBOUNDED PRECEDING AND "
        "CURRENT ROW) AS w FROM s",
        UNIQ,
    )
    assert msg.startswith("unsupported: a row-based window frame"), msg


def test_a_range_frame_over_tied_keys_serves_because_peers_share_a_frame():
    # RANGE (the default) puts every peer of the current row in the frame, so
    # a tie among the keys cannot change the value.
    fn = build("SELECT g AS o, max(v) OVER (ORDER BY v) AS w FROM s ORDER BY g")
    assert fn.backend == "constant"
    assert [r["w"] for r in fn.infer_rows([])] == [1, 1, 2]


def test_a_whole_partition_rows_frame_serves():
    # UNBOUNDED PRECEDING to UNBOUNDED FOLLOWING is the whole partition
    # whichever way it is spelled, so no neighbour count is involved -- and
    # DuckDB's parse says so, giving those bounds no ROWS/RANGE flavour.
    fn = build(
        "SELECT g AS o, max(v) OVER (ORDER BY v ROWS BETWEEN UNBOUNDED "
        "PRECEDING AND UNBOUNDED FOLLOWING) AS w FROM s ORDER BY g"
    )
    assert fn.backend == "constant"
    assert [r["w"] for r in fn.infer_rows([])] == [2, 2, 2]


def test_an_order_sensitive_aggregate_as_a_window_function_refuses():
    msg = refuses("SELECT g AS o, list(v) OVER () AS w FROM s", UNIQ)
    assert msg.startswith(AGG_MSG_HEAD + "list on a"), msg


def test_the_rank_family_over_ties_still_serves():
    fn = build("SELECT g AS o, rank() OVER (ORDER BY v) AS w FROM s ORDER BY g")
    assert fn.backend == "constant"
    assert [r["w"] for r in fn.infer_rows([])] == [1, 1, 3]


# ------------------------------------------------- a star as the sort key --


def test_order_by_star_is_order_by_all():
    # DuckDB takes the ORDER-BY-ALL path for a bare `*` exactly as for ALL (no
    # exclude list, no replace list, no COLUMNS expression). Reading it as
    # anything else measured the wrong keys and served a tie.
    assert refuses("SELECT v AS t FROM s ORDER BY *") == TIE_MSG


def test_order_by_star_over_separated_rows_serves():
    fn = build("SELECT g AS o, v AS t FROM s ORDER BY *", UNIQ)
    assert fn.backend == "constant"
    assert [r["o"] for r in fn.infer_rows([])] == ["x", "y", "z"]


def test_order_by_columns_refuses_because_it_expands_over_the_input():
    # A COLUMNS(...) term carries an expression, so DuckDB does NOT take the
    # ORDER-BY-ALL path: it expands the star over the FROM's columns. Those
    # are not the output columns this reading can measure, so it refuses
    # rather than measuring the wrong keys.
    for spelling in (
        "COLUMNS('v')",
        "COLUMNS('^v$')",
        "COLUMNS(c -> c = 'v')",
        "COLUMNS(['v'])",
    ):
        msg = refuses(f"SELECT g AS o, v AS t FROM s ORDER BY {spelling}")
        assert msg.startswith("unsupported: a star sort key this reading"), (
            spelling,
            msg,
        )


# ------------------------------------------ what the wrapper must not break --


def test_a_trailing_semicolon_does_not_turn_a_unique_key_into_a_refusal():
    # The probe wraps the statement, and a raw text wrap made a trailing `;` a
    # syntax error that arrived as a refusal. The statement goes through
    # DuckDB's own round trip instead, which normalises it away.
    fn = build("SELECT g AS o, v AS t FROM s ORDER BY t;", UNIQ)
    assert fn.backend == "constant"
    assert [r["t"] for r in fn.infer_rows([])] == [1, 2, 3]


def test_a_trailing_comment_does_not_turn_a_unique_key_into_a_refusal():
    fn = build("SELECT g AS o, v AS t FROM s ORDER BY t -- done", UNIQ)
    assert fn.backend == "constant"
    assert [r["t"] for r in fn.infer_rows([])] == [1, 2, 3]


def test_a_trailing_semicolon_still_refuses_a_real_tie():
    assert refuses("SELECT g AS o, v AS t FROM s ORDER BY v;") == TIE_MSG


# --------------------------------------------- a limit that limits nothing --


def test_limit_all_removes_no_row_and_must_not_refuse():
    fn = build("SELECT g AS o, v AS t FROM s ORDER BY v LIMIT ALL", UNIQ)
    assert fn.backend == "constant"
    assert [r["t"] for r in fn.infer_rows([])] == [1, 2, 3]


def test_offset_zero_removes_no_row_and_must_not_refuse():
    fn = build("SELECT g AS o, v AS t FROM s ORDER BY v OFFSET 0", UNIQ)
    assert fn.backend == "constant"
    assert [r["t"] for r in fn.infer_rows([])] == [1, 2, 3]


def test_a_real_limit_still_refuses():
    msg = refuses("SELECT g AS o FROM s LIMIT 1", UNIQ)
    assert msg.startswith("unsupported: row limit (LIMIT/OFFSET)"), msg


def test_a_real_offset_still_refuses():
    msg = refuses("SELECT g AS o FROM s OFFSET 1", UNIQ)
    assert msg.startswith("unsupported: row limit (LIMIT/OFFSET)"), msg


# ------------------------------------- the message names the cause it found --


def test_more_than_one_statement_is_named_as_that():
    # DuckDB serializes a multi-statement string perfectly well; only the
    # COUNT is wrong. Refusing is right, blaming the serialization is not.
    msg = refuses("SELECT 1 AS o; SELECT g AS o, v AS t FROM s ORDER BY v", UNIQ)
    assert msg == (
        "unsupported: ORDER BY in a statement string holding more than one "
        "statement -- what a static-tables-only query selects may be frozen "
        "only when it is a function of the query, and only one statement of "
        "this string is read"
    ), msg


def test_a_dynamic_query_keeps_the_row_paths_own_error():
    # A refusal that names the static-tables-only path may be pinned only on
    # a query that IS one. These two read the row table, so the fold cannot
    # run at all and the row path's own error is what reaches the caller --
    # naming the same clause, without claiming a path never taken.
    for sql in (
        "SELECT k AS o FROM __THIS__ LIMIT 1",
        "FROM __THIS__ SELECT k AS o LIMIT 1",
    ):
        with pytest.raises(ValueError) as e:
            DuckDBInferFn(sql, row_tables={"__THIS__": ROW}, static_tables={"s": UNIQ})
        msg = " ".join(str(e.value).split())
        assert msg == "unsupported: LIMIT/OFFSET", (sql, msg)


def test_a_dynamic_query_whose_only_problem_is_dialect_says_so():
    with pytest.raises(ValueError) as e:
        DuckDBInferFn(
            "FROM __THIS__ SELECT k AS o",
            row_tables={"__THIS__": ROW},
            static_tables={"s": UNIQ},
        )
    assert "unsupported: FROM-first SELECT" in str(e.value), str(e.value)


# ------------------------------- a shape the reading could not rule out --


def test_a_limit_that_is_not_a_bare_constant_refuses():
    # LIMIT ALL and OFFSET 0 remove no row because DuckDB's parse SAYS so: a
    # NULL-typed constant and a zero. Any other limit expression is one this
    # reading cannot evaluate, and an unevaluated limit is a row subset
    # nobody ruled out -- so it counts as the limit it is.
    for sql in (
        "SELECT g AS o FROM s LIMIT 1+1",
        "SELECT g AS o FROM s OFFSET 1+0",
        "SELECT g AS o FROM s LIMIT (SELECT 2)",
        "SELECT g AS o FROM s LIMIT CAST(2 AS BIGINT)",
    ):
        msg = refuses(sql, UNIQ)
        assert msg.startswith("unsupported: row limit (LIMIT/OFFSET)"), (sql, msg)


def test_more_than_one_statement_refuses_even_without_an_ordering_word():
    # Only one statement of the string is read, so every value rule was asked
    # of the wrong statement: these three froze a draw, a scan-order pick and
    # three more draws. The COUNT alone is the refusal; no clause word has to
    # appear for it to fire.
    for sql in (
        "SELECT 1 AS o; SELECT random() AS o FROM s",
        "SELECT 1 AS o; SELECT first(g) AS o FROM s",
    ):
        msg = refuses(sql)
        assert "statement string holding more than one statement" in msg, (sql, msg)
    # A DDL statement in front of the SELECT does not serialize at all, so
    # this one refuses one reading earlier -- under the serialization, not
    # under the count. Either way it is refused, which is the point: it froze
    # three draws through a macro of its own making.
    msg = refuses("CREATE MACRO m(x) AS random(); SELECT m(v) AS o FROM s")
    assert "would not expose for inspection" in msg, msg


def test_a_statement_duckdb_will_not_expose_refuses_without_an_ordering_word():
    # PIVOT runs but does not serialize, so NOTHING about it was read -- not
    # its limits, not its aggregates, not its draws. `USING first(v)` froze a
    # scan-order pick. A shape nobody could read is a shape nobody ruled out,
    # which costs the deterministic `USING max(v)` its answer as well.
    for sql in (
        "PIVOT s ON g USING first(v)",
        "PIVOT s ON g USING sum(v)",
        "PIVOT s ON g USING max(v)",
    ):
        msg = refuses(sql)
        assert "would not expose for inspection" in msg, (sql, msg)


def test_a_macro_that_reads_the_clock_refuses_like_the_clock_it_reads():
    # duckdb_functions().stability is NULL for every macro, so the catalogue
    # lookup cannot answer for one -- but the same catalogue keeps the macro's
    # DEFINITION, and pg_postmaster_start_time expands to current_timestamp.
    for name, call in (
        ("pg_postmaster_start_time", "pg_postmaster_start_time()"),
        ("ago", "ago(INTERVAL 1 DAY)"),
        ("current_query", "current_query()"),
    ):
        msg = refuses(f"SELECT {call}::VARCHAR AS o FROM s")
        assert msg.startswith(
            f"unsupported: the non-deterministic function {name}()"
        ), msg


def test_a_macro_that_reads_nothing_of_the_run_still_serves():
    # The reading is the macro's own definition, not its family: fdiv expands
    # to arithmetic and current_user to the constant 'duckdb'.
    fn = build(
        "SELECT v AS t, fdiv(v, 2) AS o, current_user AS u FROM s ORDER BY t", UNIQ
    )
    assert fn.backend == "constant"
    assert [r["u"] for r in fn.infer_rows([])] == ["duckdb"] * 3


def test_a_clock_function_duckdbs_own_flag_calls_consistent_refuses_anyway():
    # ICU registers current_localtime/current_localtimestamp with no
    # SetStability (extension/icu/icu-timezone.cpp), so they inherit
    # CONSISTENT -- and measured, the value moves between two connections 50ms
    # apart. DuckDB's own binder maps the bare words `localtime` and
    # `localtimestamp` to exactly these two functions
    # (bind_columnref_expression.cpp), and those already refused; the call
    # spelling of the same function refuses with them.
    for call in ("current_localtimestamp()", "current_localtime()"):
        msg = refuses(f"SELECT {call}::VARCHAR AS o FROM s")
        assert "on a static-tables-only query" in msg, (call, msg)


def test_a_value_that_is_a_function_of_the_build_refuses():
    # Both are CONSISTENT within one run and a function of the MACHINE across
    # runs: two builds of the same query freeze two different answers, which
    # is the skew this path exists to close.
    for call in ("version()", "current_setting('threads')"):
        msg = refuses(f"SELECT {call}::VARCHAR AS o FROM s")
        assert "on a static-tables-only query" in msg, (call, msg)


def test_rowid_is_selection_by_position_under_another_name():
    # A static table is materialized by CREATE TABLE AS SELECT, so rowid is
    # whatever physical position that produced: the same reading row_number()
    # gets, spelled as an ordinary column -- and in a WHERE it is a row limit
    # as well.
    for sql in (
        "SELECT rowid AS o FROM s",
        "SELECT s.rowid AS o FROM s",
        "SELECT g AS o FROM s WHERE rowid < 2",
    ):
        assert "rowid" in refuses(sql, UNIQ), sql
