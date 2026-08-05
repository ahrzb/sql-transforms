"""Marginalization unit tests: pins, rewrite goldens, and the refusal table."""

import duckdb
import pytest

from sql_transform._marginalize import (
    MarginalizeError,
    _serialize,
    _templates,
    marginalize,
)

# --- AST pins ----------------------------------------------------------------
# The JSON AST is a DuckDB-internal format. Every shape the walker relies on
# is asserted here from an executed serialization, so a DuckDB upgrade that
# moves the format fails loudly in one place.


def _node(sql: str) -> dict:
    return _serialize(sql)["statements"][0]["node"]


def test_pin_select_node_fields():
    node = _node("SELECT age FROM __THIS__")
    assert node["type"] == "SELECT_NODE"
    assert node["modifiers"] == []
    assert node["cte_map"] == {"map": []}
    assert node["where_clause"] is None
    assert node["group_expressions"] == []
    assert node["group_sets"] == []
    assert node["aggregate_handling"] == "STANDARD_HANDLING"
    assert node["having"] is None
    assert node["sample"] is None
    assert node["qualify"] is None
    ft = node["from_table"]
    assert ft["type"] == "BASE_TABLE"
    assert ft["table_name"] == "__THIS__"
    assert ft["alias"] == ""
    assert ft["sample"] is None


def test_pin_default_window_frame():
    (w,) = _node("SELECT avg(age) OVER () FROM __THIS__")["select_list"]
    assert w["class"] == "WINDOW"
    assert w["type"] == "WINDOW_AGGREGATE"
    assert w["function_name"] == "avg"
    assert (w["start"], w["end"]) == ("UNBOUNDED_PRECEDING", "CURRENT_ROW_RANGE")
    assert (
        w["start_expr"]
        is w["end_expr"]
        is w["offset_expr"]
        is w["default_expr"]
        is None
    )
    assert w["exclude_clause"] == "NO_OTHER"
    assert w["orders"] == [] and w["arg_orders"] == []
    assert w["filter_expr"] is None
    assert w["distinct"] is False and w["ignore_nulls"] is False


def test_pin_explicit_frame_differs_from_default():
    (w,) = _node(
        "SELECT avg(a) OVER (ROWS BETWEEN 1 PRECEDING AND CURRENT ROW) FROM __THIS__"
    )["select_list"]
    assert (w["start"], w["end"]) != ("UNBOUNDED_PRECEDING", "CURRENT_ROW_RANGE")


def test_pin_window_classification():
    (w,) = _node("SELECT row_number() OVER () FROM __THIS__")["select_list"]
    assert w["type"] == "WINDOW_ROW_NUMBER"


def test_pin_join_templates():
    t = _templates()
    assert t["left_join"]["join_type"] == "LEFT"
    assert t["left_join"]["ref_type"] == "REGULAR"
    assert t["always_true"]["type"] == "COMPARE_EQUAL"
    assert t["collapse_doc"]["statements"][0]["node"]["modifiers"] == [
        {"type": "DISTINCT_MODIFIER", "distinct_on_targets": []}
    ]
    assert t["not_distinct"]["type"] == "COMPARE_NOT_DISTINCT_FROM"
    assert t["conjunction"]["type"] == "CONJUNCTION_AND"
    assert t["column_ref"]["class"] == "COLUMN_REF"


def test_pin_subquery_class():
    (sq,) = _node("SELECT (SELECT 1) FROM __THIS__")["select_list"]
    assert sq["class"] == "SUBQUERY"


def test_pin_star_class():
    (star,) = _node("SELECT * FROM __THIS__")["select_list"]
    assert star["class"] == "STAR"
    assert star["relation_name"] == ""
    assert star["columns"] is False


# --- rewrite goldens ---------------------------------------------------------


def test_golden_single_partition():
    m = marginalize(
        "SELECT (age - avg(age) OVER (PARTITION BY country)) AS d FROM __THIS__"
    )
    assert m.serving_sql == (
        "SELECT (__cf_t.age - __cf_p0.__cf_a0) AS d "
        "FROM __THIS__ AS __cf_t LEFT JOIN __CF_PARAMS_0__ AS __cf_p0 "
        "ON ((__cf_t.country IS NOT DISTINCT FROM __cf_p0.country))"
    )
    (spec,) = m.params
    assert spec.name == "__CF_PARAMS_0__"
    assert spec.keys == ("country",)
    # The fit plan: the level step reruns the original computation (the
    # original items pin the window-operator chain; GROUP BY or a solo window
    # query would sum floats in a different order — ulp drift, fuzz-found),
    # then the collapse step picks the materialized values.
    level, collapse = m.plan
    assert level.name == "__CF_LEVEL_0__" and level.reads == ("__THIS__",)
    assert level.sql == (
        "SELECT (age - avg(age) OVER (PARTITION BY country)) AS d,"
        " avg(age) OVER (PARTITION BY country) AS __cf_w0, country AS __cf_k0"
        " FROM __THIS__"
    )
    assert collapse.name == "__CF_PARAMS_0__"
    assert collapse.reads == ("__CF_LEVEL_0__",)
    assert collapse.sql == (
        "SELECT DISTINCT __cf_k0 AS country, __cf_w0 AS __cf_a0 FROM __CF_LEVEL_0__"
    )


def test_golden_two_keysets_and_dedupe():
    m = marginalize(
        "SELECT (age - avg(age) OVER (PARTITION BY country))"
        " / stddev_samp(age) OVER (PARTITION BY country) AS age_z,"
        " avg(age) OVER (PARTITION BY country) AS mu,"
        " fare - avg(fare) OVER () AS fare_c FROM __THIS__"
    )
    p0, p1 = m.params
    sqls = {step.name: step.sql for step in m.plan}
    # avg(age) appears twice under the same key set: one column.
    assert sqls["__CF_PARAMS_0__"] == (
        "SELECT DISTINCT __cf_k0 AS country, __cf_w0 AS __cf_a0,"
        " __cf_w1 AS __cf_a1 FROM __CF_LEVEL_0__"
    )
    assert p1.keys == ()
    assert sqls["__CF_PARAMS_1__"] == (
        "SELECT DISTINCT __cf_w2 AS __cf_a0 FROM __CF_LEVEL_0__"
    )
    assert sqls["__CF_LEVEL_0__"].count("__cf_w") == 3
    assert "avg(fare) OVER () AS __cf_w2" in sqls["__CF_LEVEL_0__"]
    # Keyless params (exactly one row) join via LEFT JOIN ON (1 = 1) — never a
    # CROSS join, whose comma print form re-parses with other associativity.
    assert "LEFT JOIN __CF_PARAMS_1__ AS __cf_p1 ON ((1 = 1))" in m.serving_sql
    assert m.serving_sql.count("__cf_p0.__cf_a0") == 2


def test_golden_multi_key_join():
    m = marginalize("SELECT avg(x) OVER (PARTITION BY a, b) AS m FROM __THIS__")
    (spec,) = m.params
    assert spec.keys == ("a", "b")
    assert (
        "ON (((__cf_t.a IS NOT DISTINCT FROM __cf_p0.a)"
        " AND (__cf_t.b IS NOT DISTINCT FROM __cf_p0.b)))"
    ) in m.serving_sql


def test_no_aggregates_is_identity_modulo_normalization():
    m = marginalize("SELECT age + 1 AS b, name FROM __THIS__")
    assert m.params == ()
    assert m.serving_sql == 'SELECT (age + 1) AS b, "name" FROM __THIS__'


def test_unaliased_outputs_keep_their_derived_names():
    m = marginalize("SELECT age + 1, avg(age) OVER () FROM __THIS__")
    assert '"(age + 1)"' in m.serving_sql
    assert '"avg(age) OVER ()"' in m.serving_sql


def test_qualified_this_refs_are_normalized():
    m = marginalize(
        "SELECT __THIS__.age - avg(__THIS__.age) OVER () AS d FROM __THIS__"
    )
    assert "__cf_t.age" in m.serving_sql
    sqls = {step.name: step.sql for step in m.plan}
    assert sqls["__CF_PARAMS_0__"] == (
        "SELECT DISTINCT __cf_w0 AS __cf_a0 FROM __CF_LEVEL_0__"
    )
    # Fit-side text strips the __THIS__ qualifier (bare against the source).
    assert "avg(age) OVER () AS __cf_w0" in sqls["__CF_LEVEL_0__"]


def test_star_is_qualified_when_joins_appear():
    m = marginalize("SELECT *, avg(a) OVER () AS m FROM __THIS__")
    assert "SELECT __cf_t.*" in m.serving_sql


def test_case_insensitive_keyset_identity():
    m = marginalize(
        "SELECT avg(a) OVER (PARTITION BY c) AS x,"
        " avg(b) OVER (PARTITION BY C) AS y FROM __THIS__"
    )
    assert len(m.params) == 1


# --- widened windows (loop 2): order values join the key set -----------------


def test_golden_running_window_keys_on_order_values():
    m = marginalize(
        "SELECT sum(x) OVER (PARTITION BY k ORDER BY o) AS running FROM __THIS__"
    )
    (spec,) = m.params
    assert spec.keys == ("k", "o")
    assert (
        "ON (((__cf_t.k IS NOT DISTINCT FROM __cf_p0.k)"
        " AND (__cf_t.o IS NOT DISTINCT FROM __cf_p0.o)))"
    ) in m.serving_sql


def test_golden_rank_keys_on_order_values():
    m = marginalize("SELECT rank() OVER (PARTITION BY k ORDER BY o) AS r FROM __THIS__")
    (spec,) = m.params
    assert spec.keys == ("k", "o")


def test_golden_first_value_is_partition_keyed():
    m = marginalize(
        "SELECT first_value(x) OVER (PARTITION BY k ORDER BY o) AS f FROM __THIS__"
    )
    (spec,) = m.params
    assert spec.keys == ("k",)


def test_golden_whole_partition_frame_is_partition_keyed():
    m = marginalize(
        "SELECT sum(x) OVER (PARTITION BY k ORDER BY o"
        " ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS s FROM __THIS__"
    )
    (spec,) = m.params
    assert spec.keys == ("k",)


def test_golden_expression_partition_key():
    m = marginalize("SELECT avg(x) OVER (PARTITION BY k % 2) AS m FROM __THIS__")
    (spec,) = m.params
    assert spec.keys == ("__cf_x0",)
    assert "__cf_p0.__cf_x0" in m.serving_sql
    assert "__cf_t.k" in m.serving_sql  # the qualified expression joins back


def test_running_and_rank_share_a_keyset():
    m = marginalize(
        "SELECT sum(x) OVER (PARTITION BY k ORDER BY o) AS s,"
        " rank() OVER (PARTITION BY k ORDER BY o) AS r FROM __THIS__"
    )
    assert len(m.params) == 1
    assert m.params[0].keys == ("k", "o")


def test_order_key_deduped_against_partition():
    m = marginalize("SELECT sum(x) OVER (PARTITION BY k ORDER BY k) AS s FROM __THIS__")
    (spec,) = m.params
    assert spec.keys == ("k",)


def test_collate_order_key_strips_to_the_raw_column():
    m = marginalize("SELECT rank() OVER (ORDER BY c COLLATE NOCASE) AS r FROM __THIS__")
    (spec,) = m.params
    assert spec.keys == ("c",)


# --- projection chains and scalar subqueries (loop 3) ------------------------


def test_golden_cte_flattens_with_empty_plan():
    m = marginalize(
        "WITH a AS (SELECT x + 1 AS b FROM __THIS__) SELECT b * 2 AS c FROM a"
    )
    assert m.plan == ()
    assert m.params == ()
    assert m.serving_sql == "SELECT ((__cf_t.x + 1) * 2) AS c FROM __THIS__ AS __cf_t"


def test_golden_derived_table_flattens():
    m = marginalize("SELECT c FROM (SELECT x + 1 AS c FROM __THIS__) AS sub")
    assert m.serving_sql == "SELECT (__cf_t.x + 1) AS c FROM __THIS__ AS __cf_t"


def test_golden_cte_column_aliases():
    m = marginalize("WITH a(z) AS (SELECT x + 1 FROM __THIS__) SELECT z FROM a")
    assert m.serving_sql == "SELECT (__cf_t.x + 1) AS z FROM __THIS__ AS __cf_t"


def test_golden_nested_aggregation_dag():
    # The DAG case: the second aggregate depends on the first level's output.
    m = marginalize(
        "WITH c AS (SELECT x - avg(x) OVER () AS cx FROM __THIS__)"
        " SELECT stddev_samp(cx) OVER () AS s FROM c"
    )
    names = [step.name for step in m.plan]
    assert names == [
        "__CF_LEVEL_0__",
        "__CF_PARAMS_0__",
        "__CF_LEVEL_1__",
        "__CF_PARAMS_1__",
    ]
    steps = {s.name: s for s in m.plan}
    assert steps["__CF_LEVEL_1__"].reads == ("__CF_LEVEL_0__",)
    assert "stddev_samp(cx) OVER () AS __cf_w1" in steps["__CF_LEVEL_1__"].sql
    assert "__cf_p1.__cf_a0 AS s" in m.serving_sql


def test_golden_scalar_subquery_runs_verbatim():
    m = marginalize("SELECT x / (SELECT max(x) FROM __THIS__) AS xn FROM __THIS__")
    (step,) = m.plan
    assert step.name == "__CF_PARAMS_0__"
    assert step.sql == "SELECT (SELECT max(x) FROM __THIS__) AS __cf_a0"
    assert step.reads == ("__THIS__",)
    assert m.serving_sql == (
        "SELECT (__cf_t.x / __cf_p0.__cf_a0) AS xn FROM __THIS__ AS __cf_t"
        " LEFT JOIN __CF_PARAMS_0__ AS __cf_p0 ON ((1 = 1))"
    )


def test_golden_exists_subquery():
    m = marginalize(
        "SELECT EXISTS(SELECT 1 FROM __THIS__ WHERE x > 5) AS any_big FROM __THIS__"
    )
    (step,) = m.plan
    assert "EXISTS" in step.sql
    assert "__cf_p0.__cf_a0 AS any_big" in m.serving_sql


def test_scalar_subquery_may_contain_anything_over_this():
    # WHERE/GROUP BY inside a scalar subquery are fine: it is a scalar
    # computation run verbatim, not a projection.
    m = marginalize(
        "SELECT x - (SELECT avg(x) FROM __THIS__ WHERE x > 0) AS d FROM __THIS__"
    )
    (step,) = m.plan
    assert "WHERE" in step.sql


def test_star_expansion_through_cte():
    m = marginalize("WITH a AS (SELECT x + 1 AS b, y FROM __THIS__) SELECT * FROM a")
    assert m.serving_sql == (
        "SELECT (__cf_t.x + 1) AS b, __cf_t.y AS y FROM __THIS__ AS __cf_t"
    )


# --- schema-aware resolution (loop 4) ----------------------------------------

COLS = ["age", "fare", "country", "name"]


def test_schema_unknown_column_refuses_at_construction():
    with pytest.raises(MarginalizeError, match="unknown column nope"):
        marginalize("SELECT nope FROM __THIS__", COLS)


def test_schema_star_expands_explicitly():
    m = marginalize("SELECT * FROM __THIS__", ["a", "b"])
    assert m.serving_sql == (
        "SELECT __cf_t.a AS a, __cf_t.b AS b FROM __THIS__ AS __cf_t"
    )


def test_schema_columns_expands_via_the_oracle_regex():
    m = marginalize("SELECT COLUMNS('a.*') FROM __THIS__", ["aa", "ab", "b"])
    assert m.serving_sql == (
        "SELECT __cf_t.aa AS aa, __cf_t.ab AS ab FROM __THIS__ AS __cf_t"
    )


def test_schema_star_modifiers_compose():
    m = marginalize(
        "SELECT * EXCLUDE (b) REPLACE (a + 1 AS a) RENAME (c AS z) FROM __THIS__",
        ["a", "b", "c"],
    )
    assert m.serving_sql == (
        "SELECT (__cf_t.a + 1) AS a, __cf_t.c AS z FROM __THIS__ AS __cf_t"
    )


def test_schema_lateral_alias_inlines_when_no_column():
    m = marginalize("SELECT a + 1 AS b, b * 2 AS c FROM __THIS__", ["a"])
    assert "((__cf_t.a + 1) * 2) AS c" in m.serving_sql


def test_schema_lateral_alias_loses_to_the_column():
    m = marginalize("SELECT a + 1 AS b, b * 2 AS c FROM __THIS__", ["a", "b"])
    assert "(__cf_t.b * 2) AS c" in m.serving_sql


def test_schema_struct_access_composes_through_plain_columns():
    m = marginalize("WITH a AS (SELECT s FROM __THIS__) SELECT s.f AS f FROM a", ["s"])
    assert "__cf_t.s.f AS f" in m.serving_sql


SCHEMA_REFUSALS = [
    ("SELECT nope + 1 FROM __THIS__", "unknown column nope"),
    ("SELECT COLUMNS('zz.*') FROM __THIS__", "matched no columns"),
    ("SELECT COLUMNS(c -> c LIKE 'a%') FROM __THIS__", "lambda"),
    ("SELECT * EXCLUDE (nope) FROM __THIS__", "unknown column nope"),
    # `a + 1 AS b, avg(b) OVER ()` refused here pre-slice-2; lateral aliases
    # now β-reduce into windows (accepted, pinned in _private_test).
    ("SELECT COLUMNS('a.*') + 1 FROM __THIS__", "COLUMNS.*inside an expression"),
    ("SELECT min(COLUMNS('a.*')) FROM __THIS__", "without OVER"),
]


@pytest.mark.parametrize(
    "sql,match", SCHEMA_REFUSALS, ids=[s[:48] for s, _ in SCHEMA_REFUSALS]
)
def test_schema_refusals_are_named(sql, match):
    with pytest.raises(MarginalizeError, match=match):
        marginalize(sql, ["a", "age", "aa"])


def test_schema_validation():
    with pytest.raises(MarginalizeError, match="duplicate column names"):
        marginalize("SELECT 1 AS x FROM __THIS__", ["a", "A"])
    with pytest.raises(MarginalizeError, match="reserved prefix"):
        marginalize("SELECT 1 AS x FROM __THIS__", ["__cf_bad"])


def test_windows_at_upper_level_key_on_projected_expressions():
    m = marginalize(
        "WITH a AS (SELECT lower(k) AS lk, x FROM __THIS__)"
        " SELECT avg(x) OVER (PARTITION BY lk) AS m FROM a"
    )
    (spec,) = m.params
    assert spec.keys == ("lk",)
    # Serving joins by the substituted expression, fit keys by the level column.
    assert "lower(__cf_t.k) IS NOT DISTINCT FROM __cf_p0.lk" in m.serving_sql
    steps = {s.name: s for s in m.plan}
    assert "lk AS __cf_k0" in steps["__CF_LEVEL_1__"].sql


# --- the refusal table -------------------------------------------------------

REFUSALS = [
    ("SELECT a FROM __THIS__ WHERE a > 0", "WHERE"),
    ("SELECT a FROM __THIS__ GROUP BY a", "GROUP BY"),
    ("SELECT a FROM __THIS__ GROUP BY a HAVING a > 0", "HAVING"),
    ("SELECT a FROM __THIS__ QUALIFY row_number() OVER () = 1", "QUALIFY"),
    ("SELECT a FROM __THIS__ USING SAMPLE 10", "USING SAMPLE"),
    ("SELECT a FROM __THIS__ ORDER BY a", "ORDER BY"),
    ("SELECT a FROM __THIS__ LIMIT 5", "LIMIT"),
    ("SELECT DISTINCT a FROM __THIS__", "DISTINCT"),
    ("SELECT a FROM __THIS__ JOIN s ON true", "joins and set operations"),
    ("SELECT a FROM other_table", "row table must be __THIS__"),
    ("SELECT 1", "joins and set operations"),
    ("SELECT a FROM __THIS__ AS x", "alias on __THIS__"),
    ("SELECT a FROM __THIS__ UNION SELECT 1", "query kind"),
    ("SELECT 1; SELECT 2", "multiple SQL statements"),
    ("SELECT avg(a) FROM __THIS__", "without OVER"),
    ("SELECT sum(a) FROM __THIS__", "without OVER"),
    # Position-dependent windows: a join key cannot carry physical order.
    ("SELECT row_number() OVER () FROM __THIS__", "physical row position"),
    ("SELECT ntile(4) OVER (ORDER BY a) FROM __THIS__", "physical row position"),
    ("SELECT lag(a) OVER (PARTITION BY c ORDER BY o) FROM __THIS__", "physical"),
    ("SELECT lead(a) OVER (PARTITION BY c ORDER BY o) FROM __THIS__", "physical"),
    (
        "SELECT avg(a) OVER (ROWS BETWEEN 1 PRECEDING AND CURRENT ROW) FROM __THIS__",
        "ROWS frame",
    ),
    (
        "SELECT sum(a) OVER (PARTITION BY c ORDER BY o"
        " ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) FROM __THIS__",
        "ROWS frame",
    ),
    (
        "SELECT sum(a) OVER (ORDER BY o RANGE BETWEEN UNBOUNDED PRECEDING AND"
        " CURRENT ROW EXCLUDE CURRENT ROW) FROM __THIS__",
        "EXCLUDE",
    ),
    (
        "SELECT sum(a) OVER (ORDER BY o RANGE BETWEEN b PRECEDING AND"
        " CURRENT ROW) FROM __THIS__",
        "non-constant window frame bound",
    ),
    (
        "SELECT nth_value(a, b) OVER (PARTITION BY c ORDER BY o) FROM __THIS__",
        "nth_value with a non-constant n",
    ),
    ("SELECT avg(sum(a)) OVER () FROM __THIS__", "aggregate sum inside an aggregate"),
    (
        "SELECT avg(a) FILTER (WHERE avg(b) > 0) OVER () FROM __THIS__",
        "aggregate avg inside a FILTER",
    ),
    (
        "SELECT avg(a) OVER (PARTITION BY (SELECT 1)) FROM __THIS__",
        "subquery inside a partition or order key",
    ),
    # loop 3: chains and subqueries
    ("SELECT x IN (SELECT y FROM __THIS__) FROM __THIS__", "IN/ANY"),
    ("SELECT x = ANY(SELECT y FROM __THIS__) FROM __THIS__", "IN/ANY"),
    (
        "SELECT (SELECT max(y) FROM other) FROM __THIS__",
        "table other inside a subquery",
    ),
    ("SELECT $1 + a FROM __THIS__", "prepared-statement parameter"),
    ("SELECT a + 1 AS b, b * 2 FROM __THIS__", "lateral alias"),
    ("WITH a AS (SELECT 1) SELECT x FROM __THIS__", "unused CTE a"),
    ("WITH a AS (SELECT * FROM a) SELECT * FROM a", "referenced more than once"),
    (
        "WITH a AS (SELECT x AS n, y AS n FROM __THIS__) SELECT n FROM a",
        "duplicate output name n",
    ),
    (
        "WITH a AS (SELECT x FROM __THIS__) SELECT __THIS__.x FROM a",
        "__THIS__ is not in scope",
    ),
    (
        "WITH a AS (SELECT x + 1 AS s FROM __THIS__) SELECT s.f FROM a",
        "struct-field access through a projected expression",
    ),
    (
        "WITH a AS (SELECT * EXCLUDE (x) FROM __THIS__) SELECT y FROM a",
        "EXCLUDE",
    ),
    ("SELECT a FROM (SELECT b FROM __THIS__ WHERE b > 0)", "WHERE"),
    ("SELECT avg(a) OVER () AS __cf_x FROM __THIS__", "reserved prefix"),
    ("SELECT __CF_PARAMS_0__.a FROM __THIS__", "reserved prefix"),
    ("SELECT COLUMNS('a.*') FROM __THIS__", "COLUMNS"),
    ("SELECT FROM WHERE", "parse error"),
]


@pytest.mark.parametrize("sql,match", REFUSALS, ids=[s[:48] for s, _ in REFUSALS])
def test_refusals_are_named(sql, match):
    with pytest.raises(MarginalizeError, match=match):
        marginalize(sql)


def test_serving_sql_reparses_through_the_oracle():
    m = marginalize(
        "SELECT (age - avg(age) OVER (PARTITION BY country))"
        " / stddev_samp(age) OVER (PARTITION BY country) AS age_z,"
        " fare - avg(fare) OVER () AS fare_c FROM __THIS__"
    )
    for sql in (m.serving_sql, *(step.sql for step in m.plan)):
        assert not _serialize(sql).get("error")


def test_aggregate_catalog_is_the_oracles():
    rows = duckdb.execute(
        "SELECT count(*) FROM duckdb_functions() WHERE function_type = 'aggregate'"
    ).fetchone()
    assert rows[0] > 0
