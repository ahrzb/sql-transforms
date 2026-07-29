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
    # Fit is two-stage: windows_sql reruns the original computation (the
    # original items pin the window-operator chain; GROUP BY or a solo window
    # query would sum floats in a different order — ulp drift, fuzz-found),
    # then fit_sql collapses the materialized values.
    assert m.windows_sql == (
        "SELECT (age - avg(age) OVER (PARTITION BY country)) AS __cf_o0,"
        " avg(age) OVER (PARTITION BY country) AS __cf_w0, country AS __cf_k0"
        " FROM __THIS__"
    )
    assert spec.fit_sql == (
        "SELECT DISTINCT __cf_k0 AS country, __cf_w0 AS __cf_a0 FROM __CF_WINDOWS__"
    )


def test_golden_two_keysets_and_dedupe():
    m = marginalize(
        "SELECT (age - avg(age) OVER (PARTITION BY country))"
        " / stddev_samp(age) OVER (PARTITION BY country) AS age_z,"
        " avg(age) OVER (PARTITION BY country) AS mu,"
        " fare - avg(fare) OVER () AS fare_c FROM __THIS__"
    )
    p0, p1 = m.params
    # avg(age) appears twice under the same key set: one column.
    assert p0.fit_sql == (
        "SELECT DISTINCT __cf_k0 AS country, __cf_w0 AS __cf_a0,"
        " __cf_w1 AS __cf_a1 FROM __CF_WINDOWS__"
    )
    assert p1.keys == ()
    assert p1.fit_sql == "SELECT DISTINCT __cf_w2 AS __cf_a0 FROM __CF_WINDOWS__"
    assert m.windows_sql.count("__cf_w") == 3
    assert "avg(fare) OVER () AS __cf_w2" in m.windows_sql
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
    (spec,) = m.params
    assert spec.fit_sql == "SELECT DISTINCT __cf_w0 AS __cf_a0 FROM __CF_WINDOWS__"
    assert "avg(__THIS__.age) OVER () AS __cf_w0" in m.windows_sql


def test_star_is_qualified_when_joins_appear():
    m = marginalize("SELECT *, avg(a) OVER () AS m FROM __THIS__")
    assert "SELECT __cf_t.*" in m.serving_sql


def test_case_insensitive_keyset_identity():
    m = marginalize(
        "SELECT avg(a) OVER (PARTITION BY c) AS x,"
        " avg(b) OVER (PARTITION BY C) AS y FROM __THIS__"
    )
    assert len(m.params) == 1


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
    ("WITH x AS (SELECT 1) SELECT a FROM __THIS__", "common table"),
    ("SELECT (SELECT 1) FROM __THIS__", "subquery"),
    ("SELECT a FROM __THIS__ JOIN s ON true", "FROM must be exactly __THIS__"),
    ("SELECT a FROM other_table", "row table must be __THIS__"),
    ("SELECT 1", "FROM must be exactly __THIS__"),
    ("SELECT a FROM __THIS__ AS x", "alias on __THIS__"),
    ("SELECT a FROM __THIS__ UNION SELECT 1", "query kind"),
    ("SELECT 1; SELECT 2", "multiple SQL statements"),
    ("SELECT avg(a) FROM __THIS__", "without OVER"),
    ("SELECT sum(a) FROM __THIS__", "without OVER"),
    ("SELECT avg(a) OVER (ORDER BY b) FROM __THIS__", "ORDER BY"),
    (
        "SELECT avg(a) OVER (ROWS BETWEEN 1 PRECEDING AND CURRENT ROW) FROM __THIS__",
        "window frame",
    ),
    ("SELECT row_number() OVER () FROM __THIS__", "window function row_number"),
    ("SELECT lag(a) OVER (PARTITION BY c) FROM __THIS__", "not a per-group constant"),
    # The oracle classifies first() as its own window type, not an aggregate.
    ("SELECT first(a) OVER (PARTITION BY c) FROM __THIS__", "window function first"),
    ("SELECT string_agg(a, ',') OVER (PARTITION BY c) FROM __THIS__", "allowlist"),
    ("SELECT avg(DISTINCT a) OVER (PARTITION BY c) FROM __THIS__", "DISTINCT inside"),
    (
        "SELECT avg(a) FILTER (WHERE a > 0) OVER (PARTITION BY c) FROM __THIS__",
        "FILTER inside",
    ),
    (
        "SELECT avg(a) OVER (PARTITION BY c + 1) FROM __THIS__",
        "PARTITION BY expression",
    ),
    ("SELECT avg(a) OVER (PARTITION BY s.c) FROM __THIS__", "PARTITION BY s.c"),
    ("SELECT avg(sum(a)) OVER () FROM __THIS__", "aggregate sum inside an aggregate"),
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
    for sql in (m.serving_sql, m.windows_sql, *(s.fit_sql for s in m.params)):
        assert not _serialize(sql).get("error")


def test_aggregate_catalog_is_the_oracles():
    rows = duckdb.execute(
        "SELECT count(*) FROM duckdb_functions() WHERE function_type = 'aggregate'"
    ).fetchone()
    assert rows[0] > 0
