"""Params-join wiring vs the duckdb oracle (DRAFT-22 step 3).

The marginalizer's serving_sql joins params tables with
`IS NOT DISTINCT FROM` keys (NULL joins NULL — one bucket) and, for
keyless params, `LEFT JOIN ... ON ((1 = 1))` against a one-row static.
Differential: same SQL, same data, both engines, row-for-row.
"""

from __future__ import annotations

import pytest
from confit import DuckDBInferFn
from test_duckdb_interpreter import _row_schema, duck_check, static

DIM = static(
    {"id": "int?", "v": "int"},
    [{"id": 1, "v": 10}, {"id": None, "v": 99}],
)


def test_indf_left_join_null_joins_null():
    duck_check(
        "SELECT k, v FROM __THIS__ LEFT JOIN dim ON k IS NOT DISTINCT FROM dim.id",
        {"k": "int?"},
        [{"k": 1}, {"k": None}, {"k": 2}],
        {"dim": DIM},
    )


def test_indf_inner_join_null_joins_null():
    duck_check(
        "SELECT v FROM __THIS__ JOIN dim ON k IS NOT DISTINCT FROM dim.id",
        {"k": "int?"},
        [{"k": 1}, {"k": None}, {"k": 2}],
        {"dim": DIM},
    )


def test_indf_string_key_and_nullable_value():
    # Same column name on both sides (the marginalizer's shape) — every
    # reference qualified, exactly as generated SQL qualifies.
    dim = static(
        {"country": "str?", "est": "int?"},
        [{"country": "de", "est": 0}, {"country": None, "est": 1}],
    )
    duck_check(
        "SELECT t.country, est FROM __THIS__ AS t "
        "LEFT JOIN dim ON t.country IS NOT DISTINCT FROM dim.country",
        {"country": "str?"},
        [{"country": "de"}, {"country": None}, {"country": "fr"}],
        {"dim": dim},
    )


def test_indf_multi_key_conjunction():
    dim = static(
        {"ka": "int?", "kb": "str?", "v": "int"},
        [
            {"ka": 1, "kb": "x", "v": 10},
            {"ka": 1, "kb": None, "v": 11},
            {"ka": None, "kb": None, "v": 12},
        ],
    )
    duck_check(
        "SELECT v FROM __THIS__ LEFT JOIN dim ON "
        "(a IS NOT DISTINCT FROM dim.ka) AND (b IS NOT DISTINCT FROM dim.kb)",
        {"a": "int?", "b": "str?"},
        [
            {"a": 1, "b": "x"},
            {"a": 1, "b": None},
            {"a": None, "b": None},
            {"a": 2, "b": "x"},
        ],
        {"dim": dim},
    )


def test_indf_mixed_with_eq_key():
    dim = static(
        {"ka": "int", "kb": "int?", "v": "int"},
        [{"ka": 1, "kb": 5, "v": 10}, {"ka": 1, "kb": None, "v": 11}],
    )
    duck_check(
        "SELECT v FROM __THIS__ LEFT JOIN dim ON "
        "a = dim.ka AND (b IS NOT DISTINCT FROM dim.kb)",
        {"a": "int?", "b": "int?"},
        [
            {"a": 1, "b": 5},
            {"a": 1, "b": None},
            {"a": None, "b": None},  # `=` key: NULL never matches
            {"a": 2, "b": None},
        ],
        {"dim": dim},
    )


def test_keyless_one_row_left_join():
    params = static({"w": "int"}, [{"w": 7}])
    duck_check(
        "SELECT k, w FROM __THIS__ LEFT JOIN p ON ((1 = 1))",
        {"k": "int"},
        [{"k": 1}, {"k": 2}],
        {"p": params},
    )


def test_serving_sql_shape_end_to_end():
    # The marginalizer's exact aliased shape, minus the UDF call (step 2 of
    # DRAFT-22 wires the extern; the join layer is identical).
    params = static(
        {"country": "str?", "__cf_est": "int?"},
        [{"country": "de", "__cf_est": 0}, {"country": None, "__cf_est": 1}],
    )
    duck_check(
        'SELECT (__cf_p0.__cf_est + 1) AS z, __cf_t."name" AS "name" '
        "FROM __THIS__ AS __cf_t "
        "LEFT JOIN __CF_PARAMS_0__ AS __cf_p0 "
        "ON ((__cf_t.country IS NOT DISTINCT FROM __cf_p0.country))",
        {"country": "str?", "name": "str"},
        [
            {"country": "de", "name": "a"},
            {"country": None, "name": "b"},
            {"country": "fr", "name": "c"},
        ],
        {"__CF_PARAMS_0__": params},
    )


def test_chained_indf_and_keyless_joins_under_shape_map():
    params0 = static(
        {"id": "int?", "est": "int?"},
        [{"id": 1, "est": 0}, {"id": None, "est": 1}],
    )
    params1 = static({"w": "int"}, [{"w": 100}])
    schema = _row_schema({"k": "int?"})
    fn = DuckDBInferFn(
        "SELECT (p0.est + p1.w) AS z FROM __THIS__ AS t "
        "LEFT JOIN p0 ON ((t.k IS NOT DISTINCT FROM p0.id)) "
        "LEFT JOIN p1 ON ((1 = 1))",
        row_tables={"__THIS__": schema},
        static_tables={"p0": params0, "p1": params1},
        shape="map",
    )
    got = [r["z"] for r in fn.infer_rows([{"k": k} for k in (1, None, 2)])]
    assert got == [100, 101, None]


def test_keyless_join_multi_row_params_refuses():
    params = static({"w": "int"}, [{"w": 7}, {"w": 8}])
    with pytest.raises(ValueError, match="duplicate map key"):
        DuckDBInferFn(
            "SELECT k, w FROM __THIS__ LEFT JOIN p ON ((1 = 1))",
            row_tables={"__THIS__": _row_schema({"k": "int"})},
            static_tables={"p": params},
        )


def test_indf_duplicate_null_build_keys_refuse():
    dup = static(
        {"id": "int?", "v": "int"},
        [{"id": None, "v": 1}, {"id": None, "v": 2}],
    )
    with pytest.raises(ValueError, match="duplicate map key"):
        DuckDBInferFn(
            "SELECT v FROM __THIS__ LEFT JOIN dim ON k IS NOT DISTINCT FROM dim.id",
            row_tables={"__THIS__": _row_schema({"k": "int?"})},
            static_tables={"dim": dup},
        )


def test_indf_join_on_both_backends(monkeypatch):
    # The default build is cranelift; SPECIALIZER_FORCE_INTERP pins the
    # interpreter — both must agree with DuckDB.
    duck_check(
        "SELECT k, v FROM __THIS__ LEFT JOIN dim ON k IS NOT DISTINCT FROM dim.id",
        {"k": "int?"},
        [{"k": 1}, {"k": None}, {"k": 2}],
        {"dim": DIM},
    )
    monkeypatch.setenv("SPECIALIZER_FORCE_INTERP", "1")
    duck_check(
        "SELECT k, v FROM __THIS__ LEFT JOIN dim ON k IS NOT DISTINCT FROM dim.id",
        {"k": "int?"},
        [{"k": 1}, {"k": None}, {"k": 2}],
        {"dim": DIM},
    )


def test_indf_key_reconstruction_via_star():
    # dim.id resolves via key reconstruction: NULL-bucket hits show NULL,
    # misses show NULL, value hits show the key — exactly DuckDB.
    dim = static(
        {"rid": "int?", "v": "int"}, [{"rid": 1, "v": 10}, {"rid": None, "v": 99}]
    )
    duck_check(
        "SELECT * FROM __THIS__ LEFT JOIN dim ON k IS NOT DISTINCT FROM dim.rid",
        {"k": "int?"},
        [{"k": 1}, {"k": None}, {"k": 2}],
        {"dim": dim},
    )
