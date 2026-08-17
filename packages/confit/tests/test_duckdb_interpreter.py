"""DuckDBInferFn vs duckdb-python: the specializer's differential oracle.

`duck_check` runs the same SQL on the same data through both engines and
asserts the outputs agree row-for-row. Stretch-3 surface: the projection /
WHERE spine plus equi-joins to static tables (INNER and LEFT), which lower
to map probes.
"""

from __future__ import annotations

import math
from typing import Any

import duckdb
import pyarrow as pa
import pytest
from confit import DuckDBInferFn

_ARROW = {
    "int": pa.int64(),
    "float": pa.float64(),
    "str": pa.string(),
    "bool": pa.bool_(),
}


def _row_schema(schema: dict[str, str]) -> pa.Schema:
    return pa.schema(
        pa.field(n, _ARROW[s.rstrip("?")], nullable=s.endswith("?"))
        for n, s in schema.items()
    )


def static(schema: dict[str, str], rows: list[dict[str, Any]]) -> pa.Table:
    arrow = pa.schema(
        pa.field(n, _ARROW[s.rstrip("?")], nullable=s.endswith("?"))
        for n, s in schema.items()
    )
    return pa.Table.from_pylist(rows, schema=arrow)


def _norm(spec: str, v: Any) -> Any:
    # Pydantic used to coerce an int literal into a float field automatically;
    # the arrow boundary refuses that coercion, so replicate it here at the
    # test-data edge instead of rewriting every literal in the corpus.
    if spec.rstrip("?") == "float" and isinstance(v, int) and not isinstance(v, bool):
        return float(v)
    return v


def duck_check(
    sql: str,
    row_schema: dict[str, str],
    row_rows: list[dict[str, Any]],
    statics: dict[str, pa.Table] | None = None,
) -> None:
    statics = statics or {}
    schema = _row_schema(row_schema)
    fn = DuckDBInferFn(sql, row_tables={"__THIS__": schema}, static_tables=statics)
    inputs = [{k: _norm(row_schema[k], r.get(k)) for k in row_schema} for r in row_rows]
    got = fn.infer_rows(inputs)

    con = duckdb.connect()
    # Materialize NATIVE tables: duckdb pushes constant filters into
    # registered-arrow scans with IEEE NaN semantics, which disagrees with
    # its own native-table comparison order (adversarial probe, 2026-07-26).
    # The engine follows native-table semantics — the corpus's world.
    for name, table in statics.items():
        con.register(f"__arrow_{name}", table)
        con.execute(f'CREATE TABLE "{name}" AS SELECT * FROM "__arrow_{name}"')
    con.register("__arrow_this", static(row_schema, row_rows))
    con.execute("CREATE TABLE __THIS__ AS SELECT * FROM __arrow_this")
    want = con.execute(sql).to_arrow_table().to_pylist()

    # Row order is not part of the contract (a join may reorder); compare as
    # multisets of repr'd rows — repr keeps value types apart (1 vs '1' vs
    # 1.0) and makes NaN compare equal to itself.
    key = lambda r: sorted((k, repr(v)) for k, v in r.items())  # noqa: E731
    assert sorted(map(key, got)) == sorted(map(key, want)), f"{got} != {want}"


DIM = static(
    {"id": "int", "name": "str"},
    [{"id": 1, "name": "one"}, {"id": 3, "name": "three"}],
)


def test_projection_and_where_differential():
    duck_check(
        "SELECT a + 1 AS x, a / 2 AS h FROM __THIS__ WHERE a > 0",
        {"a": "int"},
        [{"a": 4}, {"a": -1}, {"a": 7}],
    )


def test_inner_join_hits_and_misses():
    duck_check(
        "SELECT k, name FROM __THIS__ JOIN dim ON k = dim.id",
        {"k": "int"},
        [{"k": 1}, {"k": 2}, {"k": 3}],
        {"dim": DIM},
    )


def test_left_join_miss_and_null_key():
    duck_check(
        "SELECT k, name FROM __THIS__ LEFT JOIN dim ON k = dim.id",
        {"k": "int?"},
        [{"k": 1}, {"k": 2}, {"k": None}],
        {"dim": DIM},
    )


def test_inner_join_null_key_drops():
    duck_check(
        "SELECT name FROM __THIS__ JOIN dim ON k = dim.id",
        {"k": "int?"},
        [{"k": 1}, {"k": None}],
        {"dim": DIM},
    )


def test_join_key_promotion_int_row_against_float_col():
    duck_check(
        "SELECT v FROM __THIS__ JOIN dim ON k = dim.id",
        {"k": "int"},
        [{"k": 1}, {"k": 2}],
        {"dim": static({"id": "float", "v": "int"}, [{"id": 1.0, "v": 10}])},
    )


def test_join_key_expression_and_where_interaction():
    duck_check(
        "SELECT k, name FROM __THIS__ JOIN dim ON k + 1 = dim.id WHERE k >= 0",
        {"k": "int"},
        [{"k": 0}, {"k": 2}, {"k": -5}],
        {"dim": DIM},
    )


def test_duplicate_build_keys_error():
    dup = static({"id": "int", "v": "int"}, [{"id": 1, "v": 10}, {"id": 1, "v": 11}])
    with pytest.raises(ValueError, match="duplicate map key"):
        DuckDBInferFn(
            "SELECT v FROM __THIS__ JOIN dim ON k = dim.id",
            row_tables={"__THIS__": _row_schema({"k": "int"})},
            static_tables={"dim": dup},
        )


def test_null_in_value_column_serves():
    # TASK-55: NULL VALUES flow through joins as NULL (validity+payload
    # pairs in the map); only NULL KEYS keep the drop rule.
    holed = static({"id": "int", "v": "int?"}, [{"id": 1, "v": None}])
    duck_check(
        "SELECT v FROM __THIS__ JOIN dim ON k = dim.id",
        {"k": "int"},
        [{"k": 1}, {"k": 2}],
        {"dim": holed},
    )


def test_null_key_build_rows_are_dropped():
    # A NULL build key never equi-matches, so the row is dropped rather than
    # rejected — probing k=1 still hits the valid entry.
    holed = static(
        {"id": "int?", "v": "int"}, [{"id": 1, "v": 10}, {"id": None, "v": 99}]
    )
    duck_check(
        "SELECT v FROM __THIS__ JOIN dim ON k = dim.id",
        {"k": "int"},
        [{"k": 1}],
        {"dim": holed},
    )


def test_unsupported_is_a_clean_value_error():
    with pytest.raises(ValueError, match="unsupported.*GROUP BY"):
        DuckDBInferFn(
            "SELECT a FROM __THIS__ GROUP BY a",
            row_tables={"__THIS__": _row_schema({"a": "int"})},
            static_tables={},
        )


def test_bad_sql_is_a_build_error():
    with pytest.raises(ValueError, match="parse error"):
        DuckDBInferFn(
            "SELECT FROM",
            row_tables={"__THIS__": _row_schema({"a": "int"})},
            static_tables={},
        )


def test_unknown_infer_table_is_rejected():
    # MIGRATION-NOTE: infer(tables={...}) named-table dispatch is deleted;
    # infer_rows(rows) takes no table-name argument at all, so the runtime
    # "wrong table key" error this test used to pin can no longer be
    # constructed. The remaining "unknown table" refusals are build-time
    # (SQL referencing an undeclared table name) and stay covered by
    # test_alias_shadows_original_name and
    # test_unknown_driving_table_stays_clean_unsupported.
    schema = _row_schema({"a": "int"})
    fn = DuckDBInferFn(
        "SELECT a FROM __THIS__", row_tables={"__THIS__": schema}, static_tables={}
    )
    assert fn.infer_rows([{"a": 1}]) == [{"a": 1}]


def test_output_schema_is_synthesized():
    # MIGRATION-NOTE: output_model= and the synthesized pydantic model are
    # deleted; output_schema (a pa.Schema) is the replacement contract. The
    # old assertions on pydantic field annotations have no equivalent, so
    # this checks the same shape (names + types) through the new surface.
    fn = DuckDBInferFn(
        "SELECT a + 1 AS x, b AS y FROM __THIS__",
        row_tables={"__THIS__": _row_schema({"a": "int", "b": "float?"})},
        static_tables={},
    )
    assert fn.output_schema.names == ["x", "y"]
    assert fn.output_schema.field("x").type == pa.int64()
    assert fn.output_schema.field("y").type == pa.float64()


# ------------------------------------------------------------- stretch 4:
# builtin catalogue, differential vs duckdb per the measured pins.


def test_string_builtins_differential():
    duck_check(
        "SELECT upper(s) AS u, lower(s) AS l, trim(s) AS t, ltrim(s, 'a') AS lt, "
        "rtrim(s) AS rt, substr(s, 2, 3) AS sub FROM __THIS__",
        {"s": "str?"},
        [{"s": "  aBc  "}, {"s": "abcdef"}, {"s": ""}, {"s": None}],
    )


def test_substr_edges_differential():
    duck_check(
        "SELECT substr(s, 0, 3) AS a, substr(s, -2) AS b, substr(s, -10, 8) AS c, "
        "substr(s, 1, 0) AS d, substr(s, 9) AS e FROM __THIS__",
        {"s": "str"},
        [{"s": "hello"}, {"s": "x"}],
    )


def test_concat_and_pipes_differential():
    duck_check(
        "SELECT n || '!' AS a, 'v=' || n AS b, concat(s, n, 'z') AS c, "
        "concat(s) AS d FROM __THIS__",
        {"n": "int?", "s": "str?"},
        [{"n": 1, "s": "a"}, {"n": None, "s": None}, {"n": -3, "s": ""}],
    )


def test_abs_round_differential():
    duck_check(
        "SELECT abs(n) AS an, abs(x) AS ax, round(x) AS rx, round(n) AS rn "
        "FROM __THIS__",
        {"n": "int?", "x": "float?"},
        [
            {"n": -5, "x": -2.5},
            {"n": 3, "x": 2.5},
            {"n": None, "x": None},
            {"n": 0, "x": -0.4},
        ],
    )


def test_rem_by_zero_differential():
    duck_check(
        "SELECT a % b AS r FROM __THIS__",
        {"a": "int", "b": "int"},
        [{"a": 5, "b": 0}, {"a": 5, "b": 3}, {"a": -7, "b": 2}],
    )


def test_float_rem_differential():
    duck_check(
        "SELECT x % y AS r FROM __THIS__",
        {"x": "float", "y": "float"},
        [{"x": -5.5, "y": 2.5}, {"x": 7.0, "y": 4.0}],
    )


def test_coalesce_nullif_differential():
    duck_check(
        "SELECT coalesce(n, 9) AS a, coalesce(NULL, n, 9) AS b, "
        "nullif(n, 3) AS c, nullif(s, 'x') AS d FROM __THIS__",
        {"n": "int?", "s": "str?"},
        [{"n": 3, "s": "x"}, {"n": None, "s": "y"}, {"n": 7, "s": None}],
    )


def test_nan_comparison_differential():
    # DuckDB DOUBLE order: NaN = NaN keeps the row.
    duck_check(
        "SELECT x FROM __THIS__ WHERE x = x",
        {"x": "float"},
        [{"x": float("nan")}, {"x": 1.5}],
    )


def test_simple_case_mapping_matches_duckdb():
    # Formerly a strict xfail: DuckDB uses utf8proc SIMPLE case maps, Rust
    # std only has full maps. src/specializer/exec/casemap.rs now carries the
    # measured exception table (see scripts/gen_casemap.py for why it exists
    # and why it is dependency-free).
    duck_check(
        "SELECT upper(s) AS u, lower(s) AS l FROM __THIS__",
        {"s": "str"},
        [{"s": "ß"}, {"s": "İ"}, {"s": "ᾀ"}, {"s": "ƛ"}],
    )


def test_simple_case_mapping_full_codepoint_census():
    # THE authority on casemap.rs: every Unicode scalar value, chunked into
    # long strings (per-codepoint mapping makes one string test them all),
    # through both engines. If a duckdb bump shifts utf8proc's tables, this
    # fails and scripts/gen_casemap.py regenerates the exception table.
    step = 0x8000
    rows = []
    for lo in range(1, 0x110000, step):
        s = "".join(
            chr(c)
            for c in range(lo, min(lo + step, 0x110000))
            if not (0xD800 <= c <= 0xDFFF)
        )
        if s:
            rows.append({"s": s})
    duck_check("SELECT upper(s) AS u, lower(s) AS l FROM __THIS__", {"s": "str"}, rows)


# --------------------------------------------- adversarial-fleet fixes:
# each case below was a measured divergence, now pinned differentially.


def test_trim_zs_set_differential():
    duck_check(
        "SELECT trim(s) AS t, ltrim(s) AS l, rtrim(s) AS r FROM __THIS__",
        {"s": "str"},
        [{"s": "\u00a0a\u3000"}, {"s": "\ta\n"}, {"s": " a "}, {"s": "\u2003a"}],
    )


def test_substr_negative_length_differential():
    duck_check(
        "SELECT substr(s, st, ln) AS r FROM __THIS__",
        {"s": "str", "st": "int", "ln": "int"},
        [
            {"s": "hello", "st": 3, "ln": -2},
            {"s": "hello", "st": 6, "ln": -5},
            {"s": "h\u00e9llo", "st": 4, "ln": -10},
            {"s": "hello", "st": 1, "ln": -1},
        ],
    )


def test_substr_negative_start_column_path_differential():
    # The TASK-45-review "parity bug" triples, through the path real queries
    # take (column input -> DuckDB's vectorized substr). builtin-pins \u00a74 read
    # these as "negative start clamps to 1 BEFORE the length window"; that was
    # measured on the one DuckDB path which disagrees with its own other three
    # (see test_substr_constant_fold_divergence below). The rule is the WINDOW:
    # map a negative start end-relative, then intersect [pos, pos+len) with the
    # string -- so ('ab', -4, 2) is '' here, not 'ab'.
    duck_check(
        "SELECT substr(s, st, ln) AS r FROM __THIS__",
        {"s": "str", "st": "int", "ln": "int"},
        [
            {"s": "ab", "st": -4, "ln": 2},
            {"s": "ab", "st": -3, "ln": 2},
            {"s": "ab", "st": -5, "ln": 3},
        ],
    )


# CLOSED 2026-08-17 by the oracle change, and it closed by itself. DuckDB's
# constant-fold substr disagrees with its own OPTIMIZED vectorized path on
# negative starts, and this engine used to pin the vectorized one, so the
# pure-literal spelling diverged (xfail-strict, "known residual, builtin-pins
# spec \u00a74"). Measured across all four paths for `substr('hello', -10, 8)`:
#
#   optimizer ON,  literal args   'hel'
#   optimizer ON,  column args    'hello'   <- the outlier we used to pin
#   optimizer OFF, literal args   'hel'
#   optimizer OFF, column args    'hel'
#
# The window rule the engine now implements agrees with three of the four,
# including DuckDB's own constant folder, so the literal and column spellings
# finally answer the same thing and the residual is gone.
def test_substr_constant_fold_divergence():
    duck_check(
        "SELECT substr('ab', -4, 2) AS a, substr('ab', -3, 2) AS b, "
        "substr('ab', -5, 3) AS c FROM __THIS__",
        {"s": "str"},
        [{"s": "x"}],
    )


def test_float_rendering_differential():
    duck_check(
        "SELECT x || '' AS s FROM __THIS__",
        {"x": "float"},
        [
            {"x": 1e300},
            {"x": 1e-05},
            {"x": float("nan")},
            {"x": 2.5},
            {"x": 1e16},
            {"x": -1e300},
        ],
    )


def test_null_divisor_differential():
    duck_check(
        "SELECT a % b AS r FROM __THIS__",
        {"a": "int?", "b": "int?"},
        [{"a": 7, "b": None}, {"a": None, "b": 0}, {"a": 7, "b": 3}],
    )


def test_numeric_where_differential():
    duck_check(
        "SELECT a FROM __THIS__ WHERE a % 2",
        {"a": "int"},
        [{"a": 1}, {"a": 2}, {"a": 3}],
    )


def test_nan_filter_differential_on_native_tables():
    duck_check(
        "SELECT x FROM __THIS__ WHERE x > 0",
        {"x": "float?"},
        [{"x": float("nan")}, {"x": 1.0}, {"x": None}, {"x": float("inf")}],
    )


# ------------------------------------------------- static-only queries:
# AC #2 — evaluated once at build time by DuckDB, nothing dynamic remains.


def test_static_only_query_is_a_constant_emitter():
    dim = static(
        {"id": "int", "name": "str"},
        [
            {"id": 1, "name": "one"},
            {"id": 2, "name": "two"},
            {"id": 3, "name": "three"},
        ],
    )
    schema = _row_schema({"a": "int"})
    fn = DuckDBInferFn(
        "SELECT name, id * 10 AS x FROM dim WHERE id <> 2 ORDER BY id DESC",
        row_tables={"__THIS__": schema},
        static_tables={"dim": dim},
    )
    # The result is fixed at build time — constructs like ORDER BY work
    # because DuckDB itself evaluated it. Input rows are not "irrelevant"
    # though: this build cannot READ them, so handing it any is a refusal
    # rather than a silent drop (TASK-110).
    assert fn.infer_rows([]) == [
        {"name": "three", "x": 30},
        {"name": "one", "x": 10},
    ]
    for rows_in in ([{"a": 1}], [{"a": 1}, {"a": 2}]):
        with pytest.raises(ValueError, match="infer_rows"):
            fn.infer_rows(rows_in)


def test_static_only_aggregation_works_via_duckdb():
    dim = static({"v": "int"}, [{"v": 1}, {"v": 2}, {"v": 3}])
    fn = DuckDBInferFn(
        "SELECT sum(v) AS s FROM dim",
        row_tables={"__THIS__": _row_schema({"a": "int"})},
        static_tables={"dim": dim},
    )
    assert fn.infer_rows([]) == [{"s": 6}]


def test_unknown_driving_table_stays_clean_unsupported():
    # Not a static table either -> the original clean unsupported surfaces.
    with pytest.raises(ValueError, match="driving relation"):
        DuckDBInferFn(
            "SELECT x FROM nope",
            row_tables={"__THIS__": _row_schema({"a": "int"})},
            static_tables={},
        )


# --------------------------------------------------------- M-cranelift:
# the codegen backend runs the same suite; these pin the backend choice.


def test_v0_queries_run_on_cranelift():
    fn = DuckDBInferFn(
        "SELECT k + 1 AS x, upper(s) AS u FROM __THIS__ JOIN dim ON k = dim.id",
        row_tables={"__THIS__": _row_schema({"k": "int", "s": "str"})},
        static_tables={"dim": DIM},
    )
    assert fn.backend == "cranelift"


def test_static_only_backend_is_constant():
    fn = DuckDBInferFn(
        "SELECT sum(v) AS s FROM dim",
        row_tables={"__THIS__": _row_schema({"a": "int"})},
        static_tables={"dim": static({"v": "int"}, [{"v": 1}, {"v": 2}])},
    )
    assert fn.backend == "constant"


# --------------------------------------------------------- M-boundary:
# the generated row marshaller. These pin the adversarial-review fixes
# (2026-07-26): the default boundary is the generated marshaller, the
# generic baseline accepts the same inputs, and reentrancy degrades to the
# generic path instead of erroring.


def test_default_boundary_is_the_marshaller():
    # MIGRATION-NOTE: output_model= (and the pydantic-validate semantics it
    # pinned — coercion, defaults, field_validator) is deleted along with the
    # pydantic surface; dict-out has no per-field custom-validation hook.
    # The generated marshaller is now simply the unconditional default
    # boundary (see test_generic_boundary_accepts_dict_rows for the only
    # other boundary, pinned by the SPECIALIZER_GENERIC_BOUNDARY env var).
    fn = DuckDBInferFn(
        "SELECT k + 1 AS x FROM __THIS__",
        row_tables={"__THIS__": _row_schema({"k": "int"})},
        static_tables={},
    )
    assert fn.boundary == "marshaller"
    assert fn.infer_rows([{"k": 41}]) == [{"x": 42}]


def test_generic_boundary_accepts_dict_rows(monkeypatch):
    monkeypatch.setenv("SPECIALIZER_GENERIC_BOUNDARY", "1")
    fn = DuckDBInferFn(
        "SELECT k * 2 AS d FROM __THIS__",
        row_tables={"__THIS__": _row_schema({"k": "int"})},
        static_tables={},
    )
    assert fn.boundary == "generic"
    assert fn.infer_rows([{"k": 21}])[0]["d"] == 42


def test_reentrant_infer_falls_back_instead_of_erroring():
    fn = DuckDBInferFn(
        "SELECT k * 2 AS d FROM __THIS__",
        row_tables={"__THIS__": _row_schema({"k": "int"})},
        static_tables={},
    )
    inner: list = []

    class Row:
        @property
        def k(self):
            if not inner:
                inner.append(fn.infer_rows([{"k": 5}])[0]["d"])
            return 7

    (m,) = fn.infer_rows([Row()])
    assert m["d"] == 14
    assert inner == [10]  # the nested call completed via the generic path


# --------------------------------------------------------- TASK-46:
# SELECT * star expansion — measured DuckDB 1.5.5 pins (order, EXCLUDE
# case-folding, mixed items) verified end-to-end against the oracle.


def test_star_expansion_matches_duckdb():
    rows = [
        {"a": 1, "b": 2.5, "s": "x"},
        {"a": 2, "b": None, "s": "y"},
    ]
    for sql in [
        "SELECT * FROM __THIS__",
        "SELECT __THIS__.* FROM __THIS__",
        "SELECT * EXCLUDE (b) FROM __THIS__",
        "SELECT * EXCLUDE B FROM __THIS__",  # case-insensitive, bare form
        "SELECT *, a * 2 AS a2 FROM __THIS__",
        "SELECT * EXCLUDE (s), upper(s) AS u FROM __THIS__",
    ]:
        duck_check(sql, {"a": "int", "b": "float?", "s": "str"}, rows)


def test_star_qualified_over_row_table_under_join():
    duck_check(
        "SELECT __THIS__.*, dim.name AS nm FROM __THIS__ JOIN dim ON k = dim.id",
        {"k": "int"},
        [{"k": 1}, {"k": 2}, {"k": 99}],
        statics={"dim": DIM},
    )


def test_star_over_joined_table_expands():
    # Wave-4: joined-table stars expand, the key column reconstructed from
    # the dynamic side — oracle-checked end to end (INNER and LEFT).
    for join in ["JOIN", "LEFT JOIN"]:
        duck_check(
            f"SELECT * FROM __THIS__ {join} dim ON k = dim.id",
            {"k": "int"},
            [{"k": 1}, {"k": 2}, {"k": 3}],
            {"dim": DIM},
        )


# --------------------------------------------------------- TASK-47:
# BETWEEN / IN desugars — measured DuckDB 1.5.5 truth tables (wave-1 pins).
nan, inf = float("nan"), float("inf")


def test_between_three_valued_int():
    duck_check(
        "SELECT x BETWEEN lo AND hi AS b, x NOT BETWEEN lo AND hi AS nb FROM __THIS__",
        {"x": "int?", "lo": "int?", "hi": "int?"},
        [
            {"x": 5, "lo": 1, "hi": 10},  # True
            {"x": 5, "lo": 10, "hi": 1},  # empty range -> False
            {"x": None, "lo": 1, "hi": 10},  # NULL
            {"x": None, "lo": 10, "hi": 1},  # NULL x + empty range -> still NULL
            {"x": 5, "lo": None, "hi": 10},  # NULL
            {"x": 5, "lo": None, "hi": 4},  # False — failing half decides
            {"x": 5, "lo": 1, "hi": None},  # NULL
            {"x": 5, "lo": 6, "hi": None},  # False
            {"x": None, "lo": None, "hi": None},  # NULL
            {"x": 5, "lo": 5, "hi": 5},  # inclusive -> True
        ],
    )


def test_between_int_x_double_bounds_unifies_to_double():
    duck_check(
        "SELECT x BETWEEN lo AND hi AS b FROM __THIS__",
        {"x": "int", "lo": "float", "hi": "float"},
        [
            {"x": 5, "lo": 0.5, "hi": 5.5},
            {"x": 5, "lo": 5.1, "hi": 9.9},
            # int->double is lossy at 2^53 — DuckDB does the same cast: True
            {"x": 9007199254740993, "lo": 9007199254740992.0, "hi": 9007199254740992.0},
        ],
    )


def test_between_duck_double_order_specials():
    duck_check(
        "SELECT x BETWEEN lo AND hi AS b, x NOT BETWEEN lo AND hi AS nb FROM __THIS__",
        {"x": "float?", "lo": "float?", "hi": "float?"},
        [
            {"x": nan, "lo": 1.0, "hi": 2.0},  # False (nan<=2 False)
            {"x": 5.0, "lo": 1.0, "hi": nan},  # True (nan above everything)
            {"x": 5.0, "lo": nan, "hi": 10.0},  # False
            {"x": nan, "lo": nan, "hi": nan},  # True (nan = nan)
            {"x": inf, "lo": 1.0, "hi": nan},  # True (nan above inf)
            {"x": nan, "lo": None, "hi": 1.0},  # False (NULL AND False)
            {"x": nan, "lo": 1.0, "hi": None},  # NULL (True AND NULL)
            {"x": 0.0, "lo": -0.0, "hi": -0.0},  # True (zeros equal)
            {"x": -0.0, "lo": 0.0, "hi": 0.0},  # True
            {"x": -inf, "lo": -inf, "hi": inf},  # True
        ],
    )


def test_between_strings_byte_order():
    duck_check(
        "SELECT x BETWEEN lo AND hi AS b FROM __THIS__",
        {"x": "str", "lo": "str", "hi": "str"},
        [
            {"x": "b", "lo": "a", "hi": "c"},  # True
            {"x": "B", "lo": "a", "hi": "c"},  # False (0x42 < 0x61)
            {"x": "Z", "lo": "a", "hi": "z"},  # False
            {"x": "é", "lo": "a", "hi": "z"},  # False (0xC3A9 > 'z')
            {"x": "abc", "lo": "ab", "hi": "abd"},  # True
            {"x": "", "lo": "", "hi": "a"},  # True
            {"x": "a", "lo": "c", "hi": "a"},  # False
        ],
    )


def test_in_null_element_truth_table():
    duck_check(
        "SELECT x IN (1, NULL) AS i, x NOT IN (1, NULL) AS ni,"
        " x IN (1, 2) AS i2, x NOT IN (1, 2) AS ni2, x IN (NULL) AS io FROM __THIS__",
        {"x": "int?"},
        [{"x": 1}, {"x": 3}, {"x": None}],
        # i: T/N/N  ni: F/N/N  i2: T/F/N  ni2: F/T/N  io: N/N/N
    )


def test_in_column_elements_is_the_or_chain():
    duck_check(
        "SELECT x IN (y, z) AS r, x NOT IN (y, z) AS nr FROM __THIS__",
        {"x": "int?", "y": "int?", "z": "int?"},
        [
            {"x": 1, "y": 1, "z": None},  # True / False
            {"x": 3, "y": 1, "z": None},  # NULL / NULL
            {"x": 3, "y": 1, "z": 2},  # False / True
            {"x": None, "y": 1, "z": 2},  # NULL / NULL
        ],
    )


def test_in_int_col_float_literals_small_values():
    duck_check(
        "SELECT x IN (1.0) AS a, x IN (0.5, 1.0) AS b, x IN (1.5) AS c FROM __THIS__",
        {"x": "int?"},
        [{"x": 1}, {"x": 2}, {"x": 3}, {"x": None}],  # 1.5 never rounds to 2
    )


def test_in_nan_and_signed_zero():
    duck_check(
        "SELECT x IN ('NaN'::DOUBLE) AS n, x IN (1.0, 'NaN'::DOUBLE) AS m,"
        " x IN (0.0) AS z, x NOT IN ('NaN'::DOUBLE) AS nn FROM __THIS__",
        {"x": "float"},
        [{"x": nan}, {"x": 1.0}, {"x": -0.0}],  # NaN=NaN True; -0.0 IN (0.0) True
    )


def test_in_strings_and_bools():
    duck_check(
        "SELECT s IN ('a', 'b') AS r, s IN ('b', NULL) AS rn FROM __THIS__",
        {"s": "str?"},
        [{"s": "a"}, {"s": "A"}, {"s": None}],
    )
    # BOOLEAN comparison stays the engine's pre-existing clean limit, so
    # bool IN/BETWEEN desugar into it and reject by name.
    with pytest.raises(ValueError, match="comparison on BOOLEAN"):
        DuckDBInferFn(
            "SELECT p IN (true) AS r FROM __THIS__",
            row_tables={"__THIS__": _row_schema({"p": "bool?"})},
            static_tables={},
        )


# --------------------------------------------------------- TASK-47:
# wave-1 math builtins - measured DuckDB 1.5.5 pins as oracle tests.


def test_logexp_basic_values():
    duck_check(
        "SELECT ln(x) AS a, log(x) AS b, log2(x) AS c, log10(x) AS d,"
        " exp(x) AS e FROM __THIS__",
        {"x": "float"},
        [
            {"x": v}
            for v in [
                1.0,
                2.718281828459045,
                2.0,
                10.0,
                0.5,
                1000.0,
                1e308,
                5e-324,
                709.782712893384,
            ]
        ],
    )


def test_logexp_nan_inf_pass_through():
    # NaN and +inf do NOT trip the domain checks; only <=0 does.
    duck_check(
        "SELECT ln(x) AS a, log(x) AS b, log2(x) AS c, log10(x) AS d,"
        " exp(x) AS e FROM __THIS__",
        {"x": "float"},
        [{"x": float("nan")}, {"x": float("inf")}],
    )


def test_exp_never_errors():
    # overflow -> inf, underflow -> denormal then +0.0; -inf -> +0.0;
    # boundary 709.782712893384 -> 1.7976931348622732e+308
    duck_check(
        "SELECT exp(x) AS e FROM __THIS__",
        {"x": "float"},
        [
            {"x": v}
            for v in [
                float("-inf"),
                -1000.0,
                -746.0,
                -745.0,
                -744.4400719213812,
                0.0,
                -0.0,
                709.782712893384,
                709.7827128933841,
                710.0,
                1000.0,
            ]
        ],
    )


def test_logexp_null_propagation():
    duck_check(
        "SELECT ln(x) AS a, log(x) AS b, log2(x) AS c, log10(x) AS d, exp(x) AS e,"
        " log(x, 2.0) AS f, log(2.0, x) AS g FROM __THIS__",
        {"x": "float?"},
        [{"x": None}, {"x": 4.0}],
    )


def test_log_two_arg_is_log10_ratio():
    # log(b,x) = log10(x)/log10(b) bit-exact. Discriminators: (10,1000)->3.0 (ln-ratio
    # gives 2.9999999999999996), (e,10)->2.302585092994046 (log2-ratio differs),
    # (e,5e-324)->-744.4400719213813 != ln(5e-324). (0.5,1.0)->-0.0 and (inf,0.5)->-0.0
    # pin sign-of-zero; (inf,inf)->nan.
    duck_check(
        "SELECT log(b, x) AS r FROM __THIS__",
        {"b": "float", "x": "float"},
        [
            {"b": b, "x": x}
            for b, x in [
                (2.0, 8.0),
                (10.0, 1000.0),
                (3.0, 7.0),
                (0.5, 8.0),
                (0.5, 1.0),
                (2.0, 1.0),
                (2.718281828459045, 5e-324),
                (2.718281828459045, 10.0),
                (1.0000000000000002, 4.0),
                (0.9999999999999999, 4.0),
                (5e-324, 8.0),
                (1e308, 8.0),
                (float("nan"), 8.0),
                (2.0, float("nan")),
                (float("nan"), float("nan")),
                (float("inf"), 8.0),
                (float("inf"), 0.5),
                (float("inf"), float("inf")),
                (float("inf"), 1.0),
                (2.0, float("inf")),
                (2.0, 5e-324),
                (2.0, 1e308),
            ]
        ],
    )


def test_log_two_arg_null_preempts_domain_errors():
    # NULL in either slot wins over every domain error, including base 0 / negative / 1.
    duck_check(
        "SELECT log(b, x) AS r FROM __THIS__",
        {"b": "float?", "x": "float?"},
        [
            {"b": None, "x": 8.0},
            {"b": 2.0, "x": None},
            {"b": None, "x": None},
            {"b": None, "x": -4.0},
            {"b": 0.0, "x": None},
            {"b": -2.0, "x": None},
            {"b": 1.0, "x": None},
        ],
    )


def test_logexp_integer_input_yields_double():
    duck_check(
        "SELECT ln(x) AS a, log(x) AS b, log2(x) AS c, log10(x) AS d, exp(x) AS e,"
        " log(x, 8) AS f FROM __THIS__",
        {"x": "int"},
        [{"x": 2}, {"x": 9223372036854775807}],
    )


def test_logexp_guarded_rows_do_not_trap():
    # A bad row aborts an unguarded query, but WHERE / CASE genuinely
    # prevent evaluation.
    duck_check(
        "SELECT CASE WHEN x > 0 THEN ln(x) ELSE NULL END AS a FROM __THIS__",
        {"x": "float"},
        [{"x": 1.0}, {"x": -1.0}, {"x": 4.0}, {"x": 0.0}],
    )
    duck_check(
        "SELECT ln(x) AS a FROM __THIS__ WHERE x > 0",
        {"x": "float"},
        [{"x": 1.0}, {"x": -1.0}, {"x": 4.0}, {"x": 0.0}],
    )


def test_pow_bigint_inputs_yield_double():
    # ints promote to DOUBLE before compute: i64::MAX rounds to 2^63,
    # 2^53+1 loses the +1, 3^40 is the rounded double not the exact int.
    rows = [
        (0, 0),
        (2, 10),
        (0, -1),
        (-8, 2),
        (10, 18),
        (9223372036854775807, 1),
        (-9223372036854775808, 1),
        (2, 63),
        (2, 64),
        (3, 40),
        (9007199254740993, 1),
        (5, 2),
    ]
    duck_check(
        "SELECT pow(x, y) AS p FROM __THIS__",
        {"x": "int", "y": "int"},
        [{"x": x, "y": y} for x, y in rows],
    )


def test_sqrt_non_negative_domain():
    # NOTE: any x < 0 (incl -inf) raises OutOfRange; keep those out of duck_check.
    xs = [4.0, 2.0, 0.0, -0.0, INF, NAN, None, 5e-324, 1e308]
    duck_check(
        "SELECT sqrt(x) AS s FROM __THIS__", {"x": "float?"}, [{"x": x} for x in xs]
    )


def duck_check_ulp(sql, row_schema, row_rows, max_ulp=1):
    """duck_check with a float ulp tolerance, POSITIONAL compare (no joins).

    Exists for exactly one reason so far: DuckDB's own wheels disagree with
    each other on cbrt by one ulp across platforms (Windows wheel matches
    Rust/ucrt bit-exactly; the Linux wheel's bundled std::cbrt is one ulp
    off on e.g. cbrt(27)) — CI-discovered 2026-07-26. The oracle itself is
    platform-inconsistent here, so repr-exact parity is unpinnable.
    """
    import math
    import struct

    schema = _row_schema(row_schema)
    fn = DuckDBInferFn(sql, row_tables={"__THIS__": schema}, static_tables={})
    inputs = [{k: _norm(row_schema[k], r.get(k)) for k in row_schema} for r in row_rows]
    got = fn.infer_rows(inputs)

    con = duckdb.connect()
    con.register("__arrow_this", static(row_schema, row_rows))
    con.execute("CREATE TABLE __THIS__ AS SELECT * FROM __arrow_this")
    want = con.execute(sql).to_arrow_table().to_pylist()

    def close(a, b):
        if isinstance(a, float) and isinstance(b, float):
            if math.isnan(a) and math.isnan(b):
                return True
            ia = struct.unpack("<q", struct.pack("<d", a))[0]
            ib = struct.unpack("<q", struct.pack("<d", b))[0]
            return abs(ia - ib) <= max_ulp
        return repr(a) == repr(b)

    assert len(got) == len(want)
    for g, w in zip(got, want, strict=True):
        for k in g:
            assert close(g[k], w[k]), f"{k}: {g[k]!r} vs {w[k]!r} (> {max_ulp} ulp)"


def test_sqrt_cbrt_bigint():
    xs = [4, 0, 27, 9223372036854775807, 9007199254740992, 9007199254740993, None]
    duck_check_ulp(
        "SELECT sqrt(x) AS s, cbrt(x) AS c FROM __THIS__",
        {"x": "int?"},
        [{"x": x} for x in xs],
    )


def test_cbrt_total_function():
    # cbrt has NO domain restriction: negatives, -inf, -0.0 all flow through.
    # ulp-tolerant: the oracle's own wheels disagree cross-platform on cbrt.
    xs = [8.0, -8.0, 27.0, 2.0, 0.0, -0.0, INF, -INF, NAN, None, 5e-324, -1.0]
    duck_check_ulp(
        "SELECT cbrt(x) AS c FROM __THIS__", {"x": "float?"}, [{"x": x} for x in xs]
    )


def test_sqrt_where_guard_is_lazy():
    # Filtered-out negative rows must NOT trap.
    duck_check(
        "SELECT sqrt(x) AS s FROM __THIS__ WHERE x >= 0",
        {"x": "float"},
        [{"x": 4.0}, {"x": -1.0}],
    )
    duck_check(
        "SELECT CASE WHEN x >= 0 THEN sqrt(x) ELSE NULL END AS s FROM __THIS__",
        {"x": "float"},
        [{"x": 4.0}, {"x": -1.0}],
    )


def test_sqrt_negative_is_runtime_error():
    # Not expressible as a duck_check equality: both engines must RAISE.
    # DuckDB: _duckdb.OutOfRangeException
    #   'Out of Range Error: cannot take square root of a negative number'
    with pytest.raises(Exception, match="cannot take square root of a negative number"):
        duck_check("SELECT sqrt(x) AS s FROM __THIS__", {"x": "float"}, [{"x": -1.0}])
    with pytest.raises(Exception, match="cannot take square root of a negative number"):
        duck_check(
            "SELECT sqrt(x) AS s FROM __THIS__", {"x": "float"}, [{"x": float("-inf")}]
        )


def test_trig_double_column_edges():
    duck_check(
        "SELECT sin(x) AS s, cos(x) AS c, tan(x) AS t FROM __THIS__",
        {"x": "float"},
        [{"x": v} for v in TRIG_EDGE_FLOATS],
    )  # repr-compare keeps -0.0 vs 0.0 apart: the sin/tan(-0.0) sign pins hold


def test_trig_null_propagation():
    duck_check(
        "SELECT sin(x) AS s, cos(x) AS c, tan(x) AS t FROM __THIS__",
        {"x": "float?"},
        [{"x": None}, {"x": 1.0}],
    )


def test_trig_integer_input_yields_double():
    duck_check(
        "SELECT sin(x) AS s, cos(x) AS c, tan(x) AS t FROM __THIS__",
        {"x": "int"},
        [
            {"x": 0},
            {"x": 1},
            {"x": -1},
            {"x": 9223372036854775807},  # rounds to 2^63 as f64 BEFORE sin
            {"x": -9223372036854775808},
        ],
    )


def test_trig_integer_null():
    duck_check(
        "SELECT sin(x) AS s FROM __THIS__",
        {"x": "int?"},
        [{"x": None}, {"x": 3}],
    )


def test_pi_literal_and_vectorized():
    duck_check(
        "SELECT pi() AS p, x + pi() AS xp, sin(pi() / 2) AS one, "
        "cos(pi()) AS neg_one, tan(pi() / 2) AS big, sin(-pi() / 2) AS neg_one_s "
        "FROM __THIS__",
        {"x": "float"},
        [{"x": 0.0}, {"x": 1.0}],
    )


# NOT expressible via duck_check (it asserts success on both sides) — engine needs
# dedicated tests for these, asserting against a plain duckdb replay:
#   1. sin/cos/tan(+-inf) must raise a per-row runtime error whose message mirrors
#      "Out of Range Error: input value inf is out of range for numeric function"
#      (and "... value -inf ..."); an inf row excluded by WHERE must NOT raise.
#   2. NaN *payload* preservation (e.g. input bits 0x7ff0deadbeef0001 out unchanged):
#      repr collapses every NaN to 'nan', so compare struct.pack bits directly.
def test_round_family_single_arg_double():
    vals = [
        2.5,
        3.5,
        -2.5,
        -3.5,
        0.5,
        -0.5,
        1.5,
        0.49999999999999994,
        -0.49999999999999994,
        2.675,
        -0.4,
        -0.04,
        0.0,
        -0.0,
        5e-324,
        -5e-324,
        4503599627370495.5,
        -4503599627370495.5,
        9.3e18,
        -9.3e18,
        1.7976931348623157e308,
        12345.678901234567,
        NAN,
        INF,
        -INF,
        None,
    ]
    duck_check(
        "SELECT floor(x) AS f, ceil(x) AS c, ceiling(x) AS c2,"
        " trunc(x) AS t, round(x) AS r FROM __THIS__",
        {"x": "float?"},
        [{"x": v} for v in vals],
    )


def test_floor_ceil_int64_go_through_double():
    rows = [
        {"i": v}
        for v in [7, 9007199254740993, 9223372036854775807, -9223372036854775808, None]
    ]
    duck_check(
        "SELECT floor(i) AS f, ceil(i) AS c, ceiling(i) AS c2 FROM __THIS__",
        {"i": "int?"},
        rows,
    )


# Shared constants for the wave-1 math oracle tests.

NAN = float("nan")
INF = float("inf")

POW_SPECIALS = [
    (0.0, 0.0),
    (0.0, -1.0),
    (-0.0, -1.0),
    (0.0, -2.0),
    (-0.0, -2.0),
    (-0.0, 3.0),
    (-0.0, 2.0),
    (-0.0, 0.5),
    (-8.0, 1.0 / 3.0),
    (-2.0, 0.5),
    (-2.0, -0.5),
    (NAN, 0.0),
    (NAN, -0.0),
    (1.0, NAN),
    (NAN, NAN),
    (NAN, 1.0),
    (0.0, NAN),
    (INF, 0.0),
    (-INF, 0.0),
    (-1.0, INF),
    (-1.0, -INF),
    (1.0, INF),
    (INF, -1.0),
    (-INF, -1.0),
    (-INF, -2.0),
    (-INF, 3.0),
    (-INF, 2.0),
    (-INF, 0.5),
    (0.0, INF),
    (-0.0, INF),
    (0.0, -INF),
    (-0.0, -INF),
    (2.0, INF),
    (0.5, INF),
    (2.0, -INF),
    (0.5, -INF),
    (-0.5, INF),
    (-2.0, -INF),
    (1e308, 2.0),
    (2.0, 1024.0),
    (2.0, -1075.0),
    (-2.0, -1075.0),
    (2.0, 53.0),
    (10.0, 18.0),
    (5e-324, 0.5),
    (-2.0, 3.0),
    (None, 2.0),
    (2.0, None),
    (None, None),
]

TRIG_EDGE_FLOATS = [
    0.0,
    -0.0,
    1.0,
    -1.0,
    0.5,
    math.pi / 2,
    -math.pi / 2,
    math.pi,
    2 * math.pi,
    1e15,
    float(2**53),
    1e300,
    1.7976931348623157e308,
    5e-324,
    2.2250738585072014e-308,
    9.223372036854776e18,
    float("nan"),
]


def test_pow_double_specials():
    # pow and power are one function; pin the IEEE specials per row. The ^
    # operator is deliberately NOT mapped: sqlparser parses it below *
    # while DuckDB binds it above (measured 2*x^y = 2*(x^y) vs sqlparser's
    # (2*x)^y) — mapping would silently compute the wrong tree. ** does
    # not parse at all. Both stay clean errors.
    duck_check(
        "SELECT pow(x, y) AS p, power(x, y) AS q FROM __THIS__",
        {"x": "float?", "y": "float?"},
        [{"x": x, "y": y} for x, y in POW_SPECIALS],
    )


def test_pow_operator_rejects_cleanly():
    with pytest.raises(ValueError, match="operator .*precedence differs"):
        DuckDBInferFn(
            "SELECT x ^ 2 AS r FROM __THIS__",
            row_tables={"__THIS__": _row_schema({"x": "float"})},
            static_tables={},
        )


# --------------------------------------------------------- TASK-47:
# wave-1 string search - measured DuckDB 1.5.5 pins as oracle tests.
# string-search family - contract pins measured against DuckDB 1.5.5 (2026-07-26).
# All values below were probed through the vectorized path (table columns);
# literal-fold agreed on every pair (0 divergences).

STRSEARCH_SCHEMA = {"s": "str?", "n": "str?"}
STRSEARCH_ROWS = [
    {"s": "abc", "n": "b"},  # simple hit -> 2
    {"s": "abc", "n": "z"},  # miss -> 0
    {"s": "abc", "n": ""},  # empty needle -> 1 / TRUE
    {"s": "", "n": "a"},  # empty haystack -> 0 / FALSE
    {"s": "", "n": ""},  # empty-in-empty -> 1 / TRUE
    {"s": "abc", "n": "abcd"},  # needle longer -> 0 / FALSE
    {"s": "abcabc", "n": "bc"},  # first occurrence -> 2
    {"s": "aaa", "n": "aa"},  # overlapping -> 1
    {"s": "abc", "n": "B"},  # case-sensitive -> 0
    {"s": "héllo", "n": "l"},  # CODEPOINT position -> 3 (bytes would be 4)
    {"s": "héllo", "n": "é"},  # -> 2
    {"s": "héllo", "n": "él"},  # multibyte needle -> 2
    {"s": "a\U0001f44db", "n": "b"},  # emoji = 1 codepoint -> 3 (bytes would be 6)
    {"s": "\U0001f468‍\U0001f469‍\U0001f467", "n": "\U0001f467"},  # ZWJ -> 5, not 1
    {"s": "ééa", "n": "a"},  # -> 3
    {"s": "xabc", "n": "abc"},  # suffix hit -> 2
    {"s": "abc", "n": "abc"},  # exact -> 1
    {"s": "é", "n": "é"},  # NO normalization -> 0 / FALSE
    {"s": "é", "n": "́"},  # bare combining mark -> 2
    {"s": "abc", "n": None},  # NULL-strict every argument
    {"s": None, "n": "b"},
    {"s": None, "n": None},
    {"s": None, "n": ""},  # NULL haystack + empty needle is still NULL
]


def test_strsearch_predicates():
    for fn in ["contains", "starts_with", "prefix", "ends_with", "suffix"]:
        duck_check(
            f"SELECT {fn}(s, n) AS r FROM __THIS__",
            STRSEARCH_SCHEMA,
            STRSEARCH_ROWS,
        )


def test_strsearch_three_valued_where():
    # NULL predicate rows must be dropped by WHERE, same as FALSE.
    duck_check(
        "SELECT s FROM __THIS__ WHERE contains(s, n)",
        STRSEARCH_SCHEMA,
        STRSEARCH_ROWS,
    )
    duck_check(
        "SELECT s FROM __THIS__ WHERE instr(s, n) > 1",
        STRSEARCH_SCHEMA,
        STRSEARCH_ROWS,
    )


def test_strsearch_positions():
    for fn_expr in ["instr(s, n)", "strpos(s, n)", "position(n IN s)"]:
        duck_check(
            f"SELECT {fn_expr} AS r FROM __THIS__",
            STRSEARCH_SCHEMA,
            STRSEARCH_ROWS,
        )


def test_strsearch_length_family():
    duck_check(
        "SELECT length(s) AS lc, char_length(s) AS cc, strlen(s) AS lb FROM __THIS__",
        {"s": "str?"},
        [
            {"s": "abc"},
            {"s": ""},
            {"s": "héllo"},
            {"s": "a\U0001f44db"},
            {"s": "\U0001f468‍\U0001f469‍\U0001f467"},
            {"s": "é"},
            {"s": None},
        ],
    )


# --------------------------------------------------------- TASK-47:
# least/greatest - NULL-ignoring, first-arg ties, duck order (pins).


def test_least_greatest_null_ignoring_int():
    duck_check(
        "SELECT least(a, b) AS l, greatest(a, b) AS g,"
        " least(a) AS l1, least(a, b, 3, 0) AS ln FROM __THIS__",
        {"a": "int?", "b": "int?"},
        [
            {"a": 1, "b": None},  # l=1 g=1 — NULL ignored
            {"a": None, "b": 2},  # l=2 g=2
            {"a": None, "b": None},  # NULL only when ALL null
            {"a": 1, "b": 2},
            {"a": 9223372036854775807, "b": -9223372036854775808},  # exact i64
        ],
    )


def test_least_greatest_double_nan_and_tie_keeps_first():
    duck_check(
        "SELECT least(a, b) AS l, greatest(a, b) AS g FROM __THIS__",
        {"a": "float?", "b": "float?"},
        [
            {"a": nan, "b": 1.0},  # l=1.0 g=nan (nan above everything)
            {"a": 1.0, "b": nan},
            {"a": nan, "b": inf},  # l=inf g=nan
            {"a": nan, "b": None},  # nan / nan — NULL ignored
            {"a": -inf, "b": inf},
            {"a": -0.0, "b": 0.0},  # BOTH return first arg: l=-0.0 g=-0.0
            {"a": 0.0, "b": -0.0},  # l=0.0 g=0.0
            {"a": nan, "b": nan},
        ],
    )


def test_least_greatest_mixed_int_double_unifies_to_double():
    duck_check(
        "SELECT least(i, d) AS l, greatest(i, d) AS g FROM __THIS__",
        {"i": "int", "d": "float"},
        [
            {"i": 1, "d": 2.5},
            {"i": 3, "d": 2.5},
            # DuckDB casts the i64 to double (lossy): l = 9007199254740992.0
            {"i": 9007199254740993, "d": 9007199254740994.0},
            {"i": 9223372036854775807, "d": 1.0},  # g = 9.223372036854776e+18
        ],
    )


def test_least_greatest_strings_and_bools():
    duck_check(
        "SELECT least(a, b) AS l, greatest(a, b) AS g FROM __THIS__",
        {"a": "str?", "b": "str?"},
        [
            {"a": "b", "b": "a"},
            {"a": "B", "b": "a"},  # least='B' — byte order
            {"a": "é", "b": "z"},  # greatest='é'
            {"a": "", "b": "a"},
            {"a": "a", "b": None},  # NULL ignored
        ],
    )


# --------------------------------------------------------- TASK-47:
# round/trunc with digits - oracle pow10 table semantics (pins).


def test_round_double_with_digits_constant_n():
    vals = [
        2.675,
        0.285,
        1.005,
        0.145,
        1.015,
        2.665,
        0.125,
        0.375,
        1.255,
        0.045,
        1.115,
        1234.5,
        -1234.5,
        1250.0,
        -1250.0,
        -44.0,
        -0.04,
        1.7e308,
        9.9e307,
        5e-324,
        -0.0,
        NAN,
        INF,
        -INF,
        None,
    ]
    duck_check(
        "SELECT round(x, 2) AS r2, round(x, 0) AS r0, round(x, -1) AS rn1,"
        " round(x, -2) AS rn2, round(x, 309) AS rbig, round(x, -308) AS rn308,"
        " round(x, -400) AS rneg, trunc(x, 2) AS t2, trunc(x, -1) AS tn1,"
        " trunc(x, -400) AS tneg FROM __THIS__",
        {"x": "float?"},
        [{"x": v} for v in vals],
    )


def test_round_double_pow_table_ulp_witnesses():
    # Only DuckDB's own pow(10,23) modifier (bits 0x44B52D02C7E14AF7) passes these.
    rows = [
        {"x": float.fromhex("0x1.9fc4c8ce7ceb8p+8")},  # -> 415.76868906545445
        {"x": float.fromhex("0x1.bc25ab024d0ffp+8")},  # -> 444.14714064008143
        {"x": float.fromhex("-0x1.a279f7d13a32ep+8")},
    ]  # -> -418.4764376418197
    duck_check("SELECT round(x, 23) AS r FROM __THIS__", {"x": "float"}, rows)


def test_round_variable_n_from_column():
    # engine 'int' columns are BIGINT: round(DOUBLE, BIGINT) does not bind, so CAST.
    rows = [
        {"x": 2.675, "n": 2},
        {"x": 1234.5, "n": -2},
        {"x": NAN, "n": -2},
        {"x": INF, "n": -2},
        {"x": -INF, "n": -2},
        {"x": 2.5, "n": None},
        {"x": None, "n": 2},
        {"x": 1.7e308, "n": -308},
        {"x": 2.675, "n": 400},
        {"x": 2.675, "n": -400},
        {"x": -44.0, "n": -2},
    ]
    duck_check(
        "SELECT round(x, CAST(n AS INTEGER)) AS r,"
        " trunc(x, CAST(n AS INTEGER)) AS t FROM __THIS__",
        {"x": "float?", "n": "int?"},
        rows,
    )


def test_round_family_int64_identity_and_negative_n():
    vals = [
        1250,
        -1250,
        1249,
        -1249,
        25,
        -25,
        49,
        50,
        -49,
        -50,
        0,
        15,
        9223372036854775807,
        -9223372036854775808,
        9223372036854775757,
        9223372036854775800,
        5000000000000000000,
        None,
    ]
    duck_check(
        "SELECT trunc(i) AS t1, round(i) AS r1, round(i, 2) AS rpos,"
        " round(i, -2) AS rn2, round(i, -18) AS rn18, round(i, -19) AS rcut,"
        " trunc(i, -2) AS tn2, trunc(i, -19) AS tcut FROM __THIS__",
        {"i": "int?"},
        [{"i": v} for v in vals],
    )


# --------------------------------------------------------- TASK-48:
# dynamic-table alias — the alias REPLACES the original name (measured).


def test_dynamic_table_alias():
    for sql in [
        "SELECT t.a AS x, a AS y FROM __THIS__ t",
        "SELECT t.a AS x FROM __THIS__ AS t",
        "SELECT t.a AS x, dim.name AS nm FROM __THIS__ t JOIN dim ON t.a = dim.id",
        "SELECT t.*, upper(t.s) AS u FROM __THIS__ t",
    ]:
        duck_check(sql, {"a": "int", "s": "str"}, [{"a": 1, "s": "x"}], {"dim": DIM})


def test_alias_shadows_original_name():
    # Mirrors DuckDB's 'Referenced table not found' as our bind error.
    with pytest.raises(ValueError, match="unknown table"):
        DuckDBInferFn(
            "SELECT __THIS__.a FROM __THIS__ t",
            row_tables={"__THIS__": _row_schema({"a": "int"})},
            static_tables={},
        )
    # t(y) column renaming serves since wave 5 (prefix rename, old name
    # shadowed).
    duck_check("SELECT y + 1 AS p FROM __THIS__ t(y)", {"a": "int"}, [{"a": 4}])
    with pytest.raises(ValueError, match="does not exist"):
        DuckDBInferFn(
            "SELECT a FROM __THIS__ t(y)",
            row_tables={"__THIS__": _row_schema({"a": "int"})},
            static_tables={},
        )


# --------------------------------------------------------- TASK-48:
# LIKE / NOT LIKE / ILIKE - measured DuckDB 1.5.5 pins as oracle tests.
# LIKE / NOT LIKE / ILIKE pins via duck_check (engine vs live DuckDB 1.5.5).
# Every SQL below was validated against a native-table DuckDB oracle on 2026-07-26.
# NOT expressible here: the error pins (dangling escape with chars left, multi-byte
# ESCAPE string) — those raise SyntaxException per-row; cover them with pytest.raises.


def test_like_wildcards_and_empty():
    duck_check(
        "SELECT s LIKE '%' AS c01, s LIKE '%%' AS c02, "
        "s LIKE 'a%' AS c03, s LIKE '%c' AS c04, "
        "s LIKE '%b%' AS c05, s LIKE 'a%c' AS c06, "
        "s LIKE 'a%b' AS c07, s LIKE '%abc' AS c08, "
        "s LIKE 'abc%' AS c09, s LIKE '%abcd%' AS c10, "
        "s LIKE 'abc' AS c11, s LIKE '' AS c12, "
        "s LIKE '_' AS c13, s LIKE '___' AS c14, s LIKE '_%' AS c15, "
        "s LIKE '%_' AS c16, "
        "s LIKE '%_%' AS c17, s LIKE 'a%a' AS c18, "
        "s LIKE 'a%c%f' AS c19, s LIKE '%%c' AS c20, "
        "s NOT LIKE 'a%' AS n01, s NOT LIKE '_' AS n02 FROM __THIS__",
        {"s": "str?"},
        [
            {"s": v}
            for v in [
                "abc",
                "",
                "a",
                "ab",
                "ac",
                "aa",
                "aaa",
                "abcdef",
                "a%b",
                "a_b",
                "%",
                "_",
                "xabcx",
                "A",
                None,
            ]
        ],
    )


def test_like_underscore_is_one_codepoint():
    # NFC 1cp/2B, NFD 2cp/3B, emoji 1cp/4B, ZWJ family 5cp/18B: '_' = one CODEPOINT
    duck_check(
        "SELECT s LIKE '_' AS u1, s LIKE '__' AS u2, "
        "s LIKE '___' AS u3, s LIKE '_____' AS u4, "
        "s LIKE 'a_c' AS u5, s LIKE '%é%' AS u6, s LIKE 'é' AS u7, "
        "s LIKE 'é' AS u8, s LIKE '_x' AS u9 FROM __THIS__",
        {"s": "str?"},
        [
            {"s": v}
            for v in [
                "é",
                "é",
                "\U0001f600",
                "\U0001f468‍\U0001f469‍\U0001f467",
                "aéc",
                "éx",
                None,
            ]
        ],
    )


def test_like_literals_case_and_backslash():
    # regex specials are literals; backslash is literal (NO implicit
    # escape); case-sensitive
    duck_check(
        "SELECT s LIKE 'a' AS l1, s LIKE 'A' AS l2, "
        "s LIKE 'a.c' AS l3, s LIKE 'a[b]c' AS l4, "
        "s LIKE 'a(b)c' AS l5, s LIKE 'a*' AS l6, "
        "s LIKE 'a+?' AS l7, s LIKE 'a^$c' AS l8, "
        "s LIKE 'a\\c' AS l9, s LIKE 'a\\bc' AS l10, "
        "s LIKE 'a\\%b' AS l11, s LIKE '\\' AS l12, "
        "s LIKE '\\%' AS l13 FROM __THIS__",
        {"s": "str?"},
        [
            {"s": v}
            for v in [
                "A",
                "a",
                "ABC",
                "a.c",
                "abc",
                "a[b]c",
                "a(b)c",
                "a*",
                "a+?",
                "a^$c",
                "a\\c",
                "a\\xb",
                "a\\b",
                "\\",
                "%",
                "a%b",
            ]
        ],
    )


def test_like_control_chars():
    # chr() is not in the catalogue; embed the control chars directly in
    # the SQL literals (legal in both engines).
    duck_check(
        "SELECT s LIKE 'a\nb' AS t1, s LIKE 'a_b' AS t2, s LIKE 'a%b' AS t3, "
        "s LIKE '%\nb' AS t4, s LIKE 'a\t%' AS t5, "
        "s LIKE '_\nb' AS t6 FROM __THIS__",
        {"s": "str?"},
        [{"s": v} for v in ["a\nb", "a\tb", "ab"]],
    )


def test_like_dangling_escape_false_rows():
    # Dangling escape is DATA-dependent: these rows return false (string exhausted
    # before the trailing escape is consumed). Rows with chars remaining at the
    # escape raise SyntaxException instead — engine must reproduce BOTH.
    duck_check(
        "SELECT s LIKE 'ab#' ESCAPE '#' AS d1,"
        " s LIKE 'ab#%' ESCAPE '#' AS d2 FROM __THIS__",
        {"s": "str?"},
        [{"s": v} for v in ["ab", "a", "", None]],
    )
    duck_check(
        "SELECT s LIKE 'a#' ESCAPE '#' AS d3 FROM __THIS__",
        {"s": "str?"},
        [{"s": v} for v in ["a", "", "ba", None]],
    )
    duck_check(
        "SELECT s LIKE '#' ESCAPE '#' AS d4 FROM __THIS__",
        {"s": "str?"},
        [{"s": v} for v in ["", None]],
    )


def test_like_pattern_from_column():
    # vec_sp path: pattern (and NULLs) flowing through a column
    duck_check(
        "SELECT s LIKE p AS v1, s NOT LIKE p AS v2,"
        " s LIKE p ESCAPE '#' AS v3 FROM __THIS__",
        {"s": "str?", "p": "str?"},
        [
            {"s": s, "p": p}
            for s, p in [
                ("abc", "a%c"),
                ("abc", "a_c"),
                ("abc", "x%"),
                ("a%b", "a#%b"),
                ("axb", "a#%b"),
                ("ab", "ab#"),
                ("aeb", "aeeb"),
                ("é", "_"),
                ("é", "__"),
                (None, "%"),
                ("abc", None),
                (None, None),
            ]
        ],
    )


def test_like_degenerate_equivalences():
    # LIKE 'abc%'==prefix==starts_with, '%abc'==suffix==ends_with, '%abc%'==contains,
    # bitwise incl. NULL propagation (s LIKE '%' is NULL for NULL s, not false)
    duck_check(
        "SELECT s LIKE 'abc%' AS g1, prefix(s,'abc') AS g2, s LIKE '%abc' AS g3, "
        "suffix(s,'abc') AS g4, s LIKE '%abc%' AS g5, contains(s,'abc') AS g6, "
        "s LIKE 'abc' AS g7, starts_with(s,'abc') AS g8, ends_with(s,'abc') AS g9, "
        "s LIKE '%' AS g10 FROM __THIS__",
        {"s": "str?"},
        [
            {"s": v}
            for v in [
                None,
                "",
                "a",
                "ab",
                "abc",
                "abcx",
                "xabc",
                "xabcx",
                "ABC",
                "a%c",
                "éabcé",
                "\U0001f600abc",
                "a\nb",
                "abca",
                "aabc",
            ]
        ],
    )


def test_like_escape_clause():
    duck_check(
        "SELECT s LIKE 'a#%b' ESCAPE '#' AS e1, s LIKE '%#%' ESCAPE '#' AS e2, "
        "s LIKE 'a#_b' ESCAPE '#' AS e3, s LIKE 'a##b' ESCAPE '#' AS e4, "
        "s LIKE 'a##' ESCAPE '#' AS e5, s LIKE 'a#b' ESCAPE '#' AS e6, "
        "s LIKE 'abc' ESCAPE '#' AS e7, s LIKE 'a%%b' ESCAPE '%' AS e8, "
        "s LIKE 'a%b' ESCAPE '%' AS e9, s LIKE 'a__b' ESCAPE '_' AS e10, "
        "s LIKE 'aeb' ESCAPE 'e' AS e11, s LIKE 'aeeb' ESCAPE 'e' AS e12, "
        "s LIKE 'a#%b' ESCAPE '' AS e13, s ILIKE 'A#%B' ESCAPE '#' AS e14, "
        "s LIKE 'a#%b' ESCAPE NULL AS e15 FROM __THIS__",
        {"s": "str?"},
        [
            {"s": v}
            for v in [
                "a%b",
                "axb",
                "a%",
                "a_b",
                "a#b",
                "a#",
                "ab",
                "abc",
                "aeb",
                "a#xb",
                None,
            ]
        ],
    )


def test_like_dangling_escape_is_data_dependent():
    # Exhausted string -> plain false; chars remaining -> per-row trap.
    duck_check(
        "SELECT s LIKE 'ab#' ESCAPE '#' AS d FROM __THIS__",
        {"s": "str"},
        [{"s": "ab"}],
    )
    with pytest.raises(ValueError, match="must not end with escape"):
        fn = DuckDBInferFn(
            "SELECT s LIKE 'a#' ESCAPE '#' AS d FROM __THIS__",
            row_tables={"__THIS__": _row_schema({"s": "str"})},
            static_tables={},
        )
        fn.infer_rows([{"s": "ax"}])
