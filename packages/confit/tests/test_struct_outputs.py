"""Struct-valued outputs: a projection item, star column or struct field whose
value is a whole struct. A struct leaves as one Arrow struct field, and as
one dict on the row path. It is NULL where the struct itself is NULL, which
is not a struct of NULLs.

Each expectation is the campaign verdict (`fuzz.parity`), names and types
included, against DuckDB 1.5.5 with the optimizer off.
"""

from __future__ import annotations

import decimal
import sys
from pathlib import Path

import pyarrow as pa
import pytest
from confit import DuckDBInferFn, ExternFunction, SqlFunction
from confit import sql as S

sys.path.insert(0, str(Path(__file__).parents[1]))

from fuzz.parity import assert_parity  # noqa: E402

_S = pa.struct(
    [("a", pa.int64()), ("B", pa.string()), ("n", pa.struct([("x", pa.float64())]))]
)
_W = pa.struct([("a", pa.int64())])
T = pa.Table.from_pylist(
    [
        {"k": 1, "x": 1.5, "s": {"a": 1, "B": "q", "n": {"x": 1.5}}, "w": {"a": 9}},
        {"k": 2, "x": None, "s": None, "w": None},
        {"k": 3, "x": -2.0, "s": {"a": None, "B": None, "n": None}, "w": {"a": None}},
    ],
    schema=pa.schema(
        [
            pa.field("k", pa.int64()),
            pa.field("x", pa.float64()),
            pa.field("s", _S),
            pa.field("w", _W),
        ]
    ),
)
D = pa.table(
    {
        "id": pa.array([1, 3], pa.int64()),
        "v": pa.array(
            [{"x": 1, "y": "p"}, None],
            pa.struct([("x", pa.int64()), ("y", pa.string())]),
        ),
        "dw": pa.array([{"a": 5}, None], _W),
        "n": pa.array(
            [{"m": {"q": 1}, "r": 2}, None],
            pa.struct([("m", pa.struct([("q", pa.int64())])), ("r", pa.int64())]),
        ),
        # a timestamp leaf: outside the served types
        "z": pa.array(
            [{"t": None, "i": 1}, None],
            pa.struct([("t", pa.timestamp("us")), ("i", pa.int64())]),
        ),
    }
)


def agree(sql: str, rows: pa.Table = T, **kw):
    return assert_parity(sql, rows, statics={"d": D}, expect="AGREE", **kw)


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT s FROM __THIS__",
        "SELECT s AS o, k FROM __THIS__",
        "SELECT s, s FROM __THIS__",
        "SELECT s, s.n, w FROM __THIS__",
        "SELECT s AS o FROM __THIS__ WHERE k > 1",
        # a nested struct field, in every spelling
        "SELECT s.n FROM __THIS__",
        "SELECT s['n'] FROM __THIS__",
        "SELECT (s).n FROM __THIS__",
        "SELECT struct_extract(s, 'n') FROM __THIS__",
        "SELECT __THIS__.s FROM __THIS__",
        "SELECT t.s FROM __THIS__ AS t",
        # stars keep a struct column as one field
        "SELECT * FROM __THIS__",
        "SELECT __THIS__.* FROM __THIS__",
        "SELECT s.* FROM __THIS__",
        "SELECT COLUMNS('s|w') FROM __THIS__",
        "SELECT * REPLACE (w AS s) FROM __THIS__",
        "SELECT * EXCLUDE (s) FROM __THIS__",
        # a static table's struct is NULL on a LEFT miss
        "SELECT v FROM __THIS__ LEFT JOIN d ON k = id",
        "SELECT d.v, n FROM __THIS__ LEFT JOIN d ON k = id",
        "SELECT n.m FROM __THIS__ LEFT JOIN d ON k = id",
        "SELECT n.* FROM __THIS__ LEFT JOIN d ON k = id",
        "SELECT * EXCLUDE (z) FROM __THIS__ LEFT JOIN d ON k = id",
        "SELECT v FROM __THIS__ JOIN d ON k = id",
    ],
)
def test_a_struct_column_reads_whole(sql):
    agree(sql)


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT t AS o FROM __THIS__ AS t",
        "SELECT __THIS__ FROM __THIS__",
        "SELECT d AS o, d IS NULL AS n FROM __THIS__ LEFT JOIN d ON k = id",
        "SELECT CASE WHEN k > 1 THEN d END AS o FROM __THIS__ LEFT JOIN d ON k = id",
    ],
)
def test_a_relation_reads_as_its_row_struct(sql):
    """A relation no column shares the name of is its row struct: its
    columns in order, never NULL (a LEFT miss is a struct of NULLs)."""
    # z's timestamp leaf would refuse the whole row of d.
    assert_parity(sql, T, statics={"d": D.drop_columns(["z"])}, expect="AGREE")


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT s IS NULL AS o, s.n IS NOT NULL AS p FROM __THIS__",
        "SELECT k FROM __THIS__ WHERE s IS NOT NULL",
        "SELECT v IS NULL AS o, n.m IS NULL AS p FROM __THIS__ LEFT JOIN d ON k = id",
        "SELECT t IS NULL AS o FROM __THIS__ AS t",
    ],
)
def test_a_struct_is_null_where_the_struct_is(sql):
    agree(sql)


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT struct_pack(a := k, b := s) AS p FROM __THIS__",
        "SELECT struct_pack(a := k, b := struct_pack(c := s.a, d := 'z')) AS p "
        "FROM __THIS__",
        "SELECT struct_pack(r := v, k := k) AS p FROM __THIS__ LEFT JOIN d ON k = id",
        "SELECT {'a': k, 'b': s.n} AS p FROM __THIS__",
        # unaliased literals are named `main.struct_pack(...)`
        "SELECT {'a': k, 'b': s.n} FROM __THIS__",
        "SELECT {'a': {'b': k}} FROM __THIS__",
        "SELECT {'x y': 1, 'Q': k} FROM __THIS__",
        "SELECT {'_a': 1} AS p FROM __THIS__",
        "SELECT struct_pack(a := NULL) AS p FROM __THIS__",
        "SELECT {'a': NULL} AS p FROM __THIS__",
    ],
)
def test_a_built_struct_matches_duckdb(sql):
    agree(sql)


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT CASE WHEN k > 1 THEN s END AS o FROM __THIS__",
        "SELECT CASE WHEN k > 1 THEN w ELSE dw END AS o "
        "FROM __THIS__ LEFT JOIN d ON k = id",
        "SELECT CASE WHEN k > 2 THEN w WHEN k = 2 THEN NULL ELSE dw END AS o "
        "FROM __THIS__ LEFT JOIN d ON k = id",
        "SELECT CASE k WHEN 1 THEN {'a': x} WHEN 3 THEN {'a': x + 1} END AS o "
        "FROM __THIS__",
        "SELECT CASE WHEN s IS NULL THEN NULL ELSE {'a': s.a} END AS o FROM __THIS__",
        "SELECT CASE WHEN k > 1 THEN {'a': {'b': s}} END AS o FROM __THIS__",
        "SELECT (CASE WHEN k > 1 THEN s END).a AS o FROM __THIS__",
    ],
)
def test_a_struct_case_matches_duckdb(sql):
    agree(sql)


def test_a_case_arm_not_taken_does_not_trap():
    """DuckDB computes a CASE arm only on the rows that take it, so a field
    that would trap on a row the guard sends elsewhere does not."""
    for sql in [
        "SELECT CASE WHEN x IS NULL THEN NULL ELSE "
        "struct_pack(a := CAST(coalesce(x, 1e30) AS BIGINT)) END AS o FROM __THIS__",
        "SELECT CASE WHEN k > 0 THEN NULL ELSE "
        "struct_pack(a := k * 9223372036854775807) END AS o FROM __THIS__",
    ]:
        agree(sql)
    # Without a guard, every field is built on every row.
    assert_parity(
        "SELECT struct_pack(a := k * 9223372036854775807) AS o FROM __THIS__",
        T,
        trap="Overflow",
    )


_C = pa.struct(
    [
        ("d", pa.decimal128(9, 4)),
        ("u", pa.uint64()),
        ("h", pa.int8()),
        ("b", pa.bool_()),
        ("g", pa.decimal128(38, 0)),
    ]
)
TYPED = pa.Table.from_pylist(
    [
        {
            "k": 1,
            "c": {
                "d": decimal.Decimal("1.5000"),
                "u": 2**64 - 1,
                "h": -3,
                "b": True,
                "g": decimal.Decimal(10**37),
            },
        },
        {"k": None, "c": None},
        {"k": 3, "c": {"d": None, "u": None, "h": None, "b": None, "g": None}},
    ],
    schema=pa.schema([pa.field("k", pa.int64()), pa.field("c", _C)]),
)
TYPED_D = pa.table(
    {
        "id": pa.array([1, 3], pa.int64()),
        "e": pa.array(
            [
                {
                    "d": decimal.Decimal("2.2500"),
                    "u": 7,
                    "h": 1,
                    "b": False,
                    "g": decimal.Decimal(5),
                },
                None,
            ],
            _C,
        ),
    }
)


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT c FROM __THIS__",
        "SELECT * FROM __THIS__",
        "SELECT e FROM __THIS__ LEFT JOIN d ON k = id",
        "SELECT struct_pack(c := c, e := e) AS p FROM __THIS__ LEFT JOIN d ON k = id",
        "SELECT c, c.d * 2 AS dd FROM __THIS__",
        # HUGEINT leaves as decimal128(38, 0), past 38 digits too
        "SELECT struct_pack("
        "h := CAST(170141183460469231731687303715884105727 AS HUGEINT),"
        " i := CAST(2 AS UTINYINT)) AS p FROM __THIS__",
        "SELECT struct_pack(h := CAST(k AS HUGEINT) * 10, u := c.u) AS p FROM __THIS__",
    ],
)
def test_struct_leaves_keep_their_types(sql):
    assert_parity(sql, TYPED, statics={"d": TYPED_D}, expect="AGREE")


PAIR = SqlFunction(
    "pair",
    pa.schema([("v", pa.float64())]),
    pa.struct([("lo", pa.float64()), ("hi", pa.float64())]),
    lambda v: {"lo": v - S.lit(1.0), "hi": v + S.lit(1.0)},
)
PAIR_N = SqlFunction(
    "pair_n",
    pa.schema([("v", pa.float64())]),
    pa.struct([("lo", pa.float64()), ("hi", pa.float64())]),
    lambda v: {"lo": v - S.lit(1.0), "hi": v + S.lit(1.0)},
    null_when=lambda v: v.isnull(),
)
PAIR_NEG = SqlFunction(
    "pair_neg",
    pa.schema([("v", pa.float64())]),
    pa.struct([("lo", pa.float64()), ("hi", pa.float64())]),
    lambda v: {"lo": v - S.lit(1.0), "hi": v + S.lit(1.0)},
    null_when=lambda v: v < S.lit(0.0),
)
EMB = ExternFunction(
    "emb",
    pa.schema([("v", pa.float64())]),
    pa.struct([("a", pa.float64()), ("b", pa.float64())]),
    lambda v: None if v is None else (v + 1.0, v * 2.0),
)


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT pair(x) AS o FROM __THIS__",
        "SELECT pair_n(x) AS o FROM __THIS__",
        # a null_when that is not an IS NULL test
        "SELECT pair_neg(x) AS o FROM __THIS__",
        "SELECT struct_pack(p := pair(x), q := k) AS o FROM __THIS__",
        "SELECT CASE WHEN k > 1 THEN pair(x) ELSE pair_n(x) END AS o FROM __THIS__",
        "SELECT emb(x) AS o FROM __THIS__",
        "SELECT struct_pack(e := emb(x), k := k) AS o FROM __THIS__",
        "SELECT CASE WHEN k > 1 THEN emb(x) END AS o FROM __THIS__",
        # a UDF's validity is its call, which runs
        "SELECT emb(x) IS NULL AS n, emb(x) IS NOT NULL AS m FROM __THIS__",
        "SELECT k FROM __THIS__ WHERE emb(x) IS NOT NULL",
    ],
)
def test_a_struct_valued_function_matches_duckdb(sql):
    agree(sql, udfs=[PAIR, PAIR_N, PAIR_NEG, EMB])


_K = pa.table(
    {
        "k": pa.array([1, 3], pa.int64()),
        "w": pa.array([{"a": 9}, {"a": 8}], _W),
        "v": pa.array([10, 30], pa.int64()),
    }
)


@pytest.mark.parametrize(
    "sql",
    [
        # a USING key merges into the left occurrence under `*` ...
        "SELECT * FROM __THIS__ LEFT JOIN d USING (w)",
        "SELECT * FROM __THIS__ JOIN d USING (w)",
        "SELECT * FROM __THIS__ LEFT JOIN d USING (k)",
        # ... and its own table's star keeps it, with that table's values
        "SELECT d.* FROM __THIS__ LEFT JOIN d USING (k)",
        "SELECT d.* EXCLUDE (k) FROM __THIS__ LEFT JOIN d USING (k)",
        "SELECT d FROM __THIS__ LEFT JOIN d USING (k)",
    ],
)
def test_a_using_key_under_a_star(sql):
    assert_parity(sql, T.select(["k", "w"]), statics={"d": _K}, expect="AGREE")


@pytest.mark.parametrize(
    ("sql", "why", "oracle"),
    [
        (
            "SELECT z FROM __THIS__ LEFT JOIN d ON k = id",
            "its field 't' has a non-scalar type",
            "serves",
        ),
        (
            "SELECT x FROM (SELECT s AS x FROM __THIS__)",
            "derived table",
            "serves",
        ),
        (
            "SELECT CASE WHEN k > 1 THEN {'a': k} ELSE {'a': 0} END AS o FROM __THIS__",
            "structs of different types",
            "serves",
        ),
        (
            "SELECT CASE WHEN k > 1 THEN {'a': k} ELSE 5 END AS o FROM __THIS__",
            "a struct and another value",
            "rejects",
        ),
        # DuckDB builds every field before it tests the struct
        (
            "SELECT struct_pack(a := k) IS NULL AS o FROM __THIS__",
            "struct_pack",
            "serves",
        ),
        (
            "SELECT {'a': 1, 'A': 2} AS o FROM __THIS__",
            'duplicate struct entry name "A"',
            "rejects",
        ),
    ],
)
def test_other_struct_values_refuse_by_name(sql, why, oracle):
    v = assert_parity(sql, T, statics={"d": D}, expect="REFUSED")
    assert why in v.detail, v.detail
    assert v.oracle == oracle, f"oracle moved: {v.oracle}"


def test_a_using_key_struct_under_its_own_star_refuses():
    v = assert_parity(
        "SELECT d.* FROM __THIS__ LEFT JOIN d USING (w)",
        T.select(["k", "w"]),
        statics={"d": _K},
        expect="REFUSED",
    )
    assert "non-scalar" in v.detail and v.oracle == "serves"


@pytest.mark.parametrize("generic", [False, True])
def test_both_boundaries_carry_the_struct(monkeypatch, generic):
    """The Arrow field is NULL where the struct is, and so is each child
    (DuckDB's own export); each row path's dict says the same, and the
    declared schema is the table's."""
    if generic:
        # the generic row boundary, which the marshaller replaced
        monkeypatch.setenv("SPECIALIZER_GENERIC_BOUNDARY", "1")
    fn = DuckDBInferFn(
        "SELECT s, struct_pack(p := s.n, q := k) AS t FROM __THIS__",
        row_tables={"__THIS__": T.schema},
        static_tables={},
    )
    out = fn.infer_arrow(T)
    assert fn.output_schema == out.schema
    s = out.column("s").combine_chunks()
    assert s.is_null().to_pylist() == [False, True, False]
    assert s.field("a").is_null().to_pylist() == [False, True, True]
    assert s.field("n").is_null().to_pylist() == [False, True, True]
    assert (
        fn.infer_rows(T.to_pylist())
        == out.to_pylist()
        == [
            {"s": {"a": 1, "B": "q", "n": {"x": 1.5}}, "t": {"p": {"x": 1.5}, "q": 1}},
            {"s": None, "t": {"p": None, "q": 2}},
            {"s": {"a": None, "B": None, "n": None}, "t": {"p": None, "q": 3}},
        ]
    )


def test_a_struct_without_fields_refuses_whole():
    """DuckDB has no STRUCT type without fields (it refuses one at input),
    so there is no whole value to match."""
    schema = pa.schema([pa.field("k", pa.int64()), pa.field("e", pa.struct([]))])
    for sql in ["SELECT e FROM __THIS__", "SELECT * FROM __THIS__"]:
        with pytest.raises(ValueError, match="it has no fields"):
            DuckDBInferFn(sql, row_tables={"__THIS__": schema}, static_tables={})
