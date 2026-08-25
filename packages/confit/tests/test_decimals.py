"""The decimals feature (m-8 Dec lane).

Ordinary fits produce DECIMAL statics - sum(BIGINT) is decimal128(38,0),
and it routinely leaves int64 (2^63+1 measured) - so serving them exactly
is a real demand path, not an edge case. A decimal static is held as a
scaled i128 from ingest through the join and emitted as decimal128(p,s).

Every expectation here is the LIVE oracle: an optimizer-off DuckDB
connection with the same arrow fixtures registered, compared on
`to_pylist()` AND on `schema`. A wrong row is impossible to write.
Expressions OVER a decimal (arithmetic, casts to int/varchar, mixed
coalesce) are m-8 lattice phase 5 and refuse by name; those refusals are
pinned on their message text.
"""

from __future__ import annotations

import decimal
from typing import Any

import duckdb
import pyarrow as pa
import pytest
from confit import DuckDBInferFn

_DEC_SCHEMA = pa.schema([pa.field("gid", pa.string(), nullable=False)])
_GID_ROWS = [{"gid": "a"}]


def _dec_static(val: str) -> pa.Table:
    return pa.table(
        {
            "g": pa.array(["a"], pa.string()),
            "sk": pa.array([decimal.Decimal(val)], pa.decimal128(38, 0)),
        }
    )


def _dec62(vals: list[str | None], gs: list[str] | None = None) -> pa.Table:
    """A DECIMAL(6,2) static: negative and NULL payloads included."""
    gs = gs or [chr(ord("a") + i) for i in range(len(vals))]
    return pa.table(
        {
            "g": pa.array(gs, pa.string()),
            "d": pa.array(
                [None if v is None else decimal.Decimal(v) for v in vals],
                pa.decimal128(6, 2),
            ),
        }
    )


def _fit_params() -> pa.Table:
    """A REAL fit-time aggregate, so the decimal128(38,0) comes from DuckDB
    rather than a hand-built pa.array. sum(BIGINT) here is 2^63+1."""
    con = duckdb.connect()
    con.execute("CREATE TABLE t(g VARCHAR, n BIGINT)")
    con.execute("INSERT INTO t VALUES ('a', 9223372036854775807), ('a', 2), ('b', 5)")
    return con.execute(
        "SELECT g, sum(n) AS sk, count(*) AS c, min(n) AS mn"
        " FROM t GROUP BY g ORDER BY g"
    ).to_arrow_table()


def _oracle(
    sql: str,
    row_schema: pa.Schema,
    rows: list[dict[str, Any]],
    statics: dict[str, pa.Table],
) -> pa.Table:
    con = duckdb.connect()
    con.execute("PRAGMA disable_optimizer")
    for name, table in statics.items():
        con.register(f"__arrow_{name}", table)
        con.execute(f'CREATE TABLE "{name}" AS SELECT * FROM "__arrow_{name}"')
    con.register("__arrow_this", pa.Table.from_pylist(rows, schema=row_schema))
    con.execute("CREATE TABLE __THIS__ AS SELECT * FROM __arrow_this")
    return con.execute(sql).to_arrow_table()


def _check(
    sql: str,
    row_schema: pa.Schema,
    rows: list[dict[str, Any]],
    statics: dict[str, pa.Table],
) -> pa.Table:
    """Ours vs optimizer-off DuckDB: values AND schema.

    Row ORDER is a hash-join accident on the oracle side, so values compare
    as a sorted multiset of repr'd rows (repr keeps Decimal('0.50') apart
    from Decimal('0.5') and from 0.5, which is the whole point here).
    """
    fn = DuckDBInferFn(sql, row_tables={"__THIS__": row_schema}, static_tables=statics)
    got = fn.infer_arrow(pa.Table.from_pylist(rows, schema=row_schema))
    want = _oracle(sql, row_schema, rows, statics)
    key = lambda r: sorted((k, repr(v)) for k, v in r.items())  # noqa: E731
    assert sorted(map(key, got.to_pylist())) == sorted(map(key, want.to_pylist())), (
        f"{sql}: {got.to_pylist()} != {want.to_pylist()}"
    )
    assert got.schema == want.schema, f"{sql}: {got.schema} != {want.schema}"
    return got


_JOIN = "SELECT {} FROM __THIS__ LEFT JOIN p ON gid = p.g"


def test_an_inexact_decimal_static_serves_exactly():
    """2^53+1 in a decimal static comes back as ITSELF."""
    fn = DuckDBInferFn(
        "SELECT sk AS o FROM __THIS__ LEFT JOIN p ON gid = p.g",
        row_tables={"__THIS__": _DEC_SCHEMA},
        static_tables={"p": _dec_static("9007199254740993")},
    )
    got = fn.infer_rows([{"gid": "a"}])
    assert got == [{"o": decimal.Decimal("9007199254740993")}]


def test_an_exact_decimal_static_still_serves():
    """Flip 3: an f64-representable decimal stops coming back as a double."""
    fn = DuckDBInferFn(
        "SELECT sk AS o FROM __THIS__ LEFT JOIN p ON gid = p.g",
        row_tables={"__THIS__": _DEC_SCHEMA},
        static_tables={"p": _dec_static("9007199254740992")},
    )
    got = fn.infer_rows([{"gid": "a"}])
    # By repr, not by ==: 9007199254740992.0 == Decimal("9007199254740992")
    # is True in Python, so a bare == cannot see this flip at all.
    assert repr(got) == repr([{"o": decimal.Decimal("9007199254740992")}])


def test_a_negative_inexact_decimal_static_serves_exactly():
    """Cell A1, second row."""
    _check(
        _JOIN.format("sk AS o"),
        _DEC_SCHEMA,
        _GID_ROWS,
        {"p": _dec_static("-9007199254740993")},
    )


@pytest.mark.parametrize("p,s", [(38, 0), (6, 2), (4, 2), (9, 4)])
def test_a_decimal_static_types_as_decimal128_in_infer_arrow(p, s):
    """Cells I2/I3: all four DuckDB storage tiers (int16/int32/int64/int128)
    leave as decimal128(p,s), so our boundary has one output arm."""
    val = decimal.Decimal(1).scaleb(-s)
    static = pa.table(
        {
            "g": pa.array(["a"], pa.string()),
            "d": pa.array([val], pa.decimal128(p, s)),
        }
    )
    got = _check(_JOIN.format("d AS o"), _DEC_SCHEMA, _GID_ROWS, {"p": static})
    assert got.schema.field("o").type == pa.decimal128(p, s)


def test_a_small_scale_decimal_serves_negative_and_null():
    """Cell B1, compared by repr so the SCALE SPELLING is pinned."""
    _check(
        _JOIN.format("d AS o"),
        _DEC_SCHEMA,
        [{"gid": "a"}, {"gid": "b"}, {"gid": "c"}],
        {"p": _dec62(["-12.34", None, "99.99"])},
    )


def test_a_left_join_miss_gives_a_null_decimal():
    """Cell A2: the probe misses, the decimal column is NULL not absent."""
    _check(
        _JOIN.format("sk AS o"),
        _DEC_SCHEMA,
        [{"gid": "zz"}],
        {"p": _dec_static("9007199254740993")},
    )


@pytest.mark.parametrize("proj", ["*", "* EXCLUDE (d)", "* EXCLUDE (g)"])
def test_a_decimal_static_survives_star_and_exclude(proj):
    """Cells E1/E2/E3 - including the column ORDER, which `schema` pins."""
    _check(
        _JOIN.format(proj),
        _DEC_SCHEMA,
        [{"gid": "a"}],
        {"p": _dec62(["-12.34"])},
    )


def test_a_fit_tables_sum_bigint_serves_beyond_int64():
    """Cells F1+F2, the feature's reason to exist: sum(BIGINT) over a fit
    table is 2^63+1, which no i64 lane can hold."""
    params = _fit_params()
    assert params.to_pylist()[0]["sk"] == decimal.Decimal("9223372036854775809")
    _check(
        "SELECT sk AS o, c, mn FROM __THIS__ LEFT JOIN p ON gid = p.g",
        _DEC_SCHEMA,
        [{"gid": "a"}, {"gid": "b"}, {"gid": "zz"}],
        {"p": params},
    )


def test_a_decimal_key_joins_a_bigint_probe_exactly():
    """Cell D1: DuckDB casts the BIGINT side UP to DECIMAL(38,0), exactly."""
    static = pa.table(
        {
            "sk": pa.array([decimal.Decimal("9007199254740993")], pa.decimal128(38, 0)),
            "v": pa.array(["hi"], pa.string()),
        }
    )
    schema = pa.schema([pa.field("k", pa.int64(), nullable=False)])
    _check(
        "SELECT v AS o FROM __THIS__ LEFT JOIN p ON k = p.sk",
        schema,
        [{"k": 9007199254740993}, {"k": 9007199254740992}],
        {"p": static},
    )


def test_a_decimal_key_joins_a_double_probe_like_duckdb():
    """Cell D2: only decimal->double is an implicit cast, so the DECIMAL
    side casts DOWN and the comparison is LOSSY. Probe 9007199254740992.0
    MATCHES the 2^53+1 static row. We reproduce the loss, not hide it."""
    static = pa.table(
        {
            "sk": pa.array([decimal.Decimal("9007199254740993")], pa.decimal128(38, 0)),
            "v": pa.array(["hi"], pa.string()),
        }
    )
    schema = pa.schema([pa.field("k", pa.float64(), nullable=False)])
    got = _check(
        "SELECT v AS o FROM __THIS__ LEFT JOIN p ON k = p.sk",
        schema,
        [{"k": 9007199254740992.0}, {"k": 9007199254740994.0}],
        {"p": static},
    )
    assert sorted(r["o"] or "" for r in got.to_pylist()) == ["", "hi"]


def test_a_non_integral_decimal_key_never_matches_an_integer_probe():
    """Cell D3: a non-integral build key cannot equal any integer probe, so
    the build row drops - the same semantics-preserving move a NULL `=` key
    already gets."""
    static = pa.table(
        {
            "dk": pa.array(
                [decimal.Decimal("-12.34"), decimal.Decimal("99.99")],
                pa.decimal128(6, 2),
            ),
            "v": pa.array(["neg", "pos"], pa.string()),
        }
    )
    schema = pa.schema([pa.field("k", pa.int64(), nullable=False)])
    _check(
        "SELECT v AS o FROM __THIS__ LEFT JOIN p ON k = p.dk",
        schema,
        [{"k": 100}, {"k": -12}],
        {"p": static},
    )


def test_an_integral_decimal_key_matches_both_probe_lanes():
    """Cells H3/H4, the must-not-regress control for the drop rule above."""
    static = pa.table(
        {
            "dk": pa.array(
                [decimal.Decimal("100.00"), decimal.Decimal("-2.25")],
                pa.decimal128(6, 2),
            ),
            "v": pa.array(["hundred", "negq"], pa.string()),
        }
    )
    _check(
        "SELECT v AS o FROM __THIS__ LEFT JOIN p ON k = p.dk",
        pa.schema([pa.field("k", pa.int64(), nullable=False)]),
        [{"k": 100}, {"k": -2}],
        {"p": static},
    )
    _check(
        "SELECT v AS o FROM __THIS__ LEFT JOIN p ON k = p.dk",
        pa.schema([pa.field("k", pa.float64(), nullable=False)]),
        [{"k": 100.0}, {"k": -2.25}],
        {"p": static},
    )


@pytest.mark.parametrize(
    "pred",
    [
        "sk = 9007199254740993",
        "sk = 9007199254740992",
        "sk > 0",
        "sk < 0",
    ],
)
def test_where_predicates_on_a_wide_decimal_static(pred):
    """Cells C1 + G5/G6 on the (38,0) tier: the bind-time constant fold."""
    _check(
        f"SELECT gid AS o FROM __THIS__ LEFT JOIN p ON gid = p.g WHERE {pred}",
        _DEC_SCHEMA,
        [{"gid": "a"}],
        {"p": _dec_static("9007199254740993")},
    )


@pytest.mark.parametrize(
    "pred",
    ["d = -12.34", "d = 99.99", "d = 0.50", "d > 0", "d <= -12.34", "d < 100"],
)
def test_where_predicates_on_a_small_scale_decimal_static(pred):
    """Cells C2/C3 and G5/G6: equality against a literal that does not
    divide down to the column scale is constant-false; the floor/ceil at the
    column's scale is the exact ordering bound."""
    _check(
        f"SELECT gid AS o FROM __THIS__ LEFT JOIN p ON gid = p.g WHERE {pred}",
        _DEC_SCHEMA,
        [{"gid": "a"}, {"gid": "b"}, {"gid": "c"}],
        {"p": _dec62(["-12.34", None, "99.99"])},
    )


def test_a_decimal_compared_to_a_bigint_row_column():
    """The Itod path: the integer lane scales up into the column's scale."""
    schema = pa.schema(
        [
            pa.field("gid", pa.string(), nullable=False),
            pa.field("k", pa.int64(), nullable=False),
        ]
    )
    _check(
        "SELECT gid AS o FROM __THIS__ LEFT JOIN p ON gid = p.g WHERE d = k",
        schema,
        [{"gid": "a", "k": 100}, {"gid": "b", "k": 0}, {"gid": "c", "k": 3}],
        {"p": _dec62(["100.00", "-2.25", "3.00"])},
    )


def test_a_decimal_compared_to_a_double_row_column():
    """The Dtof path on a comparison: DuckDB casts the DECIMAL side DOWN."""
    schema = pa.schema(
        [
            pa.field("gid", pa.string(), nullable=False),
            pa.field("x", pa.float64(), nullable=False),
        ]
    )
    _check(
        "SELECT gid AS o FROM __THIS__ LEFT JOIN p ON gid = p.g WHERE d = x",
        schema,
        [{"gid": "a", "x": 0.5}, {"gid": "b", "x": 1.0}, {"gid": "c", "x": -2.25}],
        {"p": _dec62(["0.50", "1.25", "-2.25"])},
    )


def test_cast_a_decimal_static_to_double():
    """Cell G9, must not regress: CAST(d AS DOUBLE) is DuckDB's div/mod
    algorithm, not Python's correctly-rounded float(Decimal)."""
    _check(
        _JOIN.format("CAST(d AS DOUBLE) AS o"),
        _DEC_SCHEMA,
        [{"gid": "a"}, {"gid": "b"}, {"gid": "c"}],
        {"p": _dec62(["0.50", None, "-2.25"])},
    )


_PHASE5 = "m-8 lattice phase 5"


def _refuses(sql: str, statics: dict[str, pa.Table], *needles: str) -> None:
    with pytest.raises(ValueError) as ei:
        DuckDBInferFn(sql, row_tables={"__THIS__": _DEC_SCHEMA}, static_tables=statics)
    msg = str(ei.value)
    for n in needles:
        assert n in msg, f"{sql}: {n!r} not in {msg!r}"


@pytest.mark.parametrize("expr", ["d + 1", "d - 1", "d * 2", "d / 2", "d % 2"])
def test_decimal_arithmetic_refuses_by_name(expr):
    """Cells G7/G8: arithmetic over a decimal was a silently wrong double."""
    _refuses(
        _JOIN.format(f"{expr} AS o"),
        {"p": _dec62(["0.50"])},
        "DECIMAL(6,2)",
        "'d'",
        _PHASE5,
    )


@pytest.mark.parametrize("target", ["BIGINT", "INTEGER", "VARCHAR"])
def test_a_decimal_cast_to_integer_or_varchar_refuses_by_name(target):
    """Cells G10/G11 - WRONG VALUES on master (0.50::BIGINT was 0, DuckDB
    says 1; '0.5' vs '0.50'), so the trade is the ladder's own preference."""
    _refuses(
        _JOIN.format(f"CAST(d AS {target}) AS o"),
        {"p": _dec62(["0.50"])},
        "DECIMAL(6,2)",
        "'d'",
        _PHASE5,
    )


@pytest.mark.parametrize(
    "expr", ["coalesce(d, 0)", "CASE WHEN d > 0 THEN d ELSE 0 END"]
)
def test_coalesce_mixing_a_decimal_with_an_integer_refuses_by_name(expr):
    """Cells G12/G13: DuckDB unifies these to a WIDER decimal; we refuse."""
    _refuses(
        _JOIN.format(f"{expr} AS o"),
        {"p": _dec62(["0.50"])},
        "DECIMAL(6,2)",
        "'d'",
        _PHASE5,
    )


def test_a_decimal256_static_column_refuses_by_name():
    """Cell I4: DuckDB refuses decimal256 at arrow register at ANY
    precision ('Unsupported Internal Arrow Type for Decimal'), so serving
    it on the f64 lane was serve-where-DuckDB-refuses."""
    static = pa.table(
        {
            "g": pa.array(["a"], pa.string()),
            "d": pa.array([decimal.Decimal("1.25")], pa.decimal256(38, 2)),
        }
    )
    _refuses(_JOIN.format("d AS o"), {"p": static}, "'d'", "decimal256")


def test_a_wide_scale_decimal_key_against_an_integer_probe_refuses_by_name():
    """Flip 6: DuckDB compares these as DECIMAL(38,30), where the integer's
    cast can FAIL per row - a row-time trap the key path has no shape for."""
    static = pa.table(
        {
            "dk": pa.array([decimal.Decimal(1)], pa.decimal128(38, 30)),
            "v": pa.array(["x"], pa.string()),
        }
    )
    schema = pa.schema([pa.field("k", pa.int64(), nullable=False)])
    with pytest.raises(ValueError) as ei:
        DuckDBInferFn(
            "SELECT v AS o FROM __THIS__ LEFT JOIN p ON k = p.dk",
            row_tables={"__THIS__": schema},
            static_tables={"p": static},
        )
    msg = str(ei.value)
    assert "cannot join" in msg and "dec(38,30)" in msg, msg


def test_an_unreferenced_decimal_static_column_no_longer_blocks_a_build():
    """Cell I1: the old exactness guard ran over EVERY static column,
    referenced or not, so an untouched inexact decimal refused the build."""
    static = pa.table(
        {
            "g": pa.array(["a"], pa.string()),
            "v": pa.array(["hit"], pa.string()),
            "d": pa.array([decimal.Decimal("9007199254740993")], pa.decimal128(38, 0)),
        }
    )
    _check(
        _JOIN.format("v AS o"),
        _DEC_SCHEMA,
        [{"gid": "a"}, {"gid": "zz"}],
        {"p": static},
    )


def test_a_many_shape_join_serves_a_decimal_value_column():
    """The MultiMap value path (interpreter-only), which no cell above
    exercises: duplicate keys fan out and each carries its own decimal."""
    static = pa.table(
        {
            "g": pa.array(["a", "a", "b"], pa.string()),
            "d": pa.array(
                [decimal.Decimal("-12.34"), decimal.Decimal("99.99"), None],
                pa.decimal128(6, 2),
            ),
        }
    )
    rows = [{"gid": "a"}, {"gid": "b"}, {"gid": "zz"}]
    fn = DuckDBInferFn(
        _JOIN.format("d AS o"),
        row_tables={"__THIS__": _DEC_SCHEMA},
        static_tables={"p": static},
        shape="many",
    )
    got = [r["o"] for r in fn.infer_rows(rows)]
    want = [
        r["o"]
        for r in _oracle(
            _JOIN.format("d AS o"), _DEC_SCHEMA, rows, {"p": static}
        ).to_pylist()
    ]
    key = lambda v: (v is None, repr(v))  # noqa: E731
    assert sorted(got, key=key) == sorted(want, key=key), f"{got} != {want}"
