"""The arrow schema surface (spec: 2026-08-13-arrow-schema-api-design.md).

Every row-table schema is a pa.Schema; rows are dict-or-object in, dict
out. Strict totality, no coercion, input range checks, nullability from
the arrow field flag. The pydantic surface is deleted.
"""

from types import SimpleNamespace

import duckdb
import pyarrow as pa
import pytest
from confit import DuckDBInferFn

SCHEMA = pa.schema(
    [
        pa.field("a", pa.float64()),
        pa.field("b", pa.int32()),
        pa.field("s", pa.string()),
        pa.field("k", pa.int64(), nullable=False),
    ]
)

ROW = {"a": 1.5, "b": 2, "s": "x", "k": 3}


def build(sql, schema=SCHEMA, **kw):
    return DuckDBInferFn(sql, row_tables={"__THIS__": schema}, static_tables={}, **kw)


def test_dict_in_dict_out():
    fn = build("SELECT a + 1.0 AS o FROM __THIS__")
    out = fn.infer_rows([ROW])
    assert out == [{"o": 2.5}]
    assert isinstance(out[0], dict)


def test_object_in_dict_out():
    fn = build("SELECT k + 1 AS o FROM __THIS__")
    assert fn.infer_rows([SimpleNamespace(**ROW)]) == [{"o": 4}]


def test_narrow_width_binds_round_digits():
    # The corpus spelling pydantic could not express: round(DOUBLE, INTEGER).
    fn = build("SELECT round(a, b) AS o FROM __THIS__")
    assert fn.infer_rows([{**ROW, "a": 2.345, "b": 2}]) == [{"o": 2.35}]


def test_narrow_width_flows_to_output_schema():
    # INTEGER + INTEGER stays INTEGER in DuckDB's lattice; the declared
    # int32 must reach the binder, not collapse to int64.
    fn = build("SELECT b + b AS o FROM __THIS__")
    assert fn.output_schema.field("o").type == pa.int32()


def test_output_schema_matches_infer_arrow():
    fn = build("SELECT a * 2.0 AS d, b AS n, s AS t FROM __THIS__")
    table = fn.infer_arrow(pa.Table.from_pylist([ROW], schema=SCHEMA))
    assert fn.output_schema == table.schema


def test_missing_key_refuses_by_name():
    fn = build("SELECT a AS o FROM __THIS__")
    row = dict(ROW)
    del row["b"]
    with pytest.raises(ValueError, match="missing attribute 'b'"):
        fn.infer_rows([row])


def test_missing_attribute_refuses_by_name():
    fn = build("SELECT a AS o FROM __THIS__")
    row = dict(ROW)
    del row["s"]
    with pytest.raises(ValueError, match="missing attribute 's'"):
        fn.infer_rows([SimpleNamespace(**row)])


def test_extra_dict_keys_are_ignored():
    fn = build("SELECT a AS o FROM __THIS__")
    assert fn.infer_rows([{**ROW, "junk": object()}]) == [{"o": 1.5}]


@pytest.mark.parametrize(
    ("col", "value", "arrow_type"),
    [
        ("b", True, "int32"),  # bool is not an int value
        ("b", "1", "int32"),
        ("b", 1.0, "int32"),
        ("a", 1, "double"),  # no int -> float coercion
        ("s", 1, "string"),
        ("k", True, "int64"),
    ],
)
def test_no_coercion(col, value, arrow_type):
    """The refusal names the ARROW type the caller declared, not DuckDB's
    spelling of it (decided 2026-08-15)."""
    fn = build("SELECT a AS o FROM __THIS__")
    with pytest.raises(ValueError, match=f"column '{col}'.*{arrow_type}"):
        fn.infer_rows([{**ROW, col: value}])


def test_numpy_scalars_cross_the_boundary():
    """Fixed-width numpy scalars are the natural Python spelling of a narrow
    column, so an integer-like (`__index__`) value crosses — np.bool_ has no
    `__index__` and stays out of the int lane, exactly like a Python bool."""
    np = pytest.importorskip("numpy")
    fn = build("SELECT a + 1.0 AS d, b + b AS n, k AS big FROM __THIS__")
    row = {**ROW, "a": np.float64(1.5), "b": np.int32(2), "k": np.int64(3)}
    assert fn.infer_rows([row]) == [{"d": 2.5, "n": 4, "big": 3}]
    # np.uint8 is integer-like too; range is still checked against the column.
    assert fn.infer_rows([{**row, "b": np.uint8(7)}])[0]["n"] == 14
    with pytest.raises(ValueError, match="column 'b'.*int32"):
        fn.infer_rows([{**row, "b": np.bool_(True)}])
    with pytest.raises(ValueError, match="column 'b' value 3000000000"):
        fn.infer_rows([{**row, "b": np.int64(3_000_000_000)}])


def test_numpy_bool_crosses_the_bool_lane():
    np = pytest.importorskip("numpy")
    fn = build("SELECT NOT f AS o FROM __THIS__", schema=pa.schema([("f", pa.bool_())]))
    assert fn.infer_rows([{"f": np.bool_(True)}]) == [{"o": False}]


def test_infer_arrow_struct_refusal_names_a_live_method():
    schema = pa.schema([pa.field("st", pa.struct([pa.field("x", pa.int64())]))])
    fn = build("SELECT st.x AS o FROM __THIS__", schema=schema)
    with pytest.raises(ValueError, match="infer_rows") as e:
        fn.infer_arrow(pa.Table.from_pylist([{"st": {"x": 1}}], schema=schema))
    assert "row model" not in str(e.value) and "infer()" not in str(e.value)


def test_input_range_refuses_by_name():
    """The message quotes the DECLARATION back — the caller wrote
    `pa.int32()`, so the refusal says `int32`, not DuckDB's `INTEGER`
    (decided 2026-08-15). Arrow is the physical vocabulary at this boundary;
    DuckDB spellings stay in dialect/, which emits SQL text."""
    fn = build("SELECT b AS o FROM __THIS__")
    with pytest.raises(
        ValueError, match="column 'b' value 3000000000 is outside its int32 range"
    ):
        fn.infer_rows([{**ROW, "b": 3_000_000_000}])


def test_none_into_non_nullable_refuses():
    fn = build("SELECT k AS o FROM __THIS__")
    with pytest.raises(ValueError, match="column 'k' is not nullable"):
        fn.infer_rows([{**ROW, "k": None}])


def test_null_is_explicit_none():
    fn = build("SELECT a + 1.0 AS o FROM __THIS__")
    assert fn.infer_rows([{**ROW, "a": None}]) == [{"o": None}]


def test_bool_column_round_trip():
    schema = pa.schema([("f", pa.bool_())])
    fn = build("SELECT NOT f AS o FROM __THIS__", schema=schema)
    assert fn.infer_rows([{"f": True}, {"f": None}]) == [{"o": False}, {"o": None}]


def test_struct_row_column_serves():
    schema = pa.schema([pa.field("st", pa.struct([pa.field("x", pa.int64())]))])
    fn = build("SELECT st.x + 1 AS o FROM __THIS__", schema=schema)
    assert fn.infer_rows([{"st": {"x": 41}}]) == [{"o": 42}]
    assert fn.infer_rows([{"st": None}]) == [{"o": None}]


# TASK-116. The same column used to bind or refuse depending only on which
# table it sat in: served in a ROW table (above), unserved in a STATIC one.
# A static table is already `map(keys) -> value lanes`, and a struct's scalar
# leaves ARE a lane set, so they flatten under their FULL ORDERED PATH.
#
# Full ordered path is the load-bearing part. Keying by leaf name, by name
# set, or by suffix would collapse `w.x.y.z.a` and `w.z.y.x.a` into one lane
# and make `w.a` start finding something instead of erroring. The lookup
# walks the path exactly or misses.
_S116 = pa.struct(
    [
        ("x", pa.struct([("y", pa.struct([("z", pa.struct([("a", pa.int64())]))]))])),
        ("z", pa.struct([("y", pa.struct([("x", pa.struct([("a", pa.int64())]))]))])),
        ("mean", pa.float64()),
    ]
)
_STATIC116 = pa.table(
    {
        "id": pa.array([5], pa.int64()),
        "w": pa.array(
            [{"x": {"y": {"z": {"a": 1}}}, "z": {"y": {"x": {"a": 2}}}, "mean": 2.5}],
            _S116,
        ),
        "v": pa.array([7], pa.int64()),
    }
)
_ROW116 = pa.schema([pa.field("k", pa.int64(), nullable=False)])


def _duck116(sql):
    con = duckdb.connect()
    con.execute("CREATE TABLE __THIS__ (k BIGINT)")
    con.execute("INSERT INTO __THIS__ VALUES (5)")
    con.register("sa", _STATIC116)
    con.execute("CREATE TABLE s AS SELECT * FROM sa")
    return con.execute(sql).to_arrow_table()


@pytest.mark.parametrize(
    "expr",
    [
        "s.w.mean",  # a leaf one hop down
        "s.w.x.y.z.a",  # a deep leaf, and its mirror below
        "s.w.z.y.x.a",
        "s.w.mean + 1",  # a lane is an ordinary operand
        "main.s.w.mean",  # the schema-qualified spelling
    ],
)
def test_struct_static_column_serves_its_lanes(expr):
    sql = f"SELECT {expr} AS o FROM __THIS__ JOIN s ON k = s.id"
    want = _duck116(sql)
    fn = DuckDBInferFn(
        sql, row_tables={"__THIS__": _ROW116}, static_tables={"s": _STATIC116}
    )
    got = fn.infer_arrow(pa.Table.from_pylist([{"k": 5}], schema=_ROW116))
    assert got.to_pylist() == want.to_pylist()
    assert got.schema == want.schema, f"{got.schema} != {want.schema}"


def test_a_static_struct_lane_is_null_on_a_left_miss():
    """A lane inherits the join's nullability like any other value column."""
    sql = "SELECT s.w.mean AS o FROM __THIS__ LEFT JOIN s ON k = s.id"
    fn = DuckDBInferFn(
        sql, row_tables={"__THIS__": _ROW116}, static_tables={"s": _STATIC116}
    )
    assert fn.infer_rows([{"k": 5}, {"k": 6}]) == [{"o": 2.5}, {"o": None}]


@pytest.mark.parametrize(
    ("expr", "match"),
    [
        # a wrong path finds nothing -- it is not resolved by name or suffix
        ("s.w.a", 'Could not find key "a"'),
        ("s.w.x.a", 'Could not find key "a"'),
        # a field asked of a scalar is the pre-struct error, unchanged
        ("s.v.bad", "not a struct"),
        # a struct VALUE is still unserved (so is the row path's); the
        # refusal names it as a struct rather than claiming it is missing
        ("s.w", "is a struct"),
        ("s.w.x", "is a struct"),
    ],
)
def test_a_static_struct_path_that_is_not_a_lane_refuses_by_name(expr, match):
    with pytest.raises(ValueError, match=match) as e:
        DuckDBInferFn(
            f"SELECT {expr} AS o FROM __THIS__ JOIN s ON k = s.id",
            row_tables={"__THIS__": _ROW116},
            static_tables={"s": _STATIC116},
        )
    assert "does not exist" not in str(e.value), str(e.value)


# TASK-132. The lane's PATH is data; the dotted spelling is display-only.
# The RFC's collision table: a struct leaf and a literal column that share
# the dotted spelling are DIFFERENT references (quoting is resolved by the
# lexer, so `s."w.mean"` is two parts and `s.w.mean` is three) and DuckDB
# serves both. A dotted name-lookup encoding cannot; a structured path can.
_S132 = pa.struct([("mean", pa.float64())])
_COLLIDE132 = pa.table(
    {
        "id": pa.array([5], pa.int64()),
        "w": pa.array([{"mean": 1.5}], _S132),
        "w.mean": pa.array([99.0], pa.float64()),
    }
)
_STRUCT_ONLY132 = pa.table(
    {
        "id": pa.array([5], pa.int64()),
        "w": pa.array([{"mean": 1.5}], _S132),
    }
)


def _duck132(sql, static):
    con = duckdb.connect()
    con.execute("CREATE TABLE __THIS__ (k BIGINT)")
    con.execute("INSERT INTO __THIS__ VALUES (5)")
    con.register("sa", static)
    con.execute("CREATE TABLE s AS SELECT * FROM sa")
    return con.execute(sql).to_arrow_table()


@pytest.mark.parametrize(
    "expr",
    [
        "s.w.mean",  # three parts: the struct leaf (1.5)
        's."w.mean"',  # two parts: the literal column (99.0)
    ],
)
def test_the_collision_table_serves_both_spellings(expr):
    sql = f"SELECT {expr} AS o FROM __THIS__ JOIN s ON k = s.id"
    want = _duck132(sql, _COLLIDE132)
    fn = DuckDBInferFn(
        sql, row_tables={"__THIS__": _ROW116}, static_tables={"s": _COLLIDE132}
    )
    got = fn.infer_arrow(pa.Table.from_pylist([{"k": 5}], schema=_ROW116))
    assert got.to_pylist() == want.to_pylist()
    assert got.schema == want.schema, f"{got.schema} != {want.schema}"


def test_a_quoted_dotted_name_is_not_a_struct_leaf():
    """Without a literal column, `s."w.mean"` is a name lookup that MISSES:
    the leaf is reachable only through the 3-part spelling, like DuckDB."""
    sql = 'SELECT s."w.mean" AS o FROM __THIS__ JOIN s ON k = s.id'
    con = duckdb.connect()
    con.register("sa", _STRUCT_ONLY132)
    con.execute("CREATE TABLE s AS SELECT * FROM sa")
    with pytest.raises(duckdb.Error):
        con.execute('SELECT s."w.mean" FROM s')
    with pytest.raises(ValueError, match="does not exist"):
        DuckDBInferFn(
            sql, row_tables={"__THIS__": _ROW116}, static_tables={"s": _STRUCT_ONLY132}
        )


def test_a_plain_static_column_with_a_dot_in_its_name_serves():
    """No struct anywhere -- just a column named 'a.b'. The data path must
    read THE COLUMN, not walk row['a']['b']."""
    static = pa.table(
        {
            "id": pa.array([5], pa.int64()),
            "a.b": pa.array([42], pa.int64()),
        }
    )
    sql = 'SELECT s."a.b" AS o FROM __THIS__ JOIN s ON k = s.id'
    want = _duck132(sql, static)
    fn = DuckDBInferFn(
        sql, row_tables={"__THIS__": _ROW116}, static_tables={"s": static}
    )
    got = fn.infer_arrow(pa.Table.from_pylist([{"k": 5}], schema=_ROW116))
    assert got.to_pylist() == want.to_pylist() == [{"o": 42}]


def test_a_non_ascii_field_name_misses_cleanly():
    """A miss beside a non-ASCII field must be the named refusal, not a
    byte-boundary panic: the probe 'w.z.' is four BYTES, which lands in the
    middle of the é of the real lane 'w.zé.y'."""
    static = pa.table(
        {
            "id": pa.array([5], pa.int64()),
            "w": pa.array(
                [{"zé": {"y": 2.0}}],
                pa.struct([("zé", pa.struct([("y", pa.float64())]))]),
            ),
        }
    )
    with pytest.raises(ValueError, match="Could not find key"):
        DuckDBInferFn(
            "SELECT s.w.z AS o FROM __THIS__ JOIN s ON k = s.id",
            row_tables={"__THIS__": _ROW116},
            static_tables={"s": static},
        )


def test_an_unreferenced_static_struct_still_builds():
    fn = DuckDBInferFn(
        "SELECT s.v AS o FROM __THIS__ JOIN s ON k = s.id",
        row_tables={"__THIS__": _ROW116},
        static_tables={"s": _STATIC116},
    )
    assert fn.infer_rows([{"k": 5}]) == [{"o": 7}]


def test_a_static_table_beats_a_row_struct_column_of_the_same_name():
    """Measured on DuckDB: with a TABLE named `w` in scope, `w.mean` is that
    table's column, not the struct's field. Qualifying forces the struct."""
    row = pa.schema(
        [
            pa.field("k", pa.int64(), nullable=False),
            pa.field("w", pa.struct([("mean", pa.float64())])),
        ]
    )
    static = pa.table(
        {"id": pa.array([5], pa.int64()), "mean": pa.array([99.0], pa.float64())}
    )
    rows = [{"k": 5, "w": {"mean": 2.0}}]
    tbl = {"__THIS__": row}
    st = {"w": static}

    con = duckdb.connect()
    con.execute("CREATE TABLE __THIS__ (k BIGINT, w STRUCT(mean DOUBLE))")
    con.execute("INSERT INTO __THIS__ VALUES (5, {'mean': 2.0})")
    con.register("wa", static)
    con.execute("CREATE TABLE w AS SELECT * FROM wa")

    for expr in ["w.mean", "__THIS__.w.mean"]:
        sql = f"SELECT {expr} AS o FROM __THIS__ JOIN w ON k = w.id"
        want = con.execute(sql).to_arrow_table().to_pylist()
        got = DuckDBInferFn(sql, row_tables=tbl, static_tables=st).infer_rows(rows)
        assert got == want, f"{expr}: {got} != {want}"
    assert con.execute(
        "SELECT w.mean AS o FROM __THIS__ JOIN w ON k = w.id"
    ).fetchall() == [(99.0,)], "oracle moved — the table no longer wins"


def test_foreign_type_unreferenced_builds_referenced_refuses():
    schema = pa.schema(
        [("a", pa.float64()), ("t", pa.timestamp("us")), ("v", pa.float32())]
    )
    fn = build("SELECT a AS o FROM __THIS__", schema=schema)
    assert fn.infer_rows([{"a": 1.0, "t": None, "v": None}]) == [{"o": 1.0}]
    with pytest.raises(ValueError, match="v"):
        build("SELECT v AS o FROM __THIS__", schema=schema)


def test_constant_static_only_query_serves_via_infer_rows():
    statics = {"s": pa.table({"v": pa.array([1, 2, 3], pa.int64())})}
    fn = DuckDBInferFn(
        "SELECT sum(v) AS o FROM s",
        row_tables={"__THIS__": SCHEMA},
        static_tables=statics,
    )
    assert fn.backend == "constant"
    assert fn.infer_rows([]) == [{"o": 6}]


def test_empty_rows_in_empty_rows_out():
    fn = build("SELECT a AS o FROM __THIS__")
    assert fn.infer_rows([]) == []


def test_infer_and_output_model_are_gone():
    fn = build("SELECT a AS o FROM __THIS__")
    assert not hasattr(fn, "infer")
    assert not hasattr(fn, "output_model")
    assert not hasattr(fn, "output")
    with pytest.raises(TypeError):
        build("SELECT a AS o FROM __THIS__", output="dict")
    with pytest.raises(TypeError):
        build("SELECT a AS o FROM __THIS__", output_model=object)


def test_non_arrow_schema_refuses_by_name():
    class Row:
        a: float

    with pytest.raises(ValueError, match="pyarrow.Schema"):
        build("SELECT a AS o FROM __THIS__", schema=Row)


# TASK-110: a static-tables-only query compiles to a fixed answer, so it
# structurally cannot see input rows. It used to DROP them without a word —
# the one input mistake at this boundary that did not refuse by name, and
# the one that hides a real caller bug: serving N request rows through a
# function that cannot read them returns 1 fixed row, and the caller's
# zip/positional assumption breaks somewhere downstream instead of here.
def _constant_fn(shape=None):
    statics = {"s": pa.table({"v": pa.array([1, 2, 3], pa.int64())})}
    kw = {"shape": shape} if shape else {}
    return DuckDBInferFn(
        "SELECT sum(v) AS o FROM s",
        row_tables={"__THIS__": SCHEMA},
        static_tables=statics,
        **kw,
    )


def test_constant_build_refuses_rows_it_cannot_read():
    fn = _constant_fn()
    assert fn.backend == "constant"
    with pytest.raises(ValueError, match="infer_rows"):
        fn.infer_rows([ROW])
    with pytest.raises(ValueError, match="infer_rows"):
        fn.infer_rows([ROW] * 3)


def test_constant_build_still_serves_on_empty_rows():
    fn = _constant_fn()
    assert fn.infer_rows([]) == [{"o": 6}]
    assert fn.infer_rows([]) == [{"o": 6}]  # repeatable, fresh dict each call


def test_constant_refusal_names_the_query_shape():
    fn = _constant_fn()
    with pytest.raises(ValueError) as e:
        fn.infer_rows([ROW])
    msg = str(e.value)
    assert "static" in msg and "infer_rows([])" in msg, msg


def test_constant_refusal_holds_under_shape_many():
    """shape='map' already refuses to BUILD a constant engine (fixed rows
    cannot be one-out-per-row-in), so only the default and 'many' shapes
    reach this boundary at all."""
    fn = _constant_fn(shape="many")
    assert fn.backend == "constant"
    with pytest.raises(ValueError, match="infer_rows"):
        fn.infer_rows([ROW])
    assert fn.infer_rows([]) == [{"o": 6}]


def test_compiled_build_is_untouched_by_the_constant_guard():
    fn = build("SELECT a + 1.0 AS o FROM __THIS__")
    assert fn.backend != "constant"
    assert fn.infer_rows([ROW, ROW]) == [{"o": 2.5}, {"o": 2.5}]


# --------------------------------------------------------- the static star --
#
# TASK-125. A star over a static relation expands the DECLARED column list, in
# declared order. We serve no struct and no non-vocabulary value, so a star
# that covers one refuses NAMING it -- never a different column set: expanding
# a struct's leaves invents columns DuckDB does not output, and dropping an
# opaque column removes one it does. EXCLUDE takes the column out of the star
# before that check, exactly as it does on the row path.
_ROW125 = pa.schema([pa.field("k", pa.int64(), nullable=False)])
_STAR125 = {
    "struct": pa.table(
        {
            "id": pa.array([1], pa.int64()),
            "w": pa.array(
                [{"mean": 1.5, "sd": 0.25}],
                pa.struct([("mean", pa.float64()), ("sd", pa.float64())]),
            ),
            "z": pa.array([7], pa.int64()),
        }
    ),
    "opaque": pa.table(
        {
            "id": pa.array([1], pa.int64()),
            "ts": pa.array([0], pa.timestamp("us")),
            "z": pa.array([7], pa.int64()),
        }
    ),
}
_STAR125_DDL = {
    "struct": (
        "CREATE TABLE s (id BIGINT, w STRUCT(mean DOUBLE, sd DOUBLE), z BIGINT)",
        "INSERT INTO s VALUES (1, {'mean': 1.5, 'sd': 0.25}, 7)",
    ),
    "opaque": (
        "CREATE TABLE s (id BIGINT, ts TIMESTAMP, z BIGINT)",
        "INSERT INTO s VALUES (1, TIMESTAMP '1970-01-01', 7)",
    ),
}


def _duck125(sql: str, kind: str):
    con = duckdb.connect()
    con.execute("CREATE TABLE __THIS__ (k BIGINT)")
    con.execute("INSERT INTO __THIS__ VALUES (1)")
    ddl, ins = _STAR125_DDL[kind]
    con.execute(ddl)
    con.execute(ins)
    res = con.execute(sql)
    return [d[0] for d in res.description], res.fetchall()


@pytest.mark.parametrize("star", ["s.*", "*"])
@pytest.mark.parametrize(("kind", "unservable"), [("struct", "w"), ("opaque", "ts")])
def test_a_static_star_refuses_a_column_it_cannot_serve(star, kind, unservable):
    sql = f"SELECT {star} FROM __THIS__ JOIN s ON s.id = __THIS__.k"
    names, _ = _duck125(sql, kind)  # oracle serves the column whole
    assert unservable in names

    with pytest.raises(ValueError, match=unservable) as e:
        DuckDBInferFn(
            sql, row_tables={"__THIS__": _ROW125}, static_tables={"s": _STAR125[kind]}
        )
    assert "does not exist" not in str(e.value), str(e.value)


@pytest.mark.parametrize(("kind", "unservable"), [("struct", "w"), ("opaque", "ts")])
def test_exclude_lets_a_static_star_drop_what_it_cannot_serve(kind, unservable):
    """TASK-127 AC #1: the star entry restores the NAME, so EXCLUDE can take
    the column out before the refusal fires -- and the rest serves."""
    sql = f"SELECT s.* EXCLUDE ({unservable}) FROM __THIS__ JOIN s ON s.id = __THIS__.k"
    names, want = _duck125(sql, kind)
    assert names == ["id", "z"] and want == [(1, 7)]

    fn = DuckDBInferFn(
        sql, row_tables={"__THIS__": _ROW125}, static_tables={"s": _STAR125[kind]}
    )
    assert [tuple(x.values()) for x in fn.infer_rows([{"k": 1}])] == want


def test_a_scalar_only_static_star_expands_in_declared_order():
    static = pa.table(
        {
            "id": pa.array([1], pa.int64()),
            "b": pa.array([2.5], pa.float64()),
            "a": pa.array([9], pa.int64()),
        }
    )
    sql = "SELECT s.* FROM __THIS__ JOIN s ON s.id = __THIS__.k"
    con = duckdb.connect()
    con.execute("CREATE TABLE __THIS__ (k BIGINT)")
    con.execute("INSERT INTO __THIS__ VALUES (1)")
    con.execute("CREATE TABLE s (id BIGINT, b DOUBLE, a BIGINT)")
    con.execute("INSERT INTO s VALUES (1, 2.5, 9)")
    res = con.execute(sql)
    names, want = [d[0] for d in res.description], res.fetchall()
    assert names == ["id", "b", "a"]

    fn = DuckDBInferFn(
        sql, row_tables={"__THIS__": _ROW125}, static_tables={"s": static}
    )
    rows = fn.infer_rows([{"k": 1}])
    assert list(rows[0].keys()) == names
    assert [tuple(x.values()) for x in rows] == want


# ------------------------------------------- row limits on the constant path --
#
# TASK-128. A static-tables-only query is evaluated ONCE at build by DuckDB
# and frozen. A row limit picks WHICH rows survive, and without a total order
# that pick is not a function of the query: measured 2026-08-19, the same
# `GROUP BY ... FETCH FIRST 1 ROWS ONLY` over the same four rows returned
# FOUR distinct answers across twelve fresh connections -- and ORDER BY does
# not fix it in general (a tie fed from a GROUP BY flipped in 20 runs). So
# the constant path refuses EVERY row limit, ORDER BY or not (decision (a),
# 2026-08-19); the provably-total case can be layered on if ever needed.
@pytest.mark.parametrize(
    "sql",
    [
        "SELECT v AS o FROM s LIMIT 1",
        "SELECT v AS o FROM s LIMIT 1 OFFSET 1",
        "SELECT v AS o FROM s OFFSET 1",
        "SELECT v AS o FROM s FETCH FIRST 1 ROWS ONLY",
        "SELECT TOP 1 v AS o FROM s",
        "SELECT v AS o, sum(v) AS t FROM s GROUP BY v FETCH FIRST 1 ROWS ONLY",
        # ORDER BY does NOT lift the refusal -- decision (a)
        "SELECT v AS o FROM s ORDER BY v LIMIT 1",
    ],
)
def test_a_row_limit_on_the_constant_path_refuses(sql):
    statics = {"s": pa.table({"v": pa.array([1, 2, 3], pa.int64())})}
    with pytest.raises(ValueError, match="row limit"):
        DuckDBInferFn(sql, row_tables={"__THIS__": SCHEMA}, static_tables=statics)


def test_the_constant_path_without_a_limit_is_untouched():
    statics = {"s": pa.table({"v": pa.array([1, 2, 3], pa.int64())})}
    fn = DuckDBInferFn(
        "SELECT sum(v) AS o FROM s ORDER BY 1",
        row_tables={"__THIS__": SCHEMA},
        static_tables=statics,
    )
    assert fn.backend == "constant"
    assert fn.infer_rows([]) == [{"o": 6}]


# ------------------------------------------ bare-name ambiguity (TASK-121) --
#
# DuckDB decides AMBIGUITY on the bare HEAD name before it looks at struct
# fields or lanes, and a static STRUCT's name binds for that purpose even
# though we serve no struct value. Three spellings, one rule; closed
# 2026-08-19, formerly the largest single class the campaign saw (78/161
# findings at 20k seeds).
def _ambig(sql, row, static):
    import duckdb as _duck

    con = _duck.connect()
    con.execute("CREATE TABLE __THIS__ (c0 STRUCT(f0 BIGINT), v BIGINT)")
    con.execute("INSERT INTO __THIS__ VALUES ({'f0': 1}, 2)")
    con.register("sa", static)
    con.execute("CREATE TABLE s0 AS SELECT * FROM sa")
    with pytest.raises(_duck.Error, match="[Aa]mbiguous"):
        con.execute(sql).fetchall()  # oracle refuses; if this stops, remeasure

    with pytest.raises(ValueError, match="[Aa]mbiguous"):
        DuckDBInferFn(sql, row_tables={"__THIS__": row}, static_tables={"s0": static})


def test_a_struct_path_head_that_binds_in_a_join_scope_is_ambiguous():
    row = pa.schema(
        [
            pa.field("c0", pa.struct([("f0", pa.int64())])),
            pa.field("v", pa.int64(), nullable=False),
        ]
    )
    static = pa.table({"c0": pa.array([1], pa.int64()), "w": pa.array([9], pa.int64())})
    _ambig("SELECT v AS o FROM __THIS__ LEFT JOIN s0 ON (c0.f0 = s0.c0)", row, static)


def test_a_bare_scalar_vs_a_static_struct_is_ambiguous():
    # the seed-2023 spelling: the row SCALAR c0 collides with s0's STRUCT c0,
    # whose bare name lives in the opaque list rather than the lane set
    import duckdb as _duck

    row = pa.schema(
        [pa.field("c0", pa.int64(), nullable=False), pa.field("k", pa.int64())]
    )
    static = pa.table(
        {
            "c0": pa.array(
                [{"f0": 1.5, "f1": 3}],
                pa.struct([("f0", pa.float64()), ("f1", pa.int64())]),
            ),
        }
    )
    sql = "SELECT coalesce(c0, 7) AS o FROM __THIS__ LEFT JOIN s0 ON (k = s0.c0.f1)"

    con = _duck.connect()
    con.execute("CREATE TABLE __THIS__ (c0 BIGINT, k BIGINT)")
    con.execute("INSERT INTO __THIS__ VALUES (5, 3)")
    con.register("sa", static)
    con.execute("CREATE TABLE s0 AS SELECT * FROM sa")
    with pytest.raises(_duck.Error, match="[Aa]mbiguous"):
        con.execute(sql).fetchall()  # oracle refuses; if this stops, remeasure

    with pytest.raises(ValueError, match="[Aa]mbiguous"):
        DuckDBInferFn(sql, row_tables={"__THIS__": row}, static_tables={"s0": static})


def test_a_sole_static_struct_bare_name_refuses_as_a_struct_not_as_missing():
    # no collision: the bare name resolves to the static STRUCT alone, and
    # the refusal must name the struct, never claim the column is missing
    row = pa.schema([pa.field("k", pa.int64(), nullable=False)])
    static = pa.table(
        {
            "id": pa.array([1], pa.int64()),
            "w": pa.array([{"m": 1.5}], pa.struct([("m", pa.float64())])),
        }
    )
    with pytest.raises(ValueError, match="is a struct") as e:
        DuckDBInferFn(
            "SELECT w AS o FROM __THIS__ JOIN s0 ON k = s0.id",
            row_tables={"__THIS__": row},
            static_tables={"s0": static},
        )
    assert "does not exist" not in str(e.value)


# --------------------------------- the unqualified ladder (TASK-127) --
#
# DuckDB's two-part `a.b` reads table-then-column FIRST and falls through to
# column-then-field only when the relation matched and its COLUMN half
# missed (bind_context.cpp:360-363). We had no fall-through at all, and a
# static struct head was invisible to the bare-name rung -- so every
# reference DuckDB resolves that way came back as `unknown table 'w'`.
_S127 = pa.struct(
    [
        ("mean", pa.float64()),
        ("sd", pa.float64()),
        ("inner", pa.struct([("val", pa.float64())])),
    ]
)
_STATIC127 = pa.table(
    {
        "id": pa.array([5], pa.int64()),
        "w": pa.array([{"mean": 1.5, "sd": 0.5, "inner": {"val": 9.0}}], _S127),
        "z": pa.array([7], pa.int64()),
    }
)


def _serves127(sql, static=_STATIC127):
    """Both engines answer, and the OUTPUT NAME is half the answer: an
    unqualified path is named by its last part, spelled as typed."""
    want = _duck132(sql, static)
    fn = DuckDBInferFn(
        sql, row_tables={"__THIS__": _ROW116}, static_tables={"s": static}
    )
    got = fn.infer_arrow(pa.Table.from_pylist([{"k": 5}], schema=_ROW116))
    assert got.to_pylist() == want.to_pylist()
    assert got.schema.names == want.schema.names
    assert got.schema == want.schema, f"{got.schema} != {want.schema}"


@pytest.mark.parametrize(
    "sql",
    [
        # SELECT: the head is a static struct nobody qualified
        "SELECT w.mean FROM __THIS__ JOIN s ON k = s.id",
        "SELECT W.MEAN FROM __THIS__ JOIN s ON k = s.id",
        "SELECT w.inner.val FROM __THIS__ JOIN s ON k = s.id",
        "SELECT w.sd + 1 AS o FROM __THIS__ JOIN s ON k = s.id",
        # the same heads in a WHERE
        "SELECT z AS o FROM __THIS__ JOIN s ON k = s.id WHERE w.mean > 1.0",
        "SELECT z AS o FROM __THIS__ JOIN s ON k = s.id WHERE w.inner.val > 1.0",
        "SELECT z AS o FROM __THIS__ JOIN s ON k = s.id WHERE w.sd < 1.0",
        # and in an ON residual
        "SELECT z AS o FROM __THIS__ JOIN s ON k = s.id AND w.mean > 1.0",
        "SELECT z AS o FROM __THIS__ JOIN s ON k = s.id AND w.inner.val > 1.0",
        "SELECT z AS o FROM __THIS__ JOIN s ON k = s.id AND w.sd < 1.0",
    ],
)
def test_an_unqualified_static_struct_path_serves(sql):
    _serves127(sql)


@pytest.mark.parametrize(
    ("expr", "match", "names_a_table"),
    [
        # the head bound: the miss is about the KEY, not about a table
        ("w.nope", 'Could not find key "nope"', False),
        ("w.inner.nope", 'Could not find key "nope"', False),
        ("z.bad", "not a struct", False),
        # a head in no scope at all still says so
        ("q.mean", "unknown table", True),
    ],
)
def test_an_unqualified_static_struct_miss_names_the_key(expr, match, names_a_table):
    sql = f"SELECT {expr} AS o FROM __THIS__ JOIN s ON k = s.id"
    with pytest.raises(duckdb.Error):
        _duck132(sql, _STATIC127)  # oracle refuses; if this stops, remeasure
    with pytest.raises(ValueError, match=match) as e:
        DuckDBInferFn(
            sql, row_tables={"__THIS__": _ROW116}, static_tables={"s": _STATIC127}
        )
    if not names_a_table:
        assert "unknown table" not in str(e.value), str(e.value)


_W127 = pa.struct([("mean", pa.float64())])
_SW127 = pa.table(
    {"id": pa.array([5], pa.int64()), "w": pa.array([{"mean": 1.5}], _W127)}
)
_SW2127 = pa.table(
    {"id": pa.array([5], pa.int64()), "w": pa.array([{"mean": 2.5}], _W127)}
)
_SWSCALAR127 = pa.table(
    {"id": pa.array([5], pa.int64()), "w": pa.array([9.0], pa.float64())}
)
_ROWW127 = pa.schema(
    [pa.field("k", pa.int64(), nullable=False), pa.field("w", pa.int64())]
)
_ROWWSTRUCT127 = pa.schema(
    [pa.field("k", pa.int64(), nullable=False), pa.field("w", _W127)]
)
_TWO127 = "SELECT w.mean AS o FROM __THIS__ JOIN s ON k = s.id JOIN s2 ON k = s2.id"


def _both_refuse_ambiguously(sql, row_ddl, row_schema, statics):
    con = duckdb.connect()
    con.execute(f"CREATE TABLE __THIS__ ({row_ddl})")
    for name, tbl in statics.items():
        con.register(f"{name}_a", tbl)
        con.execute(f"CREATE TABLE {name} AS SELECT * FROM {name}_a")
    with pytest.raises(duckdb.Error, match="[Aa]mbiguous"):
        con.execute(sql).fetchall()  # oracle refuses; if this stops, remeasure
    with pytest.raises(ValueError, match="[Aa]mbiguous"):
        DuckDBInferFn(sql, row_tables={"__THIS__": row_schema}, static_tables=statics)


@pytest.mark.parametrize(
    ("row_ddl", "row_schema", "statics", "sql"),
    [
        # a row SCALAR beside a static struct head
        (
            "k BIGINT, w BIGINT",
            _ROWW127,
            {"s": _SW127},
            "SELECT w.mean AS o FROM __THIS__ JOIN s ON k = s.id",
        ),
        # a row STRUCT beside a static struct head
        (
            "k BIGINT, w STRUCT(mean DOUBLE)",
            _ROWWSTRUCT127,
            {"s": _SW127},
            "SELECT w.mean AS o FROM __THIS__ JOIN s ON k = s.id",
        ),
        # two statics, both carrying a struct head `w`
        ("k BIGINT", _ROW116, {"s": _SW127, "s2": _SW2127}, _TWO127),
        # a static struct head beside a static SCALAR of the same name:
        # the verdict is on the HEAD, before any field is examined
        ("k BIGINT", _ROW116, {"s": _SW127, "s2": _SWSCALAR127}, _TWO127),
    ],
)
def test_an_ambiguous_unqualified_head_refuses_before_the_fields(
    row_ddl, row_schema, statics, sql
):
    _both_refuse_ambiguously(sql, row_ddl, row_schema, statics)


@pytest.mark.parametrize(
    ("expr", "want"),
    [
        ("w.z", 7),  # the ALIAS wins -- its column `z` binds
        ("w.mean", 1.5),  # the alias has no `mean`: backtrack to column `w`
        ("w.w.mean", 1.5),  # spelled out: alias `w`, column `w`, field `mean`
    ],
)
def test_a_join_alias_backtracks_to_a_struct_column(expr, want):
    sql = f"SELECT {expr} AS o FROM __THIS__ JOIN s AS w ON k = w.id"
    assert _duck132(sql, _STATIC127).to_pylist() == [{"o": want}]
    fn = DuckDBInferFn(
        sql, row_tables={"__THIS__": _ROW116}, static_tables={"s": _STATIC127}
    )
    assert fn.infer_rows([{"k": 5}]) == [{"o": want}]


def test_a_join_alias_beside_a_row_column_of_the_same_name_is_ambiguous():
    """The backtrack never runs past an ambiguity: with a row column `w`
    too, the head has two bindings and DuckDB refuses on the head alone."""
    _both_refuse_ambiguously(
        "SELECT w.mean AS o FROM __THIS__ JOIN s AS w ON k = w.id",
        "k BIGINT, w BIGINT",
        _ROWW127,
        {"s": _SW127},
    )


def test_a_row_struct_leaf_and_a_dotted_sibling_both_serve():
    """The ROW side of the RFC's collision table (TASK-127's collision
    criterion). A leaf lane's dotted name is DISPLAY post-132, so it is no
    longer a duplicate identifier: the table must build -- including for a
    query that touches neither column -- and both spellings serve."""
    row = pa.schema(
        [
            pa.field("k", pa.int64(), nullable=False),
            pa.field("w", pa.struct([("mean", pa.float64())])),
            pa.field("w.mean", pa.float64()),
        ]
    )
    rows = [{"k": 5, "w": {"mean": 1.5}, "w.mean": 99.0}]
    con = duckdb.connect()
    con.execute(
        'CREATE TABLE __THIS__ (k BIGINT, w STRUCT(mean DOUBLE), "w.mean" DOUBLE)'
    )
    con.execute("INSERT INTO __THIS__ VALUES (5, {'mean': 1.5}, 99.0)")
    for expr, want in [("k", 5), ("w.mean", 1.5), ('"w.mean"', 99.0)]:
        sql = f"SELECT {expr} AS o FROM __THIS__"
        assert con.execute(sql).fetchall() == [(want,)], f"oracle moved: {expr}"
        got = DuckDBInferFn(
            sql, row_tables={"__THIS__": row}, static_tables={}
        ).infer_rows(rows)
        assert got == [{"o": want}], f"{expr}: {got}"


def test_two_plain_row_columns_of_the_same_name_refuse_by_name():
    """The other half of D4: a name that IS an identifier still cannot
    repeat, and it refuses at build naming the column -- not as an internal
    verifier bug."""
    row = pa.schema([pa.field("a", pa.int64()), pa.field("a", pa.int64())])
    with pytest.raises(ValueError, match="two columns named 'a'") as e:
        DuckDBInferFn(
            "SELECT 1 AS o FROM __THIS__",
            row_tables={"__THIS__": row},
            static_tables={},
        )
    assert "internal" not in str(e.value).lower(), str(e.value)


# ------------------------------- refusals quoted from DuckDB (TASK-127) --
#
# Both engines already refused in these two cells; only our wording was
# poorer. DuckDB's words carry more information, so we adopt them.
def test_a_not_a_struct_refusal_enumerates_what_duckdb_enumerates():
    sql = "SELECT z.bad AS o FROM __THIS__ JOIN s ON k = s.id"
    with pytest.raises(duckdb.Error) as oracle:
        _duck132(sql, _STATIC127)
    assert "not a struct, union, map, or json" in str(oracle.value)
    with pytest.raises(ValueError, match="not a struct, union, map, or json"):
        DuckDBInferFn(
            sql, row_tables={"__THIS__": _ROW116}, static_tables={"s": _STATIC127}
        )


@pytest.mark.parametrize(
    ("star", "scope"),
    [
        ("s.*", "'s'"),  # a qualified star searched ONE relation: name it
        ("*", "FROM clause"),  # an unqualified star searched all of them
    ],
)
def test_an_exclude_miss_names_the_scope_it_searched(star, scope):
    sql = f"SELECT {star} EXCLUDE (nope) FROM __THIS__ JOIN s ON k = s.id"
    with pytest.raises(duckdb.Error) as oracle:
        _duck132(sql, _STATIC127)
    assert "in EXCLUDE list not found in" in str(oracle.value)
    with pytest.raises(ValueError, match=f"in EXCLUDE list not found in {scope}"):
        DuckDBInferFn(
            sql, row_tables={"__THIS__": _ROW116}, static_tables={"s": _STATIC127}
        )
