"""The columnar boundary: infer_arrow == infer_rows, always.

The differential is the contract: same engine, same program, two
boundaries — parity is by construction, this suite makes drift
impossible.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pyarrow as pa
import pytest
from confit import DuckDBInferFn

from benchmarks import serving_scenarios as sc

T_SCHEMA = pa.schema(
    [
        pa.field("a", pa.int64(), nullable=False),
        pa.field("s", pa.string()),
        pa.field("f", pa.float64()),
    ]
)
ROWS = [
    {"a": 1, "s": "héllo", "f": 1.5},
    {"a": 2, "s": None, "f": None},
    {"a": 3, "s": "", "f": -0.0},
]


def _table(rows, schema=None):
    return pa.Table.from_pylist(rows, schema=schema)


def _cmp(fn, rows, tbl):
    got_rows = fn.infer_rows(rows)
    got_arrow = fn.infer_arrow(tbl).to_pylist()
    assert got_rows == got_arrow


def _duck_arrow(mod, statics, rows_d):
    """The same scenario run by DuckDB itself, as an arrow table — the oracle
    for the OUTPUT schema, not just the values."""
    import duckdb

    con = duckdb.connect()
    for name, table in statics.items():
        con.register(f"__arrow_{name}", table)
        con.execute(f'CREATE TABLE "{name}" AS SELECT * FROM "__arrow_{name}"')  # noqa: S608
        con.unregister(f"__arrow_{name}")
    con.register("__THIS__", sc.rows_table(mod, rows_d))
    return con.execute(mod.SQL).to_arrow_table()


def test_differential_basic_and_nulls():
    fn = DuckDBInferFn(
        "SELECT a + 1 AS b, upper(s) AS u, f * 2 AS d FROM __THIS__ WHERE a < 3",
        row_tables={"__THIS__": T_SCHEMA},
        static_tables={},
    )
    _cmp(fn, ROWS, _table(ROWS))


def test_differential_every_serving_scenario():
    for mod in sc.all_scenarios():
        statics = mod.make_statics(sc.SEED)
        fn = sc.build_spec_fn(mod, statics)
        rows_d = mod.make_rows(sc.SEED + 7, 64)
        tbl = sc.rows_table(mod, rows_d)
        _cmp(fn, rows_d, tbl)


def test_differential_shape_many_left_join():
    dup = pa.table({"id": [1, 1, 2], "v": [10, 11, 20]})
    fn = DuckDBInferFn(
        "SELECT a, v FROM __THIS__ LEFT JOIN d ON a = d.id",
        row_tables={"__THIS__": T_SCHEMA},
        static_tables={"d": dup},
        shape="many",
    )
    _cmp(fn, ROWS, _table(ROWS))


def test_named_rejections():
    fn = DuckDBInferFn(
        "SELECT a FROM __THIS__",
        row_tables={"__THIS__": T_SCHEMA},
        static_tables={},
    )
    two_chunks = pa.Table.from_batches(
        [pa.record_batch({"a": [1]}), pa.record_batch({"a": [2]})]
    )
    with pytest.raises(ValueError, match="combine_chunks"):
        fn.infer_arrow(
            pa.Table.from_batches(
                [
                    pa.record_batch(_table(ROWS[:1]).to_batches()[0]),
                    pa.record_batch(_table(ROWS[1:]).to_batches()[0]),
                ]
            )
        )
    del two_chunks
    with pytest.raises(ValueError, match="missing column"):
        fn.infer_arrow(pa.table({"x": [1]}))
    with pytest.raises(ValueError, match="cast first"):
        fn.infer_arrow(
            _table(
                ROWS,
                schema=pa.schema(
                    [("a", pa.int32()), ("s", pa.string()), ("f", pa.float64())]
                ),
            )
        )
    # Non-nullable model column with NULLs in the batch.
    with pytest.raises(ValueError, match="not nullable"):
        fn.infer_arrow(
            _table(
                [{"a": None, "s": "x", "f": 0.0}],
                schema=pa.schema(
                    [("a", pa.int64()), ("s", pa.string()), ("f", pa.float64())]
                ),
            )
        )
    # A struct row column is NOT rejected here: infer_arrow walks the lane
    # path, so it serves. The struct pins are at the bottom of this file.


# There is no "infer_arrow refuses a supplied output model" pin because
# output_model= no longer exists: nothing can be supplied, so there is no
# separate refusal path to differential-test. The premise ("a supplied model
# flips infer_arrow's behaviour") is covered once, at the constructor, by
# test_arrow_schema_api.py::test_infer_and_output_model_are_gone.


# Integer widths are typed for real (m-8 phase 2), so the schema suite allows
# NO differences and this is a plain parity test rather than a width pin. The
# catalogue lives in test_integer_widths.py.
def test_infer_arrow_integer_width_matches_duckdb():
    import duckdb

    in_schema = pa.schema([pa.field("k", pa.int64(), nullable=False)])
    sql = "SELECT CASE WHEN k > 1 THEN 1 ELSE 0 END AS c FROM __THIS__"
    fn = DuckDBInferFn(sql, row_tables={"__THIS__": in_schema}, static_tables={})
    got = fn.infer_arrow(pa.table({"k": [0, 2]}))
    con = duckdb.connect()
    con.execute("CREATE TABLE __THIS__ (k BIGINT)")
    con.execute("INSERT INTO __THIS__ VALUES (0), (2)")
    want = con.execute(sql).to_arrow_table()
    assert got.to_pylist() == want.to_pylist()  # values already agree
    assert got.schema == want.schema


def test_output_schema_matches_duckdb_for_every_scenario():
    """Values agreeing is not enough: a schema that does not match DuckDB's
    cannot be stacked with DuckDB's own output."""
    for mod in sc.all_scenarios():
        statics = mod.make_statics(sc.SEED)
        fn = sc.build_spec_fn(mod, statics)
        rows_d = mod.make_rows(sc.SEED + 7, 16)
        got = fn.infer_arrow(sc.rows_table(mod, rows_d))
        want = _duck_arrow(mod, statics, rows_d)
        assert got.schema.names == want.schema.names, mod.__name__
        for ours, theirs in zip(got.schema.types, want.schema.types, strict=True):
            assert ours == theirs, mod.__name__


def test_sliced_and_recordbatch_inputs():
    fn = DuckDBInferFn(
        "SELECT a, s, f FROM __THIS__",
        row_tables={"__THIS__": T_SCHEMA},
        static_tables={},
    )
    tbl = _table(ROWS * 3)
    sliced = tbl.slice(2, 4).combine_chunks()
    assert fn.infer_arrow(sliced).to_pylist() == sliced.to_pylist()
    rb = tbl.to_batches()[0]
    assert fn.infer_arrow(rb).to_pylist() == tbl.to_pylist()


# ------------------------------------------------ unaligned value buffers --
#
# Arrow does not promise aligned value buffers. `np.frombuffer(blob,
# np.float64, offset=4)` — what you get parsing a packed binary record or an
# mmap with a header — is zero-copied straight through `pa.array()`, and
# pyarrow computes over it perfectly. Dereferencing a misaligned `*const f64`
# is UB in Rust: a DEBUG build traps it as a non-unwinding abort that takes
# the whole interpreter down rather than raising, and release only happens
# not to fault today.
#
# So these run in a subprocess. Observed from inside the session, an abort
# would kill the test run instead of failing a test; and the whole point is
# that a per-request input batch must never be able to end a serving process.

_PROBE_PRELUDE = """
import numpy as np, pyarrow as pa
from confit import DuckDBInferFn

def misaligned(values, dtype):
    "A pyarrow array whose value buffer is deliberately not naturally aligned."
    dt = np.dtype(dtype)
    off = max(1, dt.itemsize // 2)
    blob = bytearray(dt.itemsize * len(values) + off)
    view = np.ndarray(len(values), dtype=dt, buffer=blob, offset=off)
    view[:] = values
    arr = pa.array(view)
    addr = arr.buffers()[1].address
    # a copy anywhere upstream would make the probe vacuous
    assert addr % dt.itemsize != 0, f"{dtype} buffer came back aligned"
    return arr
"""


def _probe(body: str) -> str:
    src = _PROBE_PRELUDE + textwrap.dedent(body)
    r = subprocess.run(  # noqa: S603 — the source is this file's own literal
        [sys.executable, "-c", src], capture_output=True, text=True, timeout=180
    )
    assert r.returncode == 0, (
        f"probe exited {r.returncode} — a misaligned buffer must never end the "
        f"process\nstdout: {r.stdout}\nstderr: {r.stderr[-3000:]}"
    )
    return r.stdout.strip()


def test_unaligned_input_buffers_on_the_request_path():
    """`infer_arrow`, per request — the one that a caller controls."""
    out = _probe(
        """
        schema = pa.schema([
            pa.field("a", pa.int64(), nullable=False),
            pa.field("f", pa.float64(), nullable=False),
        ])
        fn = DuckDBInferFn(
            "SELECT a + 1 AS b, f * 2 AS d FROM __THIS__",
            row_tables={"__THIS__": schema}, static_tables={},
        )
        tbl = pa.table({
            "a": misaligned([1, 2, 3], np.int64),
            "f": misaligned([1.5, 2.5, 3.5], np.float64),
        })
        got = fn.infer_arrow(tbl).to_pylist()
        assert got == [{"b": 2, "d": 3.0}, {"b": 3, "d": 5.0}, {"b": 4, "d": 7.0}], got
        print("ok")
        """
    )
    assert out == "ok"


def test_unaligned_string_offset_buffer():
    """String offsets are read the same way as values, and are the buffer a
    caller is least likely to have thought about."""
    out = _probe(
        """
        schema = pa.schema([pa.field("s", pa.string(), nullable=False)])
        fn = DuckDBInferFn(
            "SELECT upper(s) AS u FROM __THIS__",
            row_tables={"__THIS__": schema}, static_tables={},
        )
        vals = ["ab", "cde", "f"]
        offs = np.array([0, 2, 5, 6], dtype=np.int32)
        data = b"abcdef"
        blob = bytearray(offs.nbytes + 2)
        view = np.ndarray(len(offs), dtype=np.int32, buffer=blob, offset=2)
        view[:] = offs
        obuf = pa.py_buffer(memoryview(blob)[2:])
        assert obuf.address % 4 != 0, "offset buffer came back aligned"
        arr = pa.Array.from_buffers(pa.string(), len(vals),
                                    [None, obuf, pa.py_buffer(data)])
        assert arr.to_pylist() == vals, arr.to_pylist()
        got = fn.infer_arrow(pa.table({"s": arr})).to_pylist()
        assert got == [{"u": "AB"}, {"u": "CDE"}, {"u": "F"}], got
        print("ok")
        """
    )
    assert out == "ok"


def test_unaligned_model_tables_at_construction():
    """The other path: the tree-table decode runs once at build, not per
    request, but it reads the same way."""
    out = _probe(
        """
        nodes = pa.table({
            "model_id": misaligned([0, 0, 0], np.int64),
            "tree_id": misaligned([0, 0, 0], np.int64),
            "node_id": misaligned([0, 1, 2], np.int64),
            "feature": misaligned([0, -1, -1], np.int32),
            "threshold": misaligned([0.5, 0.0, 0.0], np.float64),
            "left": misaligned([1, -1, -1], np.int32),
            "right": misaligned([2, -1, -1], np.int32),
            "missing_left": pa.array([True, True, True], pa.bool_()),
            "value": misaligned([0.0, 10.0, 20.0], np.float64),
        })
        headers = pa.table({
            "model_id": misaligned([0], np.int64),
            "base": misaligned([0.0], np.float64),
            "agg": pa.array(["sum"], pa.string()),
            "link": pa.array(["identity"], pa.string()),
        })
        class M:
            name, takes, returns = "m", pa.schema([("x", pa.float64())]), pa.float64()
            instances = {0: None}
            def tree_tables(self):
                return nodes, headers, "float32"

        schema = pa.schema([
            pa.field("id", pa.int64(), nullable=False),
            pa.field("x", pa.float64(), nullable=False),
        ])
        fn = DuckDBInferFn(
            "SELECT m(id, x) AS p FROM __THIS__",
            row_tables={"__THIS__": schema}, static_tables={},
            udfs=[M()],
        )
        got = fn.infer_rows([{"id": 0, "x": 0.0}, {"id": 0, "x": 1.0}])
        assert [r["p"] for r in got] == [10.0, 20.0], got
        print("ok")
        """
    )
    assert out == "ok"


# ---------------------------------------------- struct row columns ingest --
#
# Both entry points serve a row schema with a struct column, because ingest
# walks the lane's SEGMENT PATH through the StructArray's children.
#
# The one hazard the design must not get wrong: a NOT NULL child under a
# nullable parent has no validity buffer of its own, and its data buffer
# under a null parent holds an arbitrary live value. DuckDB folds the
# parent's mask into every child at arrow ingest
# (src/function/table/arrow_conversion.cpp, ColumnArrowToDuckDB, the STRUCT
# case) and struct_extract itself does not, so the fold belongs at OUR
# ingest too: the null lane is the OR of every level.

_NN = pa.field("x", pa.int64(), nullable=False)
_S114 = pa.schema([pa.field("st", pa.struct([_NN]), nullable=True)])
_S114_NESTED = pa.schema(
    [
        pa.field(
            "a",
            pa.struct(
                [
                    pa.field(
                        "b",
                        pa.struct([pa.field("c", pa.int64(), nullable=False)]),
                        nullable=True,
                    )
                ]
            ),
            nullable=True,
        )
    ]
)
_S114_WIDE = pa.schema(
    [
        pa.field(
            "st",
            pa.struct(
                [
                    pa.field("s", pa.string()),
                    pa.field("b", pa.bool_()),
                    pa.field("i32", pa.int32()),
                ]
            ),
            nullable=True,
        )
    ]
)
_S114_ROWS = [{"st": {"x": 41}}, {"st": None}, {"st": {"x": 7}}]
_S114_SQL = "SELECT st.x + 1 AS o FROM __THIS__"


def _duck(sql: str, table: pa.Table) -> pa.Table:
    """The live oracle over the SAME arrow batch, optimizer off."""
    import duckdb

    con = duckdb.connect()
    con.execute("PRAGMA disable_optimizer")
    con.register("__THIS__", table)
    return con.execute(sql).to_arrow_table()


def _build(sql, schema):
    return DuckDBInferFn(sql, row_tables={"__THIS__": schema}, static_tables={})


def _vs_duckdb(fn, sql, table):
    """infer_arrow == optimizer-off DuckDB, values AND schema."""
    got = fn.infer_arrow(table)
    want = _duck(sql, table)
    assert got.to_pylist() == want.to_pylist()
    assert got.schema == want.schema
    return got


def test_differential_struct_row_column():
    """The entry-point-agreement criterion, at its narrowest."""
    tbl = _table(_S114_ROWS, schema=_S114)
    for sql in (_S114_SQL, "SELECT st.* FROM __THIS__"):
        fn = _build(sql, _S114)
        _cmp(fn, _S114_ROWS, tbl)
        _vs_duckdb(fn, sql, tbl)
    # A NOT NULL parent over a NOT NULL child: no validity buffer anywhere.
    nn = pa.schema([pa.field("st", pa.struct([_NN]), nullable=False)])
    rows = [{"st": {"x": 1}}, {"st": {"x": 2}}]
    tbl = _table(rows, schema=nn)
    fn = _build(_S114_SQL, nn)
    _cmp(fn, rows, tbl)
    _vs_duckdb(fn, _S114_SQL, tbl)


def test_a_null_parent_does_not_leak_the_child_buffer():
    """The validity-OR criterion, on a fixture whose child buffer holds a
    LOUD value under the null parent. `Table.from_pylist` happens to leave a
    zero there, which would let a fold-free ingest pass by luck."""
    loud = pa.StructArray.from_arrays(
        [pa.array([41, 999999, 7])],
        fields=[_NN],
        mask=pa.array([False, True, False]),
    )
    tbl = pa.Table.from_arrays([loud], schema=_S114)
    # The hazard is live: the child, undecorated, still carries the value.
    child = tbl.column("st").chunk(0).field(0)
    assert child.to_pylist() == [41, 999999, 7]
    assert child.null_count == 0
    fn = _build(_S114_SQL, _S114)
    got = _vs_duckdb(fn, _S114_SQL, tbl)
    assert [r["o"] for r in got.to_pylist()] == [42, None, 8]
    assert fn.infer_rows(tbl.to_pylist()) == got.to_pylist()


def test_a_null_at_any_nesting_level_nulls_the_leaf():
    """The nesting criterion: an outer null, a middle null and a live leaf,
    with the innermost buffer holding real values under both nulls."""
    rows = [{"a": {"b": {"c": 1}}}, {"a": {"b": None}}, {"a": None}]
    sql = "SELECT a.b.c * 10 AS o FROM __THIS__"
    fn = _build(sql, _S114_NESTED)
    tbl = _table(rows, schema=_S114_NESTED)
    _cmp(fn, rows, tbl)
    assert _vs_duckdb(fn, sql, tbl).to_pylist() == [{"o": 10}, {"o": None}, {"o": None}]

    inner = pa.array([1, 777, 888])
    mid = pa.StructArray.from_arrays(
        [inner],
        fields=[pa.field("c", pa.int64(), nullable=False)],
        mask=pa.array([False, True, False]),
    )
    outer = pa.StructArray.from_arrays(
        [mid],
        fields=[pa.field("b", mid.type, nullable=True)],
        mask=pa.array([False, False, True]),
    )
    loud = pa.Table.from_arrays([outer], schema=_S114_NESTED)
    assert loud.column("a").chunk(0).field(0).field(0).to_pylist() == [1, 777, 888]
    assert _vs_duckdb(fn, sql, loud).to_pylist() == [
        {"o": 10},
        {"o": None},
        {"o": None},
    ]


@pytest.mark.parametrize("depth", [1, 2, 4, 8, 16, 32])
def test_nested_structs_serve_to_the_row_paths_depth(depth):
    """There is no depth constant to import: schema.rs, build_fields and
    lane_paths all recurse unbounded and the ingest walk is a segment loop,
    so this pins that the arrow path has no limit the row path lacks."""
    field = pa.field("v", pa.int64(), nullable=False)
    for i in range(depth - 1, -1, -1):
        field = pa.field(f"f{i}", pa.struct([field]), nullable=True)
    schema = pa.schema([field])
    row = {"v": 5}
    for i in range(depth - 1, -1, -1):
        row = {f"f{i}": row}
    path = ".".join(f"f{i}" for i in range(depth)) + ".v"
    sql = f"SELECT {path} + 1 AS o FROM __THIS__"  # noqa: S608 -- our own names
    fn = _build(sql, schema)
    tbl = _table([row], schema=schema)
    _cmp(fn, [row], tbl)
    assert _vs_duckdb(fn, sql, tbl).to_pylist() == [{"o": 6}]


def test_a_sliced_struct_batch_reads_the_right_rows():
    """pyarrow COMPOSES a struct child's offset with its parent's, matching
    DuckDB's GetEffectiveOffset — so each level is read through its OWN
    offset and nothing is composed by hand."""
    fn = _build(_S114_SQL, _S114)

    # (a) a parent sliced over a zero-offset child.
    big = _table([{"st": {"x": i}} for i in range(8)], schema=_S114)
    sliced = big.slice(3, 4)
    assert sliced.column("st").chunk(0).offset == 3
    assert _vs_duckdb(fn, _S114_SQL, sliced).to_pylist() == [
        {"o": 4},
        {"o": 5},
        {"o": 6},
        {"o": 7},
    ]

    # (b) a child that carries its own offset 4 under a parent sliced at 2.
    child = pa.array(list(range(100, 110)))
    parent = pa.StructArray.from_arrays([child.slice(4)], fields=[_NN])
    own = pa.Table.from_arrays([parent], schema=_S114).slice(2, 3)
    assert own.column("st").chunk(0).field(0).offset == 6  # 4 + 2, composed
    assert _vs_duckdb(fn, _S114_SQL, own).to_pylist() == [
        {"o": 107},
        {"o": 108},
        {"o": 109},
    ]

    # (c) a sliced parent whose slice CONTAINS a null, with the null slot's
    # raw value a real neighbouring row's rather than a zero.
    vals = pa.array([100 + i for i in range(8)])
    nulled = pa.StructArray.from_arrays(
        [vals],
        fields=[_NN],
        mask=pa.array([False] * 4 + [True] + [False] * 3),
    )
    inside = pa.Table.from_arrays([nulled], schema=_S114).slice(3, 4)
    assert inside.column("st").chunk(0).field(0).to_pylist() == [103, 104, 105, 106]
    assert _vs_duckdb(fn, _S114_SQL, inside).to_pylist() == [
        {"o": 104},
        {"o": None},
        {"o": 106},
        {"o": 107},
    ]


def _wrong_st(kind):
    """A batch whose 'st' column does not match the declared struct."""
    if kind == "absent":
        return pa.table({"other": [1]})
    if kind == "not_a_struct":
        return pa.table({"st": pa.array([1], pa.int64())})
    if kind == "no_field":
        arr = pa.StructArray.from_arrays(
            [pa.array([1])], fields=[pa.field("y", pa.int64())]
        )
        return pa.table({"st": arr})
    if kind == "leaf_dtype":
        arr = pa.StructArray.from_arrays(
            [pa.array([1], pa.int32())], fields=[pa.field("x", pa.int32())]
        )
        return pa.table({"st": arr})
    raise AssertionError(kind)


@pytest.mark.parametrize(
    ("kind", "reason"),
    [
        ("absent", "missing column 'st'"),
        ("not_a_struct", "'st' is int64"),
        ("no_field", "has no field 'x'"),
        ("leaf_dtype", "column 'st.x' is int32"),
    ],
)
def test_struct_leaf_lanes_refuse_by_name(kind, reason):
    """A struct-shaped batch that does not match the declared schema names
    the specific reason, not the retired blanket all-scalar sentence."""
    fn = _build(_S114_SQL, _S114)
    with pytest.raises(ValueError) as e:
        fn.infer_arrow(_wrong_st(kind))
    assert reason in str(e.value)
    assert "all-scalar" not in str(e.value)


def test_a_non_nullable_struct_leaf_refuses_a_child_level_null():
    """The declared-schema contract every NOT NULL column shares, extended
    to a leaf lane and named with the leaf's dotted DISPLAY name."""
    nn = pa.schema([pa.field("st", pa.struct([_NN]), nullable=False)])
    arr = pa.StructArray.from_arrays([pa.array([1, None])], fields=[_NN])
    fn = _build(_S114_SQL, nn)
    with pytest.raises(ValueError, match="column 'st.x' is not nullable"):
        fn.infer_arrow(pa.Table.from_arrays([arr], schema=nn))


def test_a_struct_column_with_two_chunks_names_combine_chunks():
    """The existing refusal, now that the all-scalar one no longer fires
    first and hides it."""
    fn = _build(_S114_SQL, _S114)
    two = pa.concat_tables(
        [
            _table(_S114_ROWS[:1], schema=_S114),
            _table(_S114_ROWS[1:], schema=_S114),
        ]
    )
    assert two.column("st").num_chunks == 2
    with pytest.raises(ValueError, match="combine_chunks"):
        fn.infer_arrow(two)


def test_an_empty_struct_batch_round_trips():
    fn = _build(_S114_SQL, _S114)
    empty = _table([], schema=_S114)
    got = _vs_duckdb(fn, _S114_SQL, empty)
    assert got.to_pylist() == []
    assert got.schema == fn.output_schema


def test_a_wide_struct_of_mixed_leaf_types():
    """String, bool and a narrow int leaf, sliced so a NULL parent lands
    inside the slice. Values AND the output pa.Schema against DuckDB."""
    rows = [
        {"st": None} if i == 2 else {"st": {"s": c * 2, "b": i % 2 == 1, "i32": i}}
        for i, c in enumerate("abcde")
    ]
    sql = "SELECT st.s AS s, st.b AS b, st.i32 AS i FROM __THIS__"
    fn = _build(sql, _S114_WIDE)
    sliced = _table(rows, schema=_S114_WIDE).slice(1, 3)
    got = _vs_duckdb(fn, sql, sliced)
    assert got.schema.types == [pa.string(), pa.bool_(), pa.int32()]
    assert got.to_pylist() == fn.infer_rows(sliced.to_pylist())


# The opacity criterion. "Stays opaque and does not block the build" means
# the all-opaque struct (zero lanes, never looked at) and the servable one
# (lanes, required in the batch on exactly the row path's terms).
_OPAQUE_TS = pa.struct([pa.field("t", pa.timestamp("us"))])
_K = pa.field("k", pa.int64(), nullable=False)


def test_an_unreferenced_all_opaque_struct_column_is_never_looked_at():
    """Passes today; a regression pin. Zero lanes means the column need not
    even be in the batch."""
    schema = pa.schema([_K, pa.field("ts", _OPAQUE_TS, nullable=True)])
    sql = "SELECT k + 1 AS o FROM __THIS__"
    fn = _build(sql, schema)
    flat = pa.table({"k": [1, 2]}, schema=pa.schema([_K]))
    assert fn.infer_arrow(flat).to_pylist() == [{"o": 2}, {"o": 3}]
    assert fn.infer_rows([{"k": 1, "ts": None}]) == [{"o": 2}]


def test_an_unreferenced_struct_column_with_lanes_serves_and_is_required():
    """Its leaves ARE lanes, so the batch must carry it -- and the row path
    requires the same attribute for the same input."""
    schema = pa.schema(
        [_K, pa.field("st", pa.struct([pa.field("x", pa.int64())]), nullable=True)]
    )
    sql = "SELECT k + 1 AS o FROM __THIS__"
    fn = _build(sql, schema)
    present = pa.Table.from_pylist(
        [{"k": 1, "st": {"x": 9}}, {"k": 2, "st": None}], schema=schema
    )
    assert _vs_duckdb(fn, sql, present).to_pylist() == [{"o": 2}, {"o": 3}]
    # Absent: both entry points refuse, each in its own vocabulary.
    with pytest.raises(ValueError, match="missing column 'st'"):
        fn.infer_arrow(pa.table({"k": [1]}, schema=pa.schema([_K])))
    with pytest.raises(ValueError, match="missing attribute 'st.x'"):
        fn.infer_rows([{"k": 1}])


def test_a_mixed_struct_never_touches_its_opaque_sibling():
    """One servable field and one out-of-vocabulary field: exactly ONE lane,
    and the walk only ever visits that lane's path."""
    mix = pa.struct([pa.field("x", pa.int64()), pa.field("t", pa.timestamp("us"))])
    schema = pa.schema([_K, pa.field("mix", mix, nullable=True)])
    sql = "SELECT mix.x + k AS o FROM __THIS__"
    fn = _build(sql, schema)
    rows = [{"k": 1, "mix": {"x": 10, "t": None}}, {"k": 2, "mix": None}]
    tbl = pa.Table.from_pylist(rows, schema=schema)
    assert _vs_duckdb(fn, sql, tbl).to_pylist() == [{"o": 11}, {"o": None}]
    # The opaque sibling is not type checked: a child type we do not serve
    # goes straight through, because the walk never asks about it.
    hostile = pa.struct(
        [pa.field("x", pa.int64()), pa.field("t", pa.list_(pa.int64()))]
    )
    arr = pa.StructArray.from_arrays(
        [pa.array([10, None]), pa.array([[1, 2], None], pa.list_(pa.int64()))],
        fields=list(hostile),
        mask=pa.array([False, True]),
    )
    batch = pa.table({"k": pa.array([1, 2]), "mix": arr})
    assert fn.infer_arrow(batch).to_pylist() == [{"o": 11}, {"o": None}]


def test_struct_input_does_not_change_the_output_schema():
    """The no-output-change criterion: the flattened output is unchanged,
    and still byte-identical to DuckDB's own."""
    cases = [
        (_S114_SQL, _S114, _table(_S114_ROWS, schema=_S114)),
        (
            "SELECT a.b.c * 10 AS o FROM __THIS__",
            _S114_NESTED,
            _table([{"a": {"b": {"c": 1}}}], schema=_S114_NESTED),
        ),
        ("SELECT st.* FROM __THIS__", _S114, _table(_S114_ROWS, schema=_S114)),
    ]
    for sql, schema, tbl in cases:
        fn = _build(sql, schema)
        want = _duck(sql, tbl).schema
        assert fn.output_schema == want
        for ty in fn.output_schema.types:
            assert not pa.types.is_struct(ty)
