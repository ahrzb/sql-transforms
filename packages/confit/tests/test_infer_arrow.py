"""The columnar boundary (TASK-60): infer_arrow == infer_rows, always.

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
    # Struct schemas use the row path.
    m_schema = pa.schema([pa.field("a", pa.struct([pa.field("i", pa.int64())]))])
    fs = DuckDBInferFn(
        "SELECT a.i FROM __THIS__", row_tables={"__THIS__": m_schema}, static_tables={}
    )
    with pytest.raises(ValueError, match="all-scalar"):
        fs.infer_arrow(pa.table({"a": [{"i": 1}]}))


# test_every_scenario_refuses_a_supplied_output_model (TASK-71) is gone with
# output_model=: the kwarg itself no longer exists, so there is nothing left
# to supply and no separate refusal path to differential-test. Its premise
# ("a supplied model flips infer_arrow's behaviour") is now covered once, at
# the constructor, by test_arrow_schema_api.py::test_infer_and_output_model_are_gone.


# TASK-79 landed with m-8 phase 2: integer widths are typed for real, so the
# schema suite allows NO differences and the former width pin is a plain
# parity test. The catalogue lives in test_integer_widths.py.
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
    """TASK-72: values always agreed; the SCHEMA did not, and a schema that
    does not match DuckDB's cannot be stacked with DuckDB's output."""
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


# --------------------------------------- TASK-67: unaligned value buffers --
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
