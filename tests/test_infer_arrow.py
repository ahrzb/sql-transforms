"""The columnar boundary (TASK-60): infer_arrow == infer_rows, always.

The differential is the contract: same engine, same program, two
boundaries — parity is by construction, this suite makes drift
impossible.
"""

from __future__ import annotations

import pyarrow as pa
import pytest
from pydantic import create_model

from benchmarks import serving_scenarios as sc
from sql_transform._interpreter import DuckDBInferFn

T = create_model("T", a=(int, ...), s=(str | None, None), f=(float | None, None))
ROWS = [
    {"a": 1, "s": "héllo", "f": 1.5},
    {"a": 2, "s": None, "f": None},
    {"a": 3, "s": "", "f": -0.0},
]


def _table(rows, schema=None):
    return pa.Table.from_pylist(rows, schema=schema)


def _cmp(fn, rows, tbl):
    got_rows = fn.infer({"__THIS__": rows})
    got_arrow = fn.infer_arrow(tbl).to_pylist()
    assert [dict(r) for r in got_rows] == got_arrow


def test_differential_basic_and_nulls():
    fn = DuckDBInferFn(
        "SELECT a + 1 AS b, upper(s) AS u, f * 2 AS d FROM __THIS__ WHERE a < 3",
        row_tables={"__THIS__": T},
        static_tables={},
        output="dict",
    )
    _cmp(fn, [T(**r) for r in ROWS], _table(ROWS))


def test_differential_every_serving_scenario():
    for mod in sc.all_scenarios():
        statics = mod.make_statics(sc.SEED)
        fn = sc.build_spec_fn(mod, statics, output="dict")
        rows_d = mod.make_rows(sc.SEED + 7, 64)
        model = sc.row_model(mod.ROW_SCHEMA)
        tbl = sc.rows_table(mod, rows_d)
        _cmp(fn, [model(**r) for r in rows_d], tbl)


def test_differential_shape_many_left_join():
    dup = pa.table({"id": [1, 1, 2], "v": [10, 11, 20]})
    fn = DuckDBInferFn(
        "SELECT a, v FROM __THIS__ LEFT JOIN d ON a = d.id",
        row_tables={"__THIS__": T},
        static_tables={"d": dup},
        output="dict",
        shape="many",
    )
    _cmp(fn, [T(**r) for r in ROWS], _table(ROWS))


def test_named_rejections():
    fn = DuckDBInferFn(
        "SELECT a FROM __THIS__",
        row_tables={"__THIS__": T},
        static_tables={},
        output="dict",
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
    # Struct models use the row path.
    Inner = create_model("Inner", i=(int | None, None))
    M = create_model("M", a=(Inner | None, None))
    fs = DuckDBInferFn(
        "SELECT a.i FROM __THIS__", row_tables={"__THIS__": M}, static_tables={}
    )
    with pytest.raises(ValueError, match="all-scalar"):
        fs.infer_arrow(pa.table({"a": [{"i": 1}]}))


def test_sliced_and_recordbatch_inputs():
    fn = DuckDBInferFn(
        "SELECT a, s, f FROM __THIS__",
        row_tables={"__THIS__": T},
        static_tables={},
        output="dict",
    )
    tbl = _table(ROWS * 3)
    sliced = tbl.slice(2, 4).combine_chunks()
    assert fn.infer_arrow(sliced).to_pylist() == sliced.to_pylist()
    rb = tbl.to_batches()[0]
    assert fn.infer_arrow(rb).to_pylist() == tbl.to_pylist()


def test_columnar_core_differential(monkeypatch):
    # The batch core is OPT-IN (measured: v1 computes at row-core parity
    # with a per-call allocation cost — see benchmarks/scaling_results.json
    # and the TASK-61 report). Under the flag, every scenario must still be
    # bit-identical to the row path.
    monkeypatch.setenv("SPECIALIZER_COLUMNAR", "1")
    for mod in sc.all_scenarios():
        statics = mod.make_statics(sc.SEED)
        fn = sc.build_spec_fn(mod, statics, output="dict")
        assert fn.arrow_backend == "columnar"
        rows_d = mod.make_rows(sc.SEED + 11, 96)
        model = sc.row_model(mod.ROW_SCHEMA)
        _cmp(fn, [model(**r) for r in rows_d], sc.rows_table(mod, rows_d))
    monkeypatch.delenv("SPECIALIZER_COLUMNAR")
    fn = sc.build_spec_fn(sc.load("titanic"), sc.load("titanic").make_statics(sc.SEED))
    assert fn.arrow_backend in ("cranelift", "interpreter")
