"""Native's retained consumer contracts for confit sharing and serving."""

import statistics
import time

import numpy as np
import pyarrow as pa
import pytest
from confit import DuckDBInferFn
from sklearn.preprocessing import QuantileTransformer

from sql_transform._udf import PythonTransform
from sql_transform.native import to_native
from sql_transform.native._registry import query


def _step(estimators, width):
    return PythonTransform(
        "cost_tf",
        dict(enumerate(estimators)),
        pa.schema([(f"x{i}", pa.float64()) for i in range(width)]),
        pa.struct([(f"f{i}", pa.float64()) for i in range(width)]),
    )


@pytest.mark.cost
def test_three_quantile_instances_serve_well_below_their_twin_cost():
    # #363 hoisted all instance arms: 25x faster than the twin became 1.4x.
    # Fixed shape, never the widest shape that happens to survive admission.
    rng = np.random.default_rng(0)
    fit = rng.normal(size=(128, 4))
    step = _step(
        [
            QuantileTransformer(n_quantiles=128, random_state=i).fit(fit + i)
            for i in range(3)
        ],
        4,
    )
    native = to_native(step, strict=True)
    values = rng.normal(size=(64, 4))
    rows = pa.table(
        {
            "__iid": pa.array(np.arange(64) % 3, pa.int64()),
            **{f"x{i}": values[:, i] for i in range(4)},
        }
    )
    sql = query(step)
    engines = [
        DuckDBInferFn(
            sql, row_tables={"__THIS__": rows.schema}, static_tables={}, udfs=[fn]
        )
        for fn in (native, step)
    ]
    for engine in engines:
        engine.infer_arrow(rows)
    timings = [[], []]
    for _ in range(9):
        for engine, samples in zip(engines, timings, strict=True):
            start = time.perf_counter()
            engine.infer_arrow(rows)
            samples.append(time.perf_counter() - start)
    took, twin = map(statistics.median, timings)
    print(
        f"quantile serve: native={took:.6f}s twin={twin:.6f}s ratio={took / twin:.3f}"
    )
    assert took < twin / 5, (took, twin)
