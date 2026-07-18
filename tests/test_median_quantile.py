"""TASK-26: median / quantile OVER as frozen fit-state.

median and quantile OVER (PARTITION BY ...) must freeze at fit (computed once by
DataFusion) and be looked up per row at inference -- the same mechanism the other
window aggregates use. The acceptance bar is transform == infer parity, with
DataFusion as the oracle (decision-1); we assert it with the differential
harness's row comparison.
"""

import pyarrow as pa
from differential import _rows_equal

from sql_transform import SQLTransform


def _parity(sql: str, train: pa.Table, batch: pa.Table | None = None) -> list[dict]:
    """Fit `sql`, then assert transform(batch) == infer_batch(batch)."""
    batch = train if batch is None else batch
    t = SQLTransform(sql).fit(train)
    transform_out = t.transform(batch).to_pylist()
    infer_out = [r.model_dump() for r in t.infer_batch(batch.to_pylist())]
    assert _rows_equal(transform_out, infer_out), (transform_out, infer_out)
    return transform_out


# --- median -----------------------------------------------------------------

# A skewed column: median (2.0 for [1,2,9]) differs from mean (4.0), which is the
# whole point of the ticket (LotFrontage imputation).
_SKEWED = pa.table({"g": ["a", "a", "a", "b", "b"], "x": [1.0, 2.0, 9.0, 4.0, 6.0]})


def test_median_over_partition_parity():
    out = _parity(
        "SELECT x - MEDIAN(x) OVER (PARTITION BY g) AS c FROM __THIS__", _SKEWED
    )
    # g='a' median=2.0 -> [-1, 0, 7]; g='b' median=5.0 -> [-1, 1]
    assert [r["c"] for r in out] == [-1.0, 0.0, 7.0, -1.0, 1.0]


def test_median_over_global_parity():
    # whole-column median of [1,2,9,4,6] sorted [1,2,4,6,9] = 4.0
    out = _parity("SELECT x - MEDIAN(x) OVER () AS c FROM __THIS__", _SKEWED)
    assert [r["c"] for r in out] == [-3.0, -2.0, 5.0, 0.0, 2.0]


def test_median_frozen_on_unseen_data():
    # Median is frozen at fit; applying to new rows uses the fit-time value, and
    # an unseen partition key yields NULL (like the other window aggs).
    train = pa.table({"g": ["a", "a", "a"], "x": [1.0, 2.0, 9.0]})
    test = pa.table({"g": ["a", "z"], "x": [10.0, 10.0]})
    out = _parity(
        "SELECT MEDIAN(x) OVER (PARTITION BY g) AS m FROM __THIS__", train, test
    )
    assert out[0]["m"] == 2.0  # frozen 'a' median
    assert out[1]["m"] is None  # unseen 'z'


# --- quantile (percentile_cont, exact) --------------------------------------


def test_quantile_over_partition_parity():
    out = _parity(
        "SELECT percentile_cont(x, 0.25) OVER (PARTITION BY g) AS q FROM __THIS__",
        _SKEWED,
    )
    # g='a' [1,2,9] p25 (continuous interpolation) = 1.5; g='b' [4,6] p25 = 4.5
    assert [r["q"] for r in out] == [1.5, 1.5, 1.5, 4.5, 4.5]


def test_quantile_over_global_parity():
    _parity("SELECT percentile_cont(x, 0.9) OVER () AS q FROM __THIS__", _SKEWED)


def test_two_quantiles_distinct_state_keys():
    # Two quantiles of the same column must NOT collide on the same state key.
    _parity(
        "SELECT percentile_cont(x, 0.25) OVER () AS q1, "
        "percentile_cont(x, 0.75) OVER () AS q3 FROM __THIS__",
        _SKEWED,
    )
