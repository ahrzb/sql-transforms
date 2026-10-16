"""The additive chi2 map's first field, bit-exact (kernel.py): a product
and a square root, each correctly rounded on both sides. Its cos and sin
fields stay Python."""

from __future__ import annotations

import math

import numpy as np
import pyarrow as pa
import pytest
from sklearn.kernel_approximation import AdditiveChi2Sampler
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler

from sql_transform._udf import PythonTransform
from sql_transform.native import NotNative, check, to_native

# The twin's products overflow and meet 0 * inf, as numpy warns.
pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")

# Doubles where the map can part from the twin: signed zeros, subnormals (a
# product that underflows to 0), a product that overflows, and the values
# the twin rejects: negatives (a negative subnormal too), the infinities,
# NaN and NULL.
SPECIALS = [
    0.0, -0.0, 5e-324, -5e-324, 1e-310, -1e-310, 1e-300, 0.5, 1.0, -1.0, 2.5,
    7.25, 1e300, -1e300, 1e308, 1.7976931348623157e308, math.inf, -math.inf,
    math.nan, None,
]  # fmt: skip
# The ones it answers.
VALID = [v for v in SPECIALS if v is not None and v == v and 0 <= v < math.inf]


def _step(est: AdditiveChi2Sampler, types: list[pa.DataType], returns: pa.DataType):
    takes = pa.schema([(f"x{i}", t) for i, t in enumerate(types)])
    return PythonTransform("tf", {0: est}, takes, returns)


def _rows(*cols: list) -> pa.Table:
    return pa.table(
        {
            "__iid": pa.array([0] * len(cols[0]), pa.int64()),
            **{f"x{i}": pa.array(c) for i, c in enumerate(cols)},
        }
    )


def _chi2(n: int, **kw) -> AdditiveChi2Sampler:
    return AdditiveChi2Sampler(sample_steps=1, **kw).fit(np.ones((2, n)))


@pytest.mark.parametrize("interval", [None, 0.0, 0.4, 1.0, 2.7, 1e300, 1e-300])
@pytest.mark.parametrize(
    "returns",
    [pa.struct([("a", pa.float64()), ("b", pa.float64())]), pa.list_(pa.float64(), 2)],
    ids=["struct", "list"],
)
def test_the_first_field_is_bit_exact_on_special_values(interval, returns):
    # Each pair of SPECIALS: the twin answers the pairs of VALID, and the
    # entry traps on the others, on either feature.
    est = _chi2(2, sample_interval=interval)
    step = _step(est, [pa.float64()] * 2, returns)
    k = len(SPECIALS)
    rows = _rows(
        [SPECIALS[i % k] for i in range(k * k)],
        [SPECIALS[i // k] for i in range(k * k)],
    )
    assert check(step, to_native(step, strict=True), rows) == len(VALID) ** 2


def test_one_feature_serves_a_scalar():
    step = _step(_chi2(1), [pa.float64()], pa.float64())
    assert check(step, to_native(step, strict=True), _rows(SPECIALS)) == len(VALID)


@pytest.mark.parametrize("steps, interval", [(2, None), (3, None), (5, 0.3)])
def test_its_cos_and_sin_fields_stay_python(steps, interval):
    est = AdditiveChi2Sampler(sample_steps=steps, sample_interval=interval)
    est.fit(np.ones((2, 1)))
    step = _step(est, [pa.float64()], pa.list_(pa.float64(), 2 * steps - 1))
    with pytest.raises(NotNative, match=rf"sample_steps={steps}\): its cos and sin"):
        to_native(step, strict=True)


def test_an_interval_the_fit_rejects_stays_python():
    # The fit takes no infinite interval; one set after it stays Python.
    est = _chi2(1)
    est.sample_interval = math.inf
    with pytest.raises(NotNative, match="sample_interval=inf"):
        to_native(_step(est, [pa.float64()], pa.float64()), strict=True)


BITS = [True, False, None]


def _booleans(interval: float | None) -> PythonTransform:
    est = _chi2(2, sample_interval=interval)
    return _step(est, [pa.bool_()] * 2, pa.list_(pa.float64(), 2))


@pytest.mark.parametrize("interval", [None, 2.7])
def test_booleans_only_stay_python_where_the_twin_answers_true(interval):
    # A row of booleans none NULL reaches the twin as a boolean array, which
    # it keeps boolean: True answers 1.0, not the root of the interval.
    with pytest.raises(NotNative, match="over boolean features only: the twin"):
        to_native(_booleans(interval), strict=True)


@pytest.mark.parametrize("interval", [0.0, 1.0])
def test_booleans_only_serve_where_true_answers_its_root(interval):
    step = _booleans(interval)
    rows = _rows([a for a in BITS for _ in BITS], [b for _ in BITS for b in BITS])
    # A NULL makes the row float64 with NaN, which the twin rejects.
    assert check(step, to_native(step, strict=True), rows) == 4


def test_a_boolean_beside_a_number_answers_its_root():
    # Beside a number the row is float64: True answers sqrt(0.8).
    step = _step(_chi2(2), [pa.bool_(), pa.float64()], pa.list_(pa.float64(), 2))
    rows = _rows([True, False, True, None], [2.5, -0.0, 0.0, 1.0])
    assert check(step, to_native(step, strict=True), rows) == 3


def test_a_negative_output_of_a_scaler_traps():
    # Past the fit's range a MinMaxScaler answers below 0, where the twin's
    # second step raises: the composition's guard traps it.
    est = make_pipeline(MinMaxScaler(), AdditiveChi2Sampler(sample_steps=1))
    est.fit(np.array([[0.0, 1.0], [10.0, 3.0]]))
    step = _step(est, [pa.float64()] * 2, pa.list_(pa.float64(), 2))
    rows = _rows([5.0, -1.0, 5.0, 11.0, 0.0], [2.0, 2.0, 0.5, 4.0, 1.0])
    assert check(step, to_native(step, strict=True), rows) == 3
