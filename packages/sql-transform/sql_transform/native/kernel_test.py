"""The additive chi2 map (kernel.py): its first field bit-exact, a product
and a square root, each correctly rounded on both sides; its cos and sin
fields within their error scale, as numpy's log parts from glibc's."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pyarrow as pa
import pytest
from sklearn.kernel_approximation import AdditiveChi2Sampler
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler

from sql_transform._udf import PythonTransform
from sql_transform.native import (
    NotNative,
    ParityError,
    catalog,
    check,
    explain_native,
    to_native,
)
from sql_transform.native import kernel as K

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


def _steps(steps: int, n: int, interval: Any = None) -> AdditiveChi2Sampler:
    est = AdditiveChi2Sampler(sample_steps=steps, sample_interval=interval)
    return est.fit(np.ones((2, n)))


def _pairs(est: AdditiveChi2Sampler) -> PythonTransform:
    """A step of `est` over two features, returning a list."""
    width = 2 * (2 * est.sample_steps - 1)
    return _step(est, [pa.float64()] * 2, pa.list_(pa.float64(), width))


def _apart_logs(n: int) -> list[float]:
    """Up to `n` doubles near 1 whose numpy log is not glibc's (`math.log`,
    DuckDB's `ln`): about 2% of such draws on x86-64 with AVX-512, none
    where numpy's log is glibc's."""
    x = 1 + np.random.default_rng(0).normal(0, 1e-3, 20_000)
    return [v for v in x.tolist() if float(np.log(v)) != math.log(v)][:n]


@pytest.mark.parametrize(
    "steps, interval", [(2, None), (3, None), (4, 0.2501), (2, 0.0), (2, 1e306)]
)
def test_its_cos_and_sin_fields_serve_special_values(steps, interval):
    # At 1e306, `s * ln x` overflows, and numpy's cos and sin of infinity
    # are NaN, where DuckDB's raise.
    step = _pairs(_steps(steps, 2, interval))
    k = len(SPECIALS)
    rows = _rows(
        [SPECIALS[i % k] for i in range(k * k)],
        [SPECIALS[i // k] for i in range(k * k)],
    )
    native = to_native(step, strict=True, allow_bound=True)
    assert check(step, native, rows) == len(VALID) ** 2


@pytest.mark.parametrize("steps, interval", [(3, None), (18, 0.1)])
def test_a_field_whose_log_parts_is_held_to_its_scale(steps, interval):
    xs = _apart_logs(100)
    if not xs:
        pytest.skip("numpy's log is glibc's here")
    est = _steps(steps, 1, interval)
    step = _step(est, [pa.float64()], pa.list_(pa.float64(), 2 * steps - 1))
    native = to_native(step, strict=True, allow_bound=True)
    assert check(step, native, _rows(xs)) == len(xs)
    # The fields do part: bit-exact, the same rows fail.
    with pytest.raises(ParityError, match="output field"):
        check(step, native, _rows(xs), ulps=0)


def test_its_cos_and_sin_fields_serve_only_on_request():
    est = _steps(2, 2)
    if K._exact(est):
        pytest.skip("the kernel probes read 0 here: the entry is bit-exact")
    step = _pairs(est)
    assert to_native(step) is step
    assert "within 4·eps·S + τ (S its error scale)" in explain_native(step)
    with pytest.raises(NotNative, match="only with allow_bound=True"):
        to_native(step, strict=True)


def test_where_the_kernel_probes_read_0_the_entry_is_bit_exact(monkeypatch):
    # And where numpy's cosh is glibc's for the step's constants.
    est, entry = _steps(3, 1), catalog()[AdditiveChi2Sampler]
    same = all(
        float(np.cosh(np.pi * j * 0.4)) == math.cosh(math.pi * j * 0.4) for j in (1, 2)
    )
    monkeypatch.setattr(K, "kernel_distance", lambda func: 0)
    assert entry.exact(est) == same
    # At 1e306 the cosh overflows on both sides, where `math.cosh` raises.
    assert entry.exact(_steps(2, 1, 1e306))
    monkeypatch.setattr(K, "kernel_distance", lambda func: 1)
    assert not entry.exact(est)
    assert entry.exact(_steps(1, 1))


def test_a_log_further_from_glibcs_keeps_the_cos_and_sin_fields_python(monkeypatch):
    # K = 4 reads a log within 1 ulp of glibc's.
    monkeypatch.setattr(K, "kernel_distance", lambda func: 2 if func is np.log else 0)
    with pytest.raises(NotNative, match="log, sin and cos are 2, 0 and 0 ulps"):
        to_native(_pairs(_steps(2, 2)), strict=True, allow_bound=True)


def test_a_float32_interval_keeps_the_cos_and_sin_fields_python():
    # The twin's cosh is then a float32's, which hosts round apart.
    step = _pairs(_steps(2, 2, np.float32(0.3)))
    with pytest.raises(NotNative, match="computes its cosh in float32"):
        to_native(step, strict=True, allow_bound=True)


def test_more_steps_without_an_interval_stay_python():
    # The fit rejects them; set after it, the twin raises.
    est = _steps(2, 2)
    est.sample_steps = 4
    with pytest.raises(NotNative, match=r"sample_steps=4\) without a sample_interval"):
        to_native(_pairs(est), strict=True, allow_bound=True)


def test_an_interval_the_fit_rejects_stays_python():
    # The fit takes no infinite interval; one set after it stays Python.
    est = _chi2(1)
    est.sample_interval = math.inf
    with pytest.raises(NotNative, match="sample_interval=inf"):
        to_native(_step(est, [pa.float64()], pa.float64()), strict=True)


@pytest.mark.skipif(
    np.finfo(np.longdouble).nmant <= 52, reason="a long double is a double here"
)
def test_a_long_double_interval_stays_python():
    # The twin computes the root in long doubles, then rounds it.
    est = _chi2(1, sample_interval=np.longdouble(0.3))
    with pytest.raises(NotNative, match="the twin computes in float"):
        to_native(_step(est, [pa.float64()], pa.float64()), strict=True)


@pytest.mark.parametrize("interval", [np.float32(0.3), np.float16(0.3), np.int64(2)])
def test_an_interval_of_another_numpy_type_is_read_as_a_double(interval):
    est = _chi2(2, sample_interval=interval)
    step = _step(est, [pa.float64()] * 2, pa.list_(pa.float64(), 2))
    rows = _rows([0.3, 2.0, 7.25, 1e300], [5e-324, 0.0, 1.0, 1e-300])
    assert check(step, to_native(step, strict=True), rows) == 4


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


def test_booleans_only_keep_the_cos_and_sin_fields_python():
    # The root of 1.0 is 1.0, but a cos field of True is not.
    est = _steps(2, 2, 1.0)
    step = _step(est, [pa.bool_()] * 2, pa.list_(pa.float64(), 6))
    with pytest.raises(NotNative, match="over boolean features only: the twin"):
        to_native(step, strict=True, allow_bound=True)


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
