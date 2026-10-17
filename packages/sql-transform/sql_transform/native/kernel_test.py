"""The kernel approximations (kernel.py). The additive chi2 map: its first
field bit-exact, a product and a square root, each correctly rounded on
both sides; its cos and sin fields within their error scale, as numpy's log
parts from glibc's. The RBF sampler within its error scale, as the twin's
BLAS sum and the entry's balanced tree part by rounding; the skewed chi2
sampler, the same over ln(x + s), within its own."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pyarrow as pa
import pytest
from sklearn.kernel_approximation import (
    AdditiveChi2Sampler,
    RBFSampler,
    SkewedChi2Sampler,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

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


def test_the_build_estimate_takes_what_builds_within_the_cap():
    # Measured (`kernel._additive_chi2_build_estimate`): with one sample
    # step, a struct of 768 features built in 4.5 s and of 1,024 in 8.3 to
    # 9.6 s; with three, of 128 features in 5.1 s and of 192 in 12.6 s;
    # with eight, of 128 features in 28.6 s; with 400, of 3 features in
    # 3.4 s. confit refused the struct of one step over 1,536 features,
    # past its token limit.
    for n, steps in [(768, 1), (128, 3), (3, 400)]:
        assert K._additive_chi2_build_estimate(n, steps) <= K.MAX_BUILD_S
    for n, steps in [(1024, 1), (1536, 1), (192, 3), (128, 8)]:
        assert K._additive_chi2_build_estimate(n, steps) > K.MAX_BUILD_S


def test_a_step_past_its_estimated_build_stays_python():
    step = _step(_steps(3, 256), [pa.float64()] * 256, pa.list_(pa.float64(), 1280))
    with pytest.raises(
        NotNative,
        match=r"an estimated \d+ s build, past 7 s \(3 sample steps over 256 features",
    ):
        to_native(step, strict=True, allow_bound=True)


def _rbf(n: int, m: int = 20, dtype: Any = np.float64, **kw) -> RBFSampler:
    est = RBFSampler(n_components=m, random_state=0, **kw)
    return est.fit(np.random.default_rng(n).normal(size=(8, n)).astype(dtype))


def _rbf_step(est: Any, types: list[pa.DataType] | None = None) -> PythonTransform:
    n, m = np.shape(est.random_weights_)
    return _step(est, types or [pa.float64()] * n, pa.list_(pa.float64(), m))


def _draws(seed: int, n: int) -> np.ndarray:
    """Rows of features of scales 1e-2 to 1e3, and small whole numbers."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(60, n)) * 10.0 ** rng.uniform(-2, 3, size=(60, 1))
    return np.vstack([x, rng.integers(-3, 4, size=(20, n)).astype(float)])


def _table(x: np.ndarray) -> pa.Table:
    return _rows(*(x[:, i].tolist() for i in range(x.shape[1])))


@pytest.mark.parametrize(
    "n, m, gamma", [(3, 20, 1.0), (12, 100, "scale"), (64, 30, 0.01)]
)
def test_an_rbf_sampler_parts_from_its_twin_within_its_error_scale(n, m, gamma):
    est = _rbf(n, m, gamma=gamma)
    step = _rbf_step(est)
    native = to_native(step, strict=True, allow_bound=True)
    rows = _table(_draws(n, n))
    with pytest.raises(ParityError, match="ulps, bound 0"):
        check(step, native, rows, ulps=0)
    assert check(step, native, rows) == rows.num_rows


def test_an_rbf_sampler_of_one_feature_answers_the_twins_bits():
    # The angle is one product and the offset on both sides.
    step = _rbf_step(_rbf(1))
    rows = _table(_draws(1, 1))
    assert check(step, to_native(step, strict=True, allow_bound=True), rows, ulps=0)


# Doubles where the sampler can part from its twin, and the values its
# validation rejects; none so large that a product overflows, which the
# bound does not cover (one sum can overflow where another does not).
RBF_SPECIALS = [
    0.0, -0.0, 5e-324, -5e-324, 1e-310, 1e-300, 0.5, -1.0, 2.5, 7.25, 1e300,
    -1e300, math.inf, -math.inf, math.nan, None,
]  # fmt: skip


def test_an_rbf_sampler_serves_special_values():
    step = _rbf_step(_rbf(2))
    k = len(RBF_SPECIALS)
    rows = _rows(
        [RBF_SPECIALS[i % k] for i in range(k * k)],
        [RBF_SPECIALS[i // k] for i in range(k * k)],
    )
    valid = [v for v in RBF_SPECIALS if v is not None and math.isfinite(v)]
    native = to_native(step, strict=True, allow_bound=True)
    assert check(step, native, rows) == len(valid) ** 2


def test_an_infinite_angle_answers_nan_as_numpys_cos():
    # A product past the largest double makes the angle infinite, where
    # numpy's cos answers NaN and DuckDB's raises.
    est = _rbf(2)
    big = np.finfo(np.float64).max
    rows = np.array([[big, 0.0], [-big, 0.0], [0.0, big]])
    assert np.isnan(est.transform(rows)).any()
    step = _rbf_step(est)
    assert check(step, to_native(step, strict=True, allow_bound=True), _table(rows))


def test_an_rbf_sampler_serves_only_on_request():
    step = _rbf_step(_rbf(2))
    assert to_native(step) is step
    assert "within 5·eps·S + τ (S its error scale)" in explain_native(step)
    with pytest.raises(NotNative, match="only with allow_bound=True"):
        to_native(step, strict=True)


def test_a_changed_weight_breaches_the_error_scale():
    # A weight moved by 1e-9 of itself moves an angle by about 1e-9 of its
    # terms, far past (n + 3)·eps·S.
    est = _rbf(4)
    step = _rbf_step(est)
    native = to_native(step, strict=True, allow_bound=True)
    est.random_weights_ = est.random_weights_.copy()
    est.random_weights_[0, 0] *= 1 + 1e-9
    with pytest.raises(ParityError, match=r"past its bound K\*eps\*S \+ tau"):
        check(step, native, _table(_draws(4, 4)))


def test_an_rbf_sampler_in_a_pipeline_serves_as_its_last_step():
    est = make_pipeline(StandardScaler(), RBFSampler(n_components=8, random_state=0))
    est.fit(np.random.default_rng(5).normal(size=(30, 3)) * 100)
    step = _step(est, [pa.float64()] * 3, pa.list_(pa.float64(), 8))
    rows = _table(_draws(6, 3) * 100)
    assert check(step, to_native(step, strict=True, allow_bound=True), rows)


def test_an_rbf_sampler_over_booleans_only_serves():
    step = _rbf_step(_rbf(2), [pa.bool_()] * 2)
    rows = _rows([a for a in BITS for _ in BITS], [b for _ in BITS for b in BITS])
    # A NULL makes the row float64 with NaN, which the twin rejects.
    assert check(step, to_native(step, strict=True, allow_bound=True), rows) == 4


def test_an_rbf_sampler_fitted_in_float32_serves_over_numbers():
    # numpy widens the weights and offsets to float64 for a row of doubles.
    step = _rbf_step(_rbf(3, dtype=np.float32))
    rows = _table(_draws(7, 3))
    assert check(step, to_native(step, strict=True, allow_bound=True), rows)


def test_an_rbf_sampler_fitted_in_float32_over_booleans_only_stays_python():
    # A boolean array times float32 weights is a float32 product.
    step = _rbf_step(_rbf(2, dtype=np.float32), [pa.bool_()] * 2)
    with pytest.raises(NotNative, match="over boolean features only, fitted in"):
        to_native(step, strict=True, allow_bound=True)


def test_a_cos_apart_from_confits_keeps_an_rbf_sampler_python(monkeypatch):
    monkeypatch.setattr(K, "kernel_distance", lambda func: 1)
    with pytest.raises(NotNative, match="numpy cos is 1 ulps from confit's"):
        to_native(_rbf_step(_rbf(2)), strict=True, allow_bound=True)


def test_a_weight_not_finite_keeps_an_rbf_sampler_python():
    est = _rbf(2)
    est.random_weights_[1, 3] = math.inf
    with pytest.raises(NotNative, match="a random weight or offset not finite"):
        to_native(_rbf_step(est), strict=True, allow_bound=True)


def test_an_rbf_sampler_takes_a_fit_by_its_estimated_build():
    # It reads `x - 0.0` where the build so read stays within the cap, else
    # `coalesce(x, NaN)`; past both, the cap refuses it.
    assert K._rbf_reads_apart(_rbf(64, 100))
    wide = RBFSampler(n_components=60, random_state=0).fit(np.ones((2, 1024)))
    assert not K._rbf_reads_apart(wide)
    huge = RBFSampler(n_components=400, random_state=0).fit(np.ones((2, 256)))
    with pytest.raises(NotNative, match=r"an estimated \d+ s build, past 7 s"):
        to_native(_rbf_step(huge), strict=True, allow_bound=True)


def _skewed(n: int, m: int = 20, dtype: Any = np.float64, **kw) -> SkewedChi2Sampler:
    est = SkewedChi2Sampler(n_components=m, random_state=0, **kw)
    return est.fit(np.ones((2, n), dtype=dtype))


def _above(seed: int, n: int, s: float) -> np.ndarray:
    """Rows above -s, where the twin answers: x + s of magnitudes 1e-5 to
    1e5, near 1, and small whole numbers."""
    rng = np.random.default_rng(seed)
    p = np.vstack(
        [
            10.0 ** rng.uniform(-5, 5, size=(50, n)),
            1.0 + rng.normal(size=(20, n)) * 1e-4,
            rng.integers(1, 9, size=(20, n)).astype(float),
        ]
    )
    return p - s


@pytest.mark.parametrize("n, m, s", [(3, 20, 1.0), (12, 100, 0.5), (64, 30, -2.0)])
def test_a_skewed_chi2_sampler_parts_from_its_twin_within_its_error_scale(n, m, s):
    step = _rbf_step(_skewed(n, m, skewedness=s))
    native = to_native(step, strict=True, allow_bound=True)
    rows = _table(_above(n, n, s))
    with pytest.raises(ParityError, match="ulps, bound 0"):
        check(step, native, rows, ulps=0)
    assert check(step, native, rows) == rows.num_rows


# Doubles about -s = -1: at and below it, which the twin rejects; just
# above it (ln near -37); signed zeros and subnormals; the largest double;
# the values its validation rejects.
SKEWED_SPECIALS = [
    -1.0, -1.0000000000000002, -0.9999999999999999, -1e300, -0.5, 0.0, -0.0,
    5e-324, 1e-300, 0.5, 2.5, 1e300, 1.7976931348623157e308, math.inf,
    -math.inf, math.nan, None,
]  # fmt: skip


@pytest.mark.parametrize("s", [1.0, 0.0])
def test_a_skewed_chi2_sampler_serves_special_values(s):
    # Each pair, shifted by 1 - s, and the signed zeros: the twin answers
    # where both are above -s and finite, and the entry traps on the others
    # (at s = 0, on both zeros).
    step = _rbf_step(_skewed(2, skewedness=s))
    shift = 1.0 - s
    vals = [v if v is None or not shift else v + shift for v in SKEWED_SPECIALS]
    vals += [0.0, -0.0]
    k = len(vals)
    rows = _rows(
        [vals[i % k] for i in range(k * k)], [vals[i // k] for i in range(k * k)]
    )
    valid = [v for v in vals if v is not None and -s < v < math.inf]
    native = to_native(step, strict=True, allow_bound=True)
    assert check(step, native, rows) == len(valid) ** 2


def test_a_skewed_chi2_sampler_serves_only_on_request():
    step = _rbf_step(_skewed(2))
    assert to_native(step) is step
    assert "within 6·eps·S + τ (S its error scale)" in explain_native(step)
    with pytest.raises(NotNative, match="only with allow_bound=True"):
        to_native(step, strict=True)


def test_a_changed_skewed_weight_breaches_the_error_scale():
    est = _skewed(4)
    step = _rbf_step(est)
    native = to_native(step, strict=True, allow_bound=True)
    est.random_weights_ = est.random_weights_.copy()
    est.random_weights_[0, 0] *= 1 + 1e-9
    with pytest.raises(ParityError, match=r"past its bound K\*eps\*S \+ tau"):
        check(step, native, _table(_above(4, 4, 1.0)))


def test_a_skewed_chi2_sampler_in_a_pipeline_serves_as_its_last_step():
    est = make_pipeline(
        MinMaxScaler(), SkewedChi2Sampler(n_components=8, random_state=0)
    )
    est.fit(np.random.default_rng(5).normal(size=(30, 3)))
    step = _step(est, [pa.float64()] * 3, pa.list_(pa.float64(), 8))
    # Rows past the fit's range scale below 0, and below -1 the twin raises.
    rows = _table(np.random.default_rng(6).normal(size=(80, 3)) * 2)
    assert check(step, to_native(step, strict=True, allow_bound=True), rows) > 0


@pytest.mark.parametrize("s, answered", [(1.0, 4), (0.0, 1)])
def test_a_skewed_chi2_sampler_over_booleans_only_serves(s, answered):
    # The twin's validation makes a boolean a double; NULL is NaN, which
    # it rejects, and at s = 0 so is False (0 <= -s).
    step = _rbf_step(_skewed(2, skewedness=s), [pa.bool_()] * 2)
    rows = _rows([a for a in BITS for _ in BITS], [b for _ in BITS for b in BITS])
    assert check(step, to_native(step, strict=True, allow_bound=True), rows) == answered


def test_a_skewed_chi2_sampler_over_booleans_only_traps_where_it_rejects_both():
    # At s = -1, the twin rejects True and False (x <= 1): the entry traps
    # each row, so that `check` compares none.
    step = _rbf_step(_skewed(2, skewedness=-1.0), [pa.bool_()] * 2)
    rows = _rows([a for a in BITS for _ in BITS], [b for _ in BITS for b in BITS])
    native = to_native(step, strict=True, allow_bound=True)
    with pytest.raises(ParityError, match="the step raises on each of 9 rows"):
        check(step, native, rows)


def test_a_skewed_chi2_sampler_over_a_string_feature_stays_python():
    # The twin rejects a string, so the probe row holds one.
    step = _rbf_step(_skewed(2), [pa.string(), pa.float64()])
    with pytest.raises(NotNative, match="raises on the input guard's probe row"):
        to_native(step, strict=True, allow_bound=True)


@pytest.mark.parametrize("bits", [False, True])
def test_a_skewed_chi2_sampler_fitted_in_float32_serves(bits):
    # The twin's validation makes the row doubles, and numpy widens the
    # weights and offsets to float64 over them.
    n = 3
    step = _rbf_step(_skewed(n, dtype=np.float32), [pa.bool_()] * n if bits else None)
    if bits:
        rows = _rows(*([a for a in BITS for _ in BITS] for _ in range(n)))
    else:
        rows = _table(_above(7, n, 1.0))
    assert check(step, to_native(step, strict=True, allow_bound=True), rows)


@pytest.mark.parametrize(
    "probes, read",
    [({"log": 2}, "2 and 0"), ({"log": None}, "None and 0"), ({"cos": 1}, "1 and 1")],
)
def test_a_log_or_cos_apart_keeps_a_skewed_chi2_sampler_python(
    monkeypatch, probes, read
):
    # K = n + 4 reads numpy's log within 1 ulp of confit's, and its cos equal.
    defaults = {"log": 1, "cos": 0}
    monkeypatch.setattr(
        K, "kernel_distance", lambda func: {**defaults, **probes}[func.__name__]
    )
    with pytest.raises(NotNative, match=f"numpy log and cos are {read} ulps"):
        to_native(_rbf_step(_skewed(2)), strict=True, allow_bound=True)


def test_a_log_and_cos_equal_to_confits_serve_a_skewed_chi2_sampler(monkeypatch):
    monkeypatch.setattr(K, "kernel_distance", lambda func: 0)
    step = _rbf_step(_skewed(2))
    assert check(
        step, to_native(step, strict=True, allow_bound=True), _table(_above(3, 2, 1.0))
    )


@pytest.mark.parametrize("given", [np.longdouble(1.0), math.inf, math.nan, "1"])
def test_a_skewedness_read_otherwise_than_as_a_double_stays_python(given):
    # The fit rejects the infinities and NaN, which `transform` does not
    # read again; a long double makes the twin add in long doubles.
    est = _skewed(2)
    est.skewedness = given
    with pytest.raises(NotNative, match=r"SkewedChi2Sampler\(skewedness="):
        to_native(_rbf_step(est), strict=True, allow_bound=True)


@pytest.mark.parametrize("given", [np.float32(0.3), np.float16(0.3), 2])
def test_a_skewedness_of_another_numpy_type_is_read_as_a_double(given):
    step = _rbf_step(_skewed(2, skewedness=given))
    rows = _table(_above(4, 2, float(given)))
    assert check(step, to_native(step, strict=True, allow_bound=True), rows)


def test_a_skewed_weight_not_finite_stays_python():
    est = _skewed(2)
    est.random_offset_[3] = math.nan
    with pytest.raises(NotNative, match="a random weight or offset not finite"):
        to_native(_rbf_step(est), strict=True, allow_bound=True)


def test_a_skewed_chi2_sampler_past_2048_products_stays_python():
    # It computes the logarithm in each product: past 2,048 it would serve
    # about as slowly as the twin.
    with pytest.raises(NotNative, match="3,200 products past 2,048"):
        to_native(_rbf_step(_skewed(64, 50)), strict=True, allow_bound=True)


def test_a_skewed_chi2_sampler_takes_a_fit_by_its_estimated_build():
    # A struct grows about as n·m²: within the products, the cap takes 8
    # features of 256 components and refuses 2 of 1,024.
    assert K._skewed_build_estimate(8, 256) <= 7
    with pytest.raises(NotNative, match=r"an estimated \d+ s build, past 7 s"):
        to_native(_rbf_step(_skewed(2, 1024)), strict=True, allow_bound=True)
