"""The power transforms (power.py): Box-Cox within 5·eps·|t| + τ and
Yeo-Johnson within 10·eps·(1 + max(w, 0))·|t| + τ of the twin, as the
entry's log1p and expm1, spelled from ln and exp, part from the twin's by
rounding; the entry overflows where the twin does; the kernel probe and
the build cap keep the rest Python."""

from __future__ import annotations

import math

import numpy as np
import pyarrow as pa
import pytest
from confit import DuckDBInferFn
from confit import sql as S
from sklearn.preprocessing import PowerTransformer

from sql_transform._udf import PythonTransform
from sql_transform.native import NotNative, ParityError, catalog, check, to_native
from sql_transform.native import power as P
from sql_transform.native.function import doubles_apart, probe_values

# The twin's expm1 overflows, and NaN meets the twin's branches.
pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")

E = float(np.finfo(np.float64).eps)
MAX = float(np.finfo(np.float64).max)
# Doubles where the entry can part from the twin: signed zeros, subnormals,
# values near 0 and 1, large ones up to the largest double, and the
# infinities, which the twin rejects; NaN and NULL pass.
SPECIALS = [
    0.0, -0.0, 5e-324, -5e-324, 1e-310, -1e-310, 1e-300, -1e-300, 1e-17,
    -1e-17, 0.5, -0.5, 1.0, -1.0, 2.5, -7.25, 1e15, -1e15, 1e154, -1e154,
    1e300, -1e300, MAX, -MAX, math.inf, -math.inf, math.nan, None,
]  # fmt: skip
# On and around scipy's Yeo-Johnson branches (eps under |lambda| and about
# 2), with lambdas whose transform overflows on either side.
LAMBDAS = [
    0.0, E / 2, -E / 2, E, -E, 2.0, 2 - E, 2 - 2 * E, 2 + 2 * E, 1.0, 1 + 1e-9,
    1 - 1e-9, 0.5, 1.5, 3.0, 50.0, -1.0, -3.0, -50.0, 1e300, -1e300,
]  # fmt: skip


def _fitted(
    method: str, lambdas: list[float], dtype: type = np.float64
) -> PowerTransformer:
    """A PowerTransformer(method, standardize=False) whose `lambdas_` are
    pinned to `lambdas`, in `dtype`."""
    X = np.random.default_rng(0).lognormal(size=(20, len(lambdas)))
    est = PowerTransformer(method, standardize=False).fit(X)
    est.lambdas_ = np.array(lambdas, dtype)
    return est


def _step(est: PowerTransformer) -> PythonTransform:
    n = len(est.lambdas_)
    takes = pa.schema([(f"x{i}", pa.float64()) for i in range(n)])
    returns = pa.float64() if n == 1 else pa.list_(pa.float64(), n)
    return PythonTransform("tf", {0: est}, takes, returns)


def _rows(*cols: list) -> pa.Table:
    return pa.table(
        {
            "__iid": pa.array([0] * len(cols[0]), pa.int64()),
            **{f"x{i}": pa.array(c, pa.float64()) for i, c in enumerate(cols)},
        }
    )


def _confit(expr: S.Expr, x: np.ndarray) -> np.ndarray:
    """`expr` over the column `x`, as confit computes it."""
    rows = pa.table({"x": x})
    sql = f"SELECT {expr.sql()} AS y FROM __THIS__"  # noqa: S608 — a fixed spelling
    f = DuckDBInferFn(
        sql, row_tables={"__THIS__": rows.schema}, static_tables={}, udfs=[]
    )
    return f.infer_arrow(rows).column("y").to_numpy()


def test_this_platforms_kernels_pass_the_probe():
    d = P.kernel_distances()
    assert d is not None
    assert d["ln"] == d["exp"] == 0
    assert d["log1p"] <= 1 and d["expm1"] <= 1


def test_the_spellings_are_within_2_ulps_of_the_c_librarys():
    # Measured; the derivation allows 7 (`log1p`, `expm1`).
    x = probe_values()
    z = np.abs(x[np.isfinite(x)])
    got = _confit(P.log1p(S.col("x")), z)
    assert doubles_apart(P._c_library(math.log1p, z), got) <= 2
    w = x[np.isfinite(x) & (x < 709.78)]
    got = _confit(P.expm1(S.col("x")), w)
    assert doubles_apart(P._c_library(math.expm1, w), got) <= 2


def test_the_spellings_keep_signed_zeros_subnormals_and_overflow():
    z = np.array([0.0, -0.0, 5e-324, 1e-300, MAX])
    got = _confit(P.log1p(S.col("x")), z)
    assert [repr(v) for v in got[:4].tolist()] == ["0.0", "-0.0", "5e-324", "1e-300"]
    assert math.isfinite(got[4])
    w = np.array([0.0, -0.0, 5e-324, -1e10, 709.79, 1e10])
    got = _confit(P.expm1(S.col("x")), w)
    want = ["0.0", "-0.0", "5e-324", "-1.0", "inf", "inf"]
    assert [repr(v) for v in got.tolist()] == want


@pytest.mark.parametrize("dtype", [np.float64, np.float32], ids=["f64", "f32"])
def test_yeo_johnson_matches_its_twin_on_special_values(dtype):
    # ±1e300 rounds to an infinite float32.
    lambdas = [v for v in LAMBDAS if abs(v) < 1e38 or dtype == np.float64]
    est = _fitted("yeo-johnson", lambdas, dtype)
    step = _step(est)
    native = to_native(step, strict=True, allow_bound=True)
    # Each row holds one value in every feature. The twin rejects the
    # infinities, where the entry traps, and `check` compares the rest.
    rows = _rows(*[SPECIALS] * len(lambdas))
    answered = [v for v in SPECIALS if v is None or not math.isinf(v)]
    assert check(step, native, rows) == len(answered)


def test_box_cox_matches_its_twin_on_special_values():
    lambdas = [0.0, 1e-20, 1e-19, -3e-12, 0.5, 3.0, -50.0, 2.0, 1e300]
    est = _fitted("box-cox", lambdas)
    step = _step(est)
    native = to_native(step, strict=True, allow_bound=True)
    # The twin rejects x <= 0 and the infinities, where the entry traps.
    rows = _rows(*[SPECIALS] * len(lambdas))
    answered = [v for v in SPECIALS if v is None or not (v <= 0 or math.isinf(v))]
    assert check(step, native, rows) == len(answered)


@pytest.mark.parametrize("lam", [3.0, 1.5, 1 + 1e-9, 1.0, 50.0, -1.0, 0.0, 1 - 1e-9])
def test_the_entry_overflows_where_the_twin_does(lam):
    # The twin computes expm1(w) before it divides by c (lambda, or 2 -
    # lambda below 0), so with c > 1 it is infinite where t is about
    # DBL_MAX / c; the entry answers ±inf from the twin's own edge on.
    est = _fitted("yeo-johnson", [lam])
    step = _step(est)
    native = to_native(step, strict=True, allow_bound=True)
    lam_ = np.asarray(est.lambdas_)
    k = np.arange(-3000, 3001, 7)
    for positive in (True, False):
        edge = P._edges(lam_, positive)[0]
        if math.isnan(edge):
            continue
        top = np.array([MAX]).view(np.int64)[0]
        bits = np.minimum(np.array([edge]).view(np.int64)[0] + k, top)
        xs = bits.view(np.float64) * (1.0 if positive else -1.0)
        t = P._yeo_johnson_t(lam_, np.abs(xs), positive)
        assert np.isinf(t).any() and np.isfinite(t).any()
        assert check(step, native, _rows(xs.tolist())) == len(xs)


def test_an_edge_the_twin_does_not_keep_leaves_the_step_python(monkeypatch):
    # Infinite from 1e10 on, but for a run of doubles just past it.
    def twin(lam, v, positive):
        v = np.asarray(v, np.float64)
        hole = (v > 1e10 + 1e-3) & (v < 1e10 + 2e-3)
        return np.where((v >= 1e10) & ~hole, np.inf, 1.0) + 0 * lam

    monkeypatch.setattr(P, "_yeo_johnson_t", twin)
    with pytest.raises(NotNative, match="overflows on no single edge near 1000000000"):
        to_native(_step(_fitted("yeo-johnson", [3.0])), strict=True, allow_bound=True)


def test_the_fields_part_from_the_twin():
    # Not bit-exact: the bound is the error scale.
    est = _fitted("yeo-johnson", [0.5, 3.0, -1.5])
    step = _step(est)
    native = to_native(step, strict=True, allow_bound=True)
    xs = np.random.default_rng(1).normal(size=(3, 200)) * 50
    assert check(step, native, _rows(*xs.tolist())) == 200
    with pytest.raises(ParityError, match="output field"):
        check(step, native, _rows(*xs.tolist()), ulps=0)


def test_the_error_scale_reads_the_twins_branches():
    s = catalog()[PowerTransformer].scale.s
    est = _fitted("yeo-johnson", [0.5, 0.5, 0.0, 2.0, 2 + 2 * E])
    x = np.array([3.0, -3.0, 3.0, -3.0, -3.0], np.longdouble)
    got = s(est, x)
    log4 = np.log(np.longdouble(4))
    # x >= 0: w = 0.5·log(4), t = (2 - 1) / 0.5; x < 0: w = 1.5·log(4),
    # t = -(8 - 1) / 1.5; log1p branches: w = 0, t = ±log(4); and lambda
    # 2 + 2 eps, past eps from 2: w = -2 eps·log(4), below 0.
    want = [
        (1 + 0.5 * log4) * 2,
        (1 + 1.5 * log4) * 7 / np.longdouble(1.5),
        log4,
        log4,
        log4,
    ]
    assert np.allclose(got, np.array(want, np.longdouble), rtol=1e-15)
    est = _fitted("box-cox", [0.0, 2.0])
    got = s(est, np.array([4.0, 3.0], np.longdouble))
    assert np.allclose(got, np.array([log4, 4.0], np.longdouble), rtol=1e-15)


def test_a_failing_kernel_probe_leaves_the_step_python(monkeypatch):
    def refuse(*args, **kwargs):
        raise RuntimeError("no engine")

    monkeypatch.setattr(P, "DuckDBInferFn", refuse)
    P.kernel_distances.cache_clear()
    try:
        for method in ("box-cox", "yeo-johnson"):
            step = _step(_fitted(method, [0.5]))
            with pytest.raises(NotNative, match="the kernel probe did not run"):
                to_native(step, strict=True, allow_bound=True)
    finally:
        P.kernel_distances.cache_clear()


@pytest.mark.parametrize(
    "probes, method, reason",
    [
        ({"ln": 1}, "box-cox", "confit's ln and exp are 1 and 0 ulps from the C"),
        ({"exp": 2}, "yeo-johnson", "confit's ln and exp are 0 and 2 ulps"),
        ({"log1p": 2}, "yeo-johnson", r"log1p and expm1 are 2 and 1 ulps .* \(1\)"),
        ({"expm1": 1 << 64}, "yeo-johnson", "past what K = 10 reads"),
    ],
    ids=lambda v: None,
)
def test_a_kernel_past_what_the_bound_reads_leaves_the_step_python(
    monkeypatch, probes, method, reason
):
    d = {"ln": 0, "exp": 0, "log1p": 1, "expm1": 1, **probes}
    monkeypatch.setattr(P, "kernel_distances", lambda: d)
    with pytest.raises(NotNative, match=reason):
        to_native(_step(_fitted(method, [0.5])), strict=True, allow_bound=True)


def test_box_cox_does_not_read_numpys_log1p(monkeypatch):
    d = {"ln": 0, "exp": 0, "log1p": 5, "expm1": 5}
    monkeypatch.setattr(P, "kernel_distances", lambda: d)
    step = _step(_fitted("box-cox", [0.5]))
    assert to_native(step, strict=True, allow_bound=True) is not step


@pytest.mark.parametrize("method", ["box-cox", "yeo-johnson"])
@pytest.mark.parametrize("lam", [math.inf, math.nan])
def test_a_lambda_not_finite_leaves_the_step_python(method, lam):
    with pytest.raises(NotNative, match="a lambda not finite"):
        to_native(_step(_fitted(method, [0.5, lam])), strict=True, allow_bound=True)


def test_the_build_estimate_takes_what_builds_within_the_cap():
    # Of the widths it was fitted on, these built in 6.4 s or less, and
    # those in 7.0 s or more (the slower of a list and a struct).
    assert P._build_estimate(320, "yeo-johnson") <= P.MAX_BUILD_S
    assert P._build_estimate(352, "yeo-johnson") > P.MAX_BUILD_S
    assert P._build_estimate(640, "box-cox") <= P.MAX_BUILD_S
    assert P._build_estimate(704, "box-cox") > P.MAX_BUILD_S


@pytest.mark.parametrize("method, n", [("yeo-johnson", 352), ("box-cox", 704)])
def test_a_power_transform_past_its_estimated_build_stays_python(method, n):
    # The lambdas alone: a fit over this many features is slow.
    est = PowerTransformer(method, standardize=False)
    est.lambdas_ = np.full(n, 0.5)
    reason = rf"an estimated 7 s build, past 7 s \({n} features\)"
    with pytest.raises(NotNative, match=reason):
        to_native(_step(est), strict=True, allow_bound=True)
