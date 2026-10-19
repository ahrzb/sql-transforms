import dataclasses

import numpy as np
import pyarrow as pa
import pytest
from sklearn.preprocessing import QuantileTransformer

from benchmarks import native_estimates as B
from sql_transform.native._registry import BuildEstimate, catalog


def test_every_declared_estimate_has_a_committed_calibration_grid():
    declared = {id(model) for entry in catalog().values() for model in entry.builds}
    assert declared == {id(model) for model in B.ESTIMATES.values()}
    assert B.ESTIMATES.keys() == B.CALIBRATION.keys() == B.FIT_EXTRA.keys()
    assert all(3 <= len(shapes) <= 5 for shapes in B.CALIBRATION.values())


def test_a_single_output_uses_a_scalar_instead_of_an_invalid_width_one_list():
    est = QuantileTransformer(n_quantiles=3).fit(np.arange(3.0)[:, None])
    assert B.step_for(est, "list").returns == pa.float64()
    assert pa.types.is_struct(B.step_for(est, "struct").returns)


def test_the_timer_warms_once_and_takes_three_whole_call_measurements(monkeypatch):
    # Finding 92: trial-build seconds were passed off as whole to_native.
    # A controlled clock tests the accounting without a gate timing test.
    calls = []

    def translate(step, **kwargs):
        calls.append((step, kwargs))
        B.R._builds(step, None)

    ticks = iter([0, 1, 3, 4, 10, 11, 16, 18, 20, 21, 24, 26])
    monkeypatch.setattr(B, "to_native", translate)
    monkeypatch.setattr(B.R, "_builds", lambda *args: None)
    monkeypatch.setattr(B.time, "perf_counter", lambda: next(ticks))
    step = object()
    assert B.measure(step) == (6, 3)  # not the median trial time of 3
    assert calls == [(step, {"strict": True, "allow_bound": True})] * 4


def test_fitting_recovers_each_declared_coefficient_without_refitting_the_warm_factor():
    model = BuildEstimate(
        lambda n: {"n": (n,), "n2": (n, n)}, {"n": 8.0, "n2": 9.0}, warm_factor=1.5
    )
    rows = [(model.terms(n), 1.5 * (2.0 * n + 3.0 * n * n)) for n in (1, 3, 8, 20)]
    coefficients, band = B.fit_coefficients(model, rows)
    assert coefficients == pytest.approx({"n": 2.0, "n2": 3.0})
    assert band == pytest.approx((1.0, 1.0))
    assert model.coefficients == {"n": 8.0, "n2": 9.0}
    assert model.warm_factor == 1.5


def test_fitting_refuses_a_grid_that_cannot_identify_the_formula():
    model = B.ESTIMATES["distance"]
    terms = model.terms(512, 80)
    with pytest.raises(ValueError, match="rank 1"):
        B.fit_coefficients(model, [(terms, 6.0)] * 10)


def test_a_new_refusal_fails_the_check_instead_of_substituting_a_smaller_shape(
    monkeypatch, capsys
):
    def refuse(*args):
        raise B.R.NotNative("fixed calibration shape cannot build")

    monkeypatch.setattr(B, "fitted", refuse)
    assert not B.run("quantile")
    assert capsys.readouterr().out.count("REFUSED") == len(B.CALIBRATION["quantile"])


@pytest.mark.parametrize(
    "ratio, passes", [(0.49, False), (0.5, True), (2.0, True), (2.01, False)]
)
def test_the_check_uses_the_slower_return_and_the_documented_drift_band(
    monkeypatch, ratio, passes
):
    est = QuantileTransformer(n_quantiles=3).fit(np.arange(3.0)[:, None])
    shape = B.Shape(1, 3)
    monkeypatch.setitem(B.CALIBRATION, "quantile", [shape])
    monkeypatch.setattr(B, "fitted", lambda *args: est)
    args = B.arguments("quantile", shape, est)
    predicted = B.ESTIMATES["quantile"](*args)
    monkeypatch.setattr(
        B,
        "measure",
        lambda step: (
            predicted * (ratio if pa.types.is_struct(step.returns) else 0.1),
            predicted,
        ),
    )
    assert B.run("quantile") == passes


def test_calibration_restores_admission_and_read_selection_after_a_refusal():
    shape = B.Shape(2, 1, {"apart": False})
    est = B.fitted("rbf", shape)
    before = catalog()[type(est)]
    cap = B.project.MAX_BUILD_S
    with pytest.raises(RuntimeError), B.admitted("rbf", shape, est):
        assert not B.R._CATALOG[type(est)].apart(est)
        assert B.project.MAX_BUILD_S == float("inf")
        raise RuntimeError("failed measurement")
    assert B.project.MAX_BUILD_S == cap
    assert dataclasses.asdict(catalog()[type(est)]) == dataclasses.asdict(before)
