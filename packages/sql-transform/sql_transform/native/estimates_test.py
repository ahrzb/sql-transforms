"""Admission-model checks without a timed build in the gate."""

from dataclasses import replace
from itertools import product
from types import SimpleNamespace

import numpy as np
import pytest
from confit import sql as S
from scipy import sparse
from sklearn.kernel_approximation import AdditiveChi2Sampler

from sql_transform.native import _registry as R
from sql_transform.native import (
    distance,
    isotonic,
    kernel,
    power,
    project,
    quantile,
    spline,
    trees,
)


def test_every_entry_declares_a_build_estimate_or_why_it_has_none():
    # AdditiveChi2 shipped without any cap in T49/T50; finding 91 measured
    # 28.5 s before T53 added its estimate. New entries must make this choice.
    for cls, entry in R.catalog().items():
        assert bool(entry.builds) != bool(entry.uncapped and entry.uncapped.strip()), (
            f"{cls.__name__}: declare builds or an explicit uncapped reason"
        )
        assert all(isinstance(model, R.BuildEstimate) for model in entry.builds)


def test_removing_additive_chi2s_declaration_fails_the_registry_check(monkeypatch):
    entry = replace(R._CATALOG[AdditiveChi2Sampler], builds=())
    monkeypatch.setitem(R._CATALOG, AdditiveChi2Sampler, entry)
    with pytest.raises(AssertionError, match="AdditiveChi2Sampler: declare builds"):
        test_every_entry_declares_a_build_estimate_or_why_it_has_none()


def test_named_terms_preserve_the_ten_whole_build_formulas_bit_for_bit():
    # The pre-declaration formulas, independent of BuildEstimate's evaluator.
    # Grids span each docstring's range, including the one-output branches.
    for n, k in product(
        (1, 2, 16, 64, 256, 512, 1024, 2048, 4096, 6144), (1, 2, 8, 32, 128, 256, 2000)
    ):
        p = n * k
        plain = (
            6.10e-4 * n
            + 9.28e-5 * p
            + 2.14e-8 * p * n
            + 6.82e-8 * p * k
            + 1.14e-7 * n * n
        )
        assert project._build_estimate(n, k) == plain
        assert project._centered_build_estimate(n, k) == plain + 3.01e-8 * n * n * k + (
            2.99e-7 * n * n if k > 1 else 0.0
        )
        assert (
            project._factor_build_estimate(n, k)
            == 5.97e-4 * n
            + 4.70e-5 * p
            + 1.78e-7 * p * n
            + 3.07e-7 * p * k
            + (1.89e-7 * n * n if k > 1 else 0.0)
            + 8.48e-5 * k * k
        )
        assert (
            distance._build_estimate(n, k)
            == 1.14e-3 * n
            + 8.11e-4 * k
            + 7.47e-5 * n * k
            + 3.76e-7 * n * n
            + 6.64e-8 * n * n * k
            + 7.68e-8 * n * k * k
            + 7.75e-9 * k**3
        )
        assert kernel._rbf_build_estimate(
            n, k, True
        ) == 7.61e-4 * n + 5.21e-4 * k + 1.01e-4 * n * k + 1.23e-7 * n * n * k + (
            2.20e-7 * n * n if k > 1 else 0.0
        )
        assert (
            kernel._rbf_build_estimate(n, k, False)
            == 6.48e-4 * n + 5.80e-4 * k + 2.15e-4 * n * k + 6.17e-8 * n * n * k
        )
        assert (
            kernel._skewed_build_estimate(n, k)
            == 7.40e-4 * n
            + 3.57e-4 * n * k
            + 7.48e-7 * n * n * k
            + 8.76e-6 * n * k * k
            + (6.13e-6 * n * n if k > 1 else 0.0)
        )
        assert (
            kernel._nystroem_build_estimate(n, k)
            == 4.36e-4 * n
            + 7.05e-5 * n * k
            + 5.40e-5 * k * k
            + 1.35e-7 * k**3
            + 5.55e-7 * n * n
        )
    for n, steps in product(
        (1, 3, 32, 128, 256, 896, 1024), (1, 2, 3, 8, 24, 100, 400)
    ):
        f, p = n * (2 * steps - 1), n * (steps - 1)
        assert (
            kernel._additive_chi2_build_estimate(n, steps)
            == 9.06e-4 * f
            + 3.80e-6 * n * f
            + 6.92e-5 * n * p
            + 4.28e-9 * n * n * f
            + 2.16e-7 * n * p * p
        )
    for n in (48, 64, 128, 256, 416, 640, 768, 960):
        assert (
            power._build_estimate(n, "box-cox", False) == 1.17e-3 * n + 1.25e-5 * n * n
        )
        assert (
            power._build_estimate(n, "yeo-johnson", False)
            == 5.69e-3 * n + 4.15e-5 * n * n
        )
        assert power._build_estimate(n, "box-cox", True) == 3.92e-3 * n + 9.3e-6 * n * n
        assert (
            power._build_estimate(n, "yeo-johnson", True)
            == 6.19e-3 * n + 5.13e-5 * n * n
        )
    rng = np.random.default_rng(4)
    for n, k in product((16, 64, 1024, 4096), (1, 16, 64)):
        c = sparse.random(k, n, density=0.1, rng=rng, format="csr")
        if k > 1:
            c = sparse.vstack([c[: k // 2], c[: k - k // 2]], format="csr")
        m = np.diff(c.indptr).astype(np.float64)
        p, q, r = float(m.sum()), float(m @ m), project._repeated_sums(c)
        assert (
            project._sparse_build_estimate(c)
            == 4.99e-4 * n
            + 1.08e-7 * n * n
            + 6.43e-5 * p
            + 2.72e-7 * q
            + 6.55e-6 * n * k
            + 2.22e-9 * n * n * k
            + 1.30e-7 * k * k
            + 1.68e-5 * k * r
        )


def test_the_four_trial_fits_change_only_by_the_measured_phase_correction():
    for pieces in (3417, 8000, 18000, 24702, 28523, 31724):
        old = 1.33e-4 * pieces + 4.79e-9 * pieces**2
        assert (
            isotonic._build_estimate(pieces)
            == old * isotonic._build_estimate.warm_factor
        )
    for leaves, arms, n in product(
        (9, 7999, 11984, 16000, 24000), (0, 10, 1000), (1, 16, 512, 800)
    ):
        weighted = leaves + 0.4 * arms
        old = 3.04e-4 * weighted + 1.07e-8 * weighted**2 + 7.91e-6 * n**2
        assert quantile._build_estimate(leaves, arms, n) == pytest.approx(
            (old + 9.8e-6 * n**2) * quantile._build_estimate.warm_factor,
            rel=5e-16,
            abs=0,
        )
    for fields, n in product((693, 2286, 5758, 9210, 11402), (8, 32, 64, 128)):
        est = SimpleNamespace(
            one_hot_encoder_=SimpleNamespace(categories_=[range(fields)]),
            n_features_in_=n,
        )
        old = 1.89e-4 * fields + 4.56e-8 * fields**2 + 7.19e-6 * fields * n
        assert trees._build_estimate(est) == old * trees._build_estimate.warm_factor
    for n, width in product((8, 32, 128, 320), (1, 5, 8, 32)):
        x = S.col("x")
        fields = [x + S.lit(float(i)) for i in range(width)]
        seen, stack = set(), list(fields)
        while stack:
            expr = stack.pop()
            if id(expr) not in seen:
                seen.add(id(expr))
                stack.extend(expr.children)
        old = 2.37e-5 * len(seen) + 7.53e-9 * len(seen) * width + 6.48e-6 * width * n
        assert (
            spline._build_estimate(fields, SimpleNamespace(n_features_in_=n))
            == old * spline._build_estimate.warm_factor
        )
