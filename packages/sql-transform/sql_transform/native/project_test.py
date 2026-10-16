"""Linear projections within the matvec parity bound (project.py): the
twin's BLAS sum and the entry's balanced tree part by rounding alone, on
rows near the mean where the two products cancel, and the centered
projections, which subtract the mean before the product."""

from __future__ import annotations

import math

import numpy as np
import pyarrow as pa
import pytest
from sklearn.cross_decomposition import CCA, PLSSVD, PLSCanonical, PLSRegression
from sklearn.decomposition import PCA, FastICA, IncrementalPCA, TruncatedSVD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.random_projection import GaussianRandomProjection

from sql_transform._udf import PythonTransform
from sql_transform.native import (
    NotNative,
    ParityError,
    check,
    explain_native,
    to_native,
)

# A FastICA fit may stop at `max_iter` on these data: its transform does the
# same arithmetic either way.
pytestmark = pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")


def _data(seed: int, n: int, rank: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Fit rows of `n` features of scales 1e-2 to 1e4 and large means, of
    rank `rank` (full by default), and the means."""
    rng = np.random.default_rng(seed)
    scales = 10.0 ** rng.uniform(-2, 4, size=n)
    means = rng.uniform(-100, 100, size=n) * scales
    r = rank or n
    return rng.normal(size=(200, r)) @ rng.normal(size=(r, n)) * scales + means, means


def _near(seed: int, x: np.ndarray, means: np.ndarray) -> np.ndarray:
    """Rows within 1e-6 of a feature's scale of the mean, where `x·c` and
    `m·c` cancel, then the fit's own first rows."""
    rng = np.random.default_rng(seed)
    scales = x.std(axis=0) + 1.0
    return np.vstack(
        [means + rng.normal(size=(40, len(means))) * scales * 1e-6, x[:20]]
    )


def _step(est) -> PythonTransform:
    n = est.n_features_in_
    k = np.asarray(est.transform(np.zeros((1, n)))).shape[1]
    return PythonTransform(
        "tf",
        {0: est},
        pa.schema([(f"x{i}", pa.float64()) for i in range(n)]),
        pa.struct([(f"f{j}", pa.float64()) for j in range(k)])
        if k > 1
        else pa.float64(),
    )


def _rows(x: np.ndarray) -> pa.Table:
    cols = {f"x{i}": pa.array(x[:, i]) for i in range(x.shape[1])}
    return pa.table({"__iid": pa.array([0] * len(x), pa.int64()), **cols})


MAKES = [
    lambda: PCA(n_components=3),
    lambda: PCA(n_components=3, whiten=True, svd_solver="covariance_eigh"),
    lambda: IncrementalPCA(n_components=2, whiten=True),
    lambda: TruncatedSVD(n_components=3, random_state=0),
    lambda: GaussianRandomProjection(n_components=3, random_state=0),
]


@pytest.mark.parametrize("make", MAKES, ids=lambda m: type(m()).__name__)
def test_a_projection_parts_from_its_twin_within_its_error_scale(make):
    x, means = _data(0, 12)
    est = make().fit(x)
    step = _step(est)
    native = to_native(step, strict=True, allow_bound=True)
    rows = _rows(_near(1, x, means))
    with pytest.raises(ParityError, match="ulps, bound 0"):
        check(step, native, rows, ulps=0)
    assert check(step, native, rows) == rows.num_rows


@pytest.mark.parametrize("make", MAKES[3:], ids=lambda m: type(m()).__name__)
def test_a_sum_of_negative_zeros_answers_0(make):
    # Rows of zeros signed against each component, so that every product of
    # its output field is -0.0: the twin answers 0.0 there, and so must the
    # entry, bit for bit (finding 77).
    x, _ = _data(7, 12)
    est = make().fit(x)
    step = _step(est)
    native = to_native(step, strict=True, allow_bound=True)
    zeros = [np.copysign(0.0, -c) for c in est.components_]
    rows = _rows(np.array([*zeros, np.zeros(12), -np.zeros(12)]))
    assert check(step, native, rows, ulps=0) == rows.num_rows


def test_a_projection_serves_only_on_request():
    x, _ = _data(0, 4)
    step = _step(PCA(n_components=2).fit(x))
    assert to_native(step) is step
    with pytest.raises(NotNative, match=r"PCA is within 7·eps·S \+ τ"):
        to_native(step, strict=True)
    assert explain_native(step, allow_bound=True).endswith(
        "-> SqlFunction, within 7·eps·S + τ (S its error scale)"
    )


def test_a_changed_component_breaches_the_error_scale():
    # A structural change moves an output field by about S/n, far past
    # (n + 3)·eps·S: here the largest coefficient, by 1e-9 of itself.
    x, means = _data(2, 8)
    est = PCA(n_components=2).fit(x)
    step = _step(est)
    native = to_native(step, strict=True, allow_bound=True)
    est.components_ = est.components_.copy()
    est.components_[0, np.argmax(np.abs(est.components_[0]))] *= 1 + 1e-9
    with pytest.raises(ParityError, match="past its bound"):
        check(step, native, _rows(_near(3, x, means)))


@pytest.mark.parametrize("make", [PCA, IncrementalPCA])
def test_a_whitened_component_of_rounding_noise_stays_python(make):
    # Rank 3 over 6 features: the fit's last components have a variance of
    # about eps times the largest, which whitening scales to size O(1).
    x, _ = _data(4, 6, rank=3)
    est = make(whiten=True).fit(x)
    with pytest.raises(NotNative, match="a whitened component of rounding noise"):
        to_native(_step(est), strict=True, allow_bound=True)
    # Without whitening the same fit serves.
    to_native(_step(make().fit(x)), strict=True, allow_bound=True)


def test_a_float32_mean_stays_python():
    x, _ = _data(5, 4)
    est = PCA(n_components=2).fit(x.astype(np.float32))
    with pytest.raises(NotNative, match="projects its mean in float32"):
        to_native(_step(est), strict=True, allow_bound=True)


@pytest.mark.parametrize("make", MAKES[3:], ids=lambda m: type(m()).__name__)
def test_a_float32_fit_without_a_mean_serves(make):
    # numpy widens the float32 components to float64 before the product.
    x, means = _data(5, 6)
    est = make().fit(x.astype(np.float32))
    step = _step(est)
    native = to_native(step, strict=True, allow_bound=True)
    rows = _rows(_near(6, x, means))
    assert check(step, native, rows) == rows.num_rows


def test_a_projection_takes_a_fit_by_its_estimated_build():
    from sql_transform.native.project import MAX_BUILD_S, _build_estimate

    # 128 components over 512 features built in 5.9 to 6.3 s, 32 over 2,048
    # in 8.1 to 8.8 s (project.py, `_build_estimate`).
    assert _build_estimate(512, 128) <= MAX_BUILD_S
    assert _build_estimate(2048, 32) > MAX_BUILD_S
    x = np.random.default_rng(6).normal(size=(10, 2048))
    est = GaussianRandomProjection(n_components=32, random_state=0).fit(x)
    with pytest.raises(
        NotNative,
        match=r"an estimated \d+ s build, past 7 s \(32 components over 2,048",
    ):
        to_native(_step(est), strict=True, allow_bound=True)


@pytest.mark.parametrize(
    "make",
    [lambda: PCA(n_components=3), lambda: FastICA(whiten=False, random_state=0)],
    ids=["PCA", "FastICA-unwhitened"],
)
def test_a_projection_reads_a_shared_value_where_its_build_allows(make, monkeypatch):
    # Each product reads `x - 0.0`, a value confit computes once a row, where
    # `_centered_build_estimate` keeps the build within the cap; past that it
    # reads `coalesce(x, NaN)`, and builds by `_build_estimate`
    # (`_reads_apart`). Either way it parts from the twin by rounding alone.
    from sql_transform.native import project as P

    x, means = _data(2, 12)
    est = _fit(make, x)
    step = _step(est)
    rows = _rows(_near(3, x, means))
    apart = "- CAST('0.0' AS DOUBLE)"
    native = to_native(step, strict=True, allow_bound=True)
    assert apart in native.sql_body
    assert check(step, native, rows) == rows.num_rows
    k, n = np.shape(
        P._centered(est)[2] if isinstance(est, FastICA) else est.components_
    )
    lo, hi = P._build_estimate(n, k), P._centered_build_estimate(n, k)
    monkeypatch.setattr(P, "MAX_BUILD_S", (lo + hi) / 2)
    native = to_native(step, strict=True, allow_bound=True)
    assert apart not in native.sql_body
    assert check(step, native, rows) == rows.num_rows


def _fit(make, x: np.ndarray, seed: int = 0):
    """`make()` fitted on `x`: on 4 classes for a discriminant analysis, on
    2 targets of `x` with noise for the PLS family, on `x` standardized for
    an unwhitened FastICA (which takes its data as white), else on `x`."""
    est = make()
    rng = np.random.default_rng(seed)
    if isinstance(est, FastICA) and not est.whiten:
        return est.fit((x - x.mean(axis=0)) / x.std(axis=0))
    if isinstance(est, LinearDiscriminantAnalysis):
        return est.fit(x, np.arange(len(x)) % 4)
    if isinstance(est, PLSRegression | PLSCanonical | CCA | PLSSVD):
        y = x @ rng.normal(size=(x.shape[1], 2))
        return est.fit(x, y + rng.normal(size=y.shape) * y.std())
    return est.fit(x)


CENTERED = [
    lambda: FastICA(n_components=3, random_state=0),
    lambda: FastICA(whiten=False, random_state=0),
    LinearDiscriminantAnalysis,
    lambda: LinearDiscriminantAnalysis(solver="eigen"),
    lambda: PLSRegression(n_components=3),
    lambda: PLSCanonical(n_components=2, scale=False),
    lambda: CCA(n_components=2),
    lambda: PLSSVD(n_components=2),
]


def _id(make) -> str:
    est = make()
    if isinstance(est, FastICA) and not est.whiten:
        return "FastICA-unwhitened"
    if isinstance(est, LinearDiscriminantAnalysis):
        return f"LinearDiscriminantAnalysis-{est.solver}"
    return type(est).__name__


@pytest.mark.parametrize("make", CENTERED, ids=_id)
def test_a_centered_projection_parts_from_its_twin_within_its_error_scale(make):
    x, means = _data(0, 12)
    est = _fit(make, x)
    step = _step(est)
    native = to_native(step, strict=True, allow_bound=True)
    rows = _rows(_near(1, x, means))
    with pytest.raises(ParityError, match="ulps, bound 0"):
        check(step, native, rows, ulps=0)
    assert check(step, native, rows) == rows.num_rows
    assert explain_native(step, allow_bound=True).endswith(
        "-> SqlFunction, within 13·eps·S + τ (S its error scale)"
    )


@pytest.mark.parametrize("make", CENTERED, ids=_id)
def test_a_centered_projection_answers_0_where_every_product_is_a_zero(make):
    # At the mean the twin multiplies a row of 0.0. Without a mean, rows of
    # zeros signed against each output field's matrix row make every product
    # of that field -0.0 (finding 77). The twin answers 0.0 in both.
    from sql_transform.native.project import _centered

    x, _ = _data(8, 12)
    est = _fit(make, x)
    step = _step(est)
    native = to_native(step, strict=True, allow_bound=True)
    m, _, c = _centered(est)
    zeros = [np.zeros(12)] if m is None else [m]
    if m is None:
        zeros += [np.copysign(0.0, -cj) for cj in c]
    rows = _rows(np.array(zeros))
    assert check(step, native, rows, ulps=0) == rows.num_rows


ONE_FEATURE = [
    lambda: FastICA(random_state=0),
    lambda: FastICA(whiten=False, random_state=0),
    LinearDiscriminantAnalysis,
    lambda: LinearDiscriminantAnalysis(solver="eigen"),
    lambda: PLSRegression(n_components=1),
    lambda: PLSCanonical(n_components=1, scale=False),
    lambda: CCA(n_components=1),
    lambda: PLSSVD(n_components=1),
]


@pytest.mark.parametrize("make", ONE_FEATURE, ids=_id)
def test_a_centered_projection_of_one_feature_answers_the_twins_zero(make):
    # At one feature, the twin of FastICA and of the PLS classes multiplies
    # a 1×1 row by a 1×1 matrix with `np.dot`, which answers the product
    # itself: -0.0 where the difference is a zero and the coefficient has
    # the other sign (finding 81). The `@` of LinearDiscriminantAnalysis
    # answers 0.0. The matrix is negated in turn, so that one of the two
    # signs gives -0.0: the twin and the entry read the same fitted state.
    from sql_transform.native.project import _centered

    x, _ = _data(10, 1)
    answers = []
    for negate in (False, True):
        est = _fit(make, x)
        for name in ("components_", "scalings_", "x_rotations_", "x_weights_"):
            if negate and hasattr(est, name):
                setattr(est, name, -getattr(est, name))
        m = _centered(est)[0]
        rows = _rows(np.array([[0.0], [-0.0]] if m is None else [m]))
        step = _step(est)
        native = to_native(step, strict=True, allow_bound=True)
        assert check(step, native, rows, ulps=0) == rows.num_rows
        answers += [step(0, v)[0] for v in rows.column("x0").to_pylist()]
    assert all(a == 0.0 for a in answers)
    signs = {math.copysign(1.0, a) for a in answers}
    assert signs == (
        {1.0} if isinstance(est, LinearDiscriminantAnalysis) else {1.0, -1.0}
    )


def _step_of(est, k: int) -> PythonTransform:
    n = est.n_features_in_
    return PythonTransform(
        "tf",
        {0: est},
        pa.schema([(f"x{i}", pa.float64()) for i in range(n)]),
        pa.struct([(f"f{j}", pa.float64()) for j in range(k)]),
    )


def test_a_discriminant_analysis_by_least_squares_stays_python():
    x, _ = _data(9, 6)
    est = _fit(lambda: LinearDiscriminantAnalysis(solver="lsqr"), x)
    with pytest.raises(NotNative, match=r"solver='lsqr'\) has no transform"):
        to_native(_step_of(est, 3), strict=True, allow_bound=True)


@pytest.mark.parametrize("solver", ["svd", "eigh"])
def test_a_fastica_whitened_over_rounding_noise_stays_python(solver):
    # Rank 3 over 6 features: three singular values of the centered fit are
    # rounding noise, which the whitening divides by.
    x, _ = _data(4, 6, rank=3)
    est = FastICA(whiten_solver=solver, random_state=0).fit(x)
    with pytest.raises(NotNative, match="a whitened component of rounding noise"):
        to_native(_step(est), strict=True, allow_bound=True)
    # Over the three components that are not noise, the same data serves.
    est = FastICA(n_components=3, whiten_solver=solver, random_state=0).fit(x)
    to_native(_step(est), strict=True, allow_bound=True)


@pytest.mark.parametrize("make", CENTERED[::2], ids=_id)
def test_a_centered_projection_fitted_in_float32_serves(make):
    # numpy widens the float32 mean, scale and matrix to float64 before it
    # subtracts, divides and multiplies.
    x, means = _data(5, 6)
    est = _fit(make, x.astype(np.float32))
    step = _step(est)
    native = to_native(step, strict=True, allow_bound=True)
    rows = _rows(_near(6, x, means))
    assert check(step, native, rows) == rows.num_rows


@pytest.mark.filterwarnings("ignore:divide by zero encountered:RuntimeWarning")
def test_a_fastica_fit_that_broke_down_stays_python():
    # Unwhitened, FastICA takes its data as white. On data far from white,
    # its unmixing matrix collapses to rows of norm 1/sqrt(3), not 1.
    x = np.random.default_rng(0).normal(size=(50, 3)) * 1e3 + 1e4
    est = FastICA(whiten=False, random_state=0).fit(x)
    with pytest.raises(
        NotNative, match=r"a norm of 0\.57735028, not 1: its fit broke down"
    ):
        to_native(_step(est), strict=True, allow_bound=True)
    # A constant column whose mean rounds: the source has no spread, and
    # "unit-variance" divides its row by 0.
    est = FastICA(random_state=0).fit(np.full((9, 1), 3.6))
    with pytest.raises(
        NotNative, match="components_ is not finite: its fit broke down"
    ):
        to_native(_step(est), strict=True, allow_bound=True)


def test_a_centered_projection_takes_a_fit_by_its_estimated_build():
    from sql_transform.native.project import (
        MAX_BUILD_S,
        _build_estimate,
        _centered_build_estimate,
    )

    # Centered, 96 components over 512 features built in 5.3 to 6.5 s, 48
    # over 1,024 in 7.8 to 8.4 s: past the cap, though a PCA of 48 over
    # 1,024 is within it (project.py, `_centered_build_estimate`).
    assert _centered_build_estimate(512, 96) <= MAX_BUILD_S
    assert _build_estimate(1024, 48) <= MAX_BUILD_S < _centered_build_estimate(1024, 48)
    x = np.random.default_rng(7).normal(size=(98, 1024))
    est = LinearDiscriminantAnalysis().fit(x, np.arange(98) % 49)
    with pytest.raises(
        NotNative,
        match=r"an estimated \d+ s build, past 7 s \(48 components over 1,024",
    ):
        to_native(_step_of(est, 48), strict=True, allow_bound=True)
