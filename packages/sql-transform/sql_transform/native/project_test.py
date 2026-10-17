"""Linear projections within the matvec parity bound (project.py): the
twin's BLAS sum and the entry's balanced tree part by rounding alone, on
rows near the mean where the two products cancel, and the centered
projections, which subtract the mean before the product."""

from __future__ import annotations

import math

import numpy as np
import pyarrow as pa
import pytest
from scipy import sparse
from sklearn.cross_decomposition import CCA, PLSSVD, PLSCanonical, PLSRegression
from sklearn.decomposition import (
    PCA,
    FactorAnalysis,
    FastICA,
    IncrementalPCA,
    TruncatedSVD,
)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection
from threadpoolctl import threadpool_limits

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

    # Read `coalesce(x, NaN)`, 96 components over 512 features built in 5.1
    # to 6.0 s and 2 over 4,096 in 5.5 to 5.9 s; 63 over 1,024 in 7.2 to
    # 8.5 s and one over 6,144 in 8.7 to 9.5 s (project.py,
    # `_build_estimate`).
    assert _build_estimate(512, 96) <= MAX_BUILD_S
    assert _build_estimate(4096, 2) <= MAX_BUILD_S
    assert _build_estimate(1024, 63) > MAX_BUILD_S
    assert _build_estimate(6144, 1) > MAX_BUILD_S
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


def test_a_list_of_few_components_over_wide_rows_is_estimated_so():
    # A list over shared values builds about as their number squared
    # (finding 87). Read `x - 0.0`, 2 components over 4,096 features built
    # in 11.3 s as a list and 5.9 s as a struct, and one component in 5.0
    # to 5.1 s; read `coalesce(x, NaN)`, 2 in 5.5 to 5.9 s (project.py,
    # `_centered_build_estimate`). Where the twin subtracts a mean, each
    # product reads a shared difference in either reading.
    from sql_transform.native import project as P

    assert P._build_estimate(4096, 2) <= P.MAX_BUILD_S
    assert P._centered_build_estimate(4096, 2) > P.MAX_BUILD_S
    assert P._centered_build_estimate(4096, 1) <= P.MAX_BUILD_S
    x = np.random.default_rng(8).normal(size=(10, 4096))
    assert not P._reads_apart(PCA(n_components=2).fit(x))
    assert P._reads_apart(PCA(n_components=1).fit(x))
    est = FastICA(n_components=2, max_iter=3, random_state=0).fit(x)
    with pytest.raises(
        NotNative,
        match=r"an estimated 12 s build, past 7 s \(2 components over 4,096",
    ):
        to_native(_step(est), strict=True, allow_bound=True)


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

    # Centered, 96 components over 512 features built in 6.0 to 6.8 s, 48
    # over 1,024 in 8.1 to 8.7 s: past the cap, though a PCA of 48 over
    # 1,024 that reads `coalesce(x, NaN)`, in 5.1 to 6.7 s, is within it
    # (project.py, `_centered_build_estimate`).
    assert _centered_build_estimate(512, 96) <= MAX_BUILD_S
    assert _build_estimate(1024, 48) <= MAX_BUILD_S < _centered_build_estimate(1024, 48)
    x = np.random.default_rng(7).normal(size=(98, 1024))
    est = LinearDiscriminantAnalysis().fit(x, np.arange(98) % 49)
    with pytest.raises(
        NotNative,
        match=r"an estimated \d+ s build, past 7 s \(48 components over 1,024",
    ):
        to_native(_step_of(est, 48), strict=True, allow_bound=True)


# ------------------------------------------------------ SparseRandomProjection

MAX = float(np.finfo(np.float64).max)


def _sparse(n: int, k: int, density: float | str = "auto", dtype=np.float64):
    x = np.random.default_rng(n * 100 + k).normal(size=(4, n)).astype(dtype)
    return SparseRandomProjection(n_components=k, density=density, random_state=k).fit(
        x
    )


def _wide_rows(seed: int, n: int) -> np.ndarray:
    """Rows of magnitudes 1e-300 to 1e300, where the products and the
    partial sums round, underflow and overflow; then rows of signed zeros,
    subnormals and the largest doubles."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(200, n)) * 10.0 ** rng.uniform(-300, 300, size=(200, n))
    special = np.array(
        [0.0, -0.0, 5e-324, -5e-324, 1e-310, 0.5, -1.0, 1e300, MAX, -MAX]
    )
    return np.vstack(
        [x, special[(np.arange(40)[:, None] + np.arange(n)) % len(special)]]
    )


@pytest.mark.parametrize(
    "n, k, density, dtype",
    [
        (2, 3, "auto", np.float64),
        (8, 4, 1 / 3, np.float64),
        (50, 7, "auto", np.float64),
        (60, 6, "auto", np.float32),
        (300, 20, 0.3, np.float64),
    ],
)
def test_a_sparse_random_projection_is_bit_exact(n, k, density, dtype):
    # scipy sums each output field from 0.0 in the stored order, which the
    # entry follows (`_csr_sum`); a float32 fit's components widen exactly.
    est = _sparse(n, k, density, dtype)
    assert sparse.issparse(est.components_)
    step = _step(est)
    native = to_native(step, strict=True)
    rows = _rows(_wide_rows(n, n))
    assert check(step, native, rows, ulps=0) == rows.num_rows


@pytest.mark.parametrize("order, want", [([0, 1, 2], 0.0), ([0, 2, 1], 1.0)])
def test_a_sparse_random_projection_sums_in_its_stored_order(order, want):
    # 1e16 + 1 rounds to 1e16, so the twin's sum is 0 or 1 by the order of
    # the stored entries, unsorted as the fit stores them.
    est = _sparse(3, 1, 1 / 3)
    est.components_ = sparse.csr_matrix(([1.0, 1.0, 1.0], order, [0, 3]), shape=(1, 3))
    x = np.array([[1e16, 1.0, -1e16]])
    assert est.transform(x)[0, 0] == want
    step = _step(est)
    assert check(step, to_native(step, strict=True), _rows(x), ulps=0) == 1


def test_a_sparse_random_projection_of_density_1_is_held_to_the_matvec_bound():
    # Of density 1, the fit's components are dense, and the twin hands them
    # to BLAS, as GaussianRandomProjection's (`_project`).
    est = _sparse(6, 3, 1.0)
    assert not sparse.issparse(est.components_)
    step = _step(est)
    with pytest.raises(NotNative, match=r"within 9·eps·S \+ τ"):
        to_native(step, strict=True)
    x, means = _data(1, 6)
    rows = _rows(_near(2, x, means))
    assert check(step, to_native(step, strict=True, allow_bound=True), rows) == 60


def test_a_sparse_random_projection_the_entry_does_not_read_stays_python():
    est = _sparse(8, 3)
    est.components_ = est.components_.tocsc()
    with pytest.raises(NotNative, match="components_ in csc, not the csr"):
        to_native(_step(est), strict=True)
    est = _sparse(8, 3)
    est.components_.data[0] = np.inf
    with pytest.raises(NotNative, match="components_ not all finite"):
        to_native(_step(est), strict=True)


def _repeating(k: int, u: int) -> SparseRandomProjection:
    """A projection of 16 features whose k components repeat u sums of two
    entries in turn, each sum of its own values."""
    est = _sparse(16, k, 0.1)
    rng = np.random.default_rng(u)
    sums = [(rng.permutation(16)[:2], [1.0, -1.0 - i * 2.0**-20]) for i in range(u)]
    rows = [sums[j % u] for j in range(k)]
    est.components_ = sparse.csr_matrix(
        (
            np.concatenate([v for _, v in rows]),
            np.concatenate([i for i, _ in rows]),
            np.arange(0, 2 * k + 1, 2),
        ),
        shape=(k, 16),
    )
    return est


def test_a_sparse_random_projection_takes_a_fit_by_its_estimated_build():
    from sql_transform.native import project as P

    # Over 4,096 features at the "auto" density, 32 components built in
    # 6.1 s and 64 in 8.1 s. A struct read field by field builds slowly
    # where its fields repeat a sum: 1,024 components over 512 sums in
    # 7.2 s, and over 1,024 sums, none repeated, in 0.42 s
    # (`_sparse_build_estimate`).
    assert P._sparse_build_estimate(_sparse(4096, 32).components_) <= P.MAX_BUILD_S
    assert P._sparse_build_estimate(_sparse(4096, 64).components_) > P.MAX_BUILD_S
    assert P._repeated_sums(_repeating(1024, 512).components_) == 512
    assert P._sparse_build_estimate(_repeating(1024, 1024).components_) <= 1.0
    with pytest.raises(
        NotNative,
        match=r"an estimated 9 s build, past 7 s \(2,048 stored entries of 1,024"
        r" components over 16 features\)",
    ):
        to_native(_step(_repeating(1024, 512)), strict=True)


def test_a_sparse_random_projection_of_a_long_component_stays_python():
    # Each stored entry of a sum in scipy's order nests two levels of SQL,
    # and confit's parser refuses an expression nested past 1,000 levels.
    def longest(m):
        est = _sparse(500, 2)
        est.components_ = sparse.csr_matrix(
            (np.ones(m + 1), np.r_[np.arange(m)[::-1], 0], [0, m, m + 1]),
            shape=(2, 500),
        )
        return _step(est)

    step = longest(480)
    rows = _rows(_wide_rows(0, 500)[::6])
    assert check(step, to_native(step, strict=True), rows, ulps=0) == rows.num_rows
    with pytest.raises(NotNative, match="a component of 481 stored entries, past the"):
        to_native(longest(481), strict=True)


def test_the_kernel_probe_finds_scipys_sum_and_a_fused_one(monkeypatch):
    from sql_transform.native import project as P

    assert P.csr_sum_is_scipys()
    # A fused multiply-add answers the rounding error of a·x for a·x - a·x.
    a = 1.0 + 2.0**-30
    assert math.fma(a, -a, a * a) != 0.0
    monkeypatch.setattr(P, "csr_sum_is_scipys", lambda: False)
    with pytest.raises(NotNative, match="does not sum a CSR product as the entry"):
        to_native(_step(_sparse(8, 3)), strict=True)


# ------------------------------------------------------------- FactorAnalysis


def _factor_fit(
    seed: int, n: int, k: int | None, factors: int = 2, **kw
) -> tuple[FactorAnalysis, np.ndarray]:
    """A FactorAnalysis of k components over n features, fitted to rows of
    `factors` factors, features of scales 1e-2 to 1e2 with large means, and
    noise of 1e-8 to 1 of each feature's scale; and its fit rows. Past
    `factors` components, noise_variance_ meets sklearn's floor of 1e-12,
    and the twin's cov_z is ill-conditioned: of 6 components over 8
    features, seed 0, its condition number is about 3e13."""
    rng = np.random.default_rng(seed)
    scale = 10.0 ** rng.uniform(-2, 2, n)
    f = min(factors, n)
    x = rng.normal(size=(200, f)) @ rng.normal(size=(f, n))
    x = (x + rng.normal(size=(200, n)) * 10.0 ** rng.uniform(-8, 0, n)) * scale
    x = x + rng.normal(size=n) * scale * 10
    est = FactorAnalysis(n_components=k, random_state=seed, **kw).fit(x)
    return est, x


def _factor_rows(seed: int, est: FactorAnalysis, x: np.ndarray) -> np.ndarray:
    """Rows of magnitudes 1e-150 to 1e150, where no partial sum overflows;
    rows near the mean, where the products cancel; and the fit's rows."""
    rng = np.random.default_rng(seed)
    n = x.shape[1]
    wide = rng.choice([-1.0, 1.0], (60, n)) * 10.0 ** rng.uniform(-150, 150, (60, n))
    m = est.mean_
    spread = np.abs(m) * 10.0 ** rng.uniform(-15, -3, (40, 1))
    near = m + rng.normal(size=(40, n)) * spread
    return np.vstack([wide, near, x[:20]])


@pytest.mark.parametrize(
    "n, k, kw",
    [
        (1, 1, {}),
        (5, 2, {}),
        (8, 6, {}),
        (6, None, {"rotation": "varimax"}),
        (12, 3, {"svd_method": "lapack", "rotation": "quartimax"}),
    ],
)
def test_a_factor_analysis_parts_from_its_twin_within_its_error_scale(n, k, kw):
    est, x = _factor_fit(0, n, k, **kw)
    step = _step(est)
    native = to_native(step, strict=True, allow_bound=True)
    rows = _rows(_factor_rows(1, est, x))
    if n > 1:
        with pytest.raises(ParityError, match="ulps, bound 0"):
            check(step, native, rows, ulps=0)
    assert check(step, native, rows) == rows.num_rows


def test_a_factor_analysis_serves_only_on_request():
    est, _ = _factor_fit(0, 5, 2)
    step = _step(est)
    assert to_native(step) is step
    with pytest.raises(NotNative, match=r"FactorAnalysis is within 8·eps·S \+ τ"):
        to_native(step, strict=True)


def test_a_factor_analysis_is_held_to_the_twins_own_cov_z(monkeypatch):
    from sql_transform.native import project as P

    # The entry's cov_z (`_factor_inverse`) and the twin's part by about
    # 1e-16 at entries of 1: where cov_z is ill-conditioned, the twin's
    # product then parts from the entry's by much more than its rounding,
    # and only |t·ΔC| bounds it.
    est, x = _factor_fit(0, 8, 6)
    step = _step(est)
    native = to_native(step, strict=True, allow_bound=True)
    rows = _rows(_factor_rows(1, est, x))
    assert check(step, native, rows) == rows.num_rows
    f = P._factor(est)
    assert np.linalg.cond(np.eye(6) + f.v @ est.components_.T) > 1e12
    alike = f._replace(dc=np.zeros_like(f.dc), abs_dc=np.zeros_like(f.abs_dc))
    monkeypatch.setattr(P, "_factor", lambda est: alike)
    with pytest.raises(ParityError, match="past its bound"):
        check(step, native, rows)


def test_a_factor_analysis_answers_the_twins_zero():
    # At the mean each product is a zero: the twin's sums of more than one
    # product answer 0.0, and the entry adds 0.0 to its sums. At one
    # feature and one output field, the twin's `np.dot`s of 1×1 arrays
    # answer the products themselves, as the entry does: -0.0 where they
    # are -0.0.
    for n, k in [(5, 2), (4, 1), (1, 1)]:
        est, _ = _factor_fit(2, n, k)
        step = _step(est)
        native = to_native(step, strict=True, allow_bound=True)
        rows = _rows(np.repeat(est.mean_[None], 3, axis=0))
        assert check(step, native, rows, ulps=0) == 3
    est.mean_ = np.zeros(1)
    signs = [[0.0], [-0.0]]
    assert {math.copysign(1, v) for v in est.transform(np.array(signs)[:1]).ravel()}
    step = _step(est)
    native = to_native(step, strict=True, allow_bound=True)
    assert check(step, native, _rows(np.array(signs)), ulps=0) == 2


def test_a_factor_analysis_the_entry_does_not_read_stays_python():
    est, _ = _factor_fit(3, 5, 2)
    est.noise_variance_ = est.noise_variance_.copy()
    est.noise_variance_[1] = 0.0
    with pytest.raises(NotNative, match="or a noise variance not positive"):
        to_native(_step(est), strict=True, allow_bound=True)
    est, _ = _factor_fit(3, 5, 2)
    est.components_ = est.components_.copy()
    est.components_[0, 0] = np.nan
    with pytest.raises(NotNative, match="component or noise variance not all"):
        to_native(_step(est), strict=True, allow_bound=True)
    # A subnormal noise variance: Wpsi is not finite, nor is either cov_z.
    est, _ = _factor_fit(3, 5, 2)
    est.noise_variance_ = np.full(5, 5e-324)
    with pytest.raises(NotNative, match="has no finite inverse"):
        to_native(_step(est), strict=True, allow_bound=True)


def test_the_entrys_cov_z_is_the_same_at_any_thread_count():
    from sql_transform.native import project as P

    # Without BLAS or LAPACK, the entry's inverse does not depend on the
    # thread count, as the twin's can from about 68 components; it is
    # exactly symmetric, and near the inverse of I + v·w^T.
    rng = np.random.default_rng(5)
    w = rng.normal(size=(100, 120))
    v = w / 10.0 ** rng.uniform(-3, 0, 120)
    c = P._factor_inverse(v, w)
    with threadpool_limits(limits=1):
        assert np.array_equal(P._factor_inverse(v, w).view(np.int64), c.view(np.int64))
    assert np.array_equal(c, c.T)
    exact = np.linalg.inv(np.eye(100) + v @ w.T)
    assert np.max(np.abs(c - exact)) < 1e-12 * np.max(np.abs(exact))
    # A pivot of 0 or less: I + v·w^T is not positive definite.
    assert P._factor_inverse(-2 * w[:1] / np.linalg.norm(w[0]) ** 2, w[:1]) is None


def test_the_error_scale_reads_the_twins_cov_z_at_the_thread_count_of_the_call():
    from sql_transform.native import project as P

    est, _ = _factor_fit(4, 6, 3)
    with threadpool_limits(limits=1):
        one = P._factor(est)
        assert P._factor(est) is one
    with threadpool_limits(limits=2):
        two = P._factor(est)
    assert two is not one
    assert np.array_equal(two.c, one.c)


def test_the_kernel_probe_finds_the_twins_cov_z_repeat(monkeypatch):
    from sql_transform.native import project as P

    assert P.cov_z_repeats()
    real = P._cov_z

    def by_address(w, psi):
        v, c = real(w, psi)
        return v, c * (1 + (w.ctypes.data % 64 != 0) * 2.0**-52)

    monkeypatch.setattr(P, "_cov_z", by_address)
    assert not P.cov_z_repeats.__wrapped__()
    monkeypatch.setattr(P, "cov_z_repeats", lambda: False)
    with pytest.raises(NotNative, match=r"answers cov_z apart for the same arrays"):
        to_native(_step(_factor_fit(0, 5, 2)[0]), strict=True, allow_bound=True)
