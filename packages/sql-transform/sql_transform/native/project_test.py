"""Linear projections within the matvec parity bound (project.py): the
twin's BLAS sum and the entry's balanced tree part by rounding alone, on
rows near the mean where the two products cancel."""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest
from sklearn.decomposition import PCA, IncrementalPCA, TruncatedSVD
from sklearn.random_projection import GaussianRandomProjection

from sql_transform._udf import PythonTransform
from sql_transform.native import (
    NotNative,
    ParityError,
    check,
    explain_native,
    to_native,
)


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
    n, k = est.n_features_in_, len(est.components_)
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


def test_a_float32_fit_stays_python():
    x, _ = _data(5, 4)
    est = PCA(n_components=2).fit(x.astype(np.float32))
    with pytest.raises(NotNative, match="projects its mean in float32"):
        to_native(_step(est), strict=True, allow_bound=True)
