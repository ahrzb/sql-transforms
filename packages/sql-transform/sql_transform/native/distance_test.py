"""Distances to fitted centres within their error scale on the squares
(distance.py): the twin's BLAS sum and the entry's balanced tree part by
rounding alone, which near a centre is half the digits of the distance."""

from __future__ import annotations

import dataclasses

import numpy as np
import pyarrow as pa
import pytest
from sklearn.cluster import Birch, BisectingKMeans, KMeans, MiniBatchKMeans
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sql_transform._udf import PythonTransform
from sql_transform.native import (
    NotNative,
    ParityError,
    _registry,
    check,
    explain_native,
    to_native,
)
from sql_transform.native import distance as D

# A fit may stop at `max_iter`, or find fewer distinct points than its
# clusters: its transform does the same arithmetic either way.
pytestmark = pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")


def _blobs(seed: int, n: int) -> np.ndarray:
    """Fit rows of `n` correlated features of scales 1e-2 to 1e4 and large
    means, in four blobs."""
    rng = np.random.default_rng(seed)
    scales = 10.0 ** rng.uniform(-2, 4, size=n)
    means = rng.uniform(-100, 100, size=n) * scales
    centres = rng.normal(size=(4, n)) * 5
    z = centres[rng.integers(0, 4, size=200)] + rng.normal(size=(200, n))
    return z @ rng.normal(size=(n, n)) * scales + means


def _near(seed: int, est) -> np.ndarray:
    """Each centre, then rows within 1e-9 to 1e-3 of its scale of one,
    where `-2·(x @ c)` and `‖x‖²` cancel to the square's rounding."""
    rng = np.random.default_rng(seed)
    c = D._centres(est)
    k, n = c.shape
    scale = np.abs(c).max(axis=0) + 1.0
    off = rng.normal(size=(40, n)) * scale * 10.0 ** rng.uniform(-9, -3, size=(40, 1))
    return np.vstack([c, c[rng.integers(0, k, size=40)] + off])


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
    lambda: KMeans(n_clusters=4, n_init=1, random_state=0),
    lambda: MiniBatchKMeans(n_clusters=3, n_init=1, random_state=0),
    lambda: BisectingKMeans(n_clusters=4, random_state=0),
    lambda: Birch(threshold=1e3, n_clusters=None),
]


@pytest.mark.parametrize("make", MAKES, ids=lambda m: type(m()).__name__)
def test_a_distance_parts_from_its_twin_within_its_error_scale(make):
    est = make().fit(_blobs(0, 12))
    step = _step(est)
    native = to_native(step, strict=True, allow_bound=True)
    rows = _rows(_near(1, est))
    with pytest.raises(ParityError, match="ulps, bound 0"):
        check(step, native, rows, ulps=0)
    assert check(step, native, rows) == rows.num_rows


def test_a_distance_near_a_centre_is_compared_by_its_square(monkeypatch):
    # Near a centre the squares cancel to their rounding: the two distances
    # are each about sqrt(eps·S), any number of ulps and of eps·S apart,
    # while their squares are within (n + 5)·eps·S.
    est = KMeans(n_clusters=4, n_init=1, random_state=0).fit(_blobs(2, 8))
    step = _step(est)
    native = to_native(step, strict=True, allow_bound=True)
    rows = _rows(_near(3, est))
    assert check(step, native, rows) == rows.num_rows
    entry = _registry._CATALOG[KMeans]
    plain = dataclasses.replace(entry.scale, g=None, on="")
    monkeypatch.setitem(
        _registry._CATALOG, KMeans, dataclasses.replace(entry, scale=plain)
    )
    with pytest.raises(ParityError, match="past its bound"):
        check(step, native, rows)


def test_whole_numbers_part_from_the_twin_by_nothing():
    # Products and sums of small whole numbers are exact in any order, so
    # the entry answers the twin's bits: only rounding parts them.
    est = KMeans(n_clusters=3, n_init=1, random_state=0).fit(_blobs(4, 6))
    est.cluster_centers_ = np.round(est.cluster_centers_ / 100.0)
    step = _step(est)
    native = to_native(step, strict=True, allow_bound=True)
    x = np.random.default_rng(5).integers(-1000, 1000, size=(64, 6)).astype(float)
    rows = _rows(np.vstack([est.cluster_centers_, x]))
    assert check(step, native, rows, ulps=0) == rows.num_rows


def test_a_changed_centre_breaches_the_error_scale():
    # A structural change moves a square by about S/n, far past
    # (n + 5)·eps·S: here one coordinate of a centre, by 1e-9 of itself.
    est = KMeans(n_clusters=4, n_init=1, random_state=0).fit(_blobs(6, 8))
    step = _step(est)
    native = to_native(step, strict=True, allow_bound=True)
    rows = _rows(_near(7, est))
    est.cluster_centers_ = est.cluster_centers_.copy()
    est.cluster_centers_[0, np.argmax(np.abs(est.cluster_centers_[0]))] *= 1 + 1e-9
    with pytest.raises(
        ParityError, match=r"past its bound K\*eps\*S \+ tau = \S+ on the squares"
    ):
        check(step, native, rows)


def test_a_distance_serves_only_on_request():
    est = KMeans(n_clusters=2, n_init=1, random_state=0).fit(_blobs(8, 4))
    step = _step(est)
    assert to_native(step) is step
    with pytest.raises(
        NotNative, match=r"KMeans is within 9·eps·S \+ τ on the squares \(S its"
    ):
        to_native(step, strict=True)
    assert explain_native(step, allow_bound=True).endswith(
        "-> SqlFunction, within 9·eps·S + τ on the squares (S its error scale)"
    )


@pytest.mark.parametrize("make", MAKES, ids=lambda m: type(m()).__name__)
def test_a_float32_fit_serves(make):
    # The twin widens float32 centres to float64 before it reads them, as
    # the entry does.
    est = make().fit(_blobs(9, 6).astype(np.float32))
    fitted = est.subcluster_centers_ if isinstance(est, Birch) else est.cluster_centers_
    assert fitted.dtype == np.float32
    step = _step(est)
    native = to_native(step, strict=True, allow_bound=True)
    rows = _rows(_near(10, est))
    assert check(step, native, rows) == rows.num_rows


def test_a_distance_over_1024_features_serves():
    # numpy's order for ‖x‖² is a chain 512 additions deep here, which
    # confit does not build; the entry's balanced tree is 10 deep.
    est = KMeans(n_clusters=1, n_init=1, max_iter=1, random_state=0)
    est.fit(np.random.default_rng(11).normal(size=(64, 1024)))
    step = _step(est)
    native = to_native(step, strict=True, allow_bound=True)
    rows = _rows(_near(12, est)[:8])
    assert check(step, native, rows) == rows.num_rows


def test_a_distance_in_a_composition_stays_python():
    pipe = make_pipeline(StandardScaler(), KMeans(n_clusters=2, n_init=1))
    pipe.fit(_blobs(13, 4))
    step = dataclasses.replace(_step(pipe[-1]), instances={0: pipe})
    with pytest.raises(
        NotNative,
        match="Pipeline step 'kmeans': KMeans compares through a map, which a"
        " composition does not pass on",
    ):
        to_native(step, strict=True, allow_bound=True)


def test_a_centre_not_all_finite_stays_python():
    est = KMeans(n_clusters=2, n_init=1, random_state=0).fit(_blobs(14, 3))
    step = _step(est)
    est.cluster_centers_ = est.cluster_centers_.copy()
    est.cluster_centers_[1, 0] = np.inf
    with pytest.raises(NotNative, match="a fitted centre not all finite"):
        to_native(step, strict=True, allow_bound=True)


def test_the_build_estimate_takes_what_builds_within_the_cap():
    # Measured (`distance._build_estimate`): a struct of 800 centres over 2
    # features built in 4.9 s, of 1,000 in 8.6 s and of 2,000 in 73 s; a
    # list of 4 centres over 2,048 features in 5.0 s and of 16 in 10.7 s;
    # one centre over 4,096 features in 14.8 s.
    assert D._build_estimate(2, 800) <= D.MAX_BUILD_S
    assert D._build_estimate(2048, 4) <= D.MAX_BUILD_S
    for n, k in [(2, 1000), (2, 2000), (2048, 16), (4096, 1)]:
        assert D._build_estimate(n, k) > D.MAX_BUILD_S


def test_a_distance_takes_a_fit_by_its_estimated_build():
    est = KMeans(n_clusters=64, n_init=1, max_iter=1, random_state=0)
    est.fit(np.random.default_rng(15).normal(size=(80, 1024)))
    with pytest.raises(
        NotNative,
        match=r"an estimated \d+ s build, past 7 s \(64 centres over 1,024",
    ):
        to_native(_step(est), strict=True, allow_bound=True)
