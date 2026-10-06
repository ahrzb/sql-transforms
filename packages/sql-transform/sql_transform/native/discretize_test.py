"""KBinsDiscretizer(dtype=np.float32) beyond the catalog's generated fits:
the moved cutpoints against numpy's own rounding, and the entry against
its twin at, and on both sides of, every edge and every cutpoint, over the
edges where float32 rounding decides (ties, equal edges, ±0, subnormals,
FLT_MAX and past it)."""

from __future__ import annotations

import warnings

import numpy as np
import pyarrow as pa
import pytest
from sklearn.preprocessing import KBinsDiscretizer

from sql_transform._udf import PythonTransform
from sql_transform.native import check, to_native
from sql_transform.native.discretize import _f32_cut

F32 = np.float32
FLT_MAX = float(np.finfo(F32).max)
TINY = 2.0**-149  # the smallest float32 subnormal
# The largest double that rounds to a finite float32: below the midpoint
# of FLT_MAX and 2**128, which ties to even, away from FLT_MAX's odd
# mantissa, onto infinity.
LAST_FINITE = float(np.nextafter((FLT_MAX + 2.0**128) / 2, 0.0))


def _rounds(x: float) -> float:
    """float32(x) as numpy rounds it, widened back: what the twin bins."""
    with np.errstate(over="ignore"):
        return float(np.array([x]).astype(F32)[0])


def _near(v: float, k: int = 3) -> list[float]:
    """`v` and the `k` doubles on either side of it."""
    out, lo, hi = [v], v, v
    for _ in range(k):
        lo, hi = float(np.nextafter(lo, -np.inf)), float(np.nextafter(hi, np.inf))
        out += [lo, hi]
    return out


def _probes(e: float) -> list[float]:
    """Doubles around edge `e`: at and beside the edge, its cutpoint, and
    the float32 neighbours of the edge with the midpoints between them."""
    c = float(_f32_cut(np.array([e]))[0])
    xs = _near(e) + _near(c)
    with np.errstate(over="ignore"):  # FLT_MAX's neighbour is infinity
        return [x for x in xs + _f32_neighbours(e) if not np.isnan(x)]


def _f32_neighbours(e: float) -> list[float]:
    """The float32s around `e` and the midpoints between them, with the
    doubles beside each."""
    xs: list[float] = []
    f = F32(e)
    for g in (np.nextafter(f, F32(-np.inf)), f, np.nextafter(f, F32(np.inf))):
        if np.isfinite(g):
            xs += _near(float(g), 1)
            for h in (np.nextafter(g, F32(-np.inf)), np.nextafter(g, F32(np.inf))):
                if np.isfinite(h):
                    xs += _near((float(g) + float(h)) / 2, 1)
    return xs


# Edges where float32 rounding decides: float32 values with even and odd
# mantissas (their lower midpoint ties to the even neighbour), the
# midpoints themselves, a double between two float32s, ±0, the subnormals
# and the smallest normal, FLT_MAX, past it, and arbitrary doubles.
EDGES = [
    1.0,
    float(np.nextafter(F32(1.0), F32(2.0))),
    1.0 + 2.0**-24,
    1.0 + 3 * 2.0**-24,
    1.0 + 2.0**-30,
    0.0,
    -0.0,
    TINY,
    -TINY,
    TINY / 2,
    3 * TINY / 2,
    2.0**-150 + 2.0**-170,
    2.0**-126,
    float(np.nextafter(F32(2.0**-126), F32(0.0))),
    FLT_MAX,
    -FLT_MAX,
    LAST_FINITE,
    1e39,
    -1e39,
    1e300,
    0.1,
    -2.5e-40,
    123456.789,
]


@pytest.mark.parametrize("e", EDGES)
def test_a_cutpoint_answers_numpys_rounding(e):
    c = float(_f32_cut(np.array([e]))[0])
    for x in _probes(e):
        assert (x < c) == (_rounds(x) < e), (e, c, x)


def test_cutpoints_answer_numpys_rounding_on_draws():
    rng = np.random.default_rng(0)
    e = np.concatenate(
        [
            rng.normal(size=3000) * 10.0 ** rng.integers(-45, 40, 3000),
            # float32 values and the doubles next to them
            rng.normal(size=1000).astype(F32).astype(np.float64),
            np.nextafter(rng.normal(size=1000).astype(F32).astype(np.float64), 0.0),
        ]
    )
    c = _f32_cut(e)
    assert (np.diff(c[np.argsort(e)]) >= 0).all(), "the cut is monotone"
    for ei, ci in zip(e, c, strict=True):
        for x in _near(float(ci), 2):
            assert (x < ci) == (_rounds(x) < ei), (ei, ci, x)


def test_a_tie_rounds_to_even():
    # Below 1.0 the float32s are 2**-24 apart; the midpoint of 1.0 and the
    # float32 under it ties to 1.0 (even), so it reaches the bin at 1.0.
    up = float(np.nextafter(F32(1.0), F32(2.0)))  # odd mantissa
    mid_at_1 = 1.0 - 2.0**-25
    assert _rounds(mid_at_1) == 1.0
    assert float(_f32_cut(np.array([1.0]))[0]) == mid_at_1
    # The midpoint under `up` ties down to 1.0, so it stays below `up`.
    mid_at_up = 1.0 + 2.0**-24
    assert _rounds(mid_at_up) == 1.0
    assert float(_f32_cut(np.array([up]))[0]) == float(np.nextafter(mid_at_up, 2.0))


def test_signed_zeros_and_subnormals_move_as_numpy_rounds():
    # An edge at ±0 is reached by every x that rounds to ±0: down to
    # -2**-150, which ties to -0.0.
    for z in (0.0, -0.0):
        assert float(_f32_cut(np.array([z]))[0]) == -(2.0**-150)
    assert _rounds(-(2.0**-150)) == 0.0
    # An edge at the smallest subnormal is reached from its lower
    # midpoint, which ties to 0 (even): one double past it.
    assert float(_f32_cut(np.array([TINY]))[0]) == float(np.nextafter(2.0**-150, 1.0))


def test_past_float32s_range():
    # An edge past FLT_MAX is never reached by a finite float32: its
    # cutpoint is past the largest double that rounds to one.
    for e in (1e39, 1e300, LAST_FINITE):
        c = float(_f32_cut(np.array([e]))[0])
        assert LAST_FINITE < c
        assert np.isinf(_rounds(c))
    # An infinite edge stays: it differs only where x rounds to infinity.
    assert list(_f32_cut(np.array([np.inf, -np.inf]))) == [np.inf, -np.inf]


# ------------------------------------------------------------ against the twin


def _fitted(edges: list[list[float]], encode: str) -> KBinsDiscretizer:
    """A float32 discretizer whose features have these inner edges: fitted
    with as many bins on spread data, then given the edges, which are all
    `transform` reads (the one-hot encoder's categories are the bin
    numbers). A shorter feature is padded with edges at +inf, whose bins
    stay empty."""
    n = max(len(e) for e in edges) + 1
    X = np.tile(np.arange(n + 1, dtype=np.float64)[:, None], (1, len(edges)))
    est = KBinsDiscretizer(
        n_bins=n, encode=encode, strategy="uniform", dtype=np.float32
    ).fit(X)
    assert (est.n_bins_ == n).all()
    est.bin_edges_ = np.array(
        [
            np.array([-np.inf, *sorted(e), *[np.inf] * (n - 1 - len(e)), np.inf])
            for e in edges
        ],
        dtype=object,
    )
    return est


def _step(est: KBinsDiscretizer, width: int) -> PythonTransform:
    n = est.n_features_in_
    takes = pa.schema([(f"x{i}", pa.float64()) for i in range(n)])
    returns = (
        pa.float64()
        if width == 1
        else pa.struct([(f"f{j}", pa.float64()) for j in range(width)])
    )
    return PythonTransform("tf", {0: est}, takes, returns)


def _served(step: PythonTransform, xs: list[list[float]]) -> int:
    """`check` over rows of `xs` (a column per feature, padded by
    repeating), the number of rows compared."""
    native = to_native(step, strict=True)
    n = max(len(c) for c in xs)
    table = {"__iid": pa.array([0] * n, pa.int64())}
    for i, c in enumerate(xs):
        table[f"x{i}"] = pa.array(np.resize(np.array(c, dtype=np.float64), n))
    with warnings.catch_warnings():  # the overflow the twin raises on
        warnings.simplefilter("ignore")
        return check(step, native, pa.table(table))


GROUPS = {
    "ties": [1.0, float(np.nextafter(F32(1.0), F32(2.0))), 1.0 + 2.0**-24],
    "equal": [0.5, 0.5, 0.5, 1.0 + 2.0**-30, 1.0 + 2.0**-29, 2.0],
    "zeros": [-0.0, 0.0, TINY],
    "subnormal": [-TINY, TINY / 2, 3 * TINY / 2, 2.0**-126],
    "range": [-FLT_MAX, 3e38, FLT_MAX, LAST_FINITE, 1e39],
}


@pytest.mark.parametrize("encode", ["ordinal", "onehot-dense"])
@pytest.mark.parametrize("group", list(GROUPS))
def test_the_entry_answers_its_twin_at_every_cutpoint(group, encode):
    edges = GROUPS[group]
    est = _fitted([edges, list(reversed(edges))[1:]], encode)
    width = np.asarray(est.transform(np.zeros((1, 2)))).shape[1]
    xs = [x for e in edges for x in _probes(e)]
    xs += [0.0, -0.0, 1e300, -1e300, np.inf, -np.inf]
    compared = _served(_step(est, width), [xs, list(reversed(xs))])
    assert compared > len(xs) // 2


def test_the_largest_finite_rounding_is_answered_as_the_twin():
    est = _fitted([[3e38, FLT_MAX, 1e39]], "ordinal")
    assert est.transform([[LAST_FINITE]])[0, 0] == 2.0
    with warnings.catch_warnings(), pytest.raises(ValueError, match="too large"):
        warnings.simplefilter("ignore")  # numpy's overflow, before the raise
        est.transform([[float(np.nextafter(LAST_FINITE, np.inf))]])
    assert _served(_step(est, 1), [[LAST_FINITE, -LAST_FINITE, FLT_MAX]]) == 3


def test_nan_raises_in_the_twin_and_the_entry_answers():
    est = _fitted([[0.0, 1.0]], "ordinal")
    with pytest.raises(ValueError, match="NaN"):
        est.transform([[np.nan]])
    # The NaN row (a NULL too) is not compared; the others are.
    assert _served(_step(est, 1), [[np.nan, 0.5, 2.0]]) == 2


CONFIGS = [
    {"strategy": s, "encode": c}
    for s in ("uniform", "quantile", "kmeans")
    for c in ("ordinal", "onehot-dense")
]


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("cfg", CONFIGS, ids=lambda c: "-".join(c.values()))
def test_fitted_edges_are_answered_at_their_cutpoints(cfg, seed):
    rng = np.random.default_rng(seed)
    X = np.column_stack(
        [
            rng.normal(rng.uniform(-100, 100), rng.uniform(0.01, 50), 40),
            rng.integers(-3, 4, 40).astype(float),
            rng.exponential(1e-3, 40),
        ]
    )
    with warnings.catch_warnings():  # narrow bins dropped
        warnings.simplefilter("ignore")
        est = KBinsDiscretizer(n_bins=5, dtype=np.float32, **cfg).fit(X)
    width = np.asarray(est.transform(X[:1])).shape[1]
    cols = [
        [x for e in est.bin_edges_[j][1:-1] for x in _probes(float(e))]
        for j in range(3)
    ]
    assert _served(_step(est, width), cols) == max(len(c) for c in cols)
