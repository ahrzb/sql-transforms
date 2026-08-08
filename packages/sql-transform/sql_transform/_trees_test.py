"""The gate: confit's `tree_predict` vs sklearn's own `predict`, bit-exact.

Not "close" — `==` on the raw doubles. Every tolerance in a serving path is
a place where a refactor silently changes answers, so the parity claim is
made at the only threshold that cannot rot.

`n_jobs=1` throughout. That is the reference serving configuration, not a
workaround: at our latency budget per-request parallelism is not on the
table, and a forest run with `n_jobs != 1` does not even reproduce ITSELF
run to run (measured 2026-08-07: up to 1187 of 2000 rows differ, 19 ULP).
"""

from __future__ import annotations

import numpy as np
import pytest
from confit import DuckDBInferFn
from pydantic import create_model
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.tree import DecisionTreeRegressor

from sql_transform import PythonTransform, TreePackError, pack_trees
from sql_transform._trees import _f32_grid_threshold

FEATURES = ["a", "b", "c", "d"]
ROW = create_model(
    "Row",
    id=(int, ...),
    **dict.fromkeys(FEATURES, (float | None, None)),
)
SQL = (
    "SELECT tree_predict('m', id, struct_pack("
    + ", ".join(f"{f} := {f}" for f in FEATURES)
    + ")) AS p FROM __THIS__"
)


def fit(kind, seed, n=120):
    rng = np.random.RandomState(seed)
    X = rng.rand(n, len(FEATURES)) * 4 - 2
    y = X[:, 0] * 3 - X[:, 1] ** 2 + rng.rand(n) * 0.3
    return kind.fit(X, y)


def score(entry, X, backend=None):
    fn = DuckDBInferFn(
        SQL,
        row_tables={"__THIS__": ROW},
        static_tables={},
        models={"m": entry},
    )
    if backend is not None:
        assert fn.backend == backend
    rows = [
        ROW(id=0, **{f: float(v) for f, v in zip(FEATURES, x, strict=True)}) for x in X
    ]
    return np.array([r.p for r in fn.infer({"__THIS__": rows})])


ESTIMATORS = [
    DecisionTreeRegressor(max_depth=6, random_state=1),
    RandomForestRegressor(n_estimators=12, max_depth=5, random_state=2, n_jobs=1),
    RandomForestRegressor(n_estimators=40, random_state=3, n_jobs=1),  # unbounded depth
    ExtraTreesRegressor(n_estimators=15, max_depth=4, random_state=4, n_jobs=1),
    GradientBoostingRegressor(n_estimators=25, max_depth=3, random_state=5),
    GradientBoostingRegressor(
        n_estimators=10, learning_rate=0.37, max_depth=2, random_state=6
    ),
    GradientBoostingRegressor(n_estimators=8, init="zero", random_state=7),
]


@pytest.mark.parametrize("est", ESTIMATORS, ids=lambda e: type(e).__name__)
@pytest.mark.parametrize("seed", [11, 12])
def test_matches_sklearn_bit_exactly(est, seed):
    from sklearn.base import clone

    fitted = fit(clone(est), seed)
    rng = np.random.RandomState(seed + 900)
    X = rng.rand(400, len(FEATURES)) * 5 - 2.5
    want = fitted.predict(X)
    got = score(pack_trees([fitted], FEATURES), X)
    bad = np.flatnonzero(got != want)
    assert bad.size == 0, (
        f"{bad.size}/{len(X)} rows differ; first at {bad[:3]}: "
        f"{got[bad[:3]]!r} vs {want[bad[:3]]!r}"
    )


@pytest.mark.parametrize("backend", ["cranelift", "interpreter"])
def test_both_backends_match_sklearn(backend, monkeypatch):
    """The cranelift compile error is discarded and the fallback is silent, so
    the engine is asserted, not assumed."""
    if backend == "interpreter":
        monkeypatch.setenv("SPECIALIZER_FORCE_INTERP", "1")
    else:
        monkeypatch.delenv("SPECIALIZER_FORCE_INTERP", raising=False)
    fitted = fit(RandomForestRegressor(n_estimators=20, random_state=8, n_jobs=1), 21)
    X = np.random.RandomState(22).rand(300, len(FEATURES)) * 5 - 2.5
    got = score(pack_trees([fitted], FEATURES), X, backend=backend)
    assert np.array_equal(got, fitted.predict(X))


def test_nan_features_match_sklearn():
    """sklearn routes NaN by each node's `missing_go_to_left`; so do we. The
    trees must actually have been FIT with missing values or every
    `missing_go_to_left` is the same default and this proves nothing."""
    rng = np.random.RandomState(31)
    X = rng.rand(300, len(FEATURES)) * 4 - 2
    X[rng.rand(*X.shape) < 0.25] = np.nan
    y = np.nan_to_num(X[:, 0]) * 3 - np.nan_to_num(X[:, 1]) ** 2
    fitted = RandomForestRegressor(
        n_estimators=15, max_depth=6, random_state=32, n_jobs=1
    ).fit(X, y)
    nodes = pack_trees([fitted], FEATURES)["nodes"]
    assert len(set(nodes.column("missing_left").to_pylist())) == 2, (
        "every node has the same missing direction — the fit saw no NaNs"
    )
    Xq = rng.rand(400, len(FEATURES)) * 5 - 2.5
    Xq[rng.rand(*Xq.shape) < 0.3] = np.nan
    got = score(pack_trees([fitted], FEATURES), Xq)
    assert np.array_equal(got, fitted.predict(Xq))


def test_null_and_nan_are_the_same_input():
    """SQL NULL reaches the kernel as NaN — the two spellings of missing must
    not disagree."""
    rng = np.random.RandomState(41)
    X = rng.rand(200, len(FEATURES)) * 4 - 2
    X[rng.rand(*X.shape) < 0.2] = np.nan
    fitted = RandomForestRegressor(
        n_estimators=10, max_depth=5, random_state=42, n_jobs=1
    ).fit(X, np.nan_to_num(X[:, 0]))
    entry = pack_trees([fitted], FEATURES)
    fn = DuckDBInferFn(
        SQL, row_tables={"__THIS__": ROW}, static_tables={}, models={"m": entry}
    )
    Xq = np.array([[np.nan, 0.5, np.nan, -1.0], [1.0, np.nan, 2.0, np.nan]])
    as_null = [
        ROW(
            id=0,
            **{
                f: (None if np.isnan(v) else float(v))
                for f, v in zip(FEATURES, x, strict=True)
            },
        )
        for x in Xq
    ]
    as_nan = [
        ROW(id=0, **{f: float(v) for f, v in zip(FEATURES, x, strict=True)}) for x in Xq
    ]
    null_out = [r.p for r in fn.infer({"__THIS__": as_null})]
    nan_out = [r.p for r in fn.infer({"__THIS__": as_nan})]
    assert null_out == nan_out == list(fitted.predict(Xq))


def test_per_group_models_score_by_id():
    """The serving shape: one fitted model per group, the params table's id
    column selecting between them."""
    fits = [
        fit(
            RandomForestRegressor(
                n_estimators=6, max_depth=4, random_state=s, n_jobs=1
            ),
            s,
        )
        for s in (51, 52, 53)
    ]
    entry = pack_trees(fits, FEATURES)
    X = np.random.RandomState(54).rand(90, len(FEATURES)) * 5 - 2.5
    fn = DuckDBInferFn(
        SQL, row_tables={"__THIS__": ROW}, static_tables={}, models={"m": entry}
    )
    ids = np.arange(len(X)) % 3
    rows = [
        ROW(id=int(g), **{f: float(v) for f, v in zip(FEATURES, x, strict=True)})
        for g, x in zip(ids, X, strict=True)
    ]
    got = np.array([r.p for r in fn.infer({"__THIS__": rows})])
    want = np.array([fits[g].predict(x[None])[0] for g, x in zip(ids, X, strict=True)])
    assert np.array_equal(got, want)


# --------------------------------------------- gate 2: ecall vs predict --
#
# The same estimator, the same rows, lowered two ways inside confit: once
# through the Python trampoline (`ecall` over a PythonTransform — the path
# production uses today), once through the native kernel. No DuckDB, no
# sklearn call on the confit side of the second one.
#
# This is what the swap-in has to be safe on: replacing the trampoline with
# the kernel must not move a single bit, or every model that was fit against
# the old serving path silently shifts.


class _AsTransform:
    """A regressor seen as a transformer — `PythonTransform` calls
    `transform`, a regressor offers `predict`. Counts its calls: if this
    never fires, the "ecall" side did not go through the trampoline and the
    comparison below is two runs of the same path."""

    def __init__(self, est):
        self.est = est
        self.calls = 0

    def transform(self, X):
        self.calls += 1
        return np.asarray(self.est.predict(np.asarray(X, dtype=float))).reshape(-1, 1)


ECALL_SQL = "SELECT score(id, " + ", ".join(FEATURES) + ") AS p FROM __THIS__"


def _both_paths(ests, rows, row_model=None):
    """(ecall answers, predict answers) for the same estimators and rows."""
    row_model = row_model or ROW
    shims = {i: _AsTransform(e) for i, e in enumerate(ests)}
    udf = PythonTransform(
        name="score",
        instances=shims,
        takes=("f64",) * len(FEATURES),
        returns=("f64",),
    )
    ecall_fn = DuckDBInferFn(
        ECALL_SQL, row_tables={"__THIS__": row_model}, static_tables={}, udfs=[udf]
    )
    predict_fn = DuckDBInferFn(
        SQL,
        row_tables={"__THIS__": row_model},
        static_tables={},
        models={"m": pack_trees(ests, FEATURES)},
    )
    ecall = [r.p for r in ecall_fn.infer({"__THIS__": rows})]
    predict = [r.p for r in predict_fn.infer({"__THIS__": rows})]
    scored = sum(1 for v in ecall if v is not None)
    assert sum(s.calls for s in shims.values()) == scored, (
        "the ecall side did not run the Python trampoline once per scored row"
    )
    return ecall, predict


def _rows(X, ids):
    return [
        ROW(
            id=int(g),
            **{
                f: (None if np.isnan(v) else float(v))
                for f, v in zip(FEATURES, x, strict=True)
            },
        )
        for g, x in zip(ids, X, strict=True)
    ]


def test_ecall_and_predict_agree_bit_for_bit():
    ests = [
        fit(
            RandomForestRegressor(
                n_estimators=9, max_depth=5, random_state=s, n_jobs=1
            ),
            s,
        )
        for s in (81, 82)
    ]
    X = np.random.RandomState(83).rand(200, len(FEATURES)) * 5 - 2.5
    ecall, predict = _both_paths(ests, _rows(X, np.arange(len(X)) % 2))
    assert ecall == predict


def test_ecall_and_predict_agree_on_missing_values():
    """The paths reach the estimator differently — the trampoline maps NULL to
    NaN in `_as_feature`, the kernel maps it in the lowering. They must land
    on the same number."""
    rng = np.random.RandomState(84)
    Xf = rng.rand(150, len(FEATURES)) * 4 - 2
    Xf[rng.rand(*Xf.shape) < 0.25] = np.nan
    est = RandomForestRegressor(
        n_estimators=12, max_depth=5, random_state=85, n_jobs=1
    ).fit(Xf, np.nan_to_num(Xf[:, 0]))
    X = rng.rand(200, len(FEATURES)) * 5 - 2.5
    X[rng.rand(*X.shape) < 0.3] = np.nan
    ecall, predict = _both_paths([est], _rows(X, np.zeros(len(X), int)))
    assert ecall == predict


def test_ecall_and_predict_agree_on_a_boosted_model():
    """The base term and the learning-rate scaling exist only on the kernel
    side — the trampoline gets them from sklearn. Most likely place for the
    two paths to part company."""
    est = fit(
        GradientBoostingRegressor(n_estimators=30, max_depth=3, random_state=86), 86
    )
    X = np.random.RandomState(87).rand(200, len(FEATURES)) * 5 - 2.5
    ecall, predict = _both_paths([est], _rows(X, np.zeros(len(X), int)))
    assert ecall == predict


def test_ecall_and_predict_agree_on_a_null_id():
    """A NULL id is an unseen group on both paths: NULL out, no call."""
    est = fit(RandomForestRegressor(n_estimators=5, random_state=88, n_jobs=1), 88)
    X = np.random.RandomState(89).rand(4, len(FEATURES))
    nullable_row = create_model(
        "NRow",
        id=(int | None, None),
        **dict.fromkeys(FEATURES, (float | None, None)),
    )
    rows = [
        nullable_row(id=None, **{f: float(v) for f, v in zip(FEATURES, x, strict=True)})
        for x in X
    ]
    udf = PythonTransform(
        name="score",
        instances={0: _AsTransform(est)},
        takes=("f64",) * len(FEATURES),
        returns=("f64",),
    )
    ecall_fn = DuckDBInferFn(
        ECALL_SQL, row_tables={"__THIS__": nullable_row}, static_tables={}, udfs=[udf]
    )
    predict_fn = DuckDBInferFn(
        SQL,
        row_tables={"__THIS__": nullable_row},
        static_tables={},
        models={"m": pack_trees([est], FEATURES)},
    )
    ecall = [r.p for r in ecall_fn.infer({"__THIS__": rows})]
    predict = [r.p for r in predict_fn.infer({"__THIS__": rows})]
    assert ecall == predict == [None] * len(rows)


# ------------------------------------------------------------ refusals --


def test_unfitted_family_refuses():
    from sklearn.linear_model import LinearRegression

    with pytest.raises(TreePackError, match="LinearRegression"):
        pack_trees([LinearRegression().fit(np.zeros((3, 4)), np.zeros(3))], FEATURES)


def test_classifier_refuses():
    """A classifier has a `tree_` exactly like a regressor, but its leaf
    `values` are per-class scores — packing one would score class-0
    fractions and look entirely plausible doing it."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.tree import DecisionTreeClassifier

    rng = np.random.RandomState(71)
    X = rng.rand(60, len(FEATURES))
    y = (X[:, 0] > 0.5).astype(int)
    for est in (
        DecisionTreeClassifier(max_depth=3, random_state=71).fit(X, y),
        RandomForestClassifier(n_estimators=4, random_state=71, n_jobs=1).fit(X, y),
    ):
        with pytest.raises(TreePackError, match="only regressors"):
            pack_trees([est], FEATURES)


def test_bagging_refuses():
    """`estimators_features_` gives each tree a feature SUBSET, so its
    `feature` ids are subset-local. Reading them as global indices scores the
    wrong columns and answers plausibly — refuse instead."""
    from sklearn.ensemble import BaggingRegressor

    rng = np.random.RandomState(72)
    X = rng.rand(60, len(FEATURES))
    est = BaggingRegressor(
        DecisionTreeRegressor(max_depth=3),
        n_estimators=4,
        max_features=2,
        random_state=72,
    ).fit(X, X[:, 0])
    with pytest.raises(TreePackError, match="BaggingRegressor"):
        pack_trees([est], FEATURES)


def test_multi_output_regressor_refuses():
    """`values[:, 0, 0]` takes target 0 unconditionally, so a regressor fitted
    on a 2-D y packs cleanly and serves `est.predict(X)[:, 0]` while the other
    targets vanish. Exactly the failure BaggingRegressor is refused for, one
    axis over."""
    rng = np.random.RandomState(74)
    X = rng.rand(60, len(FEATURES))
    y = np.column_stack([X[:, 0], X[:, 1] * 100])
    for est in (
        DecisionTreeRegressor(max_depth=3, random_state=74).fit(X, y),
        RandomForestRegressor(n_estimators=4, random_state=74, n_jobs=1).fit(X, y),
        ExtraTreesRegressor(n_estimators=4, random_state=74, n_jobs=1).fit(X, y),
    ):
        with pytest.raises(TreePackError, match="multi-output|n_outputs"):
            pack_trees([est], FEATURES)


def test_threshold_rewrite_reproduces_the_f32_comparison():
    """The rewrite has to be exact, not close: `x <= t'` must answer what
    sklearn's `float32(x) <= t` answers for EVERY double, not for sampled
    ones. So probe by walking f64 ULPs across the boundary — a sampled test
    would pass on a rewrite that is off by one ULP, which is the entire bug
    being fixed."""
    rng = np.random.RandomState(76)
    lo = rng.randn(400).astype(np.float32) * np.float32(50)
    hi = rng.randn(400).astype(np.float32) * np.float32(50)
    f32_max = np.float64(np.finfo(np.float32).max)
    thresholds = np.concatenate(
        [
            (lo.astype(np.float64) + hi.astype(np.float64)) / 2,  # sklearn-shaped
            lo.astype(np.float64),  # already exactly f32
            # the ends, where the midpoint has no finite neighbour to average
            # against: +inf is sklearn's own missing-only split marker
            np.array(
                [
                    0.0,
                    -0.0,
                    -2.0,
                    1.0,
                    -1.0,
                    1e-45,
                    -1e-45,
                    1e-320,
                    np.inf,
                    -np.inf,
                    f32_max,
                    -f32_max,
                    1e300,
                    -1e300,
                    np.nextafter(f32_max, 0.0),
                ]
            ),
        ]
    )
    with np.errstate(over="ignore"):  # probing the overflow end is the point
        for t, tp in zip(thresholds, _f32_grid_threshold(thresholds), strict=True):
            probes = [tp, t, np.float64(np.float32(t)), np.inf, -np.inf, np.nan]
            probes += [1e300, -1e300, 0.0, -0.0]
            for seed in (tp, np.float64(np.float32(t))):
                for toward in (-np.inf, np.inf):
                    x = seed
                    for _ in range(8):
                        x = np.nextafter(x, toward)
                        probes.append(x)
            for x in probes:
                assert (np.float64(np.float32(x)) <= t) == (x <= tp), (
                    f"threshold {t!r} -> {tp!r} disagrees at x={x!r}"
                )


def test_threshold_rewrite_flips_the_ticketed_witness():
    """TASK-65's witness, spelled out: 0.15 is BELOW the stored threshold in
    f64 and ABOVE it once narrowed, so the raw compare went left where
    sklearn went right. The rewrite has to move the split below 0.15."""
    t = 0.15000000223517418  # == mean(f32(0.1), f32(0.2))
    (tp,) = _f32_grid_threshold(np.array([t]))
    assert 0.15 <= t, "the raw f64 compare sends 0.15 left"
    assert not 0.15 <= tp, "the rewritten compare must send it right, as sklearn does"


def test_infinite_threshold_is_left_alone():
    """sklearn writes `threshold = inf` for a node whose split is "every
    non-missing value goes left" — a real fitted value, not a sentinel. It
    already admits every non-NaN double, so moving it at all would break the
    missing-value trees, and an earlier revision that refused it did."""
    got = _f32_grid_threshold(np.array([np.inf, 1.0]))
    assert got[0] == np.inf


def test_threshold_rewrite_leaves_leaf_sentinels_alone():
    """A leaf's -2.0 is never read, but a packed table should still look like
    the tree it came from."""
    entry = pack_trees(
        [fit(DecisionTreeRegressor(max_depth=3, random_state=77), 77)], FEATURES
    )
    nodes = entry["nodes"].to_pydict()
    leaves = [
        thr
        for f, thr in zip(nodes["feature"], nodes["threshold"], strict=True)
        if f < 0
    ]
    assert leaves and set(leaves) == {-2.0}


def test_quantised_features_match_sklearn():
    """The parity claim, on data that is not a continuous float64 draw. A
    2-decimal grid is what prices and percentages actually look like."""
    rng = np.random.RandomState(75)
    Xtr = np.round(rng.rand(400, len(FEATURES)) * 10, 2)
    ytr = Xtr[:, 0] * 2 - Xtr[:, 1]
    fitted = RandomForestRegressor(n_estimators=30, random_state=75, n_jobs=1).fit(
        Xtr, ytr
    )
    Xq = np.round(rng.rand(1500, len(FEATURES)) * 10, 2)
    got = score(pack_trees([fitted], FEATURES), Xq)
    want = fitted.predict(Xq)
    bad = np.flatnonzero(got != want)
    assert bad.size == 0, f"{bad.size}/{len(Xq)} rows differ"


def test_hist_gradient_boosting_refuses():
    """A different tree representation entirely (`_predictors`, binned
    thresholds) — not this layout."""
    from sklearn.ensemble import HistGradientBoostingRegressor

    rng = np.random.RandomState(73)
    X = rng.rand(60, len(FEATURES))
    est = HistGradientBoostingRegressor(max_iter=5, random_state=73).fit(X, X[:, 0])
    with pytest.raises(TreePackError, match="HistGradientBoostingRegressor"):
        pack_trees([est], FEATURES)


def test_feature_count_mismatch_refuses():
    fitted = fit(DecisionTreeRegressor(max_depth=3, random_state=61), 61)
    with pytest.raises(TreePackError, match="4 features"):
        pack_trees([fitted], ["a", "b"])


def test_empty_refuses():
    with pytest.raises(TreePackError, match="no estimators"):
        pack_trees([], FEATURES)


# ------------------------- known divergences, 2026-08-08 adversarial sweep --
#
# Both pinned xfail-strict: they fail today and cannot silently start passing.
# Full context and the confit-side pins are in
# packages/confit/tests/test_known_divergences.py; tickets are TASK-77/78.


@pytest.mark.xfail(
    strict=True,
    reason="TASK-77: an INTEGER feature is bound through `promote_f64`, so the "
    "value reaching the f32-grid compare is float32(float64(n)) — TWO "
    "roundings. sklearn's _validate_X_predict narrows the int64 array to "
    "float32 in ONE step. Above 2**53 those land a whole float32 ULP apart, so "
    "TASK-65's rewrite (exact for DOUBLE features) does not hold for integer "
    "ones. Reproduced by hand 2026-08-08.",
)
def test_integer_feature_above_2_53_matches_sklearn():
    from pydantic import create_model as _cm

    A, ulp = 1 << 53, 1 << 30  # float32 spacing at 2**53
    B, mid = A + ulp, A + (ulp >> 1)
    n = mid + 1  # just above the float32 midpoint, below the float64 one
    Xtr = np.array([[A]] * 10 + [[B]] * 10, dtype=np.int64)
    est = DecisionTreeRegressor(random_state=0).fit(
        Xtr, np.array([1.0] * 10 + [9.0] * 10)
    )
    assert est.tree_.threshold[0] == float(mid), (
        "the split must sit on the f32 midpoint"
    )

    row = _cm("IntRow", id=(int, ...), n=(int, ...))
    fn = DuckDBInferFn(
        "SELECT tree_predict('m', id, struct_pack(n := n)) AS p FROM __THIS__",
        row_tables={"__THIS__": row},
        static_tables={},
        models={"m": pack_trees([est], ["n"])},
    )
    got = [r.p for r in fn.infer({"__THIS__": [row(id=0, n=n)]})]
    want = list(est.predict(np.array([[n]], dtype=np.int64)))
    assert got == want, f"engine {got} vs sklearn {want} (int64 feature n={n})"


@pytest.mark.xfail(
    strict=True,
    reason="TASK-78 (DISPUTED by the sweep's own verifiers, not adjudicated by "
    "hand): pack_trees checks n_features_in_ but never feature_names_in_, so a "
    "regressor fitted on a DataFrame binds by POSITION against whatever names "
    "the caller passes. Same count, permuted names, silently wrong columns.",
)
def test_dataframe_fitted_model_refuses_mismatched_feature_names():
    pd = pytest.importorskip("pandas")

    rng = np.random.RandomState(90)
    df = pd.DataFrame(rng.rand(80, 2) * 4 - 2, columns=["b", "a"])
    est = DecisionTreeRegressor(max_depth=4, random_state=90).fit(df, df["b"] * 3)
    assert list(est.feature_names_in_) == ["b", "a"]
    # the packer is handed the names in the OTHER order; the count still matches
    with pytest.raises(TreePackError, match="feature_names_in_|name|order"):
        pack_trees([est], ["a", "b"])
