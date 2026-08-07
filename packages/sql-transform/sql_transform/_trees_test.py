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

from sql_transform import TreePackError, pack_trees

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
