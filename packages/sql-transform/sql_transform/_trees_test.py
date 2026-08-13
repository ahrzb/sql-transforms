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
import pyarrow as pa
import pytest
from confit import DuckDBInferFn
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.tree import DecisionTreeRegressor

from sql_transform import (
    PythonTransform,
    TreeBasedTransform,
    TreePackError,
    UDFError,
)
from sql_transform._trees import _f32_grid_threshold

FEATURES = ["a", "b", "c", "d"]
ROW = pa.schema(
    [pa.field("id", pa.int64(), nullable=False)]
    + [pa.field(f, pa.float64()) for f in FEATURES]
)
# One call shape for both lowerings: a transform is called by its own name,
# instance id first. That the ecall path and the kernel path run the SAME SQL
# is what makes the parity gate below a comparison of engines, not of surfaces.
SQL = "SELECT score(id, " + ", ".join(FEATURES) + ") AS p FROM __THIS__"


def tbt(estimators, name="score", **kw):
    return TreeBasedTransform(
        name,
        instances=dict(enumerate(estimators)),
        takes=pa.schema([(f, pa.float64()) for f in FEATURES]),
        returns=pa.float64(),
        **kw,
    )


def fit(kind, seed, n=120):
    rng = np.random.RandomState(seed)
    X = rng.rand(n, len(FEATURES)) * 4 - 2
    y = X[:, 0] * 3 - X[:, 1] ** 2 + rng.rand(n) * 0.3
    return kind.fit(X, y)


# --- the surface: a sibling of PythonTransform, called `name(id, feats...)` ---


def test_a_tree_based_transform_registers_like_every_other_transform():
    """The whole point of the surface: `udfs=[...]` and `score(id, a, b, c, d)`,
    identical in shape to a PythonTransform. No `models=`, no name-as-a-string,
    no struct at the call site, no user-visible packing step."""
    ests = [
        fit(RandomForestRegressor(n_estimators=12, random_state=s, n_jobs=1), s)
        for s in (41, 42)
    ]
    fn = DuckDBInferFn(
        SQL,
        row_tables={"__THIS__": ROW},
        static_tables={},
        udfs=[
            TreeBasedTransform(
                "score",
                instances=dict(enumerate(ests)),
                takes=pa.schema([(f, pa.float64()) for f in FEATURES]),
                returns=pa.float64(),
            )
        ],
    )
    assert fn.backend == "cranelift"
    X = np.random.RandomState(43).rand(200, len(FEATURES)) * 5 - 2.5
    rows = [
        {"id": i % 2, **{f: float(v) for f, v in zip(FEATURES, x, strict=True)}}
        for i, x in enumerate(X)
    ]
    got = [r["p"] for r in fn.infer_rows(rows)]
    want = [float(ests[i % 2].predict(x[None, :])[0]) for i, x in enumerate(X)]
    assert got == want


def score(estimators, X, backend=None):
    fn = DuckDBInferFn(
        SQL,
        row_tables={"__THIS__": ROW},
        static_tables={},
        udfs=[tbt(estimators)],
    )
    if backend is not None:
        assert fn.backend == backend
    rows = [
        {"id": 0, **{f: float(v) for f, v in zip(FEATURES, x, strict=True)}} for x in X
    ]
    return np.array([r["p"] for r in fn.infer_rows(rows)])


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
    got = score([fitted], X)
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
    got = score([fitted], X, backend=backend)
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
    nodes, _, _ = tbt([fitted]).tree_tables()
    assert len(set(nodes.column("missing_left").to_pylist())) == 2, (
        "every node has the same missing direction — the fit saw no NaNs"
    )
    Xq = rng.rand(400, len(FEATURES)) * 5 - 2.5
    Xq[rng.rand(*Xq.shape) < 0.3] = np.nan
    got = score([fitted], Xq)
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
    fn = DuckDBInferFn(
        SQL, row_tables={"__THIS__": ROW}, static_tables={}, udfs=[tbt([fitted])]
    )
    Xq = np.array([[np.nan, 0.5, np.nan, -1.0], [1.0, np.nan, 2.0, np.nan]])
    as_null = [
        {
            "id": 0,
            **{
                f: (None if np.isnan(v) else float(v))
                for f, v in zip(FEATURES, x, strict=True)
            },
        }
        for x in Xq
    ]
    as_nan = [
        {"id": 0, **{f: float(v) for f, v in zip(FEATURES, x, strict=True)}} for x in Xq
    ]
    null_out = [r["p"] for r in fn.infer_rows(as_null)]
    nan_out = [r["p"] for r in fn.infer_rows(as_nan)]
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
    X = np.random.RandomState(54).rand(90, len(FEATURES)) * 5 - 2.5
    fn = DuckDBInferFn(
        SQL, row_tables={"__THIS__": ROW}, static_tables={}, udfs=[tbt(fits)]
    )
    ids = np.arange(len(X)) % 3
    rows = [
        {"id": int(g), **{f: float(v) for f, v in zip(FEATURES, x, strict=True)}}
        for g, x in zip(ids, X, strict=True)
    ]
    got = np.array([r["p"] for r in fn.infer_rows(rows)])
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


def _both_paths(ests, rows, row_model=None):
    """(ecall answers, predict answers) for the same estimators and rows.

    Both run the IDENTICAL SQL, because both are transforms named `score`
    called `score(id, a, b, c, d)`. Swapping `PythonTransform` for
    `TreeBasedTransform` in the `udfs=` list is the entire difference — which
    is the property the surface exists to have."""
    row_model = row_model or ROW
    shims = {i: _AsTransform(e) for i, e in enumerate(ests)}
    udf = PythonTransform(
        name="score",
        instances=shims,
        takes=pa.schema([(f, pa.float64()) for f in FEATURES]),
        returns=pa.float64(),
    )
    ecall_fn = DuckDBInferFn(
        SQL, row_tables={"__THIS__": row_model}, static_tables={}, udfs=[udf]
    )
    predict_fn = DuckDBInferFn(
        SQL,
        row_tables={"__THIS__": row_model},
        static_tables={},
        udfs=[tbt(ests)],
    )
    ecall = [r["p"] for r in ecall_fn.infer_rows(rows)]
    predict = [r["p"] for r in predict_fn.infer_rows(rows)]
    scored = sum(1 for v in ecall if v is not None)
    assert sum(s.calls for s in shims.values()) == scored, (
        "the ecall side did not run the Python trampoline once per scored row"
    )
    return ecall, predict


def _rows(X, ids):
    return [
        {
            "id": int(g),
            **{
                f: (None if np.isnan(v) else float(v))
                for f, v in zip(FEATURES, x, strict=True)
            },
        }
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
    nullable_row = pa.schema(
        [pa.field("id", pa.int64())] + [pa.field(f, pa.float64()) for f in FEATURES]
    )
    rows = [
        {"id": None, **{f: float(v) for f, v in zip(FEATURES, x, strict=True)}}
        for x in X
    ]
    udf = PythonTransform(
        name="score",
        instances={0: _AsTransform(est)},
        takes=pa.schema([(f, pa.float64()) for f in FEATURES]),
        returns=pa.float64(),
    )
    ecall_fn = DuckDBInferFn(
        SQL, row_tables={"__THIS__": nullable_row}, static_tables={}, udfs=[udf]
    )
    predict_fn = DuckDBInferFn(
        SQL,
        row_tables={"__THIS__": nullable_row},
        static_tables={},
        udfs=[tbt([est])],
    )
    ecall = [r["p"] for r in ecall_fn.infer_rows(rows)]
    predict = [r["p"] for r in predict_fn.infer_rows(rows)]
    assert ecall == predict == [None] * len(rows)


# ------------------------------------------------------------ refusals --


def test_unfitted_family_refuses():
    from sklearn.linear_model import LinearRegression

    with pytest.raises(TreePackError, match="LinearRegression"):
        tbt([LinearRegression().fit(np.zeros((3, 4)), np.zeros(3))])


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
            tbt([est])


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
        tbt([est])


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
            tbt([est])


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
    nodes, _, _ = tbt(
        [fit(DecisionTreeRegressor(max_depth=3, random_state=77), 77)]
    ).tree_tables()
    nodes = nodes.to_pydict()
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
    got = score([fitted], Xq)
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
        tbt([est])


def test_feature_count_mismatch_refuses():
    """`takes` is the declared width; an estimator fitted on a different one
    would read the wrong columns."""
    fitted = fit(DecisionTreeRegressor(max_depth=3, random_state=61), 61)
    with pytest.raises(TreePackError, match="fitted on 4 features"):
        TreeBasedTransform(
            "score",
            instances={0: fitted},
            takes=pa.schema([("a", pa.float64()), ("b", pa.float64())]),
        )


def test_empty_refuses():
    with pytest.raises(UDFError, match="no fitted instances"):
        TreeBasedTransform(
            "score",
            instances={},
            takes=pa.schema([(f, pa.float64()) for f in FEATURES]),
        )


def test_sparse_instance_ids_refuse():
    """The engine indexes models by position, so a hole would silently score
    the wrong instance (or none)."""
    fitted = fit(DecisionTreeRegressor(max_depth=3, random_state=62), 62)
    with pytest.raises(UDFError, match="dense from 0"):
        TreeBasedTransform(
            "score",
            instances={0: fitted, 2: fitted},
            takes=pa.schema([(f, pa.float64()) for f in FEATURES]),
            returns=pa.float64(),
        )


# ------------------------- known divergences, 2026-08-08 adversarial sweep --
#
# Both pinned xfail-strict: they fail today and cannot silently start passing.
# Full context and the confit-side pins are in
# packages/confit/tests/test_known_divergences.py; tickets are TASK-77/78.


# FIXED 2026-08-08 (TASK-77). An integer `tree_predict` feature no longer
# binds through the ordinary `promote_f64`: it converts with `itof.f32`, which
# rounds i64 -> f32 -> f64 in ONE rounding, exactly as sklearn's
# `_validate_X_predict` narrows an integer feature array. `promote_f64` gave
# `float32(float64(n))` — TWO roundings — which above 2**53 lands a whole
# float32 ULP away.
#
# Safe to apply to EVERY integer feature rather than only large ones: below
# 2**53, `float64(n)` is exact, so `float32(float64(n)) == float32(n)` and the
# new node is a no-op. That is what made this a fix rather than a trade.
#
# The IR gains an opcode but no TYPE: `itof.f32` is f64-out, the same way
# `ftoi.nearest` is a rounding mode and not an integer type. The engine still
# computes in exactly i64 / f64 / str / bool.


def _int_split_model(seed=0):
    """A tree whose single split sits exactly on a float32 midpoint above
    2**53 — the only place the two roundings disagree."""
    a, ulp = 1 << 53, 1 << 30  # float32 spacing at 2**53
    b, mid = a + ulp, a + (ulp >> 1)
    x = np.array([[a]] * 10 + [[b]] * 10, dtype=np.int64)
    est = DecisionTreeRegressor(random_state=seed).fit(
        x, np.array([1.0] * 10 + [9.0] * 10)
    )
    assert est.tree_.threshold[0] == float(mid), (
        "the split must sit on the f32 midpoint"
    )
    return est, mid


@pytest.mark.parametrize("backend", ["cranelift", "interpreter"])
def test_integer_feature_above_2_53_matches_sklearn(backend, monkeypatch):
    if backend == "interpreter":
        monkeypatch.setenv("SPECIALIZER_FORCE_INTERP", "1")
    else:
        monkeypatch.delenv("SPECIALIZER_FORCE_INTERP", raising=False)

    est, mid = _int_split_model()
    # Walk the float32 ULP around the midpoint: the value just above it is
    # where float32(n) and float32(float64(n)) part company.
    probes = [
        mid - 1,
        mid,
        mid + 1,
        (1 << 53) + (1 << 30),
        1 << 53,
        1 << 62,
        0,
        -(1 << 55),
    ]

    row = pa.schema(
        [
            pa.field("id", pa.int64(), nullable=False),
            pa.field("n", pa.int64(), nullable=False),
        ]
    )
    fn = DuckDBInferFn(
        "SELECT m(id, n) AS p FROM __THIS__",
        row_tables={"__THIS__": row},
        static_tables={},
        udfs=[
            TreeBasedTransform(
                "m", instances={0: est}, takes=pa.schema([("n", pa.int64())])
            )
        ],
    )
    assert fn.backend == backend
    got = [r["p"] for r in fn.infer_rows([{"id": 0, "n": n} for n in probes])]
    want = list(est.predict(np.array([[n] for n in probes], dtype=np.int64)))
    assert got == want, f"engine {got} vs sklearn {want} (int64 features)"


@pytest.mark.parametrize("declared", [pa.int64(), pa.float64()])
def test_call_and_kernel_agree_on_an_integer_feature_above_2_53(declared):
    """`__call__` IS the DuckDB binding and the semantic contract the kernel is
    gated against, so the two must agree — and the sweep found they did not.

    `__call__` built its array with `float(f)`, so sklearn's own float32
    narrowing became a SECOND rounding, while the kernel narrows once
    (`itof.f32`, TASK-77). Above 2**53 that is a whole float32 ULP and a whole
    leaf. Both declarations are checked because they are different right
    answers, not one: a declared BIGINT reaches sklearn as an int64 and
    narrows once, while a declared DOUBLE is cast by DuckDB first and narrows
    from the double — the engine must follow the DECLARATION, not the column.
    """
    est, mid = _int_split_model()
    n = mid + 1
    u = TreeBasedTransform("m", instances={0: est}, takes=pa.schema([("n", declared)]))

    as_int = est.predict(np.array([[n]], dtype=np.int64))[0]
    as_dbl = est.predict(np.array([[n]], dtype=np.float64))[0]
    assert as_int != as_dbl, "the probe must sit where the two roundings differ"
    want = as_int if declared == pa.int64() else as_dbl

    assert u(0, n) == (want,), "__call__ (the contract, and DuckDB's binding)"

    row = pa.schema(
        [
            pa.field("id", pa.int64(), nullable=False),
            pa.field("n", pa.int64(), nullable=False),
        ]
    )
    fn = DuckDBInferFn(
        "SELECT m(id, n) AS p FROM __THIS__",
        row_tables={"__THIS__": row},
        static_tables={},
        udfs=[u],
    )
    assert fn.backend == "cranelift"
    got = [r["p"] for r in fn.infer_rows([{"id": 0, "n": n}])]
    assert got == [want], f"kernel {got} vs contract {want}"


def test_small_integer_features_are_unchanged_by_the_f32_narrowing():
    """AC #2: `float64(n)` is exact below 2**53, so the narrowing must be a
    no-op there — the overwhelmingly common case must not move."""
    rng = np.random.RandomState(77)
    x = rng.randint(-100_000, 100_000, size=(300, 2)).astype(np.int64)
    y = (x[:, 0] * 0.5 - x[:, 1] * 0.25).astype(np.float64)
    est = DecisionTreeRegressor(max_depth=8, random_state=77).fit(x, y)

    row = pa.schema(
        [
            pa.field("id", pa.int64(), nullable=False),
            pa.field("a", pa.int64(), nullable=False),
            pa.field("b", pa.int64(), nullable=False),
        ]
    )
    fn = DuckDBInferFn(
        "SELECT m(id, a, b) AS p FROM __THIS__",
        row_tables={"__THIS__": row},
        static_tables={},
        udfs=[
            TreeBasedTransform(
                "m",
                instances={0: est},
                takes=pa.schema([("a", pa.float64()), ("b", pa.float64())]),
            )
        ],
    )
    probe = rng.randint(-100_000, 100_000, size=(200, 2)).astype(np.int64)
    got = [
        r["p"]
        for r in fn.infer_rows([{"id": 0, "a": int(a), "b": int(b)} for a, b in probe])
    ]
    assert got == list(est.predict(probe))


def test_integer_feature_literal_is_narrowed_too():
    """The constant folder collapses an integer literal feature at build
    time, so it has to fold through f32 as well or the fix has a hole."""
    est, mid = _int_split_model()
    n = mid + 1
    row = pa.schema([pa.field("id", pa.int64(), nullable=False)])
    fn = DuckDBInferFn(
        f"SELECT m(id, {n}) AS p FROM __THIS__",
        row_tables={"__THIS__": row},
        static_tables={},
        udfs=[
            TreeBasedTransform(
                "m", instances={0: est}, takes=pa.schema([("n", pa.int64())])
            )
        ],
    )
    got = [r["p"] for r in fn.infer_rows([{"id": 0}])]
    want = list(est.predict(np.array([[n]], dtype=np.int64)))
    assert got == want, f"engine {got} vs sklearn {want} (literal feature)"


# ADJUDICATED and CONFIRMED 2026-08-08 (TASK-78). The sweep's verifiers had
# split on it; reproduced by hand, and the divergence is not subtle — a forest
# fitted on columns ['b', 'a'] and declared as ['a', 'b'] built without
# complaint and scored -2.72 where sklearn said 0.84. FIXED: the schema's
# field names must match the fitted order. Since the schema always carries
# names, the check is no longer opt-in the way the old `take_names` was; it
# stays conditional on `feature_names_in_`, which exists only for an estimator
# fitted on something with column names.


def _named(est, names):
    return TreeBasedTransform(
        "score",
        instances={0: est},
        takes=pa.schema([(n, pa.float64()) for n in names]),
    )


def test_dataframe_fitted_model_refuses_mismatched_schema_names():
    pd = pytest.importorskip("pandas")

    rng = np.random.RandomState(90)
    df = pd.DataFrame(rng.rand(80, 2) * 4 - 2, columns=["b", "a"])
    est = DecisionTreeRegressor(max_depth=4, random_state=90).fit(df, df["b"] * 3)
    assert list(est.feature_names_in_) == ["b", "a"]
    # declared in the OTHER order; the count still matches
    with pytest.raises(TreePackError, match=r"fitted on columns \['b', 'a'\]"):
        _named(est, ["a", "b"])
    # a name that was never fitted is caught by the same check
    with pytest.raises(TreePackError, match="fitted on columns"):
        _named(est, ["b", "zzz"])
    # ... and the fitted order builds
    assert _named(est, ["b", "a"]).take_names == ("b", "a")


def test_schema_names_do_not_bind_the_call_site():
    """Arguments bind by POSITION, so a call site names its columns whatever
    it likes; the schema's names are a fit-time cross-check, not a key."""
    rng = np.random.RandomState(91)
    x = rng.rand(80, 2) * 4 - 2
    est = DecisionTreeRegressor(max_depth=4, random_state=91).fit(x, x[:, 0] * 3)
    assert not hasattr(est, "feature_names_in_")
    # no `feature_names_in_`: any naming is accepted
    assert _named(est, ["anything", "at_all"]).take_names == ("anything", "at_all")

    row = pa.schema(
        [
            pa.field("id", pa.int64(), nullable=False),
            pa.field("zzz", pa.float64(), nullable=False),
            pa.field("qqq", pa.float64(), nullable=False),
        ]
    )
    fn = DuckDBInferFn(
        "SELECT score(id, zzz, qqq) AS p FROM __THIS__",
        row_tables={"__THIS__": row},
        static_tables={},
        udfs=[_named(est, ["anything", "at_all"])],
    )
    probe = rng.rand(20, 2) * 4 - 2
    got = [
        r["p"]
        for r in fn.infer_rows(
            [{"id": 0, "zzz": float(a), "qqq": float(b)} for a, b in probe]
        )
    ]
    assert got == list(est.predict(probe))


def test_call_answers_null_features_the_kernel_scores():
    """TASK-83 (fuzz seed 3112): `__call__` fed None -> NaN straight into
    `est.predict`, which REJECTS NaN on an estimator fitted without missing
    values (GradientBoosting and RandomForest raise; DecisionTree >= 1.3
    happens to accept). The kernel scored the same row via missing_left. One
    declaration, three bindings, one of them crashing where another answers —
    `__call__` now walks the PACKED tables for NaN rows, the same data the
    kernel reads, so the two cannot drift."""
    kinds = [
        DecisionTreeRegressor(max_depth=3),
        RandomForestRegressor(n_estimators=5, max_depth=4, random_state=9, n_jobs=1),
        GradientBoostingRegressor(n_estimators=5, max_depth=3, random_state=9),
    ]
    rows = [
        {"id": 0, "a": None, "b": 0.5, "c": -1.0, "d": 2.0},
        {"id": 0, "a": 1.5, "b": None, "c": 0.25, "d": None},
        {"id": 0, "a": None, "b": None, "c": None, "d": None},
        {"id": 0, "a": 0.5, "b": 0.5, "c": 0.5, "d": 0.5},  # control: no NULLs
    ]
    for kind in kinds:
        est = fit(kind, 90)  # fitted WITHOUT missing values
        udf = tbt([est])
        fn = DuckDBInferFn(
            SQL, row_tables={"__THIS__": ROW}, static_tables={}, udfs=[udf]
        )
        kernel = [r["p"] for r in fn.infer_rows(rows)]
        called = [udf(0, *(r[f] for f in FEATURES)) for r in rows]
        assert [c[0] for c in called] == kernel, type(kind).__name__


def test_registered_duckdb_udf_answers_null_features_like_the_other_two():
    """TASK-83 AC #2's third binding, executed rather than argued: the
    DuckDB-REGISTERED function (register -> _arrow_scalar_batch -> _scalar ->
    __call__) actually run by duckdb over rows with NULL features — the exact
    leg fuzz seed 3112 crashed. Three-way `==`: registered UDF, kernel,
    direct `__call__`."""
    import duckdb

    rows = [
        {"id": 0, "a": None, "b": 0.5, "c": -1.0, "d": 2.0},
        {"id": 0, "a": 1.5, "b": None, "c": 0.25, "d": None},
        {"id": 0, "a": None, "b": None, "c": None, "d": None},
        {"id": 0, "a": 0.5, "b": 0.5, "c": 0.5, "d": 0.5},  # control: no NULLs
    ]
    kinds = [
        DecisionTreeRegressor(max_depth=3),
        RandomForestRegressor(n_estimators=5, max_depth=4, random_state=9, n_jobs=1),
        GradientBoostingRegressor(n_estimators=5, max_depth=3, random_state=9),
    ]
    for kind in kinds:
        est = fit(kind, 90)  # fitted WITHOUT missing values
        udf = tbt([est])
        fn = DuckDBInferFn(
            SQL, row_tables={"__THIS__": ROW}, static_tables={}, udfs=[udf]
        )
        kernel = [r["p"] for r in fn.infer_rows(rows)]
        called = [udf(0, *(r[f] for f in FEATURES))[0] for r in rows]
        con = duckdb.connect()
        try:
            con.execute("SET threads = 1")
            con.register("__THIS__", pa.Table.from_pylist(rows, schema=ROW))
            udf.register(con)
            duck = [r[0] for r in con.execute(SQL).fetchall()]
        finally:
            con.close()
        assert duck == kernel == called, type(kind).__name__
