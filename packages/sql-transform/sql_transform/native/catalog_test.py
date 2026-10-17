"""Every catalog entry is held to its declared bound against its Python twin,
on generated fits and generated serving rows (swap-the-entry, `check`).

Adding an entry means adding its fixtures here: `FIXTURES` maps each
registered class to estimator factories covering its configurations, and
`test_every_entry_has_fixtures` fails until it does.
"""

from __future__ import annotations

import functools
import math
import os
import random
import re
import sys
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pyarrow as pa
import pytest
from sklearn.cluster import (
    Birch,
    BisectingKMeans,
    FeatureAgglomeration,
    KMeans,
    MiniBatchKMeans,
)
from sklearn.compose import ColumnTransformer
from sklearn.cross_decomposition import CCA, PLSSVD, PLSCanonical, PLSRegression
from sklearn.decomposition import (
    PCA,
    FactorAnalysis,
    FastICA,
    IncrementalPCA,
    KernelPCA,
    TruncatedSVD,
)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomTreesEmbedding
from sklearn.feature_selection import (
    RFE,
    RFECV,
    GenericUnivariateSelect,
    SelectFdr,
    SelectFpr,
    SelectFromModel,
    SelectFwe,
    SelectKBest,
    SelectPercentile,
    SequentialFeatureSelector,
    VarianceThreshold,
    f_regression,
)
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.kernel_approximation import (
    AdditiveChi2Sampler,
    Nystroem,
    RBFSampler,
    SkewedChi2Sampler,
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline, make_pipeline
from sklearn.preprocessing import (
    Binarizer,
    FunctionTransformer,
    KBinsDiscretizer,
    MaxAbsScaler,
    MinMaxScaler,
    Normalizer,
    OneHotEncoder,
    OrdinalEncoder,
    PolynomialFeatures,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    SplineTransformer,
    StandardScaler,
    TargetEncoder,
)
from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection
from threadpoolctl import threadpool_limits

from sql_transform._udf import PythonTransform, _dense
from sql_transform.native import (
    NotNative,
    ParityError,
    bound,
    bound_of,
    catalog,
    check,
    explain_native,
    to_native,
)

# An imputer whose fit saw a column only missing warns at every transform
# that it drops it, an encoder warns of an unseen category, and numpy warns
# when a twin overflows on an edge row; the fixtures make all three on
# purpose.
pytestmark = pytest.mark.filterwarnings(
    "ignore:Skipping features without any observed values:UserWarning",
    "ignore:Found unknown categories:UserWarning",
    "ignore:overflow encountered:RuntimeWarning",
    "ignore:invalid value encountered:RuntimeWarning",
    "ignore:divide by zero encountered:RuntimeWarning",
    "ignore:Feature .* is constant:UserWarning",
    "ignore:Bins whose width are too small:UserWarning",
)


@pytest.fixture(autouse=True)
def _one_blas_thread():
    # The suite runs on xdist workers, and BLAS threads per worker
    # oversubscribe the cores: an RFECV fixture took 35 s instead of 0.4 s.
    with threadpool_limits(limits=int(os.environ.get("NATIVE_BLAS_THREADS", "1"))):
        yield


FIXTURES: dict[type, list[Callable[[], Any]]] = {
    StandardScaler: [
        StandardScaler,
        lambda: StandardScaler(with_mean=False),
        lambda: StandardScaler(with_std=False),
        lambda: StandardScaler(with_mean=False, with_std=False),
    ],
    MinMaxScaler: [
        MinMaxScaler,
        lambda: MinMaxScaler(feature_range=(-2.5, 7.0)),
        lambda: MinMaxScaler(clip=True),
        lambda: MinMaxScaler(feature_range=(-1.0, 0.0), clip=True),
    ],
    MaxAbsScaler: [MaxAbsScaler, lambda: MaxAbsScaler(clip=True)],
    RobustScaler: [
        RobustScaler,
        lambda: RobustScaler(with_centering=False),
        lambda: RobustScaler(with_scaling=False),
        lambda: RobustScaler(with_centering=False, with_scaling=False),
        lambda: RobustScaler(quantile_range=(10.0, 90.0), unit_variance=True),
    ],
    Normalizer: [
        lambda: Normalizer("l1"),
        lambda: Normalizer("l2"),
        lambda: Normalizer("max"),
    ],
    Binarizer: [
        Binarizer,
        lambda: Binarizer(threshold=2.5),
        lambda: Binarizer(threshold=-40.0),
    ],
    SimpleImputer: [
        SimpleImputer,
        lambda: SimpleImputer(strategy="median"),
        lambda: SimpleImputer(strategy="most_frequent"),
        lambda: SimpleImputer(strategy="constant", fill_value=-7.5),
        lambda: SimpleImputer(add_indicator=True),
        lambda: SimpleImputer(strategy="median", keep_empty_features=True),
        lambda: SimpleImputer(
            missing_values=-1.0, strategy="most_frequent", add_indicator=True
        ),
        lambda: SimpleImputer(
            missing_values=0.0,
            strategy="constant",
            fill_value=2.5,
            keep_empty_features=True,
            add_indicator=True,
        ),
    ],
    MissingIndicator: [
        MissingIndicator,
        lambda: MissingIndicator(features="all"),
        lambda: MissingIndicator(missing_values=-1.0, error_on_new=False),
        lambda: MissingIndicator(sparse=True),
        lambda: MissingIndicator(features="all", sparse=True),
    ],
    VarianceThreshold: [VarianceThreshold],
    SelectKBest: [lambda: SelectKBest(k=1), lambda: SelectKBest(f_regression, k=1)],
    SelectPercentile: [lambda: SelectPercentile(percentile=50)],
    SelectFpr: [lambda: SelectFpr(alpha=0.5)],
    SelectFdr: [lambda: SelectFdr(alpha=0.5)],
    SelectFwe: [lambda: SelectFwe(alpha=0.5)],
    GenericUnivariateSelect: [
        lambda: GenericUnivariateSelect(mode="percentile", param=50)
    ],
    SelectFromModel: [
        lambda: SelectFromModel(LogisticRegression()),
        lambda: SelectFromModel(
            LogisticRegression(), max_features=1, threshold=-np.inf
        ),
    ],
    RFE: [lambda: RFE(LogisticRegression(), n_features_to_select=1)],
    RFECV: [lambda: RFECV(LogisticRegression(), cv=2)],
    SequentialFeatureSelector: [
        lambda: SequentialFeatureSelector(
            LinearRegression(), n_features_to_select=1, cv=2
        )
    ],
    PolynomialFeatures: [
        PolynomialFeatures,
        lambda: PolynomialFeatures(degree=3, include_bias=False),
        lambda: PolynomialFeatures(degree=3, interaction_only=True),
        lambda: PolynomialFeatures(degree=(2, 3)),
        lambda: PolynomialFeatures(degree=(2, 2), include_bias=False),
    ],
    OrdinalEncoder: [
        OrdinalEncoder,
        lambda: OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            encoded_missing_value=-2,
        ),
        lambda: OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=-1, min_frequency=3
        ),
    ],
    OneHotEncoder: [
        lambda: OneHotEncoder(sparse_output=False),
        lambda: OneHotEncoder(sparse_output=False, handle_unknown="ignore"),
        lambda: OneHotEncoder(
            sparse_output=False, drop="first", handle_unknown="ignore"
        ),
        lambda: OneHotEncoder(
            sparse_output=False, min_frequency=3, handle_unknown="infrequent_if_exist"
        ),
        lambda: OneHotEncoder(sparse_output=False, drop="if_binary"),
        # The default output is sparse, which the step reads dense.
        OneHotEncoder,
        lambda: OneHotEncoder(handle_unknown="ignore", drop="if_binary"),
    ],
    TargetEncoder: [TargetEncoder, lambda: TargetEncoder(target_type="continuous")],
}

# KBinsDiscretizer: the strategies only shape the edges; constant features
# (edges [-inf, inf]) and dropped narrow bins come from the generator's
# zero-variance and few-valued columns.
FIXTURES[KBinsDiscretizer] = [
    lambda: KBinsDiscretizer(encode="ordinal", strategy="uniform"),
    lambda: KBinsDiscretizer(n_bins=2, encode="onehot-dense", strategy="uniform"),
    lambda: KBinsDiscretizer(n_bins=10, encode="ordinal", strategy="quantile"),
    lambda: KBinsDiscretizer(n_bins=3, encode="onehot-dense", strategy="quantile"),
    lambda: KBinsDiscretizer(
        n_bins=7, encode="ordinal", quantile_method="inverted_cdf"
    ),
    lambda: KBinsDiscretizer(n_bins=4, encode="onehot-dense", quantile_method="linear"),
    lambda: KBinsDiscretizer(
        n_bins=6, encode="ordinal", quantile_method="median_unbiased"
    ),
    lambda: KBinsDiscretizer(n_bins=4, encode="ordinal", strategy="kmeans"),
    lambda: KBinsDiscretizer(n_bins=3, encode="onehot-dense", strategy="kmeans"),
    lambda: KBinsDiscretizer(
        n_bins=8, encode="onehot-dense", strategy="uniform", dtype=np.float64
    ),
]

# KBinsDiscretizer(dtype=np.float32): the twin bins float32(x), so every
# edge moves to its cutpoint; each strategy, each dense encoding.
# discretize_test.py serves the rows at and beside the cutpoints.
FIXTURES[KBinsDiscretizer] += [
    lambda: KBinsDiscretizer(encode="ordinal", strategy="uniform", dtype=np.float32),
    lambda: KBinsDiscretizer(
        n_bins=3, encode="onehot-dense", strategy="uniform", dtype=np.float32
    ),
    lambda: KBinsDiscretizer(
        n_bins=6, encode="ordinal", strategy="quantile", dtype=np.float32
    ),
    lambda: KBinsDiscretizer(
        n_bins=4, encode="onehot-dense", strategy="quantile", dtype=np.float32
    ),
    lambda: KBinsDiscretizer(
        n_bins=4, encode="ordinal", strategy="kmeans", dtype=np.float32
    ),
    lambda: KBinsDiscretizer(
        n_bins=3, encode="onehot-dense", strategy="kmeans", dtype=np.float32
    ),
]

# encode="onehot", the default: a sparse output, which the step reads dense.
FIXTURES[KBinsDiscretizer] += [
    KBinsDiscretizer,
    lambda: KBinsDiscretizer(n_bins=3, strategy="uniform", dtype=np.float32),
]

# QuantileTransformer: the fits have 5 to 60 rows, so n_quantiles_ is the
# row count past it; few distinct values (kind 4) make runs of equal
# quantiles. quantile_test.py serves the default 1,000 quantiles.
FIXTURES[QuantileTransformer] = [
    QuantileTransformer,
    lambda: QuantileTransformer(n_quantiles=2),
    lambda: QuantileTransformer(n_quantiles=7),
    lambda: QuantileTransformer(n_quantiles=40),
    lambda: QuantileTransformer(n_quantiles=4, subsample=5, random_state=0),
]


def narrow(factory: Callable[[], Any], n: int) -> Callable[[], Any]:
    """`factory`'s steps take at most `n` features: a translation whose
    build time grows faster than its width (a family's measured widths
    are in its module) is checked the same on fewer of them."""
    factory.max_features = n  # type: ignore[attr-defined]
    return factory


def _spline_with_knots(**params: Any) -> Callable[[], SplineTransformer]:
    """A SplineTransformer given an array of knots, the same for each of
    however many features its fit sees (the step draws its width)."""
    base = np.array([-60.0, -2.5, 0.0, 1.0, 37.0, 900.0])

    def make() -> SplineTransformer:
        est = SplineTransformer(**params)

        def fit(X, y=None):
            del est.fit  # the class's own again
            est.knots = np.tile(base[:, None], (1, np.shape(X)[1]))
            return SplineTransformer.fit(est, X, y)

        est.fit = fit
        return est

    return make


# SplineTransformer: degrees 0 to 4, 2 to 8 knots of each kind, the five
# extrapolations, both biases, both missing modes and both outputs (degree 5
# is in test_spline_at_the_knots). Constant columns make equal knots (all of
# them under "uniform", runs under "quantile", where few-valued columns do
# too, and a zero period under "periodic"); a column only missing makes NaN
# knots under "quantile", which stay Python when the output is sparse
# (sparse_test.py), so no sparse one fits "quantile" knots under "zeros". Rows
# at and beside the knots are in test_spline_at_the_knots. From degree 2 the
# steps take at most SPLINE_FEATURES features. Every width the generator draws
# builds in a few seconds since confit #387 (spline.py, `_build_estimate`),
# but the family's gate share grows with it: 31 s on 4 workers at 8 features,
# 49 s at 16, 60 s without the limit (2026-10-06). The 200-seed sweep
# (NATIVE_SEEDS=200, loops/native/report-format.md) covers every width.
SPLINE_FEATURES = 8
FIXTURES[SplineTransformer] = [
    SplineTransformer,
    lambda: SplineTransformer(degree=0, n_knots=2, extrapolation="continue"),
    lambda: SplineTransformer(degree=0, n_knots=4, extrapolation="linear"),
    lambda: SplineTransformer(degree=0, n_knots=3, extrapolation="periodic"),
    lambda: SplineTransformer(
        degree=1, n_knots=3, extrapolation="linear", include_bias=False
    ),
    lambda: SplineTransformer(degree=1, n_knots=2, extrapolation="periodic"),
    lambda: SplineTransformer(sparse_output=True),
    lambda: SplineTransformer(
        degree=0, n_knots=3, extrapolation="continue", sparse_output=True
    ),
    lambda: SplineTransformer(
        degree=1, n_knots=2, extrapolation="periodic", sparse_output=True
    ),
    *(
        narrow(f, SPLINE_FEATURES)
        for f in [
            lambda: SplineTransformer(
                degree=2, n_knots=6, knots="quantile", extrapolation="periodic"
            ),
            lambda: SplineTransformer(
                degree=2,
                n_knots=4,
                knots="quantile",
                extrapolation="constant",
                order="F",
            ),
            lambda: SplineTransformer(degree=3, n_knots=8, extrapolation="error"),
            lambda: SplineTransformer(
                degree=3, n_knots=4, extrapolation="continue", handle_missing="error"
            ),
            lambda: SplineTransformer(
                degree=4, n_knots=5, extrapolation="periodic", include_bias=False
            ),
            lambda: SplineTransformer(
                degree=4, n_knots=4, knots="quantile", extrapolation="continue"
            ),
            lambda: SplineTransformer(
                degree=4, n_knots=3, extrapolation="linear", handle_missing="error"
            ),
            _spline_with_knots(degree=2, extrapolation="continue"),
            _spline_with_knots(degree=3, extrapolation="periodic", include_bias=False),
            lambda: SplineTransformer(
                degree=3, n_knots=6, extrapolation="periodic", handle_missing="zeros"
            ),
            lambda: SplineTransformer(
                degree=3, n_knots=6, extrapolation="continue", sparse_output=True
            ),
            lambda: SplineTransformer(
                degree=2,
                n_knots=4,
                knots="quantile",
                extrapolation="linear",
                sparse_output=True,
            ),
            _spline_with_knots(degree=2, extrapolation="continue", sparse_output=True),
            lambda: SplineTransformer(
                degree=2,
                n_knots=4,
                extrapolation="linear",
                handle_missing="zeros",
                sparse_output=True,
            ),
        ]
    ),
]

# #432's mutation-proven "zeros" periodic shapes. The direct infinity
# regression covers degrees 0–3 and 5, but not degree 4 or degree 2 without
# the bias; these also run on the generated types, fits and instance ids.
FIXTURES[SplineTransformer] += [
    lambda: SplineTransformer(
        degree=0, n_knots=5, extrapolation="periodic", handle_missing="zeros"
    ),
    lambda: SplineTransformer(
        degree=1, n_knots=4, extrapolation="periodic", handle_missing="zeros"
    ),
    narrow(
        lambda: SplineTransformer(
            degree=2,
            n_knots=5,
            extrapolation="periodic",
            handle_missing="zeros",
            include_bias=False,
        ),
        SPLINE_FEATURES,
    ),
    narrow(
        lambda: SplineTransformer(
            degree=4,
            n_knots=9,
            knots="quantile",
            extrapolation="periodic",
            handle_missing="zeros",
        ),
        SPLINE_FEATURES,
    ),
]


# IsotonicRegression: one feature (the generator reads its one-d input tag)
# and a continuous target. `increasing` only shapes the fit; `y_max=-0.0`
# leaves -0.0 values, where numpy's exact arm at a threshold matters. Ties
# in X and fits left with one threshold come from the generator's
# few-valued and zero-variance columns, rows at thresholds from its integer
# rows; test_isotonic_at_and_between_thresholds serves the rest.
FIXTURES[IsotonicRegression] = [
    IsotonicRegression,
    lambda: IsotonicRegression(increasing=False),
    lambda: IsotonicRegression(increasing="auto"),
    lambda: IsotonicRegression(out_of_bounds="clip"),
    lambda: IsotonicRegression(out_of_bounds="raise"),
    lambda: IsotonicRegression(y_min=-1.0, y_max=1.0, out_of_bounds="clip"),
    lambda: IsotonicRegression(increasing="auto", y_max=-0.0),
    lambda: IsotonicRegression(increasing=False, y_min=0.0, out_of_bounds="raise"),
]


# FunctionTransformer: the identity validated and not, and each function
# served, unvalidated (the twin then answers NaN and infinity) and, for a
# few, validated; each bounded function both ways. Each is held to its own
# bound (`function._BOUNDS`), the rest to 0.
BOUNDED = [np.exp, np.log, np.log2, np.log10, np.tan, np.cbrt]
FUNCTIONS = [
    np.abs,
    np.fabs,
    np.negative,
    np.positive,
    np.conjugate,
    np.square,
    np.sqrt,
    np.reciprocal,
    np.floor,
    np.ceil,
    np.trunc,
    np.rint,
    np.sign,
    np.sin,
    np.cos,
    *BOUNDED,
]
FIXTURES[FunctionTransformer] = [
    FunctionTransformer,
    lambda: FunctionTransformer(validate=True),
    *((lambda f=f: FunctionTransformer(f)) for f in FUNCTIONS),
    lambda: FunctionTransformer(np.sqrt, validate=True),
    lambda: FunctionTransformer(np.rint, validate=True),
    lambda: FunctionTransformer(np.reciprocal, validate=True),
    *((lambda f=f: FunctionTransformer(f, validate=True)) for f in BOUNDED),
]

# Pipeline: compositions across families, a passthrough and a None step, and
# a nested pipeline. The generator reads the tags of the step that first
# reads the row (`_runs`), so an imputer first gets holes and an encoder
# first gets string features.
FIXTURES[Pipeline] = [
    lambda: make_pipeline(SimpleImputer(), StandardScaler()),
    lambda: make_pipeline(StandardScaler(), PolynomialFeatures()),
    lambda: make_pipeline(
        OneHotEncoder(sparse_output=False, handle_unknown="ignore"), MaxAbsScaler()
    ),
    lambda: make_pipeline(
        OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
        StandardScaler(),
    ),
    lambda: make_pipeline(
        KBinsDiscretizer(n_bins=4, encode="ordinal", strategy="uniform"),
        OneHotEncoder(sparse_output=False, handle_unknown="ignore"),
    ),
    lambda: make_pipeline(MinMaxScaler(), Binarizer(threshold=0.5)),
    lambda: make_pipeline(StandardScaler(), SelectKBest(f_regression, k=1)),
    lambda: Pipeline(
        [("skip", "passthrough"), ("scale", RobustScaler()), ("none", None)]
    ),
    lambda: make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True),
        StandardScaler(),
        MinMaxScaler(feature_range=(-1.0, 1.0), clip=True),
    ),
    lambda: Pipeline(
        [
            ("prep", make_pipeline(SimpleImputer(), StandardScaler())),
            ("poly", PolynomialFeatures(include_bias=False)),
            ("bin", Binarizer()),
        ]
    ),
    # A sparse step hands the next one a sparse matrix of 0s and 1s.
    lambda: make_pipeline(OneHotEncoder(handle_unknown="ignore"), MaxAbsScaler()),
    lambda: make_pipeline(
        KBinsDiscretizer(n_bins=3, strategy="uniform"), Binarizer(), Normalizer()
    ),
    # A step within a bound above 0 serves last (compose.py): an error scale,
    # a function's own bound (Box-Cox's, after `positive`, below).
    lambda: make_pipeline(StandardScaler(), PCA(n_components=1)),
    lambda: make_pipeline(MinMaxScaler(), FunctionTransformer(np.exp)),
]


# ColumnTransformer and FeatureUnion. Column specs hold at every width the
# generator draws (1 to 32): callables over the fit matrix, a short slice,
# a boolean mask; a part whose selection is empty is skipped, as sklearn
# skips it. An encoder part selects the string columns, beside a numeric
# part over the rest; the generator reads the tags of the first part
# (`_runs`), so an encoder or imputer goes first to get strings or holes.


def _string_cols(X: np.ndarray) -> list[int]:
    # A string feature is fitted as str or None; a number as a float.
    if X.dtype != object:
        return []
    return [
        j
        for j in range(X.shape[1])
        if any(v is None or isinstance(v, str) for v in X[:, j])
    ]


def _number_cols(X: np.ndarray) -> list[int]:
    strings = set(_string_cols(X))
    return [j for j in range(X.shape[1]) if j not in strings]


def _evens(X: np.ndarray) -> list[int]:
    return list(range(0, X.shape[1], 2))


def _odds(X: np.ndarray) -> list[int]:
    return list(range(1, X.shape[1], 2))


def _thirds(X: np.ndarray) -> np.ndarray:
    return np.arange(X.shape[1]) % 3 == 1


def _onehot() -> OneHotEncoder:
    return OneHotEncoder(sparse_output=False, handle_unknown="ignore")


FIXTURES[ColumnTransformer] = [
    lambda: ColumnTransformer(
        [("std", StandardScaler(), _evens), ("minmax", MinMaxScaler(), _odds)]
    ),
    lambda: ColumnTransformer(
        [
            ("first", RobustScaler(), slice(0, 1)),
            ("mask", Binarizer(threshold=0.5), _thirds),
            ("maxabs", MaxAbsScaler(clip=True), lambda X: [X.shape[1] - 1]),
        ],
        remainder="passthrough",
    ),
    lambda: ColumnTransformer(
        [("std", StandardScaler(), _evens)], remainder=PolynomialFeatures()
    ),
    lambda: ColumnTransformer(
        [("std", StandardScaler(), _evens), ("pass", "passthrough", _odds)],
        transformer_weights={"std": 0.1},
    ),
    lambda: ColumnTransformer(
        [("enc", _onehot(), _string_cols), ("num", StandardScaler(), _number_cols)]
    ),
    lambda: ColumnTransformer(
        [
            (
                "enc",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                _string_cols,
            ),
            ("num", make_pipeline(SimpleImputer(), StandardScaler()), _number_cols),
        ],
        transformer_weights={"enc": -3.0},
    ),
    lambda: ColumnTransformer(
        [
            ("imp", SimpleImputer(strategy="median", add_indicator=True), _evens),
            ("maxabs", MaxAbsScaler(), _odds),
        ],
        remainder="passthrough",
    ),
    # A sparse stack, which drops -0.0, and a weighted sparse part.
    lambda: ColumnTransformer(
        [
            ("enc", OneHotEncoder(handle_unknown="ignore"), _string_cols),
            ("neg", FunctionTransformer(np.negative, validate=True), _number_cols),
        ],
        sparse_threshold=1.0,
    ),
    lambda: ColumnTransformer(
        [
            ("enc", OneHotEncoder(handle_unknown="ignore"), _string_cols),
            ("num", StandardScaler(), _number_cols),
        ],
        transformer_weights={"enc": -2.0},
    ),
    # A part within a bound above 0 beside bit-exact parts (compose.py).
    lambda: ColumnTransformer(
        [("enc", _onehot(), _string_cols), ("pca", PCA(n_components=1), _number_cols)]
    ),
    lambda: ColumnTransformer(
        [("exp", FunctionTransformer(np.exp, validate=True), _evens)],
        remainder="passthrough",
    ),
]

FIXTURES[FeatureUnion] = [
    lambda: FeatureUnion([("std", StandardScaler()), ("minmax", MinMaxScaler())]),
    lambda: FeatureUnion(
        [
            ("imp", SimpleImputer()),
            ("ind", MissingIndicator(features="all")),
            ("std", StandardScaler(with_mean=False)),
        ]
    ),
    lambda: FeatureUnion(
        [("std", StandardScaler()), ("robust", RobustScaler())],
        transformer_weights={"robust": 0.3},
    ),
    lambda: FeatureUnion(
        [("minmax", MinMaxScaler()), ("gone", "drop"), ("pass", "passthrough")]
    ),
    lambda: FeatureUnion(
        [
            ("pipe", make_pipeline(SimpleImputer(), StandardScaler())),
            ("pass", "passthrough"),
        ],
        transformer_weights={"pipe": 2},
    ),
    lambda: FeatureUnion(
        [
            ("bins", KBinsDiscretizer(n_bins=3, encode="onehot-dense")),
            ("poly", PolynomialFeatures(include_bias=False)),
        ]
    ),
    lambda: FeatureUnion(
        [
            ("bins", KBinsDiscretizer(n_bins=3)),
            ("neg", FunctionTransformer(np.negative)),
        ],
        transformer_weights={"bins": -1.0},
    ),
    lambda: FeatureUnion([("pca", PCA(n_components=1)), ("std", StandardScaler())]),
    lambda: FeatureUnion(
        [
            ("pipe", make_pipeline(StandardScaler(), PCA(n_components=1))),
            ("pass", "passthrough"),
        ]
    ),
]


def _clusters(count: Callable[[int], int], **kw: Any) -> Callable[[], Any]:
    """A FeatureAgglomeration factory whose fit sets `n_clusters` to
    `count(n_features)` first: the generator draws the width, and a fit
    asked for more clusters than features raises."""

    def make() -> FeatureAgglomeration:
        est = FeatureAgglomeration(**kw)

        def fit(X, y=None):
            del est.fit  # the class's own again
            est.n_clusters = count(np.shape(X)[1])
            return FeatureAgglomeration.fit(est, X, y)

        est.fit = fit
        return est

    return make


# FeatureAgglomeration: mean pooling (the entry serves no other), one
# cluster, one feature per cluster, and between, under each linkage. A
# one-feature fit raises in sklearn, so those draws are drawn again.
FIXTURES[FeatureAgglomeration] = [
    FeatureAgglomeration,
    _clusters(lambda n: 1),
    _clusters(lambda n: n, linkage="complete"),
    _clusters(lambda n: max(1, n // 2), linkage="average", pooling_func=np.mean),
    _clusters(lambda n: max(1, n - 1), linkage="single"),
    _clusters(lambda n: min(n, 3), linkage="complete", metric="manhattan"),
    lambda: FeatureAgglomeration(n_clusters=None, distance_threshold=100.0),
]


def positive(factory: Callable[[], Any]) -> Callable[[], Any]:
    """`factory` fits positive numbers; serving also challenges its domain
    with negatives and signed zeros."""
    factory.positive = True  # type: ignore[attr-defined]
    return factory


# Fitted lambdas pinned on and around scipy's Box-Cox branches (`log`
# under |lambda| 1e-19; past `lambda * log(x)` 709.78, `exp`, which the
# rows' 1e300 and 1e-300 reach).
BOX_COX_LAMBDAS = [0.0, 1e-20, -1e-20, 1e-19, -1e-19, -3e-12, 3.0, -50.0, 2.0]
_E = float(np.finfo(np.float64).eps)
# And on and around scipy's Yeo-Johnson branches (`log1p` for x >= 0 under
# |lambda| eps, `-log1p(-x)` for x < 0 unless |lambda - 2| > eps, so at
# 2 - eps; 2 + eps rounds to 2), with lambdas whose `expm1` overflows on
# the rows' 1e300 (x >= 0 past 1, x < 0 below 1).
YEO_JOHNSON_LAMBDAS = [
    *(0.0, _E / 2, -_E / 2, _E, -_E, 2 * _E),
    *(2.0, 2 - _E, 2 - 2 * _E, 2 + 2 * _E, 2 + 4 * _E),
    *(1.0, 0.5, 3.0, 50.0, -3.0, -50.0),
]
# A fit in float32 holds float32 lambdas, and the twin rounds `lambda - 2`
# and `2 - lambda` in float32 (NEP 50): `2 - lambda` rounds there for 0.1,
# -0.7 and 1e-8 (to 2).
YEO_JOHNSON_LAMBDAS_32 = [
    0.1, -0.7, 1e-8, 3.7, 1e-30, 0.0, 2.0, 1.9999999, 2.0000002, 1.0, 0.5,
]  # fmt: skip


def _pinned(
    method: str,
    lambdas: list[float],
    dtype: Any = np.float64,
    standardize: bool = False,
):
    """A PowerTransformer(method, standardize) whose fit then pins
    `lambdas_` to `lambdas` in `dtype`, where no fit lands: one per feature
    in turn, from the fit's row count, so that narrow steps reach each of
    them over the seeds. A standardized one's scaler is fit again on the
    pinned transform of the fit data, where that is finite, so that rows
    like the fit's answer about `mean_`."""

    def make() -> PowerTransformer:
        est = PowerTransformer(method, standardize=standardize)

        def fit(X, y=None):
            del est.fit  # the class's own again
            PowerTransformer.fit(est, X, y)
            n, start = len(est.lambdas_), len(X)
            est.lambdas_ = np.array(
                [lambdas[(start + j) % len(lambdas)] for j in range(n)], dtype
            )
            if standardize:
                est.standardize = False
                t = est.transform(X)
                est.standardize = True
                if not np.isinf(t).any():
                    est._scaler.fit(t)
            return est

        est.fit = fit
        return est

    return make


# Unstandardized, then standardized (the default; power.py).
FIXTURES[PowerTransformer] = [
    positive(lambda: PowerTransformer("box-cox", standardize=False)),
    positive(_pinned("box-cox", BOX_COX_LAMBDAS)),
    lambda: PowerTransformer(standardize=False),
    _pinned("yeo-johnson", YEO_JOHNSON_LAMBDAS),
    _pinned("yeo-johnson", YEO_JOHNSON_LAMBDAS_32, np.float32),
    positive(lambda: PowerTransformer("box-cox")),
    positive(_pinned("box-cox", BOX_COX_LAMBDAS, standardize=True)),
    lambda: PowerTransformer(),
    _pinned("yeo-johnson", YEO_JOHNSON_LAMBDAS, standardize=True),
    _pinned("yeo-johnson", YEO_JOHNSON_LAMBDAS_32, np.float32, standardize=True),
]
# Box-Cox as a pipeline's last step, within its own error scale
# (compose.py), unstandardized and standardized.
FIXTURES[Pipeline] += [
    positive(
        lambda: make_pipeline(
            MaxAbsScaler(), PowerTransformer("box-cox", standardize=False)
        )
    ),
    positive(lambda: make_pipeline(MaxAbsScaler(), PowerTransformer("box-cox"))),
]

# RandomTreesEmbedding: both outputs (trees.py). One tree to 30, depth
# 1 to 5 (the default) and unbounded, min_samples_leaf above 1, and
# max_leaf_nodes (best-first growth). The trees allow NaN, so the
# generator puts holes in the fit data (columns partly and wholly
# missing), which sets missing_go_to_left by what the fit saw; serving
# NULLs route by it either way. A fixed random_state keeps the fits the
# same from run to run. Rows at and beside the cutpoints are in
# test_trees_at_the_cutpoints.
FIXTURES[RandomTreesEmbedding] = [
    lambda: RandomTreesEmbedding(n_estimators=1, sparse_output=False, random_state=0),
    lambda: RandomTreesEmbedding(
        n_estimators=3, max_depth=1, sparse_output=False, random_state=1
    ),
    lambda: RandomTreesEmbedding(
        n_estimators=10, max_depth=3, sparse_output=False, random_state=2
    ),
    lambda: RandomTreesEmbedding(
        n_estimators=30, max_depth=5, sparse_output=False, random_state=3
    ),
    lambda: RandomTreesEmbedding(
        n_estimators=5,
        max_depth=4,
        min_samples_leaf=3,
        sparse_output=False,
        random_state=4,
    ),
    lambda: RandomTreesEmbedding(
        n_estimators=8,
        max_depth=None,
        max_leaf_nodes=6,
        sparse_output=False,
        random_state=5,
    ),
    lambda: RandomTreesEmbedding(
        n_estimators=4,
        max_depth=2,
        min_samples_leaf=2,
        max_leaf_nodes=3,
        sparse_output=False,
        random_state=6,
    ),
    lambda: RandomTreesEmbedding(n_estimators=3, max_depth=3, random_state=7),
    lambda: RandomTreesEmbedding(n_estimators=10, random_state=8),
]

# Linear projections, within the matvec error scale (project.py). The fits
# draw constant columns and fewer rows than features, so a whitened
# component can be rounding noise, which stays Python (REFUSED).
# `n_components=0.9` keeps the components that explain 90% of the variance.
FIXTURES[PCA] = [
    PCA,
    lambda: PCA(n_components=1),
    lambda: PCA(n_components=1, whiten=True),
    lambda: PCA(n_components=2, svd_solver="covariance_eigh"),
    lambda: PCA(n_components=0.9, svd_solver="full", whiten=True),
    lambda: PCA(n_components=2, svd_solver="randomized", random_state=0),
]
FIXTURES[IncrementalPCA] = [
    IncrementalPCA,
    lambda: IncrementalPCA(n_components=1, whiten=True),
]
FIXTURES[TruncatedSVD] = [
    lambda: TruncatedSVD(n_components=1, random_state=0),
    lambda: TruncatedSVD(n_components=2, algorithm="arpack", random_state=0),
]
FIXTURES[GaussianRandomProjection] = [
    lambda: GaussianRandomProjection(n_components=1, random_state=0),
    lambda: GaussianRandomProjection(n_components=4, random_state=1),
]
# Bit-exact over its sparse components, in scipy's order; of one feature at
# the "auto" density, its components are dense, within the matvec bound.
FIXTURES[SparseRandomProjection] = [
    lambda: SparseRandomProjection(n_components=1, random_state=0),
    lambda: SparseRandomProjection(n_components=4, density=1 / 3, random_state=1),
    lambda: SparseRandomProjection(n_components=3, dense_output=True, random_state=2),
]
# The centered projections (project.py, `_project_centered`), within their
# own error scale. The generator fits a classifier to a binary target, so a
# discriminant analysis has one output field, and a regressor to one target,
# so PLSCanonical, CCA and PLSSVD have one component. A FastICA can whiten
# rounding noise, as a whitened PCA can, which stays Python (REFUSED).
FIXTURES[FastICA] = [
    lambda: FastICA(random_state=0),
    lambda: FastICA(n_components=2, whiten="arbitrary-variance", random_state=1),
    lambda: FastICA(n_components=1, whiten_solver="eigh", random_state=2),
    lambda: FastICA(whiten=False, random_state=3),
]
FIXTURES[LinearDiscriminantAnalysis] = [
    LinearDiscriminantAnalysis,
    lambda: LinearDiscriminantAnalysis(solver="eigen"),
    lambda: LinearDiscriminantAnalysis(solver="eigen", shrinkage="auto"),
]
FIXTURES[PLSRegression] = [
    lambda: PLSRegression(n_components=1),
    lambda: PLSRegression(scale=False),
]
FIXTURES[PLSCanonical] = [
    lambda: PLSCanonical(n_components=1),
    lambda: PLSCanonical(n_components=1, scale=False, algorithm="svd"),
]
FIXTURES[CCA] = [lambda: CCA(n_components=1)]
FIXTURES[PLSSVD] = [
    lambda: PLSSVD(n_components=1),
    lambda: PLSSVD(n_components=1, scale=False),
]
# FactorAnalysis (project.py, `_factor_analysis`), within its own error
# scale. Where I + Wpsi @ components_.T is too ill-conditioned for a bound
# that holds on any host, the step stays Python (REFUSED): a constant
# column of the fit has sklearn's floor of noise, 1e-12.
FIXTURES[FactorAnalysis] = [
    lambda: FactorAnalysis(n_components=1, random_state=0),
    lambda: FactorAnalysis(n_components=2, rotation="varimax", random_state=1),
    lambda: FactorAnalysis(svd_method="lapack", random_state=2),
]
# Distances to fitted centres, within their error scale on the squares
# (distance.py). A fit of fewer rows than its clusters raises, and is drawn
# again. Birch's subclusters are its output fields: under the default
# threshold of 0.5, about one a fit row.
FIXTURES[KMeans] = [
    lambda: KMeans(n_clusters=3, n_init=1, random_state=0),
    lambda: KMeans(n_clusters=1, n_init=1, algorithm="elkan", random_state=1),
]
FIXTURES[MiniBatchKMeans] = [
    lambda: MiniBatchKMeans(n_clusters=4, n_init=1, random_state=0),
]
FIXTURES[BisectingKMeans] = [
    lambda: BisectingKMeans(n_clusters=3, random_state=0),
    lambda: BisectingKMeans(
        n_clusters=4, bisecting_strategy="largest_cluster", random_state=1
    ),
]
FIXTURES[Birch] = [
    Birch,
    lambda: Birch(threshold=100.0, n_clusters=None, branching_factor=5),
]
# The additive chi2 map (kernel.py): its first field alone, bit-exact, at
# the default interval and one past 1; with its cos and sin fields, within
# their error scale, at sklearn's defaults and at j up to 3 of an interval
# whose rounding amplifies a log apart most (A = 2/1.0004). After a
# MinMaxScaler, a row past the fit's range is negative, where the twin's
# last step raises.
FIXTURES[AdditiveChi2Sampler] = [
    positive(lambda: AdditiveChi2Sampler(sample_steps=1)),
    positive(lambda: AdditiveChi2Sampler(sample_steps=1, sample_interval=2.7)),
    positive(AdditiveChi2Sampler),
    positive(lambda: AdditiveChi2Sampler(sample_steps=3)),
    positive(lambda: AdditiveChi2Sampler(sample_steps=4, sample_interval=0.2501)),
]
FIXTURES[Pipeline] += [
    lambda: make_pipeline(MinMaxScaler(), AdditiveChi2Sampler(sample_steps=1)),
    lambda: make_pipeline(MinMaxScaler(), AdditiveChi2Sampler()),
]
# The RBF sampler (kernel.py), within its error scale: few and many
# components, and a `gamma` the fit derives from the data's variance. The
# fit draws the weights, so each fixes `random_state`.
FIXTURES[RBFSampler] = [
    lambda: RBFSampler(n_components=5, random_state=0),
    lambda: RBFSampler(gamma="scale", n_components=100, random_state=1),
]
FIXTURES[Pipeline].append(
    lambda: make_pipeline(StandardScaler(), RBFSampler(n_components=8, random_state=2))
)
# The skewed chi2 sampler (kernel.py), within its error scale: rows above
# -s, where each row answers, and rows of either sign at s = 3, where the
# twin rejects each x <= -3.
FIXTURES[SkewedChi2Sampler] = [
    positive(lambda: SkewedChi2Sampler(n_components=5, random_state=0)),
    lambda: SkewedChi2Sampler(skewedness=3.0, n_components=50, random_state=1),
]
FIXTURES[Pipeline].append(
    lambda: make_pipeline(
        MinMaxScaler(), SkewedChi2Sampler(n_components=8, random_state=2)
    )
)
# Nystroem's rbf kernel (kernel.py), within its error scale: the default
# gamma, 1 / n_features, and one of `kernel_params`. A fit of fewer rows
# than its components keeps one a row. Few components: `check` runs a
# definition that spells each kernel out in every output field.
FIXTURES[Nystroem] = [
    lambda: Nystroem(n_components=5, random_state=0),
    lambda: Nystroem(kernel_params={"gamma": 1e-4}, n_components=20, random_state=1),
]
FIXTURES[Pipeline].append(
    lambda: make_pipeline(StandardScaler(), Nystroem(n_components=8, random_state=2))
)

# A string feature's fitted values, and the unseen ones serving adds.
VOCAB = ["a", "b", "c", "d", "é", "日本"]
UNSEEN = ["zz", "", "A"]
# Finite edges and non-finite values run separately: a wide edge row must
# not almost always trap on an infinity before its finite values are compared.
EDGES = [
    0.0, -0.0, 1e-300, -1e300, 1e300, 5e-324, -5e-324,
    sys.float_info.max, math.inf, -math.inf, math.nan,
]  # fmt: skip
FINITE_EDGES = [v for v in EDGES if math.isfinite(v)]
NONFINITE_EDGES = [v for v in EDGES if not math.isfinite(v)]
OFFSETS = (1e4, 1e8, 1e12)
# A row of only these has a norm under sklearn's zero-scale threshold.
SMALL = [0.0, -0.0, 1e-300, -5e-324, 1e-17, -2.5e-16]
# The widest step drawn, in output fields: a wider fixture checks the same
# translation, only slower. 1,000 holds degree-2 PolynomialFeatures over all
# 32 features and degree 3 over 16; with 8 seeds the classes that draw past
# 300 output fields (PolynomialFeatures, OneHotEncoder, KBinsDiscretizer)
# run in 35 s, against 31 s at 300 and 44 s at 2,000 (master 49acad5).
MAX_OUTPUT_FIELDS = 1000
# A feature's type: a double, an integer or a boolean, by these cumulative
# shares (a boolean about 15%); and the share of steps all boolean.
SHARES = [0.6, 0.85]
BOOL_STEPS = 0.07
# Refusals the generator draws, each tested by name on its own
# (test_a_function_transformer_refuses, compose_test.py, project_test.py):
# a function the twin computes in a narrower dtype on a boolean array, a
# whitened component of rounding noise, a FastICA fit that broke down, and
# a FactorAnalysis whose cov_z is too ill-conditioned for a bound that
# holds on any host, or whose LU has two pivot candidates about alike; and a
# standardized PowerTransformer whose fitted mean_ or scale_ is non-finite.
REFUSED = (
    "over boolean features only",
    "a whitened component of rounding noise",
    "its fit broke down",
    "a mean_ of 2**970 or more",
    "a mean_ or scale_ not finite",
    "too ill-conditioned for a bound",
    "two pivot candidates of the LU",
)
# Seeds per configuration: 8 in the gate; a milestone report sweeps more
# (NATIVE_SEEDS=200, loops/native/report-format.md).
SEEDS = int(os.environ.get("NATIVE_SEEDS", "8"))
SEED_BASE = int(os.environ.get("NATIVE_SEED_BASE", "0"))


@dataclass(frozen=True)
class _Fixture(PythonTransform):
    fit_rows: dict[int, np.ndarray] = field(default_factory=dict, compare=False)


def _fit_matrix(
    rng: np.random.Generator,
    kinds: list[int],
    types: list[pa.DataType],
    holes: list[str | None],
    marker: float,
    positive: bool = False,
    regression: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """One instance's fit data, a column per feature of its kind (an integer
    one rounded), and a binary target, or for a `regression` a continuous
    one: a random multiple of the first feature, standardized, plus noise.
    `kinds` and `holes` are the step's, the same for every instance, as a
    feature keeps its nature across fitted groups: `holes[j]` is None for
    nowhere missing (and never holding the marker), "some" for about a
    fifth of the rows and at least one, "all" for everywhere. `marker`
    spells missing."""
    n = int(rng.integers(5, 60))
    cols = []
    strings = [t == pa.string() for t in types]
    for kind, t, hole in zip(kinds, types, holes, strict=True):
        if t == pa.bool_():
            cols.append(_bool_column(rng, n, hole, marker))
            continue
        if t == pa.string():
            # The step's share of VOCAB for this feature: kind - 3 of them.
            c = rng.choice(np.array(VOCAB[: kind - 3], dtype=object), n)
            if hole == "all":
                c[:] = None
            elif hole == "some":
                c[rng.random(n) < 0.2] = None
                c[rng.integers(n)] = None
            cols.append(c)
            continue
        if kind == 0:
            c = rng.normal(rng.uniform(-100, 100), rng.uniform(0.01, 50), n)
        elif kind == 1:
            c = rng.integers(-1000, 1000, n).astype(float)
        elif kind == 2:
            c = np.full(n, rng.uniform(-5, 5))  # zero variance
        elif kind == 3:
            c = rng.exponential(rng.uniform(0.1, 1e6), n)
        elif kind == 5:
            c = rng.normal(rng.choice(OFFSETS), 1.0, n)
        else:
            c = rng.integers(-3, 4, n).astype(float)  # few distinct values
        if t == pa.int64():
            c = np.round(c)
        if positive:
            c = np.abs(c)
            c[c == 0] = 1.0
        if not np.isnan(marker):
            c[c == marker] = marker + 1.0
        if hole == "all":
            c[:] = marker
        elif hole == "some":
            c[rng.random(n) < 0.2] = marker
            c[rng.integers(n)] = marker
        cols.append(c)
    y = (rng.random(n) < 0.5).astype(int)
    y[:4] = (0, 1, 0, 1)  # two of each class, for a 2-fold split
    if regression and not strings[0]:
        c = np.nan_to_num(np.asarray(cols[0], dtype=float))
        t = (c - c.mean()) / (c.std() or 1.0)
        y = rng.uniform(-3, 3) * t + rng.normal(0, rng.uniform(0.1, 2), n)
    if any(strings):
        # A string beside numbers makes an object matrix, as the step's
        # rows reach `transform`.
        X = np.empty((n, len(cols)), dtype=object)
        for j, c in enumerate(cols):
            boolean = types[j] == pa.bool_()
            X[:, j] = c if strings[j] else [_scalar(v, boolean) for v in c]
        return X, y
    # Booleans only make a boolean matrix; beside a number, or with a hole,
    # a float64 one: as numpy makes the step's row.
    return np.column_stack(cols), y


def _bool_column(
    rng: np.random.Generator, n: int, hole: str | None, marker: float
) -> np.ndarray:
    """A boolean feature's fit column: True at a drawn rate, sometimes
    never or always (a constant column). A hole is the marker, NaN unless
    an imputer says otherwise, as a NULL boolean reaches `transform` as
    NaN, and it makes the column float; without holes it stays boolean.
    A marker of 0 or 1 is not avoided as a number's is: the step hands
    False and True, which equal it, and the twin reads them as missing."""
    p = (0.0, 1.0, rng.uniform(0.1, 0.9))[int(rng.choice(3, p=[0.1, 0.1, 0.8]))]
    c = rng.random(n) < p
    if hole is None:
        return c
    f = c.astype(float)
    if hole == "all":
        f[:] = marker
    else:
        f[rng.random(n) < 0.2] = marker
        f[rng.integers(n)] = marker
    return f


def _scalar(v: Any, boolean: bool) -> Any:
    """A fit value in an object matrix as the step hands it: a boolean
    feature's False and True as Python bools (from a float column too, one
    with holes), a number or a hole as a float."""
    return bool(v) if boolean and v in (0.0, 1.0) else float(v)


def _step(cls_factory, seed: int, variant: int = 0) -> PythonTransform:
    # Each of a class's configurations (`variant`) draws shapes of its own,
    # so a class's seeds are not the same few shapes for every one.
    rng = np.random.default_rng([seed, variant])
    # A shape the estimator cannot fit (a selector asked for more features
    # than the step has), that keeps no output field or that is wider than
    # MAX_OUTPUT_FIELDS is drawn again.
    for _ in range(20):
        step = _draw(rng, cls_factory)
        if step is not None:
            return step
    raise AssertionError(f"no fixture of {cls_factory} fits in 20 draws")


def _runs(est: Any) -> list[Any]:
    """The estimators `transform` runs, in order: `est`, or a pipeline's
    steps that run, or a column transformer's or union's parts (remainder
    last), nested ones flattened. A composition's own tags do not say what
    it takes: sklearn 1.9 copies only `pairwise` (a pipeline's first step)
    and `sparse` (all steps or parts) from its estimators, so `allow_nan`
    and `categorical` read False. The generator reads theirs instead."""
    if isinstance(est, Pipeline):
        return [r for _, _, s in est._iter() for r in _runs(s)]
    if isinstance(est, ColumnTransformer):
        parts = [t for _, t, _ in getattr(est, "transformers_", est.transformers)]
        if not hasattr(est, "transformers_"):
            parts.append(est.remainder)
    elif isinstance(est, FeatureUnion):
        parts = [t for _, t in est.transformer_list]
    else:
        return [est]
    return [r for t in parts if not isinstance(t, str) for r in _runs(t)]


def _draw(rng: np.random.Generator, cls_factory) -> PythonTransform | None:
    # Mostly narrow; sometimes wide enough for a row reduction's blocks.
    wide = rng.random() < 0.3
    n_features = int(rng.integers(5, 33) if wide else rng.integers(1, 5))
    n_features = min(n_features, getattr(cls_factory, "max_features", n_features))
    # Mostly doubles, some integers and booleans; a few steps all boolean,
    # whose rows reach `transform` as a boolean array.
    if rng.random() < BOOL_STEPS:
        types = [pa.bool_()] * n_features
    else:
        types = [
            (pa.float64(), pa.int64(), pa.bool_())[int(np.searchsorted(SHARES, r))]
            for r in rng.random(n_features)
        ]
    kinds = [int(rng.integers(6)) for _ in range(n_features)]
    runs = _runs(cls_factory())
    proto = runs[0]  # what reads the row
    positive = getattr(cls_factory, "positive", False)
    if positive:
        # No boolean column fits: False is not positive, and all True is
        # constant, which Box-Cox's fit rejects too.
        types = [pa.float64() if t == pa.bool_() else t for t in types]
    if not proto.__sklearn_tags__().input_tags.two_d_array:
        # A one-dimensional input (IsotonicRegression): one feature.
        n_features, types, kinds = 1, types[:1], kinds[:1]
    regression = runs[-1].__sklearn_tags__().estimator_type == "regressor"
    if proto.__sklearn_tags__().input_tags.categorical:
        # Categories: few distinct values per feature, about half of them
        # strings (kind 5..8: two to five of VOCAB), the others booleans or
        # few-valued numbers; an all-boolean step stays so.
        for j in range(n_features):
            if types[j] == pa.bool_():
                continue
            if rng.random() < 0.5:
                types[j], kinds[j] = pa.string(), int(rng.integers(5, 9))
            else:
                kinds[j] = 4
    takes = pa.schema([(f"x{i}", t) for i, t in enumerate(types)])
    # Missing values in the fit data, for an estimator that takes them: an
    # imputer's own `missing_values`, or NaN where sklearn says it allows it
    # (for a pipeline, every step: a scaler passes NaN on).
    marker = float(getattr(proto, "missing_values", np.nan))
    holes: list[str | None] = [None] * n_features
    if hasattr(proto, "missing_values") or all(
        r.__sklearn_tags__().input_tags.allow_nan for r in runs
    ):
        holes = [
            ("some", "all", None)[int(np.searchsorted([0.4, 0.48], rng.random()))]
            for _ in range(n_features)
        ]
        if hasattr(proto, "missing_values") and "some" not in holes:
            holes[int(rng.integers(n_features))] = "some"
    # Every fit gets a target (an unsupervised one ignores it). A step's
    # instances share one width, as a fitted step's do: an instance that
    # fits to another width ends the step before it.
    instances: dict[int, Any] = {}
    fit_rows: dict[int, np.ndarray] = {}
    width = 0
    with warnings.catch_warnings():  # all-missing columns, by design
        warnings.simplefilter("ignore")
        try:
            for k in range(int(rng.integers(1, 4))):
                X, y = _fit_matrix(
                    rng, kinds, types, holes, marker, positive, regression
                )
                original = X.copy()
                est = cls_factory().fit(X, y)
                w = np.asarray(_dense(est.transform(X[:1]))).reshape(1, -1).shape[1]
                if k and w != width:
                    break
                instances[k], width = est, w
                fit_rows[k] = original
        except (ValueError, TypeError, IndexError):
            # sklearn rejects the data, or numpy cannot run the fit on it
            # (a quantile of a boolean column subtracts booleans), or the
            # fit fails on it (a discriminant analysis by SVD indexes an
            # empty spectrum where no feature varies within a class).
            return None
    if width == 0 or width > MAX_OUTPUT_FIELDS:
        return None
    if width == 1:
        returns = pa.float64()
    elif rng.random() < 0.3:  # unnamed output fields
        returns = pa.list_(pa.float64(), width)
    else:
        returns = pa.struct([(f"f{j}", pa.float64()) for j in range(width)])
    return _Fixture("tf", instances, takes, returns, fit_rows)


def _value(rng: random.Random, regime: str) -> float | None:
    r = rng.random()
    if regime == "nulls" and r < 0.3:
        return None
    if regime == "finite-edges" and r < 0.3:
        return rng.choice(FINITE_EDGES)
    if regime == "nonfinite" and r < 0.3:
        return rng.choice(NONFINITE_EDGES)
    if regime == "magnitude":
        return rng.choice(OFFSETS) + rng.uniform(-10.0, 10.0)
    if regime == "small":
        return rng.choice(SMALL)
    if regime == "ints":
        return float(rng.randint(-3, 3))
    return rng.uniform(-1e3, 1e3)


def _fitted_values(est: Any) -> list[float]:
    """Float values in fitted state, including nested arrays and objects
    (for example spline knots), and the fitted parts of a composition."""
    values: list[float] = []
    seen: dict[int, Any] = {}

    def collect(v):
        if isinstance(v, float | np.floating):
            if np.isfinite(v):
                values.append(float(v))
            return
        if id(v) in seen:
            return
        seen[id(v)] = v
        if isinstance(v, np.ndarray):
            for x in v.flat:
                collect(x)
        elif isinstance(v, np.void) and v.dtype.names:
            for name in v.dtype.names:
                collect(v[name])
        elif isinstance(v, dict):
            for x in v.values():
                collect(x)
        elif isinstance(v, list | tuple):
            for x in v:
                collect(x)
        elif hasattr(v, "__dict__"):
            collect(vars(v))
        elif hasattr(v, "__getstate__"):
            state = v.__getstate__()
            if state is not None:
                collect(state)

    for part in _runs(est):
        for name, value in vars(part).items():
            if name.endswith("_"):
                collect(value)
    return values


def _rows(step: PythonTransform, seed: int, positive: bool = False) -> pa.Table:
    rng = random.Random(seed)  # noqa: S311
    ids, rows = [], []
    keys = list(step.instances)
    fit_rows = getattr(step, "fit_rows", {})

    def append(iid, row):
        ids.append(iid)
        rows.append(row)

    # Both exact replay (categorical domains) and one-ulp replay (continuous
    # boundaries). Never pair one instance's training row with another's id.
    for i in range(12):
        iid = keys[i % len(keys)]
        if iid not in fit_rows:
            continue
        X = fit_rows[iid]
        row = list(X[(seed + i) % len(X)])
        append(iid, row)
        append(
            iid,
            [
                float(np.nextafter(v, math.inf if i % 2 else -math.inf))
                if f.type == pa.float64() and v is not None
                else v
                for f, v in zip(step.takes, row, strict=True)
            ],
        )

    # A bounded sample of fitted constants, each at and beside both signs.
    # The seed rotates through the full flattened state rather than always
    # choosing its first coefficient.
    constants = {iid: _fitted_values(est) for iid, est in step.instances.items()}
    for i in range(3):
        iid = keys[i % len(keys)]
        values = constants[iid]
        if not values or iid not in fit_rows:
            continue
        v = values[(seed * 3 + i) % len(values)]
        base = list(fit_rows[iid][0])
        for sign in (v, -v):
            for value in (
                sign,
                np.nextafter(sign, -np.inf),
                np.nextafter(sign, np.inf),
            ):
                append(
                    iid,
                    [
                        float(value) if f.type == pa.float64() else x
                        for f, x in zip(step.takes, base, strict=True)
                    ],
                )

    start = len(rows)
    regimes = rng.choices(
        ["plain", "finite-edges", "nonfinite", "nulls", "small", "ints", "magnitude"],
        [25, 15, 5, 15, 10, 15, 15],
        k=40,
    )
    for i, regime in enumerate(regimes):
        row = []
        for f in step.takes:
            if f.type == pa.string():
                v = (
                    None
                    if regime == "nulls" and rng.random() < 0.3
                    else rng.choice(VOCAB + UNSEEN)
                )
            elif f.type == pa.bool_():
                v = (
                    None
                    if regime == "nulls" and rng.random() < 0.3
                    else rng.random() < 0.5
                )
            else:
                v = EDGES[i] if i < len(EDGES) else _value(rng, regime)
                if positive and v is not None:
                    v = abs(v)
            row.append(v)
        append(rng.choice(keys), row)
    if positive:
        for v in (-1.0, -5e-324, -0.0):
            append(keys[0], [v] * len(step.takes))
    # Fixed 5%, independent of the number of fitted instances; preserve
    # the replay and deterministic edge rows' own ids.
    for i in rng.sample(range(start + len(EDGES), start + 40), round(len(rows) * 0.05)):
        ids[i] = None
    cols = {"__iid": pa.array(ids, pa.int64())}
    for j, f in enumerate(step.takes):
        vals = [r[j] for r in rows]
        if f.type == pa.bool_():
            vals = [None if v is None or math.isnan(v) else bool(v) for v in vals]
        elif f.type == pa.int64():
            vals = [
                None
                if v is None or math.isnan(v)
                else round(max(-(2**62), min(2**62, v)))
                for v in vals
            ]
        cols[f.name] = pa.array(vals, f.type)
    return pa.table(cols)


def _row_can_overflow(est: Any, X: Any) -> bool:
    """The open overflow question's rows, not a failed parity comparison.

    Test the absolute terms of fitted matrix products in units of DBL_MAX,
    before multiplying: this test must also work on a 64-bit long double.
    Compositions inspect the actual row each part reads and its carried bound.
    """
    from sql_transform.native._carry import absolute
    from sql_transform.native.compose import _columns, _parts

    if isinstance(est, Pipeline):
        part_X = X
        for _, _, part in est._iter():
            if _row_can_overflow(part, part_X):
                return True
            part_X = part.transform(part_X)
    elif isinstance(est, ColumnTransformer | FeatureUnion):
        for _, part, columns in _parts(est):
            part_X = X
            if columns is not None:
                indices = _columns(columns, np.shape(X)[1])
                if not indices:
                    continue
                part_X = np.asarray(X, dtype=object)[:, indices]
            if _row_can_overflow(part, part_X):
                return True
    else:
        entry = catalog().get(type(est))
        if entry is None or entry.sum_products is None:
            return False
        x = np.asarray(_dense(X), dtype=np.longdouble).reshape(-1)
        if not np.isfinite(x).all():
            return False  # keep the separate rejected-domain rows
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            for left, right in entry.sum_products(est, x):
                scaled = np.abs(np.asarray(left, np.longdouble)) / sys.float_info.max
                terms = scaled * np.abs(np.asarray(right, np.longdouble))
                if np.any(np.sum(terms, axis=-1) >= 1):
                    return True
        return False
    entry = catalog()[type(est)]
    parts = entry.part_bounds(est)
    if parts is None:
        return False
    y = np.asarray(_dense(est.transform(X)), dtype=np.longdouble).reshape(-1)
    e = absolute(parts.at(X), y)
    return bool(np.any(np.abs(y) / sys.float_info.max + e / sys.float_info.max >= 1))


def _gate_rows(step: PythonTransform, rows: pa.Table) -> pa.Table:
    """Keep raw overflow rows for strict xfails, not the default parity gate.

    The interim rule keeps check strict and serves entries unchanged:
    loops/native/decisions/open/where-a-sum-overflows.md.
    """
    from sql_transform.native._check import _twin_input

    keep = []
    for i, iid in enumerate(rows.column("__iid").to_pylist()):
        try:
            risk = iid is not None and _row_can_overflow(
                step.instances[iid], _twin_input(step, rows, i)
            )
        except (ValueError, TypeError, IndexError):
            risk = False  # preserve rows whose twin rejects the input
        keep.append(not risk)
    return rows.filter(keep)


def test_every_entry_has_fixtures():
    assert set(catalog()) == set(FIXTURES), "add fixtures for every catalog entry"


@pytest.mark.parametrize(
    "cls, j",
    [
        pytest.param(
            c,
            j,
            id=f"{c.__name__}-{j}",
            marks=pytest.mark.skipif(
                np.finfo(np.longdouble).maxexp == 1024
                and c is FactorAnalysis
                and j in (0, 2),
                reason=(
                    "64-bit long double overflows FactorAnalysis's bound arithmetic"
                    " despite finite answers within the rescaled bound"
                ),
            ),
        )
        for c, fs in FIXTURES.items()
        for j in range(len(fs))
    ],
)
def test_an_entry_matches_its_twin(cls, j):
    make = FIXTURES[cls][j]
    answered = 0
    refused = []
    for seed in range(SEED_BASE, SEED_BASE + SEEDS):
        step = _step(make, seed, j)
        try:
            native = to_native(step, strict=True, allow_bound=True)
        except NotNative as e:
            if len(step.takes) <= 4 and not any(r in str(e) for r in REFUSED):
                raise
            refused.append((seed, str(e)))
            continue
        try:
            answered += check(
                step,
                native,
                _gate_rows(step, _rows(step, seed, getattr(make, "positive", False))),
            )
        except ParityError as e:
            raise ParityError(
                f"{cls.__name__} configuration {j}, seed {seed}: {e}"
            ) from e
    # Keep the 64-row default floor even when named fit refusals leave only
    # one accepted seed (FastICA(whiten=False) answers 63 with 8 replays).
    assert answered >= 8 * SEEDS, (answered, "answered id rows", refused)


def test_a_seed_draws_the_same_step_under_any_global_random_state():
    # A fit that reads numpy's global random state, as ARPACK's start does
    # without `random_state`, draws another step from the same seed in each
    # process, so a failure of that step may not repeat (finding 82).
    saved = np.random.get_state()
    try:
        for cls, makes in FIXTURES.items():
            for j, make in enumerate(makes):
                answers = []
                for state in (1, 2):
                    np.random.seed(state)
                    step = _step(make, 0, j)
                    answers.append(_answers(step, make))
                assert answers[0] == answers[1], f"{cls.__name__}, configuration {j}"
    finally:
        np.random.set_state(saved)


def _answers(step: PythonTransform, make) -> list[str]:
    """The twin's answer to each of seed 0's rows, or the error it raises."""
    out = []
    for row in _rows(step, 0, getattr(make, "positive", False)).to_pylist():
        iid = row.pop("__iid")
        try:
            out.append(repr(step(iid, *row.values())))
        except Exception as e:  # noqa: BLE001 - the twin's error is its answer
            out.append(type(e).__name__)
    return out


def test_replay_rows_keep_their_own_instance_and_challenge_both_sides():
    step = _step(StandardScaler, 0)
    rows = _rows(step, 0).to_pylist()
    for i in range(12):
        iid = list(step.instances)[i % len(step.instances)]
        fit = step.fit_rows[iid][i % len(step.fit_rows[iid])]
        exact, nudged = rows[2 * i : 2 * i + 2]
        assert exact["__iid"] == nudged["__iid"] == iid
        for f, v in zip(step.takes, fit, strict=True):
            if f.type == pa.float64() and math.isfinite(v):
                assert exact[f.name] == v
                assert nudged[f.name] == np.nextafter(
                    v, math.inf if i % 2 else -math.inf
                )
    assert sum(r["__iid"] is None for r in rows) == round(len(rows) * 0.05)


def test_fitted_constants_reach_rows_inside_nested_compositions():
    X = np.array([[2.0], [4.0], [6.0]])
    est = make_pipeline(make_pipeline(StandardScaler())).fit(X)
    step = _Fixture("tf", {0: est}, pa.schema([("x0", pa.float64())]), fit_rows={0: X})
    rows = _rows(step, 0).column("x0").to_pylist()
    for value in (4.0, -4.0):
        assert value in rows
        assert np.nextafter(value, -np.inf) in rows
        assert np.nextafter(value, np.inf) in rows


def test_positive_fixtures_still_challenge_the_rejected_domain():
    step = _step(FIXTURES[PowerTransformer][0], 0)
    values = _rows(step, 0, positive=True).column("x0").to_pylist()
    assert -1.0 in values and -5e-324 in values
    assert "-0.0" in list(map(repr, values))


def test_the_default_gate_keeps_safe_extremes_and_rejected_domain_rows():
    est = PCA(n_components=1).fit([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    est.components_ = np.array([[1.0, 1.0]])
    step = PythonTransform(
        "tf", {0: est}, pa.schema([("x0", pa.float64()), ("x1", pa.float64())])
    )
    rows = pa.table(
        {
            "__iid": pa.array([0, 0, None, 0], pa.int64()),
            "x0": [sys.float_info.max, 1e300, sys.float_info.max, math.inf],
            "x1": [sys.float_info.max, 1e300, sys.float_info.max, 1.0],
        }
    )
    assert _gate_rows(step, rows).equals(rows.slice(1))
    est.components_ = np.array([[0.25, 0.25]])
    assert _gate_rows(step, rows).equals(rows)
    X = np.random.default_rng(0).normal(size=(12, 2))
    y = np.arange(12) % 2
    lda = LinearDiscriminantAnalysis().fit(X, y)
    lda.scalings_ = np.array([[0.25, 2.0], [0.25, 2.0]])
    assert not _row_can_overflow(lda, [[sys.float_info.max] * 2])
    pls = PLSRegression(n_components=1, scale=False).fit(X, y)
    pls.x_rotations_[:] = 0.25
    pls.x_weights_[:] = 2.0
    assert not _row_can_overflow(pls, [[sys.float_info.max] * 2])


def test_the_default_gate_finds_the_rbf_sampler_sum_overflow():
    step = _step(FIXTURES[RBFSampler][0], 1, 0)
    rows = _rows(step, 1)
    row = rows.slice(rows.num_rows - 40 + 7, 1)
    assert row.to_pylist() == [
        {
            "__iid": 1,
            "x0": True,
            "x1": sys.float_info.max,
            "x2": sys.float_info.max,
            "x3": 2**62,
        }
    ]
    assert _gate_rows(step, row).num_rows == 0


@pytest.mark.parametrize(
    "cls",
    [
        PCA,
        IncrementalPCA,
        TruncatedSVD,
        GaussianRandomProjection,
        SparseRandomProjection,
        FastICA,
        LinearDiscriminantAnalysis,
        PLSRegression,
        PLSCanonical,
        CCA,
        PLSSVD,
        FactorAnalysis,
        KMeans,
        MiniBatchKMeans,
        BisectingKMeans,
        Birch,
        RBFSampler,
        SkewedChi2Sampler,
        Nystroem,
    ],
)
def test_each_matrix_product_entry_declares_its_actual_sum_operands(cls):
    entry = catalog()[cls]
    assert entry.sum_products is not None
    est = _ladder_step(cls, 2).instances[0]
    products = list(entry.sum_products(est, np.ones(2, np.longdouble)))
    assert products
    for left, right in products:
        a, b = np.broadcast_arrays(left, right)
        assert a.ndim >= 1 and a.shape == b.shape


def test_logarithmic_kernel_sums_use_the_transformed_row():
    est = _ladder_step(SkewedChi2Sampler, 2).instances[0]
    est.random_weights_[:] = 2.0
    assert not _row_can_overflow(est, [[sys.float_info.max] * 2])
    est.random_weights_[:] = sys.float_info.max
    assert _row_can_overflow(est, [[math.e - 1] * 2])
    assert not _row_can_overflow(est, [[-2.0] * 2])
    assert catalog()[AdditiveChi2Sampler].sum_products is None


def test_the_kernel_normalization_product_is_checked_too():
    est = Nystroem(n_components=2, random_state=0).fit(np.zeros((3, 2)))
    est.normalization_[:] = sys.float_info.max
    assert _row_can_overflow(est, [[0.0, 0.0]])


def test_the_second_factor_product_is_checked_too(monkeypatch):
    from sql_transform.native import project

    est = _ladder_step(FactorAnalysis, 2).instances[0]
    est.mean_[:] = 0.0
    f = project._factor(est)._replace(
        v=np.full((2, 2), 0.5), c=np.full((2, 2), sys.float_info.max)
    )
    monkeypatch.setattr(project, "_factor", lambda est: f)
    assert _row_can_overflow(est, [[1.0, 1.0]])


@pytest.mark.skipif(
    np.finfo(np.longdouble).maxexp == 1024,
    reason="64-bit long double overflows S or an intermediate term before K*eps*S",
)
@pytest.mark.parametrize(
    "cls, cfg, seed, offset",
    [
        (FastICA, 0, 3, 7),
        (LinearDiscriminantAnalysis, 1, 0, 7),
        (LinearDiscriminantAnalysis, 2, 7, 7),
        (PLSRegression, 1, 3, 7),
        (PLSCanonical, 1, 3, 7),
        (PLSSVD, 1, 5, 7),
        (FactorAnalysis, 0, 5, 7),
        (FactorAnalysis, 2, 5, 18),
    ],
    ids=lambda v: v.__name__ if isinstance(v, type) else str(v),
)
def test_extreme_finite_pairs_fit_a_computable_bound(cls, cfg, seed, offset):
    step = _step(FIXTURES[cls][cfg], seed, cfg)
    rows = _rows(step, seed)
    row = rows.slice(rows.num_rows - 40 + offset, 1)
    assert check(step, to_native(step, strict=True, allow_bound=True), row) == 1


@pytest.mark.xfail(
    strict=True,
    raises=ParityError,
    reason="strict interim rule: loops/native/decisions/open/where-a-sum-overflows.md",
)
def test_a_projection_sum_overflow_remains_a_strict_mismatch():
    step = _step(FIXTURES[GaussianRandomProjection][1], 3, 1)
    rows = _rows(step, 3)
    row = rows.slice(rows.num_rows - 40 + 7, 1)
    assert _gate_rows(step, row).num_rows == 0
    check(step, to_native(step, strict=True, allow_bound=True), row)


@pytest.mark.xfail(
    strict=True,
    raises=ParityError,
    reason="strict interim rule: loops/native/decisions/open/where-a-sum-overflows.md",
)
@pytest.mark.parametrize(
    "cls, cfg, seed, offset",
    [
        (BisectingKMeans, 1, 3, 32),
        (Birch, 1, 3, 32),
        (Nystroem, 1, 3, 32),
        (RBFSampler, 0, 1, 7),
    ],
)
def test_a_kernel_distance_nan_after_overflow_remains_a_strict_mismatch(
    cls, cfg, seed, offset
):
    step = _step(FIXTURES[cls][cfg], seed, cfg)
    rows = _rows(step, seed)
    row = rows.slice(rows.num_rows - 40 + offset, 1)
    if cls is RBFSampler:
        from sql_transform.native._check import _twin_input

        iid = row.column("__iid")[0].as_py()
        value = step.instances[iid].transform(_twin_input(step, row, 0))[0, 2]
        if not math.isfinite(value):
            pytest.skip("this host's BLAS also overflows the reported RBF angle")
    assert _gate_rows(step, row).num_rows == 0
    check(step, to_native(step, strict=True, allow_bound=True), row)


# Real fits only. NATIVE_WIDE=1 exercises the expensive widths nightly.
# Windows x64, 2026-10-17, four workers: the default ceilings took 37.13 s
# (183 passed, 89 skipped), below the 60 s added-gate target. No lower cap
# was needed. The wider run took 122 s: at 256 features the three encoders
# took 33.29--36.78 s each (0.70--1.21 s at 33). Encoder probes transform
# the full width for each feature/category. Other wider times are at 2048.
_LADDER_WIDTH_LIMITS = {
    FeatureAgglomeration: (256, "sklearn fits an O(n_features**2) distance matrix"),
    SequentialFeatureSelector: (
        33,
        "sklearn cross-validates once per candidate feature",
    ),
    OrdinalEncoder: (33, "33.29--36.78 s at 256: quadratic feature/category probes"),
    OneHotEncoder: (33, "33.29--36.78 s at 256: quadratic feature/category probes"),
    TargetEncoder: (33, "33.29--36.78 s at 256: quadratic feature/category probes"),
    PowerTransformer: (256, "4.61 s at 256: sklearn optimizes a lambda per feature"),
    Pipeline: (256, "25.40 s at 2048 features"),
    SkewedChi2Sampler: (256, "22.87 s at 2048 features"),
    Birch: (256, "16.32 s at 2048 features"),
    Nystroem: (256, "15.20 s at 2048 features"),
    KMeans: (256, "14.90 s at 2048 features"),
    KBinsDiscretizer: (256, "14.48 s at 2048 features"),
    SimpleImputer: (256, "13.99 s at 2048 features"),
    BisectingKMeans: (256, "13.42 s at 2048 features"),
    MiniBatchKMeans: (256, "13.14 s at 2048 features"),
    FeatureUnion: (256, "11.17 s at 2048 features"),
    PLSRegression: (256, "9.19 s at 2048 features"),
    CCA: (256, "9.11 s at 2048 features"),
    TruncatedSVD: (256, "8.93 s at 2048 features"),
    PLSCanonical: (256, "8.85 s at 2048 features"),
    RBFSampler: (256, "8.75 s at 2048 features"),
    LinearDiscriminantAnalysis: (256, "8.01 s at 2048 features"),
    FastICA: (256, "7.98 s at 2048 features"),
    FactorAnalysis: (256, "7.75 s at 2048 features"),
    PolynomialFeatures: (
        8192,
        "cheap fit; generated-term count refuses before allocation",
    ),
}
_FIT_MIN_WIDTHS = {
    FeatureAgglomeration: 2,
    RFE: 2,
    RFECV: 2,
    SequentialFeatureSelector: 2,
    TruncatedSVD: 2,
}
_NARROWEST = {
    Normalizer: 1,
    MissingIndicator: 1,
    SplineTransformer: 1,
    PCA: 1,
    Birch: 1,
}


def _ladder_step(cls, n, depth=0):
    est = FIXTURES[cls][_NARROWEST.get(cls, 0)]()
    params = est.get_params(deep=False)
    # These knobs affect fitting cost, not the translated transform form.
    changes = {}
    if "n_components" in params:
        changes["n_components"] = 1
    if "n_clusters" in params and params["n_clusters"] is not None:
        changes["n_clusters"] = 1
    if cls in (RFE, RFECV):
        changes["step"] = 0.99
    if "random_state" in params:
        changes["random_state"] = 0
    est.set_params(**changes)
    rng = np.random.default_rng(0)
    X = rng.lognormal(size=(12, n))
    y = np.arange(12) % 2
    X[:, 0] = y + 0.1 * rng.random(12) + 1
    est.fit(X, y)
    if isinstance(est, PolynomialFeatures):
        width = est.n_output_features_
    else:
        width = np.asarray(_dense(est.transform(X[:1]))).reshape(1, -1).shape[1]
    for _ in range(depth):
        est = Pipeline([("nested", est)])
    returns = pa.list_(pa.float64(), width) if width > 1 else pa.float64()
    return PythonTransform(
        "tf", {0: est}, pa.schema([(f"x{i}", pa.float64()) for i in range(n)]), returns
    )


@pytest.mark.parametrize(
    "cls, width, depth",
    [
        pytest.param(cls, n, 0, id=f"{cls.__name__}-{n}")
        for cls in FIXTURES
        for n in (1, 33, 256, 2048, 8192)
    ]
    + [pytest.param(Pipeline, 1, d, id=f"Pipeline-depth-{d}") for d in (10, 200)],
)
def test_translation_returns_a_function_or_only_not_native(cls, width, depth):
    from confit import Function

    if width < _FIT_MIN_WIDTHS.get(cls, 1):
        pytest.skip(f"sklearn {cls.__name__}.fit requires at least two features")
    if cls is IsotonicRegression and width > 1:
        pytest.skip("sklearn IsotonicRegression.fit accepts exactly one feature")
    if os.environ.get("NATIVE_WIDE", "0") != "1":
        limit, why = _LADDER_WIDTH_LIMITS.get(
            cls, (2048, "8192-feature nightly ladder")
        )
        if width > limit:
            pytest.skip(f"NATIVE_WIDE=1 exercises widths past {limit}: {why}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        step = _ladder_step(cls, width, depth)
    try:
        native = to_native(step, strict=True, allow_bound=True)
    except NotNative as e:
        assert str(e), "a refusal must name its reason"
    else:
        assert isinstance(native, Function)


def test_nested_exact_steps_do_not_rediscover_their_bounds_exponentially(monkeypatch):
    from confit import Function

    from sql_transform.native import compose

    original = compose._bounds_of
    calls = 0

    def counted(where, part):
        nonlocal calls
        calls += 1
        assert calls < 2000, "ten nested steps repeated bound discovery exponentially"
        return original(where, part)

    monkeypatch.setattr(compose, "_bounds_of", counted)
    assert isinstance(to_native(_ladder_step(Pipeline, 1, 10), strict=True), Function)


@pytest.mark.parametrize("depth", [0, 1, 200])
def test_a_polynomial_refuses_before_allocating_output_metadata(monkeypatch, depth):
    from sql_transform.native import _registry

    step = _ladder_step(PolynomialFeatures, 176, depth)

    def too_late(*args):
        raise AssertionError("output metadata was allocated before the refusal")

    monkeypatch.setattr(_registry, "_output_fields", too_late)
    with pytest.raises(NotNative, match="PolynomialFeatures.*15753 generated terms"):
        to_native(step, strict=True)


@pytest.mark.parametrize(
    "make",
    [
        lambda: FunctionTransformer(np.sin),
        lambda: FunctionTransformer(np.log),
        OrdinalEncoder,
        OneHotEncoder,
        TargetEncoder,
        Normalizer,
        QuantileTransformer,
        SplineTransformer,
        IsotonicRegression,
        lambda: SparseRandomProjection(n_components=1, random_state=0),
        lambda: PowerTransformer("box-cox"),
        PowerTransformer,
        AdditiveChi2Sampler,
        lambda: RBFSampler(n_components=1, random_state=0),
        lambda: SkewedChi2Sampler(n_components=1, random_state=0),
        lambda: Nystroem(n_components=1, random_state=0),
    ],
    ids=lambda make: repr(make()),
)
def test_every_probed_family_declines_when_confit_cannot_run(monkeypatch, make):
    from sql_transform.native import _registry, function, power

    X = np.arange(1.0, 13.0)[:, None]
    est = make().fit(X, np.arange(12) % 2)
    width = np.asarray(_dense(est.transform(X[:1]))).reshape(1, -1).shape[1]
    returns = pa.list_(pa.float64(), width) if width > 1 else pa.float64()
    step = PythonTransform("tf", {0: est}, pa.schema([("x0", pa.float64())]), returns)
    calls = []

    def refuse(*args, **kwargs):
        calls.append(1)
        raise OSError("injected engine failure")

    for module in (_registry, function, power):
        monkeypatch.setattr(module, "DuckDBInferFn", refuse)
    function.kernel_distance.cache_clear()
    power.kernel_distances.cache_clear()
    try:
        with pytest.raises(NotNative):
            to_native(step, strict=True, allow_bound=True)
        assert calls, "exercise the build/probe failure, not an unrelated refusal"
    finally:
        function.kernel_distance.cache_clear()
        power.kernel_distances.cache_clear()


@pytest.mark.parametrize(
    "cls", [OrdinalEncoder, OneHotEncoder, TargetEncoder, StandardScaler]
)
def test_a_failing_twin_probe_is_a_named_refusal(monkeypatch, cls):
    step = _ladder_step(cls, 1)
    calls = []

    def refuse(*args, **kwargs):
        calls.append(1)
        raise OSError("injected twin probe failure")

    monkeypatch.setattr(step.instances[0], "transform", refuse)
    with pytest.raises(NotNative):
        to_native(step, strict=True, allow_bound=True)
    assert calls


# ------------------------------------------------------------------ framework


def test_a_per_estimator_bound_needs_a_ceiling():
    from sql_transform.native import translates

    with pytest.raises(ValueError, match="needs a ceiling above 0"):
        translates(type("_Fake", (), {}), bound=lambda est: 0)


def test_a_bound_past_its_ceiling_raises():
    from sql_transform.native import Entry

    entry = Entry(lambda est, x, types: x, 2, lambda est: 3)
    assert entry.varies
    with pytest.raises(ValueError, match="past the class's ceiling of 2"):
        entry.bound(object())


def test_a_class_bound_reads_as_before():
    from sql_transform.native import Entry

    entry = Entry(lambda est, x, types: x, 4)
    assert not entry.varies
    assert entry.bound(object()) == 4


def _scaler_step() -> PythonTransform:
    return _step(StandardScaler, 0)


def test_a_step_without_a_translation_comes_back_unchanged():
    step = _step(lambda: KernelPCA(n_components=1), 0)
    assert to_native(step) is step
    assert "no translation for KernelPCA" in explain_native(step)
    with pytest.raises(NotNative, match="KernelPCA"):
        to_native(step, strict=True)


def test_a_step_nested_past_the_recursion_limit_stays_python():
    # `Normalizer(l2)` sums its squares in numpy's order, a chain of 1,024
    # additions over 2,048 features, and confit writes it by recursion
    # (finding 86).
    n = 2048
    est = Normalizer().fit(np.random.default_rng(0).normal(size=(4, n)))
    step = PythonTransform(
        "tf",
        {0: est},
        pa.schema([(f"x{i}", pa.float64()) for i in range(n)]),
        pa.list_(pa.float64(), n),
    )
    assert to_native(step) is step
    with pytest.raises(NotNative, match="nested past Python's recursion limit"):
        to_native(step, strict=True)


def test_to_native_is_idempotent():
    native = to_native(_scaler_step(), strict=True)
    assert to_native(native) is native


def test_a_mixed_step_stays_python():
    step = _scaler_step()
    X = np.random.default_rng(1).normal(size=(10, len(step.takes)))
    step.instances[len(step.instances)] = KernelPCA(n_components=1).fit(X)
    assert to_native(step) is step


def test_an_unknown_id_raises_as_the_twin_does():
    from confit.oracle import Oracle

    from sql_transform.native._check import _serve
    from sql_transform.native._registry import query

    step = _scaler_step()
    native = to_native(step, strict=True)
    rows = _rows(step, 0)
    rows = rows.set_column(0, "__iid", pa.array([99] * rows.num_rows, pa.int64()))
    sql = query(step)
    assert all(isinstance(a, Exception) for a in _serve(sql, rows, step))
    got = _serve(sql, rows, native)
    assert all("not in the fitted instances" in str(b) for b in got)
    with Oracle() as o:
        native.register(o)
        o.load("__THIS__", rows)
        assert "not in the fitted instances" in str(o.try_answer(sql))


@pytest.mark.parametrize("returns", ["struct", "list"])
@pytest.mark.parametrize("instances", [1, 2])
def test_an_output_field_of_one_arm_reads_no_id(returns, instances):
    # Past the first output field, an output field that is one arm (one
    # instance, or equal fits) is its expression alone: the body reads the
    # id as often at any width. The first output field still answers each id
    # as the twin does, on a read of the last output field too.
    from confit.oracle import Oracle

    from sql_transform.native._check import _serve

    def make(width: int) -> PythonTransform:
        X = np.random.default_rng(0).normal(size=(20, width))
        r = (
            pa.struct([(f"f{j}", pa.float64()) for j in range(width)])
            if returns == "struct"
            else pa.list_(pa.float64(), width)
        )
        takes = pa.schema([(f"x{i}", pa.float64()) for i in range(width)])
        fits = {k: StandardScaler().fit(X) for k in range(instances)}
        return PythonTransform("tf", fits, takes, r)

    def rows(ids: list[int | None]) -> pa.Table:
        x = pa.array([0.5] * len(ids))
        return pa.table({"__iid": pa.array(ids, pa.int64()), "x0": x, "x1": x, "x2": x})

    reads = [to_native(make(w), strict=True).sql_body.count('"__iid"') for w in (3, 6)]
    assert reads[0] == reads[1]
    step = make(3)
    native = to_native(step, strict=True)
    assert check(step, native, rows([0, None, instances - 1])) == 2
    last = ".f2" if returns == "struct" else "[3]"
    sql = f"SELECT tf(__iid, x0, x1, x2){last} AS o FROM __THIS__"
    unknown = rows([99])
    assert "not in the fitted instances" in str(_serve(sql, unknown, native))
    with Oracle() as o:
        native.register(o)
        o.load("__THIS__", unknown)
        assert "not in the fitted instances" in str(o.try_answer(sql))


@pytest.mark.parametrize(
    "make",
    [
        lambda: OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
        lambda: OneHotEncoder(sparse_output=False, handle_unknown="ignore"),
        TargetEncoder,
    ],
    ids=["OrdinalEncoder", "OneHotEncoder", "TargetEncoder"],
)
def test_an_encoder_reads_a_string_feature_missing_at_fit(make):
    # Its categories are [None], which says nothing of its kind: the
    # declared type does (read as a number, it once answered wrongly).
    X = np.empty((8, 2), dtype=object)
    X[:, 0] = None
    X[:, 1] = ["a", "b", "a", "b", None, "a", "b", "a"]
    est = make().fit(X, np.array([0, 1, 0, 1, 0, 1, 1, 0]))
    width = np.asarray(_dense(est.transform(X[:1]))).shape[1]
    step = PythonTransform(
        "tf",
        {0: est},
        pa.schema([("x0", pa.string()), ("x1", pa.string())]),
        pa.struct([(f"f{i}", pa.float64()) for i in range(width)]),
    )
    rows = pa.table(
        {
            "__iid": pa.array([0] * 5, pa.int64()),
            "x0": pa.array([None, "a", "zz", None, "b"], pa.string()),
            "x1": pa.array(["a", None, "b", "zz", "a"], pa.string()),
        }
    )
    assert check(step, to_native(step, strict=True), rows) == 5


@pytest.mark.parametrize(
    "make, answered",
    [
        (
            lambda: OrdinalEncoder(
                handle_unknown="use_encoded_value", unknown_value=-1
            ),
            10,
        ),
        (
            lambda: OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1,
                encoded_missing_value=-2,
            ),
            10,
        ),
        (OrdinalEncoder, 4),
    ],
    ids=["unknown", "missing", "raises"],
)
def test_an_ordinal_number_field_answers_past_its_ends_as_its_else(make, answered):
    # Over a number feature, an ordinal field of more than one arm first
    # tests the ends of the classes (`encode._field`): a number past them
    # answers as the ELSE does, and NaN, which DuckDB orders above every
    # number, goes on to the arms. Where the twin raises (on an infinity,
    # or on an unknown number by default), the input guard traps.
    est = make().fit(np.array([[1.0], [2.0], [3.0], [np.nan], [-1.0], [2.0]]))
    step = PythonTransform(
        "tf",
        {0: est},
        pa.schema([("x0", pa.float64())]),
        pa.struct([("f0", pa.float64())]),
    )
    values = [-1.0, 3.0, 2.5, -1.5, 3.5, -math.inf, math.inf, math.nan, None]
    values += [0.0, -0.0, 1e300]
    rows = pa.table(
        {
            "__iid": pa.array([0] * len(values), pa.int64()),
            "x0": pa.array(values, pa.float64()),
        }
    )
    assert check(step, to_native(step, strict=True), rows) == answered


def _encoder_fit(beside: str | None, hole: bool) -> tuple[np.ndarray, list]:
    """Two boolean columns (the first constant True), and a third beside
    them: none, a number or a string; with a hole (NaN) in the second."""
    bits = np.array([[True, False], [True, True], [True, False], [True, True]] * 2)
    cols: list[Any] = [bits[:, 0], bits[:, 1].astype(float) if hole else bits[:, 1]]
    if hole:
        cols[1][3] = np.nan
    types = [pa.bool_(), pa.bool_()]
    if beside is None:
        return np.column_stack(cols), types
    if beside == "number":
        cols.append(np.arange(8) % 3 - 1.0)
        return np.column_stack(cols), [*types, pa.float64()]
    X = np.empty((8, 3), dtype=object)
    for j in (0, 1):
        X[:, j] = [v if v != v else bool(v) for v in cols[j]]
    X[:, 2] = ["a", "b"] * 4
    return X, [*types, pa.string()]


@pytest.mark.parametrize(
    "beside, hole, dtype",
    [
        (None, False, np.bool_),
        (None, True, np.float64),
        ("number", False, np.float64),
        ("string", False, object),
        ("string", True, object),
    ],
    ids=["bool", "bool-hole", "float", "object", "object-hole"],
)
@pytest.mark.parametrize(
    "make",
    [
        OrdinalEncoder,
        lambda: OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            encoded_missing_value=-2,
        ),
        lambda: OneHotEncoder(sparse_output=False, handle_unknown="ignore"),
        lambda: OneHotEncoder(sparse_output=False, drop="if_binary"),
        TargetEncoder,
    ],
    ids=["ordinal", "ordinal-unknown", "onehot", "onehot-if-binary", "target"],
)
def test_an_encoder_reads_a_boolean_feature(make, beside, hole, dtype):
    # Its categories are booleans, doubles or objects as the fit matrix
    # is; the step hands False, True or NaN, and the first feature's False
    # was never fitted.
    X, types = _encoder_fit(beside, hole)
    est = make().fit(X, np.array([0, 1] * 4))
    assert est.categories_[1].dtype == dtype
    width = np.asarray(_dense(est.transform(X[:1]))).shape[1]
    step = PythonTransform(
        "tf",
        {0: est},
        pa.schema([(f"x{i}", t) for i, t in enumerate(types)]),
        pa.struct([(f"f{i}", pa.float64()) for i in range(width)]),
    )
    cols = {
        "__iid": pa.array([0] * 6, pa.int64()),
        "x0": pa.array([True, False, None, True, False, True]),
        "x1": pa.array([False, True, True, None, False, True]),
    }
    if beside == "number":
        cols["x2"] = pa.array([0.0, 1.0, -1.0, None, 7.0, 0.0])
    elif beside == "string":
        cols["x2"] = pa.array(["a", "b", None, "a", "zz", "b"])
    native = to_native(step, strict=True)
    assert check(step, native, pa.table(cols)) > 0


@pytest.mark.parametrize(
    "make",
    [
        lambda: OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
        lambda: OneHotEncoder(sparse_output=False, handle_unknown="ignore"),
        lambda: TargetEncoder(target_type="continuous"),
    ],
    ids=["OrdinalEncoder", "OneHotEncoder", "TargetEncoder"],
)
def test_an_encoder_reads_a_boolean_row_by_its_dtype(make):
    # A first feature fitted only missing (categories [nan]): True is
    # unknown, which a float64 row (a NULL in it) answers and a boolean
    # row (none) may not; the entry follows each.
    X = np.array([[np.nan, 1.0], [np.nan, 0.0], [np.nan, 1.0], [np.nan, 0.0]])
    est = make().fit(X, np.array([0.5, 1.5, 2.0, -1.0]))
    width = np.asarray(_dense(est.transform(X[:1]))).shape[1]
    takes = pa.schema([("x0", pa.bool_()), ("x1", pa.bool_())])
    step = PythonTransform("tf", {0: est}, takes, pa.list_(pa.float64(), width))
    rows = pa.table(
        {
            "__iid": pa.array([0] * 5, pa.int64()),
            "x0": pa.array([True, True, None, False, True]),
            "x1": pa.array([False, None, True, None, True]),
        }
    )
    assert check(step, to_native(step, strict=True), rows) >= 3


@pytest.mark.parametrize("held", [0, 1, 4], ids=["free", "held-one", "held-all"])
@pytest.mark.parametrize("kind", ["ordinal", "ordinal-missing", "onehot"])
def test_an_ordinal_guard_finds_many_strings_by_one_search(monkeypatch, kind, held):
    # Past SEARCH_PAST strings (lowered here), the input guard of an
    # encoder with one output field a feature finds a value among them by
    # one substring search of the strings joined by a separator that none
    # holds; where a string holds the first separator, the next; where one
    # holds them all, an IN list. A one-hot encoder keeps the IN list. A
    # part or a join of the strings, or a value holding a separator, is
    # none of them; NULL is one where it was fitted.
    from sql_transform.native import encode

    monkeypatch.setattr(encode, "SEARCH_PAST", 4)
    holds = {0: [], 1: ["q\x1fr"], 4: ["q" + "".join(encode._SEPARATORS)]}[held]
    cats = ["a", "b", "", "a b", "日本", "ab", "x%y", *holds]
    X = np.array(
        [[c] for c in cats + (["a", None] if kind == "ordinal-missing" else [])]
    )
    X = X.astype(object)
    est = (
        OneHotEncoder(sparse_output=False) if kind == "onehot" else OrdinalEncoder()
    ).fit(X)
    width = np.asarray(_dense(est.transform(X[:1]))).shape[1]
    step = PythonTransform(
        "tf",
        {0: est},
        pa.schema([("x0", pa.string())]),
        pa.struct([(f"f{i}", pa.float64()) for i in range(width)]),
    )
    native = to_native(step, strict=True)
    assert ("contains(" in native.sql_body) == (kind != "onehot" and held < 4)
    values = [
        *[*cats, None, "zz", "A", "日", "q", "r", "ba", "a b "],
        *["a\x1fb", "\x1fa\x1f", "\x1f", "a\x1eb", "\x1ea\x1e", "a\x1f"],
    ]
    rows = pa.table(
        {
            "__iid": pa.array([0] * len(values), pa.int64()),
            "x0": pa.array(values, pa.string()),
        }
    )
    assert check(step, native, rows) >= len(cats)


@pytest.mark.parametrize("kind", ["ordinal", "ordinal-missing", "onehot"])
def test_an_encoder_guard_finds_a_run_of_integers_by_a_range(kind):
    # Numbers in a run of RUN_FROM or more consecutive integers are found
    # by `a <= x <= b` and `floor(x) = x`, the others by equality. A number
    # between two of a run's, a fraction, an integer just past a run's end,
    # NaN and ±inf are none of them; -0.0 is 0.0, and NaN is one where it
    # was fitted.
    cats = [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 7.0, 9.0, 10.0, 11.0, 12.0, 2.5]
    cats.append(2.0**53)
    X = np.array([[c] for c in cats + ([np.nan] if kind == "ordinal-missing" else [])])
    est = (
        OneHotEncoder(sparse_output=False) if kind == "onehot" else OrdinalEncoder()
    ).fit(X)
    width = np.asarray(_dense(est.transform(X[:1]))).shape[1]
    step = PythonTransform(
        "tf",
        {0: est},
        pa.schema([("x0", pa.float64())]),
        pa.struct([(f"f{i}", pa.float64()) for i in range(width)]),
    )
    native = to_native(step, strict=True)
    assert native.sql_body.count("floor(") == 1
    values = [
        *cats,
        *[None, math.nan, math.inf, -math.inf, 1e300, -0.0, 0.5, -2.5],
        *[1e-300, -5e-324, 2.5 + 2**-51, -4.0, 3.0, 6.0, 8.0, 13.0],
        *[2.0**53 - 1, 2.0**53 + 2],
    ]
    rows = pa.table(
        {
            "__iid": pa.array([0] * len(values), pa.int64()),
            "x0": pa.array(values, pa.float64()),
        }
    )
    assert check(step, native, rows) >= len(cats) + 1


def test_a_onehot_output_field_tests_its_category_alone():
    # Under handle_unknown="error" the ELSE answers the largest group, 0:
    # every other value it meets traps in the input guard, whose one test
    # holds the body's one IN list.
    X = np.array([[f"c{i}"] for i in range(12)], dtype=object)
    est = OneHotEncoder(sparse_output=False).fit(X)
    step = PythonTransform(
        "tf",
        {0: est},
        pa.schema([("x0", pa.string())]),
        pa.struct([(f"f{i}", pa.float64()) for i in range(12)]),
    )
    native = to_native(step, strict=True)
    assert native.sql_body.count(" IN (") == 1
    values = [f"c{i}" for i in range(12)] + [None, "zz", "c1 c2", ""]
    rows = pa.table(
        {
            "__iid": pa.array([0] * len(values), pa.int64()),
            "x0": pa.array(values, pa.string()),
        }
    )
    assert check(step, native, rows) == 12


@pytest.mark.parametrize(
    "make",
    [OrdinalEncoder, lambda: OneHotEncoder(sparse_output=False)],
    ids=["OrdinalEncoder", "OneHotEncoder"],
)
def test_an_encoder_refuses_a_feature_whose_every_value_raises(make):
    # A string feature fitted only on NaN: the step hands None, which
    # sklearn does not match to NaN, so every call raises in Python too.
    X = np.empty((4, 2), dtype=object)
    X[:, 0] = np.nan
    X[:, 1] = ["a", "b", "a", "b"]
    est = make().fit(X)
    width = np.asarray(_dense(est.transform(X[:1]))).shape[1]
    step = PythonTransform(
        "tf",
        {0: est},
        pa.schema([("x0", pa.string()), ("x1", pa.string())]),
        pa.struct([(f"f{i}", pa.float64()) for i in range(width)]),
    )
    with pytest.raises(NotNative, match="raises on every value of feature 0"):
        to_native(step, strict=True)


@pytest.mark.parametrize("beside", [None, "number", "string"])
@pytest.mark.parametrize(
    "make",
    [
        OrdinalEncoder,
        lambda: OneHotEncoder(sparse_output=False, handle_unknown="ignore"),
        TargetEncoder,
    ],
    ids=["ordinal", "onehot", "target"],
)
def test_an_encoder_answers_a_probe_row_as_alone(make, beside):
    # `_probes` hands `transform` a feature's probe rows in one call and,
    # where it raises, each half in a call of its own: each row answers as
    # in a call of its own, the rows that raise (an unfitted False or NaN,
    # an unknown value under OrdinalEncoder) included.
    from sql_transform.native.encode import (
        _classes,
        _first_classes,
        _probe,
        _probes,
        _unknown,
    )

    X, types = _encoder_fit(beside, hole=True)
    est = make().fit(X, np.array([0, 1] * 4))
    base = _first_classes(est, types)
    dtype = np.float64 if beside is None else None
    raised = 0
    for j, t in enumerate(types):
        values = _classes(est.categories_[j], t)
        if t != pa.bool_():
            values = [*values, _unknown(values, t == pa.string())]
        alone = [_probe(est, base, j, v, dtype) for v in values]
        assert repr(_probes(est, base, j, values, dtype)) == repr(alone)
        raised += alone.count(None)
    assert raised > 0 or make is not OrdinalEncoder


def test_a_null_id_is_a_null_struct():
    # The whole struct, as DuckDB reads it from each definition (confit
    # serves field reads, which are NULL either way).
    from confit.oracle import Oracle

    step = _step(StandardScaler, 4)
    assert pa.types.is_struct(step.returns)
    args = ", ".join(["__iid", *step.takes.names])
    rows = _rows(step, 4).slice(0, 2)
    rows = rows.set_column(0, "__iid", pa.array([None, 0], pa.int64()))
    answers = []
    for fn in (step, to_native(step, strict=True)):
        with Oracle() as o:
            fn.register(o)
            o.load("__THIS__", rows)
            answers.append(o.answer(f"SELECT tf({args}) AS s FROM __THIS__"))
    twin, native = (a.column("s").to_pylist() for a in answers)
    assert twin[0] is None and native[0] is None
    assert native[1] is not None and twin[1].keys() == native[1].keys()


def test_explain_names_the_kind():
    assert explain_native(_scaler_step()) == "'tf': StandardScaler -> SqlFunction"


def test_a_reordered_translation_is_caught(monkeypatch):
    # `x * (1/s)` is one rounding off `x / s` on some inputs: the bit-exact
    # bound must see it.
    from sql_transform.native import _registry
    from sql_transform.native._helpers import f64

    def reordered(est, x, types):
        terms = zip(x, est.mean_, est.scale_, strict=True)
        return [(xi - f64(m)) * f64(1.0 / s) for xi, m, s in terms]

    entry = _registry._CATALOG[StandardScaler]
    monkeypatch.setitem(
        _registry._CATALOG, StandardScaler, _registry.Entry(reordered, 0)
    )
    caught = 0
    for seed in range(8):
        step = _step(StandardScaler, seed)
        try:
            check(step, to_native(step, strict=True), _rows(step, seed))
        except AssertionError as e:
            assert "ulps, bound 0" in str(e)
            caught += 1
    assert caught, "a one-rounding reorder passed the bit-exact bound"
    assert entry.ulps == 0


# ------------------------------------------------------------ KBinsDiscretizer


def _kbins_step(est: Any) -> PythonTransform:
    width = np.asarray(_dense(est.transform(np.zeros((1, 2))))).shape[1]
    return PythonTransform(
        "tf",
        {0: est},
        pa.schema([("x0", pa.float64()), ("x1", pa.float64())]),
        pa.struct([(f"f{i}", pa.float64()) for i in range(width)]),
    )


@pytest.mark.parametrize("encode", ["ordinal", "onehot-dense"])
def test_kbins_counts_equal_edges_as_numpy_does(encode):
    # "uniform" keeps every edge of np.linspace, so a range a few doubles
    # wide repeats edges; served on the edges themselves and their
    # neighbours. The second feature's edges straddle zero, served with
    # both signed zeros.
    lo = 1.0
    hi = float(np.nextafter(np.nextafter(lo, 2.0), 2.0))
    X = np.array([[lo, -2.0], [hi, 2.0], [lo, 0.5]])
    est = KBinsDiscretizer(n_bins=8, encode=encode, strategy="uniform").fit(X)
    edges = est.bin_edges_[0][1:-1]
    assert len(set(edges)) < len(edges), "no repeated edge"
    assert 0.0 in est.bin_edges_[1]
    probes = sorted(
        {float(v) for e in edges for v in (np.nextafter(e, -9), e, np.nextafter(e, 9))}
    )
    zeros = [0.0, -0.0, 5e-324, -5e-324, 0.5, -0.5, 2.0, -2.0]
    n = max(len(probes), len(zeros))
    rows = pa.table(
        {
            "__iid": pa.array([0] * n, pa.int64()),
            "x0": pa.array([probes[i % len(probes)] for i in range(n)]),
            "x1": pa.array([zeros[i % len(zeros)] for i in range(n)]),
        }
    )
    step = _kbins_step(est)
    assert check(step, to_native(step, strict=True), rows) == n


# --------------------------------------------------------- FunctionTransformer

# Doubles where an elementwise function's spelling can part from numpy's:
# signed zeros, subnormals, halves (rint), the largest non-integer doubles,
# infinities, NaN (and NULL, read as NaN), near pi, and where exp overflows
# (past 709.78) and underflows to a subnormal and to 0 (past -745.13).
SPECIALS = [
    0.0, -0.0, 5e-324, -5e-324, 2.2250738585072014e-308, -1e-310, 0.5, -0.5,
    1.5, -1.5, 2.5, -2.5, 0.49999999999999994, -0.49999999999999994, 1.0, -1.0,
    4503599627370495.5, -4503599627370495.5, 4503599627370497.0, 1e300, -1e300,
    math.inf, -math.inf, math.nan, None, 3.141592653589793, -7.25, 1e-300,
    709.78, 709.79, 710.0, -740.0, -745.2, -746.0, 1e308, -1.7976931348623157e308,
]  # fmt: skip


@pytest.mark.parametrize("func", FUNCTIONS, ids=lambda f: f.__name__)
def test_a_function_matches_numpy_on_special_values(func):
    est = FunctionTransformer(func).fit(np.zeros((2, 1)))
    step = PythonTransform("tf", {0: est}, pa.schema([("x0", pa.float64())]))
    rows = pa.table(
        {
            "__iid": pa.array([0] * len(SPECIALS), pa.int64()),
            "x0": pa.array(SPECIALS, pa.float64()),
        }
    )
    native = to_native(step, strict=True, allow_bound=True)
    assert check(step, native, rows) == len(SPECIALS)


def test_the_identity_passes_a_boolean_as_its_double():
    est = FunctionTransformer(validate=True).fit(np.zeros((2, 2)))
    takes = pa.schema([("x0", pa.bool_()), ("x1", pa.float64())])
    step = PythonTransform("tf", {0: est}, takes, pa.list_(pa.float64(), 2))
    rows = pa.table(
        {
            "__iid": pa.array([0, 0, 0], pa.int64()),
            "x0": pa.array([True, False, True]),
            "x1": pa.array([1.5, -0.0, 2.0]),
        }
    )
    assert check(step, to_native(step, strict=True), rows) == 3


@pytest.mark.parametrize(
    "func", [np.abs, np.square, np.sqrt, np.floor, np.log, np.sin, np.negative]
)
@pytest.mark.parametrize("beside", [False, True], ids=["booleans", "beside-double"])
def test_a_function_over_booleans_answers_as_numpy(func, beside):
    # Over booleans only, a row none NULL is a boolean array: served where
    # numpy answers it as it answers 0.0 and 1.0. Beside a double (or with
    # a NULL) the row is float64, and every served function serves.
    est = FunctionTransformer(func)
    if not beside and func in (np.sin, np.negative):
        with pytest.raises(NotNative, match="over boolean features only"):
            to_native(_bool_step(est, beside), strict=True)
        return
    step = _bool_step(est, beside)
    rows = {
        "__iid": pa.array([0, 0, 0, 0, None], pa.int64()),
        "x0": pa.array([True, False, None, True, False]),
        "x1": pa.array([False, False, True, None, True]),
    }
    if beside:
        rows["x2"] = pa.array([2.5, -0.0, None, 4.0, 1.0])
    native = to_native(step, strict=True, allow_bound=True)
    assert check(step, native, pa.table(rows)) == 4


def _bool_step(est: Any, beside: bool) -> PythonTransform:
    types = [pa.bool_(), pa.bool_(), *([pa.float64()] if beside else [])]
    est.fit(np.zeros((2, len(types))))
    takes = pa.schema([(f"x{i}", t) for i, t in enumerate(types)])
    return PythonTransform("tf", {0: est}, takes, pa.list_(pa.float64(), len(types)))


def _sum_rows(X):
    return np.asarray(X, dtype=float) + 1.0


@pytest.mark.parametrize(
    "est, types, reason",
    [
        (FunctionTransformer(np.log1p), None, "DuckDB has no log1p"),
        (FunctionTransformer(np.expm1), None, "DuckDB has no expm1"),
        (FunctionTransformer(np.arctan), None, r"func=np\.arctan\): not a function"),
        (FunctionTransformer(lambda X: X), None, r"func=<lambda>\): not a function"),
        (FunctionTransformer(_sum_rows), None, r"func=_sum_rows\): not a function"),
        (
            FunctionTransformer(functools.partial(np.round, decimals=0)),
            None,
            r"func=functools\.partial.*not a function",
        ),
        (FunctionTransformer(np.round), None, r"func=np\.round\): not a function"),
        (
            FunctionTransformer(np.abs, kw_args={"dtype": np.float32}),
            None,
            "kw_args",
        ),
        (FunctionTransformer(), [pa.string()], "string feature"),
        (
            FunctionTransformer(np.negative),
            [pa.bool_()],
            r"func=np\.negative\) over boolean features only: numpy raises TypeError",
        ),
        (
            FunctionTransformer(np.sin),
            [pa.bool_(), pa.bool_()],
            r"over boolean features only: it answers float16 \[0\.0, 0\.8413",
        ),
        (
            FunctionTransformer(np.reciprocal, validate=True),
            [pa.bool_()],
            r"over boolean features only: it answers int8 \[0, 1\], not \[inf, 1\.0\]",
        ),
    ],
    ids=lambda v: None,
)
def test_a_function_transformer_refuses(est, types, reason):
    types = types or [pa.float64()]
    est.fit(np.zeros((2, len(types))))
    takes = pa.schema([(f"x{i}", t) for i, t in enumerate(types)])
    step = PythonTransform("tf", {0: est}, takes)
    with pytest.raises(NotNative, match=reason):
        to_native(step, strict=True)


@pytest.mark.parametrize("func", [np.sin, *BOUNDED], ids=lambda f: f.__name__)
def test_a_failing_kernel_probe_leaves_the_function_python(monkeypatch, func):
    from sql_transform.native import function

    def refuse(*args, **kwargs):
        raise RuntimeError("no engine")

    monkeypatch.setattr(function, "DuckDBInferFn", refuse)
    function.kernel_distance.cache_clear()
    try:
        est = FunctionTransformer(func).fit(np.zeros((2, 1)))
        step = PythonTransform("tf", {0: est}, pa.schema([("x0", pa.float64())]))
        with pytest.raises(NotNative, match="the kernel probe did not run"):
            to_native(step, strict=True)
    finally:
        function.kernel_distance.cache_clear()


@pytest.mark.parametrize(
    "func, distance, reason",
    [
        (np.sin, 1, "1 ulps from confit's on the probe, past the entry's bound of 0"),
        (np.exp, 2, "2 ulps from confit's on the probe, past the entry's bound of 1"),
        (np.log10, 3, "3 ulps .* past the entry's bound of 2"),
        (np.tan, 1 << 64, "past the entry's bound of 1"),
        (np.cbrt, 4, "4 ulps .* past the entry's bound of 3"),
    ],
    ids=lambda v: None,
)
def test_a_kernel_past_its_bound_leaves_the_function_python(
    monkeypatch, func, distance, reason
):
    # numpy picks its kernel by CPU: one further from confit's than the
    # bound measured here is not served.
    from sql_transform.native import function

    monkeypatch.setattr(function, "kernel_distance", lambda f: distance)
    est = FunctionTransformer(func).fit(np.zeros((2, 1)))
    step = PythonTransform("tf", {0: est}, pa.schema([("x0", pa.float64())]))
    with pytest.raises(NotNative, match=reason):
        to_native(step, strict=True)


def test_this_platforms_kernels_are_within_their_bounds():
    from sql_transform.native import function

    for func in function._PROBED:
        est = FunctionTransformer(func).fit(np.zeros((2, 1)))
        assert function.kernel_distance(func) <= bound_of(est), func.__name__


def _kernels_part(monkeypatch, distance: int | None = None) -> None:
    """Pin the kernel probe: each function `distance` ulps from confit's,
    by default its own bound, as on x86-64 with AVX-512. A bounded
    function's bound is 0 where the probe reads 0, so a test of the bound
    above 0 pins it rather than read this platform's numpy."""
    from sql_transform.native import function

    monkeypatch.setattr(
        function,
        "kernel_distance",
        lambda f: function._BOUNDS.get(f, 0) if distance is None else distance,
    )


@pytest.mark.parametrize(
    "funcs, ulps",
    [
        ([None, np.sqrt, np.sin], 0),
        ([np.exp], 1),
        ([None, np.log10], 2),
        ([np.exp, np.log10, np.abs], 2),
        ([None, np.cbrt, np.exp], 3),
    ],
    ids=lambda v: None,
)
def test_a_step_is_held_to_its_loosest_instance(monkeypatch, funcs, ulps):
    _kernels_part(monkeypatch)
    instances = {
        k: FunctionTransformer(f).fit(np.zeros((2, 1))) for k, f in enumerate(funcs)
    }
    step = PythonTransform("tf", instances, pa.schema([("x0", pa.float64())]))
    assert bound(step) == ulps


def test_a_bounded_function_is_held_to_its_bound_not_to_0():
    # numpy picks its log10 kernel by CPU: on x86-64 with AVX-512 it parts
    # from DuckDB's (glibc's) on some of these rows, elsewhere it may not.
    # The check at 0 fails exactly where numpy and the native answer part;
    # at the function's own bound it passes either way.
    from confit import DuckDBInferFn

    from sql_transform.native import function
    from sql_transform.native._registry import query

    est = FunctionTransformer(np.log10).fit(np.zeros((2, 1)))
    step = PythonTransform("tf", {0: est}, pa.schema([("x0", pa.float64())]))
    x = np.random.default_rng(0).uniform(1e-3, 1e3, 2000)
    rows = pa.table({"__iid": pa.array([0] * len(x), pa.int64()), "x0": x})
    native = to_native(step, strict=True, allow_bound=True)
    # Where the probe reads 0, numpy's kernel is glibc's and the bound is 0.
    d = function.kernel_distance(np.log10)
    assert bound_of(est) == (2 if d else 0)
    assert check(step, native, rows) == len(x)
    served = DuckDBInferFn(
        query(step),
        row_tables={"__THIS__": rows.schema},
        static_tables={},
        udfs=[native],
    ).infer_arrow(rows)
    got = served.column("o").to_numpy()
    parts = not np.array_equal(np.log10(x).view(np.int64), got.view(np.int64))
    if parts:
        with pytest.raises(ParityError, match="bound 0"):
            check(step, native, rows, ulps=0)
    else:
        assert check(step, native, rows, ulps=0) == len(x)


# --------------------------------------------- a bound above 0 (allow_bound)

# What `to_native` raises under `strict` for a step within a bound above 0
# (decisions/closed/matvec-parity-bound.md, the ruling's amendment).
ON_REQUEST = (
    "of its twin, not bit-exact; to_native serves a bound above 0 only with"
    " allow_bound=True"
)


@pytest.mark.parametrize(
    "j, k",
    [(0, 5), (2, 10), (5, 7), (7, 12)],
    ids=["box-cox", "yeo-johnson", "box-cox-standardized", "yeo-johnson-standardized"],
)
def test_a_power_transform_serves_only_on_request(j, k):
    make = FIXTURES[PowerTransformer][j]
    step = _step(make, 0, j)
    within = f"within {k}·eps·S + τ (S its error scale)"
    assert to_native(step) is step
    with pytest.raises(
        NotNative, match=re.escape(f"PowerTransformer is {within} ") + ON_REQUEST
    ):
        to_native(step, strict=True)
    assert "stays Python" in explain_native(step)
    native = to_native(step, strict=True, allow_bound=True)
    rows = _rows(step, 0, positive=getattr(make, "positive", False))
    assert check(step, native, rows) > 0
    assert explain_native(step, allow_bound=True).endswith(f"-> SqlFunction, {within}")


@pytest.mark.parametrize("func", BOUNDED, ids=lambda f: f.__name__)
def test_a_bounded_function_serves_only_on_request(monkeypatch, func):
    from sql_transform.native import function

    _kernels_part(monkeypatch)
    b = function._BOUNDS[func]
    est = FunctionTransformer(func).fit(np.zeros((2, 1)))
    step = PythonTransform("tf", {0: est}, pa.schema([("x0", pa.float64())]))
    assert bound_of(est) == b
    assert to_native(step) is step
    with pytest.raises(
        NotNative, match=f"instance 0: FunctionTransformer is within {b} ulps"
    ):
        to_native(step, strict=True)
    assert to_native(step, strict=True, allow_bound=True) is not step


@pytest.mark.parametrize("func", BOUNDED, ids=lambda f: f.__name__)
def test_a_bounded_function_whose_kernel_reads_0_serves_by_default(monkeypatch, func):
    # numpy's kernel is glibc's there, as DuckDB's and confit's are: the
    # function is bit-exact on this platform, and so composes too.
    _kernels_part(monkeypatch, 0)
    est = FunctionTransformer(func).fit(np.zeros((2, 1)))
    step = PythonTransform("tf", {0: est}, pa.schema([("x0", pa.float64())]))
    assert bound_of(est) == 0
    assert to_native(step, strict=True) is not step
    pipe = make_pipeline(FunctionTransformer(func), StandardScaler())
    pipe.fit(np.ones((3, 1)) + np.arange(3.0)[:, None])
    composed = PythonTransform("p", {0: pipe}, pa.schema([("x0", pa.float64())]))
    assert to_native(composed, strict=True) is not composed


def test_a_step_serves_on_request_by_its_loosest_instance(monkeypatch):
    _kernels_part(monkeypatch)
    funcs = [None, np.sqrt, np.exp, np.log10, np.log10]
    instances = {
        k: FunctionTransformer(f).fit(np.zeros((2, 1))) for k, f in enumerate(funcs)
    }
    step = PythonTransform("tf", instances, pa.schema([("x0", pa.float64())]))
    with pytest.raises(
        NotNative, match="instance 3: FunctionTransformer is within 2 ulps"
    ):
        to_native(step, strict=True)
    exact = {k: est for k, est in instances.items() if k < 2}
    step = PythonTransform("tf", exact, pa.schema([("x0", pa.float64())]))
    assert to_native(step, strict=True) is not step


def test_a_refused_configuration_says_why_before_its_bound():
    # A power transform with a lambda not finite stays Python: allow_bound
    # does not serve it, so the refusal names the configuration, not the
    # bound.
    step = _step(positive(_pinned("box-cox", [math.inf])), 0)
    for allow in (False, True):
        with pytest.raises(NotNative, match="a lambda not finite"):
            to_native(step, strict=True, allow_bound=allow)


# ----------------------------------------------------------- SplineTransformer


def _knot_rows(est: SplineTransformer, n_features: int) -> pa.Table:
    """Rows on every knot of each feature's spline and one double either
    side, with the fit range's ends, signed zeros, extremes and NaN."""
    cols = []
    for j in range(n_features):
        t = np.asarray(est.bsplines_[j].t, dtype=np.float64)
        vals = sorted(
            {
                float(w)
                for v in t
                for w in (np.nextafter(v, -np.inf), v, np.nextafter(v, np.inf))
            }
        )
        cols.append([*vals, 0.0, -0.0, 1e300, -1e300, 5e-324, math.nan])
    n = max(len(c) for c in cols)
    table = {"__iid": pa.array([0] * n, pa.int64())}
    for j, c in enumerate(cols):
        table[f"x{j}"] = pa.array([c[i % len(c)] for i in range(n)], pa.float64())
    return pa.table(table)


SPLINE_AT_KNOTS = [
    # Without the bias, one spline per feature leaves no output field: kept
    # there.
    {
        "degree": d,
        "n_knots": m,
        "extrapolation": e,
        "knots": kn,
        "include_bias": b or (m - 1 if e == "periodic" else m + d - 1) == 1,
    }
    for d, m in [(0, 2), (0, 5), (1, 2), (1, 4), (2, 3), (3, 5), (3, 8), (5, 6)]
    for e in ["continue", "error", "periodic", "constant", "linear"]
    for kn, b in [("uniform", True), ("quantile", False)]
    if not (e == "periodic" and m <= d)
    and not (e == "linear" and (d, m) == (0, 2))  # refused, below
]


@pytest.mark.parametrize(
    "params, sparse",
    [
        pytest.param(
            p,
            sparse,
            id="-".join([*(str(v) for v in p.values()), "sparse" * sparse or "dense"]),
        )
        for p in SPLINE_AT_KNOTS
        for sparse in (False, True)
        # Stays Python (sparse_test.py).
        if not (
            sparse
            and p["degree"] == 0
            and p["extrapolation"] in ("constant", "periodic")
        )
    ],
)
def test_spline_at_the_knots(params, sparse):
    # Three features: spread ones, and one whose quantile knots repeat
    # (few distinct values). `_find_interval` must land where scipy's does
    # on each knot, and the extrapolations take over past the ends; under
    # "linear" at degree 0 or 1 the twin's range narrows from the second
    # feature on, and again from the third. The sparse output
    # (`sparse_output=True`) keeps no `0.0 * h` terms.
    rng = np.random.default_rng(7)
    X = np.column_stack(
        [
            rng.normal(3.0, 40.0, 30),
            rng.integers(-2, 3, 30).astype(float),
            rng.exponential(5.0, 30),
        ]
    )
    est = SplineTransformer(**params, sparse_output=sparse).fit(X)
    width = est.n_features_out_
    step = PythonTransform(
        "tf",
        {0: est},
        pa.schema([(f"x{j}", pa.float64()) for j in range(3)]),
        pa.struct([(f"f{i}", pa.float64()) for i in range(width)]),
    )
    rows = _knot_rows(est, 3)
    assert check(step, to_native(step, strict=True), rows) > 0


@pytest.mark.parametrize(
    "degree, n_knots", [(0, 2), (0, 5), (1, 4), (2, 5), (3, 5), (3, 6), (5, 8)]
)
def test_a_periodic_spline_answers_infinity_as_its_twin(degree, n_knots):
    # Under handle_missing="zeros" the twin validates with infinity allowed:
    # the remainder of ±inf is NaN, and scipy answers NaN on every output
    # field. An output field with no basis in the last interval read 0.0
    # there (from n_knots = degree + 3, and at degree 0), until the entry
    # tested for it.
    X = np.random.default_rng(0).normal(size=(40, 2)) * 3
    est = SplineTransformer(
        degree=degree,
        n_knots=n_knots,
        extrapolation="periodic",
        handle_missing="zeros",
    ).fit(X)
    step = PythonTransform(
        "tf",
        {0: est},
        pa.schema([("x0", pa.float64()), ("x1", pa.float64())]),
        pa.struct([(f"f{i}", pa.float64()) for i in range(est.n_features_out_)]),
    )
    vals = [math.inf, -math.inf, math.nan, None, 1e308, -1e308, 0.0, -0.0, 1.5]
    rows = pa.table(
        {
            "__iid": pa.array([0] * len(vals) ** 2, pa.int64()),
            "x0": pa.array([a for a in vals for _ in vals], pa.float64()),
            "x1": pa.array([b for _ in vals for b in vals], pa.float64()),
        }
    )
    assert check(step, to_native(step, strict=True), rows) == len(vals) ** 2


@pytest.mark.parametrize(
    "params, reason",
    [
        # From the second feature on the twin continues two output fields of
        # one: a row below raises, a row above writes the previous
        # feature's.
        (
            {"degree": 0, "n_knots": 2, "extrapolation": "linear"},
            "feature 1 continues 2 output fields of 1",
        ),
    ],
    ids=["linear-past-the-fields"],
)
def test_spline_refuses(params, reason):
    X = np.column_stack([np.arange(10.0), np.arange(10.0) ** 2])
    est = SplineTransformer(**params).fit(X)
    step = PythonTransform(
        "tf", {0: est}, pa.schema([("x0", pa.float64()), ("x1", pa.float64())])
    )
    with pytest.raises(NotNative, match=reason):
        to_native(step, strict=True)


def test_spline_refuses_a_build_past_the_cap():
    # 128 features of degree 5, 8 knots, "continue": estimated at 10 s, and
    # built in 9.4 s (spline.py, `_build_estimate`). The estimate refuses it
    # before confit is asked.
    X = np.random.default_rng(0).normal(size=(50, 128)) * 10
    est = SplineTransformer(degree=5, n_knots=8, extrapolation="continue").fit(X)
    takes = pa.schema([(f"x{j}", pa.float64()) for j in range(128)])
    returns = pa.struct([(f"f{i}", pa.float64()) for i in range(est.n_features_out_)])
    step = PythonTransform("tf", {0: est}, takes, returns)
    with pytest.raises(NotNative, match=r"an estimated \d+ s build, past 7 s"):
        to_native(step, strict=True)


def test_this_platform_evaluates_splines_as_the_entry():
    from sql_transform.native.spline import bspline_is_scipys

    assert bspline_is_scipys()


def test_a_failing_spline_probe_leaves_the_step_python(monkeypatch):
    from sql_transform.native import spline

    monkeypatch.setattr(spline, "bspline_is_scipys", lambda: False)
    est = SplineTransformer().fit(np.arange(10.0)[:, None])
    step = PythonTransform(
        "tf",
        {0: est},
        pa.schema([("x0", pa.float64())]),
        pa.struct([(f"f{i}", pa.float64()) for i in range(est.n_features_out_)]),
    )
    with pytest.raises(NotNative, match="bspline_is_scipys"):
        to_native(step, strict=True)


# ---------------------------------------------------------- IsotonicRegression


def _isotonic_step(est: Any) -> PythonTransform:
    return PythonTransform("tf", {0: est}, pa.schema([("x0", pa.float64())]))


def _probes(xp: np.ndarray) -> list[float]:
    """Each threshold, its neighbouring doubles, the midpoints, beyond both
    ends, and both signed zeros."""
    vals = {0.0, -0.0, float(xp[0]) - 1.0, float(xp[-1]) + 1.0}
    for t in xp:
        vals |= {
            float(t),
            float(np.nextafter(t, -np.inf)),
            float(np.nextafter(t, np.inf)),
        }
    vals |= {float(v) for v in (xp[:-1] + xp[1:]) / 2}
    return sorted(vals, key=lambda v: (v, math.copysign(1.0, v)))


@pytest.mark.parametrize("out_of_bounds", ["nan", "clip", "raise"])
@pytest.mark.parametrize("increasing", [True, False])
def test_isotonic_at_and_between_thresholds(out_of_bounds, increasing):
    # Thresholds straddling zero, with -0.0 values (y_max=-0.0 or y_min
    # with a negated target), where numpy's exact arm at a threshold keeps
    # the sign.
    rng = np.random.default_rng(3)
    X = np.round(rng.normal(0, 4, 80), 1)
    X[:3] = (0.0, -2.5, 2.5)
    y = (X if increasing else -X) + rng.normal(0, 1, 80)
    est = IsotonicRegression(
        increasing=increasing, y_max=-0.0, out_of_bounds=out_of_bounds
    ).fit(X, y)
    xp, fp = est.X_thresholds_, est.y_thresholds_
    assert len(xp) > 4
    assert any(v == 0.0 and math.copysign(1.0, v) < 0 for v in fp)
    probes = _probes(xp)
    rows = pa.table(
        {
            "__iid": pa.array([0] * len(probes), pa.int64()),
            "x0": pa.array(probes, pa.float64()),
        }
    )
    step = _isotonic_step(est)
    compared = check(step, to_native(step, strict=True), rows)
    outside = sum(not xp[0] <= v <= xp[-1] for v in probes)
    assert compared == len(probes) - (outside if out_of_bounds == "raise" else 0)


@pytest.mark.parametrize("out_of_bounds", ["nan", "clip", "raise"])
def test_isotonic_with_one_threshold_is_a_constant(out_of_bounds):
    est = IsotonicRegression(out_of_bounds=out_of_bounds).fit(
        np.full(6, 2.0), np.arange(6.0)
    )
    assert len(est.X_thresholds_) == 1
    probes = [2.0, -1e300, 0.0, -0.0, 7.5, 1e300, None]
    rows = pa.table(
        {
            "__iid": pa.array([0] * len(probes), pa.int64()),
            "x0": pa.array(probes, pa.float64()),
        }
    )
    step = _isotonic_step(est)
    # The twin raises on NaN (a NULL) only.
    assert check(step, to_native(step, strict=True), rows) == len(probes) - 1


def _fitted_isotonic(**kw) -> IsotonicRegression:
    X = np.arange(10.0)
    return IsotonicRegression(**kw).fit(X, X + np.sin(X))


def _mutate(est: IsotonicRegression, **attrs) -> IsotonicRegression:
    for k, v in attrs.items():
        setattr(est, k, v)
    return est


@pytest.mark.parametrize(
    "make, reason",
    [
        (
            lambda: IsotonicRegression().fit(
                np.arange(10, dtype=np.float32), np.arange(10.0)
            ),
            "float32 thresholds",
        ),
        (
            lambda: IsotonicRegression().fit(
                np.array([-1e308, 1e308]), np.array([0.0, 1.0])
            ),
            "further apart than a double",
        ),
        (
            lambda: _mutate(_fitted_isotonic(), f_=lambda T: T),
            "does not delegate to np.interp",
        ),
        (
            lambda: _mutate(
                _fitted_isotonic(),
                y_thresholds_=np.where(np.arange(10) == 4, np.inf, np.arange(10.0)),
            ),
            "not all finite",
        ),
        (
            lambda: _mutate(
                _fitted_isotonic(),
                X_thresholds_=np.array([0.0, 2.0, 1.0, *range(3, 10)]),
            ),
            "not strictly increasing",
        ),
    ],
    ids=["float32", "infinite-width", "not-np-interp", "infinite-value", "unsorted"],
)
def test_isotonic_refuses(make, reason):
    with pytest.raises(NotNative, match=reason):
        to_native(_isotonic_step(make()), strict=True)


def test_isotonic_refuses_two_features():
    takes = pa.schema([("x0", pa.float64()), ("x1", pa.float64())])
    step = PythonTransform("tf", {0: _fitted_isotonic()}, takes)
    with pytest.raises(NotNative, match="over 2 features"):
        to_native(step, strict=True)


def test_isotonic_takes_a_fit_by_its_estimated_build():
    from sql_transform.native.isotonic import MAX_BUILD_S, _build_estimate

    # 24,703 thresholds built in 5.9 s and 28,524 in 7.4 s (isotonic.py,
    # `_build_estimate`), where a count of 8,000 thresholds was the cap.
    assert _build_estimate(24_702) <= MAX_BUILD_S
    assert _build_estimate(28_523) > MAX_BUILD_S
    X = np.arange(32_000.0)
    step = _isotonic_step(IsotonicRegression().fit(X, X))
    with pytest.raises(NotNative, match=r"an estimated 9 s build, past 7 s"):
        to_native(step, strict=True)


def test_isotonic_refuses_a_fused_interp(monkeypatch):
    from sql_transform.native import isotonic

    monkeypatch.setattr(isotonic, "interp_is_numpys", lambda: False)
    with pytest.raises(NotNative, match="unfused C loop"):
        to_native(_isotonic_step(_fitted_isotonic()), strict=True)


# ------------------------------------------------------- FeatureAgglomeration


def _agglomeration_step(est: Any, n: int = 4) -> PythonTransform:
    width = len(np.unique(est.labels_))
    takes = pa.schema([(f"x{i}", pa.float64()) for i in range(n)])
    return PythonTransform("tf", {0: est}, takes, pa.list_(pa.float64(), width))


def test_agglomeration_pools_signed_zeros_from_zero():
    # Mean pooling sums from 0.0: a cluster of -0.0 pools to 0.0, a cluster
    # of one -0.0 too; a 0.0 beside a -0.0 in any order as well.
    X = np.array([[0.0, 0.0, 9.0, 9.0], [1.0, 1.0, -9.0, -9.0], [2.0, 2.0, 5, 5]])
    est = _clusters(lambda n: 2, linkage="single")().fit(X)
    assert len(set(est.labels_[:2])) == 1 and len(set(est.labels_[2:])) == 1
    zeros = [(a, b, c, d) for a in (0.0, -0.0) for b in (0.0, -0.0)
             for c in (0.0, -0.0) for d in (0.0, -0.0)]  # fmt: skip
    rows = pa.table(
        {
            "__iid": pa.array([0] * len(zeros), pa.int64()),
            **{f"x{i}": pa.array([z[i] for z in zeros]) for i in range(4)},
        }
    )
    singles = FeatureAgglomeration(n_clusters=4).fit(X)
    for e in (est, singles):
        step = _agglomeration_step(e)
        assert check(step, to_native(step, strict=True), rows) == len(zeros)


def _median_rows(X, axis):
    return np.median(X, axis=axis)


@pytest.mark.parametrize(
    "func, reason",
    [
        (np.max, r"pooling_func=np\.max\)"),
        (np.min, r"pooling_func=np\.min\)"),
        (np.median, r"pooling_func=np\.median\)"),
        (lambda X, axis: np.mean(X, axis=axis), r"pooling_func=<lambda>\)"),
        (_median_rows, r"pooling_func=_median_rows\)"),
    ],
    ids=["max", "min", "median", "lambda", "function"],
)
def test_agglomeration_refuses_other_pooling(func, reason):
    X = np.random.default_rng(0).normal(size=(10, 4))
    est = FeatureAgglomeration(pooling_func=func).fit(X)
    with pytest.raises(NotNative, match=reason):
        to_native(_agglomeration_step(est), strict=True)


def test_agglomeration_refuses_labels_that_skip_a_cluster():
    X = np.random.default_rng(0).normal(size=(10, 4))
    est = FeatureAgglomeration(n_clusters=2).fit(X)
    est.labels_ = np.array([0, 0, 2, 2])
    with pytest.raises(NotNative, match="skip a cluster"):
        to_native(_agglomeration_step(est), strict=True)


# -------------------------------------------------------- RandomTreesEmbedding


def _beside(v: float) -> list[float]:
    """`v` and the doubles either side of it."""
    return [float(np.nextafter(v, -np.inf)), v, float(np.nextafter(v, np.inf))]


def _cutpoint_values(t: float) -> list[float]:
    """A threshold, the float32 values either side of it, and the doubles
    at and beside the cutpoint the entry compares against instead."""
    from sql_transform._trees import _f32_grid_threshold

    below = np.float32(t)
    if float(below) > t:
        below = np.nextafter(below, np.float32(-np.inf))
    above = np.nextafter(below, np.float32(np.inf))
    cut = float(_f32_grid_threshold(np.array([t]))[0])
    return [t, float(below), float(above), *_beside(cut)]


# The values a float32 grid makes special: signed zeros, the double and
# float32 subnormals and the smallest normals, float32's largest finite
# value, and the largest double that still rounds to it (the next one up
# rounds to infinity); then past float32's range, and the infinities,
# where the twin raises; and NaN.
F32_MAX = float(np.finfo(np.float32).max)
F32_LAST = float(np.nextafter(2.0**128 * (1 - 2.0**-25), 0.0))
TREE_SPECIALS = [
    0.0,
    -0.0,
    5e-324,
    -5e-324,
    *_beside(float(np.finfo(np.float32).smallest_subnormal)),
    -float(np.finfo(np.float32).smallest_subnormal),
    float(np.finfo(np.float32).smallest_normal),
    float(np.finfo(np.float64).smallest_normal),
    F32_MAX,
    -F32_MAX,
    F32_LAST,
    -F32_LAST,
    2.0**128 * (1 - 2.0**-25),
    1e39,
    -1e300,
    math.inf,
    -math.inf,
    math.nan,
]


def test_trees_at_the_cutpoints():
    # Four features: spread values, float32 subnormals, values near
    # float32's largest, and few integers with holes (so some nodes saw
    # NaN and some did not). Every feature of a row is a training row's
    # but one, which takes each threshold's values in turn, and the
    # specials.
    rng = np.random.default_rng(11)
    n = 60
    X = np.column_stack(
        [
            rng.normal(0.0, 3.0, n),
            rng.integers(-40, 40, n) * float(np.finfo(np.float32).smallest_subnormal),
            rng.uniform(-1.0, 1.0, n) * F32_MAX,
            np.where(rng.random(n) < 0.2, np.nan, rng.integers(-3, 4, n)),
        ]
    )
    est = RandomTreesEmbedding(
        n_estimators=12, max_depth=5, sparse_output=False, random_state=0
    ).fit(X)
    assert any(t.tree_.missing_go_to_left[t.tree_.feature >= 0].any() for t in est)
    by_feature: dict[int, set[float]] = {j: set(TREE_SPECIALS) for j in range(4)}
    for tree in est.estimators_:
        t = tree.tree_
        for f, thr in zip(t.feature, t.threshold, strict=True):
            if f >= 0:
                by_feature[int(f)].update(_cutpoint_values(float(thr)))
    rows: list[list[float]] = []
    for j, values in by_feature.items():
        for v in sorted(values, key=repr):
            row = [float(c) for c in X[int(rng.integers(n))]]
            row[j] = v
            rows.append(row)
    step = _tree_step(est, 4)

    # The twin answers a row whose values are finite as float32s
    # (F32_LAST included, the next double up not), and raises on the rest;
    # served apart, so the step serves the first in one batch.
    def finite(r: list[float]) -> bool:
        return all(math.isnan(v) or abs(v) <= F32_LAST for v in r)

    def table(rs: list[list[float]]) -> pa.Table:
        cols = {f"x{j}": pa.array([r[j] for r in rs], pa.float64()) for j in range(4)}
        return pa.table({"__iid": pa.array([0] * len(rs), pa.int64()), **cols})

    native = to_native(step, strict=True)
    answered = [r for r in rows if finite(r)]
    assert check(step, native, table(answered)) == len(answered)
    # Where the twin raises, the entry traps (check asserts it on each
    # row), and check compares none.
    with pytest.raises(ParityError, match="compares no row"):
        check(step, native, table([r for r in rows if not finite(r)]))


def test_trees_trap_where_the_twin_raises():
    from sql_transform.native._check import _serve
    from sql_transform.native._registry import query

    X = np.random.default_rng(0).normal(size=(20, 2))
    est = RandomTreesEmbedding(n_estimators=2, sparse_output=False).fit(X)
    with pytest.raises(ValueError, match="infinity or a value too large"):
        est.transform([[2.0**128 * (1 - 2.0**-25), 0.0]])
    est.transform([[F32_LAST, 0.0]])  # the twin answers the double below
    step = _tree_step(est, 2)
    rows = pa.table(
        {
            "__iid": pa.array([0, 0], pa.int64()),
            "x0": pa.array([F32_LAST, 2.0**128 * (1 - 2.0**-25)]),
            "x1": pa.array([0.0, 0.0]),
        }
    )
    got = _serve(query(step), rows, to_native(step, strict=True))
    assert not isinstance(got[0], Exception)
    assert "the fitted estimator rejects" in str(got[1])


def _tree_step(est: RandomTreesEmbedding, n_features: int) -> PythonTransform:
    width = len(np.concatenate(est.one_hot_encoder_.categories_))
    return PythonTransform(
        "tf",
        {0: est},
        pa.schema([(f"x{j}", pa.float64()) for j in range(n_features)]),
        pa.list_(pa.float64(), width),
    )


def test_trees_serve_the_default_forest():
    # n_estimators=100, max_depth=5: up to 3,200 output fields, past
    # MAX_OUTPUT_FIELDS, so the generator never draws it (trees.py has its
    # builds).
    X = np.random.default_rng(3).normal(size=(500, 6))
    est = RandomTreesEmbedding(sparse_output=False, random_state=0).fit(X)
    step = _tree_step(est, 6)
    # Rows the twin answers, NULLs among them, so it serves them at once.
    R = np.random.default_rng(4).normal(size=(40, 6))
    cols = {f"x{j}": [None if v > 1.5 else float(v) for v in R[:, j]] for j in range(6)}
    rows = pa.table({"__iid": pa.array([0] * 40, pa.int64()), **cols})
    assert check(step, to_native(step, strict=True), rows) == 40


def _forest(
    n_estimators: int, max_depth: int | None, rows: int, features: int = 8
) -> RandomTreesEmbedding:
    X = np.random.default_rng(0).normal(size=(rows, features))
    return RandomTreesEmbedding(
        n_estimators=n_estimators,
        max_depth=max_depth,
        sparse_output=False,
        random_state=0,
    ).fit(X)


def test_trees_take_a_forest_by_the_estimated_build():
    from sql_transform.native.trees import MAX_BUILD_S, _build_estimate

    # The entry's build is estimated within 7 s for the default forest, the
    # default forest over 32 features, 400 trees of depth 5 (9,210 output
    # fields, built in 6.3 s) and one unbounded tree over 8,000 rows (4.7 s;
    # T46 on master 12aca98, trees.py).
    assert _build_estimate(_forest(100, 5, 2000)) <= MAX_BUILD_S
    assert _build_estimate(_forest(100, 5, 2000, features=32)) <= MAX_BUILD_S
    assert _build_estimate(_forest(400, 5, 2000)) <= MAX_BUILD_S
    assert _build_estimate(_forest(1, None, 8000)) <= MAX_BUILD_S


def test_trees_refuse_a_build_past_the_cap():
    # 300 trees of depth 6: 11,402 output fields, estimated at 9 s (8.1 s
    # measured).
    est = _forest(300, 6, 2000)
    with pytest.raises(NotNative, match=r"an estimated 9 s build, past 7 s"):
        to_native(_tree_step(est, 8), strict=True)
