"""The input guard: where the validation of the twin raises, the entry
traps, and nowhere else (loops/native/decisions/closed/tolerated-differences.md,
the ruling). `check` holds every row to it; each test here also says which
rows trap, so that none passes with both sides answering."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pyarrow as pa
import pytest
from confit import DuckDBInferFn
from confit import sql as S
from confit.oracle import Oracle
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.pipeline import FeatureUnion, make_pipeline
from sklearn.preprocessing import (
    Binarizer,
    KBinsDiscretizer,
    OneHotEncoder,
    OrdinalEncoder,
    PowerTransformer,
    SplineTransformer,
    StandardScaler,
)

from sql_transform._udf import PythonTransform
from sql_transform.native import Entry, NotNative, ParityError, check, to_native
from sql_transform.native._check import TRAP_QUERIES, _definition, _serve
from sql_transform.native._helpers import F32_INF, f64
from sql_transform.native._registry import query, rejects

# The double below F32_INF, the largest that float32 narrowing keeps finite.
F32_LAST = float(np.nextafter(F32_INF, 0.0))

# The twin narrowing a row to float32 warns where it overflows, on purpose.
pytestmark = pytest.mark.filterwarnings(
    "ignore:overflow encountered in cast:RuntimeWarning"
)


def _step(est: Any, types: list[pa.DataType], width: int) -> PythonTransform:
    """One fitted instance over features x0, x1, ... of `types`, returning
    a struct of `width` lanes."""
    return PythonTransform(
        "tf",
        {0: est},
        pa.schema([(f"x{i}", t) for i, t in enumerate(types)]),
        pa.struct([(f"f{j}", pa.float64()) for j in range(width)]),
    )


def _rows(step: PythonTransform, rows: list[tuple]) -> pa.Table:
    cols = {
        f.name: pa.array([r[i] for r in rows], f.type) for i, f in enumerate(step.takes)
    }
    return pa.table({"__iid": pa.array([0] * len(rows), pa.int64()), **cols})


def _traps(step: PythonTransform, native: Any, rows: pa.Table) -> list[bool]:
    """Whether confit traps on each row, served with the native twin."""
    return [isinstance(g, Exception) for g in _serve(query(step), rows, native)]


@pytest.mark.parametrize("returns", ["struct", "list"])
def test_a_read_of_any_lane_fires_the_guard(returns):
    # StandardScaler's validation rejects ±inf and lets NaN through. The
    # guard is in the first lane, and a read of the second alone fires it,
    # in confit and in DuckDB.
    est = StandardScaler().fit(np.random.default_rng(0).normal(size=(20, 2)))
    r = (
        pa.struct([("f0", pa.float64()), ("f1", pa.float64())])
        if returns == "struct"
        else pa.list_(pa.float64(), 2)
    )
    takes = pa.schema([("x0", pa.float64()), ("x1", pa.float64())])
    step = PythonTransform("tf", {0: est}, takes, r)
    native = to_native(step, strict=True)
    last = ".f1" if returns == "struct" else "[2]"
    sql = f"SELECT tf(__iid, x0, x1){last} AS o FROM __THIS__"
    for x0, traps in (
        (1.0, False),
        (math.inf, True),
        (-math.inf, True),
        (math.nan, False),
    ):
        rows = pa.table(
            {
                "__iid": pa.array([0], pa.int64()),
                "x0": pa.array([x0]),
                "x1": pa.array([2.0]),
            }
        )
        f = DuckDBInferFn(
            sql, row_tables={"__THIS__": rows.schema}, static_tables={}, udfs=[native]
        )
        with Oracle() as o:
            native.register(o)
            o.load("__THIS__", rows)
            want = o.try_answer(sql)
        if traps:
            with pytest.raises(ValueError, match="tf: a value in this row that"):
                f.infer_arrow(rows)
            assert "the fitted estimator rejects" in str(want)
        else:
            assert f.infer_arrow(rows).to_pylist() == want.to_pylist()


@pytest.mark.parametrize(
    "est, traps",
    [
        # Validation that lets NaN through, and validation that does not.
        (StandardScaler(), [False, True, True, False, False]),
        (Binarizer(), [False, True, True, True, True]),
    ],
    ids=["allow-nan", "finite"],
)
def test_the_guard_traps_where_the_validation_raises(est, traps):
    est.fit(np.random.default_rng(0).normal(size=(20, 2)))
    step = _step(est, [pa.float64(), pa.float64()], 2)
    native = to_native(step, strict=True)
    rows = _rows(
        step,
        [(0.5, 1.0), (math.inf, 1.0), (1.0, -math.inf), (math.nan, 1.0), (None, 1.0)],
    )
    assert _traps(step, native, rows) == traps
    assert check(step, native, rows) == traps.count(False)


# Over boolean features only, the step hands a row none NULL as a boolean
# array, which SimpleImputer(strategy="most_frequent") rejects, and a row
# with a NULL as float64. A ColumnTransformer hands its part an object
# array, which it accepts; a selector hands the next step the array it
# was handed.
@pytest.mark.parametrize(
    "make, traps",
    [
        (lambda: SimpleImputer(strategy="most_frequent"), [True, False, False, False]),
        (
            lambda: make_pipeline(
                VarianceThreshold(), SimpleImputer(strategy="most_frequent")
            ),
            [True, False, False, False],
        ),
        (
            lambda: ColumnTransformer(
                [("i", SimpleImputer(strategy="most_frequent"), [0, 1])]
            ),
            [False, False, False, False],
        ),
    ],
    ids=["list", "booleans", "objects"],
)
def test_a_boolean_row_traps_as_its_container_does(make, traps):
    X = np.array([[1.0, 0.0], [np.nan, 1.0], [1.0, 1.0], [0.0, np.nan]])
    est = make().fit(X)
    step = _step(est, [pa.bool_(), pa.bool_()], 2)
    native = to_native(step, strict=True)
    rows = _rows(step, [(True, False), (None, True), (False, None), (None, None)])
    assert _traps(step, native, rows) == traps
    assert check(step, native, rows) == traps.count(False)


def test_a_later_step_traps_on_what_the_earlier_one_hands_it():
    # OrdinalEncoder answers NaN for an unknown or a missing string, and
    # Binarizer rejects NaN: the pipeline raises there. Binarizer's guard
    # tests its own input, the encoder's lane.
    est = make_pipeline(
        OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan),
        Binarizer(threshold=0.5),
    ).fit(np.array([["a"], ["b"], ["c"]], dtype=object))
    step = _step(est, [pa.string()], 1)
    native = to_native(step, strict=True)
    rows = _rows(step, [("a",), ("c",), ("zz",), (None,)])
    assert _traps(step, native, rows) == [False, False, True, True]
    assert check(step, native, rows) == 2


def test_a_composition_handed_objects_and_handing_on_stays_python():
    # A ColumnTransformer hands its part an object array. The guard does not
    # probe what a composition there hands the next step.
    est = ColumnTransformer(
        [
            (
                "p",
                make_pipeline(
                    FeatureUnion([("s", StandardScaler())]), StandardScaler()
                ),
                [0],
            )
        ]
    ).fit(np.random.default_rng(0).normal(size=(20, 1)))
    step = _step(est, [pa.float64()], 1)
    with pytest.raises(NotNative, match="a FeatureUnion handed objects"):
        to_native(step, strict=True)


def _positive(n: int) -> np.ndarray:
    return np.random.default_rng(0).uniform(0.5, 3.0, size=(n, 1))


# Each family whose twin raises past the values the probe finds: its
# domain test (`rejects`), on rows on both sides of it.
DOMAINS = {
    "box-cox": (
        lambda: PowerTransformer(method="box-cox", standardize=False).fit(
            _positive(20)
        ),
        [pa.float64()],
        [(2.0,), (0.0,), (-0.0,), (-1.0,), (math.inf,), (math.nan,), (None,)],
        [False, True, True, True, True, False, False],
    ),
    "isotonic-raise": (
        lambda: IsotonicRegression(out_of_bounds="raise").fit(
            np.arange(10.0), np.arange(10.0) ** 2
        ),
        [pa.float64()],
        [(0.0,), (4.5,), (9.0,), (-0.5,), (9.5,), (math.nan,), (None,)],
        [False, False, False, True, True, True, True],
    ),
    "spline-error": (
        lambda: SplineTransformer(n_knots=4, extrapolation="error").fit(
            np.linspace(0.0, 1.0, 20).reshape(-1, 1)
        ),
        [pa.float64()],
        [(0.0,), (0.5,), (1.0,), (-0.25,), (1.25,), (math.nan,), (None,)],
        [False, False, False, True, True, True, True],
    ),
    "indicator-error-on-new": (
        lambda: MissingIndicator(missing_values=-1.0).fit(
            np.array([[-1.0, 2.0], [3.0, 4.0]])
        ),
        [pa.float64(), pa.float64()],
        [(-1.0, 2.0), (3.0, 4.0), (3.0, -1.0), (math.nan, 2.0)],
        [False, False, True, True],
    ),
    "float32-bins": (
        lambda: KBinsDiscretizer(
            n_bins=3, encode="ordinal", dtype=np.float32, quantile_method="linear"
        ).fit(np.arange(12.0).reshape(-1, 1)),
        [pa.float64()],
        [(5.0,), (F32_LAST,), (-F32_LAST,), (F32_INF,), (-1e39,), (math.inf,)],
        [False, False, False, True, True, True],
    ),
    "unknown-category": (
        lambda: OneHotEncoder(sparse_output=False).fit(
            np.array([["a"], ["b"]], dtype=object)
        ),
        [pa.string()],
        [("a",), ("b",), ("zz",), ("",), (None,)],
        [False, False, True, True, True],
    ),
    "unknown-number": (
        lambda: OrdinalEncoder().fit(np.array([[1.0], [2.0]])),
        [pa.float64()],
        [(1.0,), (2.0,), (3.0,), (math.nan,), (math.inf,)],
        [False, False, True, True, True],
    ),
}


@pytest.mark.parametrize("family", list(DOMAINS))
def test_a_family_traps_past_its_domain(family):
    make, types, values, traps = DOMAINS[family]
    est = make()
    width = np.asarray(est.transform([list(values[0])])).reshape(1, -1).shape[1]
    step = _step(est, types, width)
    native = to_native(step, strict=True, allow_bound=True)
    rows = _rows(step, values)
    assert _traps(step, native, rows) == traps
    assert check(step, native, rows) == traps.count(False)


def test_a_twin_error_the_ruling_lists_is_let_through():
    # SplineTransformer(degree=0, extrapolation="constant") fails to
    # broadcast above its knots, an sklearn bug and not validation: the
    # ruling lets the entry answer there, and check lets the row through.
    est = SplineTransformer(degree=0, n_knots=3, extrapolation="constant").fit(
        np.linspace(0.0, 1.0, 20).reshape(-1, 1)
    )
    with pytest.raises(ValueError, match="could not be broadcast"):
        est.transform([[2.0]])
    width = est.transform([[0.5]]).shape[1]
    step = _step(est, [pa.float64()], width)
    native = to_native(step, strict=True)
    rows = _rows(step, [(0.5,), (2.0,), (-1.0,)])
    assert _traps(step, native, rows) == [False, False, False]
    assert check(step, native, rows) == 2


class _Positive(BaseEstimator, TransformerMixin):
    """The identity, which raises on a number at most 0 as a validation
    does (NaN passes: it is not at most 0)."""

    def fit(self, X, y=None):
        self.n_features_in_ = 1
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        if (X <= 0).any():
            raise ValueError("a number at most 0")
        return X


def _guarded(at: float):
    def translate(est, x, types):
        rejects(x[0] <= f64(at))
        return [x[0]]

    return translate


def _bare(est, x, types):
    return [x[0]]


def _fails(est, x, types):
    return [S.case(x[0] <= f64(0.0), S.fn("error", S.lit("boom"))).otherwise(x[0])]


@pytest.mark.parametrize(
    "translate, why",
    [
        (_guarded(0.0), None),
        (_bare, r"row 1: the step raises \(.*a number at most 0.*\) where the native"),
        (_guarded(1.0), r"row 3: the native twin traps \(.*\) where the step answers"),
        (_fails, r"row 1: the native twin fails \(.*boom.*\), not by one of its traps"),
    ],
    ids=["exact", "a-missed-trap", "a-trap-too-many", "an-error-not-a-trap"],
)
def test_check_asserts_a_trap_where_the_step_raises_and_nowhere_else(
    monkeypatch, translate, why
):
    from sql_transform.native import _registry

    monkeypatch.setitem(_registry._CATALOG, _Positive, Entry(translate, 0))
    step = _step(_Positive().fit(None), [pa.float64()], 1)
    native = to_native(step, strict=True)
    rows = _rows(step, [(2.0,), (-1.0,), (-math.inf,), (0.5,), (math.nan,)])
    if why is None:
        assert _traps(step, native, rows) == [False, True, True, False, False]
        assert check(step, native, rows) == 3
    else:
        with pytest.raises(ParityError, match=why):
            check(step, native, rows)


# DuckDB runs each of the first TRAP_QUERIES rows confit traps on alone,
# and the rest together. Here confit's answers are marked as traps on rows
# it answers (`fake`), where DuckDB answers too.
INF = math.inf


@pytest.mark.parametrize(
    "values, fake, why",
    [
        ([0.5, 0.7] + [INF] * TRAP_QUERIES, [0], r"row 0: confit traps"),
        (
            [INF] * TRAP_QUERIES + [0.5, 0.7],
            [TRAP_QUERIES, TRAP_QUERIES + 1],
            rf"rows \[{TRAP_QUERIES}, {TRAP_QUERIES + 1}\]: confit traps on each",
        ),
    ],
    ids=["alone", "together"],
)
def test_check_holds_each_trap_of_confit_to_duckdb(values, fake, why):
    est = StandardScaler().fit(np.random.default_rng(0).normal(size=(20, 1)))
    step = _step(est, [pa.float64()], 1)
    native = to_native(step, strict=True)
    rows = _rows(step, [(v,) for v in values])
    got = _serve(query(step), rows, native)
    assert [isinstance(g, Exception) for g in got] == [v == INF for v in values]
    _definition(step, native, rows, got, "__iid")
    said = [ValueError("a trap") if i in fake else g for i, g in enumerate(got)]
    with pytest.raises(ParityError, match=why):
        _definition(step, native, rows, said, "__iid")


class _Never(_Positive):
    def transform(self, X):
        raise ValueError("never")


def test_a_twin_that_raises_on_the_probe_row_stays_python(monkeypatch):
    from sql_transform.native import _registry

    monkeypatch.setitem(_registry._CATALOG, _Never, Entry(_bare, 0))
    step = _step(_Never().fit(None), [pa.float64()], 1)
    assert to_native(step) is step
    with pytest.raises(
        NotNative, match=r"_Never: the twin raises on the input guard's"
    ):
        to_native(step, strict=True)
