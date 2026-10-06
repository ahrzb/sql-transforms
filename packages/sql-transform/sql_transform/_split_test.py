"""Inline fit/apply scopes, authored direct IDs and ordered named bundles."""

import duckdb
import numpy as np
import pyarrow as pa
import pytest
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from sql_transform import SQLProjection, TransformError

from ._transformers_test import TRAIN, _by_name, _reference


class CountingScaler(StandardScaler):
    fits = 0

    def fit(self, X, y=None, sample_weight=None):
        type(self).fits += 1
        return super().fit(X, y, sample_weight=sample_weight)


def test_split_spelling_fits_per_partition():
    sc = StandardScaler()
    p = SQLProjection.marginalize(
        "SELECT sc_transform(sc_fit(age) OVER (PARTITION BY country), age).age"
        " AS z, name FROM __THIS__",
        captured={"sc": sc},
    ).fit(TRAIN)
    assert len(p.instances) == len(set(TRAIN.column("country").to_pylist()))
    got = _by_name(p.transform(TRAIN), "z")
    feats = np.array([TRAIN.column("age").to_pylist()], dtype=float).T
    ref = _reference(sc, feats, [(c,) for c in TRAIN.column("country").to_pylist()])
    for i, n in enumerate(TRAIN.column("name").to_pylist()):
        np.testing.assert_allclose(got[n], ref[i][0], rtol=1e-12)


def test_fit_here_apply_there():
    """fit on age, apply to fare: previously unspellable (DRAFT-25)."""
    sc = StandardScaler()
    p = SQLProjection.marginalize(
        "SELECT sc_transform(sc_fit(struct_pack(v := age)) OVER (),"
        " struct_pack(v := fare)).v AS z, name FROM __THIS__",
        captured={"sc": sc},
    ).fit(TRAIN)
    got = _by_name(p.transform(TRAIN), "z")
    ages = np.array([TRAIN.column("age").to_pylist()], dtype=float).T
    est = StandardScaler().fit(ages)
    fares = np.array([TRAIN.column("fare").to_pylist()], dtype=float).T
    ref = est.transform(fares)[:, 0]
    for i, n in enumerate(TRAIN.column("name").to_pylist()):
        np.testing.assert_allclose(got[n], ref[i], rtol=1e-12)


def test_one_fit_feeds_two_transform_calls():
    """Identical inline fit scopes fit once even when applications differ."""
    CountingScaler.fits = 0
    sc = CountingScaler()
    p = SQLProjection.marginalize(
        "SELECT sc_transform(sc_fit(struct_pack(v := age)) OVER (),"
        " struct_pack(v := age)).v AS za,"
        " sc_transform(sc_fit(struct_pack(v := age)) OVER (),"
        " struct_pack(v := fare)).v AS zf, name FROM __THIS__",
        captured={"sc": sc},
    ).fit(TRAIN)
    assert CountingScaler.fits == 1
    assert len(p.instances) == 1
    ages = np.array([TRAIN.column("age").to_pylist()], dtype=float).T
    est = StandardScaler().fit(ages)
    out = p.transform(TRAIN)
    got_a = _by_name(out, "za")
    got_f = _by_name(out, "zf")
    ref_a = est.transform(ages)[:, 0]
    fares = np.array([TRAIN.column("fare").to_pylist()], dtype=float).T
    ref_f = est.transform(fares)[:, 0]
    for i, n in enumerate(TRAIN.column("name").to_pylist()):
        np.testing.assert_allclose(got_a[n], ref_a[i], rtol=1e-12)
        np.testing.assert_allclose(got_f[n], ref_f[i], rtol=1e-12)


def test_bare_call_is_global_fit_transform():
    sc = StandardScaler()
    p = SQLProjection.marginalize(
        "SELECT sc(age).age AS z, name FROM __THIS__", captured={"sc": sc}
    ).fit(TRAIN)
    assert len(p.instances) == 1
    got = _by_name(p.transform(TRAIN), "z")
    feats = np.array([TRAIN.column("age").to_pylist()], dtype=float).T
    ref = _reference(sc, feats, [()] * TRAIN.num_rows)
    for i, n in enumerate(TRAIN.column("name").to_pylist()):
        np.testing.assert_allclose(got[n], ref[i][0], rtol=1e-12)


REFUSALS = [
    ("SELECT sc(age) OVER ().age AS z FROM __THIS__", "aggregate|fit scope"),
    (
        "SELECT sc(age) OVER (PARTITION BY country).age AS z FROM __THIS__",
        "aggregate|fit scope",
    ),
    ("SELECT sc(age) + 1 AS s FROM __THIS__", r"STRUCT|struct|No function"),
    ("SELECT sc_fit(age).age AS z FROM __THIS__", "inline|fit"),
    (
        "SELECT sc_transform(sc_fit(age), age).age AS z FROM __THIS__",
        "canonical|inline",
    ),
    (
        "SELECT sc_transform(pca_fit(age) OVER (), age).age AS z FROM __THIS__",
        "outside its transform|canonical",
    ),
    (
        "SELECT sc_transform({'type': 'sc', 'id': 3}, age).age AS z FROM __THIS__",
        "canonical|direct",
    ),
    (
        "SELECT sc_transform(sc_fit(struct_pack(v := age)) OVER (),"
        " struct_pack(w := age)).v AS z FROM __THIS__",
        "match.*fit fields",
    ),
    (
        "SELECT sc_transform(sc_fit(age) OVER (ORDER BY fare), age).age AS z "
        "FROM __THIS__",
        "running fit",
    ),
    ("SELECT sc(age).age.x AS z FROM __THIS__", "chained.*field"),
    (
        "SELECT sc_transform(sc_fit(age) OVER (), age).age.x AS z FROM __THIS__",
        "chained.*field",
    ),
    (
        "SELECT pca(struct_pack(v := sc(age).age)).pca0 AS z FROM __THIS__",
        "nests.*estimator|nested",
    ),
    (
        "SELECT sc_transform(sc_fit(age) OVER (PARTITION BY pca(fare).pca0),"
        " age).age AS z FROM __THIS__",
        "nests.*estimator",
    ),
    (
        "SELECT sc_transform(sc_fit(struct_pack(v := age)) OVER (),"
        " struct_pack(v := pca(fare).pca0)).v AS z FROM __THIS__",
        "nests.*estimator|nested",
    ),
    (
        "SELECT ns.sc_fit(age) OVER () AS t FROM __THIS__",
        "namespaced|outside its transform",
    ),
    (
        "SELECT sc_transform(ns.sc_fit(age) OVER (), age).age AS z FROM __THIS__",
        "namespaced",
    ),
    (
        "SELECT sc_transform(sc_fit(age) OVER (), age) OVER () AS z FROM __THIS__",
        "scalar|aggregate|nests",
    ),
    (
        "SELECT (SELECT sc_fit(age) OVER () FROM __THIS__) AS t FROM __THIS__",
        "fit|window",
    ),
    (
        "SELECT sc_transform(sc_fit(struct_pack(a := age, f := fare)) OVER (),"
        " struct_pack(f := fare, a := age)).a AS z FROM __THIS__",
        "match.*fit fields",
    ),
    (
        "SELECT avg(sc_transform({'type': 'sc', 'id': 3}, age).age) OVER () AS z "
        "FROM __THIS__",
        "nests.*estimator|canonical",
    ),
]


@pytest.mark.parametrize("sql,match", REFUSALS)
def test_split_refusals(sql, match):
    sc = StandardScaler()
    pca = PCA(n_components=1)
    with pytest.raises((TransformError, duckdb.Error), match=match):
        SQLProjection.marginalize(
            sql,
            captured={"sc": sc, "pca": pca},
        ).fit(TRAIN).transform(TRAIN)


def test_reserved_fit_name_collision_on_bare_spelling_refuses():
    """Review round (2026-08-05): the bare x_fit(...).field spelling used to
    silently serve the x_fit registry object instead of refusing the
    reservation — the one spelling the struct-value hint points users at."""
    with pytest.raises(TransformError, match="reserve"):
        SQLProjection.marginalize(
            "SELECT x_fit(age).age AS z FROM __THIS__",
            captured={"x": StandardScaler(), "x_fit": StandardScaler()},
        )


def test_distinct_bundle_names_mint_distinct_fits():
    """Review round (2026-08-05): fit-step identity must include the bundle
    FIELD NAMES (S's names are the type, P16a) — _stripped erases aliases,
    so two fits differing only in struct_pack names used to collide into
    one step and serve the wrong fit's lanes."""
    CountingScaler.fits = 0
    sc = CountingScaler()
    p = SQLProjection.marginalize(
        "SELECT sc_transform(sc_fit(struct_pack(v := age)) OVER (),"
        " struct_pack(v := age)).v AS a,"
        " sc_transform(sc_fit(struct_pack(w := age)) OVER (),"
        " struct_pack(w := age)).w AS b, name FROM __THIS__",
        captured={"sc": sc},
    ).fit(TRAIN)
    assert CountingScaler.fits == 2
    assert len(p.instances) == 2
    out = p.transform(TRAIN)
    got_a = _by_name(out, "a")
    got_b = _by_name(out, "b")
    feats = np.array([TRAIN.column("age").to_pylist()], dtype=float).T
    ref = _reference(StandardScaler(), feats, [()] * TRAIN.num_rows)
    for i, n in enumerate(TRAIN.column("name").to_pylist()):
        np.testing.assert_allclose(got_a[n], ref[i][0], rtol=1e-12)
        np.testing.assert_allclose(got_b[n], ref[i][0], rtol=1e-12)


def test_field_read_is_case_insensitive():
    """DuckDB struct reads are ASCII-case-insensitive and so is the confit
    binder; the fit-time field check must match (review round 2026-08-05)."""
    sc = StandardScaler()
    p = SQLProjection.marginalize(
        "SELECT sc(struct_pack(V := age)).v AS z, name FROM __THIS__",
        captured={"sc": sc},
    ).fit(TRAIN)
    got = _by_name(p.transform(TRAIN), "z")
    feats = np.array([TRAIN.column("age").to_pylist()], dtype=float).T
    ref = _reference(StandardScaler(), feats, [()] * TRAIN.num_rows)
    for i, n in enumerate(TRAIN.column("name").to_pylist()):
        np.testing.assert_allclose(got[n], ref[i][0], rtol=1e-12)


def test_reserved_name_collision_refuses():
    """A registry entry literally named sc_transform while sc is registered
    is ambiguous — refuse, never silently shadow either reading."""
    sc = StandardScaler()
    other = StandardScaler()
    with pytest.raises(TransformError, match="reserve"):
        SQLProjection.marginalize(
            "SELECT sc_transform(sc_fit(age) OVER (), age).age AS z FROM __THIS__",
            captured={"sc": sc, "sc_transform": other},
        )


def test_migrated_shared_fit_cte_feeds_two_requests():
    CountingScaler.fits = 0
    fitted = SQLProjection(
        "WITH p AS (SELECT sc_fit(struct_pack(v := age)) AS iid FROM __FIT__)"
        " SELECT sc_transform(p.iid, struct_pack(v := t.age)).v AS za,"
        " sc_transform(p.iid, struct_pack(v := t.fare)).v AS zf, t.name"
        " FROM __THIS__ t LEFT JOIN p ON 1 = 1",
        captured={"sc": CountingScaler()},
    ).fit(TRAIN)
    assert CountingScaler.fits == 1
    ages = np.array(TRAIN.column("age").to_pylist(), dtype=float).reshape(-1, 1)
    fares = np.array(TRAIN.column("fare").to_pylist(), dtype=float).reshape(-1, 1)
    reference = StandardScaler().fit(ages)
    out = fitted.transform(TRAIN)
    np.testing.assert_allclose(
        out.column("za").to_pylist(), reference.transform(ages)[:, 0], rtol=1e-12
    )
    np.testing.assert_allclose(
        out.column("zf").to_pylist(), reference.transform(fares)[:, 0], rtol=1e-12
    )


def test_migrated_fit_partition_and_bundle_repeat_alias_expressions():
    sql = (
        "SELECT sc_transform(sc_fit(age) OVER(PARTITION BY age >= 30), age).age AS z,"
        " name FROM __THIS__"
    )
    fitted = SQLProjection.marginalize(sql, captured={"sc": StandardScaler()}).fit(
        TRAIN
    )
    feats = np.array(TRAIN.column("age").to_pylist(), dtype=float).reshape(-1, 1)
    reference = _reference(
        StandardScaler(), feats, [(age >= 30,) for age in feats[:, 0]]
    )
    got = _by_name(fitted.transform(TRAIN), "z")
    for i, name in enumerate(TRAIN.column("name").to_pylist()):
        np.testing.assert_allclose(got[name], reference[i][0], rtol=1e-12)
    half = SQLProjection.marginalize(
        "SELECT sc(struct_pack(v := age / 2)).v AS z, name FROM __THIS__",
        captured={"sc": StandardScaler()},
    ).fit(TRAIN)
    reference_half = _reference(StandardScaler(), feats / 2, [()] * TRAIN.num_rows)
    got_half = _by_name(half.transform(TRAIN), "z")
    for i, name in enumerate(TRAIN.column("name").to_pylist()):
        np.testing.assert_allclose(got_half[name], reference_half[i][0], rtol=1e-12)


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT sc(struct_pack(v := struct_pack(a := age))).v AS z FROM __THIS__",
        "SELECT sc(struct_pack(v := sc(age).age)).v AS z FROM __THIS__",
    ],
)
def test_nested_raw_bundle_is_not_scalar(sql):
    with pytest.raises(TransformError, match="nested|scalar|nests"):
        SQLProjection.marginalize(sql, captured={"sc": StandardScaler()})


def test_struct_typed_feature_refuses_by_declared_type():
    data = pa.table(
        {"age": [30.0, 40.0], "s": [{"a": 1.0}, {"a": 2.0}], "name": ["x", "y"]}
    )
    projection = SQLProjection.marginalize(
        "SELECT sc(struct_pack(v := s)).v AS z, name FROM __THIS__",
        captured={"sc": StandardScaler()},
    )
    with pytest.raises(TransformError, match="struct|unsupported"):
        projection.fit(data)
