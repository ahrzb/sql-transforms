"""The fit/transform split (2026-08-05 spec): bare ``tfm(x)`` is the one
sugar; every fit scope lives on ``tfm_fit`` where window semantics are
true; ``tfm_transform`` is an ordinary scalar; the ``tfm(x) OVER w``
sugar is deleted.

Spec: docs/superpowers/specs/2026-08-05-fit-transform-split-design.md
"""

import numpy as np
import pytest
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from sql_transform import MarginalizeError, SQLProjection

from ._transformers_test import TRAIN, _by_name, _reference


def test_split_spelling_fits_per_partition():
    sc = StandardScaler()
    p = SQLProjection(
        "SELECT sc_transform(sc_fit(age) OVER (PARTITION BY country), age).age"
        " AS z, name FROM __THIS__",
        transformers={"sc": sc},
    ).fit(TRAIN)
    (step,) = [s for s in p.plan if s.kind == "fit"]
    assert step.transformer == "sc" and step.keys != ()
    got = _by_name(p.transform(TRAIN), "z")
    feats = np.array([TRAIN.column("age").to_pylist()], dtype=float).T
    ref = _reference(sc, feats, [(c,) for c in TRAIN.column("country").to_pylist()])
    for i, n in enumerate(TRAIN.column("name").to_pylist()):
        np.testing.assert_allclose(got[n], ref[i][0], rtol=1e-12)


def test_fit_here_apply_there():
    """fit on age, apply to fare: previously unspellable (DRAFT-25)."""
    sc = StandardScaler()
    p = SQLProjection(
        "SELECT sc_transform(sc_fit(struct_pack(v := age)) OVER (),"
        " struct_pack(v := fare)).v AS z, name FROM __THIS__",
        transformers={"sc": sc},
    ).fit(TRAIN)
    got = _by_name(p.transform(TRAIN), "z")
    ages = np.array([TRAIN.column("age").to_pylist()], dtype=float).T
    est = StandardScaler().fit(ages)
    fares = np.array([TRAIN.column("fare").to_pylist()], dtype=float).T
    ref = est.transform(fares)[:, 0]
    for i, n in enumerate(TRAIN.column("name").to_pylist()):
        np.testing.assert_allclose(got[n], ref[i], rtol=1e-12)


def test_one_fit_feeds_two_transform_calls():
    """The same inline fit node mints ONE fit step; each transform call
    brings its own bundle."""
    sc = StandardScaler()
    p = SQLProjection(
        "SELECT sc_transform(sc_fit(struct_pack(v := age)) OVER (),"
        " struct_pack(v := age)).v AS za,"
        " sc_transform(sc_fit(struct_pack(v := age)) OVER (),"
        " struct_pack(v := fare)).v AS zf, name FROM __THIS__",
        transformers={"sc": sc},
    ).fit(TRAIN)
    assert len([s for s in p.plan if s.kind == "fit"]) == 1
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
    p = SQLProjection(
        "SELECT sc(age).age AS z, name FROM __THIS__",
        transformers={"sc": sc},
    ).fit(TRAIN)
    (step,) = [s for s in p.plan if s.kind == "fit"]
    assert step.transformer == "sc" and step.keys == ()
    got = _by_name(p.transform(TRAIN), "z")
    feats = np.array([TRAIN.column("age").to_pylist()], dtype=float).T
    ref = _reference(sc, feats, [()] * TRAIN.num_rows)
    for i, n in enumerate(TRAIN.column("name").to_pylist()):
        np.testing.assert_allclose(got[n], ref[i][0], rtol=1e-12)


REFUSALS = [
    # The deleted sugar: fit scope belongs on {tf}_fit (2026-08-05 spec).
    ("SELECT sc(age) OVER ().age AS z FROM __THIS__", "fit scope"),
    ("SELECT sc(age) OVER (PARTITION BY country).age AS z FROM __THIS__", "fit scope"),
    # Struct values outside a field read (output boundary is a later slice).
    ("SELECT sc(age) AS s FROM __THIS__", "struct value"),
    (
        "SELECT sc_transform(sc_fit(age) OVER (), age) AS s FROM __THIS__",
        "struct value",
    ),
    # θ positions that have no lawful reading yet.
    ("SELECT sc_fit(age) OVER () AS t FROM __THIS__", "later slice"),
    ("SELECT sc_fit(age).age AS z FROM __THIS__", "window aggregate"),
    (
        "SELECT sc_transform(sc_fit(age), age).age AS z FROM __THIS__",
        "lawful provenance",
    ),
    (
        "SELECT sc_transform(pca_fit(age) OVER (), age).age AS z FROM __THIS__",
        "lawful provenance",
    ),
    (
        "SELECT sc_transform({'type': 'sc', 'id': 3}, age).age AS z FROM __THIS__",
        "lawful provenance",
    ),
    # Bundle fields are name-keyed against the fit bundle.
    (
        "SELECT sc_transform(sc_fit(struct_pack(v := age)) OVER (),"
        " struct_pack(w := age)).v AS z FROM __THIS__",
        "do not match",
    ),
    # Fit-scope clauses that are later slices or unsupported.
    (
        "SELECT sc_transform(sc_fit(age) OVER (ORDER BY fare), age).age"
        " AS z FROM __THIS__",
        "running fit",
    ),
    (
        "SELECT sc_transform(sc_fit(age ORDER BY fare) OVER (), age).age"
        " AS z FROM __THIS__",
        "later slice",
    ),
    (
        "SELECT sc_transform(sc_fit(age) FILTER (WHERE fare > 6) OVER (), age).age"
        " AS z FROM __THIS__",
        "later slice",
    ),
    # A field is a scalar; chaining can never resolve.
    ("SELECT sc(age).age.x AS z FROM __THIS__", "chained field access"),
    (
        "SELECT sc_transform(sc_fit(age) OVER (), age).age.x AS z FROM __THIS__",
        "chained field access",
    ),
    # Nested transformer calls refuse at construction, never mid-fit
    # (composition is TASK-65, parked; review round 2026-08-05).
    (
        "SELECT pca(struct_pack(v := sc(age).age)).pca0 AS z FROM __THIS__",
        "inside a transformer bundle",
    ),
    (
        "SELECT sc_transform(sc_fit(age) OVER (PARTITION BY pca(fare).pca0),"
        " age).age AS z FROM __THIS__",
        "inside a partition",
    ),
    (
        "SELECT sc_transform(sc_fit(struct_pack(v := age)) OVER (),"
        " struct_pack(v := pca(fare).pca0)).v AS z FROM __THIS__",
        "inside a transformer bundle",
    ),
    # The namespace is the named mistake, on every half.
    ("SELECT ns.sc_fit(age) OVER () AS t FROM __THIS__", "namespaced"),
    (
        "SELECT sc_transform(ns.sc_fit(age) OVER (), age).age AS z FROM __THIS__",
        "namespaced",
    ),
    # OVER on the transform half names the actual mistake.
    (
        "SELECT sc_transform(sc_fit(age) OVER (), age) OVER () AS z FROM __THIS__",
        "is a scalar",
    ),
    # A fit window inside a subquery dies at construction, not mid-fit.
    (
        "SELECT (SELECT sc_fit(age) OVER () FROM __THIS__) AS t FROM __THIS__",
        "inside a subquery",
    ),
    # Bundle matching is name-keyed IN ORDER — same names, swapped order.
    (
        "SELECT sc_transform(sc_fit(struct_pack(a := age, f := fare)) OVER (),"
        " struct_pack(f := fare, a := age)).a AS z FROM __THIS__",
        "do not match",
    ),
    # A split call with a non-window θ inside an aggregate keeps its name.
    (
        "SELECT avg(sc_transform({'type': 'sc', 'id': 3}, age).age) OVER ()"
        " AS z FROM __THIS__",
        "inside a window aggregate",
    ),
]


@pytest.mark.parametrize("sql,match", REFUSALS)
def test_split_refusals(sql, match):
    sc = StandardScaler()
    pca = PCA(n_components=1)
    with pytest.raises(MarginalizeError, match=match):
        SQLProjection(sql, transformers={"sc": sc, "pca": pca})


def test_reserved_fit_name_collision_on_bare_spelling_refuses():
    """Review round (2026-08-05): the bare x_fit(...).field spelling used to
    silently serve the x_fit registry object instead of refusing the
    reservation — the one spelling the struct-value hint points users at."""
    with pytest.raises(MarginalizeError, match="reserve"):
        SQLProjection(
            "SELECT x_fit(age).age AS z FROM __THIS__",
            transformers={"x": StandardScaler(), "x_fit": StandardScaler()},
        )


def test_distinct_bundle_names_mint_distinct_fits():
    """Review round (2026-08-05): fit-step identity must include the bundle
    FIELD NAMES (S's names are the type, P16a) — _stripped erases aliases,
    so two fits differing only in struct_pack names used to collide into
    one step and serve the wrong fit's lanes."""
    sc = StandardScaler()
    p = SQLProjection(
        "SELECT sc_transform(sc_fit(struct_pack(v := age)) OVER (),"
        " struct_pack(v := age)).v AS a,"
        " sc_transform(sc_fit(struct_pack(w := age)) OVER (),"
        " struct_pack(w := age)).w AS b, name FROM __THIS__",
        transformers={"sc": sc},
    ).fit(TRAIN)
    assert len([s for s in p.plan if s.kind == "fit"]) == 2
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
    p = SQLProjection(
        "SELECT sc(struct_pack(V := age)).v AS z, name FROM __THIS__",
        transformers={"sc": sc},
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
    with pytest.raises(MarginalizeError, match="reserve"):
        SQLProjection(
            "SELECT sc_transform(sc_fit(age) OVER (), age).age AS z FROM __THIS__",
            transformers={"sc": sc, "sc_transform": other},
        )
