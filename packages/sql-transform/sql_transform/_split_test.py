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
]


@pytest.mark.parametrize("sql,match", REFUSALS)
def test_split_refusals(sql, match):
    sc = StandardScaler()
    pca = PCA(n_components=1)
    with pytest.raises(MarginalizeError, match=match):
        SQLProjection(sql, transformers={"sc": sc, "pca": pca})


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
