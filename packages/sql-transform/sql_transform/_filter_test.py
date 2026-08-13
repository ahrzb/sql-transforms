"""FILTER on ``tfm_fit`` — fit/transform-split slice 3.

Leakage control is a spelling, not a policy: ``sc_fit(b) FILTER (WHERE
split = 'train') OVER (...)`` fits each scope on predicate-TRUE rows only
and transforms every row. A group with no passing rows is an unseen group
(params-miss NULL, P14); FILTER on any scalar call refuses like DuckDB.
Spec: docs/superpowers/specs/2026-08-05-fit-transform-split-design.md.
"""

import numpy as np
import pyarrow as pa
import pytest
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from sql_transform import MarginalizeError, SQLProjection

from ._transformers_test import TRAIN, _by_name

ROW = TRAIN.schema


def _fit(sql: str) -> SQLProjection:
    return SQLProjection(
        sql,
        this_schema=ROW,
        transformers={"sc": StandardScaler(), "pca": PCA(n_components=1)},
    ).fit(TRAIN)


def _ref_filtered(proto, feats, keys, mask):
    """Fit per group on mask-passing rows; transform every row; None where
    the row's group has no passing rows."""
    fit_idx: dict = {}
    for i, k in enumerate(keys):
        if mask[i]:
            fit_idx.setdefault(k, []).append(i)
    ests = {k: clone(proto).fit(feats[idx]) for k, idx in fit_idx.items()}
    return [
        float(np.asarray(ests[k].transform(feats[[i]]))[0, 0]) if k in ests else None
        for i, k in enumerate(keys)
    ]


def test_global_filtered_fit():
    p = _fit(
        "SELECT sc_transform(sc_fit(struct_pack(v := age))"
        " FILTER (WHERE fare > 6) OVER (), struct_pack(v := age)).v AS z,"
        " name FROM __THIS__"
    )
    feats = np.array([TRAIN.column("age").to_pylist()], dtype=float).T
    mask = [f > 6 for f in TRAIN.column("fare").to_pylist()]
    ref = _ref_filtered(StandardScaler(), feats, [()] * TRAIN.num_rows, mask)
    got = _by_name(p.transform(TRAIN), "z")
    for i, n in enumerate(TRAIN.column("name").to_pylist()):
        np.testing.assert_allclose(got[n], ref[i], rtol=1e-12)
    # row path serves the same artifact (C3)
    row = {"country": "US", "age": 40.0, "fare": 7.0, "name": "x"}
    np.testing.assert_allclose(p.infer(row)["z"], ref[0], rtol=1e-12)


def test_partitioned_filtered_fit_unseen_group_is_null():
    """FR's only row fails the predicate: its group is unseen — NULL, the
    same params-miss story as any unseen group (P14)."""
    p = _fit(
        "SELECT sc_transform(sc_fit(age) FILTER (WHERE fare > 6)"
        " OVER (PARTITION BY country), age).age AS z, name FROM __THIS__"
    )
    feats = np.array([TRAIN.column("age").to_pylist()], dtype=float).T
    mask = [f > 6 for f in TRAIN.column("fare").to_pylist()]
    keys = TRAIN.column("country").to_pylist()
    ref = _ref_filtered(StandardScaler(), feats, keys, mask)
    got = _by_name(p.transform(TRAIN), "z")
    for i, n in enumerate(TRAIN.column("name").to_pylist()):
        if ref[i] is None:
            assert got[n] is None, n
        else:
            np.testing.assert_allclose(got[n], ref[i], rtol=1e-12)


def test_non_boolean_predicate_uses_duckdb_cast():
    """Measured: DuckDB accepts a non-boolean FILTER predicate and casts
    (nonzero is true) — the fit side must inherit exactly that."""
    p = _fit(
        "SELECT sc_transform(sc_fit(age) FILTER (WHERE fare - 6) OVER (), age)"
        ".age AS z, name FROM __THIS__"
    )
    feats = np.array([TRAIN.column("age").to_pylist()], dtype=float).T
    mask = [f - 6 != 0 for f in TRAIN.column("fare").to_pylist()]
    ref = _ref_filtered(StandardScaler(), feats, [()] * TRAIN.num_rows, mask)
    got = _by_name(p.transform(TRAIN), "z")
    for i, n in enumerate(TRAIN.column("name").to_pylist()):
        np.testing.assert_allclose(got[n], ref[i], rtol=1e-12)


def test_filtered_and_unfiltered_fits_are_distinct_steps():
    p = _fit(
        "SELECT sc_transform(sc_fit(age) FILTER (WHERE fare > 6) OVER (), age)"
        ".age AS a,"
        " sc_transform(sc_fit(age) OVER (), age).age AS b, name FROM __THIS__"
    )
    assert len([s for s in p.plan if s.kind == "fit"]) == 2
    feats = np.array([TRAIN.column("age").to_pylist()], dtype=float).T
    mask = [f > 6 for f in TRAIN.column("fare").to_pylist()]
    ref_a = _ref_filtered(StandardScaler(), feats, [()] * TRAIN.num_rows, mask)
    ref_b = _ref_filtered(
        StandardScaler(), feats, [()] * TRAIN.num_rows, [True] * TRAIN.num_rows
    )
    out = p.transform(TRAIN)
    got_a, got_b = _by_name(out, "a"), _by_name(out, "b")
    for i, n in enumerate(TRAIN.column("name").to_pylist()):
        np.testing.assert_allclose(got_a[n], ref_a[i], rtol=1e-12)
        np.testing.assert_allclose(got_b[n], ref_b[i], rtol=1e-12)


def test_theta_lateral_with_filter_equals_inline():
    lateral = _fit(
        "SELECT sc_fit(age) FILTER (WHERE fare > 6) OVER () AS _th,"
        " sc_transform(_th, age).age AS z, name FROM __THIS__"
    )
    inline = _fit(
        "SELECT sc_transform(sc_fit(age) FILTER (WHERE fare > 6) OVER (), age)"
        ".age AS z, name FROM __THIS__"
    )
    assert lateral.serving_sql == inline.serving_sql


def test_everything_filtered_refuses_at_fit_by_name():
    p = SQLProjection(
        "SELECT sc_transform(sc_fit(age) FILTER (WHERE fare > 100) OVER (), age)"
        ".age AS z FROM __THIS__",
        transformers={"sc": StandardScaler()},
    )
    with pytest.raises(MarginalizeError, match="no training rows"):
        p.fit(TRAIN)


def test_author_udf_in_predicate_serves():
    """Review round: the predicate skipped UDF resolution, so a registered
    author UDF crashed raw at fit (CatalogException)."""
    from sql_transform import PythonUDF

    gt6 = PythonUDF(
        name="gt6",
        fn=lambda v: v is not None and v > 6.0,
        takes=pa.schema([("v", pa.float64())]),
        returns=pa.bool_(),
    )
    p = SQLProjection(
        "SELECT sc_transform(sc_fit(age) FILTER (WHERE gt6(fare)) OVER (), age)"
        ".age AS z, name FROM __THIS__",
        this_schema=ROW,
        transformers={"sc": StandardScaler(), "gt6": gt6},
    ).fit(TRAIN)
    ref = _fit(
        "SELECT sc_transform(sc_fit(age) FILTER (WHERE fare > 6) OVER (), age)"
        ".age AS z, name FROM __THIS__"
    )
    got = _by_name(p.transform(TRAIN), "z")
    want = _by_name(ref.transform(TRAIN), "z")
    for n, v in want.items():
        np.testing.assert_allclose(got[n], v, rtol=1e-12)


def test_predicates_differing_in_named_args_are_distinct_steps():
    """Review round: _stripped erases the named-arg aliases inside the
    predicate, collapsing two different fits into one step."""
    p = _fit(
        "SELECT sc_transform(sc_fit(age) FILTER (WHERE struct_extract("
        "struct_pack(a := fare, b := age), 'a') > 6) OVER (), age).age AS za,"
        " sc_transform(sc_fit(age) FILTER (WHERE struct_extract("
        "struct_pack(b := fare, a := age), 'a') > 6) OVER (), age).age AS zb,"
        " name FROM __THIS__"
    )
    assert len([s for s in p.plan if s.kind == "fit"]) == 2
    feats = np.array([TRAIN.column("age").to_pylist()], dtype=float).T
    mask_a = [f > 6 for f in TRAIN.column("fare").to_pylist()]  # 'a' := fare
    mask_b = [a > 6 for a in TRAIN.column("age").to_pylist()]  # 'a' := age
    ref_a = _ref_filtered(StandardScaler(), feats, [()] * TRAIN.num_rows, mask_a)
    ref_b = _ref_filtered(StandardScaler(), feats, [()] * TRAIN.num_rows, mask_b)
    out = p.transform(TRAIN)
    got_a, got_b = _by_name(out, "za"), _by_name(out, "zb")
    for i, n in enumerate(TRAIN.column("name").to_pylist()):
        np.testing.assert_allclose(got_a[n], ref_a[i], rtol=1e-12)
        np.testing.assert_allclose(got_b[n], ref_b[i], rtol=1e-12)


def test_schema_free_forward_alias_in_predicate_refuses():
    """Review round: schema-free, a predicate naming a LATER select alias
    served text DuckDB refuses (binds backward in the level table)."""
    with pytest.raises(MarginalizeError, match="lateral alias"):
        SQLProjection(
            "SELECT sc_transform(sc_fit(age) FILTER (WHERE m > 0) OVER (), age)"
            ".age AS z, fare - 6 AS m, name FROM __THIS__",
            transformers={"sc": StandardScaler()},
        )


REFUSALS = [
    # The predicate resolves like any expression (review round): unknown
    # columns and functions refuse at construction, not mid-fit.
    (
        "SELECT sc_transform(sc_fit(age) FILTER (WHERE nope > 0) OVER (), age)"
        ".age AS z FROM __THIS__",
        "unknown column nope",
    ),
    (
        "SELECT sc_transform(sc_fit(age) FILTER (WHERE nosuchfn(fare))"
        " OVER (), age).age AS z FROM __THIS__",
        "unknown function nosuchfn",
    ),
    (
        "SELECT sc_transform(sc_fit(age) FILTER (WHERE fare >"
        " (SELECT max(age) FROM __THIS__)) OVER (), age).age AS z FROM __THIS__",
        "subquery inside a FILTER",
    ),
    # FILTER on any scalar call refuses (measured: DuckDB binds FILTER only
    # on aggregates) — at construction, by name.
    (
        "SELECT (sc(age) FILTER (WHERE fare > 6)).age AS z FROM __THIS__",
        "FILTER on",
    ),
    (
        "SELECT (sc_transform(sc_fit(age) OVER (), age) FILTER (WHERE fare > 6))"
        ".age AS z FROM __THIS__",
        "FILTER on",
    ),
    ("SELECT round(age) FILTER (WHERE fare > 6) AS r FROM __THIS__", "FILTER on"),
    # No transformer calls inside the predicate.
    (
        "SELECT sc_transform(sc_fit(age) FILTER (WHERE pca(fare).pca0 > 0)"
        " OVER (), age).age AS z FROM __THIS__",
        "inside a FILTER",
    ),
]


@pytest.mark.parametrize("sql,match", REFUSALS)
def test_filter_refusals(sql, match):
    with pytest.raises(MarginalizeError, match=match):
        SQLProjection(
            sql,
            this_schema=ROW,
            transformers={"sc": StandardScaler(), "pca": PCA(n_components=1)},
        )
