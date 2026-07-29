"""Transformers-as-UDAFs v0: fit runs sklearn per group; the gate's oracle
for transformer columns is an independent clone/fit/transform reference."""

import numpy as np
import pyarrow as pa
import pytest
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sql_transform import MarginalizeError, SQLProjection

TRAIN = pa.table(
    {
        "country": ["US", "US", None, "DE", None, "DE", "FR"],
        "age": [40.0, 30.0, 20.0, 25.0, 50.0, 45.0, 35.0],
        "fare": [7.0, 8.0, 6.0, 9.0, 10.0, 11.0, 5.0],
        "name": ["x", "y", "z", "w", "v", "u", "t"],
    }
)


def _reference(proto, feats, keys):
    """Independent oracle: clone-per-group fit_transform, row-aligned."""
    groups = {}
    for i, k in enumerate(keys):
        groups.setdefault(k, []).append(i)
    out = [None] * len(keys)
    for _k, idx in groups.items():
        est = clone(proto)
        block = np.asarray(est.fit(feats[idx]).transform(feats[idx]))
        if block.ndim == 1:
            block = block.reshape(-1, 1)
        for row, vals in zip(idx, block, strict=True):
            out[row] = [float(v) for v in vals]
    return out


def _by_name(table, value_col):
    names = table.column("name").to_pylist()
    vals = table.column(value_col).to_pylist()
    return dict(zip(names, vals, strict=True))


def test_global_scaler_from_scope():
    sc = StandardScaler()
    p = SQLProjection("SELECT sc(age) OVER () AS z, name FROM __THIS__").fit(TRAIN)
    (step,) = [s for s in p.plan if s.kind == "fit"]
    assert step.transformer == "sc" and step.keys == ()
    got = _by_name(p.transform(TRAIN), "z")
    feats = np.array([TRAIN.column("age").to_pylist()], dtype=float).T
    ref = _reference(sc, feats, [()] * TRAIN.num_rows)
    for i, n in enumerate(TRAIN.column("name").to_pylist()):
        np.testing.assert_allclose(got[n], ref[i], rtol=1e-12)


def test_per_country_pipeline_with_struct_bundle():
    embed = Pipeline([("s", StandardScaler()), ("p", PCA(n_components=1))])
    p = SQLProjection(
        "SELECT embed(struct_pack(a := age, f := fare))"
        " OVER (PARTITION BY country) AS e, name FROM __THIS__",
        transformers={"embed": embed},
    ).fit(TRAIN)
    got = _by_name(p.transform(TRAIN), "e")
    feats = np.array(
        [TRAIN.column("age").to_pylist(), TRAIN.column("fare").to_pylist()],
        dtype=float,
    ).T
    ref = _reference(embed, feats, [(c,) for c in TRAIN.column("country").to_pylist()])
    for i, n in enumerate(TRAIN.column("name").to_pylist()):
        np.testing.assert_allclose(got[n], ref[i], rtol=1e-9)


def test_unseen_group_gets_null():
    sc = StandardScaler()
    p = SQLProjection(
        "SELECT sc(age) OVER (PARTITION BY country) AS z, name FROM __THIS__",
        transformers={"sc": sc},
    ).fit(TRAIN)
    new = pa.table({"country": ["JP"], "age": [1.0], "fare": [1.0], "name": ["q"]})
    assert p.transform(new).column("z").to_pylist() == [None]


def test_mixed_sql_and_transformer_columns():
    sc = StandardScaler()
    p = SQLProjection(
        "SELECT age - avg(age) OVER () AS c, sc(fare) OVER () AS z, name FROM __THIS__",
        transformers={"sc": sc},
    ).fit(TRAIN)
    out = p.transform(TRAIN)
    mean = np.mean(TRAIN.column("age").to_pylist())
    got = _by_name(out, "c")
    for i, n in enumerate(TRAIN.column("name").to_pylist()):
        np.testing.assert_allclose(got[n], TRAIN.column("age")[i].as_py() - mean)
    assert out.column_names.index("c") < out.column_names.index("z")


@pytest.mark.parametrize(
    "sql,match",
    [
        ("SELECT nope(age) OVER () FROM __THIS__", "unknown window function nope"),
        (
            "SELECT sc(age) OVER (ORDER BY age) FROM __THIS__",
            "ORDER BY on a transformer",
        ),
        ("SELECT sc(*) OVER () FROM __THIS__", "zero arguments"),
        ("SELECT sc(age, fare) OVER () FROM __THIS__", "exactly one bundle"),
        ("SELECT sc(struct_pack(age)) OVER () FROM __THIS__", "must be named"),
        ("SELECT sc(age) OVER () + 1 FROM __THIS__", "top-level select item"),
        (
            "WITH a AS (SELECT sc(age) OVER () AS z FROM __THIS__) SELECT z FROM a",
            "non-final level",
        ),
        ("SELECT sk.sc(age) OVER () FROM __THIS__", "namespaced transformer"),
    ],
)
def test_transformer_refusals(sql, match):
    sc = StandardScaler()
    with pytest.raises(MarginalizeError, match=match):
        SQLProjection(sql, transformers={"sc": sc})


def test_registry_beats_scope():
    sc = StandardScaler()  # noqa: F841 — the scope decoy
    p = SQLProjection(
        "SELECT sc(age) OVER () AS z FROM __THIS__",
        transformers={"sc": Pipeline([("s", StandardScaler())])},
    )
    assert isinstance(p.transformers["sc"], Pipeline)
