"""Named outputs and field access (DRAFT-24, loop 1).

A fitted transform is ``S -> T`` between named structs: S's field types come
from the bundle's real column types, T's field names are learned at fit
(sklearn's ``get_feature_names_out``, else canonical ``f0..``). Field access
resolves at marginalize time to a width-1 lane UDF, so no struct flows at
serving time; identity is name-keyed, so a refit that renumbers lanes breaks
loudly instead of rewiring silently.
"""

import numpy as np
import pyarrow as pa
import pytest
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sql_transform import MarginalizeError, SQLProjection

TRAIN = pa.table(
    {
        "color": ["red", "blue", "red", "blue"],
        "country": ["US", "US", "DE", "DE"],
        "age": [40.0, 30.0, 20.0, 25.0],
        "fare": [7.0, 8.0, 6.0, 9.0],
        "name": ["x", "y", "z", "w"],
    }
)


def _ohe():
    return OneHotEncoder(sparse_output=False, handle_unknown="ignore")


# --- T: learned output names ---------------------------------------------------


def test_sklearn_names_are_used_and_addressable():
    p = SQLProjection(
        "SELECT ohe(struct_pack(color := color)) OVER ().color_red AS is_red,"
        " name FROM __THIS__",
        transformers={"ohe": _ohe()},
    ).fit(TRAIN)
    (lane,) = [u for u in p.udfs.values() if u.name.startswith("__cf_tf0_g")]
    assert lane.return_names == ("color_red",)
    got = {r["name"]: r["is_red"] for r in p.transform(TRAIN).to_pylist()}
    assert got == {"x": 1.0, "y": 0.0, "z": 1.0, "w": 0.0}


def test_canonical_names_when_sklearn_offers_none():
    class Duck:  # no get_feature_names_out
        def fit(self, X):
            return self

        def transform(self, X):
            return np.asarray(X) * 2.0

    p = SQLProjection(
        "SELECT d(struct_pack(a := age)) OVER ().f0 AS z, name FROM __THIS__",
        transformers={"d": Duck()},
    ).fit(TRAIN)
    got = {r["name"]: r["z"] for r in p.transform(TRAIN).to_pylist()}
    assert got["x"] == 80.0


def test_pca_generated_names_mid_expression():
    pca = Pipeline([("p", PCA(n_components=2))])
    p = SQLProjection(
        "SELECT pca(struct_pack(a := age, f := fare)) OVER ().pca0 * 2 AS c,"
        " name FROM __THIS__",
        transformers={"pca": pca},
    ).fit(TRAIN)
    ref = clone_ref(pca, TRAIN)
    got = {r["name"]: r["c"] for r in p.transform(TRAIN).to_pylist()}
    for i, n in enumerate(TRAIN.column("name").to_pylist()):
        np.testing.assert_allclose(got[n], ref[i][0] * 2, rtol=1e-9)


def clone_ref(proto, table):
    from sklearn.base import clone

    feats = np.array(
        [table.column("age").to_pylist(), table.column("fare").to_pylist()], dtype=float
    ).T
    est = clone(proto)
    return np.asarray(est.fit(feats).transform(feats))


# --- S: feature types come from the data ---------------------------------------


def test_string_feature_keeps_its_type():
    p = SQLProjection(
        "SELECT ohe(struct_pack(color := color)) OVER ().color_blue AS b FROM __THIS__",
        transformers={"ohe": _ohe()},
    ).fit(TRAIN)
    (lane,) = [u for u in p.udfs.values() if u.name.startswith("__cf_tf0_g")]
    assert lane.takes == ("str",) and lane.take_names == ("color",)


def test_mixed_feature_types():
    ints = TRAIN.set_column(
        TRAIN.schema.get_field_index("fare"),
        "fare",
        pa.array([7, 8, 6, 9], type=pa.int64()),
    )

    class Widths:
        def fit(self, X):
            return self

        def transform(self, X):
            return np.asarray([[len(str(r[0])) + float(r[1])] for r in X])

    p = SQLProjection(
        "SELECT w(struct_pack(c := color, f := fare)) OVER ().f0 AS z, name"
        " FROM __THIS__",
        transformers={"w": Widths()},
    ).fit(ints)
    assert p.udfs["__cf_tf0_g0"].takes == ("str", "i64")
    got = {r["name"]: r["z"] for r in p.transform(ints).to_pylist()}
    assert got["x"] == len("red") + 7.0


# --- name-keyed identity: the decision -----------------------------------------


def test_refit_that_drops_a_field_refuses_by_name():
    p = SQLProjection(
        "SELECT ohe(struct_pack(color := color)) OVER ().color_red AS r FROM __THIS__",
        transformers={"ohe": _ohe()},
    ).fit(TRAIN)
    without_red = pa.table(
        {
            "color": ["blue", "aqua"],
            "country": ["US", "US"],
            "age": [1.0, 2.0],
            "fare": [1.0, 2.0],
            "name": ["a", "b"],
        }
    )
    with pytest.raises(MarginalizeError, match="no output field 'color_red'"):
        p.fit(without_red)


def test_per_group_shape_disagreement_refuses():
    grouped = pa.table(
        {
            "g": ["u", "u", "d", "d"],
            "c": ["red", "blue", "x", "y"],
            "name": list("abcd"),
        }
    )
    p = SQLProjection(
        "SELECT ohe(struct_pack(c := c)) OVER (PARTITION BY g).c_red AS r"
        " FROM __THIS__",
        transformers={"ohe": _ohe()},
    )
    with pytest.raises(MarginalizeError, match="different output shapes per group"):
        p.fit(grouped)


# --- field access mechanics ----------------------------------------------------


def test_two_fields_share_one_fit_step():
    p = SQLProjection(
        "SELECT ohe(struct_pack(color := color)) OVER ().color_red AS r,"
        " ohe(struct_pack(color := color)) OVER ().color_blue AS b, name FROM __THIS__",
        transformers={"ohe": _ohe()},
    ).fit(TRAIN)
    assert [s.name for s in p.plan] == ["__CF_LEVEL_0__", "__CF_PARAMS_0__"]
    assert "__cf_tf0_g0(__cf_p0.__cf_est, __cf_t.color)" in p.serving_sql
    assert "__cf_tf0_g1(__cf_p0.__cf_est, __cf_t.color)" in p.serving_sql
    rows = p.transform(TRAIN).to_pylist()
    assert rows[0]["r"] == 1.0 and rows[0]["b"] == 0.0


def test_field_access_serves_row_at_a_time():
    p = SQLProjection(
        "SELECT ohe(struct_pack(color := color)) OVER ().color_red AS r, name"
        " FROM __THIS__",
        transformers={"ohe": _ohe()},
    ).fit(TRAIN)
    want = p.transform(TRAIN).to_pylist()
    got = [r.model_dump() for r in p.infer_batch(TRAIN.to_pylist())]
    assert got == want


def test_field_access_under_partition_unseen_group_is_null():
    # StandardScaler passes input names through, so the output field of
    # `struct_pack(a := age)` is `a` — names follow the producer.
    p = SQLProjection(
        "SELECT sc(struct_pack(a := age)) OVER (PARTITION BY country).a AS z, name"
        " FROM __THIS__",
        transformers={"sc": StandardScaler()},
    ).fit(TRAIN)
    out = p.infer(
        {"color": "red", "country": "JP", "age": 1.0, "fare": 1.0, "name": "q"}
    )
    assert out.z is None


def test_bare_call_still_emits_the_list():
    p = SQLProjection(
        "SELECT pca(struct_pack(a := age, f := fare)) OVER () AS e, name FROM __THIS__",
        transformers={"pca": PCA(n_components=2)},
    ).fit(TRAIN)
    assert p.transform(TRAIN).column("e").to_pylist()[0].__class__ is list


def test_unknown_field_name_refuses_at_fit():
    p = SQLProjection(
        "SELECT pca(struct_pack(a := age, f := fare)) OVER ().nope AS z FROM __THIS__",
        transformers={"pca": PCA(n_components=2)},
    )
    with pytest.raises(MarginalizeError, match="no output field 'nope'"):
        p.fit(TRAIN)


def test_computed_field_name_refuses():
    with pytest.raises(MarginalizeError, match="computed field name"):
        SQLProjection(
            "SELECT struct_extract(sc(struct_pack(a := age)) OVER (), name) AS z"
            " FROM __THIS__",
            transformers={"sc": StandardScaler()},
        )
