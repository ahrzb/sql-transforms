"""Named outputs and field access (DRAFT-24, loops 1-4).

A fitted transform is ``S -> T`` between named structs: S's field types come
from the bundle's real column types, T's field names are learned at fit
(sklearn's ``get_feature_names_out``, else canonical ``f0..``). Field access
is validated at fit and serves as a field read over the ONE whole-value call
(TASK-63) — k addressed fields cost one evaluation per row on both engines;
identity is name-keyed, so a refit that renumbers lanes breaks loudly
instead of rewiring silently.
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
    assert p.udfs["__cf_tf0"].return_names == ("color_blue", "color_red")
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
    u = p.udfs["__cf_tf0"]
    assert u.takes == ("str",) and u.take_names == ("color",)


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
    assert p.udfs["__cf_tf0"].takes == ("str", "i64")
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
    # ONE call, two field reads (TASK-63) — no per-lane UDFs. The identical
    # mentions cost one evaluation per row (DuckDB CSE / confit's shared
    # ecall site; counted in _single_eval_test.py).
    assert "(__cf_tf0(__cf_p0.__cf_est, __cf_t.color)).color_red" in p.serving_sql
    assert "(__cf_tf0(__cf_p0.__cf_est, __cf_t.color)).color_blue" in p.serving_sql
    assert "__cf_tf0_g" not in p.serving_sql
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


# --- the boundary: bare width-k expands to flat named columns (loop 3) ---------


def test_bare_wide_item_expands_to_alias_prefixed_columns():
    p = SQLProjection(
        "SELECT pca(struct_pack(a := age, f := fare)) OVER () AS e, name FROM __THIS__",
        transformers={"pca": PCA(n_components=2)},
    ).fit(TRAIN)
    out = p.transform(TRAIN)
    assert out.column_names == ["e_pca0", "e_pca1", "name"]
    ref = clone_ref(PCA(n_components=2), TRAIN)
    got = {r["name"]: (r["e_pca0"], r["e_pca1"]) for r in out.to_pylist()}
    for i, n in enumerate(TRAIN.column("name").to_pylist()):
        np.testing.assert_allclose(got[n], ref[i], rtol=1e-9)


def test_bare_wide_item_uses_declared_names():
    from sql_transform import Named

    p = SQLProjection(
        "SELECT pca(struct_pack(a := age, f := fare)) OVER () AS e FROM __THIS__",
        transformers={"pca": Named(PCA(n_components=2), returns=("size", "cost"))},
    ).fit(TRAIN)
    assert p.transform(TRAIN).column_names == ["e_size", "e_cost"]


def test_bare_wide_item_serves_row_at_a_time():
    p = SQLProjection(
        "SELECT ohe(struct_pack(color := color)) OVER () AS oh, name FROM __THIS__",
        transformers={"ohe": _ohe()},
    ).fit(TRAIN)
    want = p.transform(TRAIN)
    assert want.column_names == ["oh_color_blue", "oh_color_red", "name"]
    got = [r.model_dump() for r in p.infer_batch(TRAIN.to_pylist())]
    assert got == want.to_pylist()


def test_bare_width_one_item_stays_scalar():
    p = SQLProjection(
        "SELECT sc(struct_pack(a := age)) OVER () AS z, name FROM __THIS__",
        transformers={"sc": StandardScaler()},
    ).fit(TRAIN)
    assert p.transform(TRAIN).column_names == ["z", "name"]
    assert isinstance(p.transform(TRAIN).to_pylist()[0]["z"], float)


def test_wide_call_inside_an_expression_refuses_at_fit():
    p = SQLProjection(
        "SELECT pca(struct_pack(a := age, f := fare)) OVER () AS e, age + 1 AS b"
        " FROM __THIS__",
        transformers={"pca": PCA(n_components=2)},
    )
    ok = p.fit(TRAIN)  # bare item is fine
    assert ok.transform(TRAIN).column_names == ["e_pca0", "e_pca1", "b"]

    from sql_transform import MarginalizeError as _E

    p2 = SQLProjection(
        "SELECT list_extract(pca(struct_pack(a := age, f := fare)) OVER (), 1) AS e"
        " FROM __THIS__",
        transformers={"pca": PCA(n_components=2)},
    )
    with pytest.raises(_E, match="width-2 transformer call inside an expression"):
        p2.fit(TRAIN)


def test_unseen_group_nulls_every_expanded_column():
    p = SQLProjection(
        "SELECT pca(struct_pack(a := age, f := fare)) OVER (PARTITION BY country) AS e,"
        " name FROM __THIS__",
        transformers={"pca": PCA(n_components=2)},
    ).fit(TRAIN)
    assert p.transform(TRAIN).column_names == ["e_pca0", "e_pca1", "name"]
    out = p.infer(
        {"color": "red", "country": "JP", "age": 1.0, "fare": 1.0, "name": "q"}
    )
    # Flattening trades away P16's NULL-list vs list-of-NULLs distinction:
    # an unseen group is k NULL columns (DRAFT-24, a deliberate consequence).
    assert out.e_pca0 is None and out.e_pca1 is None


def test_unknown_field_name_refuses_at_fit():
    p = SQLProjection(
        "SELECT pca(struct_pack(a := age, f := fare)) OVER ().nope AS z FROM __THIS__",
        transformers={"pca": PCA(n_components=2)},
    )
    with pytest.raises(MarginalizeError, match="no output field 'nope'"):
        p.fit(TRAIN)


# --- Named(...): the author's declaration wins (DRAFT-24 loop 2) ---------------


def test_named_override_replaces_generated_names():
    from sql_transform import Named

    p = SQLProjection(
        "SELECT pca(struct_pack(a := age, f := fare)) OVER ().size AS s,"
        " pca(struct_pack(a := age, f := fare)) OVER ().cost AS c, name FROM __THIS__",
        transformers={"pca": Named(PCA(n_components=2), returns=("size", "cost"))},
    ).fit(TRAIN)
    assert p.udfs["__cf_tf0"].return_names == ("size", "cost")
    ref = clone_ref(PCA(n_components=2), TRAIN)
    got = {r["name"]: (r["s"], r["c"]) for r in p.transform(TRAIN).to_pylist()}
    for i, n in enumerate(TRAIN.column("name").to_pylist()):
        np.testing.assert_allclose(got[n], ref[i], rtol=1e-9)


def test_named_override_serves_row_at_a_time():
    from sql_transform import Named

    p = SQLProjection(
        "SELECT pca(struct_pack(a := age, f := fare)) OVER ().size AS s, name"
        " FROM __THIS__",
        transformers={"pca": Named(PCA(n_components=2), returns=("size", "cost"))},
    ).fit(TRAIN)
    assert [r.model_dump() for r in p.infer_batch(TRAIN.to_pylist())] == (
        p.transform(TRAIN).to_pylist()
    )


def test_named_override_clones_per_group():
    from sql_transform import Named

    p = SQLProjection(
        "SELECT sc(struct_pack(a := age)) OVER (PARTITION BY country).z AS z, name"
        " FROM __THIS__",
        transformers={"sc": Named(StandardScaler(), returns=("z",))},
    ).fit(TRAIN)
    (step,) = [s for s in p.plan if s.kind == "fit"]
    base = p.udfs["__cf_tf0"]
    assert len(base.instances) == 2  # one fitted clone per country
    assert base.instances[0].estimator is not base.instances[1].estimator
    got = {r["name"]: r["z"] for r in p.transform(TRAIN).to_pylist()}
    assert got["x"] == 1.0 and got["y"] == -1.0  # US: mean 35, per-group z


def test_named_override_width_mismatch_refuses():
    from sql_transform import Named, UDFError

    p = SQLProjection(
        "SELECT pca(struct_pack(a := age, f := fare)) OVER ().a AS s FROM __THIS__",
        transformers={"pca": Named(PCA(n_components=2), returns=("a", "b", "c"))},
    )
    with pytest.raises(UDFError, match="declares 3 output names.*fits to width 2"):
        p.fit(TRAIN)


def test_named_override_cannot_paper_over_a_learned_width():
    from sql_transform import Named, UDFError

    # Declared width matches the first fit's vocabulary, not the second's —
    # exactly the case a fixed override must refuse rather than mislabel.
    p = SQLProjection(
        "SELECT ohe(struct_pack(color := color)) OVER ().is_red AS r FROM __THIS__",
        transformers={"ohe": Named(_ohe(), returns=("is_blue", "is_red"))},
    ).fit(TRAIN)
    assert p.transform(TRAIN).to_pylist()[0]["r"] == 1.0
    three = pa.table(
        {
            "color": ["red", "blue", "aqua"],
            "country": ["US"] * 3,
            "age": [1.0, 2.0, 3.0],
            "fare": [1.0, 2.0, 3.0],
            "name": list("abc"),
        }
    )
    with pytest.raises(UDFError, match="declares 2 output names.*fits to width 3"):
        p.fit(three)


def test_named_declaration_is_validated_eagerly():
    from sql_transform import Named, UDFError

    with pytest.raises(UDFError, match="duplicate output names"):
        Named(PCA(n_components=2), returns=("a", "a"))
    with pytest.raises(UDFError, match="at least one output name"):
        Named(PCA(n_components=2), returns=())


def test_computed_field_name_refuses():
    with pytest.raises(MarginalizeError, match="computed field name"):
        SQLProjection(
            "SELECT struct_extract(sc(struct_pack(a := age)) OVER (), name) AS z"
            " FROM __THIS__",
            transformers={"sc": StandardScaler()},
        )


def test_struct_literal_bundle_is_the_same_as_struct_pack():
    # DuckDB desugars {'a': x} to struct_pack(a := x) — same AST, so the
    # bundle rules and field names are identical. Pinned, not incidental.
    kw = {"transformers": {"pca": PCA(n_components=2)}}
    a = SQLProjection(
        "SELECT pca({'a': age, 'f': fare}) OVER ().pca0 AS c FROM __THIS__", **kw
    ).fit(TRAIN)
    b = SQLProjection(
        "SELECT pca(struct_pack(a := age, f := fare)) OVER ().pca0 AS c FROM __THIS__",
        **kw,
    ).fit(TRAIN)
    assert a.serving_sql == b.serving_sql
    assert a.transform(TRAIN).to_pylist() == b.transform(TRAIN).to_pylist()
    assert a.udfs["__cf_tf0"].take_names == ("a", "f")
