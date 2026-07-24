"""Differential parity for fitted transformers referenced as {ref} in a t-string."""

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from sklearn.decomposition import PCA
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from sql_transform import SQLTransform, _transformer_ref

# The nameless-input warning is a known false positive (see test_transformer_udf).
pytestmark = pytest.mark.filterwarnings(
    "ignore:X does not have valid feature names:UserWarning"
)


def _both_engines(t, test_df):
    """transform (DataFusion) and infer (Rust) as plain dicts; assert equal."""
    batch = t.transform(pa.Table.from_pandas(test_df)).to_pylist()
    infer = [r.model_dump() for r in t.infer_batch(test_df.to_dict("records"))]
    assert infer == batch, (infer, batch)
    return batch


def test_single_scaler_ref_parity():
    train = pd.DataFrame(
        {"age": [10.0, 20.0, 30.0, 40.0], "income": [1.0, 2.0, 3.0, 4.0]}
    )
    sc = StandardScaler().fit(train)
    t = SQLTransform(t"SELECT {sc}(age, income) AS out FROM __THIS__").fit(
        pa.Table.from_pandas(train)
    )

    test = pd.DataFrame({"age": [25.0, 35.0], "income": [2.5, 3.5]})
    batch = _both_engines(t, test)

    expected = sc.transform(test)
    got = np.array([[b["out"]["age"], b["out"]["income"]] for b in batch])
    assert np.allclose(got, expected)


@pytest.mark.parametrize(
    "sql_of",
    [
        pytest.param(
            lambda sc: t"SELECT age FROM __THIS__ QUALIFY {sc}(age) > 1", id="qualify"
        ),
        pytest.param(
            lambda sc: t"SELECT DISTINCT ON ({sc}(age)) age FROM __THIS__",
            id="distinct_on",
        ),
        pytest.param(
            lambda sc: t"SELECT age FROM __THIS__ CLUSTER BY {sc}(age)", id="cluster_by"
        ),
        pytest.param(
            lambda sc: t"SELECT age FROM __THIS__ SORT BY {sc}(age)", id="sort_by"
        ),
        pytest.param(
            lambda sc: (
                t"SELECT AVG(age) OVER w AS a FROM __THIS__ "
                t"WINDOW w AS (PARTITION BY {sc}(age))"
            ),
            id="window_clause",
        ),
    ],
)
def test_transformer_call_outside_projection_raises(sql_of):
    # TASK-2 AC#1. The native engine resolves transformer calls only over the
    # projection (src/lib.rs); DataFusion plans the whole statement. These five
    # clauses survive parse_and_validate, so before this guard fit() ACCEPTED
    # them and the disagreement surfaced late and asymmetrically: transform()
    # raised a DataFusion planning error while infer() silently ignored the
    # clause and returned rows (WINDOW crashed at fit with a bare
    # AttributeError). Reject at build time with one clear message instead.
    # ORDER BY/WHERE/GROUP BY/HAVING/LIMIT/JOIN are already rejected upstream
    # by parse_and_validate, so they cannot reach this guard.
    train = pd.DataFrame({"age": [10.0, 20.0, 30.0, 40.0]})
    sc = StandardScaler().fit(train)
    t = SQLTransform(sql_of(sc))
    with pytest.raises(ValueError, match="projection"):
        t.fit(pa.Table.from_pandas(train))


def test_nested_threading_parity():
    train = pd.DataFrame(
        {"age": [10.0, 20.0, 30.0, 40.0], "income": [1.0, 2.0, 3.0, 4.0]}
    )
    sc = StandardScaler().fit(train)
    # Wrap as a DataFrame (not the bare ndarray sc.transform returns) so PCA's
    # fit records feature_names_in_ == sc.get_feature_names_out() -- required
    # for is_transformer(pca) to hold and for the nested schema match.
    scaled = pd.DataFrame(sc.transform(train), columns=sc.get_feature_names_out())
    pca = PCA(n_components=1).fit(scaled)
    t = SQLTransform(t"SELECT {pca}({sc}(age, income)) AS out FROM __THIS__").fit(
        pa.Table.from_pandas(train)
    )

    test = pd.DataFrame({"age": [25.0, 35.0], "income": [2.5, 3.5]})
    batch = _both_engines(t, test)

    expected = pca.transform(sc.transform(test))
    out_names = [str(n) for n in pca.get_feature_names_out()]
    got = np.array([[b["out"][n] for n in out_names] for b in batch])
    assert np.allclose(got, expected)


def test_ordinal_encoder_ref_string_in_int_out():
    train = pd.DataFrame(
        {"color": ["red", "green", "blue", "red"], "size": ["S", "M", "L", "M"]}
    )
    enc = OrdinalEncoder(dtype=np.int64).fit(train)
    t = SQLTransform(t"SELECT {enc}(color, size) AS out FROM __THIS__").fit(
        pa.Table.from_pandas(train)
    )

    test = pd.DataFrame({"color": ["blue", "red"], "size": ["L", "S"]})
    batch = _both_engines(t, test)

    expected = enc.transform(test)
    got = np.array([[b["out"]["color"], b["out"]["size"]] for b in batch])
    assert (got == expected).all()


def test_transformer_and_native_window_agg_coexist():
    # A native window agg over __THIS__ alongside a transformer call: proves the
    # two compose without either engine choking. No agg reads the transformer output.
    train = pd.DataFrame(
        {"age": [10.0, 20.0, 30.0, 40.0], "income": [1.0, 2.0, 3.0, 4.0]}
    )
    sc = StandardScaler().fit(train[["age", "income"]])
    t = SQLTransform(
        t"SELECT {sc}(age, income) AS out, age / (MEAN(age) OVER ()) AS an FROM __THIS__"  # noqa: E501
    ).fit(pa.Table.from_pandas(train))

    test = pd.DataFrame({"age": [25.0, 35.0], "income": [2.5, 3.5]})
    batch = _both_engines(t, test)
    assert np.allclose(
        [b["an"] for b in batch], test["age"].to_numpy() / train["age"].mean()
    )


def test_camelcase_columns_quoted_compose():
    # A transformer ref on case-sensitive columns works when the user quotes them
    # (DataFusion keeps a quoted identifier case-exact). The ref carries the
    # quoting through verbatim, so both engines resolve `"LotArea"` identically.
    train = pd.DataFrame(
        {"LotArea": [1.0, 2.0, 3.0, 4.0], "YearBuilt": [1990.0, 2000.0, 2010.0, 2020.0]}
    )
    sc = StandardScaler().fit(train)
    t = SQLTransform(t'SELECT {sc}("LotArea", "YearBuilt") AS out FROM __THIS__').fit(
        pa.Table.from_pandas(train)
    )

    test = pd.DataFrame({"LotArea": [2.5], "YearBuilt": [2005.0]})
    batch = _both_engines(t, test)

    expected = sc.transform(test)
    got = np.array([[b["out"]["LotArea"], b["out"]["YearBuilt"]] for b in batch])
    assert np.allclose(got, expected)


def test_camelcase_columns_unquoted_folds_and_fails():
    # Unquoted CamelCase folds to lowercase in DataFusion (the oracle) and misses,
    # so the transformer-ref path must NOT silently make it work (the reverted
    # TASK-25 force-quote did). The user quotes instead (see the quoted test).
    # TASK-28.
    train = pd.DataFrame(
        {"LotArea": [1.0, 2.0, 3.0, 4.0], "YearBuilt": [1990.0, 2000.0, 2010.0, 2020.0]}
    )
    sc = StandardScaler().fit(train)
    t = SQLTransform(t"SELECT {sc}(LotArea, YearBuilt) AS out FROM __THIS__")
    with pytest.raises(Exception, match="(?i)lotarea|no field|schema"):
        fitted = t.fit(pa.Table.from_pandas(train))
        fitted.transform(pa.Table.from_pandas(train))


def test_ndarray_fit_transformer_binds_positionally():
    # sklearn records feature_names_in_ only for DataFrame fit. An ndarray-fit
    # transformer has no names, so arguments bind positionally in call order.
    train = pd.DataFrame(
        {"age": [10.0, 20.0, 30.0, 40.0], "income": [1.0, 2.0, 3.0, 4.0]}
    )
    sc = StandardScaler().fit(train.to_numpy())
    assert not hasattr(sc, "feature_names_in_")

    t = SQLTransform(t"SELECT {sc}(age, income) AS out FROM __THIS__").fit(
        pa.Table.from_pandas(train)
    )
    batch = _both_engines(t, train)
    expected = sc.transform(train.to_numpy())
    got = np.array([[b["out"]["age"], b["out"]["income"]] for b in batch])
    assert np.allclose(got, expected)

    # clone contract: the user's object must be left untouched
    assert not hasattr(sc, "feature_names_in_")


def test_ndarray_fit_arity_mismatch_raises():
    train = pd.DataFrame(
        {"age": [10.0, 20.0, 30.0, 40.0], "income": [1.0, 2.0, 3.0, 4.0]}
    )
    sc = StandardScaler().fit(train.to_numpy())  # n_features_in_ == 2
    t = SQLTransform(t"SELECT {sc}(age) AS out FROM __THIS__")
    with pytest.raises(ValueError, match="bind positionally"):
        t.fit(pa.Table.from_pandas(train))


def test_named_fit_column_mismatch_still_raises():
    # The named path is unchanged: names are validated as a set.
    train = pd.DataFrame(
        {"age": [10.0, 20.0, 30.0, 40.0], "income": [1.0, 2.0, 3.0, 4.0]}
    )
    sc = StandardScaler().fit(train)
    t = SQLTransform(t"SELECT {sc}(age) AS out FROM __THIS__")
    with pytest.raises(ValueError, match="must match feature_names_in_"):
        t.fit(pa.Table.from_pandas(train))


class _SpyTransformer:
    """Wraps a fitted transformer and counts .transform() calls."""

    def __init__(self, obj):
        self._obj = obj
        self.calls = 0
        self.feature_names_in_ = obj.feature_names_in_
        self.n_features_in_ = obj.n_features_in_

    def transform(self, x):
        self.calls += 1
        return self._obj.transform(x)

    def get_feature_names_out(self):
        return self._obj.get_feature_names_out()


def test_leaf_ref_probes_transform_once():
    # A leaf ref's materialised output is only ever read by an OUTER ref's probe.
    # With no outer, materialising it is a discarded .transform() call.
    train = pd.DataFrame(
        {"age": [10.0, 20.0, 30.0, 40.0], "income": [1.0, 2.0, 3.0, 4.0]}
    )
    spy = _SpyTransformer(StandardScaler().fit(train))

    SQLTransform(t"SELECT {spy}(age, income) AS out FROM __THIS__").fit(
        pa.Table.from_pandas(train)
    )
    assert spy.calls == 1, f"expected a single probe, got {spy.calls}"


def test_nested_refs_probe_once_each():
    # The inner IS consumed, so it must still be materialised -- but from the
    # probe's own output, not a second .transform() call.
    train = pd.DataFrame(
        {"age": [10.0, 20.0, 30.0, 40.0], "income": [1.0, 2.0, 3.0, 4.0]}
    )
    sc = StandardScaler().fit(train)
    scaled = pd.DataFrame(sc.transform(train), columns=sc.get_feature_names_out())
    inner = _SpyTransformer(sc)
    outer = _SpyTransformer(PCA(n_components=1).fit(scaled))

    SQLTransform(t"SELECT {outer}({inner}(age, income)) AS out FROM __THIS__").fit(
        pa.Table.from_pandas(train)
    )
    assert (inner.calls, outer.calls) == (1, 1), (inner.calls, outer.calls)


def test_leaf_ref_skips_materialization(monkeypatch):
    # The `consumed` gate saves a _table_from_probe call (a table allocation),
    # not a .transform() call -- _table_from_probe reuses the probe's `y` and
    # never calls .transform. So the spy tests above cannot see this gate: they
    # count .transform(), which is identical whether or not the gate exists.
    # Count _table_from_probe calls directly instead.
    calls = []
    orig = _transformer_ref._table_from_probe

    def counting(y, out_schema):
        calls.append(1)
        return orig(y, out_schema)

    monkeypatch.setattr(_transformer_ref, "_table_from_probe", counting)

    train = pd.DataFrame(
        {"age": [10.0, 20.0, 30.0, 40.0], "income": [1.0, 2.0, 3.0, 4.0]}
    )
    sc = StandardScaler().fit(train)

    # Leaf-only: no outer consumer, so the materialised table is never built.
    SQLTransform(t"SELECT {sc}(age, income) AS out FROM __THIS__").fit(
        pa.Table.from_pandas(train)
    )
    assert len(calls) == 0, f"expected 0 materializations for a leaf, got {len(calls)}"

    calls.clear()
    scaled = pd.DataFrame(sc.transform(train), columns=sc.get_feature_names_out())
    pca = PCA(n_components=1).fit(scaled)

    # Nested: the inner ref IS consumed by the outer, so it must be built once.
    SQLTransform(t"SELECT {pca}({sc}(age, income)) AS out FROM __THIS__").fit(
        pa.Table.from_pandas(train)
    )
    assert len(calls) == 1, (
        f"expected 1 materialization for the inner, got {len(calls)}"
    )
