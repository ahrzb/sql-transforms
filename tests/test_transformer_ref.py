"""Differential parity for fitted transformers referenced as {ref} in a t-string."""

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from sklearn.decomposition import PCA
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from sql_transform import SQLTransform

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
