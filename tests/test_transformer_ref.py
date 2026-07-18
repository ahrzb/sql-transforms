"""Differential parity for fitted transformers referenced as {ref} in a t-string."""

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

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
    train = pd.DataFrame({"age": [10.0, 20.0, 30.0, 40.0], "income": [1.0, 2.0, 3.0, 4.0]})
    sc = StandardScaler().fit(train)
    t = SQLTransform(
        t"SELECT {sc}(age, income) AS out FROM __THIS__"
    ).fit(pa.Table.from_pandas(train))

    test = pd.DataFrame({"age": [25.0, 35.0], "income": [2.5, 3.5]})
    batch = _both_engines(t, test)

    expected = sc.transform(test)
    got = np.array([[b["out"]["age"], b["out"]["income"]] for b in batch])
    assert np.allclose(got, expected)


def test_nested_threading_parity():
    train = pd.DataFrame({"age": [10.0, 20.0, 30.0, 40.0], "income": [1.0, 2.0, 3.0, 4.0]})
    sc = StandardScaler().fit(train)
    # Wrap as a DataFrame (not the bare ndarray sc.transform returns) so PCA's
    # fit records feature_names_in_ == sc.get_feature_names_out() -- required
    # for is_transformer(pca) to hold and for the nested schema match.
    scaled = pd.DataFrame(sc.transform(train), columns=sc.get_feature_names_out())
    pca = PCA(n_components=1).fit(scaled)
    t = SQLTransform(
        t"SELECT {pca}({sc}(age, income)) AS out FROM __THIS__"
    ).fit(pa.Table.from_pandas(train))

    test = pd.DataFrame({"age": [25.0, 35.0], "income": [2.5, 3.5]})
    batch = _both_engines(t, test)

    expected = pca.transform(sc.transform(test))
    out_names = [str(n) for n in pca.get_feature_names_out()]
    got = np.array([[b["out"][n] for n in out_names] for b in batch])
    assert np.allclose(got, expected)
