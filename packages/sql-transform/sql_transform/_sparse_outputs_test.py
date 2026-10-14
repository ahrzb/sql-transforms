"""A transformer whose output is a scipy sparse matrix serves as its dense
configuration does: the step reads the output as `.toarray()` makes it
dense, at fit, where the probe counts the output fields a field read
names, and at each row (loops/native/decisions/closed/sparse-outputs.md,
the ruling). sklearn 1.9 picks the container per thread
(`sparse_interface`): a sparse matrix by default (`spmatrix`), a sparse
array on request (`sparray`); each test runs under both."""

from __future__ import annotations

import struct
from collections.abc import Callable
from typing import Any

import pyarrow as pa
import pytest
import sklearn
from scipy import sparse
from sklearn.ensemble import RandomTreesEmbedding
from sklearn.impute import MissingIndicator
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder, SplineTransformer

from sql_transform import SQLProjection

TRAIN = pa.table(
    {
        "color": ["red", "blue", "red", "green", "blue", "red", "green", "blue"],
        "side": ["l", "r", "r", "l", "l", "r", "l", "r"],
        "age": [40.0, 30.0, 52.5, 25.0, 61.5, 18.0, 33.0, 47.0],
        "fare": [7.0, 8.0, 6.0, 0.0, 9.5, 12.25, -3.0, 1.0],
        "name": ["a", "b", "c", "d", "e", "f", "g", "h"],
    }
)
HOLES = TRAIN.set_column(
    TRAIN.schema.get_field_index("age"),
    "age",
    pa.array([40.0, None, 52.5, 25.0, None, 18.0, 33.0, 47.0]),
)
LIVE = pa.table(
    {
        "color": ["green", "aqua", "red"],
        "side": ["r", "l", "l"],
        "age": [0.0, 99.0, -5.0],
        "fare": [-0.0, 30.0, 6.5],
        "name": ["x", "y", "z"],
    }
)
CONTAINERS = {"spmatrix": sparse.spmatrix, "sparray": sparse.sparray}


@pytest.fixture(params=list(CONTAINERS))
def interface(request, monkeypatch):
    """sklearn's sparse container, in this thread and in a thread an engine
    starts, which copies sklearn's global configuration at its first use."""
    monkeypatch.setitem(
        sklearn._config._global_config, "sparse_interface", request.param
    )
    with sklearn.config_context(sparse_interface=request.param):
        yield request.param


class _Seen(OneHotEncoder):
    """A `OneHotEncoder` that keeps each output it returns, to show which
    container the step was handed."""

    outputs: list[Any] = []

    def transform(self, X):
        out = super().transform(X)
        self.outputs.append(out)
        return out


def _fit(sql: str, est: Any, data: pa.Table = TRAIN) -> SQLProjection:
    return SQLProjection.marginalize(sql, captured={"t": est}).fit(data)


def _names(p: SQLProjection) -> tuple[str, ...]:
    return next(iter(p.udfs.values())).return_names


def _bits(t: pa.Table) -> list[Any]:
    """The rows with each double as its 8 bytes, so that -0.0 and NaN
    compare."""

    def b(v: Any) -> Any:
        if isinstance(v, float):
            return struct.pack("<d", v)
        if isinstance(v, dict):
            return {k: b(w) for k, w in v.items()}
        return v

    return [b(r) for r in t.to_pylist()]


def test_a_field_read_names_an_output_field_of_a_sparse_encoder(interface):
    # Before the step densified, the fit probe counted one output field
    # (`np.asarray` of a sparse matrix is a 0-d object array), and the
    # field read failed at fit.
    sql = "SELECT t(struct_pack(color := color)).color_red AS red, name FROM __THIS__"
    _Seen.outputs = []
    p = _fit(sql, _Seen(handle_unknown="ignore"))
    dense = _fit(sql, OneHotEncoder(sparse_output=False, handle_unknown="ignore"))
    assert _names(p) == _names(dense) == ("color_blue", "color_green", "color_red")
    for data in (TRAIN, LIVE):
        assert _bits(p.transform(data)) == _bits(dense.transform(data))
    assert p.transform(LIVE).column("red").to_pylist() == [0.0, 0.0, 1.0]
    assert _Seen.outputs
    assert all(isinstance(o, CONTAINERS[interface]) for o in _Seen.outputs)


def test_an_encoder_of_one_output_field_serves_it(interface):
    # `drop="if_binary"` over two categories: a 1-by-1 sparse matrix a row.
    sql = "SELECT t(struct_pack(side := side)).side_r AS r, name FROM __THIS__"
    p = _fit(sql, OneHotEncoder(drop="if_binary"))
    assert _names(p) == ("side_r",)
    assert p.transform(LIVE).column("r").to_pylist() == [1.0, 0.0, 0.0]
    dense = _fit(sql, OneHotEncoder(sparse_output=False, drop="if_binary"))
    assert _bits(p.transform(TRAIN)) == _bits(dense.transform(TRAIN))


PAIRS: list[tuple[str, Callable[[], Any], Callable[[], Any], pa.Table]] = [
    (
        "color := color",
        lambda: OneHotEncoder(handle_unknown="ignore"),
        lambda: OneHotEncoder(sparse_output=False, handle_unknown="ignore"),
        TRAIN,
    ),
    (
        "a := age, f := fare",
        lambda: KBinsDiscretizer(n_bins=3, strategy="uniform"),
        lambda: KBinsDiscretizer(n_bins=3, encode="onehot-dense", strategy="uniform"),
        TRAIN,
    ),
    (
        "a := age, f := fare",
        lambda: SplineTransformer(n_knots=4, sparse_output=True),
        lambda: SplineTransformer(n_knots=4),
        TRAIN,
    ),
    (
        "a := age, f := fare",
        lambda: MissingIndicator(features="all", sparse=True),
        lambda: MissingIndicator(features="all"),
        HOLES,
    ),
    (
        "a := age, f := fare",
        lambda: RandomTreesEmbedding(n_estimators=3, max_depth=2, random_state=0),
        lambda: RandomTreesEmbedding(
            n_estimators=3, max_depth=2, sparse_output=False, random_state=0
        ),
        TRAIN,
    ),
]


@pytest.mark.parametrize(
    "bundle, make, make_dense, data",
    PAIRS,
    ids=[
        "OneHotEncoder",
        "KBinsDiscretizer",
        "SplineTransformer",
        "MissingIndicator",
        "RandomTreesEmbedding",
    ],
)
def test_a_sparse_output_serves_as_its_dense_configuration(
    interface, bundle, make, make_dense, data
):
    sql = f"SELECT t(struct_pack({bundle})) AS s, name FROM __THIS__"
    p, dense = _fit(sql, make(), data), _fit(sql, make_dense(), data)
    assert (
        next(iter(p.udfs.values())).returns == next(iter(dense.udfs.values())).returns
    )
    assert _bits(p.transform(data)) == _bits(dense.transform(data))
