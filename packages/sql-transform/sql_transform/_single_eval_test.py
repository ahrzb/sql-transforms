"""Single-evaluation field access (TASK-63, DRAFT-24 loop 4).

k addressed fields of one fitted transformer cost ONE ``transform()`` call
per row on BOTH serving paths: the batch path (DuckDB — one struct-returning
call; its CSE merges the identical mentions) and the row path (confit — k
lane reads off one ecall). These tests COUNT real ``transform()``
invocations; timing is not proof. Groups alternate row-by-row so any
cross-group state sharing would corrupt the values against the independent
clone-per-group reference.
"""

import numpy as np
import pyarrow as pa
from sklearn.base import clone
from sklearn.decomposition import PCA

from sql_transform import SQLProjection

TRAIN = pa.table(
    {
        "grp": ["u", "v", "u", "v", "u", "v"],
        "a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "b": [9.0, 7.0, 4.0, 1.0, 8.0, 3.0],
        "name": ["r0", "r1", "r2", "r3", "r4", "r5"],
    }
)

TWO_FIELDS = (
    "SELECT pca_transform(pca_fit(struct_pack(a := a, b := b))"
    " OVER (PARTITION BY grp), struct_pack(a := a, b := b)).pca0 AS x,"
    " pca_transform(pca_fit(struct_pack(a := a, b := b))"
    " OVER (PARTITION BY grp), struct_pack(a := a, b := b)).pca1 AS y,"
    " name FROM __THIS__"
)


class CountingPCA(PCA):
    """PCA counting ``transform()`` calls across every per-group clone."""

    calls = 0

    def transform(self, X):
        type(self).calls += 1
        return super().transform(X)

    def get_feature_names_out(self, input_features=None):
        # Keep PCA's own field names — the subclass would otherwise learn
        # countingpca0/1 (sklearn derives them from the class name).
        return np.asarray([f"pca{i}" for i in range(self.n_components)])


def _reference(feats, keys):
    """Independent clone-per-group oracle, row-aligned (C4's shape)."""
    groups: dict = {}
    for i, k in enumerate(keys):
        groups.setdefault(k, []).append(i)
    out = [None] * len(keys)
    for idx in groups.values():
        est = clone(PCA(n_components=2))
        block = np.asarray(est.fit(feats[idx]).transform(feats[idx]))
        for row, vals in zip(idx, block, strict=True):
            out[row] = [float(v) for v in vals]
    return out


def _fitted_two_fields():
    p = SQLProjection(
        TWO_FIELDS, transformers={"pca": CountingPCA(n_components=2)}
    ).fit(TRAIN)
    CountingPCA.calls = 0  # fit probes transform once per group; not measured
    return p


def test_two_fields_cost_one_call_per_row_on_the_batch_path():
    p = _fitted_two_fields()
    out = p.transform(TRAIN)
    assert CountingPCA.calls == TRAIN.num_rows, (
        f"batch path: {CountingPCA.calls} transform() calls for"
        f" {TRAIN.num_rows} rows — k fields must share one call"
    )
    feats = np.array(
        [TRAIN.column("a").to_pylist(), TRAIN.column("b").to_pylist()], dtype=float
    ).T
    ref = _reference(feats, TRAIN.column("grp").to_pylist())
    xs = out.column("x").to_pylist()
    ys = out.column("y").to_pylist()
    for i in range(TRAIN.num_rows):
        np.testing.assert_allclose([xs[i], ys[i]], ref[i], rtol=1e-9, atol=1e-12)


def test_two_fields_cost_one_call_per_row_on_the_row_path():
    p = _fitted_two_fields()
    rows = TRAIN.to_pylist()
    got = [r.model_dump() for r in p.infer_batch(rows)]
    assert CountingPCA.calls == len(rows), (
        f"row path: {CountingPCA.calls} transform() calls for"
        f" {len(rows)} rows — k fields must share one call"
    )
    CountingPCA.calls = 0
    one = p.infer(rows[0]).model_dump()
    assert CountingPCA.calls == 1, "one row, two fields: exactly one call"
    assert one == got[0]
    # Interleaved groups: values must match the clone-per-group reference —
    # sharing state across the alternating instance ids would corrupt them.
    feats = np.array(
        [TRAIN.column("a").to_pylist(), TRAIN.column("b").to_pylist()], dtype=float
    ).T
    ref = _reference(feats, TRAIN.column("grp").to_pylist())
    for i, r in enumerate(got):
        np.testing.assert_allclose([r["x"], r["y"]], ref[i], rtol=1e-9, atol=1e-12)


def test_bare_wide_item_refuses_at_construction():
    # Struct-valued calls: the flat expansion is gone; a bare item refuses
    # until DRAFT-25's nested outputs. (Counting for the field-read shape
    # is covered above — the bare shape no longer exists to count.)
    import pytest

    from sql_transform import MarginalizeError

    with pytest.raises(MarginalizeError, match="struct value"):
        SQLProjection(
            "SELECT pca_transform(pca_fit(struct_pack(a := a, b := b))"
            " OVER (PARTITION BY grp), struct_pack(a := a, b := b)) AS e,"
            " name FROM __THIS__",
            transformers={"pca": CountingPCA(n_components=2)},
        )
