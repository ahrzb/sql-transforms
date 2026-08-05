"""Nested struct outputs — fit/transform-split slice 5 (engine-touching).

The output boundary learns struct columns on BOTH paths: a bare
transformer call as an output item serves its whole output struct
(measured: DuckDB already serves it via the arrow-typed UDF; confit
refused by name at specializer/frontend.rs — this slice teaches it to
emit the lanes as a struct value). C2 gates batch ≡ DuckDB; C3 gates
row ≡ batch. Spec: docs/superpowers/specs/
2026-08-05-fit-transform-split-design.md, slice 5.
"""

import numpy as np
import pydantic
import pytest
from sklearn.preprocessing import StandardScaler

from sql_transform import SQLProjection

from ._transformers_test import TRAIN, _reference

ROW = pydantic.create_model("Row", **dict.fromkeys(TRAIN.column_names, (object, None)))


def _fit(sql: str) -> SQLProjection:
    return SQLProjection(
        sql, this_model=ROW, transformers={"sc": StandardScaler()}
    ).fit(TRAIN)


def _ref_struct():
    """Per-row {'a': ..., 'f': ...} reference for the global 2-wide scaler."""
    feats = np.array(
        [TRAIN.column("age").to_pylist(), TRAIN.column("fare").to_pylist()],
        dtype=float,
    ).T
    ref = _reference(StandardScaler(), feats, [()] * TRAIN.num_rows)
    return [{"a": r[0], "f": r[1]} for r in ref]


def test_bare_call_serves_struct_column_batch():
    p = _fit("SELECT sc(struct_pack(a := age, f := fare)) AS s, name FROM __THIS__")
    got = p.transform(TRAIN).column("s").to_pylist()
    for row, want in zip(got, _ref_struct(), strict=True):
        assert set(row) == {"a", "f"}
        np.testing.assert_allclose(row["a"], want["a"], rtol=1e-12)
        np.testing.assert_allclose(row["f"], want["f"], rtol=1e-12)


def test_bare_call_serves_struct_column_row_path():
    """C3: infer emits the same struct the batch path serves."""
    p = _fit("SELECT sc(struct_pack(a := age, f := fare)) AS s, name FROM __THIS__")
    want = _ref_struct()[0]
    row = p.infer({"country": "US", "age": 40.0, "fare": 7.0, "name": "x"})
    got = row.s if not isinstance(row.s, dict) else row.s
    got = dict(got) if not isinstance(got, dict) else got
    np.testing.assert_allclose(got["a"], want["a"], rtol=1e-12)
    np.testing.assert_allclose(got["f"], want["f"], rtol=1e-12)


def test_struct_output_and_field_read_share_one_fit():
    p = _fit(
        "SELECT sc(struct_pack(a := age, f := fare)) AS s,"
        " sc(struct_pack(a := age, f := fare)).a AS za, name FROM __THIS__"
    )
    assert len([st for st in p.plan if st.kind == "fit"]) == 1
    out = p.transform(TRAIN)
    ref = _ref_struct()
    for i in range(TRAIN.num_rows):
        np.testing.assert_allclose(
            out.column("za").to_pylist()[i], ref[i]["a"], rtol=1e-12
        )
        np.testing.assert_allclose(
            out.column("s").to_pylist()[i]["a"], ref[i]["a"], rtol=1e-12
        )


def test_split_spelling_serves_struct_column():
    p = _fit(
        "SELECT sc_transform(sc_fit(struct_pack(a := age, f := fare))"
        " OVER (PARTITION BY country), struct_pack(a := age, f := fare)) AS s,"
        " name FROM __THIS__"
    )
    feats = np.array(
        [TRAIN.column("age").to_pylist(), TRAIN.column("fare").to_pylist()],
        dtype=float,
    ).T
    ref = _reference(
        StandardScaler(), feats, [(c,) for c in TRAIN.column("country").to_pylist()]
    )
    got = p.transform(TRAIN).column("s").to_pylist()
    for i in range(TRAIN.num_rows):
        np.testing.assert_allclose(got[i]["a"], ref[i][0], rtol=1e-12)
        np.testing.assert_allclose(got[i]["f"], ref[i][1], rtol=1e-12)


def test_star_over_a_call_is_the_oracles_parser_error():
    # Measured 2026-08-05: `call.*` / `(call).*` are DuckDB PARSER errors —
    # the star-over-expression spelling does not exist in the oracle. The
    # lawful expansion spelling is unnest (below).
    from sql_transform import MarginalizeError

    with pytest.raises(MarginalizeError, match="parse error"):
        _fit("SELECT sc(struct_pack(a := age, f := fare)).*, name FROM __THIS__")


@pytest.mark.xfail(
    strict=True,
    reason="slice-5 follow-up: unnest over a struct-valued transformer call"
    " expands to per-field columns named by the LEARNED field names"
    " (measured: alias ignored, one column per field, in place)",
)
def test_unnest_expands_struct_output():
    p = _fit("SELECT unnest(sc(struct_pack(a := age, f := fare))), name FROM __THIS__")
    out = p.transform(TRAIN)
    assert out.column_names == ["a", "f", "name"]
