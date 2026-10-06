"""Struct-valued estimator outputs and explicitly named field projections."""

import numpy as np
import pyarrow as pa
import pytest
from sklearn.preprocessing import StandardScaler

from sql_transform import FittedProjection, SQLProjection, TransformError

from ._transformers_test import TRAIN, _reference


def _fit(sql: str) -> FittedProjection:
    return SQLProjection.marginalize(sql, captured={"sc": StandardScaler()}).fit(TRAIN)


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
    row = p.compile().infer_rows(
        [{"country": "US", "age": 40.0, "fare": 7.0, "name": "x"}]
    )[0]
    got = row["s"]
    np.testing.assert_allclose(got["a"], want["a"], rtol=1e-12)
    np.testing.assert_allclose(got["f"], want["f"], rtol=1e-12)


def test_struct_output_and_field_read_share_one_fit():
    p = _fit(
        "SELECT sc(struct_pack(a := age, f := fare)) AS s,"
        " sc(struct_pack(a := age, f := fare)).a AS za, name FROM __THIS__"
    )
    assert len(p.instances) == 1
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


def test_unseen_group_serves_null_whole_struct_both_paths():
    """P14: an unseen group misses the params LEFT JOIN — the WHOLE struct
    is NULL on both paths, distinct from a struct of NULLs."""
    p = _fit(
        "SELECT sc_transform(sc_fit(struct_pack(a := age, f := fare))"
        " OVER (PARTITION BY country), struct_pack(a := age, f := fare)) AS s,"
        " name FROM __THIS__"
    )
    unseen = pa.table({"country": ["JP"], "age": [33.0], "fare": [4.0], "name": ["q"]})
    assert p.transform(unseen).column("s").to_pylist() == [None]
    row = {"country": "JP", "age": 33.0, "fare": 4.0, "name": "q"}
    assert p.compile().infer_rows([row])[0]["s"] is None


def test_underscore_fitted_field_serves_whole_struct():
    # Dict outputs and Arrow structs keep underscore-leading fields visible;
    # the former pydantic output model treated them as private attributes.
    p = _fit('SELECT sc(struct_pack("_a" := age, f := fare)) AS s, name FROM __THIS__')
    ref = _ref_struct()
    got = p.transform(TRAIN).column("s").to_pylist()
    for row, want in zip(got, ref, strict=True):
        assert set(row) == {"_a", "f"}
        np.testing.assert_allclose(row["_a"], want["a"], rtol=1e-12)
        np.testing.assert_allclose(row["f"], want["f"], rtol=1e-12)
    row0 = p.compile().infer_rows(
        [{"country": "US", "age": 40.0, "fare": 7.0, "name": "x"}]
    )[0]
    np.testing.assert_allclose(row0["s"]["_a"], ref[0]["a"], rtol=1e-12)
    np.testing.assert_allclose(row0["s"]["f"], ref[0]["f"], rtol=1e-12)


def test_underscore_fitted_field_still_fits_for_field_reads():
    p = _fit(
        'SELECT sc(struct_pack("_a" := age, f := fare)).f AS zf, name FROM __THIS__'
    )
    assert isinstance(p.transform(TRAIN).column("zf").to_pylist()[0], float)


def test_previously_pydantic_reserved_fitted_field_serves():
    # These names were reserved by the former pydantic output model, not by
    # dict outputs or Arrow structs; neither binding may drop or reject them.
    ref = _ref_struct()
    for member in ("model_config", "model_validate", "__init__"):
        p = _fit(
            f'SELECT sc(struct_pack("{member}" := age, f := fare)) AS s,'
            " name FROM __THIS__"
        )
        got = p.transform(TRAIN).column("s").to_pylist()
        for row, want in zip(got, ref, strict=True):
            assert set(row) == {member, "f"}
            np.testing.assert_allclose(row[member], want["a"], rtol=1e-12)
            np.testing.assert_allclose(row["f"], want["f"], rtol=1e-12)
        row0 = p.compile().infer_rows(
            [{"country": "US", "age": 40.0, "fare": 7.0, "name": "x"}]
        )[0]
        np.testing.assert_allclose(row0["s"][member], ref[0]["a"], rtol=1e-12)


def test_distinct_on_the_call_refuses_at_construction():
    # DuckDB binds DISTINCT only on aggregates, not registered scalar calls;
    # refusing each spelling preserves that binding boundary.
    for sql in [
        "SELECT sc(DISTINCT age) AS s FROM __THIS__",
        "SELECT sc(DISTINCT age).age AS z FROM __THIS__",
        "SELECT sc_transform(DISTINCT sc_fit(age) OVER (), age) AS s FROM __THIS__",
    ]:
        with pytest.raises(TransformError, match="DISTINCT|modifier|scalar bundle"):
            _fit(sql)


def test_star_over_a_call_is_the_oracles_parser_error():
    with pytest.raises(TransformError, match="parse|Parser"):
        _fit("SELECT sc(struct_pack(a := age, f := fare)).*, name FROM __THIS__")


def test_migrated_struct_expansion_uses_named_fields():
    # Automatic transformer unnest is removed; select its computed fields explicitly.
    p = _fit(
        "SELECT sc(struct_pack(a := age, f := fare)).a AS a,"
        " sc(struct_pack(a := age, f := fare)).f AS f, name FROM __THIS__"
    )
    out = p.transform(TRAIN)
    assert out.column_names == ["a", "f", "name"]
    ref = _ref_struct()
    for i in range(TRAIN.num_rows):
        np.testing.assert_allclose(out.column("a")[i].as_py(), ref[i]["a"], rtol=1e-12)
        np.testing.assert_allclose(out.column("f")[i].as_py(), ref[i]["f"], rtol=1e-12)
    compiled = p.compile()
    assert compiled.infer_rows(TRAIN.to_pylist()) == out.to_pylist()
    assert compiled.infer_arrow(TRAIN).to_pylist() == out.to_pylist()


def test_migrated_named_fields_and_repeated_read_share_fit():
    p = _fit(
        "SELECT sc(struct_pack(a := age, f := fare)).a AS a,"
        " sc(struct_pack(a := age, f := fare)).f AS f,"
        " sc(struct_pack(a := age, f := fare)).a AS ra, name FROM __THIS__"
    )
    assert len(p.instances) == 1
    out = p.transform(TRAIN)
    assert out.column_names == ["a", "f", "ra", "name"]
    assert out.column("a").to_pylist() == out.column("ra").to_pylist()
