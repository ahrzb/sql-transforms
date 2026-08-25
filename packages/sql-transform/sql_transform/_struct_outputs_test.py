"""Nested struct outputs — fit/transform-split slice 5 (engine-touching).

The output boundary learns struct columns on BOTH paths: a bare
transformer call as an output item serves its whole output struct
(measured: DuckDB already serves it via the arrow-typed UDF; confit
refused by name at specializer/frontend.rs — this slice teaches it to
emit the lanes as a struct value). C2 gates batch ≡ DuckDB; C3 gates
row ≡ batch. Spec: docs/specs/2026-08-05-fit-transform-split-design.md,
slice 5.
"""

import numpy as np
import pyarrow as pa
import pytest
from sklearn.preprocessing import StandardScaler

from sql_transform import MarginalizeError, SQLProjection

from ._transformers_test import TRAIN, _reference

ROW = TRAIN.schema


def _fit(sql: str) -> SQLProjection:
    return SQLProjection(
        sql, this_schema=ROW, transformers={"sc": StandardScaler()}
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
    got = row["s"]
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
    assert p.infer(row)["s"] is None


def test_underscore_fitted_field_serves_whole_struct():
    # MIGRATION-NOTE: the old pydantic reserved-name refusal ("row-path model
    # boundary" — pydantic silently reclassified _-leading create_model
    # kwargs as private attributes) died with the pydantic output model. A
    # learned field named "_a" is an ordinary dict/struct key now, on both
    # engines, so the fit that used to refuse here now succeeds and serves.
    p = _fit('SELECT sc(struct_pack("_a" := age, f := fare)) AS s, name FROM __THIS__')
    ref = _ref_struct()
    got = p.transform(TRAIN).column("s").to_pylist()
    for row, want in zip(got, ref, strict=True):
        assert set(row) == {"_a", "f"}
        np.testing.assert_allclose(row["_a"], want["a"], rtol=1e-12)
        np.testing.assert_allclose(row["f"], want["f"], rtol=1e-12)
    row0 = p.infer({"country": "US", "age": 40.0, "fare": 7.0, "name": "x"})
    np.testing.assert_allclose(row0["s"]["_a"], ref[0]["a"], rtol=1e-12)
    np.testing.assert_allclose(row0["s"]["f"], ref[0]["f"], rtol=1e-12)


def test_underscore_fitted_field_still_fits_for_field_reads():
    # The refusal is scoped to whole-value serving: a field read never
    # turns the learned name into a pydantic field, so it keeps working.
    p = _fit(
        'SELECT sc(struct_pack("_a" := age, f := fare)).f AS zf, name FROM __THIS__'
    )
    assert isinstance(p.transform(TRAIN).column("zf").to_pylist()[0], float)


def test_previously_pydantic_reserved_fitted_field_serves():
    # MIGRATION-NOTE: names pydantic used to reserve (config/protected
    # namespaces, dunders) would raw-crash or silently vanish at first infer
    # against the old output model, so the fit-time probe refused them by
    # name. Dict-out and arrow structs don't reserve any name, so the fit
    # now succeeds and serves correctly on both paths.
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
        row0 = p.infer({"country": "US", "age": 40.0, "fare": 7.0, "name": "x"})
        np.testing.assert_allclose(row0["s"][member], ref[0]["a"], rtol=1e-12)


def test_distinct_on_the_call_refuses_at_construction():
    # Measured: DuckDB refuses DISTINCT on any registered scalar function
    # ("only applicable to aggregate functions") — the original text has
    # no oracle reading, so every spelling refuses by name (review round).
    for sql in [
        "SELECT sc(DISTINCT age) AS s FROM __THIS__",
        "SELECT sc(DISTINCT age).age AS z FROM __THIS__",
        "SELECT sc_transform(DISTINCT sc_fit(age) OVER (), age) AS s FROM __THIS__",
    ]:
        with pytest.raises(MarginalizeError, match="DISTINCT"):
            _fit(sql)


def test_star_over_a_call_is_the_oracles_parser_error():
    # Measured 2026-08-05: `call.*` / `(call).*` are DuckDB PARSER errors —
    # the star-over-expression spelling does not exist in the oracle. The
    # lawful expansion spelling is unnest (below).
    with pytest.raises(MarginalizeError, match="parse error"):
        _fit("SELECT sc(struct_pack(a := age, f := fare)).*, name FROM __THIS__")


def test_unnest_expands_struct_output():
    p = _fit("SELECT unnest(sc(struct_pack(a := age, f := fare))), name FROM __THIS__")
    out = p.transform(TRAIN)
    assert out.column_names == ["a", "f", "name"]
    ref = _ref_struct()
    for i in range(TRAIN.num_rows):
        np.testing.assert_allclose(
            out.column("a").to_pylist()[i], ref[i]["a"], rtol=1e-12
        )
        np.testing.assert_allclose(
            out.column("f").to_pylist()[i], ref[i]["f"], rtol=1e-12
        )


def test_unnest_expands_struct_output_row_path():
    """C3: the row path expands to the same columns, same values."""
    p = _fit("SELECT unnest(sc(struct_pack(a := age, f := fare))), name FROM __THIS__")
    want = p.transform(TRAIN).to_pylist()
    got = p.infer_batch(TRAIN.to_pylist())
    assert got == want


def test_unnest_ignores_its_alias():
    # Measured: DuckDB ignores an alias on an unnest item — the learned
    # field names are the output names, at every width.
    p = _fit(
        "SELECT unnest(sc(struct_pack(a := age, f := fare))) AS zzz, name FROM __THIS__"
    )
    assert p.transform(TRAIN).column_names == ["a", "f", "name"]


def test_unnest_alongside_a_field_read_shares_one_fit():
    p = _fit(
        "SELECT unnest(sc(struct_pack(a := age, f := fare))),"
        " sc(struct_pack(a := age, f := fare)).a AS ra, name FROM __THIS__"
    )
    assert len([st for st in p.plan if st.kind == "fit"]) == 1
    out = p.transform(TRAIN)
    assert out.column_names == ["a", "f", "ra", "name"]
    assert out.column("a").to_pylist() == out.column("ra").to_pylist()


def test_unnest_name_collision_refuses_at_fit():
    # Measured: DuckDB emits DUPLICATE result columns (a, b, a) — the row
    # path's output model cannot carry that, so a learned name colliding
    # with any sibling output refuses by name at fit (P7 carve-out: the
    # names are learned, so construction cannot know them).
    with pytest.raises(MarginalizeError, match="collides"):
        _fit(
            "SELECT unnest(sc(struct_pack(a := age, f := fare))),"
            " name AS a FROM __THIS__"
        )


def test_two_unnests_of_the_same_call_refuse_at_fit():
    with pytest.raises(MarginalizeError, match="collides"):
        _fit(
            "SELECT unnest(sc(struct_pack(a := age, f := fare))),"
            " unnest(sc(struct_pack(a := age, f := fare))) FROM __THIS__"
        )


@pytest.mark.parametrize(
    "clause,sql",
    [
        ("DISTINCT", "SELECT unnest(DISTINCT sc(%s)), name FROM __THIS__"),
        ("FILTER", "SELECT unnest(sc(%s)) FILTER (WHERE age > 0), name FROM __THIS__"),
        ("ORDER BY", "SELECT unnest(sc(%s) ORDER BY age), name FROM __THIS__"),
    ],
)
def test_modifiers_on_the_unnest_item_refuse(clause, sql):
    # Measured: DuckDB refuses all three on UNNEST itself ('"DISTINCT",
    # "FILTER", and "ORDER BY" are not applicable to "UNNEST"'). The
    # unnest branch rebuilds the item, so it must re-screen them —
    # they were silently dropped (review round).
    with pytest.raises(MarginalizeError, match="UNNEST"):
        _fit(sql % "struct_pack(a := age, f := fare)")


@pytest.mark.parametrize(
    "sql,match",
    [
        # Measured binder errors — the oracle has no reading for these.
        (
            "SELECT unnest(sc(struct_pack(a := age, f := fare))) + 1 FROM __THIS__",
            "root element",
        ),
        (
            "SELECT unnest(unnest(sc(struct_pack(a := age, f := fare)))) FROM __THIS__",
            "[Nn]ested UNNEST",
        ),
        # WHERE is refused wholesale (filter shape, not a projection) —
        # it never reaches the unnest position check.
        (
            "SELECT name FROM __THIS__ WHERE"
            " unnest(sc(struct_pack(a := age, f := fare))) > 0",
            "filter shape",
        ),
    ],
)
def test_unlawful_unnest_positions_refuse(sql, match):
    with pytest.raises(MarginalizeError, match=match):
        _fit(sql)
