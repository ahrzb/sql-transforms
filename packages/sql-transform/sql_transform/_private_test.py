"""Private columns (TASK-64) + θ laterals — fit/transform-split slice 2.

An output field named ``_...`` is a same-SELECT macro: usable by later
items (β-reduced at the raw AST, so windows and transformer bundles see
closed expressions), never crossing the output boundary. ``_th`` is the
ergonomic split spelling. Spec: docs/superpowers/specs/
2026-08-05-fit-transform-split-design.md, slice-2 addendum.
"""

import duckdb
import numpy as np
import pyarrow as pa
import pydantic
import pytest
from sklearn.preprocessing import StandardScaler

from sql_transform import MarginalizeError, SQLProjection

from ._transformers_test import TRAIN, _by_name, _reference

ROW = pydantic.create_model("Row", **dict.fromkeys(TRAIN.column_names, (object, None)))


def _fit(sql: str) -> SQLProjection:
    return SQLProjection(
        sql, this_model=ROW, transformers={"sc": StandardScaler()}
    ).fit(TRAIN)


def _duck(sql: str) -> dict:
    con = duckdb.connect()
    try:
        con.execute("SET threads = 1")
        con.register("__THIS__", TRAIN)
        return con.execute(sql).to_arrow_table().to_pydict()
    finally:
        con.close()


# --- θ laterals ---------------------------------------------------------------


def test_theta_lateral_equals_inline_spelling():
    """The private-θ spelling serves the exact SQL of the inline spelling."""
    lateral = _fit(
        "SELECT sc_fit(age) OVER (PARTITION BY country) AS _th,"
        " sc_transform(_th, age).age AS z, name FROM __THIS__"
    )
    inline = _fit(
        "SELECT sc_transform(sc_fit(age) OVER (PARTITION BY country), age).age"
        " AS z, name FROM __THIS__"
    )
    assert lateral.serving_sql == inline.serving_sql
    (step,) = [s for s in lateral.plan if s.kind == "fit"]
    assert step.transformer == "sc" and step.keys != ()
    got = _by_name(lateral.transform(TRAIN), "z")
    feats = np.array([TRAIN.column("age").to_pylist()], dtype=float).T
    ref = _reference(
        StandardScaler(), feats, [(c,) for c in TRAIN.column("country").to_pylist()]
    )
    for i, n in enumerate(TRAIN.column("name").to_pylist()):
        np.testing.assert_allclose(got[n], ref[i][0], rtol=1e-12)


def test_theta_lateral_feeds_two_transforms_one_fit():
    p = _fit(
        "SELECT sc_fit(struct_pack(v := age)) OVER () AS _th,"
        " sc_transform(_th, struct_pack(v := age)).v AS za,"
        " sc_transform(_th, struct_pack(v := fare)).v AS zf, name FROM __THIS__"
    )
    assert len([s for s in p.plan if s.kind == "fit"]) == 1
    out = p.transform(TRAIN)
    assert "_th" not in out.column_names
    ages = np.array([TRAIN.column("age").to_pylist()], dtype=float).T
    est = StandardScaler().fit(ages)
    fares = np.array([TRAIN.column("fare").to_pylist()], dtype=float).T
    got_a, got_f = _by_name(out, "za"), _by_name(out, "zf")
    ref_a, ref_f = est.transform(ages)[:, 0], est.transform(fares)[:, 0]
    for i, n in enumerate(TRAIN.column("name").to_pylist()):
        np.testing.assert_allclose(got_a[n], ref_a[i], rtol=1e-12)
        np.testing.assert_allclose(got_f[n], ref_f[i], rtol=1e-12)


def test_private_feeds_fit_partition_key():
    """The old lateral-in-window refusal, lifted via substitution (AC#5)."""
    p = _fit(
        "SELECT age >= 30 AS _o,"
        " sc_transform(sc_fit(age) OVER (PARTITION BY _o), age).age AS z,"
        " name FROM __THIS__"
    )
    got = _by_name(p.transform(TRAIN), "z")
    feats = np.array([TRAIN.column("age").to_pylist()], dtype=float).T
    ref = _reference(
        StandardScaler(), feats, [(a >= 30,) for a in TRAIN.column("age").to_pylist()]
    )
    for i, n in enumerate(TRAIN.column("name").to_pylist()):
        np.testing.assert_allclose(got[n], ref[i][0], rtol=1e-12)


def test_private_feeds_transformer_bundle():
    lateral = _fit(
        "SELECT age / 2 AS _h, sc(struct_pack(v := _h)).v AS z, name FROM __THIS__"
    )
    inline = _fit("SELECT sc(struct_pack(v := age / 2)).v AS z, name FROM __THIS__")
    assert lateral.serving_sql == inline.serving_sql


def test_private_struct_valued_call_read_via_struct_extract():
    """Name the call once privately, read a field — the TASK-64 pattern
    (dotted ``_t.v`` refuses like DuckDB; struct_extract binds laterally)."""
    lateral = _fit(
        "SELECT sc(struct_pack(v := age)) AS _t,"
        " struct_extract(_t, 'v') AS z, name FROM __THIS__"
    )
    inline = _fit("SELECT sc(struct_pack(v := age)).v AS z, name FROM __THIS__")
    assert len([s for s in lateral.plan if s.kind == "fit"]) == 1
    got = _by_name(lateral.transform(TRAIN), "z")
    ref = _by_name(inline.transform(TRAIN), "z")
    for n, v in ref.items():
        np.testing.assert_allclose(got[n], v, rtol=1e-12)


# --- plain private columns ----------------------------------------------------


def test_private_chain_lowers_by_substitution():
    p = _fit("SELECT age * 2 AS _d, _d + 1 AS _e, _e * 3 AS out, name FROM __THIS__")
    assert "_d" not in p.serving_sql and "_e" not in p.serving_sql
    out = p.transform(TRAIN)
    assert out.column_names == ["out", "name"]
    got = _by_name(out, "out")
    for a, n in zip(
        TRAIN.column("age").to_pylist(), TRAIN.column("name").to_pylist(), strict=True
    ):
        assert got[n] == (a * 2 + 1) * 3


def test_private_window_value_consumed_by_scalar():
    p = _fit(
        "SELECT avg(age) OVER (PARTITION BY country) AS _m, age - _m AS c,"
        " name FROM __THIS__"
    )
    out = p.transform(TRAIN)
    assert out.column_names == ["c", "name"]
    ref = _duck(
        "SELECT age - avg(age) OVER (PARTITION BY country) AS c, name FROM __THIS__"
    )
    got = _by_name(out, "c")
    for n, c in zip(ref["name"], ref["c"], strict=True):
        np.testing.assert_allclose(got[n], c, rtol=1e-12)


def test_output_model_excludes_privates():
    p = _fit("SELECT age * 2 AS _d, _d + 1 AS out, name FROM __THIS__")
    assert list(p.output_model.model_fields) == ["out", "name"]
    row = p.infer({"country": "US", "age": 40.0, "fare": 7.0, "name": "x"})
    assert row.out == 81.0


# --- public laterals on the new mechanism -------------------------------------


def test_public_lateral_in_window_now_lawful():
    """Was refused pre-slice-2; the substitution mechanism lifts it."""
    p = _fit(
        "SELECT age >= 30 AS o, avg(fare) OVER (PARTITION BY o) AS m,"
        " name FROM __THIS__"
    )
    ref = _duck(
        "SELECT age >= 30 AS o, avg(fare) OVER (PARTITION BY age >= 30) AS m,"
        " name FROM __THIS__"
    )
    got = _by_name(p.transform(TRAIN), "m")
    for n, m in zip(ref["name"], ref["m"], strict=True):
        np.testing.assert_allclose(got[n], m, rtol=1e-12)


def test_duplicate_public_alias_is_last_wins_like_duckdb():
    """Measured 2026-08-05: DuckDB resolves a duplicated alias to its LAST
    definition; the old first-wins store served the wrong column."""
    p = _fit("SELECT age AS d, name AS d, d AS out FROM __THIS__")
    out = p.transform(TRAIN)
    assert out.column("out").to_pylist() == TRAIN.column("name").to_pylist()


def test_real_column_still_beats_alias():
    p = _fit("SELECT name AS age, age AS out FROM __THIS__")
    out = p.transform(TRAIN)
    assert out.column("out").to_pylist() == TRAIN.column("age").to_pylist()


# --- the boundary and the refusals --------------------------------------------


def test_schema_free_star_over_private_table_column_refuses_at_fit():
    t = pa.table({"age": [1.0, 2.0], "_meta": ["a", "b"]})
    p = SQLProjection("SELECT * FROM __THIS__")
    with pytest.raises(MarginalizeError, match="_meta"):
        p.fit(t)
    SQLProjection("SELECT age FROM __THIS__").fit(t)  # no leak, no refusal


def test_declared_schema_cannot_express_a_private_column():
    """Pydantic drops ``_``-leading names from model_fields, so the model
    canonicalization already excludes a ``_meta`` table column."""
    t = pa.table({"age": [1.0, 2.0], "_meta": ["a", "b"]})
    model = pydantic.create_model("Row", age=(object, None))
    p = SQLProjection("SELECT * FROM __THIS__", this_model=model).fit(t)
    assert p.transform(t).column_names == ["age"]


REFUSALS = [
    ("SELECT age * 2 AS _d, name FROM __THIS__", "never read"),
    ("SELECT age * 2 AS _d, _d + 1 AS _e FROM __THIS__", "every output column"),
    (
        "SELECT sc(struct_pack(v := age)) AS _t, _t.v AS z FROM __THIS__",
        "struct_extract",
    ),
    ("SELECT avg(age) OVER () AS _m, sum(_m) OVER () AS s FROM __THIS__", "nested"),
    (
        "SELECT avg(age) OVER () AS _m,"
        " avg(fare) OVER (PARTITION BY _m) AS s FROM __THIS__",
        "nested",
    ),
    (
        "SELECT age * 2 AS _d, (SELECT max(age) + _d FROM __THIS__) AS s FROM __THIS__",
        "inside a subquery",
    ),
    ("SELECT age AS _d, fare AS _d, _d + 1 AS out FROM __THIS__", "duplicate private"),
    (
        "WITH c AS (SELECT age * 2 AS _d, _d + 1 AS out FROM __THIS__)"
        " SELECT out, _d AS y FROM c",
        "same-SELECT",
    ),
    ("SELECT _d + 1 AS out, age * 2 AS _d, name FROM __THIS__", "unknown column"),
    ("SELECT * RENAME (age AS _age) FROM __THIS__", "private name"),
    (
        "SELECT sc_fit(age) OVER () AS th, sc_transform(th, age).age AS z"
        " FROM __THIS__",
        "private column",
    ),
]


@pytest.mark.parametrize("sql,match", REFUSALS)
def test_private_refusals(sql, match):
    with pytest.raises(MarginalizeError, match=match):
        SQLProjection(sql, this_model=ROW, transformers={"sc": StandardScaler()})


def test_private_without_schema_refuses():
    with pytest.raises(MarginalizeError, match="this_model"):
        SQLProjection("SELECT age * 2 AS _d, _d + 1 AS out FROM __THIS__")
