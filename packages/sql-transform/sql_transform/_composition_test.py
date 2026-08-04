"""Composition (TASK-65): projections as vocabulary members.

A member is called like a transformer — windowed, bundled,
field-addressed — and its output struct T is AUTHORED (its select
aliases), so unknown fields refuse at construction, not fit. Slice 1:
SQL-only members (no internal windows/transformers) lower by pure
β-reduction — the member's item expression substitutes into the call
site with the caller's bundle expressions bound to its column names.
"""

import pyarrow as pa
import pytest

from sql_transform import MarginalizeError, SQLProjection

# TASK-65 slice 1, RED checkpoint (2026-08-04): written test-first, watched
# fail (6 failed / bare-call refusal already passes), parked for a context
# compaction. UNSKIP as the first act of the implementation.
pytestmark = pytest.mark.skip(
    reason="TASK-65 slice 1 red checkpoint — unskip with the implementation"
)

TRAIN = pa.table(
    {
        "a": [1.0, 2.0, 3.0, 4.0],
        "b": [10.0, 20.0, 30.0, 40.0],
        "name": ["r0", "r1", "r2", "r3"],
    }
)


def test_sql_only_member_end_to_end():
    m = SQLProjection("SELECT (age + fare) * 2 AS s, age - fare AS d FROM __THIS__")
    p = SQLProjection(
        "SELECT m(struct_pack(age := a, fare := b)) OVER ().s AS v, name FROM __THIS__",
        transformers={"m": m},
    ).fit(TRAIN)
    out = {r["name"]: r["v"] for r in p.transform(TRAIN).to_pylist()}
    for i, n in enumerate(TRAIN.column("name").to_pylist()):
        av = TRAIN.column("a")[i].as_py()
        bv = TRAIN.column("b")[i].as_py()
        assert out[n] == (av + bv) * 2
    # C3: row path == batch path, value for value
    got = [r.model_dump() for r in p.infer_batch(TRAIN.to_pylist())]
    assert got == p.transform(TRAIN).to_pylist()


def test_member_field_read_composes_in_arithmetic():
    m = SQLProjection("SELECT age + fare AS s FROM __THIS__")
    p = SQLProjection(
        "SELECT m(struct_pack(age := a, fare := b)) OVER ().s * 10 AS v FROM __THIS__",
        transformers={"m": m},
    ).fit(TRAIN)
    assert p.transform(TRAIN).column("v").to_pylist() == [
        (r["a"] + r["b"]) * 10 for r in TRAIN.to_pylist()
    ]


def test_two_member_fields_from_one_call_site_text():
    m = SQLProjection("SELECT age + fare AS s, age - fare AS d FROM __THIS__")
    p = SQLProjection(
        "SELECT m(struct_pack(age := a, fare := b)) OVER ().s AS s,"
        " m(struct_pack(age := a, fare := b)) OVER ().d AS d FROM __THIS__",
        transformers={"m": m},
    ).fit(TRAIN)
    rows = p.transform(TRAIN).to_pylist()
    assert rows[0] == {"s": 11.0, "d": -9.0}


def test_unknown_member_field_refuses_at_construction():
    # T is AUTHORED for members — no fit needed to refuse.
    m = SQLProjection("SELECT age * 2 AS s FROM __THIS__")
    with pytest.raises(MarginalizeError, match="no output field 'nope'"):
        SQLProjection(
            "SELECT m(struct_pack(age := a)) OVER ().nope AS v FROM __THIS__",
            transformers={"m": m},
        )


def test_bundle_missing_a_member_column_refuses_at_construction():
    m = SQLProjection("SELECT age + fare AS s FROM __THIS__")
    with pytest.raises(MarginalizeError, match="fare"):
        SQLProjection(
            "SELECT m(struct_pack(age := a)) OVER ().s AS v FROM __THIS__",
            transformers={"m": m},
        )


def test_fitted_member_refuses():
    m = SQLProjection("SELECT age * 2 AS s FROM __THIS__").fit(pa.table({"age": [1.0]}))
    with pytest.raises(MarginalizeError, match="fitted"):
        SQLProjection(
            "SELECT m(struct_pack(age := a)) OVER ().s AS v FROM __THIS__",
            transformers={"m": m},
        )


def test_bare_member_call_refuses():
    m = SQLProjection("SELECT age * 2 AS s FROM __THIS__")
    with pytest.raises(MarginalizeError, match="struct value"):
        SQLProjection(
            "SELECT m(struct_pack(age := a)) OVER () AS v FROM __THIS__",
            transformers={"m": m},
        )
