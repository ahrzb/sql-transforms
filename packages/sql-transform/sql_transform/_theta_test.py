"""θ public export — fit/transform-split slice 6.

θ is an ordinary value: ``Struct<type, id>`` where ``type`` names the
minted UDF binding and ``id`` is the params-table ``__cf_est`` joined per
group. A public ``sc_fit(bundle) OVER (...) AS theta`` column therefore
serves as ``struct_pack(type := '__cf_tf0', id := __cf_p0.__cf_est)`` —
pure existing parts, the same value for every row of a fit scope. Spec:
docs/superpowers/specs/2026-08-05-fit-transform-split-design.md, slice 6.
"""

import pyarrow as pa
import pydantic
import pytest
from sklearn.preprocessing import StandardScaler

from sql_transform import MarginalizeError, SQLProjection

from ._transformers_test import TRAIN

ROW = pydantic.create_model("Row", **dict.fromkeys(TRAIN.column_names, (object, None)))
B = "struct_pack(a := age, f := fare)"


def _fit(sql: str) -> SQLProjection:
    return SQLProjection(
        sql, this_model=ROW, transformers={"sc": StandardScaler()}
    ).fit(TRAIN)


def test_theta_export_serves_a_handle_per_group():
    p = _fit(
        f"SELECT sc_fit({B}) OVER (PARTITION BY country) AS theta,"
        " country, name FROM __THIS__"
    )
    out = p.transform(TRAIN)
    thetas = out.column("theta").to_pylist()
    countries = out.column("country").to_pylist()
    assert all(set(t) == {"type", "id"} for t in thetas)
    # One handle per fit scope: same θ within a group, distinct between.
    by_group = {}
    for c, t in zip(countries, thetas, strict=True):
        by_group.setdefault(c, set()).add((t["type"], t["id"]))
    assert all(len(v) == 1 for v in by_group.values())
    assert len({next(iter(v)) for v in by_group.values()}) == len(by_group)


def test_theta_export_row_path():
    """C3: the row path serves the same handle the batch path does."""
    p = _fit(
        f"SELECT sc_fit({B}) OVER (PARTITION BY country) AS theta,"
        " country, name FROM __THIS__"
    )
    want = p.transform(TRAIN).to_pylist()
    got = [r.model_dump() for r in p.infer_batch(TRAIN.to_pylist())]
    assert got == want


def test_theta_export_and_consumption_share_one_fit():
    # Exporting θ and applying it in the same SELECT is ONE fit step —
    # the export is the same handle the transform half consumes.
    p = _fit(
        f"SELECT sc_fit({B}) OVER (PARTITION BY country) AS _th,"
        f" sc_transform(_th, {B}).a AS za,"
        f" sc_fit({B}) OVER (PARTITION BY country) AS theta,"
        " name FROM __THIS__"
    )
    assert len([st for st in p.plan if st.kind == "fit"]) == 1
    out = p.transform(TRAIN)
    assert set(out.column_names) == {"za", "theta", "name"}


def test_global_fit_scope_exports_one_handle():
    p = _fit(f"SELECT sc_fit({B}) OVER () AS theta, name FROM __THIS__")
    thetas = p.transform(TRAIN).column("theta").to_pylist()
    assert len({(t["type"], t["id"]) for t in thetas}) == 1


def test_unseen_group_theta_is_null_on_both_paths():
    """P14: no fitted instance for the group — no handle to hand out."""
    p = _fit(
        f"SELECT sc_fit({B}) OVER (PARTITION BY country) AS theta, name FROM __THIS__"
    )
    unseen = pa.table({"country": ["JP"], "age": [33.0], "fare": [4.0], "name": ["q"]})
    assert p.transform(unseen).column("theta").to_pylist() == [None]
    assert (
        p.infer({"country": "JP", "age": 33.0, "fare": 4.0, "name": "q"}).theta is None
    )


@pytest.mark.parametrize(
    "sql",
    [
        f"SELECT name AS th, sc_fit({B}) OVER () AS th FROM __THIS__",
        f"SELECT sc_fit({B}) OVER () AS th, name AS th FROM __THIS__",
    ],
)
def test_theta_export_obeys_the_duplicate_output_name_law(sql):
    # A θ item is an output column like any other: DuckDB would emit two
    # same-named columns, which a row (a named struct) cannot carry — the
    # law refuses at EVERY level, θ included (review round).
    with pytest.raises(MarginalizeError, match="duplicate output name"):
        _fit(sql)


def test_theta_export_colliding_with_a_star_column_refuses():
    # The duplicate law counts star-expanded names too — the row path
    # cannot carry two `age` columns (review round).
    with pytest.raises(MarginalizeError, match="duplicate output name age"):
        _fit(f"SELECT *, sc_fit({B}) OVER () AS age FROM __THIS__")
    ok = _fit(f"SELECT *, sc_fit({B}) OVER () AS theta FROM __THIS__")
    assert ok.transform(TRAIN).column_names[-1] == "theta"


def test_schema_free_theta_export_refuses_by_name_not_by_crash():
    # Schema-free: a bare reference to an earlier alias is undecidable
    # (lateral alias vs table column), so it must refuse BY NAME — the θ
    # item used to skip risky_aliases and die as a raw binder error.
    with pytest.raises(MarginalizeError, match="lateral alias|not found|unknown"):
        SQLProjection(
            f"SELECT sc_fit({B}) OVER () AS q, q AS r FROM __THIS__",
            transformers={"sc": StandardScaler()},
        ).fit(TRAIN)


def test_hand_written_theta_still_refuses():
    # Export makes θ readable, NOT constructible: a hand-written handle
    # has no lawful provenance (it graduates with composition).
    with pytest.raises(MarginalizeError, match="lawful provenance"):
        _fit(
            f"SELECT sc_transform({{'type': 'sc', 'id': 0}}, {B}).a AS z FROM __THIS__"
        )


def test_exported_theta_column_cannot_be_consumed_across_levels():
    # Cross-level θ is θ-as-data (composition territory): the CTE level
    # parks a fit, the level above tries to apply it.
    with pytest.raises(MarginalizeError, match="non-final level|lawful provenance"):
        _fit(
            f"WITH c AS (SELECT sc_fit({B}) OVER () AS th, age, fare FROM __THIS__)"
            " SELECT sc_transform(th, struct_pack(a := age, f := fare)).a AS z"
            " FROM c"
        )
