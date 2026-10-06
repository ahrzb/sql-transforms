"""Explicit display handles and the canonical direct estimator-ID boundary.

Migrated display values are ordinary SQL structs, not consumable fit handles.
"""

import pyarrow as pa
import pytest
from sklearn.preprocessing import StandardScaler

from sql_transform import SQLProjection, TransformError

from ._transformers_test import TRAIN

B = "struct_pack(a := age, f := fare)"
PARAMS = f"SELECT country, sc_fit({B}) AS iid FROM __FIT__ GROUP BY country"
DISPLAY = (
    "CASE WHEN p.iid IS NULL THEN NULL ELSE "
    "struct_pack(type := 'sc', id := p.iid) END AS theta"
)


def _display(*, consume=False):
    items = DISPLAY + ", t.country, t.name"
    if consume:
        items += ", sc_transform(p.iid, struct_pack(a := t.age, f := t.fare)).a AS za"
    return SQLProjection(
        f"WITH p AS ({PARAMS}) SELECT {items} FROM __THIS__ t "
        "LEFT JOIN p ON t.country IS NOT DISTINCT FROM p.country",
        captured={"sc": StandardScaler()},
    ).fit(TRAIN)


def test_migrated_display_handle_per_group():
    p = _display()
    out = p.transform(TRAIN)
    by_group = {}
    for row in out.to_pylist():
        theta = row["theta"]
        assert set(theta) == {"type", "id"}
        assert theta["type"] == "sc"
        assert theta["id"] in p.instances
        by_group.setdefault(row["country"], set()).add(theta["id"])
    assert all(len(ids) == 1 for ids in by_group.values())
    assert len({next(iter(ids)) for ids in by_group.values()}) == len(by_group)
    compiled = p.compile()
    assert compiled.infer_rows(TRAIN.to_pylist()) == out.to_pylist()
    assert compiled.infer_arrow(TRAIN).to_pylist() == out.to_pylist()


def test_migrated_display_and_direct_consumption_share_instances():
    p = _display(consume=True)
    assert len(p.instances) == len(set(TRAIN.column("country").to_pylist()))
    rows = p.transform(TRAIN).to_pylist()
    assert set(rows[0]) == {"za", "theta", "country", "name"}
    assert rows[0]["za"] == 1.0 and rows[1]["za"] == -1.0


def test_migrated_global_display_handle():
    p = SQLProjection(
        f"WITH p AS (SELECT sc_fit({B}) AS iid FROM __FIT__) "
        f"SELECT {DISPLAY}, t.name FROM __THIS__ t LEFT JOIN p ON 1 = 1",
        captured={"sc": StandardScaler()},
    ).fit(TRAIN)
    thetas = p.transform(TRAIN).column("theta").to_pylist()
    assert len({(theta["type"], theta["id"]) for theta in thetas}) == 1


def test_migrated_unseen_group_has_no_display_handle():
    p = _display()
    unseen = pa.table({"country": ["JP"], "age": [33.0], "fare": [4.0], "name": ["q"]})
    assert p.transform(unseen).column("theta").to_pylist() == [None]
    assert p.compile().infer_rows(unseen.to_pylist())[0]["theta"] is None


@pytest.mark.parametrize("alias", ["theta", "_th"])
def test_parked_fit_window_is_removed(alias):
    with pytest.raises(TransformError, match="inline|explicit"):
        SQLProjection.marginalize(
            f"SELECT sc_fit({B}) OVER () AS {alias}, name FROM __THIS__",
            captured={"sc": StandardScaler()},
        )


@pytest.mark.parametrize(
    "sql",
    [
        f"SELECT sc_transform({{'type': 'sc', 'id': 0}}, {B}).a AS z FROM __THIS__",
        f"SELECT sc_transform(0, {B}).a AS z FROM __THIS__",
        f"WITH p AS ({PARAMS}) SELECT sc_transform(p.iid + 1, {B}).a AS z "
        "FROM __THIS__ t, p",
        f"WITH p AS ({PARAMS}), q AS (SELECT iid FROM p) "
        f"SELECT sc_transform(q.iid, {B}).a AS z FROM __THIS__ t, q",
        f"WITH p AS ({PARAMS}) "
        f"SELECT sc_transform(struct_pack(type := 'sc', id := p.iid), {B}).a AS z "
        "FROM __THIS__ t, p",
        f"WITH p AS (SELECT country, other_fit({B}) AS iid FROM __FIT__ "
        f"GROUP BY country) SELECT sc_transform(p.iid, {B}).a AS z FROM __THIS__ t, p",
    ],
)
def test_forged_altered_intermediate_and_cross_scope_ids_refuse(sql):
    with pytest.raises(TransformError, match="canonical|direct|raw params"):
        SQLProjection(sql, captured={"sc": StandardScaler(), "other": StandardScaler()})
