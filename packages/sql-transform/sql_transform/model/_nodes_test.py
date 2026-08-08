"""The typed tree: lossless, complete, and unable to interpret what it does not know.

Three properties, in the order they matter:

* **Lossless.** ``to_json(from_json(raw)) == raw`` for every statement we have.
  Structural equality, not "it still prints" — because the measurement this
  whole model rests on is that a *dropped* field still prints, just differently:

      dropping BASE_TABLE.table_name -> accepted, DIFFERENT SQL
      dropping BASE_TABLE.type       -> rejected

  ``json_deserialize_sql`` requires exactly one field. Everything else silently
  defaults, so a model that forgets a field emits another query rather than an
  error.

* **Complete.** Every field DuckDB emits for a typed tag is carried by its
  class. This is the property a code generator would have guaranteed; checking
  it is cheaper than generating and fails in the same place on a version bump.

* **Opaque descends.** An unrecognised tag is carried verbatim and never
  interpreted — but its children are still typed, so a ``__FIT__`` reference
  nested under a node DuckDB adds next year is still visible to the walk.
"""

import json
from collections import defaultdict
from pathlib import Path

import duckdb
import pytest

from sql_transform.model._ast import _deserialize, _serialize
from sql_transform.model._nodes import (
    _STRUCTURAL,
    INTERPRETED,
    BaseTable,
    ColumnRef,
    Function,
    Opaque,
    Select,
    child_nodes,
    from_json,
    to_json,
)

# --------------------------------------------------------------- the corpus

# The window corpus is deep but narrow — it is all SELECT with aggregates. These
# reach the shapes it never does: set operations, every join form, recursive
# CTEs, table functions, subqueries in both of their two lives, casts, CASE.
EXTRA_SHAPES = [
    "SELECT 1 UNION SELECT 2",
    "SELECT 1 UNION ALL SELECT 2",
    "SELECT 1 EXCEPT SELECT 2",
    "SELECT 1 INTERSECT SELECT 2",
    "WITH RECURSIVE t(n) AS (SELECT 1 UNION ALL SELECT n+1 FROM t WHERE n<5)"
    " SELECT * FROM t",
    "WITH a AS (SELECT 1 AS x) SELECT * FROM a",
    "WITH a AS MATERIALIZED (SELECT 1 AS x) SELECT * FROM a",
    "SELECT * FROM (SELECT 1 AS x) s",
    "SELECT * FROM (SELECT 1 AS x) s(y)",
    "SELECT (SELECT max(x) FROM t) AS m FROM u",
    "SELECT * FROM t WHERE x IN (SELECT y FROM u)",
    "SELECT * FROM t WHERE EXISTS (SELECT 1 FROM u)",
    "SELECT * FROM t WHERE x > ALL (SELECT y FROM u)",
    "SELECT * FROM range(3)",
    "SELECT * FROM range(3) WITH ORDINALITY",
    "SELECT * FROM a JOIN b ON a.i = b.i",
    "SELECT * FROM a LEFT JOIN b ON a.i = b.i",
    "SELECT * FROM a FULL OUTER JOIN b USING (i)",
    "SELECT * FROM a NATURAL JOIN b",
    "SELECT * FROM a CROSS JOIN b",
    "SELECT * FROM a POSITIONAL JOIN b",
    "SELECT * FROM a ASOF JOIN b ON a.i >= b.i",
    "SELECT a.* FROM a",
    "SELECT * EXCLUDE (x) FROM a",
    "SELECT * REPLACE (x+1 AS x) FROM a",
    "SELECT CAST(x AS DECIMAL(10,2)) FROM a",
    "SELECT x::VARCHAR FROM a",
    "SELECT CASE WHEN x > 1 THEN 'a' ELSE 'b' END FROM a",
    "SELECT x IS NOT NULL, NOT x, -x, x[1], x['k'] FROM a",
    "SELECT [1,2,3], {'a': 1}, (1,2) FROM a",
    "SELECT count(*) FILTER (WHERE x > 1) FROM a",
    "SELECT string_agg(x, ',' ORDER BY x) FROM a",
    "SELECT x FROM a GROUP BY GROUPING SETS ((x), ())",
    "SELECT x FROM a GROUP BY CUBE (x)",
    "SELECT x FROM a ORDER BY x DESC NULLS LAST LIMIT 3 OFFSET 1",
    "SELECT DISTINCT ON (x) x FROM a",
    "SELECT x FROM a QUALIFY row_number() OVER (PARTITION BY x) = 1",
    "SELECT x FROM a TABLESAMPLE 10%",
    "SELECT x FROM a HAVING count(*) > 1",
    "SELECT $1 AS a",
    "SELECT INTERVAL 1 DAY, DATE '2020-01-01', TIMESTAMP '2020-01-01 00:00:00'",
    "SELECT * FROM VALUES (1, 'a'), (2, 'b') AS t(i, s)",
    "SELECT lambda_test FROM (SELECT list_transform([1,2], x -> x + 1) AS lambda_test)",
]


def _corpus() -> list[str]:
    from sql_transform._corpus_test import (  # noqa: PLC0415
        CURATED_MARGINALIZED,
        CURATED_REFUSED,
        MINED,
    )

    everything = MINED + CURATED_MARGINALIZED + CURATED_REFUSED + EXTRA_SHAPES
    return list(dict.fromkeys(everything))


CORPUS = _corpus()


def _parseable() -> list[str]:
    out = []
    for sql in CORPUS:
        try:
            _serialize(sql)
        except Exception:  # noqa: BLE001, S112 - not every corpus line is a query
            continue
        out.append(sql)
    return out


PARSEABLE = _parseable()


# ------------------------------------------------------------------ lossless


def test_the_corpus_is_wide_enough_to_mean_something():
    assert len(PARSEABLE) >= 110, f"corpus shrank to {len(PARSEABLE)}"


@pytest.mark.parametrize("sql", PARSEABLE)
def test_the_tree_is_lossless(sql):
    """Structural equality, both directions, on every statement we have."""
    raw = _serialize(sql)
    assert to_json(from_json(raw)) == raw


@pytest.mark.parametrize("sql", PARSEABLE)
def test_the_tree_still_prints_what_duckdb_prints(sql):
    """Belt and braces: the oracle agrees the round-tripped tree is the query."""
    raw = _serialize(sql)
    assert _deserialize(to_json(from_json(raw))) == _deserialize(raw)


# ------------------------------------------------------------------ complete


def _fields_in_corpus() -> dict[str, set[str]]:
    """Every field DuckDB emits, per (tag, kind) — the shape the classes must match."""
    found: dict[str, set[str]] = defaultdict(set)

    def walk(obj):
        if isinstance(obj, dict):
            if isinstance(t := obj.get("type"), str):
                is_q = "cte_map" in obj
                kind = "query" if is_q else "ref" if "sample" in obj else "expr"
                found[f"{t}/{kind}"] |= set(obj)
            for v in obj.values():
                walk(v)
        elif isinstance(obj, list):
            for v in obj:
                walk(v)

    for sql in PARSEABLE:
        walk(_serialize(sql))
    return dict(found)


FIELDS_IN_CORPUS = _fields_in_corpus()


@pytest.mark.parametrize("model", INTERPRETED, ids=lambda m: m.__name__)
def test_every_field_duckdb_emits_is_carried(model):
    """A field we do not carry is a field we drop, and a dropped field is
    accepted by the deserializer as *different SQL*. So this is not style."""
    key = model.shape_key()
    observed = FIELDS_IN_CORPUS.get(key)
    assert observed is not None, f"{model.__name__}: corpus never produced {key}"
    # Compared under DuckDB's names, not Python's: `class` is a keyword, so
    # three of these carry it as `class_` behind an alias.
    carried = {spec.alias or name for name, spec in model.model_fields.items()}
    assert carried == observed


def test_the_structural_shapes_are_what_we_think():
    """`_STRUCTURAL` is the one place a *key set* decides a node's class, so a
    DuckDB change there would silently turn a CTE map into an Opaque rather
    than fail. Held down by asking the oracle for each shape directly."""
    doc = _serialize("WITH a AS (SELECT 1) SELECT * FROM a")
    entry = doc["statements"][0]["node"]["cte_map"]["map"][0]
    for shape in (
        doc,  # Document
        doc["statements"][0],  # Statement
        doc["statements"][0]["node"]["cte_map"],  # CteMap
        entry,  # CteEntry
        entry["value"],  # CteValue
        entry["value"]["query"],  # Subquery
    ):
        assert frozenset(shape) in _STRUCTURAL, sorted(shape)


def test_a_dropped_field_is_accepted_and_changes_the_query():
    """The measurement the whole design rests on. If DuckDB ever starts
    rejecting a partial node, this goes red and the completeness gate above
    stops being load-bearing — which is worth knowing."""
    raw = _serialize("SELECT a.x FROM t a")
    partial = json.loads(json.dumps(raw))
    del partial["statements"][0]["node"]["from_table"]["table_name"]
    assert _deserialize(partial) != _deserialize(raw)


# -------------------------------------------------------------------- opaque


def test_an_unknown_tag_is_carried_not_interpreted():
    raw = _serialize("SELECT x FROM a WHERE y BETWEEN 1 AND 2")
    node = from_json(raw)
    assert to_json(node) == raw
    assert any(isinstance(n, Opaque) for n in _walk(node)), "no Opaque in a BETWEEN"


def test_opaque_children_are_still_typed():
    """The bug this prevents: a `__FIT__` reference hidden under a tag we do
    not know, invisible to freezing, silently never frozen."""
    raw = _serialize(
        "SELECT x FROM a WHERE y BETWEEN (SELECT min(z) FROM __FIT__) AND 2"
    )
    tables = [n.table_name for n in _walk(from_json(raw)) if isinstance(n, BaseTable)]
    assert "__FIT__" in tables


def test_opaque_cannot_be_mistaken_for_a_typed_node():
    raw = _serialize("SELECT x + 1 FROM a")
    opaques = [n for n in _walk(from_json(raw)) if isinstance(n, Opaque)]
    assert opaques
    for n in opaques:
        assert not isinstance(n, Select | BaseTable | ColumnRef | Function)


def _walk(node):
    """Every typed node in the tree, including the ones under an Opaque."""
    yield node
    for child in child_nodes(node):
        yield from _walk(child)


# --------------------------------------------------------------- the manifest

MANIFEST = Path(__file__).with_name("_shapes.json")


def test_the_pinned_shapes_still_describe_this_duckdb():
    """The drift gate. Covers every tag, not just the typed ones: Opaque makes
    drift in an uninterpreted tag harmless, which also makes it invisible.

    Re-pin with `uv run python scripts/pin_ast_shapes.py` and read the diff.
    """
    pinned = json.loads(MANIFEST.read_text(encoding="utf-8"))
    observed = {k: sorted(v) for k, v in sorted(FIELDS_IN_CORPUS.items())}
    moved = {
        k: (pinned["shapes"].get(k), observed.get(k))
        for k in set(pinned["shapes"]) | set(observed)
        if pinned["shapes"].get(k) != observed.get(k)
    }
    assert not moved, (
        f"AstDrift: duckdb {duckdb.__version__} (pinned {pinned['duckdb']}) "
        f"moved {len(moved)} shapes:\n"
        + "\n".join(f"  {k}: {was} -> {now}" for k, (was, now) in sorted(moved.items()))
    )
