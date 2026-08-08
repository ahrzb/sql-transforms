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
from collections import Counter, defaultdict
from pathlib import Path

import duckdb
import pytest

from sql_transform.model._ast import _deserialize, _serialize
from sql_transform.model._nodes import (
    INTERPRETED,
    AstNode,
    BaseTable,
    ColumnRef,
    Function,
    Opaque,
    Select,
    _is_structural,
    child_nodes,
    descendants,
    from_json,
    is_query,
    is_ref,
    rebuild,
    to_json,
)

# --------------------------------------------- the dict walk, as the reference
# `_ast.py` used to read the oracle's output through these. They live here now,
# and nowhere else, as the other side of the differential below: the typed walk
# has to visit the same nodes in the same order, or the migration changed what
# freezing sees. Deleting them means giving that up.


def _is_query(value):
    return isinstance(value, dict) and "cte_map" in value


def _is_ref(value):
    return isinstance(value, dict) and "sample" in value and not _is_query(value)


def _under(obj, *, deep):
    """Yield ``(parent, key, dict)`` for every dict below ``obj``."""
    if isinstance(obj, dict):
        items = list(obj.items())
    elif isinstance(obj, list):
        items = list(enumerate(obj))
    else:
        return
    for key, value in items:
        if key == "cte_map" and not deep:
            continue
        if isinstance(value, dict):
            yield obj, key, value
            if _is_query(value) and not deep:
                continue
        if isinstance(value, dict | list):
            yield from _under(value, deep=deep)


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
        assert _is_structural(shape), sorted(shape)


@pytest.mark.parametrize("sql", PARSEABLE)
def test_no_raw_dict_survives_the_conversion(sql):
    """The invariant the walk rests on: after `from_json` every dict in the
    tree is a model. A raw one is a place `isinstance` cannot look, and that
    is where a `__FIT__` hides."""

    def raw_dicts(value):
        if isinstance(value, Opaque):
            return sum(raw_dicts(v) for v in value.fields.values())
        if isinstance(value, AstNode):
            return sum(raw_dicts(getattr(value, n)) for n in type(value).model_fields)
        if isinstance(value, list):
            return sum(raw_dicts(v) for v in value)
        return 1 if isinstance(value, dict) else 0

    assert raw_dicts(from_json(_serialize(sql))) == 0


def test_a_star_replace_entry_is_not_a_cte_entry():
    """The one shape collision: DuckDB spells both `{key, value}`. Claiming the
    replace entry as a CTE entry left it a plain dict inside an Opaque, where
    nothing validates it and the walk stopped yielding it."""
    star = _serialize("SELECT * REPLACE (x+1 AS x) FROM a")["statements"][0]["node"]
    entry = star["select_list"][0]["replace_list"][0]
    assert frozenset(entry) == frozenset({"key", "value"})
    assert not _is_structural(entry)


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


# ------------------------------------------------- the walk, against the old one


def _shape(node):
    """A node as a comparable key: its tag plus its own field names.

    Identity is no use across the two walks — one yields dicts, the other
    models — and printing is no use either, since two `SELECT 1` nodes are
    indistinguishable and both walks may yield both.

    A `type` that is not a string is not a tag: DuckDB nests a whole type
    descriptor under that key inside a VALUE_CONSTANT, and the typed side
    reports no tag for it.
    """
    if isinstance(node, dict):
        tag = node.get("type")
        return (tag if isinstance(tag, str) else None, tuple(sorted(node)))
    if isinstance(node, Opaque):
        return (node.tag_name or None, tuple(sorted(node.fields)))
    fields = type(node).model_fields
    keys = sorted(spec.alias or name for name, spec in fields.items())
    return (type(node).tag or None, tuple(keys))


@pytest.mark.parametrize("deep", [True, False])
@pytest.mark.parametrize("sql", PARSEABLE)
def test_the_typed_walk_visits_what_the_dict_walk_visited(sql, deep):
    """`descendants` replaces `_under`. Same nodes, same order, or the refactor
    changed what freezing sees — which is exactly how a silent one happens."""
    doc = _serialize(sql)
    raw = doc["statements"][0]["node"]
    typed = from_json(doc).statements[0].node
    old = [_shape(v) for _, _, v in _under(raw, deep=deep)]
    new = [_shape(n) for n in descendants(typed, deep=deep)]
    assert new == old


@pytest.mark.parametrize("sql", PARSEABLE)
def test_the_typed_predicates_agree_with_the_structural_ones(sql):
    """`is_query`/`is_ref` replace `_is_query`/`_is_ref`, including for tags
    no class claims — a query node DuckDB adds later arrives as an Opaque, and
    answering False for it would treat a whole subquery as an expression."""
    raw = _serialize(sql)["statements"][0]["node"]
    typed = from_json(_serialize(sql)).statements[0].node
    old = [(_is_query(v), _is_ref(v)) for _, _, v in _under(raw, deep=True)]
    new = [(is_query(n), is_ref(n)) for n in descendants(typed, deep=True)]
    assert new == old


@pytest.mark.parametrize("sql", PARSEABLE)
def test_rebuild_without_a_replacement_is_the_identity(sql):
    doc = from_json(_serialize(sql))
    assert to_json(rebuild(doc, lambda _: None, deep=True)) == _serialize(sql)
    assert to_json(rebuild(doc, lambda _: None, deep=False)) == _serialize(sql)


@pytest.mark.parametrize("deep", [True, False])
@pytest.mark.parametrize("sql", PARSEABLE)
def test_rebuild_offers_exactly_what_descendants_yields(sql, deep):
    """Same nodes, deliberately not the same order.

    `descendants` reads pre-order, matching `_under`, because `_correlation`
    returns its *first* qualified reference. `rebuild` writes post-order, so a
    replacement is never offered back to `fn`. Sets must agree or a caller
    that reads with one and writes with the other misses a site.
    """
    typed = from_json(_serialize(sql)).statements[0].node
    offered: list = []
    rebuild(typed, lambda n: offered.append(n) or None, deep=deep)
    assert Counter(_shape(n) for n in offered) == Counter(
        _shape(n) for n in descendants(typed, deep=deep)
    )


def test_rebuild_replaces_every_site_and_never_rewalks_one():
    """A replacement that itself matches must not be replaced again — the
    property `_bind_parameters` used to get by collecting sites up front."""
    doc = from_json(_serialize("SELECT * FROM a, (SELECT * FROM a) s"))
    seen = 0

    def rename(node):
        nonlocal seen
        if isinstance(node, BaseTable) and node.table_name == "a":
            seen += 1
            return node.model_copy(update={"table_name": "a"})  # matches again
        return None

    rebuild(doc, rename, deep=True)
    assert seen == 2


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
