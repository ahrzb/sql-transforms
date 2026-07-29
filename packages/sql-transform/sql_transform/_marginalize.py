"""Marginalization over ``__THIS__``: the fit half of a binding-time analysis.

Aggregates over ``__THIS__`` are static — computable once, at fit time. This
module rewrites a projection so every window aggregate becomes a column of a
materialized params table joined back in:

    SELECT (age - avg(age) OVER (PARTITION BY country)) AS d FROM __THIS__
        -->  serving_sql:
    SELECT (__cf_t.age - __cf_p0.__cf_a0) AS d
    FROM __THIS__ AS __cf_t
    LEFT JOIN __CF_PARAMS_0__ AS __cf_p0
      ON (__cf_t.country IS NOT DISTINCT FROM __cf_p0.country)
        -->  fit (two stages; see Marginalized): windows_sql reruns the
             original computation with each window re-projected, then
    SELECT DISTINCT __cf_k0 AS country, __cf_w0 AS __cf_a0 FROM __CF_WINDOWS__

Parsing is DuckDB's own (``json_serialize_sql``/``json_deserialize_sql``): the
oracle's grammar, the oracle's printer.

Typing discipline for the JSON AST (a DuckDB-internal format): two tiers.
Subtrees this module merely carries pass through as opaque ``Node`` dicts —
their contents are the oracle's business, and the correctness argument never
depends on them (see below). Every node this module *interprets* is read
through a pydantic view (``_Select``, ``_Window``, ...) that validates shape
at the read site, so a DuckDB format change fails as one named "AST shape
drift" error, not a ``KeyError`` mid-walk. Mutation happens only on raw
dicts (views are read-only lenses) and only via nodes cloned from
``_templates()``, which the oracle itself serialized — so synthetic nodes
always carry every field the deserializer expects. The shapes are pinned by
executed example in ``_marginalize_test.py``.

Correctness split (why pass-through is safe): an expression that survives
untouched appears identically in the original and the rewritten text, so the
two cannot disagree on it. The only constructs that can smuggle whole-table
semantics into a projection are aggregates (marginalized or refused), window
functions (marginalized or refused), subqueries (refused), and the top-level
clauses (WHERE/GROUP BY/... — refused). Everything else passes through.

Join predicates use IS NOT DISTINCT FROM, never ``=``: window PARTITION BY
groups NULL keys into one partition; an equality join would drop them.
Multiplicity holds by construction: every allowlisted aggregate is
deterministic per group, so DISTINCT over (keys, aggs) collapses to exactly
one params row per group, and the LEFT JOIN matches exactly one per input row.

Design spec: docs/superpowers/specs/2026-07-29-sql-projection-marginalization-design.md
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from functools import cache
from typing import Any

import duckdb
import pydantic

Node = dict[str, Any]
"""An opaque oracle AST node: carried, cloned, or grafted — never interpreted."""

_RESERVED = "__cf_"
_NOLOC = 18446744073709551615  # DuckDB's "no query_location" sentinel (u64 max)

# Aggregates whose per-group value is a deterministic function of the group as
# a *multiset* — grown one measured entry at a time, like the corpus. Order-
# sensitive aggregates (first, last, string_agg, array_agg, ...) are refused by
# absence: their value depends on scan order, which would make fit and the
# differential gate flaky.
_ALLOWLIST = frozenset(
    "avg sum count count_star min max "
    "stddev stddev_pop stddev_samp var_pop var_samp variance median".split()
)


class MarginalizeError(ValueError):
    """The SQL is not a projection this slice can marginalize; names why."""


@dataclass(frozen=True)
class ParamsSpec:
    """One params table: its name, its join keys, and the SQL that collapses
    it out of the materialized ``__CF_WINDOWS__`` result (see Marginalized)."""

    name: str
    keys: tuple[str, ...]
    fit_sql: str


@dataclass(frozen=True)
class Marginalized:
    """The rewrite plan.

    Fitting is two stages. ``windows_sql`` runs once over ``__THIS__``: it is
    the *original* select list verbatim (aliased ``__cf_oN``, discarded) plus
    every distinct window aggregate re-projected as ``__cf_wN`` plus the key
    columns as ``__cf_kM``. Keeping the original items is load-bearing: DuckDB
    chains window operators, each reordering rows for the next, so a float
    aggregate's summation order depends on which other windows share the
    query — re-projecting an already-present window is CSE'd and therefore
    yields the original text's value bit-exactly. Each ``params[i].fit_sql``
    then collapses the materialized result (registered as ``__CF_WINDOWS__``)
    with SELECT DISTINCT — pure value picking, no arithmetic left to drift.
    ``windows_sql`` is None when there is nothing to marginalize.
    """

    serving_sql: str
    windows_sql: str | None
    params: tuple[ParamsSpec, ...]


def _refuse(what: str, loc: int | None = None) -> None:
    pos = f" (at position {loc})" if loc not in (None, _NOLOC) else ""
    raise MarginalizeError(f"not a supported projection: {what}{pos}")


# --- typed views over the node classes this module interprets -----------------


class _View(pydantic.BaseModel):
    """Validated, read-only lens over one raw AST node."""

    model_config = pydantic.ConfigDict(extra="ignore", frozen=True)

    @classmethod
    def of(cls, node: Node) -> _View:
        try:
            return cls.model_validate(node)
        except pydantic.ValidationError as e:
            raise MarginalizeError(
                f"DuckDB AST shape drift in {cls.__name__[1:]}: {e}"
            ) from e


class _CteMap(_View):
    map: list[Node]


class _Select(_View):
    type: str
    modifiers: list[Node]
    cte_map: _CteMap
    select_list: list[Node]
    from_table: Node
    where_clause: Node | None
    group_expressions: list[Node]
    group_sets: list[Any]
    aggregate_handling: str
    having: Node | None
    sample: Node | None
    qualify: Node | None


class _BaseTable(_View):
    type: str
    table_name: str = ""
    alias: str = ""
    sample: Node | None = None


class _ColumnRef(_View):
    column_names: list[str]
    query_location: int


class _Window(_View):
    type: str
    function_name: str
    alias: str
    query_location: int
    children: list[Node]
    partitions: list[Node]
    orders: list[Node]
    arg_orders: list[Node]
    start: str
    end: str
    start_expr: Node | None
    end_expr: Node | None
    offset_expr: Node | None
    default_expr: Node | None
    exclude_clause: str
    filter_expr: Node | None
    distinct: bool
    ignore_nulls: bool


class _Star(_View):
    relation_name: str
    columns: bool
    expr: Node | None
    query_location: int


# --- the oracle as parser and printer ----------------------------------------


def _serialize(sql: str) -> dict:
    (out,) = duckdb.execute("SELECT json_serialize_sql(?)", [sql]).fetchone()
    doc = json.loads(out)
    if doc.get("error"):
        raise MarginalizeError(
            f"parse error at position {doc.get('position')}: {doc.get('error_message')}"
        )
    return doc


def _deserialize(doc: dict) -> str:
    (out,) = duckdb.execute(
        "SELECT json_deserialize_sql(?::JSON)", [json.dumps(doc)]
    ).fetchone()
    return out


@cache
def _aggregate_names() -> frozenset[str]:
    """Every aggregate the oracle knows — bare uses of these are refused."""
    rows = duckdb.execute(
        "SELECT DISTINCT function_name FROM duckdb_functions()"
        " WHERE function_type = 'aggregate'"
    ).fetchall()
    return frozenset(name.lower() for (name,) in rows)


@cache
def _templates() -> dict[str, Node]:
    """Shape templates, cut from the oracle's own serialization of reference
    SQL, so grafted nodes always carry every field the deserializer expects."""
    join_doc = _serialize(
        "SELECT 1 FROM __THIS__ AS __cf_t LEFT JOIN p"
        " ON (__cf_t.k IS NOT DISTINCT FROM p.k AND __cf_t.k IS NOT DISTINCT FROM p.k)"
    )
    join = join_doc["statements"][0]["node"]["from_table"]
    true_doc = _serialize("SELECT 1 FROM __THIS__ AS __cf_t LEFT JOIN p ON (1 = 1)")
    collapse = _serialize("SELECT DISTINCT k FROM __CF_WINDOWS__")
    return {
        "doc": _serialize("SELECT 1 FROM __THIS__"),
        "base_table": join["left"],
        "left_join": join,
        "always_true": true_doc["statements"][0]["node"]["from_table"]["condition"],
        "conjunction": join["condition"],
        "not_distinct": join["condition"]["children"][0],
        "column_ref": collapse["statements"][0]["node"]["select_list"][0],
        "collapse_doc": collapse,
    }


def _clone(template_name: str, **fields: Any) -> Node:
    node = copy.deepcopy(_templates()[template_name])
    node.update(fields)
    return node


def _stripped(node: Node) -> str:
    """Structural identity: the node minus locations and aliases."""

    def clean(x: Any) -> Any:
        if isinstance(x, dict):
            return {
                k: clean(v)
                for k, v in x.items()
                if k not in ("query_location", "alias")
            }
        if isinstance(x, list):
            return [clean(v) for v in x]
        return x

    return json.dumps(clean(node), sort_keys=True)


def _expr_text(expr: Node) -> str:
    """The oracle's printed text for one expression (used as output names)."""
    doc = copy.deepcopy(_templates()["doc"])
    doc["statements"][0]["node"]["select_list"] = [dict(expr, alias="")]
    sql = _deserialize(doc)
    prefix, suffix = "SELECT ", " FROM __THIS__"
    if not (sql.startswith(prefix) and sql.endswith(suffix)):
        raise MarginalizeError(f"DuckDB AST shape drift: unexpected print form {sql!r}")
    return sql[len(prefix) : -len(suffix)]


# --- validation and rewrite ---------------------------------------------------


def _partition_key(part: Node) -> str:
    """A PARTITION BY entry must be a plain ``__THIS__`` column; its bare name."""
    if part.get("class") != "COLUMN_REF":
        _refuse(
            "PARTITION BY expression (plain columns only)", part.get("query_location")
        )
    ref = _ColumnRef.of(part)
    if len(ref.column_names) == 2 and ref.column_names[0].upper() == "__THIS__":
        return ref.column_names[1]
    if len(ref.column_names) == 1:
        return ref.column_names[0]
    _refuse(f"PARTITION BY {'.'.join(ref.column_names)}", ref.query_location)
    raise AssertionError("unreachable")


def _check_window(w: _Window) -> None:
    name = w.function_name.lower()
    if w.type != "WINDOW_AGGREGATE":
        _refuse(
            f"window function {name} (position-dependent, not a per-group constant)",
            w.query_location,
        )
    if name not in _ALLOWLIST:
        _refuse(
            f"aggregate {name} is not in the marginalization allowlist",
            w.query_location,
        )
    if w.orders or w.arg_orders:
        _refuse(
            "OVER (... ORDER BY ...) — a running window, not a per-group constant",
            w.query_location,
        )
    if (
        w.start != "UNBOUNDED_PRECEDING"
        or w.end != "CURRENT_ROW_RANGE"
        or w.start_expr is not None
        or w.end_expr is not None
        or w.offset_expr is not None
        or w.default_expr is not None
        or w.exclude_clause != "NO_OTHER"
    ):
        _refuse("an explicit window frame", w.query_location)
    if w.filter_expr is not None:
        _refuse("FILTER inside an aggregate", w.query_location)
    if w.distinct:
        _refuse("DISTINCT inside an aggregate", w.query_location)
    if w.ignore_nulls:
        _refuse("IGNORE NULLS", w.query_location)
    for child in w.children:
        _walk_agg_arg(child)


def _walk_agg_arg(x: Any) -> None:
    """Aggregate arguments must be row-wise: no nesting of table semantics."""
    if isinstance(x, dict):
        cls = x.get("class")
        if cls == "WINDOW":
            _refuse("a window function inside an aggregate", x.get("query_location"))
        if cls == "SUBQUERY":
            _refuse("a subquery", x.get("query_location"))
        if (
            cls == "FUNCTION"
            and x.get("function_name", "").lower() in _aggregate_names()
        ):
            _refuse(
                f"aggregate {x['function_name']} inside an aggregate",
                x.get("query_location"),
            )
        for v in x.values():
            _walk_agg_arg(v)
    elif isinstance(x, list):
        for v in x:
            _walk_agg_arg(v)


@dataclass
class _Group:
    """All aggregates sharing one partition-key set."""

    keys: tuple[str, ...]
    agg_keys: list[str]  # structural identities, in appearance order


class _Collector:
    """Finds window aggregates, groups them per partition-key set, swaps each
    for a reference to its params column, and qualifies everything else.

    ``rewrite`` is pure: it returns fresh nodes and never mutates its input,
    so the no-aggregate identity path can reuse the original tree."""

    def __init__(self) -> None:
        self.groups: dict[tuple[str, ...], _Group] = {}
        self.windows: dict[str, Node] = {}  # structural identity -> raw node

    def _params_ref(self, raw: Node, w: _Window) -> Node:
        keys = tuple(_partition_key(p) for p in w.partitions)
        ident = tuple(k.lower() for k in keys)
        group = self.groups.setdefault(ident, _Group(keys=keys, agg_keys=[]))
        agg_key = _stripped(raw)
        self.windows.setdefault(agg_key, copy.deepcopy(raw))
        if agg_key not in group.agg_keys:
            group.agg_keys.append(agg_key)
        table_idx = list(self.groups).index(ident)
        col_idx = group.agg_keys.index(agg_key)
        return _clone(
            "column_ref",
            column_names=[f"__cf_p{table_idx}", f"__cf_a{col_idx}"],
            alias=w.alias,
        )

    def rewrite(self, x: Any) -> Any:
        if isinstance(x, dict):
            cls = x.get("class")
            if cls == "WINDOW":
                w = _Window.of(x)
                _check_window(w)
                return self._params_ref(x, w)
            if cls == "SUBQUERY":
                _refuse("a subquery", x.get("query_location"))
            if (
                cls == "FUNCTION"
                and x.get("function_name", "").lower() in _aggregate_names()
            ):
                _refuse(
                    f"aggregate {x['function_name']} without OVER"
                    " (that would aggregate the whole table, not project rows)",
                    x.get("query_location"),
                )
            if cls == "COLUMN_REF":
                names = _ColumnRef.of(x).column_names
                if names and names[0].lower().startswith(_RESERVED):
                    return copy.deepcopy(x)  # a reference this module created
                if names and names[0].upper() == "__THIS__":
                    qualified = ["__cf_t", *names[1:]]
                else:
                    qualified = ["__cf_t", *names]
                return dict(copy.deepcopy(x), column_names=qualified)
            if cls == "STAR":
                star = _Star.of(x)
                if star.columns or star.expr is not None:
                    _refuse("COLUMNS(...)", star.query_location)
                if star.relation_name and star.relation_name.upper() != "__THIS__":
                    _refuse(
                        f"star qualified by unknown relation {star.relation_name}",
                        star.query_location,
                    )
                y = {k: self.rewrite(v) for k, v in x.items()}
                y["relation_name"] = "__cf_t"
                return y
            return {k: self.rewrite(v) for k, v in x.items()}
        if isinstance(x, list):
            return [self.rewrite(v) for v in x]
        return x


def _validate_top(raw: Node) -> None:
    if raw.get("type") != "SELECT_NODE":
        _refuse(f"query kind {raw.get('type')} (UNION/INTERSECT/EXCEPT and friends)")
    node = _Select.of(raw)
    if node.cte_map.map:
        _refuse("WITH / common table expressions")
    for mod in node.modifiers:
        _refuse(
            {
                "ORDER_MODIFIER": "ORDER BY (meaningless row-at-a-time)",
                "LIMIT_MODIFIER": "LIMIT/OFFSET",
                "DISTINCT_MODIFIER": "SELECT DISTINCT",
            }.get(mod.get("type"), str(mod.get("type")))
        )
    for clause, pretty in (
        (node.where_clause, "WHERE (that is filter shape, not a projection)"),
        (node.having, "HAVING"),
        (node.qualify, "QUALIFY"),
    ):
        if clause is not None:
            _refuse(pretty, clause.get("query_location"))
    if node.sample is not None:
        _refuse("USING SAMPLE")
    if node.group_expressions or node.group_sets:
        _refuse("GROUP BY")
    if node.aggregate_handling != "STANDARD_HANDLING":
        _refuse(f"aggregate handling {node.aggregate_handling}")
    if node.from_table.get("type") != "BASE_TABLE":
        _refuse("FROM must be exactly __THIS__ (joins and subqueries: a later loop)")
    ft = _BaseTable.of(node.from_table)
    if ft.table_name.upper() != "__THIS__":
        _refuse(f"FROM {ft.table_name} — the row table must be __THIS__")
    if ft.alias:
        _refuse(f"an alias on __THIS__ ({ft.alias})")
    if ft.sample is not None:
        _refuse("USING SAMPLE")


def _windows_sql(original_items: list[Node], collector: _Collector) -> str:
    """The one fit-time execution: the original select list verbatim (chain-
    pinning, discarded), each distinct window as ``__cf_wN``, keys as
    ``__cf_kM``. See the Marginalized docstring for why the original items
    must ride along."""
    doc = copy.deepcopy(_templates()["doc"])
    originals = [
        item if item.get("class") == "STAR" else dict(item, alias=f"__cf_o{n}")
        for n, item in enumerate(original_items)
    ]
    windows = [
        dict(copy.deepcopy(raw), alias=f"__cf_w{g}")
        for g, raw in enumerate(collector.windows.values())
    ]
    keys = [
        _clone("column_ref", column_names=[spelled], alias=f"__cf_k{m}")
        for m, spelled in enumerate(_key_union(collector).values())
    ]
    doc["statements"][0]["node"]["select_list"] = [*originals, *windows, *keys]
    return _deserialize(doc)


def _key_union(collector: _Collector) -> dict[str, str]:
    """Casefolded key name -> first spelling, over every group."""
    union: dict[str, str] = {}
    for group in collector.groups.values():
        for key in group.keys:
            union.setdefault(key.lower(), key)
    return union


def _fit_sql(group: _Group, collector: _Collector) -> str:
    """``SELECT DISTINCT <keys>, <this group's window columns> FROM
    __CF_WINDOWS__`` — pure value picking over the materialized windows
    result; every allowlisted aggregate is deterministic per group, so the
    tuple is constant within a group and DISTINCT collapses to exactly one
    row per group (which is also the serving-side multiplicity argument)."""
    doc = copy.deepcopy(_templates()["collapse_doc"])
    union_order = list(_key_union(collector))
    key_refs = [
        _clone(
            "column_ref",
            column_names=[f"__cf_k{union_order.index(k.lower())}"],
            alias=k,
        )
        for k in group.keys
    ]
    windows_order = list(collector.windows)
    agg_refs = [
        _clone(
            "column_ref",
            column_names=[f"__cf_w{windows_order.index(agg_key)}"],
            alias=f"__cf_a{j}",
        )
        for j, agg_key in enumerate(group.agg_keys)
    ]
    doc["statements"][0]["node"]["select_list"] = [*key_refs, *agg_refs]
    return _deserialize(doc)


def _serving_from(specs: list[ParamsSpec]) -> Node:
    """Left-deep LEFT JOIN chain — every params join, keyed or not, is a LEFT
    JOIN so multiplicity is one uniform argument. A keyless params table is
    exactly one row, so its condition is the always-true ``1 = 1``. (Never a
    CROSS join: the oracle prints those in comma form, which re-parses with
    different associativity once another join follows.)"""
    from_table = _clone("base_table")
    for i, spec in enumerate(specs):
        right = _clone("base_table", table_name=spec.name, alias=f"__cf_p{i}")
        comparisons = [
            _clone(
                "not_distinct",
                left=_clone("column_ref", column_names=["__cf_t", key], alias=""),
                right=_clone("column_ref", column_names=[f"__cf_p{i}", key], alias=""),
            )
            for key in spec.keys
        ]
        if not comparisons:
            condition = _clone("always_true")
        elif len(comparisons) == 1:
            condition = comparisons[0]
        else:
            condition = _clone("conjunction", children=comparisons)
        from_table = _clone(
            "left_join", left=from_table, right=right, condition=condition
        )
    return from_table


def marginalize(sql: str) -> Marginalized:
    """Parse a strict projection over ``__THIS__`` and marginalize its window
    aggregates. Pure: parses, rewrites, and plans — never executes."""
    if _RESERVED in sql.lower():
        _refuse(f"the reserved prefix {_RESERVED} in the SQL")
    doc = _serialize(sql)
    if len(doc["statements"]) != 1:
        _refuse("multiple SQL statements")
    node = doc["statements"][0]["node"]
    _validate_top(node)

    # Output names must survive the rewrite: DuckDB names an unaliased
    # expression column by its own printed text, and qualification changes
    # that text — so freeze every derived name as an explicit alias first.
    # Column refs are exempt: their name is the last path part, which
    # qualification never touches.
    for item in node["select_list"]:
        if item.get("class") not in ("STAR", "COLUMN_REF") and not item.get("alias"):
            item["alias"] = _expr_text(item)

    original_items = copy.deepcopy(node["select_list"])
    collector = _Collector()
    rewritten_list = collector.rewrite(node["select_list"])
    if not collector.groups:
        # No aggregates: marginalization is the identity (modulo normalization).
        return Marginalized(serving_sql=_deserialize(doc), windows_sql=None, params=())

    specs = [
        ParamsSpec(
            name=f"__CF_PARAMS_{i}__", keys=g.keys, fit_sql=_fit_sql(g, collector)
        )
        for i, g in enumerate(collector.groups.values())
    ]
    node["select_list"] = rewritten_list
    node["from_table"] = _serving_from(specs)
    return Marginalized(
        serving_sql=_deserialize(doc),
        windows_sql=_windows_sql(original_items, collector),
        params=tuple(specs),
    )
