"""Marginalization over ``__THIS__``: the fit half of a binding-time analysis.

A query is a **chain of strict projections** — the final SELECT over a CTE or
derived table, which projects over another, down to ``__THIS__``. Every
window aggregate in the chain is computed once at fit time and rewritten into
a join against a materialized params table; every scalar subquery becomes a
params table run verbatim; the chain itself is flattened by substitution, so
``serving_sql`` is one flat projection over ``__THIS__`` plus params joins:

    SELECT (age - avg(age) OVER (PARTITION BY country)) AS d FROM __THIS__
        -->  serving_sql:
    SELECT (__cf_t.age - __cf_p0.__cf_a0) AS d
    FROM __THIS__ AS __cf_t
    LEFT JOIN __CF_PARAMS_0__ AS __cf_p0
      ON (__cf_t.country IS NOT DISTINCT FROM __cf_p0.country)

Fitting is an explicit DAG — ``Marginalized.plan`` is a topologically ordered
list of named steps, each one SQL statement over previously registered
tables. ``__CF_LEVEL_{i}__`` materializes projection level i (its original
select items under their original names — the next level's verbatim input —
plus that level's windows as ``__cf_wN`` and key expressions as ``__cf_kM``);
``__CF_PARAMS_{n}__`` steps collapse params out of a level table with SELECT
DISTINCT, or run a scalar subquery verbatim. The executor in
``SQLProjection.fit`` just runs the steps in order; every intermediate is
inspectable by name and every step is plain SQL you can run by hand.

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
functions (marginalized or refused), subqueries (marginalized verbatim or
refused), and the top-level clauses (WHERE/GROUP BY/... — refused).
Everything else passes through. Subqueries are provably uncorrelated:
``FROM __THIS__`` carries no alias and no other relation is in scope, so
there is no syntax to reference the outer row.

Join predicates use IS NOT DISTINCT FROM, never ``=``: window PARTITION BY
groups NULL keys into one partition; an equality join would drop them.
Multiplicity holds by construction: every admitted window's value is a
function of its join keys (constant within the key tuple in one execution),
so DISTINCT over (keys, values) collapses to exactly one params row per key
tuple, and the LEFT JOIN matches exactly one per input row.

Design specs, in order:
docs/superpowers/specs/2026-07-29-sql-projection-marginalization-design.md
docs/superpowers/specs/2026-07-29-window-widening-design.md
docs/superpowers/specs/2026-07-29-projection-chains-fit-plan-design.md
"""

from __future__ import annotations

import copy
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import cache
from typing import Any

import duckdb
import pydantic

Node = dict[str, Any]
"""An opaque oracle AST node: carried, cloned, or grafted — never interpreted."""

_RESERVED = "__cf_"
_NOLOC = 18446744073709551615  # DuckDB's "no query_location" sentinel (u64 max)

# The marginalizability rule (spec: window-widening design): a window's value
# must be a function of row-visible values — the partition keys plus, when
# ORDER BY discriminates, the order values (RANGE/GROUPS peers share values).
# Physical position is the one thing a join key cannot carry, so positional
# windows are refused. Any WINDOW_AGGREGATE function is admitted: the fit
# plan reruns the original computation, so even order-sensitive aggregates
# freeze exactly the value the original text produced.
_RANK_FAMILY = frozenset(
    {"WINDOW_RANK", "WINDOW_RANK_DENSE", "WINDOW_PERCENT_RANK", "WINDOW_CUME_DIST"}
)
_POSITIONAL = frozenset(
    {"WINDOW_ROW_NUMBER", "WINDOW_NTILE", "WINDOW_LAG", "WINDOW_LEAD"}
)
_VALUE_FNS = frozenset({"WINDOW_FIRST_VALUE", "WINDOW_LAST_VALUE", "WINDOW_NTH_VALUE"})


class MarginalizeError(ValueError):
    """The SQL is not a projection this slice can marginalize; names why."""


@dataclass(frozen=True)
class FitStep:
    """One fit-time execution. ``kind == "sql"``: register ``name`` as the
    result of ``sql`` over previously registered tables (``reads``).
    ``kind == "fit"``: group ``reads[0]`` by the ``keys`` columns, fit a
    clone of registry transformer ``transformer`` per group on the
    ``features`` columns, and store the fitted clones under ``name``."""

    name: str
    sql: str
    reads: tuple[str, ...]
    kind: str = "sql"
    transformer: str = ""
    features: tuple[str, ...] = ()
    keys: tuple[str, ...] = ()


@dataclass(frozen=True)
class TransformSpec:
    """One transformer column in the serving output: which fit step supplies
    the fitted clones, the output column name, and the helper columns in
    ``serving_sql`` carrying its features and partition keys (the apply layer
    replaces the helper block with the transformed column in place)."""

    step: str
    alias: str
    feature_cols: tuple[str, ...]
    key_cols: tuple[str, ...]


@dataclass(frozen=True)
class ParamsSpec:
    """One params table the serving query joins: its name and the params-table
    column names of its join keys (empty for the always-one-row tables)."""

    name: str
    keys: tuple[str, ...]


@dataclass(frozen=True)
class Marginalized:
    """The rewrite: a flat serving projection plus the fit plan (a
    topologically ordered DAG of named SQL steps) that materializes its
    params tables. ``plan`` is empty when nothing needs fitting."""

    serving_sql: str
    plan: tuple[FitStep, ...]
    params: tuple[ParamsSpec, ...]
    transforms: tuple[TransformSpec, ...] = ()


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


class _CteEntry(_View):
    key: str
    value: Node  # {aliases, key_targets, materialized, query: {node}}


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
    column_name_alias: list[str] = []


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
    exclude_list: list[Any]
    replace_list: list[Any]
    rename_list: list[Any]
    qualified_exclude_list: list[Any] = []
    query_location: int


class _Subquery(_View):
    subquery_type: str
    subquery: Node  # {named_param_map, node}
    alias: str
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
    plain = _serialize("SELECT 1 FROM __THIS__")
    scalar = _serialize("SELECT 1")
    return {
        "doc": plain,
        "scalar_doc": scalar,  # from_table type EMPTY
        "base_table": join["left"],
        "left_join": join,
        "always_true": true_doc["statements"][0]["node"]["from_table"]["condition"],
        "conjunction": join["condition"],
        "not_distinct": join["condition"]["children"][0],
        "column_ref": collapse["statements"][0]["node"]["select_list"][0],
        "collapse_doc": collapse,
        "star": _serialize("SELECT __cf_t.* FROM __THIS__")["statements"][0]["node"][
            "select_list"
        ][0],
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


def _select_doc(select_list: list[Node], from_table: Node) -> str:
    doc = copy.deepcopy(_templates()["doc"])
    node = doc["statements"][0]["node"]
    node["select_list"] = select_list
    node["from_table"] = from_table
    return _deserialize(doc)


# --- the chain resolver -------------------------------------------------------


@dataclass
class _Level:
    """One projection level: its SELECT node, the names that qualify its
    source inside its expressions, and the renames its consumer applies to
    its outputs (CTE/derived-table column aliases)."""

    node: Node
    source_quals: frozenset[str]  # lowered
    output_aliases: list[str]


def _validate_level(raw: Node) -> None:
    if raw.get("type") != "SELECT_NODE":
        _refuse(f"query kind {raw.get('type')} (UNION/INTERSECT/EXCEPT and friends)")
    node = _Select.of(raw)
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


def _resolve_chain(root: Node) -> list[_Level]:
    """Walk the FROM chain down to ``__THIS__``; returns levels base-first."""
    ctes: dict[str, tuple[Node, list[str], str]] = {}
    for raw_entry in _CteMap.of(root.get("cte_map", {"map": []})).map:
        entry = _CteEntry.of(raw_entry)
        cte_node = entry.value["query"]["node"]
        ctes[entry.key.lower()] = (cte_node, list(entry.value["aliases"]), entry.key)
    used: set[str] = set()

    top_first: list[_Level] = []
    node, output_aliases = root, []
    while True:
        _validate_level(node)
        if node is not root and _CteMap.of(node["cte_map"]).map:
            _refuse("WITH nested inside a CTE or derived table")
        ft = node["from_table"]
        kind = ft.get("type")
        if kind == "BASE_TABLE":
            bt = _BaseTable.of(ft)
            if bt.sample is not None:
                _refuse("USING SAMPLE")
            name_ci = bt.table_name.lower()
            if name_ci == "__this__":
                if bt.alias:
                    _refuse(f"an alias on __THIS__ ({bt.alias})")
                if bt.column_name_alias:
                    _refuse("column aliases on __THIS__")
                top_first.append(_Level(node, frozenset({"__this__"}), output_aliases))
                break
            if name_ci in ctes:
                if name_ci in used:
                    _refuse(f"CTE {bt.table_name} referenced more than once")
                used.add(name_ci)
                cte_node, cte_aliases, spelled = ctes[name_ci]
                qual = (bt.alias or spelled).lower()
                top_first.append(_Level(node, frozenset({qual}), output_aliases))
                node = cte_node
                output_aliases = list(bt.column_name_alias) or cte_aliases
                continue
            _refuse(
                f"FROM {bt.table_name} — the row table must be __THIS__"
                " (or a CTE/derived table over it)"
            )
        if kind == "SUBQUERY":
            qual = ft.get("alias", "")
            quals = frozenset({qual.lower()}) if qual else frozenset()
            top_first.append(_Level(node, quals, output_aliases))
            node = ft["subquery"]["node"]
            output_aliases = list(ft.get("column_name_alias", []))
            continue
        _refuse(f"FROM {kind} (joins and set operations are not projections)")

    unused = [spelled for ci, (_, _, spelled) in ctes.items() if ci not in used]
    if unused:
        _refuse(f"unused CTE {unused[0]}")
    if len(top_first) > 64:
        _refuse("a projection chain deeper than 64 levels")
    return list(reversed(top_first))


# --- window and key rules (spec: window-widening design) ----------------------


def _walk_row_wise(x: Any, where: str) -> None:
    """The subtree must be row-wise: no nesting of table semantics."""
    if isinstance(x, dict):
        cls = x.get("class")
        if cls == "WINDOW":
            _refuse(f"a window function inside {where}", x.get("query_location"))
        if cls == "SUBQUERY":
            _refuse(f"a subquery inside {where}", x.get("query_location"))
        if (
            cls == "FUNCTION"
            and x.get("function_name", "").lower() in _aggregate_names()
        ):
            _refuse(
                f"aggregate {x['function_name']} inside {where}",
                x.get("query_location"),
            )
        for v in x.values():
            _walk_row_wise(v, where)
    elif isinstance(x, list):
        for v in x:
            _walk_row_wise(v, where)


def _check_frame(w: _Window) -> bool:
    """Generic frame validation; True when the frame varies with the order
    values (so they must join the key set), False when it always covers the
    whole partition."""
    if w.exclude_clause != "NO_OTHER":
        _refuse("EXCLUDE in a window frame (distinguishes tied rows)", w.query_location)
    for bound in (w.start_expr, w.end_expr):
        if bound is not None and bound.get("class") != "CONSTANT":
            _refuse("a non-constant window frame bound", w.query_location)
    if w.start == "UNBOUNDED_PRECEDING" and w.end == "UNBOUNDED_FOLLOWING":
        return False
    if w.start.endswith("_ROWS") or w.end.endswith("_ROWS"):
        _refuse(
            "a bounded ROWS frame (depends on physical row order;"
            " RANGE/GROUPS frames are supported)",
            w.query_location,
        )
    return bool(w.orders)


def _check_window(w: _Window) -> bool:
    """Validate one window; True means the order values join the key set."""
    t = w.type
    if t in _POSITIONAL:
        _refuse(
            f"window function {w.function_name} (its value depends on physical"
            " row position, which no join key can carry)",
            w.query_location,
        )
    for child in w.children:
        _walk_row_wise(child, "an aggregate")
    if w.filter_expr is not None:
        _walk_row_wise(w.filter_expr, "a FILTER")
    if t in _RANK_FAMILY:
        return True  # a function of the order values; frames don't apply
    if t == "WINDOW_AGGREGATE":
        if w.ignore_nulls:
            _refuse("IGNORE NULLS on an aggregate", w.query_location)
        return _check_frame(w)
    if t in _VALUE_FNS:
        if t == "WINDOW_NTH_VALUE" and (
            len(w.children) < 2 or w.children[1].get("class") != "CONSTANT"
        ):
            _refuse("nth_value with a non-constant n", w.query_location)
        discriminates = _check_frame(w)
        if (
            t == "WINDOW_FIRST_VALUE"
            and not w.ignore_nulls
            and w.start == "UNBOUNDED_PRECEDING"
            and w.end in ("CURRENT_ROW_RANGE", "UNBOUNDED_FOLLOWING")
        ):
            # The frame always starts at the partition start and is never
            # empty, so the value is the partition's first row regardless of
            # the current row's order value.
            return False
        return discriminates
    _refuse(f"window type {t}", w.query_location)
    raise AssertionError("unreachable")


@dataclass
class _Key:
    """One join key: ``fit_expr`` projects it in the level table (source
    terms, qualifiers stripped); ``serving_expr`` joins it back in the flat
    serving query; ``name`` is the natural column name for plain columns."""

    ident: str
    fit_expr: Node
    serving_expr: Node
    name: str | None


# --- substitution environments ------------------------------------------------


@dataclass
class _Env:
    """What one level's outputs look like from above: ordered entries of
    ("*",) — base columns pass through — or (spelled name, serving expr)."""

    entries: list[tuple[str, Node | None]]

    @property
    def exprs(self) -> dict[str, Node]:
        return {n.lower(): e for n, e in self.entries if e is not None}

    @property
    def star(self) -> bool:
        return any(e is None for _, e in self.entries)


_BASE_ENV = _Env(entries=[("*", None)])


def _item_name(item: Node) -> str | None:
    """The output name of a select item: its alias, or a column ref's last
    path part. None for stars (and anything unresolvable)."""
    if item.get("class") == "STAR":
        return None
    if item.get("alias"):
        return item["alias"]
    if item.get("class") == "COLUMN_REF":
        return _ColumnRef.of(item).column_names[-1]
    return None


def _strip_quals(x: Any, quals: frozenset[str]) -> Any:
    """Drop source-qualifier heads so an expression runs against the
    materialized level table (whose columns are bare)."""
    if isinstance(x, dict):
        if x.get("class") == "COLUMN_REF":
            names = x["column_names"]
            if len(names) > 1 and names[0].lower() in quals:
                return dict(copy.deepcopy(x), column_names=list(names[1:]))
            return copy.deepcopy(x)
        if x.get("class") == "STAR" and x.get("relation_name", "").lower() in quals:
            return dict(
                {k: _strip_quals(v, quals) for k, v in x.items()}, relation_name=""
            )
        return {k: _strip_quals(v, quals) for k, v in x.items()}
    if isinstance(x, list):
        return [_strip_quals(v, quals) for v in x]
    return x


# --- the planner --------------------------------------------------------------


@dataclass
class _Join:
    """One params table in the serving join chain."""

    keys: list[_Key] = field(default_factory=list)
    agg_keys: list[str] = field(default_factory=list)  # window identities
    level: int | None = None  # None for scalar-subquery params

    def colnames(self) -> list[str]:
        return [
            k.name if k.name is not None else f"__cf_x{j}"
            for j, k in enumerate(self.keys)
        ]


class _LevelRewriter:
    """Rewrites one level's select items: windows become params references,
    scalar subqueries become params references, column refs resolve through
    the substitution environment. Pure — never mutates its input."""

    def __init__(self, planner: _Planner, level_i: int, env: _Env, level: _Level):
        self.planner = planner
        self.level_i = level_i
        self.env = env
        self.level = level
        self.risky_aliases: set[str] = set()  # schema-free mode only
        self.alias_exprs: dict[str, Node] = {}  # explicit-env lateral aliases

    # -- column resolution

    def _resolve_ref(self, x: Node) -> Node:
        names = _ColumnRef.of(x).column_names
        if names and names[0].lower().startswith(_RESERVED):
            return copy.deepcopy(x)  # a reference this module created
        was_bare = len(names) == 1
        if len(names) > 1 and names[0].lower() in self.level.source_quals:
            names = names[1:]
        elif len(names) > 1 and names[0].lower() == "__this__" and self.level_i > 0:
            _refuse(
                "__THIS__ is not in scope here (this level reads a CTE or"
                " derived table)",
                x.get("query_location"),
            )
        head = names[0].lower() if names else ""
        if was_bare and head in self.risky_aliases:
            # DuckDB prefers the table column over a lateral alias when both
            # exist — undecidable without a schema, so refuse with a hint.
            # (An alias that merely re-states a column of the same name is
            # benign: both readings are the same value.)
            _refuse(
                f"{names[0]} may be a lateral alias of an earlier select item"
                " — qualify the column (e.g. __THIS__.{0}) or rename the"
                " alias".format(names[0]),
                x.get("query_location"),
            )
        if head in self.env.exprs:
            target = self.env.exprs[head]
            if len(names) > 1:
                # Struct access composes through a plain column: extend the
                # path. Through a computed expression it stays refused.
                if target.get("class") == "COLUMN_REF" and target["column_names"][
                    :1
                ] == ["__cf_t"]:
                    return dict(
                        copy.deepcopy(target),
                        column_names=[*target["column_names"], *names[1:]],
                    )
                _refuse(
                    "struct-field access through a projected expression",
                    x.get("query_location"),
                )
            return copy.deepcopy(target)
        if was_bare and head in self.alias_exprs:
            # DuckDB's lateral-alias rule, resolvable because the environment
            # is explicit: the column did not exist, so the alias applies.
            return copy.deepcopy(self.alias_exprs[head])
        if self.env.star:
            return dict(copy.deepcopy(x), column_names=["__cf_t", *names])
        _refuse(f"unknown column {'.'.join(names)}", x.get("query_location"))
        raise AssertionError("unreachable")

    # -- windows

    def _key_of(self, entry: Node) -> _Key:
        expr = entry
        if expr.get("class") == "COLLATE":
            # Key on the raw child value: raw-equal implies collated-equal
            # implies peer, so multiplicity survives (joining on the
            # *collated* value could match several params rows).
            return self._key_of(expr["child"])
        quals = self.level.source_quals
        if expr.get("class") == "COLUMN_REF":
            names = _ColumnRef.of(expr).column_names
            if len(names) > 1 and names[0].lower() in quals:
                names = names[1:]
            if len(names) == 1:
                bare = names[0]
                return _Key(
                    ident=bare.lower(),
                    fit_expr=_clone("column_ref", column_names=[bare], alias=""),
                    serving_expr=self.rewrite(expr),
                    name=bare,
                )
        _walk_row_wise(expr, "a partition or order key")
        return _Key(
            ident=_stripped(_strip_quals(expr, quals)),
            fit_expr=dict(_strip_quals(expr, quals), alias=""),
            serving_expr=self.rewrite(expr),
            name=None,
        )

    def _params_ref(self, raw: Node, w: _Window, discriminates: bool) -> Node:
        keys = [self._key_of(p) for p in w.partitions]
        if discriminates:
            keys += [self._key_of(o["expression"]) for o in w.orders]
        seen: set[str] = set()
        uniq = [k for k in keys if not (k.ident in seen or seen.add(k.ident))]
        idx, join = self.planner.group_for(self.level_i, uniq)
        agg_key = _stripped(raw)
        self.planner.level_window(self.level_i, agg_key, raw)
        if agg_key not in join.agg_keys:
            join.agg_keys.append(agg_key)
        col_idx = join.agg_keys.index(agg_key)
        return _clone(
            "column_ref",
            column_names=[f"__cf_p{idx}", f"__cf_a{col_idx}"],
            alias=w.alias,
        )

    # -- the walk

    def _is_transformer_call(self, item: Node) -> str | None:
        return _is_tf_node(item)

    def _tf_bundle(self, w_raw: Node) -> list[tuple[str, Node, Node]]:
        """(field name, fit expr in level terms, serving expr) per feature."""
        children = w_raw["children"]
        if len(children) == 0:
            _refuse(
                "a transformer call with no arguments — note fn(*) parses to"
                " zero arguments; pass a column or struct_pack(...)",
                w_raw.get("query_location"),
            )
        if len(children) != 1:
            _refuse(
                "a transformer call takes exactly one bundle argument (a"
                " column or struct_pack(...)); configuration belongs in the"
                " registered object",
                w_raw.get("query_location"),
            )
        c = children[0]
        quals = self.level.source_quals
        if c.get("class") == "COLUMN_REF":
            fname = _ColumnRef.of(c).column_names[-1]
            return [(fname, _strip_quals(c, quals), dict(self.rewrite(c), alias=""))]
        if (
            c.get("class") == "FUNCTION"
            and c.get("function_name", "").lower() == "struct_pack"
        ):
            out: list[tuple[str, Node, Node]] = []
            names: set[str] = set()
            for ch in c["children"]:
                if not ch.get("alias"):
                    _refuse(
                        "struct_pack fields in a transformer bundle must be"
                        " named (f := expr)",
                        c.get("query_location"),
                    )
                if ch["alias"].lower() in names:
                    _refuse(f"duplicate bundle field {ch['alias']}")
                names.add(ch["alias"].lower())
                _walk_row_wise(ch, "a transformer bundle")
                out.append(
                    (
                        ch["alias"],
                        dict(_strip_quals(ch, quals), alias=""),
                        dict(self.rewrite(ch), alias=""),
                    )
                )
            return out
        _refuse(
            "a transformer bundle must be a column or struct_pack(...)",
            c.get("query_location"),
        )
        raise AssertionError("unreachable")

    def rewrite(self, x: Any) -> Any:
        if isinstance(x, dict):
            cls = x.get("class")
            if cls == "WINDOW":
                if self._is_transformer_call(x) is not None:
                    _refuse(
                        "a transformer call is only supported as a top-level"
                        " select item",
                        x.get("query_location"),
                    )
                if self.alias_exprs:
                    self._check_no_lateral_in_window(x)
                w = _Window.of(x)
                discriminates = _check_window(w)
                return self._params_ref(x, w, discriminates)
            if cls == "SUBQUERY":
                return self.planner.scalar_ref(_Subquery.of(x), x)
            if cls == "PARAMETER":
                _refuse("a prepared-statement parameter", x.get("query_location"))
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
                return self._resolve_ref(x)
            if cls == "STAR":
                return self._rewrite_star(x)
            return {k: self.rewrite(v) for k, v in x.items()}
        if isinstance(x, list):
            return [self.rewrite(v) for v in x]
        return x

    def _rewrite_star(self, x: Node) -> Node:
        """A star inside an expression (``count(*)``): plain row reference,
        qualified to the base table. Anything fancier is refused here —
        expansion is a select-item affair."""
        star = _Star.of(x)
        if star.columns or star.expr is not None:
            _refuse("COLUMNS(...) inside an expression", star.query_location)
        if (
            star.exclude_list
            or star.replace_list
            or star.rename_list
            or star.qualified_exclude_list
        ):
            _refuse("* with modifiers inside an expression", star.query_location)
        rel = star.relation_name.lower()
        if rel and rel not in self.level.source_quals and rel != "__this__":
            _refuse(
                f"star qualified by unknown relation {star.relation_name}",
                star.query_location,
            )
        y = {k: self.rewrite(v) for k, v in x.items()}
        y["relation_name"] = "__cf_t"
        return y

    def _check_no_lateral_in_window(self, x: Any) -> None:
        """A window's fit-side text runs against the source relation, which
        has no same-level lateral aliases — refuse them by name."""
        if isinstance(x, dict):
            if x.get("class") == "COLUMN_REF":
                names = x["column_names"]
                if (
                    len(names) == 1
                    and names[0].lower() in self.alias_exprs
                    and names[0].lower() not in self.env.exprs
                ):
                    _refuse(
                        f"lateral alias {names[0]} inside a window function"
                        " (name the column or repeat the expression)",
                        x.get("query_location"),
                    )
            for v in x.values():
                self._check_no_lateral_in_window(v)
        elif isinstance(x, list):
            for v in x:
                self._check_no_lateral_in_window(v)

    def _expand_star(self, star_node: Node) -> list[tuple[str, Node]]:
        """Expand a star/COLUMNS over a fully explicit environment, composing
        EXCLUDE/REPLACE/RENAME; returns (output name, rewritten expr) pairs."""
        star = _Star.of(star_node)
        if star.qualified_exclude_list:
            _refuse("qualified EXCLUDE", star.query_location)
        names = [n for n, _ in self.env.entries]
        known = {n.lower() for n in names}
        if star.columns:
            expr = star.expr
            if not (isinstance(expr, dict) and expr.get("class") == "CONSTANT"):
                _refuse(
                    "COLUMNS with a lambda or non-constant pattern",
                    star.query_location,
                )
            pattern = str(expr["value"]["value"])
            # The oracle matches its own regex dialect — never Python's re.
            matched = {
                n
                for n in names
                if duckdb.execute(
                    "SELECT regexp_matches(?, ?)", [n, pattern]
                ).fetchone()[0]
            }
            if not matched:
                _refuse(f"COLUMNS({pattern!r}) matched no columns", star.query_location)
        else:
            matched = set(names)
        exclude = {e.lower() for e in star.exclude_list}
        replace = {
            entry["key"].lower(): self.rewrite(entry["value"])
            for entry in star.replace_list
        }
        rename = {
            entry["key"]["column"].lower(): entry["value"] for entry in star.rename_list
        }
        for modifier in (exclude, replace, rename):
            unknown = set(modifier) - known
            if unknown:
                _refuse(
                    f"a star modifier names unknown column {sorted(unknown)[0]}",
                    star.query_location,
                )
        out: list[tuple[str, Node]] = []
        for name, expr in self.env.entries:
            if name not in matched or name.lower() in exclude:
                continue
            chosen = replace.get(name.lower())
            chosen = copy.deepcopy(chosen if chosen is not None else expr)
            out.append((rename.get(name.lower(), name), dict(chosen, alias="")))
        if not out:
            _refuse("* expanded to no columns", star.query_location)
        return out

    def rewrite_items(
        self, items: list[Node], is_final: bool
    ) -> tuple[list[Node], list[tuple[str, Node | None]]]:
        """Rewrite the select list, expanding stars over explicit
        environments. Returns the rewritten items plus this level's output
        entries (the environment the next level resolves against)."""
        out: list[Node] = []
        entries: list[tuple[str, Node | None]] = []
        names_seen: set[str] = set()
        explicit = not self.env.star
        for item in items:
            tf_name = self._is_transformer_call(item)
            if tf_name is not None:
                helpers = self.planner.transformer_item(self, item, tf_name, is_final)
                out.extend(helpers)
                continue
            if item.get("class") == "STAR":
                star = _Star.of(item)
                if explicit:
                    for name, expr in self._expand_star(item):
                        entries.append((name, expr))
                        out.append(dict(copy.deepcopy(expr), alias=name))
                    continue
                # Schema-free: base columns are unknowable, so modifiers are
                # only expressible in a final level directly over __THIS__.
                has_mods = star.exclude_list or star.replace_list or star.rename_list
                if len(self.env.entries) == 1:
                    if star.columns or star.expr is not None:
                        _refuse(
                            "COLUMNS(...) without a schema (declare one via"
                            " this_model)",
                            star.query_location,
                        )
                    if has_mods and not is_final:
                        _refuse(
                            "* with EXCLUDE/REPLACE/RENAME in a non-final"
                            " level (declare a schema via this_model)",
                            star.query_location,
                        )
                    entries.append(("*", None))
                    if has_mods:
                        y = {k: self.rewrite(v) for k, v in item.items()}
                        y["relation_name"] = "__cf_t"
                        out.append(y)
                    else:
                        out.append(self.rewrite(item))
                    continue
                if has_mods or star.columns or star.expr is not None:
                    _refuse(
                        "* with modifiers over a CTE or derived table without"
                        " a schema (declare one via this_model)",
                        star.query_location,
                    )
                for name, expr in self.env.entries:
                    if expr is None:
                        entries.append(("*", None))
                        out.append(_clone("star"))
                    else:
                        entries.append((name, copy.deepcopy(expr)))
                        out.append(dict(copy.deepcopy(expr), alias=name))
                continue
            rewritten = self.rewrite(item)
            if item.get("alias"):
                # Substitution may return an env expression whose alias field
                # is empty; the item's output name must survive it.
                rewritten = dict(rewritten, alias=item["alias"])
            out.append(rewritten)
            name = _item_name(item)
            if explicit:
                if name is not None:
                    self.alias_exprs.setdefault(
                        name.lower(), dict(copy.deepcopy(rewritten), alias="")
                    )
            elif item.get("alias") and not (
                item.get("class") == "COLUMN_REF"
                and _ColumnRef.of(item).column_names[-1].lower()
                == item["alias"].lower()
            ):
                self.risky_aliases.add(item["alias"].lower())
            if not is_final:
                if name is None:
                    _refuse("an unnameable select item in a non-final level")
                if name.lower() in names_seen:
                    _refuse(f"duplicate output name {name} in a non-final level")
                names_seen.add(name.lower())
                entries.append((name, dict(copy.deepcopy(rewritten), alias="")))
            else:
                entries.append((name or "", None))
        return out, entries


class _Planner:
    """Accumulates the fit plan, the params joins, and the per-level windows
    while the chain is rewritten level by level."""

    def __init__(self) -> None:
        self.plan: list[FitStep] = []
        self.joins: list[_Join] = []
        self.groups: dict[tuple[int, tuple[str, ...]], int] = {}
        self.windows: dict[int, dict[str, Node]] = {}  # level -> ident -> raw
        self.scalars: dict[str, int] = {}
        self.key_count = 0
        self.window_count = 0
        self.feature_count = 0
        self.window_names: dict[tuple[int, str], str] = {}
        self.key_names: dict[tuple[int, str], str] = {}
        self.lookup: Any = None  # transformer resolver: name -> object | None
        self.tf_steps: list[dict] = []
        self.tf_specs: list[TransformSpec] = []

    def key_name(self, level_i: int, key: _Key) -> str:
        kid = (level_i, key.ident)
        if kid not in self.key_names:
            self.key_names[kid] = f"__cf_k{self.key_count}"
            self.key_count += 1
        return self.key_names[kid]

    def transformer_item(
        self, rw: _LevelRewriter, raw: Node, name: str, is_final: bool
    ) -> list[Node]:
        """One transformer window as a top-level item: validate, plan its fit
        step, and return the serving helper items (features + keys)."""
        if raw.get("schema"):
            _refuse(
                f"namespaced transformer {raw['schema']}.{name}"
                " (the curated-namespace index is a later loop)",
                raw.get("query_location"),
            )
        if not is_final:
            _refuse(
                "a transformer call in a non-final level", raw.get("query_location")
            )
        obj = self.lookup(name) if self.lookup is not None else None
        if obj is None:
            _refuse(
                f"unknown window function {name} — not a DuckDB aggregate, and"
                " no transformer of that name in the registry or caller scope",
                raw.get("query_location"),
            )
        if not (hasattr(obj, "fit") and hasattr(obj, "transform")):
            _refuse(f"transformer {name} has no fit/transform")
        w = _Window.of(raw)
        if w.orders or w.arg_orders:
            _refuse("ORDER BY on a transformer window", w.query_location)
        if (
            w.start != "UNBOUNDED_PRECEDING"
            or w.end != "CURRENT_ROW_RANGE"
            or w.start_expr is not None
            or w.end_expr is not None
        ):
            _refuse("a frame on a transformer window", w.query_location)
        if w.filter_expr is not None or w.distinct or w.ignore_nulls:
            _refuse(
                "FILTER/DISTINCT/IGNORE NULLS on a transformer window",
                w.query_location,
            )
        keys = [rw._key_of(p) for p in w.partitions]
        seen: set[str] = set()
        keys = [k for k in keys if not (k.ident in seen or seen.add(k.ident))]
        bundle = rw._tf_bundle(raw)
        j = len(self.tf_steps)
        step_name = f"__CF_FIT_{j}__"
        feature_fit = []
        helpers: list[Node] = []
        for i, (fname, fit_expr, serving_expr) in enumerate(bundle):
            col = f"__cf_f{self.feature_count}"
            self.feature_count += 1
            feature_fit.append((fname, col, fit_expr))
            helpers.append(dict(serving_expr, alias=f"__cf_tf{j}_f{i}"))
        key_cols = []
        for m, k in enumerate(keys):
            self.key_name(rw.level_i, k)
            helpers.append(
                dict(copy.deepcopy(k.serving_expr), alias=f"__cf_tf{j}_k{m}")
            )
            key_cols.append(f"__cf_tf{j}_k{m}")
        self.tf_steps.append(
            {
                "level": rw.level_i,
                "name": step_name,
                "transformer": name,
                "feature_fit": feature_fit,
                "keys": keys,
            }
        )
        self.tf_specs.append(
            TransformSpec(
                step=step_name,
                alias=raw.get("alias") or name,
                feature_cols=tuple(f"__cf_tf{j}_f{i}" for i in range(len(bundle))),
                key_cols=tuple(key_cols),
            )
        )
        return helpers

    def group_for(self, level_i: int, keys: list[_Key]) -> tuple[int, _Join]:
        ident = (level_i, tuple(k.ident for k in keys))
        if ident not in self.groups:
            self.groups[ident] = len(self.joins)
            self.joins.append(_Join(keys=list(keys), level=level_i))
            for k in keys:
                kid = (level_i, k.ident)
                if kid not in self.key_names:
                    self.key_names[kid] = f"__cf_k{self.key_count}"
                    self.key_count += 1
        idx = self.groups[ident]
        return idx, self.joins[idx]

    def level_window(self, level_i: int, agg_key: str, raw: Node) -> None:
        per_level = self.windows.setdefault(level_i, {})
        if agg_key not in per_level:
            per_level[agg_key] = copy.deepcopy(raw)
            self.window_names[(level_i, agg_key)] = f"__cf_w{self.window_count}"
            self.window_count += 1

    def scalar_ref(self, view: _Subquery, raw: Node) -> Node:
        if view.subquery_type not in ("SCALAR", "EXISTS"):
            _refuse(
                "an IN/ANY/ALL subquery (its value varies per row — a"
                " membership function, not a scalar)",
                view.query_location,
            )
        _validate_subquery_tables(view.subquery["node"])
        ident = _stripped(raw)
        if ident not in self.scalars:
            idx = len(self.joins)
            self.scalars[ident] = idx
            self.joins.append(_Join(keys=[], level=None))
            doc = copy.deepcopy(_templates()["scalar_doc"])
            doc["statements"][0]["node"]["select_list"] = [
                dict(copy.deepcopy(raw), alias="__cf_a0")
            ]
            self.plan.append(
                FitStep(
                    name=f"__CF_PARAMS_{idx}__",
                    sql=_deserialize(doc),
                    reads=("__THIS__",),
                )
            )
        idx = self.scalars[ident]
        return _clone(
            "column_ref",
            column_names=[f"__cf_p{idx}", "__cf_a0"],
            alias=raw.get("alias", ""),
        )


def _validate_subquery_tables(sub_node: Node) -> None:
    """Every base table inside a subquery must be ``__THIS__`` (the
    subquery's own CTEs are fine — the step runs verbatim)."""
    own_ctes: set[str] = set()

    def collect(x: Any) -> None:
        if isinstance(x, dict):
            if "cte_map" in x and isinstance(x["cte_map"], dict):
                for entry in x["cte_map"].get("map", []):
                    own_ctes.add(entry.get("key", "").lower())
            for v in x.values():
                collect(v)
        elif isinstance(x, list):
            for v in x:
                collect(v)

    def check(x: Any) -> None:
        if isinstance(x, dict):
            if x.get("type") == "BASE_TABLE":
                name = x.get("table_name", "")
                if name.lower() != "__this__" and name.lower() not in own_ctes:
                    _refuse(
                        f"table {name} inside a subquery (only __THIS__ and"
                        " the subquery's own CTEs)",
                        x.get("query_location"),
                    )
            for v in x.values():
                check(v)
        elif isinstance(x, list):
            for v in x:
                check(v)

    collect(sub_node)
    check(sub_node)


def _is_tf_node(item: Node) -> str | None:
    """The lowered name when the node is a transformer window call (an
    unknown-to-DuckDB or namespaced function with OVER), else None."""
    if item.get("class") != "WINDOW" or item.get("type") != "WINDOW_AGGREGATE":
        return None
    name = item.get("function_name", "").lower()
    if name in _aggregate_names() and not item.get("schema"):
        return None
    return name


def _has_windows(x: Any) -> bool:
    if isinstance(x, dict):
        if x.get("class") == "WINDOW":
            return True
        return any(_has_windows(v) for v in x.values())
    if isinstance(x, list):
        return any(_has_windows(v) for v in x)
    return False


# --- fit-plan step builders ---------------------------------------------------


def _level_step(
    planner: _Planner, level_i: int, level: _Level, source_name: str
) -> FitStep:
    """``__CF_LEVEL_{i}__``: the level's original items under their original
    output names (the next level's verbatim input) plus its windows and key
    expressions, reading from the previous level's table."""
    quals = level.source_quals
    originals = []
    for item in level.node["select_list"]:
        if _is_tf_node(item) is not None:
            continue  # not executable SQL; its features ride as __cf_f cols
        fit_item = _strip_quals(item, quals)
        originals.append(fit_item)
    windows = [
        dict(_strip_quals(raw, quals), alias=planner.window_names[(level_i, ident)])
        for ident, raw in planner.windows.get(level_i, {}).items()
    ]
    keys: list[Node] = []
    emitted: set[str] = set()
    key_lists = [j.keys for j in planner.joins if j.level == level_i] + [
        t["keys"] for t in planner.tf_steps if t["level"] == level_i
    ]
    for klist in key_lists:
        for k in klist:
            kname = planner.key_names[(level_i, k.ident)]
            if kname not in emitted:
                emitted.add(kname)
                keys.append(dict(copy.deepcopy(k.fit_expr), alias=kname))
    features = [
        dict(copy.deepcopy(fit_expr), alias=col)
        for t in planner.tf_steps
        if t["level"] == level_i
        for (_, col, fit_expr) in t["feature_fit"]
    ]
    from_table = _clone("base_table", table_name=source_name, alias="")
    return FitStep(
        name=f"__CF_LEVEL_{level_i}__",
        sql=_select_doc([*originals, *windows, *keys, *features], from_table),
        reads=(source_name,),
    )


def _collapse_step(planner: _Planner, idx: int, join: _Join) -> FitStep:
    doc = copy.deepcopy(_templates()["collapse_doc"])
    level_name = f"__CF_LEVEL_{join.level}__"
    key_refs = [
        _clone(
            "column_ref",
            column_names=[planner.key_names[(join.level, k.ident)]],
            alias=colname,
        )
        for k, colname in zip(join.keys, join.colnames(), strict=True)
    ]
    agg_refs = [
        _clone(
            "column_ref",
            column_names=[planner.window_names[(join.level, agg_key)]],
            alias=f"__cf_a{j}",
        )
        for j, agg_key in enumerate(join.agg_keys)
    ]
    node = doc["statements"][0]["node"]
    node["select_list"] = [*key_refs, *agg_refs]
    node["from_table"] = _clone("base_table", table_name=level_name, alias="")
    return FitStep(
        name=f"__CF_PARAMS_{idx}__", sql=_deserialize(doc), reads=(level_name,)
    )


def _serving_from(joins: list[_Join]) -> Node:
    """Left-deep LEFT JOIN chain — every params join, keyed or not, is a LEFT
    JOIN so multiplicity is one uniform argument. A keyless params table is
    exactly one row, so its condition is the always-true ``1 = 1``. (Never a
    CROSS join: the oracle prints those in comma form, which re-parses with
    different associativity once another join follows.) Keys join by their
    serving expression, which may reference earlier params tables — creation
    order keeps the chain topological."""
    from_table = _clone("base_table")
    for i, join in enumerate(joins):
        right = _clone(
            "base_table", table_name=f"__CF_PARAMS_{i}__", alias=f"__cf_p{i}"
        )
        comparisons = [
            _clone(
                "not_distinct",
                left=copy.deepcopy(key.serving_expr),
                right=_clone(
                    "column_ref", column_names=[f"__cf_p{i}", colname], alias=""
                ),
            )
            for key, colname in zip(join.keys, join.colnames(), strict=True)
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


# --- the entry point ----------------------------------------------------------


def marginalize(
    sql: str,
    columns: Sequence[str] | None = None,
    transformers: Any = None,
) -> Marginalized:
    """Parse a chain of strict projections over ``__THIS__`` and marginalize
    it. Pure: parses, rewrites, and plans — never executes.

    ``columns``, when given, is the declared ``__THIS__`` schema (names, in
    order). The base environment then becomes explicit: unknown columns
    refuse at construction, stars and COLUMNS expand with their modifiers at
    any level, and lateral aliases resolve by DuckDB's column-wins rule."""
    if _RESERVED in sql.lower():
        _refuse(f"the reserved prefix {_RESERVED} in the SQL")
    base_env = _BASE_ENV
    if columns is not None:
        cols = list(columns)
        for c in cols:
            if c.lower().startswith(_RESERVED):
                _refuse(f"the reserved prefix {_RESERVED} in column {c}")
        if len({c.lower() for c in cols}) != len(cols):
            _refuse("duplicate column names in the declared schema")
        base_env = _Env(
            [
                (c, _clone("column_ref", column_names=["__cf_t", c], alias=""))
                for c in cols
            ]
        )
    doc = _serialize(sql)
    if len(doc["statements"]) != 1:
        _refuse("multiple SQL statements")
    root = doc["statements"][0]["node"]
    levels = _resolve_chain(root)

    # Output names must survive the rewrite: DuckDB names an unaliased
    # expression column by its own printed text, and substitution changes
    # that text — so freeze every derived name as an explicit alias first.
    # Column refs are exempt: their name is the last path part, which
    # neither qualification nor a params reference touches... except that a
    # substituted expression *does* change a ref's name, so at upper levels
    # refs resolving through the env are frozen too.
    for i, level in enumerate(levels):
        for item in level.node["select_list"]:
            if item.get("class") == "STAR" or item.get("alias"):
                continue
            if item.get("class") == "COLUMN_REF" and i == 0:
                continue
            if item.get("class") == "COLUMN_REF":
                item["alias"] = _ColumnRef.of(item).column_names[-1]
            else:
                item["alias"] = _expr_text(item)

    # CTE / derived-table column aliases rename the level's outputs — on the
    # items themselves, so both the serving env and the fit-side level table
    # carry the renamed columns.
    for level in levels:
        if not level.output_aliases:
            continue
        items = level.node["select_list"]
        if len(level.output_aliases) > len(items):
            _refuse("more column aliases than columns")
        if any(it.get("class") == "STAR" for it in items[: len(level.output_aliases)]):
            _refuse("column aliases on a level that projects *")
        for pos, alias in enumerate(level.output_aliases):
            items[pos]["alias"] = alias

    windows_at = [i for i, lvl in enumerate(levels) if _has_windows(lvl.node)]
    deepest = max(windows_at, default=-1)

    planner = _Planner()
    planner.lookup = transformers.get if hasattr(transformers, "get") else transformers
    env = base_env
    final_items: list[Node] = []
    for i, level in enumerate(levels):
        is_final = i == len(levels) - 1
        rewriter = _LevelRewriter(planner, i, env, level)
        rewritten, entries = rewriter.rewrite_items(level.node["select_list"], is_final)
        if i <= deepest:
            source = "__THIS__" if i == 0 else f"__CF_LEVEL_{i - 1}__"
            planner.plan.append(_level_step(planner, i, level, source))
            for idx, join in enumerate(planner.joins):
                if join.level == i:
                    planner.plan.append(_collapse_step(planner, idx, join))
            for t in planner.tf_steps:
                if t["level"] == i:
                    planner.plan.append(
                        FitStep(
                            name=t["name"],
                            sql="",
                            reads=(f"__CF_LEVEL_{i}__",),
                            kind="fit",
                            transformer=t["transformer"],
                            features=tuple(col for (_, col, _) in t["feature_fit"]),
                            keys=tuple(
                                planner.key_names[(i, k.ident)] for k in t["keys"]
                            ),
                        )
                    )
        if is_final:
            final_items = rewritten
            break
        env = _Env(entries)

    if (
        not planner.joins
        and not planner.tf_steps
        and len(levels) == 1
        and base_env is _BASE_ENV
    ):
        # No aggregates, no chain, no schema: marginalization is the identity
        # (modulo normalization). With a declared schema the rewrite always
        # canonicalizes — stars/COLUMNS expand, lateral aliases inline.
        return Marginalized(serving_sql=_deserialize(doc), plan=(), params=())

    root["select_list"] = final_items
    root["from_table"] = _serving_from(planner.joins)
    root["cte_map"] = {"map": []}
    specs = tuple(
        ParamsSpec(name=f"__CF_PARAMS_{i}__", keys=tuple(j.colnames()))
        for i, j in enumerate(planner.joins)
    )
    return Marginalized(
        serving_sql=_deserialize(doc),
        plan=tuple(planner.plan),
        params=specs,
        transforms=tuple(planner.tf_specs),
    )
