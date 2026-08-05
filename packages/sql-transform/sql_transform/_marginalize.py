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
docs/superpowers/specs/2026-07-29-udf-protocol-serving-calls-design.md
"""

from __future__ import annotations

import copy
import json
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
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
    # BOOLEAN level-table column gating which rows enter the fit (FILTER on
    # the fit call); "" fits on every row.
    filter_col: str = ""
    # In-call ORDER BY keys as (level-table column, ascending, nulls_first,
    # collation) in declared order — each fit scope stably sorts by them
    # before fit; "" collation means the default (binary).
    order_by: tuple[tuple[str, bool, bool, str], ...] = ()
    # The bundle's field names, positionally aligned with ``features`` — the
    # input struct type S (DRAFT-24), passed to sklearn as input_features so
    # the learned output names can refer to them.
    feature_names: tuple[str, ...] = ()


@dataclass(frozen=True)
class UDFSpec:
    """One UDF the serving SQL calls: the scalar function name in the SQL,
    the ``kind="fit"`` plan step that produces both its params table (same
    name: the join keys plus the ``__cf_est`` instance-id column) and its
    fitted instances, and the registry name of the prototype.

    ``field`` is None for a whole-value call (``__cf_tf{j}``) and the
    requested output field name for a field access, which serves as a field
    read over that same one call (TASK-63) — the name is checked against
    the fitted output at fit time (DRAFT-24, the P7 carve-out). One spec
    per distinct (name, field) request."""

    name: str
    step: str
    transformer: str
    field: str | None = None
    # True when the UDF's OWN output struct crosses the output boundary, so
    # its learned field names become row-model fields. θ export registers a
    # spec (the step must exist) without ever serving that struct — the
    # handle is struct_pack of a tag and an id, not the transform's output.
    whole: bool = False


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
    udfs: tuple[UDFSpec, ...] = ()
    scalar_udfs: tuple[str, ...] = ()
    # One entry per ``unnest(tfm(...))`` output item, in output order: the
    # fit step whose LEARNED field names become that item's columns. Their
    # collision check runs at fit, where the names exist.
    unnest_items: tuple[str, ...] = ()
    # The output names that are known at construction (every non-unnest
    # final item) — what an expanded name must not collide with.
    unnest_siblings: tuple[str, ...] = ()


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
def _known_functions() -> frozenset[str]:
    """Every function name the oracle knows, of any type. A scalar call
    outside this set is a UDF candidate — resolved against the registry or
    the caller's scope, never guessed."""
    rows = duckdb.execute(
        "SELECT DISTINCT function_name FROM duckdb_functions()"
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
        # FILTER fit semantics ride on DuckDB's own cast: the level table
        # materializes CAST(pred AS BOOLEAN), so a non-boolean predicate
        # keeps DuckDB's nonzero-is-true reading (measured 2026-08-05).
        "bool_cast": _serialize("SELECT CAST(k AS BOOLEAN) FROM __CF_WINDOWS__")[
            "statements"
        ][0]["node"]["select_list"][0],
        "function": _serialize("SELECT __cf_fn(k) FROM __CF_WINDOWS__")["statements"][
            0
        ]["node"]["select_list"][0],
        # θ export (slice 6): the handle IS the wire mechanism — a type tag
        # naming the minted UDF plus the joined instance id. NULL id (an
        # unseen group) makes the WHOLE handle NULL: P14's story, not a
        # half-built handle with a live type tag. Graft slots: the two
        # `k` refs (the params id column) and the type constant's value.
        "theta": _serialize(
            "SELECT CASE WHEN k IS NULL THEN NULL ELSE"
            " struct_pack(type := '__cf_ty', id := k) END FROM __CF_WINDOWS__"
        )["statements"][0]["node"]["select_list"][0],
        "star": _serialize("SELECT __cf_t.* FROM __THIS__")["statements"][0]["node"][
            "select_list"
        ][0],
    }


def _clone(template_name: str, **fields: Any) -> Node:
    node = copy.deepcopy(_templates()[template_name])
    node.update(fields)
    return node


def _alias_sig(node: Node) -> str:
    """Lowered INNER aliases (named arguments, struct_pack field names) in
    traversal order. ``_stripped`` erases every alias — right for the output
    name, wrong for named args, whose names are semantic (P16a's family;
    review round 2026-08-05: two windows differing only in a struct field
    name collapsed to one params column and served the first's value)."""
    out: list[str] = []

    def walk(x: Any, root: bool) -> None:
        if isinstance(x, dict):
            if not root and x.get("alias"):
                out.append(str(x["alias"]).lower())
            for v in x.values():
                walk(v, False)
        elif isinstance(x, list):
            for v in x:
                walk(v, False)

    walk(node, True)
    return ",".join(out)


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


def _walk_row_wise(x: Any, where: str, lookup: Any = None) -> None:
    """The subtree must be row-wise: no nesting of table semantics. With a
    ``lookup``, transformer calls (bare sugar, split halves) are screened
    too — composition is TASK-65, parked, and un-screened text would ride
    verbatim into fit-side SQL and die mid-fit (review round 2026-08-05)."""
    if isinstance(x, dict):
        cls = x.get("class")
        if cls == "WINDOW":
            _refuse(f"a window function inside {where}", x.get("query_location"))
        if cls == "SUBQUERY":
            _refuse(f"a subquery inside {where}", x.get("query_location"))
        if cls == "FUNCTION":
            if x.get("function_name", "").lower() in _aggregate_names():
                _refuse(
                    f"aggregate {x['function_name']} inside {where}",
                    x.get("query_location"),
                )
            if lookup is not None and (
                _bare_tf_name(x, lookup) is not None
                or _split_transform_name(x, lookup) is not None
                or _fit_scalar_name(x, lookup) is not None
            ):
                _refuse(
                    f"a transformer call inside {where}",
                    x.get("query_location"),
                )
        for v in x.values():
            _walk_row_wise(v, where, lookup)
    elif isinstance(x, list):
        for v in x:
            _walk_row_wise(v, where, lookup)


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
    ("*",) — base columns pass through — or (spelled name, serving expr).
    ``private`` names the lower level's private columns, which never cross
    levels — reading one refuses by name."""

    entries: list[tuple[str, Node | None]]
    private: frozenset[str] = frozenset()

    @property
    def exprs(self) -> dict[str, Node]:
        return {n.lower(): e for n, e in self.entries if e is not None}

    @property
    def star(self) -> bool:
        return any(e is None for _, e in self.entries)


_BASE_ENV = _Env(entries=[("*", None)])


def _contains_class(x: Any, cls: str) -> bool:
    if isinstance(x, dict):
        return x.get("class") == cls or any(_contains_class(v, cls) for v in x.values())
    if isinstance(x, list):
        return any(_contains_class(v, cls) for v in x)
    return False


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
        # Explicit-env lateral aliases, stored as EXPANDED raw expressions in
        # level terms — consumers β-reduce before any rewrite, so windows and
        # fit-side SQL always see closed expressions (slice-2 addendum).
        self.raw_aliases: dict[str, Node] = {}
        self.private_names: set[str] = set()
        self.private_unread: set[str] = set()
        # Where each output name is defined, set upfront by rewrite_items; a
        # reference is FORWARD (refused, DuckDB's rule) only when its
        # definition is a strictly later item — a self-named bare ref falls
        # through to normal resolution.
        self.item_pos: dict[str, int] = {}
        self.item_i = 0
        # The fit-side view of this level's items: expanded (alias-free of
        # laterals) and private-filtered — what __CF_LEVEL_{i}__ projects.
        self.fit_items: list[Node] = []
        # Fit step name per unnest(tfm(...)) item, in output order — their
        # columns are the LEARNED field names, checked for collisions at fit.
        self.unnest_steps: list[str] = []
        self.is_final = False  # set by rewrite_items

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
        # The ref's own alias (a struct_pack field name, a named argument)
        # must survive resolution — the env expression carries alias="".
        al = x.get("alias", "")
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
                        **({"alias": al} if al else {}),
                    )
                _refuse(
                    "struct-field access through a projected expression",
                    x.get("query_location"),
                )
            resolved = copy.deepcopy(target)
            return dict(resolved, alias=al) if al else resolved
        if head in self.env.private:
            _refuse(
                f"private column {names[0]} is same-SELECT scope — it never"
                " crosses levels (make it public in its own level)",
                x.get("query_location"),
            )
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
        _walk_row_wise(expr, "a partition or order key", self.planner.lookup)
        stripped = _strip_quals(expr, quals)
        return _Key(
            ident=_stripped(stripped) + "|" + _alias_sig(stripped),
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
        agg_key = _stripped(raw) + "|" + _alias_sig(raw)
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
        return self._bundle_of(children[0])

    def _bundle_of(self, c: Node) -> list[tuple[str, Node, Node]]:
        """Parse ONE bundle node (a column or struct_pack(...)):
        (field name, fit expr in level terms, serving expr) per feature."""
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
                if (
                    ch.get("class") == "FUNCTION"
                    and ch.get("function_name", "").lower() == "struct_pack"
                ):
                    # Statically visible; anything else struct-typed is
                    # caught by name at fit, before est.fit sees it.
                    _refuse(
                        f"bundle field {ch['alias']} is a struct — a bundle"
                        " field must be a scalar",
                        ch.get("query_location"),
                    )
                _walk_row_wise(ch, "a transformer bundle", self.planner.lookup)
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
            if _is_struct_extract(x):
                kids = x.get("children") or []
                if (
                    len(kids) == 2
                    and isinstance(kids[0], dict)
                    and _is_struct_extract(kids[0])
                    and len(kids[0].get("children") or []) == 2
                    and (
                        _is_tf_node(kids[0]["children"][0]) is not None
                        or _bare_tf_name(kids[0]["children"][0], self.planner.lookup)
                        is not None
                        or _split_transform_name(
                            kids[0]["children"][0], self.planner.lookup
                        )
                        is not None
                    )
                ):
                    # `t(...).a.b` — a field is a scalar; the outer name
                    # could never be validated at fit and would only die
                    # mid-serving, differently per path.
                    _refuse(
                        "chained field access on a transformer call"
                        " (a field is a scalar)",
                        x.get("query_location"),
                    )
                split_tf = (
                    _split_transform_name(kids[0], self.planner.lookup)
                    if len(kids) == 2
                    else None
                )
                bare_tf = (
                    _bare_tf_name(kids[0], self.planner.lookup)
                    if len(kids) == 2
                    else None
                )
                win_name = _is_tf_node(kids[0]) if len(kids) == 2 else None
                if split_tf is not None or bare_tf is not None or win_name is not None:
                    fname = _const_str(kids[1])
                    if fname is None:
                        _refuse(
                            "a computed field name on a transformer call"
                            " (write the field literally)",
                            x.get("query_location"),
                        )
                    if win_name is not None:
                        # θ field reads and the deleted OVER sugar refuse
                        # the same way here as anywhere else.
                        self._refuse_window_tf(kids[0], win_name)
                    # The field read SURVIVES over the whole-value call
                    # (TASK-63): one call, k lane reads — DuckDB CSEs the
                    # identical mentions, confit shares the ecall site.
                    # Width-1 collapses to the bare call at fit, where the
                    # width is known.
                    if split_tf is not None:
                        call = self.planner.split_ref(
                            self, kids[0], split_tf, field_name=fname, alias=""
                        )
                    else:
                        call = self.planner.transformer_ref(
                            self,
                            kids[0],
                            bare_tf,
                            field_name=fname,
                            alias="",
                        )
                    out = copy.deepcopy(x)
                    out["children"] = [call, copy.deepcopy(kids[1])]
                    return out
            if cls == "WINDOW":
                tf_name = _is_tf_node(x)
                if tf_name is not None:
                    self._refuse_window_tf(x, tf_name)
                if _contains_tf(x, self.planner.lookup):
                    _refuse(
                        "a transformer call inside a window aggregate",
                        x.get("query_location"),
                    )
                w = _Window.of(x)
                discriminates = _check_window(w)
                self.planner.note_udfs(x)
                return self._params_ref(x, w, discriminates)
            if cls == "SUBQUERY":
                return self.planner.scalar_ref(_Subquery.of(x), x)
            if cls == "PARAMETER":
                _refuse("a prepared-statement parameter", x.get("query_location"))
            if cls == "FUNCTION":
                fname = x.get("function_name", "").lower()
                if fname in _aggregate_names():
                    _refuse(
                        f"aggregate {x['function_name']} without OVER"
                        " (that would aggregate the whole table, not project"
                        " rows)",
                        x.get("query_location"),
                    )
                if x.get("filter") is not None:
                    # Measured: DuckDB binds FILTER only on aggregates.
                    _refuse(
                        f"FILTER on the scalar call {x['function_name']}"
                        " (DuckDB binds FILTER only on aggregates — a fit"
                        " scope takes it on tf_fit(...) FILTER (...)"
                        " OVER (...))",
                        x.get("query_location"),
                    )
                if (x.get("order_bys") or {}).get("orders"):
                    # Same binder rule for in-call ORDER BY.
                    _refuse(
                        f"ORDER BY inside the scalar call {x['function_name']}"
                        " (DuckDB binds in-call ORDER BY only on aggregates)",
                        x.get("query_location"),
                    )
                if x.get("distinct"):
                    # Same binder rule again (review round 2026-08-05: the
                    # flag used to drop silently).
                    _refuse(
                        f"DISTINCT inside the scalar call {x['function_name']}"
                        " (DuckDB binds DISTINCT only on aggregates)",
                        x.get("query_location"),
                    )
                bare = _bare_tf_name(x, self.planner.lookup)
                if bare is not None:
                    _refuse(
                        f"a transformer call is a struct value inside an"
                        f" expression — serve it whole as its own output item"
                        f" ({bare}(...) AS out) or address an output field"
                        f" ({bare}(...).name)",
                        x.get("query_location"),
                    )
                split = _split_transform_name(x, self.planner.lookup)
                if split is not None:
                    _refuse(
                        f"a transformer call is a struct value inside an"
                        f" expression — serve it whole as its own output item"
                        f" ({split}_transform(...) AS out) or address an"
                        f" output field ({split}_transform(...).name)",
                        x.get("query_location"),
                    )
                fit_scalar = _fit_scalar_name(x, self.planner.lookup)
                if fit_scalar is not None:
                    _refuse(
                        f"{fit_scalar}_fit is a window aggregate — write"
                        f" {fit_scalar}_fit(bundle) OVER (...)",
                        x.get("query_location"),
                    )
                if fname not in _known_functions() and not x.get("is_operator"):
                    self.planner.scalar_udf(
                        fname, len(x.get("children", [])), x.get("query_location")
                    )
                if fname == "unnest" and _contains_tf(x, self.planner.lookup):
                    # Measured: DuckDB binds UNNEST over a struct only as a
                    # whole select item — anywhere else is a binder error.
                    _refuse(
                        "UNNEST() on a struct column can only be applied as"
                        " the root element of a SELECT expression",
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

    def _refuse_window_tf(self, x: Node, tf_name: str) -> None:
        """Every transformer-flavored WINDOW node refuses by name
        (2026-08-05 spec): a standalone fit (θ export is a later slice),
        then the identity refusals (namespaced/unknown/non-transformer),
        then — for a genuine registered transformer — the deleted sugar."""
        loc = x.get("query_location")
        if x.get("schema"):
            # The namespace is the actual mistake — name it before any
            # split-shape advice (review round 2026-08-05).
            _refuse(
                f"namespaced transformer {x['schema']}.{tf_name}"
                " (the curated-namespace index is a later loop)",
                loc,
            )
        tr_stem = _reserved_stem(tf_name, "_transform", self.planner.lookup)
        if tr_stem is not None:
            _refuse(
                f"{tr_stem}_transform is a scalar function — the fit scope"
                f" lives on {tr_stem}_fit(...) OVER (...)",
                loc,
            )
        fit_stem = _fit_node_name(x, self.planner.lookup)
        if fit_stem is not None:
            _refuse(
                f"{fit_stem}_fit(...) OVER (...) in a non-final level — a θ"
                " exported here would be θ-as-data one level up, which has"
                " no lawful provenance (export it from the final level, or"
                " park it in a private column AS _theta and consume it in"
                " the same SELECT)",
                loc,
            )
        self.planner.check_transformer_identity(x, tf_name)
        _refuse(
            f"the {tf_name}(...) OVER (...) sugar is deleted — the fit"
            f" scope moved to {tf_name}_fit: write"
            f" {tf_name}_transform({tf_name}_fit(bundle) OVER (...),"
            f" bundle).field, or {tf_name}(bundle).field for a global fit",
            loc,
        )

    def _expand(self, x: Any, in_window: bool = False) -> Any:
        """β-reduce same-level lateral aliases at the raw AST, before any
        rewrite: consumers receive closed level-term expressions, so windows,
        partition keys, and transformer bundles never mention an alias.
        Column-wins (DuckDB's rule) is preserved via the env check."""
        if isinstance(x, dict):
            cls = x.get("class")
            if cls == "LAMBDA":
                # A lambda binds its own parameters; substituting into it
                # would break the binder. Left alone, the rewrite refuses
                # lambdas by name (unknown column) — never a raw crash.
                return copy.deepcopy(x)
            if cls == "COLUMN_REF":
                names = x["column_names"]
                head = names[0].lower() if names else ""
                if (
                    head in self.raw_aliases
                    and head not in self.env.exprs
                    and not (len(names) > 1 and head in self.level.source_quals)
                ):
                    if len(names) > 1:
                        # Measured: DuckDB binds a dotted name as
                        # table.column and refuses ("Referenced table not
                        # found") — the FUNCTION spelling binds laterally.
                        _refuse(
                            f"{'.'.join(names)} reads lateral alias"
                            f" {names[0]} as a table — write"
                            f" struct_extract({names[0]}, {names[1]!r})",
                            x.get("query_location"),
                        )
                    self.private_unread.discard(head)
                    sub = copy.deepcopy(self.raw_aliases[head])
                    if x.get("alias"):
                        # The ref's own alias (an output name, a struct_pack
                        # field name) must survive the substitution.
                        sub = dict(sub, alias=x["alias"])
                    if in_window and _contains_class(sub, "WINDOW"):
                        _refuse(
                            f"window-valued alias {names[0]} inside a window"
                            " (DuckDB: window function calls cannot be"
                            " nested)",
                            x.get("query_location"),
                        )
                    if _contains_class(sub, "SUBQUERY"):
                        # Measured: DuckDB refuses referencing an alias whose
                        # expression has a subquery.
                        _refuse(
                            f"lateral alias {names[0]} holds a subquery"
                            " (DuckDB refuses referencing it — repeat the"
                            " expression)",
                            x.get("query_location"),
                        )
                    return sub
                if (
                    len(names) == 1
                    and self.item_pos.get(head, -1) > self.item_i
                    and head not in self.env.exprs
                ):
                    # DuckDB's own rule: select aliases bind left to right.
                    _refuse(
                        f"{names[0]} is referenced before it is defined",
                        x.get("query_location"),
                    )
                return copy.deepcopy(x)
            if cls == "SUBQUERY":
                # A subquery step runs verbatim where no same-level alias
                # exists (DuckDB binds these laterally; v0 refuses by name).
                self._refuse_lateral_in_subquery(x)
                return copy.deepcopy(x)
            inner = in_window or cls == "WINDOW"
            return {k: self._expand(v, inner) for k, v in x.items()}
        if isinstance(x, list):
            return [self._expand(v, in_window) for v in x]
        return x

    def validate_fit_expr(self, pred: Node) -> None:
        """A fit FILTER predicate or in-call ORDER BY key has no serving
        side, so nothing else resolves it. Rewrite-and-discard validates
        columns at construction and registers author UDFs; the fit-side text
        then binds laterally in the level table, which matches DuckDB for
        backward names — a FORWARD name is undecidable schema-free (DuckDB
        refuses the text), so it refuses with the risky-alias hint."""
        self.rewrite(pred)
        if not self.env.star:
            return  # explicit envs screened forward refs at item expansion

        def walk(x: Any) -> None:
            if isinstance(x, dict):
                if x.get("class") == "COLUMN_REF":
                    names = x["column_names"]
                    if (
                        len(names) == 1
                        and self.item_pos.get(names[0].lower(), -1) > self.item_i
                    ):
                        _refuse(
                            f"{names[0]} may be a lateral alias of a later"
                            " select item — qualify the column (e.g."
                            " __THIS__.{0}) or rename the alias".format(names[0]),
                            x.get("query_location"),
                        )
                for v in x.values():
                    walk(v)
            elif isinstance(x, list):
                for v in x:
                    walk(v)

        walk(pred)

    def _refuse_lateral_in_subquery(self, x: Any) -> None:
        if isinstance(x, dict):
            if x.get("class") == "COLUMN_REF":
                names = x["column_names"]
                head = names[0].lower() if names else ""
                if head not in self.env.exprs:
                    # Name-based, both directions (a forward alias would ride
                    # unresolved into the fit step) — conservatively also
                    # hits a subquery-internal binding of the same name.
                    if head in self.item_pos and self.item_pos[head] != self.item_i:
                        _refuse(
                            f"lateral alias {names[0]} inside a subquery (the"
                            " subquery step runs where the alias does not"
                            " exist)",
                            x.get("query_location"),
                        )
                    if head in self.env.private:
                        _refuse(
                            f"private column {names[0]} is same-SELECT scope"
                            " — it never crosses levels (make it public in"
                            " its own level)",
                            x.get("query_location"),
                        )
            for v in x.values():
                self._refuse_lateral_in_subquery(v)
        elif isinstance(x, list):
            for v in x:
                self._refuse_lateral_in_subquery(v)

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
        self.is_final = is_final
        out: list[Node] = []
        entries: list[tuple[str, Node | None]] = []
        names_seen: set[str] = set()
        explicit = not self.env.star
        self.item_pos = {
            n.lower(): i
            for i, it in enumerate(items)
            if (n := _item_name(it)) is not None
        }
        for self.item_i, raw_item in enumerate(items):
            name = _item_name(raw_item)
            lname = name.lower() if name is not None else None
            private = lname is not None and lname.startswith("_")
            item = self._expand(raw_item) if explicit else raw_item
            if private and not explicit:
                _refuse(
                    f"private column {name} without a declared schema (lateral"
                    " resolution is undecidable against unknown table columns"
                    " — declare one via this_model)",
                    raw_item.get("query_location"),
                )
            if private:
                # A private column is a same-SELECT macro: consumers already
                # β-reduced it above; it is never rewritten as an item and
                # never becomes an output or a next-level entry.
                if lname in self.private_names:
                    _refuse(f"duplicate private column {name}")
                self.raw_aliases[lname] = dict(copy.deepcopy(item), alias="")
                self.private_names.add(lname)
                self.private_unread.add(lname)
                continue
            self.fit_items.append(copy.deepcopy(item))
            if item.get("class") == "STAR":
                star = _Star.of(item)
                for entry in star.rename_list:
                    renamed = entry["value"]
                    if renamed.lower().startswith("_"):
                        _refuse(
                            f"a star modifier renames"
                            f" {entry['key']['column']} to the private name"
                            f" {renamed}",
                            star.query_location,
                        )
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
            # A bare or split transformer call as a WHOLE item serves its
            # output struct (slice 5) — the whole-value call crosses the
            # boundary as a struct column on both paths. An unaliased item
            # already carries DuckDB's derived name (the parse step stamps
            # it as the alias), so the oracle's column name survives.
            # unnest(tfm(...)): the oracle expands the struct into one column
            # per field, in place, IGNORING any alias (measured). The names
            # are learned at fit, so the item carries no construction-time
            # output name and its collision check waits for fit.
            if (unnested := _unnest_child(item, self.planner.lookup)) is not None:
                if not is_final:
                    _refuse(
                        "unnest of a transformer call in a non-final level",
                        item.get("query_location"),
                    )
                tf_bare = _bare_tf_name(unnested, self.planner.lookup)
                tf_split = (
                    None
                    if tf_bare is not None
                    else _split_transform_name(unnested, self.planner.lookup)
                )
                ref = (
                    self.planner.transformer_ref
                    if tf_bare is not None
                    else (self.planner.split_ref)
                )
                call = ref(
                    self, unnested, tf_bare or tf_split, field_name=None, alias=""
                )
                j = int(call["function_name"].removeprefix("__cf_tf"))
                self.unnest_steps.append(self.planner.tf_steps[j]["name"])
                out.append(
                    _clone(
                        "function", function_name="unnest", children=[call], alias=""
                    )
                )
                entries.append(("*", None))
                continue
            # A public fit call IS θ export (slice 6): the handle serves as
            # an ordinary struct value. Private θ laterals never reach here
            # (they β-reduce above), and every other fit position still
            # refuses through _refuse_window_tf.
            theta_stem = _fit_node_name(item, self.planner.lookup)
            if theta_stem is not None and is_final:
                self.planner.check_transformer_identity(item, theta_stem)
                oname = name or "theta"
                # An exported handle is an output column like any other: it
                # obeys the duplicate-name law, and in schema-free mode its
                # alias is as undecidable as any other (review round — the
                # branch used to skip both). It is deliberately NOT put in
                # raw_aliases: export is readable, not consumable.
                taken = names_seen | {
                    n.lower() for n, _ in entries if n not in ("*", "")
                }
                if oname.lower() in taken:
                    _refuse(f"duplicate output name {oname}")
                if not explicit:
                    self.risky_aliases.add(oname.lower())
                rewritten = self.planner.theta_ref(self, item, theta_stem)
                out.append(dict(rewritten, alias=oname))
                entries.append((oname, None))
                names_seen.add(oname.lower())
                continue
            whole_bare = _bare_tf_name(item, self.planner.lookup)
            whole_split = (
                None
                if whole_bare is not None
                else _split_transform_name(item, self.planner.lookup)
            )
            if whole_bare is not None or whole_split is not None:
                tf = whole_bare or whole_split
                if whole_bare is not None:
                    rewritten = self.planner.transformer_ref(
                        self, item, tf, field_name=None, alias=item.get("alias", "")
                    )
                else:
                    rewritten = self.planner.split_ref(
                        self, item, tf, field_name=None, alias=item.get("alias", "")
                    )
            else:
                rewritten = self.rewrite(item)
            if item.get("alias"):
                # Substitution may return an env expression whose alias field
                # is empty; the item's output name must survive it.
                rewritten = dict(rewritten, alias=item["alias"])
            out.append(rewritten)
            if explicit:
                if name is not None:
                    self.raw_aliases[lname] = dict(copy.deepcopy(item), alias="")
            elif item.get("alias") and not (
                item.get("class") == "COLUMN_REF"
                and _ColumnRef.of(item).column_names[-1].lower()
                == item["alias"].lower()
            ):
                self.risky_aliases.add(item["alias"].lower())
            if name is not None:
                # Refused at EVERY level (review round): DuckDB's duplicate
                # rules (last-wins after, refuse between) cannot be honored —
                # the fit-side level table cannot carry two same-named columns.
                if name.lower() in names_seen:
                    _refuse(f"duplicate output name {name}")
                names_seen.add(name.lower())
            if not is_final:
                if name is None:
                    _refuse("an unnameable select item in a non-final level")
                entries.append((name, dict(copy.deepcopy(rewritten), alias="")))
            else:
                entries.append((name or "", None))
        if not out:
            _refuse(
                "every output column is private (nothing would cross the"
                " output boundary)"
            )
        if self.private_unread:
            first = sorted(self.private_unread)[0]
            _refuse(
                f"private column {first} is never read (a private column is a"
                " same-SELECT intermediate — read it or drop it)"
            )
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
        self.lookup: Any = None  # UDF/transformer resolver: name -> obj | None
        self.tf_steps: list[dict] = []
        self.tf_calls: dict[str, int] = {}  # structural identity -> step index
        self.udf_specs: list[UDFSpec] = []
        self.fit_joins: set[int] = set()  # join idxs whose params come from fit
        self.scalar_udfs: dict[str, int] = {}  # author UDF name -> arity

    def scalar_udf(self, name: str, nargs: int, loc: int | None) -> None:
        """A scalar call to a function the oracle doesn't know: resolve it as
        a declared UDF or refuse by name. Runs at every call site, so arity
        is checked everywhere the name appears."""
        obj = self.lookup(name) if self.lookup is not None else None
        if obj is None:
            _refuse(
                f"unknown function {name} — not a DuckDB function, and no UDF"
                " of that name in the registry or caller scope",
                loc,
            )
        if hasattr(obj, "fit") and hasattr(obj, "transform"):
            # Normally caught earlier in rewrite; defensive for other paths.
            _refuse(
                f"a transformer call is a struct value inside an expression"
                f" — serve it whole as its own output item ({name}(...) AS"
                f" out) or address an output field ({name}(...).name)",
                loc,
            )
        takes = getattr(obj, "takes", None)
        returns = getattr(obj, "returns", None)
        if takes is None or returns is None or not callable(obj):
            _refuse(
                f"{name} is not a UDF (declare takes/returns — e.g. PythonUDF)",
                loc,
            )
        declared = getattr(obj, "name", name)
        if declared.lower() != name:
            _refuse(f"UDF {name} resolves to an object named {declared!r}", loc)
        if len(takes) != nargs:
            _refuse(
                f"UDF {name} declares {len(takes)} arguments, called with {nargs}",
                loc,
            )
        if len(returns) != 1:
            _refuse(f"scalar UDF {name} must declare exactly one return", loc)
        self.scalar_udfs[name] = nargs

    def note_udfs(self, x: Any) -> None:
        """Scan a subtree that survives verbatim into fit-side SQL (window
        arguments and their clauses) for unknown scalar calls, so every UDF
        is resolved or refused at construction, never at fit."""
        if isinstance(x, dict):
            if x.get("class") == "FUNCTION" and not x.get("is_operator"):
                f = x.get("function_name", "").lower()
                if f and f not in _known_functions():
                    self.scalar_udf(
                        f, len(x.get("children", [])), x.get("query_location")
                    )
            for v in x.values():
                self.note_udfs(v)
        elif isinstance(x, list):
            for v in x:
                self.note_udfs(v)

    def transformer_ref(
        self,
        rw: _LevelRewriter,
        raw: Node,
        name: str,
        field_name: str | None = None,
        alias: str | None = None,
    ) -> Node:
        """One BARE transformer call — the sugar ``tfm(bundle).field``
        (global fit-transform) — rewritten in place to a scalar call over
        its own params join: ``__cf_tf{j}(__cf_p{idx}.__cf_est, features...)``.
        The join's params table (keys + ``__cf_est``) is produced by the
        ``kind="fit"`` plan step; an unseen group misses the LEFT JOIN, so
        the id is NULL and the call returns NULL — the one NULL story.

        Structurally identical calls share one fit step, so ``t(...).a``
        and ``t(...).b`` fit once. Windowed fit scopes take the split path
        (``split_ref``)."""
        if raw.get("filter") is not None:
            _refuse(
                f"FILTER on the {name}(...) sugar ({name} is a scalar"
                f" fit-transform — leakage control lives on {name}_fit(...)"
                " FILTER (...) OVER (...))",
                raw.get("query_location"),
            )
        if (raw.get("order_bys") or {}).get("orders"):
            _refuse(
                f"ORDER BY inside the {name}(...) sugar — the order belongs"
                f" on {name}_fit(bundle ORDER BY key) OVER (...)",
                raw.get("query_location"),
            )
        if raw.get("distinct"):
            _refuse(
                f"DISTINCT inside the {name}(...) sugar (DuckDB binds"
                " DISTINCT only on aggregates)",
                raw.get("query_location"),
            )
        if getattr(self.lookup(name), "order_sensitive", False):
            _refuse(
                f"{name} is order-sensitive — the bare sugar cannot name an"
                f" order; write {name}_transform({name}_fit(bundle ORDER BY"
                " key) OVER (), bundle)",
                raw.get("query_location"),
            )
        ident = "|".join([_stripped(raw), _bundle_names_key(raw), _alias_sig(raw)])
        if ident in self.tf_calls:
            return self._tf_call_node(self.tf_calls[ident], field_name, alias)
        self.check_transformer_identity(raw, name)
        if not rw.is_final:
            _refuse(
                "a transformer call in a non-final level", raw.get("query_location")
            )
        # The bare sugar: tfm(x) ≡ tfm_transform(tfm_fit(x) OVER (), x)
        # — a global fit scope, so no keys. (Windowed spellings never reach
        # here: every transformer-flavored WINDOW node refuses in rewrite.)
        keys: list[_Key] = []
        bundle = rw._tf_bundle(raw)
        j = self._mint_step(rw, name, keys, bundle)
        self.tf_calls[ident] = j
        return self._tf_call_node(j, field_name, alias or raw.get("alias", ""))

    def _mint_step(
        self,
        rw: _LevelRewriter,
        name: str,
        keys: list[_Key],
        bundle: list[tuple[str, Node, Node]],
        filter_fit: Node | None = None,
        order_fit: list[tuple[Node, bool, bool, str]] | None = None,
    ) -> int:
        """One fit step + its dedicated params join, never shared with
        aggregate groups. (Sharing keys with an aggregate join is a dedup
        optimization for a later loop.)"""
        j = len(self.tf_steps)
        idx = self.new_join(rw.level_i, keys)
        self.fit_joins.add(idx)
        feature_fit = []
        for fname, fit_expr, _serving in bundle:
            col = f"__cf_f{self.feature_count}"
            self.feature_count += 1
            feature_fit.append((fname, col, fit_expr))
        self.tf_steps.append(
            {
                "level": rw.level_i,
                "name": f"__CF_PARAMS_{idx}__",
                "transformer": name,
                "feature_fit": feature_fit,
                "keys": keys,
                "join": idx,
                "filter_fit": filter_fit,
                "filter_col": f"__cf_flt{j}" if filter_fit is not None else "",
                # (column, key expr in level terms, asc, nulls_first, collation)
                "order_fit": [
                    (f"__cf_ord{j}_{n}", expr, asc, nf, coll)
                    for n, (expr, asc, nf, coll) in enumerate(order_fit or [])
                ],
                # The serving-side argument nodes, kept so a second field
                # access on the same call rebuilds the call without refitting.
                "children": [
                    _clone(
                        "column_ref",
                        column_names=[f"__cf_p{idx}", "__cf_est"],
                        alias="",
                    ),
                    *[copy.deepcopy(se) for (_, _, se) in bundle],
                ],
            }
        )
        return j

    def split_ref(
        self,
        rw: _LevelRewriter,
        raw: Node,
        tf_name: str,
        field_name: str | None = None,
        alias: str | None = None,
    ) -> Node:
        """A ``{tf}_transform(θ, bundle)`` call (2026-08-05 spec): θ must be
        an inline ``{tf}_fit(...) OVER (...)`` call — the only lawful
        provenance in this slice. The fit half mints the step (fit scope =
        its partitions, fit features = its bundle); the transform half
        supplies the serving arguments, name-keyed against the fit bundle."""
        loc = raw.get("query_location")
        if raw.get("schema"):
            _refuse(
                f"namespaced transformer {raw['schema']}.{tf_name}_transform"
                " (the curated-namespace index is a later loop)",
                loc,
            )
        if raw.get("filter") is not None:
            _refuse(
                f"FILTER on {tf_name}_transform ({tf_name}_transform is a"
                f" scalar — leakage control lives on {tf_name}_fit(...)"
                " FILTER (...) OVER (...))",
                loc,
            )
        if (raw.get("order_bys") or {}).get("orders"):
            _refuse(
                f"ORDER BY on {tf_name}_transform ({tf_name}_transform is a"
                f" scalar — the order lives on {tf_name}_fit(bundle ORDER BY"
                " key) OVER (...))",
                loc,
            )
        if raw.get("distinct"):
            _refuse(
                f"DISTINCT on {tf_name}_transform (DuckDB binds DISTINCT"
                " only on aggregates)",
                loc,
            )
        kids = raw.get("children") or []
        if len(kids) != 2:
            _refuse(
                f"{tf_name}_transform takes exactly two arguments"
                f" ({tf_name}_fit(...) OVER (...), bundle)",
                loc,
            )
        theta = kids[0]
        if isinstance(theta, dict) and theta.get("schema") and _is_tf_node(theta):
            _refuse(
                f"namespaced transformer {theta['schema']}"
                f".{theta.get('function_name', '')}"
                " (the curated-namespace index is a later loop)",
                theta.get("query_location"),
            )
        if _fit_node_name(theta, self.lookup) != tf_name:
            _refuse(
                f"the first argument of {tf_name}_transform must be an inline"
                f" {tf_name}_fit(...) OVER (...) call — θ has no other lawful"
                " provenance (an EXPORTED handle is readable, not"
                f" consumable; to apply one, park it privately: {tf_name}"
                f"_fit(bundle) OVER (...) AS _th, {tf_name}_transform(_th,"
                " bundle))",
                loc,
            )
        if not rw.is_final:
            _refuse("a transformer call in a non-final level", loc)
        j = self.fit_half(rw, theta, tf_name)
        tbundle = rw._bundle_of(kids[1])
        step = self.tf_steps[j]
        fit_names = [f.lower() for (f, _, _) in step["feature_fit"]]
        got_names = [f.lower() for (f, _, _) in tbundle]
        if got_names != fit_names:
            _refuse(
                f"{tf_name}_transform bundle fields {got_names} do not match"
                f" the fit bundle fields {fit_names} (name-keyed, in order)",
                loc,
            )
        children = [
            _clone(
                "column_ref",
                column_names=[f"__cf_p{step['join']}", "__cf_est"],
                alias="",
            ),
            *[copy.deepcopy(se) for (_, _, se) in tbundle],
        ]
        return self._tf_call_node(
            j, field_name, alias or raw.get("alias", ""), children=children
        )

    def theta_ref(self, rw: _LevelRewriter, theta: Node, tf_name: str) -> Node:
        """A PUBLIC ``{tf}_fit(bundle) OVER (...)`` item — θ export (slice
        6). θ IS the wire mechanism, so the handle is built from existing
        parts: ``struct_pack(type := '__cf_tf{j}', id := __cf_p{idx}
        .__cf_est)`` — the same value for every row of a fit scope, NULL id
        for an unseen group. Same fit half as the split spelling, so an
        exported handle and a consumed one share one step."""
        if not rw.is_final:
            _refuse(
                "a transformer call in a non-final level",
                theta.get("query_location"),
            )
        j = self.fit_half(rw, theta, tf_name)
        step = self.tf_steps[j]
        # The whole handle is NULL when the group was never fitted: the id
        # comes from the params LEFT JOIN, so P14's NULL story decides the
        # struct, not a half-built handle with a live type tag.
        fn_name = f"__cf_tf{j}"
        if not any(s.name == fn_name for s in self.udf_specs):
            # The step must exist (it fits the instances the handle names),
            # but nothing serves the transform's own struct here.
            self.udf_specs.append(
                UDFSpec(
                    name=fn_name,
                    step=step["name"],
                    transformer=step["transformer"],
                )
            )
        est = _clone(
            "column_ref", column_names=[f"__cf_p{step['join']}", "__cf_est"], alias=""
        )
        node = _clone("theta")
        node["case_checks"][0]["when_expr"]["children"] = [copy.deepcopy(est)]
        pack = node["else_expr"]
        pack["children"][0]["value"]["value"] = f"__cf_tf{j}"
        pack["children"][1] = dict(copy.deepcopy(est), alias="id")
        return node

    def fit_half(self, rw: _LevelRewriter, theta: Node, tf_name: str) -> int:
        """The fit half of a θ node — window shape, in-call ORDER BY,
        FILTER, frame/DISTINCT refusals, then the deduped fit step. Shared
        by the split spelling and θ export so the two cannot diverge."""
        w = _Window.of(theta)
        if w.orders:
            _refuse(
                f"ORDER BY in the window clause of {tf_name}_fit means a"
                " running fit (a θ per row-prefix) — not supported",
                w.query_location,
            )
        # In-call ORDER BY — DuckDB's ordered-aggregate spelling. Each key
        # resolves like any fit-side expression; direction and null placement
        # default to DuckDB's (ASC, NULLS LAST — measured 2026-08-05).
        order_fit: list[tuple[Node, bool, bool, str]] = []
        for o in w.arg_orders:
            key = o["expression"]
            collation = ""
            if isinstance(key, dict) and key.get("class") == "COLLATE":
                # The collation is a comparison annotation — it cannot ride
                # the level table (Arrow strips it), so it is carried by name
                # and re-emitted in the fit-side sort. Validated against the
                # oracle here, never mid-fit.
                collation = key.get("collation", "")
                try:
                    duckdb.execute(f"SELECT '' COLLATE {collation}")  # noqa: S608
                except duckdb.Error:
                    _refuse(
                        f"unknown collation {collation} in an ORDER BY key",
                        key.get("query_location"),
                    )
                key = key["child"]
            if _contains_class(key, "COLLATE"):
                _refuse(
                    "a COLLATE inside an order-key expression (put it at the"
                    " top level of the key)",
                    w.query_location,
                )
            if key.get("class") == "CONSTANT":
                ty = ((key.get("value") or {}).get("type") or {}).get("id", "")
                if "INT" not in str(ty).upper():
                    # DuckDB's binder rule, mirrored.
                    _refuse(
                        "ORDER BY non-integer literal has no effect"
                        " (DuckDB's binder refuses it)",
                        key.get("query_location"),
                    )
            _walk_row_wise(key, "an ORDER BY key", self.lookup)
            rw.validate_fit_expr(key)
            order_fit.append(
                (
                    _strip_quals(key, rw.level.source_quals),
                    o.get("type") != "DESCENDING",
                    o.get("null_order") == "NULLS FIRST",
                    collation,
                )
            )
        if getattr(self.lookup(tf_name), "order_sensitive", False) and not order_fit:
            _refuse(
                f"{tf_name} is order-sensitive — name the order in the call:"
                f" {tf_name}_fit(bundle ORDER BY key) OVER (...)",
                w.query_location,
            )
        filter_fit = None
        if w.filter_expr is not None:
            _walk_row_wise(w.filter_expr, "a FILTER", self.lookup)
            rw.validate_fit_expr(w.filter_expr)
            filter_fit = _strip_quals(w.filter_expr, rw.level.source_quals)
        if (
            w.start != "UNBOUNDED_PRECEDING"
            or w.end != "CURRENT_ROW_RANGE"
            or w.start_expr is not None
            or w.end_expr is not None
        ):
            _refuse(f"a frame on {tf_name}_fit", w.query_location)
        if w.distinct or w.ignore_nulls:
            _refuse(f"DISTINCT/IGNORE NULLS on {tf_name}_fit", w.query_location)
        ident = "|".join(
            [_stripped(theta), _bundle_names_key(theta), _alias_sig(theta)]
        )
        if ident in self.tf_calls:
            return self.tf_calls[ident]
        keys = [rw._key_of(p) for p in w.partitions]
        seen: set[str] = set()
        keys = [k for k in keys if not (k.ident in seen or seen.add(k.ident))]
        j = self._mint_step(
            rw,
            tf_name,
            keys,
            rw._tf_bundle(theta),
            filter_fit=filter_fit,
            order_fit=order_fit,
        )
        self.tf_calls[ident] = j
        return j

    def check_transformer_identity(self, raw: Node, name: str) -> None:
        """The identity refusals shared by field-addressed and bare calls —
        a namespaced name, an unknown name, or a non-transformer object
        refuses the same way wherever the call appears."""
        if raw.get("schema"):
            _refuse(
                f"namespaced transformer {raw['schema']}.{name}"
                " (the curated-namespace index is a later loop)",
                raw.get("query_location"),
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

    def _tf_call_node(
        self,
        j: int,
        field_name: str | None,
        alias: str | None,
        children: list[Node] | None = None,
    ) -> Node:
        """The serving call for transformer step ``j`` — always the ONE
        whole-value UDF (TASK-63). A requested field is recorded on its spec
        for fit-time validation; the field read wraps this call at the call
        site. ``children`` overrides the argument nodes (a split call's
        transform bundle may differ from the fit bundle)."""
        step = self.tf_steps[j]
        fn_name = f"__cf_tf{j}"
        for i, s in enumerate(self.udf_specs):
            if s.name == fn_name and s.field == field_name:
                # A θ export may have registered this step already; a real
                # whole-value call upgrades it (its struct now IS served).
                if field_name is None and not s.whole:
                    self.udf_specs[i] = replace(s, whole=True)
                break
        else:
            self.udf_specs.append(
                UDFSpec(
                    whole=field_name is None,
                    name=fn_name,
                    step=step["name"],
                    transformer=step["transformer"],
                    field=field_name,
                )
            )
        return _clone(
            "function",
            function_name=fn_name,
            children=[
                copy.deepcopy(c)
                for c in (children if children is not None else step["children"])
            ],
            alias=alias or "",
        )

    def new_join(self, level_i: int, keys: list[_Key]) -> int:
        idx = len(self.joins)
        self.joins.append(_Join(keys=list(keys), level=level_i))
        for k in keys:
            kid = (level_i, k.ident)
            if kid not in self.key_names:
                self.key_names[kid] = f"__cf_k{self.key_count}"
                self.key_count += 1
        return idx

    def group_for(self, level_i: int, keys: list[_Key]) -> tuple[int, _Join]:
        ident = (level_i, tuple(k.ident for k in keys))
        if ident not in self.groups:
            self.groups[ident] = self.new_join(level_i, keys)
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
        ident = _stripped(raw) + "|" + _alias_sig(raw)
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
            if x.get("class") == "WINDOW":
                f = x.get("function_name", "").lower()
                if f and f not in _aggregate_names():
                    # Pre-existing hole closed in the 2026-08-05 review
                    # round: this used to ride verbatim into the fit step
                    # and die there as a raw CatalogException.
                    _refuse(
                        f"unknown window function {f} inside a subquery",
                        x.get("query_location"),
                    )
            if x.get("type") == "BASE_TABLE":
                name = x.get("table_name", "")
                if name.lower() != "__this__" and name.lower() not in own_ctes:
                    _refuse(
                        f"table {name} inside a subquery (only __THIS__ and"
                        " the subquery's own CTEs)",
                        x.get("query_location"),
                    )
            if x.get("class") == "FUNCTION" and not x.get("is_operator"):
                f = x.get("function_name", "").lower()
                if f and f not in _known_functions():
                    _refuse(
                        f"unknown function {f} inside a subquery (UDFs in"
                        " subqueries are a later loop)",
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


def _const_str(node: Node) -> str | None:
    """The literal VARCHAR a constant node carries, else None."""
    if node.get("class") != "CONSTANT":
        return None
    value = node.get("value") or {}
    if (value.get("type") or {}).get("id") != "VARCHAR" or value.get("is_null"):
        return None
    got = value.get("value")
    return got if isinstance(got, str) else None


def _unnest_child(item: Node, lookup: Any) -> Node | None:
    """The transformer call inside a root ``unnest(tfm(...))`` select item,
    else None. Nested unnest refuses here — measured: DuckDB's binder does
    the same ("Nested UNNEST calls are not supported")."""
    if item.get("class") != "FUNCTION" or item.get("is_operator"):
        return None
    if item.get("function_name", "").lower() != "unnest" or item.get("schema"):
        return None
    # The branch below rebuilds the item, so every modifier the oracle
    # rejects on UNNEST itself must be screened HERE — measured:
    # '"DISTINCT", "FILTER", and "ORDER BY" are not applicable to "UNNEST"'
    # (review round: they used to drop silently).
    for key, what in (
        ("distinct", "DISTINCT"),
        ("filter", "FILTER"),
        ("order_bys", "ORDER BY"),
    ):
        got = item.get(key)
        if key == "order_bys":
            got = (got or {}).get("orders")
        if got and _contains_tf(item, lookup):
            _refuse(
                f'{what} is not applicable to "UNNEST"',
                item.get("query_location"),
            )
    kids = item.get("children") or []
    if len(kids) != 1:
        # recursive := / max_depth := are no-ops over a depth-1 struct;
        # refusing by name beats guessing at deeper shapes.
        if any(_contains_tf(k, lookup) for k in kids):
            _refuse(
                "unnest of a transformer call with extra arguments",
                item.get("query_location"),
            )
        return None
    child = kids[0]
    if (
        isinstance(child, dict)
        and child.get("class") == "FUNCTION"
        and child.get("function_name", "").lower() == "unnest"
        and _contains_tf(child, lookup)
    ):
        _refuse("Nested UNNEST calls are not supported", item.get("query_location"))
    if (
        _bare_tf_name(child, lookup) is not None
        or _split_transform_name(child, lookup) is not None
    ):
        return child
    return None


def _bare_tf_name(node: Node, lookup: Any) -> str | None:
    """The lowered name when the node is a bare transformer call — the one
    sugar (global fit-transform, 2026-08-05 spec): a scalar FUNCTION the
    oracle doesn't know whose name resolves to a fit/transform object."""
    if node.get("class") != "FUNCTION" or node.get("is_operator"):
        return None
    fname = node.get("function_name", "").lower()
    if not fname or fname in _known_functions():
        return None
    obj = lookup(fname) if lookup is not None else None
    if obj is not None and hasattr(obj, "fit") and hasattr(obj, "transform"):
        # A bare call spelled like a reserved name is ambiguous — refuse
        # (review round 2026-08-05: x_fit on a {x, x_fit} registry used to
        # silently serve the x_fit object).
        for suffix in ("_fit", "_transform"):
            _reserved_stem(fname, suffix, lookup)
        return fname
    return None


def _bundle_names_key(call: Node) -> str:
    """The lowered bundle field names of a transformer/fit call, for fit-step
    identity. ``_stripped`` erases aliases — but S's field names ARE the type
    (P16a), so two fits differing only in struct_pack names must never share
    a step (review round 2026-08-05: the collision served the wrong fit's
    lanes)."""
    kids = call.get("children") or []
    if len(kids) != 1 or not isinstance(kids[0], dict):
        return ""
    c = kids[0]
    if c.get("class") == "COLUMN_REF":
        names = c.get("column_names") or [""]
        return str(names[-1]).lower()
    if (
        c.get("class") == "FUNCTION"
        and c.get("function_name", "").lower() == "struct_pack"
    ):
        return ",".join(
            str(ch.get("alias") or "").lower() for ch in c.get("children") or []
        )
    return ""


def _reserved_stem(fname: str, suffix: str, lookup: Any) -> str | None:
    """The transformer stem when ``fname`` is ``{stem}{suffix}`` for a
    registered transformer — the split's reserved names. A registry entry
    under the full reserved name is ambiguous and refuses (never silently
    shadow either reading)."""
    if not fname.endswith(suffix):
        return None
    stem = fname[: -len(suffix)]
    if not stem or lookup is None:
        return None
    obj = lookup(stem)
    if obj is None or not (hasattr(obj, "fit") and hasattr(obj, "transform")):
        return None
    if lookup(fname) is not None:
        _refuse(
            f"transformer {stem} reserves the name {fname} — rename the"
            f" {fname} registry entry"
        )
    return stem


def _split_transform_name(node: Node, lookup: Any) -> str | None:
    """The transformer stem when the node is a ``{tf}_transform(θ, bundle)``
    scalar call (the split's application half, 2026-08-05 spec)."""
    if node.get("class") != "FUNCTION" or node.get("is_operator"):
        return None
    fname = node.get("function_name", "").lower()
    if fname in _known_functions():
        return None
    return _reserved_stem(fname, "_transform", lookup)


def _fit_node_name(node: Node, lookup: Any) -> str | None:
    """The transformer stem when the node is a ``{tf}_fit(...) OVER (...)``
    window call (the split's aggregate half)."""
    name = _is_tf_node(node)
    if name is None:
        return None
    return _reserved_stem(name, "_fit", lookup)


def _fit_scalar_name(node: Node, lookup: Any) -> str | None:
    """The transformer stem when the node is a ``{tf}_fit`` call WITHOUT
    OVER — a fit is a window aggregate, so this always refuses upstream."""
    if node.get("class") != "FUNCTION" or node.get("is_operator"):
        return None
    fname = node.get("function_name", "").lower()
    if fname in _known_functions():
        return None
    return _reserved_stem(fname, "_fit", lookup)


def _contains_tf(x: Any, lookup: Any = None) -> bool:
    """True when a transformer call (window or bare sugar) appears anywhere
    in the subtree (such an expression is not executable SQL on the fit
    side)."""
    if isinstance(x, dict):
        if _is_tf_node(x) is not None:
            return True
        if _bare_tf_name(x, lookup) is not None:
            return True
        if _split_transform_name(x, lookup) is not None:
            return True
        return any(_contains_tf(v, lookup) for v in x.values())
    if isinstance(x, list):
        return any(_contains_tf(v, lookup) for v in x)
    return False


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
    planner: _Planner,
    level_i: int,
    level: _Level,
    source_name: str,
    fit_items: list[Node],
) -> FitStep:
    """``__CF_LEVEL_{i}__``: the level's items — laterals expanded, privates
    dropped — under their original output names (the next level's verbatim
    input) plus its windows and key expressions, reading from the previous
    level's table."""
    quals = level.source_quals
    originals = []
    for item in fit_items:
        if _contains_tf(item, planner.lookup):
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
    filters = [
        _clone(
            "bool_cast",
            child=copy.deepcopy(t["filter_fit"]),
            alias=t["filter_col"],
        )
        for t in planner.tf_steps
        if t["level"] == level_i and t["filter_col"]
    ]
    orders = [
        dict(copy.deepcopy(expr), alias=col)
        for t in planner.tf_steps
        if t["level"] == level_i
        for (col, expr, _asc, _nf, _coll) in t["order_fit"]
    ]
    from_table = _clone("base_table", table_name=source_name, alias="")
    return FitStep(
        name=f"__CF_LEVEL_{level_i}__",
        sql=_select_doc(
            [*originals, *windows, *keys, *features, *filters, *orders], from_table
        ),
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

    planner = _Planner()
    planner.lookup = transformers.get if hasattr(transformers, "get") else transformers

    windows_at = [
        i
        for i, lvl in enumerate(levels)
        if _has_windows(lvl.node) or _contains_tf(lvl.node, planner.lookup)
    ]
    deepest = max(windows_at, default=-1)
    env = base_env
    final_items: list[Node] = []
    unnest_items: tuple[str, ...] = ()
    unnest_siblings: tuple[str, ...] = ()
    for i, level in enumerate(levels):
        is_final = i == len(levels) - 1
        rewriter = _LevelRewriter(planner, i, env, level)
        rewritten, entries = rewriter.rewrite_items(level.node["select_list"], is_final)
        if i <= deepest:
            source = "__THIS__" if i == 0 else f"__CF_LEVEL_{i - 1}__"
            planner.plan.append(
                _level_step(planner, i, level, source, rewriter.fit_items)
            )
            for idx, join in enumerate(planner.joins):
                if join.level == i and idx not in planner.fit_joins:
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
                            feature_names=tuple(f for (f, _, _) in t["feature_fit"]),
                            keys=tuple(
                                planner.key_names[(i, k.ident)] for k in t["keys"]
                            ),
                            filter_col=t["filter_col"],
                            order_by=tuple(
                                (col, asc, nf, coll)
                                for (col, _e, asc, nf, coll) in t["order_fit"]
                            ),
                        )
                    )
        if is_final:
            final_items = rewritten
            unnest_items = tuple(rewriter.unnest_steps)
            unnest_siblings = tuple(n for n, _ in entries if n not in ("*", ""))
            break
        env = _Env(entries, private=frozenset(rewriter.private_names))

    if (
        not planner.joins
        and not planner.tf_steps
        and len(levels) == 1
        and base_env is _BASE_ENV
    ):
        # No aggregates, no chain, no schema: marginalization is the identity
        # (modulo normalization). With a declared schema the rewrite always
        # canonicalizes — stars/COLUMNS expand, lateral aliases inline.
        return Marginalized(
            serving_sql=_deserialize(doc),
            plan=(),
            params=(),
            scalar_udfs=tuple(planner.scalar_udfs),
        )

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
        udfs=tuple(planner.udf_specs),
        scalar_udfs=tuple(planner.scalar_udfs),
        unnest_items=unnest_items,
        unnest_siblings=unnest_siblings,
    )


def _is_struct_extract(x: Node) -> bool:
    """Either spelling of a struct field read: the OPERATOR node the dot
    form parses to, or the ``struct_extract(...)`` FUNCTION."""
    return (x.get("class") == "OPERATOR" and x.get("type") == "STRUCT_EXTRACT") or (
        x.get("class") == "FUNCTION"
        and x.get("function_name", "").lower() == "struct_extract"
    )
