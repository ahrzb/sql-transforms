"""A typed model of what ``json_serialize_sql`` returns.

Typed where the walk interprets; carried verbatim everywhere else. A class
exists for a tag when something branches on its contents — eleven of the 53
tags the corpus reaches. The rest are data we move without reading, and a class
per tag would buy nothing that carrying the dict does not already give.

Where a class does exist it carries **every** field DuckDB emits for that tag,
including the ones nothing reads. That is not tidiness. Measured on 1.5.5,
``json_deserialize_sql`` requires exactly one field:

    dropping BASE_TABLE.table_name  ->  accepted, and prints DIFFERENT SQL
    dropping BASE_TABLE.type        ->  rejected

Everything but ``type`` silently defaults, so a model that forgets a field does
not fail — it emits another query. ``_nodes_test.py`` holds both halves down:
that every emitted field is carried, and that the tree is lossless.

The shapes here are internal DuckDB details with no stability promise.
``_shapes.json`` pins them per version and the drift gate reads the diff.
"""

from collections.abc import Callable, Iterator
from typing import Annotated, Any, ClassVar, Literal, Union

from pydantic import BaseModel, ConfigDict, Discriminator, Field, Tag

# --------------------------------------------------------------------- base


class AstNode(BaseModel):
    """One node of the oracle's serialization.

    ``extra="forbid"`` is the second drift gate: a field DuckDB adds fails
    here, by name, at parse time — the first being the pinned manifest, which
    also sees tags nothing reads.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    # The discriminator pair. `kind` separates SUBQUERY's two lives: DuckDB
    # uses the one tag for a table ref (carries `sample`) and for a subquery
    # expression (carries `class`), and they share barely half their fields.
    tag: ClassVar[str] = ""
    kind: ClassVar[str] = ""

    @classmethod
    def shape_key(cls) -> str:
        return f"{cls.tag}/{cls.kind}"


# ------------------------------------------------------------ the structure
# Nodes that carry no `type` of their own. They are shape, not vocabulary, so
# no drift gate applies to them — a change here fails as a validation error.


class Subquery(AstNode):
    """The ``{node, named_param_map}`` wrapper around a query.

    One class for two positions, because DuckDB spells them identically: the
    wrapper around a nested subquery, and the wrapper around a top-level
    statement. Keeping them apart would need a class the shape cannot pick
    between, which is the ambiguity `_structural` exists to avoid.
    """

    node: "Node"
    named_param_map: list[Any]


class CteValue(AstNode):
    aliases: list[str]
    query: Subquery
    materialized: str
    key_targets: list[Any]


class CteEntry(AstNode):
    key: str
    value: CteValue


class CteMap(AstNode):
    map: list[CteEntry]


class Document(AstNode):
    """What ``_serialize`` hands back."""

    error: bool
    statements: list[Subquery]


# ------------------------------------------------------------- query nodes
# `cte_map` is what makes a node a query node — the discriminator the walk has
# always used, now written down.


class Select(AstNode):
    tag: ClassVar[str] = "SELECT_NODE"
    kind: ClassVar[str] = "query"

    type: Literal["SELECT_NODE"] = "SELECT_NODE"
    modifiers: list["Node"]
    cte_map: CteMap
    select_list: list["Node"]
    from_table: "Node"
    where_clause: "Node | None"
    group_expressions: list["Node"]
    group_sets: list[list[int]]
    aggregate_handling: str
    having: "Node | None"
    sample: "Node | None"
    qualify: "Node | None"


class SetOperation(AstNode):
    tag: ClassVar[str] = "SET_OPERATION_NODE"
    kind: ClassVar[str] = "query"

    type: Literal["SET_OPERATION_NODE"] = "SET_OPERATION_NODE"
    modifiers: list["Node"]
    cte_map: CteMap
    setop_type: str
    left: "Node"
    right: "Node"
    setop_all: bool


class RecursiveCte(AstNode):
    tag: ClassVar[str] = "RECURSIVE_CTE_NODE"
    kind: ClassVar[str] = "query"

    type: Literal["RECURSIVE_CTE_NODE"] = "RECURSIVE_CTE_NODE"
    modifiers: list["Node"]
    cte_map: CteMap
    cte_name: str
    union_all: bool
    left: "Node"
    right: "Node"
    aliases: list[str]
    key_targets: list["Node"]


# --------------------------------------------------------------- table refs
# `sample` without `cte_map` is what makes a node a table ref.


class BaseTable(AstNode):
    tag: ClassVar[str] = "BASE_TABLE"
    kind: ClassVar[str] = "ref"

    type: Literal["BASE_TABLE"] = "BASE_TABLE"
    alias: str
    sample: "Node | None"
    query_location: int
    schema_name: str
    table_name: str
    column_name_alias: list[str]
    catalog_name: str
    at_clause: "Node | None"


class SubqueryRef(AstNode):
    tag: ClassVar[str] = "SUBQUERY"
    kind: ClassVar[str] = "ref"

    type: Literal["SUBQUERY"] = "SUBQUERY"
    alias: str
    sample: "Node | None"
    query_location: int
    subquery: Subquery
    column_name_alias: list[str]


class Join(AstNode):
    tag: ClassVar[str] = "JOIN"
    kind: ClassVar[str] = "ref"

    type: Literal["JOIN"] = "JOIN"
    alias: str
    sample: "Node | None"
    query_location: int
    left: "Node"
    right: "Node"
    condition: "Node | None"
    join_type: str
    ref_type: str
    using_columns: list[str]
    delim_flipped: bool
    duplicate_eliminated_columns: list["Node"]


class TableFunction(AstNode):
    tag: ClassVar[str] = "TABLE_FUNCTION"
    kind: ClassVar[str] = "ref"

    type: Literal["TABLE_FUNCTION"] = "TABLE_FUNCTION"
    alias: str
    sample: "Node | None"
    query_location: int
    function: "Node"
    column_name_alias: list[str]
    with_ordinality: str


class EmptyTable(AstNode):
    """``FROM`` nothing — what a bare ``SELECT 1`` reads."""

    tag: ClassVar[str] = "EMPTY"
    kind: ClassVar[str] = "ref"

    type: Literal["EMPTY"] = "EMPTY"
    alias: str
    sample: "Node | None"
    query_location: int


# -------------------------------------------------------------- expressions
# Only the two the walk reads, plus SUBQUERY's expression half.


class ColumnRef(AstNode):
    tag: ClassVar[str] = "COLUMN_REF"
    kind: ClassVar[str] = "expr"

    class_: str = Field(alias="class")
    type: Literal["COLUMN_REF"] = "COLUMN_REF"
    alias: str
    query_location: int
    column_names: list[str]


class Function(AstNode):
    tag: ClassVar[str] = "FUNCTION"
    kind: ClassVar[str] = "expr"

    class_: str = Field(alias="class")
    type: Literal["FUNCTION"] = "FUNCTION"
    alias: str
    query_location: int
    function_name: str
    schema_: str = Field(alias="schema")
    children: list["Node"]
    filter_: "Node | None" = Field(alias="filter")
    order_bys: "Node"
    distinct: bool
    is_operator: bool
    export_state: bool
    catalog: str


class SubqueryExpr(AstNode):
    tag: ClassVar[str] = "SUBQUERY"
    kind: ClassVar[str] = "expr"

    class_: str = Field(alias="class")
    type: Literal["SUBQUERY"] = "SUBQUERY"
    alias: str
    query_location: int
    subquery_type: str
    subquery: Subquery
    child: "Node | None"
    comparison_type: str


# ------------------------------------------------------------------- opaque


class Opaque(AstNode):
    """A tag with no class here. Carried verbatim, never interpreted.

    Its children are typed all the same. If this held a raw dict, a ``__FIT__``
    reference nested under a node DuckDB adds next year would be invisible to
    freezing — silently never frozen, which is the whole bug class the typed
    tree exists to close.

    Nothing can branch on what an ``Opaque`` *means*; the walk can only carry
    it and descend through it.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    # Empty for the nodes DuckDB serializes with no `type` at all — a
    # TABLESAMPLE clause is `{sample_size, is_percentage, method, seed}`.
    tag_name: str = ""
    fields: dict[str, Any]


INTERPRETED: tuple[type[AstNode], ...] = (
    Select,
    SetOperation,
    RecursiveCte,
    BaseTable,
    SubqueryRef,
    Join,
    TableFunction,
    EmptyTable,
    ColumnRef,
    Function,
    SubqueryExpr,
)

_BY_KEY: dict[str, type[AstNode]] = {m.shape_key(): m for m in INTERPRETED}


def _which(value: Any) -> str:
    """The shape key of a raw node, or ``opaque``.

    Reads the same two structural facts the walk always did — ``cte_map`` makes
    a query node, ``sample`` makes a table ref — so SUBQUERY's two lives land
    on different classes instead of one ragged union.
    """
    if isinstance(value, AstNode):
        return key if (key := value.shape_key()) in _BY_KEY else "opaque"
    if not isinstance(value, dict) or not isinstance(tag := value.get("type"), str):
        return "opaque"
    kind = "query" if "cte_map" in value else "ref" if "sample" in value else "expr"
    return key if (key := f"{tag}/{kind}") in _BY_KEY else "opaque"


type Node = Annotated[
    Union[  # noqa: UP007 - Annotated members need the explicit form
        Annotated[Select, Tag("SELECT_NODE/query")],
        Annotated[SetOperation, Tag("SET_OPERATION_NODE/query")],
        Annotated[RecursiveCte, Tag("RECURSIVE_CTE_NODE/query")],
        Annotated[BaseTable, Tag("BASE_TABLE/ref")],
        Annotated[SubqueryRef, Tag("SUBQUERY/ref")],
        Annotated[Join, Tag("JOIN/ref")],
        Annotated[TableFunction, Tag("TABLE_FUNCTION/ref")],
        Annotated[EmptyTable, Tag("EMPTY/ref")],
        Annotated[ColumnRef, Tag("COLUMN_REF/expr")],
        Annotated[Function, Tag("FUNCTION/expr")],
        Annotated[SubqueryExpr, Tag("SUBQUERY/expr")],
        Annotated[Opaque, Tag("opaque")],
    ],
    Discriminator(_which),
]


# ------------------------------------------------------------ the two doors


def from_json(raw: dict[str, Any]) -> Document:
    """DuckDB's JSON into the typed tree."""
    built = _convert(raw)
    if not isinstance(built, Document):
        raise TypeError(f"not a serialized statement: {sorted(raw)}")
    return built


def to_json(doc: Document) -> dict[str, Any]:
    """The typed tree back into DuckDB's JSON, field for field.

    Key order is free — measured: a node with its keys reversed deserializes
    identically — so this need only be complete, not ordered.
    """
    return _revert(doc)


# The shapes above that carry no `type`. Everything else in a node slot is
# vocabulary, and becomes an `Opaque` if no class claims it — including the
# nodes DuckDB serializes untagged, like a TABLESAMPLE clause. Pinned by
# `test_the_structural_shapes_are_what_we_think`, so a change here fails as a
# named mismatch rather than as a node quietly turning opaque.
_CTE_VALUE = frozenset({"aliases", "query", "materialized", "key_targets"})

_STRUCTURAL: frozenset[frozenset[str]] = frozenset(
    {
        frozenset({"node", "named_param_map"}),  # Subquery, Statement
        frozenset({"map"}),  # CteMap
        frozenset({"key", "value"}),  # CteEntry
        _CTE_VALUE,  # CteValue
        frozenset({"error", "statements"}),  # Document
    }
)


_BY_SHAPE: dict[frozenset[str], type[AstNode]] = {
    frozenset({"node", "named_param_map"}): Subquery,
    frozenset({"map"}): CteMap,
    frozenset({"key", "value"}): CteEntry,
    _CTE_VALUE: CteValue,
    frozenset({"error", "statements"}): Document,
}


def _structural(value: dict[str, Any]) -> type[AstNode] | None:
    """The class a shape names, or ``None`` if the dict is vocabulary.

    ``{key, value}`` alone does not settle it. Measured:
    ``SELECT * REPLACE (x+1 AS x)`` puts ``{"key", "value"}`` entries in a
    STAR's ``replace_list``, the same two keys a CTE entry has. Claiming those
    as CTE entries left them plain dicts inside an ``Opaque``, where nothing
    validates, and the walk stopped yielding them — one node fewer than the
    dict walk saw. The inner shape tells the two apart: a CTE entry's value is
    a CteValue, a replace entry's is an expression.
    """
    keys = frozenset(value)
    if keys == frozenset({"key", "value"}):
        inner = value.get("value")
        ok = isinstance(inner, dict | CteValue) and (
            isinstance(inner, CteValue) or frozenset(inner) == _CTE_VALUE
        )
        return CteEntry if ok else None
    return _BY_SHAPE.get(keys)


def _is_structural(value: dict[str, Any]) -> bool:
    return _structural(value) is not None


def _convert(value: Any) -> Any:
    """Raw JSON into something the models will validate.

    Anything not structural becomes an ``Opaque`` unless a class claims it, and
    an ``Opaque``'s fields are converted too — which is what keeps a typed node
    reachable underneath one.
    """
    if isinstance(value, list):
        return [_convert(v) for v in value]
    if not isinstance(value, dict):
        return value
    inner = {k: _convert(v) for k, v in value.items()}
    # Everything is built here rather than left for pydantic, because a node
    # inside an `Opaque` sits in a `dict[str, Any]` slot where pydantic
    # validates nothing. Leaving one raw is exactly the bug this file exists to
    # prevent: measured, a `__FIT__` under a BETWEEN stayed a plain dict and
    # the walk never saw it. So the invariant is uniform — after `_convert`,
    # every dict in the tree is a model.
    if (shape := _structural(value)) is not None:
        return shape.model_validate(inner)
    if (key := _which(value)) != "opaque":
        return _BY_KEY[key].model_validate(inner)
    tag = value.get("type")
    return Opaque(tag_name=tag if isinstance(tag, str) else "", fields=inner)


def _revert(value: Any) -> Any:
    if isinstance(value, list):
        return [_revert(v) for v in value]
    if isinstance(value, Opaque):
        return {k: _revert(v) for k, v in value.fields.items()}
    if isinstance(value, AstNode):
        fields = type(value).model_fields
        return {
            (spec.alias or name): _revert(getattr(value, name))
            for name, spec in fields.items()
        }
    if isinstance(value, dict):
        return {k: _revert(v) for k, v in value.items()}
    return value


QUERY_NODES = (Select, SetOperation, RecursiveCte)
REF_NODES = (BaseTable, SubqueryRef, Join, TableFunction, EmptyTable)


def is_query(node: Any) -> bool:
    """A query node — something with a ``cte_map``.

    The structural test survives alongside the class test on purpose: a query
    node DuckDB adds later arrives as an ``Opaque``, and answering ``False``
    for it would let the walk treat a whole subquery as an expression.
    """
    if isinstance(node, QUERY_NODES):
        return True
    return isinstance(node, Opaque) and "cte_map" in node.fields


def is_ref(node: Any) -> bool:
    """A table ref — something with a ``sample`` that is not a query node."""
    if isinstance(node, REF_NODES):
        return True
    return isinstance(node, Opaque) and "sample" in node.fields and not is_query(node)


def descendants(node: Any, *, deep: bool) -> Iterator[AstNode]:
    """Every node below ``node``, itself excluded.

    ``deep=False`` stops at nested query nodes — it still yields them, so a
    caller can replace one, but does not look inside — and skips ``cte_map``,
    which callers walk in definition order instead. Exactly ``_under``'s
    contract, minus the ``(parent, key)`` pair nothing can assign to now.
    """
    for child in child_nodes(node, skip_ctes=not deep):
        yield child
        if deep or not is_query(child):
            yield from descendants(child, deep=deep)


def rebuild(node: Any, fn: Callable[[AstNode], AstNode | None], *, deep: bool) -> Any:
    """``node`` with every descendant ``fn`` claims replaced by what it returns.

    Bottom-up, so a replacement is never re-visited — the property
    ``_bind_parameters`` used to get by collecting every site before touching
    any of them, now structural rather than a discipline to remember.

    ``fn`` returning ``None`` means *leave this one alone*.
    """

    def go(value: Any, *, inside_query: bool) -> Any:
        if isinstance(value, list):
            return [go(v, inside_query=inside_query) for v in value]
        if isinstance(value, Opaque):
            fields = {
                k: go(v, inside_query=inside_query) for k, v in value.fields.items()
            }
            return fn(rebuilt := value.model_copy(update={"fields": fields})) or rebuilt
        if isinstance(value, AstNode):
            if not deep and inside_query and is_query(value):
                return fn(value) or value  # yielded, not descended into
            fields = {}
            for name in type(value).model_fields:
                if name == "cte_map" and not deep:
                    continue
                fields[name] = go(getattr(value, name), inside_query=True)
            rebuilt = value.model_copy(update=fields)
            return fn(rebuilt) or rebuilt
        if isinstance(value, dict):
            return {k: go(v, inside_query=inside_query) for k, v in value.items()}
        return value

    return go(node, inside_query=False)


def child_nodes(node: Any, *, skip_ctes: bool = False) -> Iterator[AstNode]:
    """Every ``AstNode`` directly under ``node``, ``Opaque`` included.

    A free function rather than a method: ``Function`` has a field called
    ``children``, and the field name is not ours to rename.
    """
    if isinstance(node, Opaque):
        values: Any = [
            v for k, v in node.fields.items() if not (skip_ctes and k == "cte_map")
        ]
    elif isinstance(node, AstNode):
        values = [
            getattr(node, name)
            for name in type(node).model_fields
            if not (skip_ctes and name == "cte_map")
        ]
    elif isinstance(node, dict):
        values = list(node.values())
    elif isinstance(node, list):
        values = node
    else:
        return
    for value in values:
        if isinstance(value, AstNode):
            yield value
        elif isinstance(value, dict | list):
            yield from child_nodes(value, skip_ctes=skip_ctes)
