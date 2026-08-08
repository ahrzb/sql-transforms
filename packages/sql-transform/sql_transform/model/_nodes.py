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

from collections.abc import Iterator
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
    """The ``{node, named_param_map}`` wrapper around a nested query."""

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


class Statement(AstNode):
    node: "Node"
    named_param_map: list[Any]


class Document(AstNode):
    """What ``_serialize`` hands back."""

    error: bool
    statements: list[Statement]


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
    setop_all: bool
    left: "Node"
    right: "Node"


class RecursiveCte(AstNode):
    tag: ClassVar[str] = "RECURSIVE_CTE_NODE"
    kind: ClassVar[str] = "query"

    type: Literal["RECURSIVE_CTE_NODE"] = "RECURSIVE_CTE_NODE"
    modifiers: list["Node"]
    cte_map: CteMap
    cte_name: str
    union_all: bool
    aliases: list[str]
    key_targets: list["Node"]
    left: "Node"
    right: "Node"


# --------------------------------------------------------------- table refs
# `sample` without `cte_map` is what makes a node a table ref.


class BaseTable(AstNode):
    tag: ClassVar[str] = "BASE_TABLE"
    kind: ClassVar[str] = "ref"

    type: Literal["BASE_TABLE"] = "BASE_TABLE"
    alias: str
    sample: "Node | None"
    query_location: int
    catalog_name: str
    schema_name: str
    table_name: str
    column_name_alias: list[str]
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

    type: Literal["COLUMN_REF"] = "COLUMN_REF"
    class_: str = Field(alias="class")
    alias: str
    query_location: int
    column_names: list[str]


class Function(AstNode):
    tag: ClassVar[str] = "FUNCTION"
    kind: ClassVar[str] = "expr"

    type: Literal["FUNCTION"] = "FUNCTION"
    class_: str = Field(alias="class")
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

    type: Literal["SUBQUERY"] = "SUBQUERY"
    class_: str = Field(alias="class")
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
    return Document.model_validate(_convert(raw))


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
_STRUCTURAL: frozenset[frozenset[str]] = frozenset(
    {
        frozenset({"node", "named_param_map"}),  # Subquery, Statement
        frozenset({"map"}),  # CteMap
        frozenset({"key", "value"}),  # CteEntry
        frozenset({"aliases", "query", "materialized", "key_targets"}),  # CteValue
        frozenset({"error", "statements"}),  # Document
    }
)


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
    if frozenset(value) in _STRUCTURAL:
        return inner  # shape, not vocabulary: pydantic builds it from the slot
    # Built here rather than left for pydantic, because a node inside an
    # `Opaque` sits in a `dict[str, Any]` slot, where pydantic validates
    # nothing. Leaving it raw is exactly the bug this file exists to prevent:
    # measured, a `__FIT__` under a BETWEEN stayed a plain dict and the walk
    # never saw it.
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


def child_nodes(node: Any) -> Iterator[AstNode]:
    """Every ``AstNode`` directly under ``node``, ``Opaque`` included.

    A free function rather than a method: ``Function`` has a field called
    ``children``, and the field name is not ours to rename.
    """
    if isinstance(node, Opaque):
        values: Any = node.fields.values()
    elif isinstance(node, AstNode):
        values = (getattr(node, name) for name in type(node).model_fields)
    elif isinstance(node, dict):
        values = node.values()
    elif isinstance(node, list):
        values = node
    else:
        return
    for value in values:
        if isinstance(value, AstNode):
            yield value
        elif isinstance(value, dict | list):
            yield from child_nodes(value)
