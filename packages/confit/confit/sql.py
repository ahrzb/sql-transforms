"""Build SQL as a tree, render it losslessly.

The shape mirrors DuckDB's Python expression API (`ColumnExpression`,
`ConstantExpression`, `FunctionExpression`, `CaseExpression`,
`CoalesceOperator`, and the operators), with three differences that make it
fit for writing serving SQL:

* **A constant carries its type, and renders it.** `lit(0.1)` is a DOUBLE
  and renders as `CAST('0.1' AS DOUBLE)`, so the text means the value it was
  built from. DuckDB's own `str()` of an expression drops the type: run as
  text, `0.1 + 0.2` is DECIMAL arithmetic and answers 0.3, while the
  expression object answers 0.30000000000000004.
* **The tree is plain data.** Every node is a frozen dataclass; `children`
  and `walk()` traverse it, so a caller can inspect what it built.
* **It renders to text, and converts to DuckDB.** `sql()` is what confit and
  DuckDB both read; `to_duckdb()` gives the equivalent DuckDB expression
  object, for a relation.

    from confit import sql as S
    a, x = S.col("a"), S.col("x")
    e = S.case(a.isnull(), S.lit(0.0)).otherwise((x - S.lit(3.5)) * S.lit(2.0))
    q = S.select(e.alias("z")).from_("__THIS__").where(a > S.lit(0))
    q.sql()

Python's `and`/`or`/`not` cannot be overloaded, so as in DuckDB's API, `&`,
`|` and `~` are AND, OR and NOT; `==` builds a comparison, so a node is not
hashable.
"""

from __future__ import annotations

import decimal
import math
import re
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import pyarrow as pa

__all__ = [
    "Expr",
    "Query",
    "case",
    "coalesce",
    "col",
    "fn",
    "lit",
    "select",
]

_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")

_TYPE_NAMES = {
    pa.bool_(): "BOOLEAN",
    pa.int8(): "TINYINT",
    pa.int16(): "SMALLINT",
    pa.int32(): "INTEGER",
    pa.int64(): "BIGINT",
    pa.uint8(): "UTINYINT",
    pa.uint16(): "USMALLINT",
    pa.uint32(): "UINTEGER",
    pa.uint64(): "UBIGINT",
    pa.float32(): "FLOAT",
    pa.float64(): "DOUBLE",
    pa.string(): "VARCHAR",
}


def type_name(t: pa.DataType | str) -> str:
    """The SQL spelling of a type: an Arrow type, or a SQL type name as is."""
    if isinstance(t, str):
        return t
    if t in _TYPE_NAMES:
        return _TYPE_NAMES[t]
    if pa.types.is_decimal(t):
        return f"DECIMAL({t.precision},{t.scale})"
    raise TypeError(f"no SQL spelling for the Arrow type {t}")


def _quote_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _quote_str(s: str) -> str:
    return "'" + s.replace("'", "''") + "'"


def _wrap(v: Any) -> Expr:
    """An operand: an `Expr` as is, a Python value as a typed constant."""
    return v if isinstance(v, Expr) else lit(v)


class Expr:
    """A node. Operators build new nodes; nothing is evaluated here."""

    __hash__ = None  # type: ignore[assignment]  # `==` builds a node

    @property
    def children(self) -> tuple[Expr, ...]:
        return ()

    def walk(self) -> Iterator[Expr]:
        """Every node, pre-order."""
        yield self
        for c in self.children:
            yield from c.walk()

    def sql(self) -> str:
        raise NotImplementedError

    def to_duckdb(self) -> Any:
        """The equivalent `duckdb.Expression`."""
        raise NotImplementedError

    def __str__(self) -> str:
        return self.sql()

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.sql()}>"

    def __bool__(self) -> bool:
        raise TypeError("an expression has no truth value; use & | ~ for AND OR NOT")

    # ---- operators
    def _bin(self, op: str, other: Any) -> Expr:
        return BinOp(op, self, _wrap(other))

    def _rbin(self, op: str, other: Any) -> Expr:
        return BinOp(op, _wrap(other), self)

    def __add__(self, o):
        return self._bin("+", o)

    def __radd__(self, o):
        return self._rbin("+", o)

    def __sub__(self, o):
        return self._bin("-", o)

    def __rsub__(self, o):
        return self._rbin("-", o)

    def __mul__(self, o):
        return self._bin("*", o)

    def __rmul__(self, o):
        return self._rbin("*", o)

    def __truediv__(self, o):
        return self._bin("/", o)

    def __rtruediv__(self, o):
        return self._rbin("/", o)

    def __floordiv__(self, o):
        return self._bin("//", o)

    def __rfloordiv__(self, o):
        return self._rbin("//", o)

    def __mod__(self, o):
        return self._bin("%", o)

    def __rmod__(self, o):
        return self._rbin("%", o)

    def __pow__(self, o):
        return Call("power", (self, _wrap(o)))

    def __rpow__(self, o):
        return Call("power", (_wrap(o), self))

    def __neg__(self):
        return Unary("-", self)

    def __eq__(self, o):  # type: ignore[override]
        return self._bin("=", o)

    def __ne__(self, o):  # type: ignore[override]
        return self._bin("<>", o)

    def __lt__(self, o):
        return self._bin("<", o)

    def __le__(self, o):
        return self._bin("<=", o)

    def __gt__(self, o):
        return self._bin(">", o)

    def __ge__(self, o):
        return self._bin(">=", o)

    def __and__(self, o):
        return self._bin("AND", o)

    def __rand__(self, o):
        return self._rbin("AND", o)

    def __or__(self, o):
        return self._bin("OR", o)

    def __ror__(self, o):
        return self._rbin("OR", o)

    def __invert__(self):
        return Unary("NOT", self)

    # ---- methods, named as in DuckDB's API
    def isnull(self) -> Expr:
        return Postfix("IS NULL", self)

    def isnotnull(self) -> Expr:
        return Postfix("IS NOT NULL", self)

    def isin(self, *items: Any) -> Expr:
        return In(self, tuple(_wrap(i) for i in items), negated=False)

    def isnotin(self, *items: Any) -> Expr:
        return In(self, tuple(_wrap(i) for i in items), negated=True)

    def between(self, lo: Any, hi: Any) -> Expr:
        return Between(self, _wrap(lo), _wrap(hi))

    def is_not_distinct_from(self, o: Any) -> Expr:
        return self._bin("IS NOT DISTINCT FROM", o)

    def is_distinct_from(self, o: Any) -> Expr:
        return self._bin("IS DISTINCT FROM", o)

    def concat(self, o: Any) -> Expr:
        return self._bin("||", o)

    def cast(self, t: pa.DataType | str) -> Expr:
        return Cast(self, type_name(t))

    def alias(self, name: str) -> Expr:
        return Alias(self, name)

    def field(self, name: str) -> Expr:
        """A struct field: `e.field("a")` is `(e).a`."""
        return Field(self, name)


@dataclass(frozen=True, eq=False, repr=False)
class Column(Expr):
    """A column, optionally qualified: `col("t", "a")` is `"t"."a"`."""

    path: tuple[str, ...]

    def sql(self) -> str:
        return ".".join(_quote_ident(p) for p in self.path)

    def to_duckdb(self):
        import duckdb

        return duckdb.ColumnExpression(*self.path)


@dataclass(frozen=True, eq=False, repr=False)
class Const(Expr):
    """A typed constant; `value` None is a typed NULL."""

    value: Any
    type: str

    def sql(self) -> str:
        v, t = self.value, self.type
        if v is None:
            return f"CAST(NULL AS {t})"
        if isinstance(v, bool):
            return f"CAST({'TRUE' if v else 'FALSE'} AS {t})"
        if isinstance(v, int):
            return f"CAST({v} AS {t})"
        if isinstance(v, float):
            # Through a string, which DuckDB reads correctly rounded; a bare
            # numeral would be a DECIMAL first, which rounds twice.
            if math.isnan(v):
                text = "nan"
            elif math.isinf(v):
                text = "inf" if v > 0 else "-inf"
            else:
                text = repr(v)
            return f"CAST({_quote_str(text)} AS {t})"
        if isinstance(v, decimal.Decimal):
            # A numeral is a DECIMAL literal; the cast fixes its width.
            return f"CAST({v:f} AS {t})"
        if isinstance(v, str):
            return f"CAST({_quote_str(v)} AS {t})"
        raise TypeError(f"no SQL constant for {v!r}")

    def to_duckdb(self):
        import duckdb

        return duckdb.ConstantExpression(self.value).cast(duckdb.sqltype(self.type))


@dataclass(frozen=True, eq=False, repr=False)
class Call(Expr):
    """A function call by name: a builtin, or a function passed in `udfs=`."""

    name: str
    args: tuple[Expr, ...]

    @property
    def children(self):
        return self.args

    def sql(self) -> str:
        return f"{self.name}({', '.join(a.sql() for a in self.args)})"

    def to_duckdb(self):
        import duckdb

        args = [a.to_duckdb() for a in self.args]
        if self.name.lower() == "coalesce":
            # An operator in DuckDB's binder, not a catalog function.
            return duckdb.CoalesceOperator(*args)
        return duckdb.FunctionExpression(self.name, *args)


_DUCK_BIN = {
    "+": lambda a, b: a + b,
    "-": lambda a, b: a - b,
    "*": lambda a, b: a * b,
    "/": lambda a, b: a / b,
    "//": lambda a, b: a // b,
    "%": lambda a, b: a % b,
    "=": lambda a, b: a == b,
    "<>": lambda a, b: a != b,
    "<": lambda a, b: a < b,
    "<=": lambda a, b: a <= b,
    ">": lambda a, b: a > b,
    ">=": lambda a, b: a >= b,
    "AND": lambda a, b: a & b,
    "OR": lambda a, b: a | b,
}


@dataclass(frozen=True, eq=False, repr=False)
class BinOp(Expr):
    op: str
    left: Expr
    right: Expr

    @property
    def children(self):
        return (self.left, self.right)

    def sql(self) -> str:
        return f"({self.left.sql()} {self.op} {self.right.sql()})"

    def to_duckdb(self):
        import duckdb

        a, b = self.left.to_duckdb(), self.right.to_duckdb()
        if self.op in _DUCK_BIN:
            return _DUCK_BIN[self.op](a, b)
        if self.op == "||":
            # `concat` would skip a NULL operand; `||` propagates it.
            return duckdb.FunctionExpression("||", a, b)
        # DuckDB's expression API has no IS [NOT] DISTINCT FROM operator.
        return duckdb.SQLExpression(self.sql())


@dataclass(frozen=True, eq=False, repr=False)
class Unary(Expr):
    op: str
    operand: Expr

    @property
    def children(self):
        return (self.operand,)

    def sql(self) -> str:
        return f"({self.op} {self.operand.sql()})"

    def to_duckdb(self):
        x = self.operand.to_duckdb()
        return -x if self.op == "-" else ~x


@dataclass(frozen=True, eq=False, repr=False)
class Postfix(Expr):
    op: str
    operand: Expr

    @property
    def children(self):
        return (self.operand,)

    def sql(self) -> str:
        return f"({self.operand.sql()} {self.op})"

    def to_duckdb(self):
        x = self.operand.to_duckdb()
        return x.isnull() if self.op == "IS NULL" else x.isnotnull()


@dataclass(frozen=True, eq=False, repr=False)
class In(Expr):
    operand: Expr
    items: tuple[Expr, ...]
    negated: bool

    @property
    def children(self):
        return (self.operand, *self.items)

    def sql(self) -> str:
        op = "NOT IN" if self.negated else "IN"
        return f"({self.operand.sql()} {op} ({', '.join(i.sql() for i in self.items)}))"

    def to_duckdb(self):
        x, items = self.operand.to_duckdb(), [i.to_duckdb() for i in self.items]
        return x.isnotin(*items) if self.negated else x.isin(*items)


@dataclass(frozen=True, eq=False, repr=False)
class Between(Expr):
    operand: Expr
    lo: Expr
    hi: Expr

    @property
    def children(self):
        return (self.operand, self.lo, self.hi)

    def sql(self) -> str:
        return f"({self.operand.sql()} BETWEEN {self.lo.sql()} AND {self.hi.sql()})"

    def to_duckdb(self):
        return self.operand.to_duckdb().between(
            self.lo.to_duckdb(), self.hi.to_duckdb()
        )


@dataclass(frozen=True, eq=False, repr=False)
class Cast(Expr):
    operand: Expr
    type: str

    @property
    def children(self):
        return (self.operand,)

    def sql(self) -> str:
        return f"CAST({self.operand.sql()} AS {self.type})"

    def to_duckdb(self):
        import duckdb

        return self.operand.to_duckdb().cast(duckdb.sqltype(self.type))


@dataclass(frozen=True, eq=False, repr=False)
class Field(Expr):
    operand: Expr
    name: str

    @property
    def children(self):
        return (self.operand,)

    def sql(self) -> str:
        return f"({self.operand.sql()}).{_quote_ident(self.name)}"

    def to_duckdb(self):
        import duckdb

        return duckdb.FunctionExpression(
            "struct_extract",
            self.operand.to_duckdb(),
            duckdb.ConstantExpression(self.name),
        )


@dataclass(frozen=True, eq=False, repr=False)
class Case(Expr):
    """`CASE WHEN c THEN v ... [ELSE e] END`; build with `case(c, v)`, then
    `.when(c, v)` and `.otherwise(e)`."""

    whens: tuple[tuple[Expr, Expr], ...]
    default: Expr | None = None

    @property
    def children(self):
        out = [x for pair in self.whens for x in pair]
        return (*out, self.default) if self.default is not None else tuple(out)

    def when(self, cond: Any, value: Any) -> Case:
        if self.default is not None:
            raise ValueError("a WHEN after the ELSE")
        return Case((*self.whens, (_wrap(cond), _wrap(value))))

    def otherwise(self, value: Any) -> Case:
        if self.default is not None:
            raise ValueError("a second ELSE")
        return Case(self.whens, _wrap(value))

    def sql(self) -> str:
        arms = " ".join(f"WHEN {c.sql()} THEN {v.sql()}" for c, v in self.whens)
        tail = f" ELSE {self.default.sql()}" if self.default is not None else ""
        return f"CASE {arms}{tail} END"

    def to_duckdb(self):
        import duckdb

        (c0, v0), *rest = self.whens
        e = duckdb.CaseExpression(c0.to_duckdb(), v0.to_duckdb())
        for c, v in rest:
            e = e.when(c.to_duckdb(), v.to_duckdb())
        if self.default is not None:
            e = e.otherwise(self.default.to_duckdb())
        return e


@dataclass(frozen=True, eq=False, repr=False)
class Alias(Expr):
    operand: Expr
    name: str

    @property
    def children(self):
        return (self.operand,)

    def sql(self) -> str:
        return f"{self.operand.sql()} AS {_quote_ident(self.name)}"

    def to_duckdb(self):
        return self.operand.to_duckdb().alias(self.name)


# ------------------------------------------------------------------ builders


def col(*path: str) -> Column:
    """A column: `col("a")`, or qualified `col("t", "a")`."""
    if not path or not all(isinstance(p, str) and p for p in path):
        raise ValueError("a column is one or more non-empty names")
    return Column(tuple(path))


def _infer(v: Any) -> str:
    if isinstance(v, bool):
        return "BOOLEAN"
    if isinstance(v, int):
        return "BIGINT"
    if isinstance(v, float):
        return "DOUBLE"
    if isinstance(v, str):
        return "VARCHAR"
    if isinstance(v, decimal.Decimal):
        _, digits, exp = v.as_tuple()
        if not isinstance(exp, int):
            raise ValueError(f"no DECIMAL for {v}")
        scale = max(0, -exp)
        return f"DECIMAL({max(len(digits), scale, 1)},{scale})"
    raise TypeError(f"no SQL type for {v!r}; pass one: lit(v, type)")


def lit(value: Any, type: pa.DataType | str | None = None) -> Const:  # noqa: A002
    """A typed constant. Without `type`: bool is BOOLEAN, int BIGINT, float
    DOUBLE, str VARCHAR, Decimal the narrowest DECIMAL holding it. `None`
    needs a type."""
    if type is None:
        if value is None:
            raise ValueError("a NULL constant needs a type: lit(None, pa.int64())")
        return Const(value, _infer(value))
    return Const(value, type_name(type))


def fn(name: str, *args: Any) -> Call:
    """A call by name: `fn("least", a, 10)`."""
    if not _IDENT.match(name):
        raise ValueError(f"{name!r} is not a function name")
    return Call(name, tuple(_wrap(a) for a in args))


def coalesce(*args: Any) -> Call:
    return fn("coalesce", *args)


def case(cond: Any, value: Any) -> Case:
    """`CASE WHEN cond THEN value`; continue with `.when` and `.otherwise`."""
    return Case(((_wrap(cond), _wrap(value)),))


# ------------------------------------------------------------------ queries


@dataclass(frozen=True, eq=False)
class _Join:
    how: str
    table: str
    alias: str | None
    on: Expr | None


@dataclass(frozen=True, eq=False)
class Query:
    """`SELECT ... FROM ... [JOIN ...] [WHERE ...]`, built immutably."""

    items: tuple[Expr, ...]
    source: tuple[str, str | None] | None = None
    joins: tuple[_Join, ...] = ()
    predicate: Expr | None = None

    def from_(self, table: str, alias: str | None = None) -> Query:
        return Query(self.items, (table, alias), self.joins, self.predicate)

    def join(
        self,
        table: str,
        on: Expr | None = None,
        *,
        how: str = "inner",
        alias: str | None = None,
    ) -> Query:
        how = how.upper()
        if how not in ("INNER", "LEFT", "CROSS"):
            raise ValueError(f"join kind {how!r} is not INNER, LEFT or CROSS")
        if (on is None) != (how == "CROSS"):
            raise ValueError("a CROSS join takes no ON; any other join takes one")
        return Query(
            self.items,
            self.source,
            (*self.joins, _Join(how, table, alias, on)),
            self.predicate,
        )

    def where(self, predicate: Expr) -> Query:
        p = predicate if self.predicate is None else self.predicate & predicate
        return Query(self.items, self.source, self.joins, p)

    def sql(self) -> str:
        if self.source is None:
            raise ValueError("a query needs .from_(table)")
        # A builder of SQL text by design: identifiers are quoted, constants
        # rendered by `Const`.
        items = ", ".join(i.sql() for i in self.items)
        out = f"SELECT {items} FROM {_rel(*self.source)}"  # noqa: S608
        for j in self.joins:
            out += f" {j.how} JOIN {_rel(j.table, j.alias)}"
            if j.on is not None:
                out += f" ON {j.on.sql()}"
        if self.predicate is not None:
            out += f" WHERE {self.predicate.sql()}"
        return out

    def __str__(self) -> str:
        return self.sql()


def _rel(table: str, alias: str | None) -> str:
    # `__THIS__` is a placeholder the caller substitutes, never quoted.
    name = table if table == "__THIS__" else _quote_ident(table)
    return name if alias is None else f"{name} AS {_quote_ident(alias)}"


def select(*items: Any) -> Query:
    if not items:
        raise ValueError("a SELECT needs at least one item")
    return Query(tuple(_wrap(i) for i in items))
