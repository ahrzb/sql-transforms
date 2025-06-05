import ast
import dataclasses
from typing import Protocol

import datafusion
import sqlglot.expressions
from datafusion import functions as F


@dataclasses.dataclass
class Register:
    id: int
    hint: str

    @property
    def name(self):
        return f"r{self.hint}_{self.id}"


@dataclasses.dataclass
class PythonCode:
    def __init__(self):
        self.next_register_id = 2
        self.code: list[ast.stmt] = []
        self.data = Register(0, "data")
        self.result = self.assign("result", ast.Dict(elts=[], ctx=ast.Load()))

    def lookup(self, register: Register) -> ast.expr:
        return ast.Name(id=register.hint, ctx=ast.Load())

    def read_field(self, name: str) -> ast.expr:
        return ast.Subscript(
            value=ast.Constant(self.data.name), slice=ast.Constant(name), ctx=ast.Load()
        )

    def assign(self, register_hint: str, value: ast.expr) -> Register:
        register = Register(self.next_register_id, register_hint)
        self.next_register_id += 1
        self.code.append(
            ast.Assign(targets=[ast.Name(register.name, ctx=ast.Store())], value=value)
        )
        return register

    def write_field(self, name: str, value: ast.expr):
        self.code.append(
            ast.Assign(
                targets=[
                    ast.Subscript(
                        value=self.result.name,
                        slice=ast.Constant(name),
                        ctx=ast.Store(),
                    )
                ],
                value=value,
            )
        )

    def render(self):
        pass

    def eval(self):
        pass


class Expression(Protocol):
    def hint_name(self) -> str: ...
    def datafusion(self) -> datafusion.Expr: ...
    def codegen(self, codegen: PythonCode) -> Register: ...


@dataclasses.dataclass(frozen=True)
class ColumnRef(Expression):
    name: str

    def hint_name(self):
        return self.name

    def datafusion(self) -> datafusion.Expr:
        return F.col(self.name)

    def codegen(self, codegen: PythonCode) -> Register:
        return codegen.assign(self.name, codegen.read_field(self.name))


@dataclasses.dataclass(frozen=True)
class AggregationRef(Expression):
    id: int
    hint: str

    def hint_name(self) -> str:
        return self.hint

    @property
    def name(self) -> str:
        return f"{self.hint}_agg{self.id}"

    def datafusion(self) -> datafusion.Expr:
        return F.col(self.name)

    def codegen(self, codegen: PythonCode) -> Register:
        return codegen.assign(self.name, codegen.read_field(self.name))


@dataclasses.dataclass(frozen=True)
class ApplyFunction(Expression):
    name: str
    args: list[Expression]

    def hint_name(self):
        op = {
            "+": "add",
            "-": "sub",
            "/": "div",
            "*": "mul",
        }
        name = op.get(self.name, "unknown")
        args = "_".join(arg.hint_name() for arg in self.args)
        return f"{name}_{args}"

    def datafusion(self):
        match self.name, self.args:
            case "+", [a, b]:
                return a.datafusion() + b.datafusion()
            case "/", [a, b]:
                return a.datafusion() / b.datafusion()
            case "*", [a, b]:
                return a.datafusion() * b.datafusion()
            case "-", [a, b]:
                return a.datafusion() - b.datafusion()
            case _:
                raise NotImplementedError()

        return F.col(self.name)

    def codegen(self, codegen: PythonCode) -> Register:
        match self.name, self.args:
            case "+", [a, b]:
                return codegen.assign(
                    self.name,
                    ast.BinOp(
                        left=codegen.lookup(a.codegen(codegen)),
                        op=ast.Add(),
                        right=codegen.lookup(b.codegen(codegen)),
                    ),
                )
            case "/", [a, b]:
                return codegen.assign(
                    self.name,
                    ast.BinOp(
                        left=codegen.lookup(a.codegen(codegen)),
                        op=ast.Div(),
                        right=codegen.lookup(b.codegen(codegen)),
                    ),
                )
            case "*", [a, b]:
                return codegen.assign(
                    self.name,
                    ast.BinOp(
                        left=codegen.lookup(a.codegen(codegen)),
                        op=ast.Mult(),
                        right=codegen.lookup(b.codegen(codegen)),
                    ),
                )
            case "-", [a, b]:
                return codegen.assign(
                    self.name,
                    ast.BinOp(
                        left=codegen.lookup(a.codegen(codegen)),
                        op=ast.Sub(),
                        right=codegen.lookup(b.codegen(codegen)),
                    ),
                )
            case _:
                raise NotImplementedError()


@dataclasses.dataclass(frozen=True)
class WindowSpecification:
    partition_by: list[Expression] = dataclasses.field(default_factory=list)


@dataclasses.dataclass(frozen=True)
class AggregateFunction:
    operation: str
    args: list[Expression]
    over: WindowSpecification

    def datafusion(self):
        match self.operation, self.args:
            case "avg", [a]:
                return F.avg(a.datafusion())
            case "stddev", [a]:
                return F.stddev(a.datafusion())
            case _:
                raise NotImplementedError()


@dataclasses.dataclass
class Query:
    columns: dict[str, Expression] = dataclasses.field(default_factory=dict)
    aggregations: dict[AggregationRef, AggregateFunction] = dataclasses.field(
        default_factory=dict
    )


def parse(sql: str):
    query = sqlglot.parse_one(sql)

    aggregations: dict[AggregationRef, AggregateFunction] = {}

    def parse_expression(expression: sqlglot.expressions.Expression) -> Expression:
        match expression:
            case sqlglot.expressions.Column():
                return ColumnRef(expression.this.this)
            case sqlglot.expressions.Avg():
                expr = parse_expression(expression.this)
                ref = AggregationRef(len(aggregations), expr.hint_name())
                aggregations[ref] = AggregateFunction(
                    "avg", [expr], over=WindowSpecification()
                )
                return ref
            case sqlglot.expressions.Stddev():
                expr = parse_expression(expression.this)
                ref = AggregationRef(len(aggregations), expr.hint_name())
                aggregations[ref] = AggregateFunction(
                    "stddev", [expr], over=WindowSpecification()
                )
                return ref
            case sqlglot.expressions.Sub():
                return ApplyFunction(
                    "-",
                    [
                        parse_expression(expression.this),
                        parse_expression(expression.expression),
                    ],
                )
            case sqlglot.expressions.Window():
                over = WindowSpecification(
                    partition_by=[
                        parse_expression(p)
                        for p in expression.args.get("partition_by", [])
                    ]
                )
                agg_ref = parse_expression(expression.this)
                if isinstance(agg_ref, AggregationRef):
                    aggregations[agg_ref] = dataclasses.replace(
                        aggregations[agg_ref], over=over
                    )
                    return agg_ref
                else:
                    raise NotImplementedError(
                        "Window functions only supported on aggregations"
                    )
            case _:
                raise NotImplementedError(f"Cannot parse {expression}")

    columns = {}
    for expression in query.expressions:
        if not isinstance(expression, sqlglot.expressions.Alias):
            raise NotImplementedError("Non aliased expressions not supported yet")
        columns[expression.alias] = parse_expression(expression.this)

    return Query(columns=columns, aggregations=aggregations)
