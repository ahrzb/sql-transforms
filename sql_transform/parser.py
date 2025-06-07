import ast
import dataclasses
from typing import TYPE_CHECKING, Any, Protocol

import datafusion
import sqlglot.expressions
from datafusion import functions as F

if TYPE_CHECKING:
    from sql_transform.context import SqlTransformContext


@dataclasses.dataclass
class Register:
    id: int
    hint: str

    @property
    def name(self):
        return f"r{self.hint}_{self.id}"


@dataclasses.dataclass
class PythonCode:
    def __init__(self) -> None:
        self.next_register_id = 2
        self.code: list[ast.stmt] = []
        self.data = Register(0, "data")
        self.result = self.assign("result", ast.Dict(keys=[], values=[]))

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
    def to_datafusion_expr(self) -> datafusion.Expr: ...
    def codegen(self, codegen: PythonCode) -> Register: ...


@dataclasses.dataclass(frozen=True)
class ColumnRef(Expression):
    name: str

    def hint_name(self):
        return self.name

    def to_datafusion_expr(self) -> datafusion.Expr:
        return F.col(self.name)

    def codegen(self, codegen: PythonCode) -> Register:
        return codegen.assign(self.name, codegen.read_field(self.name))


@dataclasses.dataclass(frozen=True)
class LiteralValue(Expression):
    value: Any

    def hint_name(self):
        return str(self.value)

    def to_datafusion_expr(self) -> datafusion.Expr:
        return datafusion.literal(self.value)

    def codegen(self, codegen: PythonCode) -> Register:
        return codegen.assign(f"lit_{self.value}", ast.Constant(value=self.value))


@dataclasses.dataclass(frozen=True)
class AggregationRef(Expression):
    id: int
    hint: str

    def hint_name(self) -> str:
        return self.hint

    @property
    def name(self) -> str:
        return f"{self.hint}_agg{self.id}"

    def to_datafusion_expr(self) -> datafusion.Expr:
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

    def to_datafusion_expr(self):
        match self.name, self.args:
            case "+", [a, b]:
                return a.to_datafusion_expr() + b.to_datafusion_expr()
            case "/", [a, b]:
                return a.to_datafusion_expr() / b.to_datafusion_expr()
            case "*", [a, b]:
                return a.to_datafusion_expr() * b.to_datafusion_expr()
            case "-", [a, b]:
                return a.to_datafusion_expr() - b.to_datafusion_expr()
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
class TransformFunction:
    """Represents a custom transform function that needs to be resolved at runtime."""

    operation: str
    args: list[Expression]
    over: WindowSpecification

    def to_datafusion_expr(self, context=None):
        """Convert to datafusion expression, resolving transforms via context."""
        # This will be resolved at runtime by the transformer
        raise NotImplementedError(
            f"Transform {self.operation} needs context resolution"
        )


@dataclasses.dataclass(frozen=True)
class AggregateFunction:
    operation: str
    args: list[Expression]
    over: WindowSpecification

    def to_datafusion_expr(self):
        base_expr = self._get_base_expression()

        # Apply window specification if present
        if self.over.partition_by:
            # For now, we'll handle windowing in the transformer
            # DataFusion window functions need special handling
            return base_expr
        else:
            return base_expr

    def _get_base_expression(self):
        """Get the base datafusion expression without windowing."""
        match self.operation, self.args:
            case "avg", [a]:
                return F.avg(a.to_datafusion_expr())
            case "stddev", [a]:
                return F.stddev(a.to_datafusion_expr())
            case "sum", [a]:
                return F.sum(a.to_datafusion_expr())
            case "count", [a]:
                return F.count(a.to_datafusion_expr())
            case "min", [a]:
                return F.min(a.to_datafusion_expr())
            case "max", [a]:
                return F.max(a.to_datafusion_expr())
            case _:
                raise NotImplementedError(f"Unsupported aggregation: {self.operation}")

    def to_window_expr(self):
        """Convert to a DataFusion window expression."""
        import datafusion

        base_expr = self._get_base_expression()

        if self.over.partition_by:
            partition_exprs = [p.to_datafusion_expr() for p in self.over.partition_by]
            # DataFusion window syntax
            return datafusion.WindowExpr(
                fun=base_expr,
                partition_by=partition_exprs,
                order_by=[],  # We can extend this later
                window_frame=None,  # We can extend this later
            )
        else:
            return base_expr


@dataclasses.dataclass
class Query:
    columns: dict[str, Expression] = dataclasses.field(default_factory=dict)
    aggregations: dict[AggregationRef, AggregateFunction | TransformFunction] = (
        dataclasses.field(default_factory=dict)
    )


def parse_dot_expression(expression, aggregations, parse_expression):
    """Parse dot expressions like sklearn.standardize."""
    namespace = expression.this
    func_expr = expression.expression
    if (
        isinstance(func_expr, sqlglot.expressions.Anonymous)
        and str(namespace).lower() == "sklearn"
    ):
        # Parse sklearn.function_name as transform function
        func_name = f"sklearn.{func_expr.this.lower()}"
        args = [parse_expression(arg) for arg in func_expr.expressions]
        hint = f"{func_name}_{args[0].hint_name() if args else 'none'}"
        ref = AggregationRef(len(aggregations), hint)
        aggregations[ref] = TransformFunction(
            func_name, args, over=WindowSpecification()
        )
        return ref
    else:
        raise NotImplementedError(f"Unsupported dot expression: {expression}")


def parse_anonymous_function(expression, aggregations, parse_expression, context):
    """Parse anonymous function calls.

    Distinguish between aggregations and transforms.
    """
    func_name = expression.this.lower()
    args = [parse_expression(arg) for arg in expression.expressions]
    hint = f"{func_name}_{args[0].hint_name() if args else 'none'}"
    ref = AggregationRef(len(aggregations), hint)

    # Use context to resolve function
    agg_or_transform = context.resolve_function(func_name, args)
    aggregations[ref] = agg_or_transform

    return ref


def parse(sql: str, context: "SqlTransformContext") -> Query:  # noqa: C901
    query = sqlglot.parse_one(sql)

    aggregations: dict[AggregationRef, AggregateFunction | TransformFunction] = {}

    def parse_expression(expression: sqlglot.expressions.Expression) -> Expression:  # noqa: C901
        match expression:
            case sqlglot.expressions.Column():
                return ColumnRef(expression.this.this)
            case sqlglot.expressions.Literal():
                # Try to convert to appropriate type
                value = expression.this
                if isinstance(value, str):
                    # Try to parse as number
                    try:
                        if "." in value:
                            value = float(value)
                        else:
                            value = int(value)
                    except ValueError:
                        pass
                return LiteralValue(value)
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
            case sqlglot.expressions.Add():
                return ApplyFunction(
                    "+",
                    [
                        parse_expression(expression.this),
                        parse_expression(expression.expression),
                    ],
                )
            case sqlglot.expressions.Mul():
                return ApplyFunction(
                    "*",
                    [
                        parse_expression(expression.this),
                        parse_expression(expression.expression),
                    ],
                )
            case sqlglot.expressions.Div():
                return ApplyFunction(
                    "/",
                    [
                        parse_expression(expression.this),
                        parse_expression(expression.expression),
                    ],
                )
            case sqlglot.expressions.Paren():
                # Parentheses are just for grouping, parse the inner expression
                return parse_expression(expression.this)
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
            case sqlglot.expressions.Dot():
                return parse_dot_expression(expression, aggregations, parse_expression)
            case sqlglot.expressions.Anonymous():
                return parse_anonymous_function(
                    expression, aggregations, parse_expression, context
                )
            case _:
                raise NotImplementedError(f"Cannot parse {expression}")

    columns = {}
    for expression in query.expressions:
        if isinstance(expression, sqlglot.expressions.Alias):
            columns[expression.alias] = parse_expression(expression.this)
        else:
            # Auto-generate alias for non-aliased expressions
            parsed_expr = parse_expression(expression)
            alias = parsed_expr.hint_name()
            columns[alias] = parsed_expr

    return Query(columns=columns, aggregations=aggregations)
