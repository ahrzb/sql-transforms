"""A transform is a function ``(F, T) -> R`` over relations.

``__FIT__`` and ``__THIS__`` are its two parameters. At the top level ``fit``
binds one and ``transform`` binds the other. Which half is learned and which
is live is read off the text — there is no annotation to remember and none to
forget.

This module exposes ``SQLTransform`` and its output modes. The compiled text
itself lives in ``_program``: resolution, binding, freezing, ``Fitted``,
``Program``.

Implements `packages/sql-transform/spec/general-transforms.md`.
DuckDB is both the parser and the oracle — a construct means what DuckDB
computes.
"""

import sys
from typing import Any, Self

import pyarrow as pa

from sql_transform._ast import Captured, Connection, Relation
from sql_transform._errors import NotFitted, TransformError
from sql_transform._nodes import Node
from sql_transform._nodes import field as node_field
from sql_transform._program import Fitted, Program

OUTPUTS = ("default", "arrow", "duckdb", "pandas", "numpy")


def _as_output(
    table: pa.Table, output: str, source: Relation = None, aligned: bool = True
) -> Any:
    """Convert eager Arrow output to the selected output mode.

    Preserve a source pandas index only when ``aligned`` and the row counts
    agree. Otherwise leave the result's default index: attaching unrelated
    labels would silently misalign downstream index-based joins.
    """
    match output:
        case "default" | "arrow" | "duckdb":
            return table
        case "pandas":
            return _with_index(table.to_pandas(), source, aligned)
        case "numpy":
            return _with_index(table.to_pandas(), source, aligned).to_numpy()
    raise TransformError(f"output must be one of {OUTPUTS}; got {output!r}")


def _keeps_row_order(node: Node) -> bool:
    """Whether output row *i* still stands for input row *i*.

    An index claims positional correspondence. Top-level modifiers such as
    ORDER BY and LIMIT invalidate that claim. Nested modifiers are irrelevant:
    even ordinary expressions can contain empty ORDER_MODIFIER nodes.

    This is a heuristic, not an ordering proof; SQL without an
    ORDER BY does not guarantee row order.
    """
    return not node_field(node, "modifiers")


def _with_index(frame: Any, source: Relation, aligned: bool) -> Any:
    index = getattr(source, "index", None)
    if aligned and index is not None and len(index) == len(frame):
        frame.index = index
    return frame


class SQLTransform:
    """``F -> Fitted``, and an sklearn estimator.

    ``fit`` returns a ``Fitted`` artifact and also stores it for subsequent
    estimator-style ``transform`` calls:

        t.fit(D).transform(X)
        t.fit(D); t.transform(X)

    ``sql`` is authored two-parameter SQL. ``captured`` supplies explicit
    Python bindings, overriding caller-frame names, and is adopted and completed
    in place for replay. See ``Program.compile`` for mapping ownership.

    Output modes are Arrow (``default``/``arrow``), lazy ``duckdb``, ``pandas``,
    and ``numpy``. Lazy chaining requires a shared caller-owned ``connection``;
    keep the fitted artifact alive until its lazy outputs are consumed.

    Construction performs structural planning. Fit binds the data schema,
    evaluates learned tables and preserves general SQL cardinality.
    """

    def __init__(
        self,
        sql: str,
        output: str = "default",
        connection: Connection | None = None,
        captured: Captured | None = None,
    ) -> None:
        # See _program's module contract: the scope must come from this caller,
        # and no frame is retained after resolution.
        frame = sys._getframe(1)
        scope = frame.f_globals | frame.f_locals
        del frame

        if output not in OUTPUTS:
            raise TransformError(f"output must be one of {OUTPUTS}; got {output!r}")
        self.output = output
        program = Program.compile(sql, scope, connection=connection, captured=captured)
        self._program = program
        self.connection = program.connection  # borrowed; None uses owned connections
        self.captured = program.captured  # adopted author mapping for source replay
        self.foreign = program.foreign  # declared relation-batch callbacks
        # Live captured relations, separate from params.
        self.bindings = program.bindings
        self.node = program.node  # resolved SQL, before fit freezing
        self.depth = program.depth  # member-call nesting
        self.source = program.source  # the exact object: clone's identity check
        self.sql = program.sql  # resolved diagnostics, not replay input
        self._steps = program.steps
        self._residual = program.residual
        self._shadowable = program.shadowable
        self.fitted_: Fitted | None = None  # most recent fit, or None before fit
        # Learned on transform, not fit.
        self.feature_names_out_: list[str] | None = None

    def __repr__(self) -> str:
        state = "fitted" if self.fitted_ is not None else "unfitted"
        return f"SQLTransform({self.sql!r}, output={self.output!r}, {state})"

    def fit(self, data: Relation, y: Any = None) -> Fitted:
        """Partial application — and the estimator remembers the result.

        ``y`` is accepted and ignored: a target belongs in the relation, as a
        column ``__FIT__`` can read, not in a second argument the SQL cannot
        name.
        """
        self.fitted_ = self._program.fit(data)
        return self.fitted_

    __call__ = fit

    @property
    def params_(self) -> dict[str, pa.Table]:
        """The stored learned-table mapping; raises ``NotFitted`` before fit."""
        return self._require_fit().params

    @property
    def instances_(self) -> dict[int, Any]:
        """The stored opaque-ID instance mapping belonging to the current params."""
        return self._require_fit().instances

    def _require_fit(self) -> Fitted:
        if self.fitted_ is None:
            raise NotFitted("this transform has not been fit; call fit first")
        return self.fitted_

    def transform(self, data: Relation) -> Any:
        """Apply the latest fit in the selected output mode; refuse before fit.

        Lazy ``duckdb`` output retains registrations on the fitted artifact;
        see ``Fitted.relation`` for its consumption and release lifetime.
        """
        fitted = self._require_fit()
        if self.output == "duckdb":
            lazy = fitted.relation(data)
            self.feature_names_out_ = list(lazy.columns)
            return lazy
        out = fitted.transform(data)
        self.feature_names_out_ = out.column_names
        return _as_output(out, self.output, data, _keeps_row_order(fitted.node))

    def fit_transform(self, data: Relation, y: Any = None) -> Any:
        """On the training relation this is exactly ``run(t, D)`` — that is
        the *freezing is faithful* law, not a coincidence."""
        self.fit(data)
        return self.transform(data)

    def get_feature_names_out(self, input_features: Any = None) -> list[str]:
        """Return names from the latest transform; ``input_features`` is ignored."""
        if self.feature_names_out_ is None:
            raise NotFitted(
                "output column names are only known once something has been "
                "transformed; call transform or fit_transform first"
            )
        return list(self.feature_names_out_)

    def set_output(self, *, transform: str | None = None) -> Self:
        """sklearn's opt-in: ``pandas`` or ``numpy`` for a downstream
        estimator, ``default`` for the model's own arrow tables."""
        if transform is not None:
            if transform not in OUTPUTS:
                raise TransformError(
                    f"output must be one of {OUTPUTS}; got {transform!r}"
                )
            self.output = transform
        return self

    def __sklearn_clone__(self) -> Self:
        """Rebuild from authored source, sharing captures and the connection.

        A DuckDB connection is a borrowed resource and cannot be deep-copied.
        The clone has its own fit state and derives a fresh plan.
        """
        return type(self)(
            self.source,
            output=self.output,
            connection=self.connection,
            captured=self.captured,
        )

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Return constructor arguments; keep captures shared and ignore ``deep``."""
        return {
            "sql": self.source,
            "output": self.output,
            "connection": self.connection,
            "captured": self.captured,
        }

    def set_params(self, **params: Any) -> Self:
        """Update constructor arguments.

        Changes to SQL, captures or connection reset fit.
        """
        unknown = set(params) - set(self.get_params())
        if unknown:
            raise TransformError(f"unknown parameters {sorted(unknown)}")
        if {"sql", "captured", "connection"} & set(params):
            # The plan is derived from all three, so rebuild rather than let
            # them drift apart.
            rebuilt = type(self)(
                params.get("sql", self.source),
                output=params.get("output", self.output),
                connection=params.get("connection", self.connection),
                captured=params.get("captured", self.captured),
            )
            self.__dict__.update(rebuilt.__dict__)
        elif "output" in params:
            self.set_output(transform=params["output"])
        return self


def run(transform: SQLTransform, data: Relation) -> pa.Table:
    """Both parameters bound to the same relation, with no freezing at all.

    The reference side of "freezing is faithful". It is a *binding*, not a
    rewrite, which is what keeps that law from restating the implementation.
    """
    return transform._program.run(data)
