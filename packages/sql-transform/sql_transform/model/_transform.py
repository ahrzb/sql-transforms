"""A transform is a function ``(F, T) -> R`` over relations.

``__FIT__`` and ``__THIS__`` are its two parameters. At the top level ``fit``
binds one and ``transform`` binds the other. Which half is learned and which
is live is read off the text — there is no annotation to remember and none to
forget.

This module is the estimator surface — ``SQLTransform``, the output
currencies, and the sklearn contract. The compiled text itself lives in
``_program``: resolution, binding, freezing, ``Fitted``, ``Program``.

Implements `docs/superpowers/specs/2026-08-07-datamodel-redesign-design.md`.
DuckDB is both the parser and the oracle — a construct means what DuckDB
computes.
"""

import sys
from typing import Any, Self

import pyarrow as pa

from sql_transform.model._ast import Captured, Connection, Relation
from sql_transform.model._errors import NotFitted, TransformError
from sql_transform.model._nodes import Node
from sql_transform.model._nodes import field as node_field
from sql_transform.model._program import Fitted, Program

OUTPUTS = ("default", "arrow", "duckdb", "pandas", "numpy")


def _as_output(
    table: pa.Table, output: str, source: Relation = None, aligned: bool = True
) -> Any:
    """The result in the caller's currency.

    ``pandas`` carries the caller's index when there is one to carry, which is
    what every sklearn transformer does. Resetting it was silent: in a
    ``FeatureUnion`` alongside an estimator that preserves the index, pandas
    aligns on index rather than position and NaN-pads the difference — four
    rows in, seven out, no error.

    A SQL transform may change cardinality, which an sklearn one cannot. When
    the row counts disagree there is no row correspondence to express, so no
    index is attached rather than one invented.
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

    An index is a claim about which input row each output row came from, and
    positional correspondence is the only evidence available. Any ORDER BY or
    LIMIT in the residual destroys it — measured, and silently: a three-row
    frame indexed a/b/c through ``ORDER BY v`` came back with a's label on b's
    value, no error.

    The query's own ORDER BY / LIMIT lives in the top-level ``modifiers``,
    which is what this reads. A deep scan is wrong: an ordinary projection
    carries an empty nested ORDER_MODIFIER, so scanning everything never
    carries an index at all.

    Even so this is a good-faith reading rather than a proof — SQL guarantees
    no row order without ORDER BY. Losing the index is loud (a FeatureUnion
    misaligns visibly) while attaching a wrong one is not, so where the two
    compete the doubt resolves toward dropping it.
    """
    return not node_field(node, "modifiers")


def _with_index(frame: Any, source: Relation, aligned: bool) -> Any:
    index = getattr(source, "index", None)
    if aligned and index is not None and len(index) == len(frame):
        frame.index = index
    return frame


class SQLTransform:
    """``F -> Fitted``, and an sklearn estimator.

    ``fit`` returns the ``Fitted`` artifact rather than ``self``. That is the
    currying the model is built on — ``.params`` is a thing you can ship —
    and it costs nothing with sklearn, which never reads what ``fit``
    returned: ``Pipeline`` keeps the object it called and asks *it* to
    ``transform`` later. So ``fit`` also remembers, and both spellings agree:

        t.fit(D).transform(X)     # curried: the artifact transforms
        t.fit(D); t.transform(X)  # stateful: the estimator transforms

    ``bindings`` and ``foreign`` are constructor parameters as well as frame
    lookups, because ``clone`` rebuilds an estimator inside sklearn's own
    frame, where a member or a lookup table is not in scope. They ride along
    in ``get_params`` so a clone resolves to the very same objects.

    Those two mappings are *adopted*, not copied, and completed in place with
    whatever the frame supplied. ``clone`` demands that ``get_params`` hand
    back the very object the constructor was given — a defensive copy fails
    its identity check — and carrying the completed set is the whole point.

    Construction parses, plans and refuses; nothing else does — all of it in
    ``Program.compile``, which this class holds the result of.
    """

    def __init__(
        self,
        sql: str,
        output: str = "default",
        connection: Connection | None = None,
        captured: Captured | None = None,
    ) -> None:
        # Resolution happens once, in `compile`, and captures by value:
        # `scope` is a local that dies with this call, so no caller frame is
        # retained and rebinding a member afterwards cannot change what was
        # built. The frame is read *here*, not inside `compile` — one level
        # deeper would capture from the wrong caller.
        frame = sys._getframe(1)
        scope = frame.f_globals | frame.f_locals
        del frame

        if output not in OUTPUTS:
            raise TransformError(f"output must be one of {OUTPUTS}; got {output!r}")
        self.output = output
        # Given rather than conjured. A transform that makes its own hidden
        # connection cannot compose with anything: a DuckDBPyRelation belongs
        # to the connection that built it, so lazy output only chains when
        # both stages share one. Pass it and you own it.
        program = Program.compile(sql, scope, connection=connection, captured=captured)
        self._program = program
        self.connection = program.connection
        # Adopted, not copied: see the class docstring. Explicit entries win,
        # and are how a clone keeps names the frame it was rebuilt in cannot
        # see.
        self.captured = program.captured
        self.foreign = program.foreign
        self.bindings = program.bindings
        self.node = program.node
        self.depth = program.depth
        self.source = program.source  # the exact object: clone's identity check
        self.sql = program.sql
        self._steps = program.steps
        self._residual = program.residual
        self._shadowable = program.shadowable
        self.fitted_: Fitted | None = None
        self.feature_names_out_: list[str] | None = None

    # -- the model's own surface ----------------------------------------------

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
        return self._require_fit().params

    @property
    def instances_(self) -> dict[int, Any]:
        return self._require_fit().instances

    # -- the sklearn surface ---------------------------------------------------

    def _require_fit(self) -> Fitted:
        if self.fitted_ is None:
            raise NotFitted("this transform has not been fit; call fit first")
        return self.fitted_

    def transform(self, data: Relation) -> Any:
        fitted = self._require_fit()
        if self.output == "duckdb":
            lazy = fitted.relation(data)  # the whole point: never materialise
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
        """sklearn's own hook, because the default clones by deep-copying
        every parameter and a live DuckDB connection cannot be deep-copied —
        ``clone``, and so ``GridSearchCV``/``cross_val_score``/``Pipeline``,
        died with a raw TypeError on any transform built with ``connection=``.

        A connection is a resource, not a value: the clone shares it. Rebuilt
        from ``source`` so the plan is derived rather than copied.
        """
        return type(self)(
            self.source,
            output=self.output,
            connection=self.connection,
            captured=self.captured,
        )

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return {
            "sql": self.source,
            "output": self.output,
            "connection": self.connection,
            "captured": self.captured,
        }

    def set_params(self, **params: Any) -> Self:
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
