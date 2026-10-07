"""Every refusal the model can raise.

Its own hierarchy rather than sklearn's: sklearn is not a runtime dependency,
and a SQL transform must not need one in order to refuse.
"""


class TransformError(Exception):
    """Base error for named construction, fit, and execution refusals."""


class CorrelatedFit(TransformError):
    """A ``__FIT__`` subtree correlates out of itself, and no rewrite reaches it.

    The equality case is lifted to a ``GROUP BY`` and served as a keyed table
    (`_correlate`). What is left raises this, and ``reason`` says which of the
    named shapes it is — the set of reasons is the refusal list, kept short on
    purpose and written down in
    ``packages/sql-transform/spec/fit/decorrelation-refusals.md``.

    ``reason`` is required so every refusal identifies its unsupported shape.
    """

    def __init__(self, message: str, reason: str) -> None:
        super().__init__(message)
        self.reason = reason


class WholeTrainingSet(TransformError):
    """Honouring this text would put every row of ``__FIT__`` in the artifact.

    Not a correlation problem: there is simply no smaller table that answers
    it. Retention is allowed — but only where the author wrote a query whose
    value *is* those rows, so the artifact's size is visible in the text
    instead of being an implementation detail of freezing.
    """


class NotRowWise(TransformError):
    """The text cannot serve one output row per ``__THIS__`` row.

    A projection's whole promise, checked at construction against the residual
    — the text that survives freezing, where only ``__THIS__`` and params
    remain. ``reason`` names which of the closed set of shapes it is; the set
    is the refusal list, kept short on purpose, and the projection test walks
    it looking for gaps.

    ``reason`` has no default, for the same reason ``CorrelatedFit``'s has
    none: a default reason is a refusal nobody has to name.
    """

    def __init__(self, message: str, reason: str) -> None:
        super().__init__(message)
        self.reason = reason


class KeyNotUnique(TransformError):
    """A relation the spine joins can answer one serving row with many rows.

    Cardinality is measured at fit, where params exist. Equality keys must be
    unique; a keyless CROSS side must have exactly one row, while a
    request-preserving outer side may have zero or one. Syntax alone cannot
    establish this: DISTINCT and QUALIFY can both produce lawful unique keys.
    """


class UnknownName(TransformError):
    """An identifier resolved to nothing in the caller's frame."""


class NestingTooDeep(TransformError):
    """More than ``MAX_DEPTH`` levels of member calls."""


class NotFitted(TransformError):
    """The estimator surface was used before ``fit``.

    Ours rather than sklearn's ``NotFittedError``, because sklearn is not a
    runtime dependency — a SQL transform must not need one to refuse.
    """
