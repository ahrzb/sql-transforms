"""Every refusal the model can raise.

Its own hierarchy rather than sklearn's: sklearn is not a runtime dependency,
and a SQL transform must not need one in order to refuse.
"""


class TransformError(Exception):
    """Every refusal. Named, and raised at construction (P7)."""


class CorrelatedFit(TransformError):
    """A ``__FIT__`` subtree correlates out of itself.

    It is per-outer-row, so it cannot be evaluated once into a table.
    Supporting it means lifting the correlation to a ``GROUP BY`` and
    rewriting it as a join — which is marginalization. Future work, not a
    permanent boundary.
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
