---
id: TASK-87
title: 'Extract Program: the compiled two-parameter text, out of SQLTransform'
status: Done
assignee: []
created_date: '2026-08-11 18:07'
updated_date: '2026-08-11 18:07'
labels: []
dependencies: []
ordinal: 80000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`SQLTransform` is two things wearing one name: **the two-parameter text,
compiled** — parse, splice, resolve, freeze, fit — and **an sklearn estimator**
— `output=`, `set_output`, `get_params`, `feature_names_out_`, the index
alignment in `_as_output`.

`SQLProjection` needs the first half and none of the second. Subclassing would
hand it the estimator surface it does not want and make every future refusal
ask "which class am I in"; the two are siblings, not a specialisation.

Extract the first half into `model/_program.py` as a plain value both classes
*hold*:

    @dataclass(frozen=True, slots=True)
    class Program:
        node: Node                       # resolved text, both parameters live
        steps: list[tuple[str, Node]]    # the fit DAG, in dependency order
        residual: Node
        shadowable: set[str]
        bindings: Bindings
        foreign: Foreign
        captured: Captured
        source: str
        connection: Connection | None

        @classmethod
        def compile(cls, sql, scope, *, connection=None, captured=None) -> Self
        def fit(self, data) -> Fitted
        def run(self, data) -> pa.Table   # both parameters bound to one relation

What moves out of `_transform.py`: `_splice`, `_argument`, `_reserve`,
`_resolve`, `_lease`, `_rendered`, `_give_back`, `_EXECUTIONS`, `Fitted`, the
body of `SQLTransform.__init__`, the body of `SQLTransform.fit`, and the body
of `run`.

What stays: `OUTPUTS`, `_as_output`, `_keeps_row_order`, `_with_index`,
`transform`, `fit_transform`, `get_feature_names_out`, `set_output`,
`__sklearn_clone__`, `get_params`, `set_params`, `fitted_`,
`feature_names_out_`.

Behaviour-neutral. No refusal changes name, no message changes text, no public
spelling changes — `run(t, D)` stays a module-level function and delegates.
<!-- SECTION:DESCRIPTION:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Three things that will bite:

**The frame depth.** `SQLTransform.__init__` reads its caller with
`sys._getframe(1)`. Moving compilation one call deeper silently reads the
wrong frame — every `FROM df` replacement-scan idiom would resolve to nothing,
or worse, to a name inside `Program.compile`. So `scope` is a *parameter*:
each public class does its own `_getframe(1)` and passes the mapping in.
`Program` never touches the stack. Pin it with a test that captures a local
from the caller of `SQLProjection(...)`.

**`clone`'s identity check.** `get_params` must hand back the very object the
constructor was given — a defensive copy fails sklearn's check. `Program`
holds the *identical* `captured` dict (adopted, not copied), and
`SQLTransform.get_params` reads it through. `frozen=True` on the dataclass
freezes the binding, not the dict, which is what is wanted: `_resolve`
completes `captured` in place during `compile`.

**`Fitted.connection`.** It is on `Fitted` today and reads `self._own` logic
off `SQLTransform`. It moves with `Program`; `_own` becomes
`connection is None` on the Program.

Gate: the existing suite, unchanged, from the repo root. If any test needs
editing beyond an import line, the extraction is not behaviour-neutral and the
diff is wrong.

Blocks the new `SQLProjection` (row-wise projections over explicit `__FIT__`,
serving 1-1, usable as a `tfm_fit`/`tfm_transform` leaf).

Closed by the 2026-08-13 grooming pass: landed in bb60533 - Program
(compile/fit/run + Fitted) lives in model/_program.py, SQLTransform keeps the
sklearn estimator surface and delegates. The frame-depth, clone-identity and
Fitted.connection traps above are each pinned (_calling_test, _sklearn_test,
_connection_test). It unblocked SQLProjection (f157c16) and later
marginalize, both shipped on top of it.
<!-- SECTION:NOTES:END -->
