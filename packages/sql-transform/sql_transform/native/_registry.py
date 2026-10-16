"""The catalog and the translation of one `PythonTransform`.

A translator turns ONE fitted estimator into its output fields, as
`confit.sql` expressions over the feature expressions it is handed:

    @translates(StandardScaler)
    def _(est, x: list[S.Expr], types: list[pa.DataType]) -> list[S.Expr]: ...

`x` holds one expression per declared feature, `types` the declared type
of each. A numeric or boolean feature reads as a DOUBLE (a boolean as 0/1)
with NULL as NaN, as `transform` sees it in a numeric row; a string passes
as is, NULL as NULL (the step hands it None). A number reaches the
translator as `x - 0.0`, the same double, which confit can compute once a
row (`_apart`), unless its entry says otherwise (`Entry.apart`). A
translator that cannot serve this estimator's configuration raises
`NotNative` naming why.

The framework does the rest, the same way for every entry: one `SqlFunction`
named like the step, taking the instance id then the features, whose every
output field selects the instance's expression by id
(`CASE WHEN id = 0 THEN ... WHEN id IN (1, 2) THEN ... END`, instances whose
output field is the same tree sharing an arm). Past the first output field,
an output field that is one arm (every output field, in a step of one
instance) is the tree alone, without the CASE. As in the step, a NULL id
answers NULL (a NULL struct or list, for a struct or list return) and an id
the step does not know raises. Before it is returned, confit builds it into
the query reading every output field (`query`): a translation confit refuses
leaves the step Python.

Bit-exact is the default: a translation whose parity bound is above 0
serves only with `to_native(step, allow_bound=True)`. Such an entry can
change a prediction: on repeated training values, HistGradientBoosting
flipped labels under the Box-Cox entry, where the twin of an elementwise
family flips none (loops/native/decisions/closed/matvec-parity-bound.md).
A parity bound is an ulp bound (`ulps`), or an error scale
(`ErrorScale`) for a family that rounds in an order its twin does not
follow.

Where the validation of the twin raises, the entry traps
(loops/native/decisions/closed/tolerated-differences.md, the ruling): the
input guard. Each estimator a translation reads, the step's own or a
composition's part, adds tests on its own input expressions: the values
its twin rejects in each feature, found by handing the fitted estimator
±inf, NaN and NULL in the container its twin is handed (`_probed`), and
its family's domain tests (`rejects`), such as Box-Cox's `x <= 0`. The
framework joins an instance's tests with OR into the first output field,
`CASE WHEN ... THEN error(...)`, which DuckDB and confit fire on a read of
any field, as they fire the unknown-id trap.
"""

from __future__ import annotations

import contextvars
import dataclasses
import math
import sys
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import reduce
from typing import Any, NamedTuple

import numpy as np
import pyarrow as pa
from confit import DuckDBInferFn, Function, SqlFunction
from confit import sql as S

from sql_transform._udf import PythonTransform, _dense
from sql_transform.native._helpers import SameTree, f64, isnan

Translator = Callable[[Any, list[S.Expr], list[pa.DataType]], list[S.Expr]]

EPS = np.longdouble(2.0**-52)


@dataclass(frozen=True)
class ErrorScale:
    """A parity bound in units of each output field's error scale S:
    `|g(entry) - g(twin)| <= K*eps*S + tau`, field by field, eps = 2**-52
    (decisions/closed/matvec-parity-bound.md, the ruling). For a family
    that rounds in an order its twin does not follow, such as a dot
    product the twin hands to BLAS; each operand is the family's to derive.

    `k(est)` is K and `tau(est)` the floor for underflow, each a number or
    one per output field. `s(est, x)` is S for each output field at one
    row `x`: the features as the twin's `transform` sees them (NaN for
    NULL, a boolean as 0 or 1), as long doubles, so that a formula kept in
    numpy's arithmetic does not overflow where the field does not (where a
    long double is a double, as on macOS on arm64, an S past DBL_MAX
    bounds nothing). `g` maps both sides before they are compared: the
    identity unless the family names one (a distance compares its
    square). `on` names what a map compares, as a message reads it after
    the bound ("on the squares"): a map and its name come together."""

    k: Callable[[Any], Any]
    s: Callable[[Any, np.ndarray], Any]
    tau: Callable[[Any], Any]
    g: Callable[[Any], Any] | None = None
    on: str = ""

    def __post_init__(self) -> None:
        if (self.g is None) != (not self.on):
            raise ValueError("an error scale names its map with `on`, and only a map")

    def at(self, est: Any, x: np.ndarray, width: int) -> np.ndarray:
        """Each of the `width` output fields' bound K*eps*S + tau at one
        row `x`, as long doubles."""
        k, tau = self.k(est), self.tau(est)
        b = (
            np.asarray(k, np.longdouble)
            * EPS
            * np.asarray(self.s(est, x), np.longdouble)
        )
        return np.broadcast_to(b + np.asarray(tau, np.longdouble), (width,))


class FieldBounds(NamedTuple):
    """Each output field's parity bound at one row of a composition
    (`PartBounds`): where `near[j]` is NaN, within `ulps[j]` doubles of the
    twin (0 = bit-exact); else |entry - twin| <= near[j], the K*eps*S + tau
    of the error scale of the part that gives the field."""

    ulps: np.ndarray
    near: np.ndarray


@dataclass(frozen=True)
class PartBounds:
    """The parity bound of a composition that holds a part whose bound is
    above 0 (compose.py): each output field keeps the bound of the part
    that gives it, which reads what its twin reads. `at(X)` answers each
    output field's bound at the one-row input `X`, as the twin's
    `transform` takes it (`FieldBounds`); `within` names that bound as a
    message reads it."""

    at: Callable[[Any], FieldBounds]
    within: str


@dataclass(frozen=True)
class Entry:
    """A catalog entry: the translator and its declared parity bound, in
    doubles per output field (0 = bit-exact) or in an error scale, fixed by
    the derivation or the measurement its module cites.

    `ulps` is the class's ceiling. An entry whose bound depends on the
    configuration also has `per_estimator`, which answers one fitted
    estimator's own bound, from 0 to the ceiling; without it, every
    estimator's bound is the ceiling. An entry with a `scale` is bounded
    by that error scale instead, never bit-exact, unless `scaled_of(est)`
    says that one estimator is bit-exact (`scale_for`). A step whose bound
    is above 0 serves only with `to_native(step, allow_bound=True)`.

    `base(est, types)` is the input guard's probe row, a value the twin
    accepts in each feature, for a family where the default (`_base`)
    may not be one (an encoder's categories). A composition (`composes`)
    is not probed: its parts are. Its bound is its parts' too:
    `part_bounds_of(est)` answers it (`PartBounds`), None where each part
    is bit-exact, or raises `NotNative` where a part whose bound is above
    0 sits where no bound is derived.

    `apart(est)` says whether the translator reads each number as `x -
    0.0` (`_apart`). Every entry does, unless `apart_of(est)` says no for
    one estimator: a projection whose build would pass its cap where it
    reads them so (project.py)."""

    translate: Translator
    ulps: int
    per_estimator: Callable[[Any], int] | None = None
    scale: ErrorScale | None = None
    base: Callable[[Any, list[pa.DataType]], list[Any]] | None = None
    composes: bool = False
    part_bounds_of: Callable[[Any], PartBounds | None] | None = None
    apart_of: Callable[[Any], bool] | None = None
    scaled_of: Callable[[Any], bool] | None = None

    def apart(self, est: Any) -> bool:
        """The translator reads each number of `est`'s row as `x - 0.0`."""
        return self.apart_of is None or self.apart_of(est)

    @property
    def varies(self) -> bool:
        """The bound is the estimator's, not one for the whole class."""
        return self.per_estimator is not None or self.scaled_of is not None

    def scale_for(self, est: Any) -> ErrorScale | None:
        """`est`'s error scale: the entry's, None where it has none or
        where `scaled_of(est)` says that `est` is bit-exact."""
        if self.scale is None or (self.scaled_of and not self.scaled_of(est)):
            return None
        return self.scale

    def part_bounds(self, est: Any) -> PartBounds | None:
        """A composition's bound from its parts; None where each part is
        bit-exact, and for any other entry."""
        return None if self.part_bounds_of is None else self.part_bounds_of(est)

    def bound(self, est: Any) -> int:
        """`est`'s own ulp bound: what its translation is checked to."""
        name = type(est).__name__
        if self.scale_for(est) is not None:
            raise ValueError(f"{name}: its parity bound is an error scale, not ulps")
        if self.part_bounds(est) is not None:
            raise ValueError(f"{name}: its parity bound is its parts', not ulps")
        if self.per_estimator is None:
            return self.ulps
        b = self.per_estimator(est)
        if not 0 <= b <= self.ulps:
            raise ValueError(
                f"{type(est).__name__}: a bound of {b} ulps, past the class's"
                f" ceiling of {self.ulps}"
            )
        return b

    def exact(self, est: Any) -> bool:
        """The translation of `est` is bit-exact."""
        return (
            self.scale_for(est) is None
            and self.part_bounds(est) is None
            and self.bound(est) == 0
        )

    def within(self, est: Any) -> str:
        """`est`'s parity bound, as a message reads it."""
        parts = self.part_bounds(est)
        if parts is not None:
            return parts.within
        scale = self.scale_for(est)
        if scale is None:
            return f"within {self.bound(est)} ulps"
        k = scale.k(est)
        kk = f"{float(k):g}" if np.ndim(k) == 0 else "K"
        on = f" {scale.on}" if scale.on else ""
        return f"within {kk}·eps·S + τ{on} (S its error scale)"


_CATALOG: dict[type, Entry] = {}

# The instance-id parameter. Double underscores keep it apart from any
# feature name a step can declare (features come from user columns).
_ID = "__iid"

# What an id the step does not know raises with (`error()` takes a constant
# message, so the id itself is not in it).
_UNKNOWN = (
    "{}: an instance id not in the fitted instances (params table and"
    " instances are from different fits)"
)
# What the input guard traps with: the twin raises on the row.
_REJECTED = (
    "{}: a value in this row that the fitted estimator rejects, as its twin raises"
)


class NotNative(Exception):  # noqa: N818 — a reason, raised and caught
    """This step has no native translation; the message says why."""


def translates(
    *classes: type,
    ulps: int = 0,
    bound: Callable[[Any], int] | None = None,
    scale: ErrorScale | None = None,
    base: Callable[[Any, list[pa.DataType]], list[Any]] | None = None,
    composes: bool = False,
    part_bounds: Callable[[Any], PartBounds | None] | None = None,
    apart: Callable[[Any], bool] | None = None,
    scaled: Callable[[Any], bool] | None = None,
) -> Callable[[Translator], Translator]:
    """Register a translator for exactly these estimator classes, bit-exact
    unless `ulps` or `scale` says otherwise. With `bound`, `ulps` is the
    ceiling and `bound(est)` each fitted estimator's own bound (a class
    whose configurations differ: `FunctionTransformer` is bit-exact for the
    identity and within 2 ulps for `np.log10`). With `scale`, the bound is
    that error scale, and with `scaled` too, only where `scaled(est)` says
    so: elsewhere the estimator is bit-exact (`AdditiveChi2Sampler` of one
    sample step). `base`, `composes`, a composition's `part_bounds` and
    `apart` (`apart_of`) are `Entry`'s. A subclass is not covered: it may
    override what `transform` computes."""
    if bound is not None and ulps == 0:
        raise ValueError("a per-estimator bound needs a ceiling above 0")
    if scale is not None and (ulps or bound is not None):
        raise ValueError("an error scale is the whole bound, without ulps")
    if scaled is not None and scale is None:
        raise ValueError("`scaled` says where an error scale applies, and needs one")

    def deco(fn: Translator) -> Translator:
        for c in classes:
            if c in _CATALOG:
                raise ValueError(f"{c.__name__} is already in the catalog")
            _CATALOG[c] = Entry(
                fn, ulps, bound, scale, base, composes, part_bounds, apart, scaled
            )
        return fn

    return deco


def catalog() -> dict[type, Entry]:
    """The registered estimator classes and their entries."""
    return dict(_CATALOG)


# How the twin hands an estimator its row, which fixes what its validation
# sees: "list", the step's own one-row list, which numpy makes a boolean
# array over boolean features none NULL, objects beside a string, and
# float64 otherwise; "objects", an object array (a ColumnTransformer's
# part, as its `_check_X` makes it from the list); "doubles", a float64
# array (a pipeline's later step); "booleans", a later step over boolean
# features only, a boolean array where the step's row has no NULL and
# float64 where it has one.
HANDS = ("list", "objects", "doubles", "booleans")


@dataclass
class _Guard:
    """The input guard of the instance being translated: its tests so far;
    how the twin hands the estimator now being translated its row
    (`HANDS`); and, over boolean features only, the test that the step's
    row has no NULL."""

    hand: str
    no_null: S.Expr | None
    tests: list[S.Expr] = field(default_factory=list)
    # How the last estimator probed hands on its output (`handed_on`).
    out: str = "doubles"


_GUARD: contextvars.ContextVar[_Guard] = contextvars.ContextVar("guard")


def is_trap(name: str, error: BaseException) -> bool:
    """`error` is one of the traps of the native twin of step `name`: its
    input guard's, or an unknown instance id's."""
    s = str(error)
    return _REJECTED.format(name) in s or _UNKNOWN.format(name) in s


def rejects(test: S.Expr) -> None:
    """Trap where `test` holds: a family's domain test on the input
    expressions of the estimator now being translated, where its twin
    raises past the values `_probed` finds (Box-Cox's `x <= 0`, an
    unknown category)."""
    _GUARD.get().tests.append(test)


def handed() -> str:
    """How the twin hands the estimator now being translated its row
    (`HANDS`): a composition hands its parts theirs from it."""
    return _GUARD.get().hand


def handed_on() -> str:
    """How the estimator last translated (not a composition) hands on its
    output to a pipeline's next step: as objects where it returns its
    probe row as an object array (a selector handed objects keeps them),
    else as doubles."""
    return _GUARD.get().out


def translate_part(
    est: Any, x: list[S.Expr], types: list[pa.DataType], hand: str
) -> list[S.Expr]:
    """`est`'s output fields over `x`, a part of the composition now being
    translated, which the twin hands its row as `hand`. Its input guard's
    tests join the instance's."""
    if hand not in HANDS:
        raise ValueError(f"a hand of {hand!r}")
    entry = _CATALOG[type(est)]
    g = _GUARD.get()
    outer, g.hand = g.hand, hand
    try:
        if entry.composes:
            return list(entry.translate(est, x, types))
        reads = _apart(x, types) if entry.apart(est) else x
        start = len(g.tests)
        out = list(entry.translate(est, reads, types))
        g.tests[start:] = _plain(g.tests[start:], reads, x)
        g.tests += _probed(est, x, types, entry, g)
    finally:
        g.hand = outer
    return out


def _apart(x: list[S.Expr], types: list[pa.DataType]) -> list[S.Expr]:
    """`x` as a translator reads it: each number as `x - 0.0`, the same
    double (-0.0 - 0.0 is -0.0, NaN - 0.0 is NaN), which confit computes
    once a row where the function reads it twice or more outside any CASE
    arm, or twice in one arm (its shared values, `share.rs`). The input
    guard reads `coalesce(x, NaN)`, five nodes, under the six that confit
    shares, so each read computes it again, and where the guard and many
    sums read it, each read is slow (reading 7, finding 85). With confit
    alone, 64 sums of 256 products over such values served a row in 32 to
    33 µs beside the guard, 17 to 18 µs without it, and 19 to 22 µs where
    they read `x - 0.0`; `PCA` at 256 features of 64 components serves a
    row of a 1,024-row call in 20 to 30 µs, against 40 to 51 µs (release
    build, master 9f74bc0, 2026-10-16). Confit folds only constant
    subtrees, so it keeps the `- 0.0`. A translator whose reads of a
    feature each sit in a different CASE arm (a search tree) reads it
    twice in one arm, so that confit shares it (isotonic.py). A
    translator's own tests for the guard read each number as the guard
    does (`_plain`)."""
    return [
        xi if t == pa.string() else xi - f64(0.0)
        for xi, t in zip(x, types, strict=True)
    ]


def _plain(tests: list[S.Expr], reads: list[S.Expr], x: list[S.Expr]) -> list[S.Expr]:
    """`tests`, a translator's tests for its input guard (`rejects`), with
    each of its reads of a number (`_apart`) back as the guard reads it:
    the same double. confit computes a value it shares once a row, where
    the function first reads it, and carries it to its last reader. Read
    by the guard, ahead of the body, each feature's value rode through
    the whole body. A cubic spline of 128 features of 8 knots under
    `extrapolation="error"` built in 9.2 s, against 3.6 s with the
    guard's tests plain and 4.3 s where each number read `coalesce(x,
    NaN)` (master 9f74bc0's reads); a Box-Cox `PowerTransformer` of 128
    features in 0.70 s, against 0.26 s and 0.53 s (the trial build,
    `_builds`, warm; release build, master 12aca98, 2026-10-16). A row
    took about the same time either way, except in an ordinal encoder,
    whose chains then read the value the guard's test computed: it now
    tests its ends first (`encode._field`)."""
    swap = {id(r): xi for r, xi in zip(reads, x, strict=True) if r is not xi}
    if not swap:
        return tests
    new: dict[int, S.Expr] = {}
    for node in _post_order(tests):
        new[id(node)] = (
            swap[id(node)] if id(node) in swap else _with_children(node, new)
        )
    return [new[id(t)] for t in tests]


def _post_order(roots: list[S.Expr]) -> list[S.Expr]:
    """Every node under `roots` once, each after its children (no
    recursion: a test can be deep)."""
    seen: set[int] = set()
    out: list[S.Expr] = []
    for r in roots:
        if id(r) in seen:
            continue
        seen.add(id(r))
        stack = [(r, iter(r.children))]
        while stack:
            node, it = stack[-1]
            for c in it:
                if id(c) not in seen:
                    seen.add(id(c))
                    stack.append((c, iter(c.children)))
                    break
            else:
                stack.pop()
                out.append(node)
    return out


def _with_children(node: S.Expr, new: dict[int, S.Expr]) -> S.Expr:
    """`node` with each child `c` replaced by `new[id(c)]`, or `node`
    itself where no child changes."""
    changed = False

    def put(v: Any) -> Any:
        nonlocal changed
        if isinstance(v, S.Expr):
            w = new.get(id(v), v)
            changed |= w is not v
            return w
        if isinstance(v, tuple):
            return tuple(put(u) for u in v)
        return v

    fields = {f.name: put(getattr(node, f.name)) for f in dataclasses.fields(node)}
    return dataclasses.replace(node, **fields) if changed else node


def _base(types: list[pa.DataType]) -> list[Any]:
    """The default probe row: 1.0 in a number, False in a boolean, an
    empty string in a string."""
    return [
        "" if t == pa.string() else False if t == pa.bool_() else 1.0 for t in types
    ]


def _specials(t: pa.DataType) -> list[Any]:
    """The values of a feature of type `t` that a validation may reject: a
    string's NULL (None); a double's ±inf and NaN (NULL); an integer's or
    a boolean's NULL, which the step reads as NaN."""
    if t == pa.string():
        return [None]
    if t == pa.float64():
        return [math.inf, -math.inf, math.nan]
    return [math.nan]


def _container(hand: str, row: list[Any]) -> Any:
    """The one-row input of `transform` that `row`'s values make in `hand`."""
    if hand == "list":
        return [row]
    if hand == "objects":
        return np.array([row], dtype=object)
    if hand == "doubles":
        return np.array([row], dtype=np.float64)
    return np.asarray([row])  # "booleans": a boolean array, or float64 with NaN


def _answer(est: Any, arg: Any) -> np.ndarray | None:
    """`est.transform(arg)` as the step reads it (a sparse output
    densified), or None where it raises: whatever it raises, the step
    raises."""
    try:
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.simplefilter("ignore")
            return np.asarray(_dense(est.transform(arg)))
    except Exception:  # noqa: BLE001 — any error is the twin raising
        return None


def _raises(est: Any, arg: Any) -> bool:
    return _answer(est, arg) is None


def _probed(
    est: Any, x: list[S.Expr], types: list[pa.DataType], entry: Entry, g: _Guard
) -> list[S.Expr]:
    """The tests on `x` where `est`'s twin raises, by probing: handed the
    probe row with each feature in turn set to each of its `_specials`,
    as the twin hands it its row (`g.hand`).

    The validation of every catalog estimator raises per value (an
    infinity, a NaN), so one feature's probe answers for every row with
    that value there. One raise is per row: over boolean features only, a
    row none NULL is a boolean array, which some estimators reject
    whatever its values (`SimpleImputer(strategy="most_frequent")`); the
    probe row, then a boolean array, raising with each value flipped says
    so, and the test is that the row has no NULL. A probe row that raises
    otherwise leaves nothing to probe from: the step stays Python."""
    name = type(est).__name__
    base = list(entry.base(est, types)) if entry.base else _base(types)
    tests: list[S.Expr] = []
    got = _answer(est, _container(g.hand, base))
    g.out = "objects" if got is not None and got.dtype == object else "doubles"
    if got is None:
        boolean = (
            g.hand in ("list", "booleans")
            and bool(types)
            and all(t == pa.bool_() for t in types)
            and g.no_null is not None
        )
        flipped = [not v for v in base] if boolean else base
        if not boolean or not _raises(est, _container(g.hand, flipped)):
            raise NotNative(
                f"{name}: the twin raises on the input guard's probe row {base!r}"
            )
        tests.append(g.no_null)
    # One test for the features that reject the same values (`_holds`).
    kinds: dict[tuple[bool, tuple[str, ...]], tuple[pa.DataType, list[Any], list]] = {}
    for j, (xj, t) in enumerate(zip(x, types, strict=True)):
        hits = [
            v
            for v in _specials(t)
            if _raises(est, _container(g.hand, [*base[:j], v, *base[j + 1 :]]))
        ]
        if hits:
            kind = (t == pa.string(), tuple(sorted(map(repr, hits))))
            kinds.setdefault(kind, (t, hits, []))[2].append(xj)
    return tests + [_holds(xs, t, hits) for t, hits, xs in kinds.values()]


def _holds(xs: list[S.Expr], t: pa.DataType, values: list[Any]) -> S.Expr:
    """One of `xs`, features of type `t` (a string, or a number as a
    DOUBLE), holds one of `values`, their probe values. Over numbers, one
    `greatest` or `least` reads them all where it can, not a test each
    joined by OR: the widest bench `Pipeline` builds in 0.9 to 1.2 s with
    it, 2.4 to 2.7 s with a test each, and serves 64 rows in 5.4 ms
    against 6.4 ms (release build, master 9976d96)."""
    if t == pa.string():
        return any_of([x.isnull() for x in xs])
    pos, neg = math.inf in values, -math.inf in values
    nan = any(isinstance(v, float) and math.isnan(v) for v in values)
    if pos and neg and nan:
        # DuckDB orders NaN above every number: one comparison for the three.
        return _extreme("greatest", [S.fn("abs", x) for x in xs]) > f64(
            sys.float_info.max
        )
    if pos and neg and len(xs) > 1:
        # The least of the -|x| is -inf where one x is infinite. NaN, above
        # every number, is the least only where every x is NaN.
        return _extreme("least", [-S.fn("abs", x) for x in xs]) == f64(-math.inf)
    if nan and not (pos or neg):
        # The greatest is NaN where one x is.
        return isnan(_extreme("greatest", xs))
    return any_of([_holds_one(x, pos, neg, nan) for x in xs])


def _holds_one(x: S.Expr, pos: bool, neg: bool, nan: bool) -> S.Expr:
    """`x` is +inf where `pos`, -inf where `neg`, NaN where `nan`."""
    if pos and neg and not nan:
        return S.fn("abs", x) == f64(math.inf)
    terms = [
        *([x == f64(math.inf)] if pos else []),
        *([x == f64(-math.inf)] if neg else []),
        *([isnan(x)] if nan else []),
    ]
    return any_of(terms)


def _extreme(name: str, xs: list[S.Expr]) -> S.Expr:
    """`greatest` or `least` (`name`) of `xs`: the one, where it is one."""
    return xs[0] if len(xs) == 1 else S.fn(name, *xs)


def any_of(tests: list[S.Expr]) -> S.Expr:
    """The tests joined with OR."""
    return reduce(lambda a, b: a | b, tests)


def all_of(tests: list[S.Expr]) -> S.Expr:
    """The tests joined with AND."""
    return reduce(lambda a, b: a & b, tests)


def _feature(param: S.Expr, t: pa.DataType) -> S.Expr:
    """A declared feature as `PythonTransform` hands it to `transform`:
    numbers (and booleans, as 0/1) as DOUBLE with NULL read as NaN, a string
    as is."""
    if t == pa.string():
        return param
    # `SqlFunction` already cast the parameter to its declared type.
    x = param if t == pa.float64() else param.cast(pa.float64())
    return S.coalesce(x, S.lit(float("nan")))


def _output_fields(step: PythonTransform) -> list[tuple[str | None, pa.DataType]]:
    """The output fields, each `(field name, type)`; unnamed in a scalar
    or a fixed-size list return."""
    r = step.returns
    if pa.types.is_struct(r):
        return [(r.field(i).name, r.field(i).type) for i in range(r.num_fields)]
    if pa.types.is_fixed_size_list(r):
        return [(None, r.value_type)] * r.list_size
    return [(None, r)]


def _translate(step: Any, allow_bound: bool) -> SqlFunction:
    if not isinstance(step, PythonTransform):
        raise NotNative(f"{type(step).__name__} is not a PythonTransform")
    if not step.instances:
        raise NotNative("the step has no fitted instances")
    kinds = {type(e) for e in step.instances.values()}
    missing = sorted(k.__name__ for k in kinds if k not in _CATALOG)
    if missing:
        raise NotNative(f"no translation for {', '.join(missing)}")
    fields = _output_fields(step)
    types = list(step.takes.types)
    takes = pa.schema([pa.field(_ID, pa.int64()), *step.takes])

    def body(iid: S.Expr, *params: S.Expr) -> Any:
        feats = [_feature(p, t) for p, t in zip(params, types, strict=True)]
        no_null = None
        if types and all(t == pa.bool_() for t in types):
            no_null = all_of([~isnan(f) for f in feats])
        per_id: list[tuple[int, list[S.Expr]]] = []
        for k, est in sorted(step.instances.items()):
            token = _GUARD.set(_Guard("list", no_null))
            try:
                out = translate_part(est, feats, types, "list")
                tests = _GUARD.get().tests
            finally:
                _GUARD.reset(token)
            if len(out) != len(fields):
                raise NotNative(
                    f"instance {k}: {type(est).__name__} translates to"
                    f" {len(out)} output fields, the step declares {len(fields)}"
                )
            if tests:
                # The input guard, in the first output field: a read of any
                # output field fires it, as it fires the unknown-id trap
                # below.
                trap = S.fn("error", S.lit(_REJECTED.format(step.name)))
                out[0] = S.case(any_of(tests), trap).otherwise(out[0])
            per_id.append((k, out))
        # After the translators, so that a configuration they refuse says
        # why; `SqlFunction` calls this body before confit builds anything.
        why = None if allow_bound else _bounded(step)
        if why:
            raise NotNative(why)

        same = SameTree()

        def select(j: int) -> S.Expr:
            # Instances whose output field is the same tree (one instance, a
            # stateless estimator, or equal fits) share one arm.
            arms: dict[int, tuple[list[int], S.Expr]] = {}
            for k, out in per_id:
                key = same.key(out[j]) if len(per_id) > 1 else 0
                arms.setdefault(key, ([], out[j]))[0].append(k)
            (ks0, v0), *rest = arms.values()
            if j and not rest:
                # One arm needs no CASE past the first output field: a known
                # id answers it, an unknown one raises from the first output
                # field (below), and a NULL one is a NULL struct or list
                # (`null_when`). Without a CASE per output field, a step of
                # one instance builds 1.1-1.4x and serves 1.1-1.3x as fast
                # (release build, master 4c831d8).
                return v0

            def hit(ks: list[int]) -> S.Expr:
                return iid == S.lit(ks[0]) if len(ks) == 1 else iid.isin(*ks)

            e = S.case(hit(ks0), v0)
            for ks, v in rest:
                e = e.when(hit(ks), v)
            if j == 0:
                # An unknown id raises, from the first output field only:
                # DuckDB builds every field of a struct, so a read of any
                # output field fires it, and the other output fields carry
                # no trap that every field read would have to keep.
                e = e.when(iid.isnull(), S.lit(None, fields[0][1])).otherwise(
                    S.fn("error", S.lit(_UNKNOWN.format(step.name)))
                )
            return e

        if pa.types.is_struct(step.returns):
            return {name: select(j) for j, (name, _) in enumerate(fields)}
        if pa.types.is_fixed_size_list(step.returns):
            return [select(j) for j in range(len(fields))]
        return select(0)

    whole = pa.types.is_struct(step.returns) or pa.types.is_fixed_size_list(
        step.returns
    )
    try:
        fn = SqlFunction(
            step.name,
            takes,
            step.returns,
            body,
            # A NULL id is a NULL struct or list, not one of NULL output fields.
            null_when=(lambda iid, *_: iid.isnull()) if whole else None,
        )
    except RecursionError:
        # confit writes each value the body reads twice as SQL by recursion
        # (`sql.shared_texts`), so a value nested deeper than Python's
        # recursion limit raises here, before `_builds`: `Normalizer(l2)`
        # over 2,048 features, whose sum of squares follows numpy's chain
        # (finding 86). The step stays Python.
        raise NotNative(
            "confit does not build it: an expression nested past Python's"
            " recursion limit"
        ) from None
    _builds(step, fn)
    return fn


def query(step: Any, id_col: str = _ID) -> str:
    """The query reading every output field of one call of `step`: what
    `to_native` builds and `check` serves with both entries."""
    args = ", ".join([S.col(id_col).sql(), *(S.col(n).sql() for n in step.takes.names)])
    call = f"{step.name}({args})"
    r = step.returns
    if pa.types.is_struct(r):
        items = [
            f"{call}.{S.col(r.field(i).name).sql()} AS {S.col(r.field(i).name).sql()}"
            for i in range(r.num_fields)
        ]
    else:
        items = [f"{call} AS o"]
    return f"SELECT {', '.join(items)} FROM __THIS__"  # noqa: S608


def _builds(step: PythonTransform, fn: SqlFunction) -> None:
    """Raise `NotNative` unless confit builds `fn` into `query(step)`, so a
    translation it refuses (one too large to expand, say) leaves the step
    Python rather than failing where it is served."""
    schema = pa.schema([pa.field(_ID, pa.int64()), *step.takes])
    try:
        DuckDBInferFn(
            query(step, _ID),
            row_tables={"__THIS__": schema},
            static_tables={},
            udfs=[fn],
        )
    except (ValueError, RuntimeError) as e:
        # A refusal (ValueError), or a limit met while compiling, such as
        # Cranelift's function size (RuntimeError): the step stays Python.
        raise NotNative(f"confit does not build it: {e}") from None


def bound_of(est: Any) -> int:
    """One fitted estimator's ulp bound, by its class's entry."""
    return _CATALOG[type(est)].bound(est)


def bound(step: Any) -> int:
    """The ulp bound of a step's translation: the loosest of its
    instances' own bounds."""
    return max(bound_of(e) for e in step.instances.values())


def _loosest(step: PythonTransform) -> tuple[int, Any]:
    """The instance whose bound is the loosest, the first by id: one whose
    bound is an error scale or its parts' (a composition), else the one
    with the most ulps."""

    def looseness(ke: tuple[int, Any]) -> tuple[bool, int]:
        entry = _CATALOG[type(ke[1])]
        if entry.scale_for(ke[1]) is not None or entry.part_bounds(ke[1]) is not None:
            return (True, 0)
        return (False, entry.bound(ke[1]))

    return max(sorted(step.instances.items()), key=looseness)


def _bounded(step: PythonTransform) -> str | None:
    """Why `to_native` serves `step` only with `allow_bound`: its loosest
    instance (the first by id) is within a bound above 0. None when every
    instance is bit-exact."""
    k, est = _loosest(step)
    entry = _CATALOG[type(est)]
    if entry.exact(est):
        return None
    return (
        f"instance {k}: {type(est).__name__} is {entry.within(est)} of its"
        " twin, not bit-exact; to_native serves a bound above 0 only with"
        " allow_bound=True"
    )


def to_native(
    step: Any, *, strict: bool = False, allow_bound: bool = False
) -> Function | Any:
    """The native twin of `step`, or `step` itself when there is none.

    Bit-exact only, unless `allow_bound`: then a translation within its
    parity bound above 0 of the twin serves too (the module's docstring
    says why that is not the default). With `strict`, a step without a
    translation raises `NotNative` instead. Idempotent: a step that is
    already native comes back unchanged."""
    if isinstance(step, Function):
        return step
    try:
        return _translate(step, allow_bound)
    except NotNative:
        if strict:
            raise
        return step


def explain_native(step: Any, *, allow_bound: bool = False) -> str:
    """What `to_native(step, allow_bound=...)` returns, or why it returns
    the step."""
    if isinstance(step, Function):
        return f"{step!r} is already a confit function"
    try:
        fn = _translate(step, allow_bound)
    except NotNative as e:
        return f"{getattr(step, 'name', step)!r} stays Python: {e}"
    kinds = sorted({type(e).__name__ for e in step.instances.values()})
    _, est = _loosest(step)
    entry = _CATALOG[type(est)]
    within = "" if entry.exact(est) else f", {entry.within(est)}"
    return f"{fn.name!r}: {', '.join(kinds)} -> SqlFunction{within}"
