"""Splines: each feature's B-spline basis values, dense (sklearn 1.9,
`SplineTransformer.transform` in `sklearn/preprocessing/_polynomial.py`,
evaluating scipy 1.18's `BSpline.__call__`).

Per feature, `transform` evaluates the fitted `bsplines_[j]` (knots `t`,
coefficients `c`, the identity, and degree `k`) at the column:

- `continue`, `error`, `periodic`: `spl(x)`, where `periodic` first maps
  `x` to `t[k] + np.remainder(x - t[k], t[n] - t[k])` (zeros when that
  period is not positive). The splines are built with `extrapolate=True`
  for `continue` and `periodic`, False for `error`, whose NaN outputs past
  the knots then raise. NaN rows become 0.
- `constant`, `linear`: `spl(x)` where `t[k] <= x <= t[n]`, else 0; below
  and above, the first and last `degree` output fields take `spl(t[k])` and
  `spl(t[n])` (`constant`), or `f + (x - xb) * fp` with `fp` the derivative
  there (`linear`, `degree + 1` output fields when `degree <= 1`). NaN meets
  no test, so it gives 0. The twin reads that range as `t[degree]`,
  `t[-degree-1]`, and its `linear` branch raises its local `degree` inside
  the loop over features: at `degree <= 1`, the features after the first
  read a narrower range (inverted at `degree=1, n_knots=2`) and continue two
  output fields. The entry follows (`_spline`).

`spl(x)` is scipy's `_evaluate_spline` (`scipy/interpolate/src/
__fitpack.cc`, compiled C++): `_find_interval` picks the largest `l` in
`[k, n-1]` with `t[l] <= x` (`k` below the knots; so `x == t[n]` falls in
the last interval, closed), whichever interval it starts from; `_deBoor_D`
runs de Boor's recurrence over `t[l-k+1 .. l+k]` into `h[0..k]`; and output
field `j` is `0.0 + c[l-k, j]*h[0] + ... + c[l, j]*h[k]`, in that order.

The entry spells the same: per output field, a CASE over the intervals in
`_find_interval`'s order, each arm the recurrence's own operations on the
fitted knots (the knot differences folded as the C code computes them) and
the output field's sum in order. With `c` the identity, an output field's
sum is `0.0` plus one basis value (two under `periodic`, which folds the
last `degree` bases onto the first) plus `0.0 * h[a]` terms. The sum starts
at +0.0 and IEEE gives -0.0 only for -0.0 + -0.0, so it is never -0.0;
adding a `0.0 * h[a]`, a signed zero, then changes nothing as long as `h[a]`
is finite. Over an interval `x` is bounded in, the basis values are (they
lie in [0, 1] up to rounding, every divisor being a knot gap the `knots`
check below keeps away from the subnormals), so those terms are left out,
and an interval where the output field has no basis is `0.0`. Where `x` is
not bounded, the outer intervals of `continue`, the terms stay: there
`x = -1e300` can overflow a basis value, and `0.0 * inf` is the NaN the twin
answers. `0.0 + h` stays too: it is `h` except that it turns -0.0 into 0.0.

`periodic`'s `np.remainder(d, p)` is numpy's `npy_divmod`: `m = fmod(d, p)`
(DuckDB's `%` on DOUBLE; its `fmod` floors), then `m + p` when `m` and `p`
differ in sign, and `copysign(0, p)` when `m` is zero. `p > 0`, so: 0.0 when
`m == 0`, `m + p` when `m < 0`, else `m`; that is `m + p` or `m + 0.0`. At
±inf the remainder is NaN, and scipy answers NaN on every output field;
`handle_missing="zeros"` validates with infinity allowed, so the twin
answers it, and the entry tests for it before the intervals.

The boundary constants of `constant` and `linear` (`spl(t[k])`,
`spl(t[k], nu=1)`, ...) are the fitted splines' own numbers, computed with
them at translation time.

Where the validation of the twin raises, the entry traps: on a row past
the knots under `extrapolation="error"` (`rejects`), and on NaN under
`handle_missing="error"` (the input guard's probe finds it). Behind the
trap the output fields answer NaN past the knots, which is what scipy hands
sklearn before it raises, and 0.0 on NaN. Under `constant` with
`degree=0` above the knots the twin's slice assignment fails to
broadcast (or, with one output field, assigns nothing): an sklearn bug, not
validation, where the entry answers 0.0
(decisions/closed/tolerated-differences.md, the ruling lists it).

Sparse. Under `sparse_output=True` the twin builds each feature's block
with scipy's `BSpline.design_matrix`, which stores the recurrence's `k + 1`
basis values as they are, and the step makes the block dense
(`_udf._dense`): `0.0 + h[a]`, or under `periodic` the two folded bases'
sum, which the dense sum's order matches. So no output field keeps the
`0.0 * h` terms, where the dense twin keeps them: under `continue`, where a
basis value overflows past the knots, the dense twin answers NaN and the
sparse one the basis value. The step stays Python where the sparse twin
parts from the dense one otherwise (`_sparse_parts`): at degree 0 under
`constant` (above the knots it answers the basis at the last knot) and
`periodic` (it raises on every row); under `handle_missing="zeros"` over
two or more features with `continue`, `error` or `periodic` (sklearn sets a
NaN's output fields to 0.0 in the first feature only, and the others read
the basis at 0.0); and over knots that are NaN (the design matrix raises).
Where the sparse twin raises on ±inf (`handle_missing="zeros"`), the input
guard's probe finds it. The sweep that found these parts compared the two
twins a row a call on 3,840 configurations (degrees 0 to 5, 2 to 6 knots,
the five extrapolations, `include_bias`, both knot strategies, both
`handle_missing`, 1 and 3 features) at every knot and its neighbours, ±0,
±1e12, ±1e155, ±1e300, ±inf and NaN: no other row parted (2026-10-14).

Widths. The recurrence reads each round's values twice, so an output field's
text doubles per degree (about 3 KB of SQL per output field at degree 3, 8
knots, 33 KB at degree 5); `continue` keeps three times that (its outer
sums), and `periodic` repeats its mapped `x` at every read. confit binds once
a value that a SQL function body reads more than once (#412), so the build
follows the distinct nodes, not the text. Release build, master 318575b with
confit #459 and #461, one instance, a struct return, 10,000-row batches, 32
features at degree 3 and 8 knots (320 output fields; `periodic` 224):
`constant`, `linear` and `error` build in 0.7 s and serve 94-99 us per row
(the input guard of `error` tests the knots); `continue` 0.85 s and 71 us;
`periodic` 0.6 s and 33 us; the twin serves the `continue` step in 894 us per
row. 64 features of `continue`: 1.8 s and 154 us; 128 features: 4.5 s and 335
us; 16 features of degree 5, 7 knots, `continue`: 0.6 s and 37 us; 64
features of degree 5, 8 knots, `continue`: 4.0 s and 229 us; 128 features:
9.2 s and 508 us, with the cap lifted (the entry refuses it). One feature at
degree 3, 5 knots: 0.01 s and 0.3 us, against the twin's 135 us
(2026-10-15). The entry refuses a step
whose estimated build passes MAX_BUILD_S (`_build_estimate`), as quantile.py
and isotonic.py cap theirs. Up to 64 features it takes on every configuration
measured (degrees 1, 3 and 5, 5 and 8 knots, the five extrapolations). Of 128
features it refuses 4, which build in 7.7 to 9.4 s (degree 5 at 8 knots), and
of 192 and 256 features 11 of the 12 measured, which build in 6.3 to 16.3 s,
before confit is asked. confit builds every configuration measured but the
struct return of 256 features of degree 5 at 8 knots, past 4,000,000 tokens;
before #430 it refused four past Cranelift's size limit (128 features of
degree 3 to 5 at 8 knots, `error`).

Comparisons, constants and DOUBLE arithmetic in scipy's order, so the
entry is bit-exact: 8 seeds in the gate, and 200 seeds of each of the 16
fixture configurations, 3,200 steps, with none apart (2026-10-05; again
at every width the generator draws, up to 32 features, and with ±inf
among the rows, 2026-10-06). The
x86-64 build of scipy does not contract `h += w*(xb - x)` into a fused
multiply-add; a build that does (aarch64) would part from it, and
`bspline_is_scipys` probes for that on first use, the entry refusing where
it finds one.
"""

from __future__ import annotations

import functools
import math
from typing import Any

import numpy as np
import pyarrow as pa
from confit import sql as S
from sklearn.preprocessing import SplineTransformer

from sql_transform.native._helpers import SameTree, f64, isnan
from sql_transform.native._registry import NotNative, rejects, translates

# The longest build the entry takes on (seconds, warm, release), as
# quantile.py's and isotonic.py's caps.
MAX_BUILD_S = 7.0


def _build_estimate(fields: list[S.Expr], est: Any) -> float:
    """The seconds confit takes to build these output fields (warm, release
    build) into the query that reads each output field. confit binds once a
    value that the body reads more than once (#412), so the build follows
    the distinct nodes the output fields read, not the text they spell, plus
    a term in the nodes times the output fields. The input guard adds a term
    in the output fields times the features, a tenth to a fifth of what it
    added before confit #430 stopped building the guard's tests again at
    each field read. The guard has a test a feature on NaN and ±inf under
    `handle_missing="error"`; under `extrapolation="error"`, a test on the
    knots, whose term is about 1.6 times as large, with or without the
    first. Fitted on the slower of a struct and a list return over 114 warm
    builds of one instance over 0.3 s (8 to 256 features, degrees 1, 3 and
    5, 5 and 8 knots, the five extrapolations, up to 2,560 output fields;
    15 of them again under `handle_missing="zeros"`; 0.3 to 16.3 s): each
    within 0.79 to 1.20 times the estimate (master 318575b with confit #459
    and #461, 2026-10-15). The cap took on none slower than 6.32 s and
    refused none faster than 6.25 s, and it reaches 7 s at about 170 to 230
    features of degree 3 and 8 knots, or 100 to 140 of degree 5, by the
    extrapolation (150 to 200, and 86 to 120, fitted on master b4b8d03 with
    confit #430). Past the estimate, confit refuses the struct return of
    256 features of degree 5 and 8 knots (past 4,000,000 tokens). The
    estimate is one estimator's: a step's instances compound it linearly."""
    seen: set[int] = set()
    stack = list(fields)
    while stack:  # no recursion: a recurrence can be deep
        e = stack.pop()
        if id(e) not in seen:
            seen.add(id(e))
            stack.extend(e.children)
    nodes, width = len(seen), len(fields)
    if est.extrapolation == "error":
        guard = 5.66e-6
    elif est.handle_missing == "error":
        guard = 3.64e-6
    else:
        guard = 0.0
    return (
        2.94e-5 * nodes + 9.12e-9 * nodes * width + guard * width * est.n_features_in_
    )


# A knot gap under this, but not zero, could overflow `1 / gap` in the
# recurrence, so the intervals are no longer known to give finite basis
# values: every output field then keeps its whole sum.
_TINY_GAP = 1e-300

Val = float | S.Expr


def _lift(v: Val) -> S.Expr:
    return f64(v) if isinstance(v, float) else v


def _add(a: Val, b: Val) -> Val:
    # Two fitted numbers fold as the C code adds them: Python's float is
    # the same IEEE double.
    if isinstance(a, float) and isinstance(b, float):
        return a + b
    return _lift(a) + _lift(b)


def _sub(a: Val, b: Val) -> Val:
    if isinstance(a, float) and isinstance(b, float):
        return a - b
    return _lift(a) - _lift(b)


def _mul(a: Val, b: Val) -> Val:
    if isinstance(a, float) and isinstance(b, float):
        return a * b
    return _lift(a) * _lift(b)


def _div(a: Val, b: Val) -> Val:
    if isinstance(a, float) and isinstance(b, float):
        return a / b
    return _lift(a) / _lift(b)


def _de_boor(t: list[float], k: int, ell: int, x: S.Expr) -> list[Val]:
    """`_deBoor_D(t, x, k, ell, 0, h)`: the `k + 1` basis values non-zero
    on `[t[ell], t[ell+1])`, in its operations and order. `hh` is the
    previous round's copy; `h[n-1]` already holds this round's update."""
    h: list[Val] = [1.0] + [0.0] * k
    for j in range(1, k + 1):
        hh = h[:j]
        h[0] = 0.0
        for n in range(1, j + 1):
            xb, xa = t[ell + n], t[ell + n - j]
            if xb == xa:
                h[n] = 0.0
                continue
            w = _div(hh[n - 1], xb - xa)
            h[n - 1] = _add(h[n - 1], _mul(w, _sub(xb, x)))
            h[n] = _mul(w, _sub(x, xa))
    return h


def _field_sum(h: list[Val], col: list[float], bounded: bool) -> S.Expr:
    """`out = 0.0; out += c[l-k+a, j] * h[a]` for `a = 0..k`, over the
    output field's column of `c` at this interval (`col[a]`). A term
    `1.0 * h` is `h`; a term `0.0 * h` is left out where `h` is finite
    (`bounded`), as the module says."""
    acc: Val = 0.0
    for cv, ha in zip(col, h, strict=True):
        if cv == 1.0:
            acc = _add(acc, ha)
        elif cv == 0.0 and bounded:
            continue
        else:
            acc = _add(acc, _mul(cv, ha))
    return _lift(acc)


def _spline_py(t: list[float], k: int, c: np.ndarray, x: float) -> list[float]:
    """`_evaluate_spline` at one `x`, on Python floats (the same IEEE
    doubles, unfused): `_find_interval`, `_de_boor`, then every output
    field's whole sum in order."""
    n = len(t) - k - 1
    ell = k
    while ell < n - 1 and t[ell + 1] <= x:
        ell += 1
    h = _de_boor(t, k, ell, x)  # type: ignore[arg-type]
    out = []
    for j in range(c.shape[1]):
        acc = 0.0
        for a in range(k + 1):
            acc = acc + float(c[ell + a - k, j]) * float(h[a])
        out.append(acc)
    return out


@functools.cache
def bspline_is_scipys() -> bool:
    """Whether this platform's scipy `BSpline` answers as the recurrence
    the entry spells, bit for bit: degrees 0 to 5, uneven knots with runs of
    equal ones, points on and beside every knot, between them and past
    them. A scipy build that contracts `h += w*(xb - x)` into a fused
    multiply-add (aarch64) parts from it."""
    from scipy.interpolate import BSpline

    rng = np.random.default_rng(20261005)
    for k in range(6):
        for _ in range(6):
            m = int(rng.integers(2, 9))
            base = np.sort(rng.normal(size=m) * 10.0 ** rng.integers(-3, 4))
            base[rng.random(m) < 0.2] = base[0]
            base = np.sort(base)
            gap = float(base[-1] - base[0]) or 1.0
            lo = np.sort(base[0] - gap * rng.random(k))
            hi = np.sort(base[-1] + gap * rng.random(k))
            t = np.concatenate([lo, base, hi])
            if len(t) - k - 1 <= k:
                continue
            c = np.eye(len(t) - k - 1)
            spl = BSpline.construct_fast(t, c, k, extrapolate=True)
            xs = [
                w
                for v in t
                for w in (np.nextafter(v, -np.inf), v, np.nextafter(v, np.inf))
            ]
            xs += list(rng.uniform(t[0] - gap, t[-1] + gap, size=40))
            xs = [float(v) for v in xs]
            tl = [float(v) for v in t]
            got = spl(np.array(xs))
            for i, v in enumerate(xs):
                want = _spline_py(tl, k, c, v)
                if [repr(float(g)) for g in got[i]] != [repr(w) for w in want]:
                    return False
    return True


def _remainder(d: S.Expr, p: float) -> S.Expr:
    """`np.remainder(d, p)` for a period `p > 0` (the module says how),
    spelled `m + (p if m < 0 else 0.0)`: `m + 0.0` is `m` but for a -0.0,
    which it makes the 0.0 numpy answers for either zero. Every read of
    the mapped `x` repeats this text, so `m` is read twice, not three
    times."""
    m = d % f64(p)
    return m + S.case(m < f64(0.0), f64(p)).otherwise(f64(0.0))


def _arms(
    t: list[float], k: int, c: np.ndarray, x: S.Expr, unbounded: set[int], every: bool
) -> list[list[tuple[float | None, S.Expr]]]:
    """Per output field, `_find_interval`'s arms in order: `(upper, value)`,
    the arm taken when `x < upper` (None: the last, closed one).
    Neighbouring arms with the same value (the same tree, `SameTree`) merge:
    the conditions are cumulative."""
    n = len(t) - k - 1
    fields: list[list[tuple[float | None, S.Expr]]] = [[] for _ in range(c.shape[1])]
    same = SameTree()
    for ell in range(k, n):
        h = _de_boor(t, k, ell, x)
        upper = t[ell + 1] if ell < n - 1 else None
        bounded = not every and ell not in unbounded
        for j, arms in enumerate(fields):
            col = [float(c[ell + a - k, j]) for a in range(k + 1)]
            v = _field_sum(h, col, bounded)
            if arms and same.key(arms[-1][1]) == same.key(v):
                arms[-1] = (upper, v)
            else:
                arms.append((upper, v))
    return fields


def _field(
    head: list[tuple[S.Expr, S.Expr]], arms: list[tuple[float | None, S.Expr]], x
) -> S.Expr:
    """`CASE <head arms> WHEN x < t[k+1] THEN ... ELSE <last arm> END`,
    the interval arms reading `x` (the mapped `x`, under `periodic`)."""
    conds = [*head, *((x < f64(u), v) for u, v in arms[:-1])]
    last = arms[-1][1]
    if not conds:
        return last
    (c0, v0), *rest = conds
    e = S.case(c0, v0)
    for cond, v in rest:
        e = e.when(cond, v)
    return e.otherwise(last)


def _knots(spl: Any, j: int) -> tuple[list[float], int, np.ndarray]:
    t = np.asarray(spl.t, dtype=np.float64)
    k = int(spl.k)
    c = np.asarray(spl.c, dtype=np.float64)
    n = len(t) - k - 1
    if c.ndim != 2 or c.shape[0] != n or n <= k:
        raise NotNative(f"SplineTransformer: feature {j}'s spline is not sklearn's")
    if not np.isfinite(t).all() or not math.isfinite(float(t[-1] - t[0])):
        # Past the doubles, a knot gap is infinite and so is the arithmetic
        # the bounded intervals rely on being finite.
        raise NotNative(f"SplineTransformer: feature {j}'s knots span past DOUBLE")
    if (np.diff(t) < 0).any():
        raise NotNative(f"SplineTransformer: feature {j}'s knots are not sorted")
    if not np.isfinite(c).all():
        raise NotNative(f"SplineTransformer: feature {j}'s coefficients")
    return [float(v) for v in t], k, c


def _unknotted(est: Any, spl: Any, x: S.Expr) -> list[S.Expr]:
    """A feature whose knots are all NaN (`knots="quantile"` fit on a column
    only missing). Every C comparison with a NaN knot is false, so
    `_find_interval` answers `k` and the recurrence runs on NaN gaps:
    `spl(x)` is its arm `k`, with the whole sums. `periodic` reads its
    period as not positive and evaluates at 0.0; `constant` and `linear`
    find no row inside, below or above, so every output field is 0.0. NaN
    rows are 0.0 throughout."""
    t = [float(v) for v in spl.t]
    k = int(spl.k)
    c = np.asarray(spl.c, dtype=np.float64)
    zero = f64(0.0)
    if est.extrapolation in ("constant", "linear"):
        return [zero] * c.shape[1]
    xv = zero if est.extrapolation == "periodic" else x
    h = _de_boor(t, k, k, xv)
    return [
        S.case(isnan(x), zero).otherwise(
            _field_sum(h, [float(c[a, j]) for a in range(k + 1)], bounded=False)
        )
        for j in range(c.shape[1])
    ]


def _bare(x: S.Expr) -> S.Expr:
    """The feature without its NULL-as-NaN wrapper (`coalesce(p, NaN)`,
    `_registry._feature`): past an output field's first arm, which answers
    NaN and NULL, the two are the same value, and `p` is a fraction of the
    text that every one of an output field's many reads of `x` repeats."""
    if (
        isinstance(x, S.Call)
        and x.name.lower() == "coalesce"
        and len(x.args) == 2
        and isinstance(x.args[1], S.Const)
        and isinstance(x.args[1].value, float)
        and math.isnan(x.args[1].value)
    ):
        return x.args[0]
    return x


def _feature(est: Any, spl: Any, j: int, x: S.Expr, degree: int) -> list[S.Expr]:
    if np.isnan(np.asarray(spl.t, dtype=np.float64)).all():
        if est.sparse_output:
            raise NotNative(
                f"SplineTransformer(sparse_output=True): feature {j}'s knots are"
                " NaN, where scipy's design matrix raises on every row"
            )
        if est.extrapolation == "error":
            raise NotNative(
                f"SplineTransformer(extrapolation='error'): feature {j}'s knots"
                " are NaN, where the twin raises on every number"
            )
        return _unknotted(est, spl, x)
    t, k, c = _knots(spl, j)
    nan = isnan(x)
    if est.extrapolation == "error":
        # Past the knots scipy answers NaN, and the twin raises; NaN is
        # not past them (`handle_missing="zeros"` answers 0.0 there), but
        # DuckDB orders it above every number.
        lo, hi = t[k], t[len(t) - k - 1]
        rejects((x < f64(lo)) | ((x > f64(hi)) & ~nan))
    x = _bare(x)
    n = len(t) - k - 1
    n_splines = c.shape[1]
    gaps = np.diff(np.asarray(t))
    # A sparse output stores the basis values alone: no `0.0 * h` terms.
    every = bool(((gaps > 0) & (gaps < _TINY_GAP)).any()) and not est.sparse_output
    ext = est.extrapolation
    lo, hi = t[k], t[n]
    zero = f64(0.0)

    if ext == "periodic":
        period = hi - lo
        xv: S.Expr = f64(lo) + _remainder(x - f64(lo), period) if period > 0 else zero
        head = [(nan, zero)]
        if period > 0:
            # The remainder of ±inf is NaN, which scipy answers with NaN on
            # every output field; the arms would take it to the last
            # interval, where an output field with no basis there is 0.0
            # (`handle_missing="zeros"` validates with infinity allowed, so
            # the twin answers it).
            head.append((S.fn("abs", x) == f64(math.inf), f64(math.nan)))
        fields = _arms(t, k, c, xv, set(), every)
        return [_field(head, arms, xv) for arms in fields]

    if ext in ("continue", "error"):
        unbounded = {k, n - 1} if ext == "continue" and not est.sparse_output else set()
        fields = _arms(t, k, c, x, unbounded, every)
        head = [(nan, zero)]
        if ext == "error":
            # Past the knots scipy answers NaN, and the twin raises: the
            # input guard traps there (above), and the output fields answer
            # NaN.
            out = f64(math.nan)
            head += [(x < f64(lo), out), (x > f64(hi), out)]
        return [_field(head, arms, x) for arms in fields]

    if ext not in ("constant", "linear"):
        raise NotNative(f"SplineTransformer(extrapolation={ext!r})")
    # The twin's range and boundary values read its running `degree`
    # (`_spline`): `t[degree]` and `t[-degree-1]`, inside `[t[k], t[n]]`
    # since `degree >= k`, so `spl` evaluates inside its knots there.
    lo, hi = t[degree], t[len(t) - degree - 1]
    fields = _arms(t, k, c, x, set(), every)
    f_min = [float(v) for v in spl(lo)]
    f_max = [float(v) for v in spl(hi)]
    # Per output field, what a row below and a row above it get, and, when
    # the range is inverted (`lo > hi`, a row then both), which comes last.
    below: list[S.Expr] = [zero] * n_splines
    above: list[S.Expr] = [zero] * n_splines
    both: list[S.Expr] = [zero] * n_splines
    if ext == "constant":
        # `XBS[below, :degree] = f_min[:degree]`, and the last `degree`
        # output fields above; at degree 0 the twin's slice is empty (it
        # raises, or with one output field assigns nothing), so 0.0 stays.
        for i in range(degree):
            below[i] = f64(f_min[i])
            above[n_splines - degree + i] = f64(f_max[n_splines - degree + i])
    else:
        fp_min = [float(v) for v in spl(lo, nu=1)]
        fp_max = [float(v) for v in spl(hi, nu=1)]
        lines = degree + 1 if degree <= 1 else degree
        if lines > n_splines:
            # Its step `j` indexes `f_min[j]` past the output fields (an
            # IndexError on a row below) and writes a row above into the
            # previous feature's output fields (`n_splines - 1 - j < 0`).
            raise NotNative(
                f"SplineTransformer(extrapolation='linear'): feature {j}"
                f" continues {lines} output fields of {n_splines}, past its own"
            )
        # Step `i` writes output field `i` below, then output field
        # `n_splines - 1 - i` above; a row in both keeps the later write.
        last: dict[int, str] = {}
        for i in range(lines):
            below[i] = f64(f_min[i]) + (x - f64(lo)) * f64(fp_min[i])
            last[i] = "below"
            r = n_splines - 1 - i
            above[r] = f64(f_max[r]) + (x - f64(hi)) * f64(fp_max[r])
            last[r] = "above"
        for r, side in last.items():
            both[r] = below[r] if side == "below" else above[r]
    if lo > hi:
        # No row is inside; NaN aside, every row is below, above or both.
        return [
            S.case(nan, zero)
            .when((x < f64(lo)) & (x > f64(hi)), w)
            .when(x < f64(lo), b)
            .when(x > f64(hi), a)
            .otherwise(zero)
            for b, a, w in zip(below, above, both, strict=True)
        ]
    return [
        _field([(nan, zero), (x < f64(lo), b), (x > f64(hi), a)], arms, x)
        for arms, b, a in zip(fields, below, above, strict=True)
    ]


def _sparse_parts(est: Any) -> None:
    """`NotNative` where the twin of `sparse_output=True` answers other
    than the dense twin the entry spells (module docstring, "Sparse")."""
    name = f"SplineTransformer(sparse_output=True, degree={est.degree}"
    ext = est.extrapolation
    if est.degree == 0 and ext == "constant":
        raise NotNative(
            f"{name}, extrapolation='constant'): above the knots the twin"
            " answers the basis at the last knot, where the entry answers 0.0"
        )
    if est.degree == 0 and ext == "periodic":
        raise NotNative(
            f"{name}, extrapolation='periodic'): the twin raises on every row"
        )
    if (
        est.handle_missing == "zeros"
        and ext in ("continue", "error", "periodic")
        and est.n_features_in_ > 1
    ):
        raise NotNative(
            f"{name}, extrapolation={ext!r}, handle_missing='zeros'): the twin"
            " answers 0.0 for a NaN in the first feature only"
        )


def _inside(est: Any, types: list[pa.DataType]) -> list[Any]:
    """The input guard's probe row: each feature at its first inner knot,
    `t[k]`, inside the knots `extrapolation="error"` raises past (1.0
    where the knots are NaN)."""
    row = []
    for spl in est.bsplines_:
        lo = float(spl.t[spl.k])
        row.append(1.0 if math.isnan(lo) else lo)
    return row


@translates(SplineTransformer, base=_inside)
def _spline(est: Any, x: list[S.Expr], types: list[pa.DataType]) -> list[S.Expr]:
    if est.sparse_output:
        _sparse_parts(est)
    if not bspline_is_scipys():
        raise NotNative(
            "SplineTransformer: this platform's scipy BSpline does not round as"
            " de Boor's recurrence unfused, which the entry follows"
            " (spline.bspline_is_scipys)"
        )
    out: list[S.Expr] = []
    # The twin's `linear` branch raises its local `degree` by one when it
    # is at most 1, inside the loop over features and never back; the
    # next feature then reads its range at the raised degree too. The
    # entry hands each feature the count the twin has when it starts it.
    degree = est.degree
    for j, xj in enumerate(x):
        fields = _feature(est, est.bsplines_[j], j, xj, degree)
        if est.extrapolation == "linear" and degree <= 1:
            degree += 1
        out += fields if est.include_bias else fields[:-1]
    estimate = _build_estimate(out, est)
    if estimate > MAX_BUILD_S:
        raise NotNative(
            f"SplineTransformer: an estimated {estimate:.0f} s build, past"
            f" {MAX_BUILD_S:.0f} s"
        )
    if len(out) != est.n_features_out_:
        raise NotNative(
            f"SplineTransformer: {len(out)} output fields where sklearn counts"
            f" {est.n_features_out_}"
        )
    return out
