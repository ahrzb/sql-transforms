"""Value semantics for the codegen engine — the DataFusion-parity helpers that
generated code calls.

Mirrors src/expr.rs. Generated code never uses bare Python operators, because
Python diverges from the Rust engine in ways that are silent rather than loud:
integer division floors instead of truncating, `%` takes the divisor's sign,
`bool` is an `int` subclass, float division by zero raises instead of yielding
IEEE inf/nan, and `1 == 1.0`. Every such case is contained in this module.

NULL is Python None throughout. Type tests are `type(v) is X` on purpose --
`isinstance` would let `True` pass as an int, which Rust's Value never does.
"""

from __future__ import annotations

import math
import re
from typing import Any

# DataFusion str->int accepts only an optional sign and digits.
_INT_STR = re.compile(r"[+-]?[0-9]+")


def type_name(v: Any) -> str:
    """Human-readable type name, matching expr::type_name for error messages."""
    if v is None:
        return "null"
    t = type(v)
    if t is bool:
        return "bool"
    if t is int:
        return "int"
    if t is float:
        return "float"
    if t is str:
        return "string"
    return "object"


def getfield(v: Any, name: str) -> Any:
    """Struct field access: a NULL struct yields NULL; a missing field is an
    error (mirrors native FieldAccess in expr.rs)."""
    if v is None:
        return None
    if isinstance(v, dict) and name in v:
        return v[name]
    raise ValueError(f"cannot access field {name!r} on a {type_name(v)} value")


def unnest_rows(v: Any, name: str) -> list:
    """One binding per list element for the Unnest loop. An empty list yields
    zero rows, and so does a NULL list -- the input row disappears rather than
    producing a NULL (matches DataFusion / native execute_rel)."""
    if v is None:
        return []
    if not isinstance(v, list):
        raise ValueError(f"unnest() expected a list, got a {type_name(v)} value")
    return [{name: item} for item in v]


def _fmt_float(f: float) -> str:
    """Render a float the way DataFusion does (measured 2026-07-17).

    DataFusion uses fixed notation for exponents in [-5, 15] and exponential
    outside it, always with a bare exponent (no leading zero, no '+'):
        1.0 -> "1.0"      0.1 -> "0.1"        1e-5 -> "0.00001"
        1e-6 -> "1e-6"    1e16 -> "1e16"      NaN/inf -> "NaN"/"inf"/"-inf"
    Python's repr() gives the same shortest round-trip digits but diverges in
    two spots: it switches to exponential one decade earlier (1e-5 -> "1e-05")
    and pads/​signs the exponent ("1e-06", "1e+16"). We fix up only those.
    """
    if math.isnan(f):
        return "NaN"
    if math.isinf(f):
        return "inf" if f > 0 else "-inf"
    s = repr(f)  # shortest round-trip
    if "e" not in s:
        return s  # already fixed notation; DataFusion agrees
    mant, exp_str = s.split("e")
    exp = int(exp_str)
    if exp == -5:
        # The one decade (1e-5..1e-4) where Python uses exp but DataFusion fixed.
        sign = "-" if mant.startswith("-") else ""
        intpart, _, frac = mant.lstrip("-").partition(".")
        return f"{sign}0.{'0' * (5 - len(intpart))}{intpart}{frac}"
    return f"{mant}e{exp}"  # bare exponent


def display(v: Any) -> str:
    """String form used by CONCAT and CAST(.. AS VARCHAR).

    Matches DataFusion, NOT Rust's expr::display_value -- see _fmt_float.
    """
    if v is None:
        return ""
    t = type(v)
    if t is bool:
        return "true" if v else "false"
    if t is str:
        return v
    if t is int:
        return str(v)
    if t is float:
        return _fmt_float(v)
    return "<object>"


def tag(v: Any) -> int:
    """Variant tag mirroring Value's Eq/Hash: Int(1) != Float(1.0), True != 1."""
    if v is None:
        return 4
    t = type(v)
    if t is bool:
        return 3
    if t is int:
        return 0
    if t is float:
        return 1
    if t is str:
        return 2
    return 5


def key(v: Any) -> tuple:
    """Hashable, type-tagged key component for lookup-join indexes."""
    return (tag(v), v)


def val_eq(a: Any, b: Any) -> bool:
    """Value-level equality: type-strict, like Value's PartialEq."""
    return tag(a) == tag(b) and a == b


def truthy(v: Any) -> bool:
    """RelNode::Filter keeps a row only when the predicate is Value::Bool(true)."""
    return v is True


def as_f(v: Any) -> float:
    t = type(v)
    if t is int:
        return float(v)
    if t is float:
        return v
    raise ValueError(f"Cannot use a {type_name(v)} value in an arithmetic expression")


def _trunc_div(a: int, b: int) -> int:
    q = abs(a) // abs(b)
    return -q if (a < 0) != (b < 0) else q


def _float_div(a: float, b: float) -> float:
    if b == 0.0:
        if a == 0.0 or math.isnan(a):
            return math.nan
        return math.copysign(math.inf, a) * math.copysign(1.0, b)
    return a / b


def _wrap_i64(x: int) -> int:
    """Wrap a Python bigint into i64 two's-complement range, like DataFusion's
    integer arithmetic (measured: 9223372036854775807 * 2 -> -2). Python ints
    never overflow, so without this codegen diverges at the i64 boundary."""
    return (x + 0x8000000000000000) % 0x10000000000000000 - 0x8000000000000000


def add(l: Any, r: Any) -> Any:  # noqa: E741
    if l is None or r is None:
        return None
    if type(l) is int and type(r) is int:
        return _wrap_i64(l + r)
    return as_f(l) + as_f(r)


def sub(l: Any, r: Any) -> Any:  # noqa: E741
    if l is None or r is None:
        return None
    if type(l) is int and type(r) is int:
        return _wrap_i64(l - r)
    return as_f(l) - as_f(r)


def mul(l: Any, r: Any) -> Any:  # noqa: E741
    if l is None or r is None:
        return None
    if type(l) is int and type(r) is int:
        return _wrap_i64(l * r)
    return as_f(l) * as_f(r)


def div(l: Any, r: Any) -> Any:  # noqa: E741
    if l is None or r is None:
        return None
    if type(l) is int and type(r) is int:
        if r == 0:
            raise ValueError("division by zero")
        return _wrap_i64(_trunc_div(l, r))  # i64_MIN / -1 overflows
    return _float_div(as_f(l), as_f(r))


def mod(l: Any, r: Any) -> Any:  # noqa: E741
    if l is None or r is None:
        return None
    if type(l) is int and type(r) is int:
        if r == 0:
            raise ValueError("division by zero")
        return l - r * _trunc_div(l, r)
    a, b = as_f(l), as_f(r)
    if b == 0.0:
        return math.nan
    return math.fmod(a, b)


def _cmp(l: Any, r: Any) -> int:  # noqa: E741
    """Ordering mirroring expr::compare_values: same-type int/str/bool compare
    directly, everything else goes through as_f (which errors on non-numbers)."""
    tl, tr = type(l), type(r)
    if tl is tr and tl in (int, str, bool):
        return -1 if l < r else (0 if l == r else 1)
    a, b = as_f(l), as_f(r)
    a_nan, b_nan = math.isnan(a), math.isnan(b)
    if a_nan or b_nan:
        # DataFusion total order (measured): NaN == NaN, and NaN sorts BELOW
        # every non-NaN value, including -inf.
        if a_nan and b_nan:
            return 0
        return -1 if a_nan else 1
    return -1 if a < b else (0 if a == b else 1)


def _veq(a: Any, b: Any) -> bool:
    """Type-tagged structural equality mirroring native Value::PartialEq: dicts
    (structs) by key order + values, lists elementwise, scalars by variant tag +
    value (Int(1) != Float(1.0), via val_eq)."""
    if isinstance(a, dict) and isinstance(b, dict):
        return list(a.keys()) == list(b.keys()) and all(_veq(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_veq(x, y) for x, y in zip(a, b, strict=True))
    if isinstance(a, (dict, list)) or isinstance(b, (dict, list)):
        return False  # container vs scalar (or struct vs list) never equal
    return val_eq(a, b)


def eq(l: Any, r: Any) -> Any:  # noqa: E741
    if l is None or r is None:
        return None
    if isinstance(l, (dict, list)) or isinstance(r, (dict, list)):
        return _veq(l, r)
    return _cmp(l, r) == 0


def neq(l: Any, r: Any) -> Any:  # noqa: E741
    if l is None or r is None:
        return None
    if isinstance(l, (dict, list)) or isinstance(r, (dict, list)):
        return not _veq(l, r)
    return _cmp(l, r) != 0


def lt(l: Any, r: Any) -> Any:  # noqa: E741
    return None if l is None or r is None else _cmp(l, r) < 0


def gt(l: Any, r: Any) -> Any:  # noqa: E741
    return None if l is None or r is None else _cmp(l, r) > 0


def lte(l: Any, r: Any) -> Any:  # noqa: E741
    return None if l is None or r is None else _cmp(l, r) <= 0


def gte(l: Any, r: Any) -> Any:  # noqa: E741
    return None if l is None or r is None else _cmp(l, r) >= 0


def _tribool(v: Any) -> bool | None:
    if v is True:
        return True
    if v is False:
        return False
    if v is None:
        return None
    raise ValueError(f"Expected a boolean expression, got a {type_name(v)} value")


def and_(l: Any, r: Any) -> Any:  # noqa: E741
    # Both operands are converted before matching (expr::logic), so a non-bool
    # errors even when the other operand would decide the result. No short-circuit.
    lb, rb = _tribool(l), _tribool(r)
    if lb is False or rb is False:
        return False
    if lb is True and rb is True:
        return True
    return None


def or_(l: Any, r: Any) -> Any:  # noqa: E741
    lb, rb = _tribool(l), _tribool(r)
    if lb is True or rb is True:
        return True
    if lb is False and rb is False:
        return False
    return None


def not_(v: Any) -> Any:
    b = _tribool(v)
    return None if b is None else (not b)


def join_eq(a: Any, b: Any) -> bool:
    """JOIN ON equality: a NULL on either side never matches, and equality is
    type-strict (RelNode::Join compares Values, not Python numbers)."""
    if a is None or b is None:
        return False
    return val_eq(a, b)


def as_s(v: Any) -> str:
    if type(v) is str:
        return v
    raise ValueError(f"Expected a string argument, got {type_name(v)}")


def as_i(v: Any) -> int:
    if type(v) is int:
        return v
    raise ValueError(f"Expected an integer argument, got {type_name(v)}")


def cast_str(v: Any) -> Any:
    return None if v is None else display(v)


def cast_int(v: Any) -> Any:
    if v is None:
        return None
    t = type(v)
    if t is bool:
        return 1 if v else 0
    if t is int:
        return v
    if t is float:
        if math.isnan(v) or math.isinf(v):
            raise ValueError("Cannot cast this value to INT")
        return int(math.trunc(v))
    if t is str:
        # DataFusion accepts only an optional sign + digits -- no surrounding
        # whitespace, decimals, or exponents (measured). Python int() would strip
        # whitespace and accept underscores, so validate the exact shape first.
        if not _INT_STR.fullmatch(v):
            raise ValueError(f"Cannot cast '{v}' to INT")
        return int(v)
    raise ValueError("Cannot cast this value to INT")


def cast_float(v: Any) -> Any:
    if v is None:
        return None
    t = type(v)
    if t is bool:
        return 1.0 if v else 0.0
    if t is int:
        return float(v)
    if t is float:
        return v
    if t is str:
        # DataFusion rejects surrounding whitespace (measured); Python float()
        # would strip it. Otherwise Python's parse matches (1e3, .5, 5., inf, nan).
        if v != v.strip():
            raise ValueError(f"Cannot cast '{v}' to FLOAT")
        try:
            return float(v)
        except ValueError:
            raise ValueError(f"Cannot cast '{v}' to FLOAT") from None
    raise ValueError("Cannot cast this value to FLOAT")


def cast_bool(v: Any) -> Any:
    if v is None:
        return None
    t = type(v)
    if t is bool:
        return v
    if t is int:
        return v != 0
    if t is float:
        return v != 0.0
    if t is str:
        # DataFusion accepts a token set (case-insensitive, whitespace-trimmed)
        # and errors on anything else (measured).
        low = v.strip().lower()
        if low in ("true", "t", "1", "yes", "y", "on"):
            return True
        if low in ("false", "f", "0", "no", "n", "off"):
            return False
        raise ValueError(f"Cannot cast '{v}' to BOOLEAN")
    raise ValueError("Cannot cast this value to BOOLEAN")


def upper(*a: Any) -> Any:
    return None if any(x is None for x in a) else as_s(a[0]).upper()


def lower(*a: Any) -> Any:
    return None if any(x is None for x in a) else as_s(a[0]).lower()


def trim(*a: Any) -> Any:
    return None if any(x is None for x in a) else as_s(a[0]).strip()


def substr(*a: Any) -> Any:
    if any(x is None for x in a):
        return None
    s = as_s(a[0])
    start = as_i(a[1])
    n = len(s)
    if len(a) <= 2:
        return s[max(start, 1) - 1 :]
    length = as_i(a[2])
    if length < 0:
        raise ValueError("negative substring length not allowed")
    # DataFusion/Postgres windowing: the 1-indexed window is [start, start+length),
    # intersected with the string's [1, n+1) -- positions < 1 consume the length
    # rather than being clamped away. (Clamping start first, as before, was wrong
    # for start <= 0: substr('hello', 0, 3) is 'he', not 'hel'.)
    begin = max(start, 1)
    end = min(start + length, n + 1)
    return s[begin - 1 : end - 1] if end > begin else ""


def concat(*a: Any) -> str:
    return "".join(display(x) for x in a if x is not None)


def dpipe(l: Any, r: Any) -> Any:  # noqa: E741
    """The `||` operator: NULL-propagating string concat -- any NULL operand
    yields NULL (unlike concat(), which skips NULLs). Mirrors expr.rs concat_op:
    non-NULL operands are rendered via display() and joined."""
    if l is None or r is None:
        return None
    return display(l) + display(r)


def abs_(*a: Any) -> Any:
    v = a[0]
    if v is None:
        return None
    t = type(v)
    if t is int or t is float:
        return abs(v)
    raise ValueError(f"ABS expects a number, got a {type_name(v)} value")


def round_(*a: Any) -> Any:
    """ROUND always returns a float, including for an int argument.

    Measured: DataFusion's ROUND(3) is 3.0; Rust returns 3 (int) -- a Rust bug
    (xfail-ed + ticketed, Task 11). Rounding is half-away-from-zero, which
    DataFusion and Rust agree on and Python's banker's round() does NOT.
    """
    v = a[0]
    if v is None:
        return None
    t = type(v)
    if t is int:
        return float(v)
    if t is float:
        if math.isnan(v) or math.isinf(v):
            return v
        # math.floor/ceil return ints, so re-wrap to keep this a float.
        return float(math.floor(v + 0.5)) if v >= 0 else float(math.ceil(v - 0.5))
    raise ValueError(f"ROUND expects a number, got a {type_name(v)} value")


def coalesce(*a: Any) -> Any:
    for v in a:
        if v is not None:
            return v
    return None


def nullif(*a: Any) -> Any:
    """NULLIF compares numerically across int/float, not by variant.

    Measured: DataFusion's NULLIF(1, 1.0) is NULL; Rust returns 1 because its
    Value equality is variant-tagged -- a Rust bug (xfail-ed + ticketed).
    Deliberately NOT val_eq: lookup-join keys stay type-strict, NULLIF coerces.
    `eq` returns None when either side is NULL, and NULLIF(NULL, NULL) must be
    NULL -- which falls out, since a[0] is then NULL anyway.
    """
    if len(a) != 2:
        raise ValueError("NULLIF expects 2 arguments")
    return None if eq(a[0], a[1]) is True else a[0]


def miss(table: str, k: tuple) -> str:
    """Message for an inner lookup-join miss (plan.rs InterpError::MissingKey)."""
    rendered = ", ".join(display(part[1]) for part in k)
    return f"No row in static table '{table}' matches key ({rendered})"
