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
from typing import Any


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


def _fmt_float(f: float) -> str:
    """Render a float the way DataFusion does (NOT the way Rust does).

    Measured 2026-07-17 -- all three engines disagree here, and plain str() is
    wrong twice:
                      DataFusion    Rust        Python str()
        1.0           "1.0"         "1"         "1.0"   <- str ok
        0.1           "0.1"         "0.1"       "0.1"   <- str ok
        NaN           "NaN"         "NaN"       "nan"   <- str WRONG
        inf/-inf      "inf"/"-inf"  same        same    <- str ok
        1e300         "1e300"       "1000...0"  "1e+300" <- str WRONG

    Rust's "1" and its 300-digit 1e300 are both bugs (xfail-ed + ticketed).
    """
    if math.isnan(f):
        return "NaN"
    if math.isinf(f):
        return "inf" if f > 0 else "-inf"
    # Python writes exponents as "1e+300"; DataFusion omits the "+".
    return str(f).replace("e+", "e")


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


def add(l: Any, r: Any) -> Any:  # noqa: E741
    if l is None or r is None:
        return None
    if type(l) is int and type(r) is int:
        return l + r
    return as_f(l) + as_f(r)


def sub(l: Any, r: Any) -> Any:  # noqa: E741
    if l is None or r is None:
        return None
    if type(l) is int and type(r) is int:
        return l - r
    return as_f(l) - as_f(r)


def mul(l: Any, r: Any) -> Any:  # noqa: E741
    if l is None or r is None:
        return None
    if type(l) is int and type(r) is int:
        return l * r
    return as_f(l) * as_f(r)


def div(l: Any, r: Any) -> Any:  # noqa: E741
    if l is None or r is None:
        return None
    if type(l) is int and type(r) is int:
        if r == 0:
            raise ValueError("division by zero")
        return _trunc_div(l, r)
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
