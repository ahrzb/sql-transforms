"""Pieces shared by catalog entries, on `confit.sql`.

Each spells one numpy operation in the order numpy performs it, so an entry
built from them is bit-exact with its twin wherever the twin performs that
same sequence (docs/native/goal.md, "Parity").
"""

from __future__ import annotations

from typing import Any

from confit import sql as S


def f64(v: Any) -> S.Const:
    """A fitted number as a DOUBLE constant (a numpy scalar included)."""
    return S.lit(float(v))


def dot(xs: list[S.Expr], ws: list[Any], bias: Any | None = None) -> S.Expr:
    """`sum(x_i * w_i) + bias`, accumulated left to right. NumPy/BLAS may
    reassociate a matvec, so an entry built on this declares an ulp bound."""
    acc: S.Expr | None = None
    for x, w in zip(xs, ws, strict=True):
        term = x * f64(w)
        acc = term if acc is None else acc + term
    if acc is None:
        raise ValueError("an empty dot product")
    return acc if bias is None else acc + f64(bias)
