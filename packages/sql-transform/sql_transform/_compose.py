"""Template desugaring for `SQLTransform(t"...")`.

A t-string with no interpolations is plain SQL and passes straight through.

**Composition is reset.** Referencing one transform from another --
`t"SELECT {a}(age) ..."` and the frozen `{a.transform}` form -- was built on the
DataFusion batch engine this package no longer contains: it read the other
transform's `_state_tables` / `_rewritten_sql` and re-fitted its definition into
a scoped state table. Rather than leave a half-working version behind, the
surface is kept and refuses by name until it is rebuilt on confit.

Fitted sklearn transformers (`{scaler}`) are still classified here so that
`SQLTransform` can raise its own error for them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from string.templatelib import Template

import datafusion
import pyarrow as pa
from sqlglot import exp

from sql_transform._transformer_ref import is_transformer


@dataclass(frozen=True)
class Ref:
    transform: object  # a fitted transformer if is_transformer
    frozen: bool
    expr_text: str = ""  # interpolation source, for error messages
    is_transformer: bool = False


@dataclass(frozen=True)
class InlineResult:
    scoped_state: dict[str, pa.Table] = field(default_factory=dict)


def _sql_transform_cls() -> type:
    from sql_transform import SQLTransform

    return SQLTransform


def desugar_template(template: Template) -> tuple[str, dict[str, Ref]]:
    """Turn a t-string into (plain SQL with __COMPOSE_i__ placeholders, ref map)."""
    parts: list[str] = []
    refs: dict[str, Ref] = {}
    i = 0
    for item in template:
        if isinstance(item, str):
            parts.append(item)
            continue
        v = item.value  # Interpolation
        if is_transformer(v):
            ref = Ref(v, frozen=False, expr_text=item.expression, is_transformer=True)
        elif isinstance(v, _sql_transform_cls()):
            raise NotImplementedError(
                f"interpolation {{{item.expression}}}: composing one SQLTransform "
                f"into another is not supported yet -- it is being rebuilt on "
                f"confit. Inline the other transform's SQL for now."
            )
        elif hasattr(v, "transform") and not hasattr(v, "n_features_in_"):
            # Transformer-shaped but unfitted. Without this the generic TypeError
            # below blames the interpolation's TYPE, which sends users looking in
            # entirely the wrong place.
            raise ValueError(
                f"interpolation {{{item.expression}}}: {type(v).__name__} is not "
                f"fitted (or does not expose n_features_in_) -- call .fit(...) "
                f"before referencing it"
            )
        else:
            raise TypeError(
                f"interpolation {{{item.expression}}} must be a fitted "
                f"transformer, got {type(v).__name__}"
            )
        name = f"__COMPOSE_{i}__"
        refs[name] = ref
        parts.append(name)
        i += 1
    return "".join(parts), refs


def inline_references(
    select: exp.Select,
    refs: dict[str, Ref],
    ctx: datafusion.SessionContext,
    training: pa.Table,
) -> InlineResult:
    """No-op while composition is reset; `refs` can only be empty here.

    `SQLTransform.__init__` rejects transformer refs and `desugar_template`
    raises on cross-transform refs, so a non-empty map means a caller built
    `Ref`s by hand.
    """
    if refs:
        raise NotImplementedError(
            "reference inlining is not supported yet -- it is being rebuilt on confit"
        )
    return InlineResult()
