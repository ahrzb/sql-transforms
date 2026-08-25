"""Confit — SQL specialized once, served bit-exact.

Fixed SQL plus static tables frozen at fit time are partially evaluated, once,
into a native function whose only remaining input is the request row. The
contract has exactly two outcomes: serve bit-for-bit identical to DuckDB, or
refuse at build time with a named error. There is no third mode -- nothing is
approximated or silently dropped at inference.

    fn = DuckDBInferFn(sql, row_tables={"t": pa.schema(...)},
                       static_tables=..., shape="filter")
    fn.infer_rows(rows)     # dict-or-object rows in, dict rows out
    fn.infer_arrow(table)   # pa.Table in, pa.Table out

See packages/confit/docs/known-limitations.md for the constructs that are
refused, each with an executable twin asserting the refusal.
"""

from __future__ import annotations

from confit._engine import BUILD_PROFILE, DuckDBInferFn

__all__ = ["BUILD_PROFILE", "DuckDBInferFn"]
