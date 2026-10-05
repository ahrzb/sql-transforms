"""One parity check for hand-written queries: the campaign's own verdict.

A test that compares confit with DuckDB used to carry its own helper, and each
helper chose its own strictness: one backend or both, rows sorted or not,
types checked or not, traps compared by message or not at all. This module
routes a hand-written query through `fuzz.oracle.run_case`, the same code that
grades every campaign case. So a test asserts exactly what the campaign
asserts:
- both backends build or both refuse, and agree with each other;
- DuckDB is read with the optimizer off and on;
- values compare by `repr`, and output types and names are checked;
- the non-null promise is checked;
- the extra boundary legs run.

    v = assert_parity("SELECT a + 1 AS b FROM __THIS__", pa.table({"a": [1, None]}))
    assert_parity(sql, rows, statics={"d": d}, expect="REFUSED")

Inputs are Arrow tables. Their types must lie in the campaign's storage
vocabulary (`fuzz.oracle._ARROW` plus structs); anything else raises
TypeError, and such a test keeps a helper of its own. `udfs=` takes UDF
protocol objects, registered on both engines as given.

Deliberately on their own helpers: Arrow-boundary fixtures
(`tests/test_infer_arrow.py`), whose crafted buffers a rebuilt table would
lose; `duck_check_ulp`, a float tolerance by design; and call-count checks
(`udf_check(after_engine=...)`), since the verdict runs the engine more
than once.
"""

from __future__ import annotations

import re

import pyarrow as pa

from fuzz import gen as G
from fuzz import oracle as O

AGREEMENT = ("AGREE", "AGREE_TRAP")

_SPEC = {v: k for k, v in O._ARROW.items() if k not in ("int", "float", "str")}


def spec_of(field: pa.Field):
    """The campaign storage spec for one Arrow field (the inverse of
    `fuzz.oracle._arrow_field`)."""
    t = field.type
    if pa.types.is_struct(t):
        return G.Struct(
            tuple((t.field(i).name, spec_of(t.field(i))) for i in range(t.num_fields)),
            nullable=field.nullable,
        )
    if t not in _SPEC:
        raise TypeError(f"{field.name}: {t} is outside the campaign vocabulary")
    return _SPEC[t] + ("?" if field.nullable else "")


def _spec_table(t: pa.Table) -> tuple[dict, list[dict]]:
    return {f.name: spec_of(f) for f in t.schema}, t.to_pylist()


def case(
    sql: str,
    rows: pa.Table,
    *,
    statics: dict[str, pa.Table] | None = None,
    shape: str | None = None,
    udfs: list | None = None,
) -> G.Case:
    row_schema, row_list = _spec_table(rows)
    return G.Case(
        seed=0,
        row_schema=row_schema,
        rows=row_list,
        statics={n: _spec_table(t) for n, t in (statics or {}).items()},
        udfs=[],
        tree=None,
        query=None,
        shape=shape,
        output=None,
        sql=sql,
        udf_objs=list(udfs or []),
    )


def verdict(sql, rows, *, statics=None, shape=None, udfs=None) -> O.Verdict:
    """The campaign verdict for one query over the given inputs. `udfs` are
    UDF protocol objects, registered on both engines as given."""
    return O.run_case(case(sql, rows, statics=statics, shape=shape, udfs=udfs))


def table(schema: dict, rows: list[dict]) -> pa.Table:
    """An Arrow table from campaign specs: `{"c1": "int?", "s": "str"}`.
    The semantic names (`int`, `float`, `str`) are 64-bit and UTF-8."""
    return O._arrow_table(schema, rows)


def assert_parity(
    sql: str,
    rows: pa.Table,
    *,
    statics: dict[str, pa.Table] | None = None,
    shape: str | None = None,
    udfs: list | None = None,
    expect: str | tuple[str, ...] = AGREEMENT,
    trap: str | None = None,
) -> O.Verdict:
    """Assert the verdict is one of `expect`, which defaults to agreement
    (rows or a trap on both sides). `trap` is a regex confit's error must
    match, and DuckDB's too when both trap; with the default `expect` it
    narrows it to AGREE_TRAP. Returns the verdict for further checks, for
    example `.klass` on a refusal.

    `expect="DIVERGE_OPT"` states that DuckDB's optimizer changes the answer
    (usually by pruning a column that traps) while confit matches the
    optimizer-off reading, which is the contract."""
    if trap is not None and expect == AGREEMENT:
        expect = "AGREE_TRAP"
    expect = (expect,) if isinstance(expect, str) else expect
    v = verdict(sql, rows, statics=statics, shape=shape, udfs=udfs)
    assert v.kind in expect, (
        f"{v.kind} ({v.klass}), expected {' or '.join(expect)}\n"
        f"  sql: {sql}\n  detail: {v.detail[:400]}"
        + (f"\n  duckdb: {v.oracle}" if v.oracle else "")
    )
    if trap is not None:
        assert re.search(trap, v.detail), f"confit: {v.detail}"
        if v.kind == "AGREE_TRAP":
            assert re.search(trap, v.oracle_detail), f"DuckDB: {v.oracle_detail}"
    return v
