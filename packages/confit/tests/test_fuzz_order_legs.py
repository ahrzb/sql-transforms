"""The oracle's order legs must actually FIRE.

A comparison leg that never fails verifies nothing (a review found `_key`
sorting every leg, so ANY permutation of the row path passed). Each test here
wraps a real engine in a deliberate order bug and requires the leg to report
it -- these are capability pins: delete the leg, or quietly route it back
through the multiset form, and they go red.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pytest
from confit import DuckDBInferFn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fuzz import oracle  # noqa: E402

ROWS = [{"a": 1}, {"a": 2}, {"a": 3}]
SCHEMA = pa.schema([pa.field("a", pa.int64(), nullable=False)])


def _case():
    # the minimal Case surface _extra_legs touches: rows for the
    # infer_rows-vs-arrow leg, a query body the sklearn leg never reaches
    # (ests={} short-circuits it)
    return SimpleNamespace(
        rows=ROWS,
        row_schema={"a": "int"},
        query=SimpleNamespace(body=SimpleNamespace(items=[], order_by=None)),
        tree=None,
    )


class _Scrambled:
    """A real engine whose BATCH output is reversed -- per-row calls are
    untouched, so values are all correct and only the order is wrong.
    Exactly the bug class `_key` could never see."""

    def __init__(self, fn):
        self._fn = fn

    def infer_arrow(self, table):
        out = self._fn.infer_arrow(table)
        if len(out) > 1:
            return out.take(list(range(len(out) - 1, -1, -1)))
        return out

    def infer_rows(self, rows):
        out = self._fn.infer_rows(rows)
        return out[::-1] if len(out) > 1 else out

    def __getattr__(self, name):
        return getattr(self._fn, name)


def _fn():
    return DuckDBInferFn(
        "SELECT a + 1 AS o FROM __THIS__",
        row_tables={"__THIS__": SCHEMA},
        static_tables={},
    )


def test_a_correct_engine_passes_the_order_legs():
    fn = _fn()
    table = pa.Table.from_pylist(ROWS, schema=SCHEMA)
    got = fn.infer_arrow(table).to_pylist()
    assert oracle._extra_legs(fn, _case(), table, got, {}, []) is None


def test_a_scrambled_batch_is_caught_as_an_order_bug():
    fn = _Scrambled(_fn())
    table = pa.Table.from_pylist(ROWS, schema=SCHEMA)
    got = fn.infer_arrow(table).to_pylist()
    v = oracle._extra_legs(fn, _case(), table, got, {}, [])
    assert v is not None, "the order legs accepted a permuted batch"
    assert "order" in v.klass or v.klass == "reversal", v.klass
    # and the bug really was order-only: the multiset never differed
    singles = [fn.infer_arrow(table.slice(i, 1)).to_pylist() for i in range(3)]
    assert oracle._key([r for s in singles for r in s]) == oracle._key(got)


def test_sortedness_follows_duckdb_defaults():
    ok = [{"o": 1}, {"o": 2}, {"o": 2}, {"o": None}]
    assert oracle._sorted_by(ok, "o")  # ties fine, NULLS LAST fine
    assert not oracle._sorted_by([{"o": 2}, {"o": 1}], "o")
    assert not oracle._sorted_by([{"o": None}, {"o": 1}], "o")  # NULL first: no
    nan = float("nan")
    assert oracle._sorted_by([{"o": 1.0}, {"o": nan}, {"o": None}], "o")
    assert not oracle._sorted_by([{"o": nan}, {"o": 1.0}], "o")


@pytest.mark.parametrize(
    ("static_only", "order_by", "want"),
    [
        (False, None, "row-path"),
        (True, None, "constant-unordered"),
        (True, "o", "constant-ordered"),
    ],
)
def test_compare_mode_is_derived_from_the_case(static_only, order_by, want):
    case = SimpleNamespace(
        query=SimpleNamespace(body=SimpleNamespace(order_by=order_by))
    )
    assert oracle.compare_mode(case, static_only) == want
