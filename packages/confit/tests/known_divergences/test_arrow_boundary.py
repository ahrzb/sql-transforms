"""The infer_arrow boundary: output types and round-trips.

Split out of test_known_divergences.py 2026-08-16; see README.md for what
belongs here (kept behaviour + its ground) versus in
../test_open_divergences.py (behaviour we intend to change).
"""

from __future__ import annotations

import pyarrow as pa
import pytest
from confit import DuckDBInferFn, compare

# ------------------------------------------------- the infer_arrow path --
#
# The documented entry points — infer_rows and infer_arrow — are supposed to be
# the same function behind different boundaries. Two ways they were not, both
# FIXED 2026-08-08.
#
# The FIRST was resolved by REFUSAL, the other half of the contract.
# `infer_arrow` builds no Python rows — that is its entire reason to exist — so
# there was nothing to call `model_validate` on. Running the rows through
# pydantic anyway would have made the columnar path exactly as slow as the row
# one, which is to say pointless; silently skipping it gave two answers from one
# function. That whole surface is gone now: the pydantic `output_model=` kwarg
# and the synthesized-model machinery beside it were deleted by the
# arrow-schema-api migration (2026-08-13, spec
# 2026-08-13-arrow-schema-api-design.md), so the refusal has no construction
# left to express — `output_model=` raises TypeError at
# `DuckDBInferFn.__init__` itself, for every entry point, pinned in
# `test_arrow_schema_api.py::test_infer_and_output_model_are_gone`.
#
# The SECOND was resolved by matching DuckDB: `pa.string()`, 32-bit offsets. The
# 2 GiB-per-batch ceiling that comes with them is refused by name rather than
# wrapped.


def test_infer_arrow_without_an_output_model_still_works():
    """infer_arrow needs nothing extra supplied to serve — the fast path both
    of those fixes protect is exercised directly here."""
    schema = pa.schema([pa.field("x", pa.int64(), nullable=False)])
    fn = DuckDBInferFn(
        "SELECT x * 5 AS y FROM __THIS__",
        row_tables={"__THIS__": schema},
        static_tables={},
    )
    assert fn.infer_arrow(pa.table({"x": [2, 4]})).to_pylist() == [
        {"y": 10},
        {"y": 20},
    ]


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT upper(s) AS u FROM __THIS__",
        # a NULL in the column, so the validity buffer is exercised too
        "SELECT NULLIF(s, 'a') AS u FROM __THIS__",
        # several string lanes at once
        "SELECT s AS a, lower(s) AS b, s || s AS c FROM __THIS__",
    ],
)
def test_infer_arrow_string_type_matches_duckdb(sql, oracle):
    schema = pa.schema([pa.field("s", pa.string(), nullable=False)])
    fn = DuckDBInferFn(sql, row_tables={"__THIS__": schema}, static_tables={})
    got = fn.infer_arrow(pa.table({"s": ["a", "bb", "ccc"]}))
    oracle.table("__THIS__", "s VARCHAR", [("a",), ("bb",), ("ccc",)])
    want = oracle.answer(sql)
    compare.assert_schema(got.schema, want.schema, ctx=sql)
    compare.assert_rows(compare.rows(got), compare.rows(want), ordered=True, ctx=sql)
    # The point of the schema agreeing: the two stack.
    assert pa.concat_tables([want, got]).num_rows == 6


# ---------------------------------------- arrow output round-trips --


def test_infer_arrow_string_output_feeds_back_in():
    """Our own output is a valid input — `ingest` takes 32-bit offsets."""
    schema = pa.schema([pa.field("s", pa.string(), nullable=False)])
    fn = DuckDBInferFn(
        "SELECT upper(s) AS s FROM __THIS__",
        row_tables={"__THIS__": schema},
        static_tables={},
    )
    once = fn.infer_arrow(pa.table({"s": ["a", "bb"]}))
    assert fn.infer_arrow(once).to_pylist() == [{"s": "A"}, {"s": "BB"}]


# ---------------------------------------------------------------------------
# Adjudicated 2026-08-19 as a fuzzer fix, not an engine bug: a join
# star with colliding names. DuckDB's TOP-LEVEL arrow export keeps the
# DUPLICATES; its own subquery/CTE/CTAS boundaries and .df() rename them
# `<name>_N` -- which is the wave-5 client contract this engine adopted
# (pins-wave5/dup-names-client-contract.json), because dict-shaped
# infer_rows output cannot hold two `c0` keys losslessly. KEPT divergence:
# our arrow names are the deduped contract names; the campaign's schema leg
# normalizes DuckDB through the same rule.
# ---------------------------------------------------------------------------
def test_join_star_collisions_keep_the_wave5_dedup_contract(oracle):
    row = pa.schema(
        [pa.field("c0", pa.int64(), nullable=False), pa.field("c2", pa.int64())]
    )
    static = pa.table(
        {"c0": pa.array([1], pa.int64()), "c1": pa.array([9], pa.int64())}
    )
    sql = "SELECT * FROM __THIS__ LEFT JOIN s0 ON (c2 = s0.c0)"

    oracle.table("__THIS__", "c0 BIGINT, c2 BIGINT", [(5, 1)])
    oracle.load("s0", static)
    duck_names = oracle.answer(sql).schema.names
    assert duck_names == ["c0", "c2", "c0", "c1"], (
        f"oracle moved -- arrow export no longer keeps duplicates: {duck_names}"
    )

    fn = DuckDBInferFn(sql, row_tables={"__THIS__": row}, static_tables={"s0": static})
    assert fn.output_schema.names == ["c0", "c2", "c0_1", "c1"]
    # and the dict path stays LOSSLESS, which is what the contract buys
    rows = fn.infer_rows([{"c0": 5, "c2": 1}])
    assert rows == [{"c0": 5, "c2": 1, "c0_1": 1, "c1": 9}]
