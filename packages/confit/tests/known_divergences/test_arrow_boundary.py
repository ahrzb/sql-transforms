"""The infer_arrow boundary: output types and round-trips (TASK-71, TASK-72).

Split out of test_known_divergences.py 2026-08-16; see README.md for what
belongs here (kept behaviour + its ground) versus in
../test_open_divergences.py (behaviour we intend to change).
"""

from __future__ import annotations

import duckdb
import pyarrow as pa
import pytest
from confit import DuckDBInferFn

# ------------------------------------------------- the infer_arrow path --
#
# Three documented entry points — infer, infer_rows, infer_arrow — are supposed
# to be the same function behind different boundaries. Two ways they were not.
#
# FIXED 2026-08-08 (TASK-71, TASK-72).
#
# TASK-71 is resolved by REFUSAL, the other half of the contract. `infer_arrow`
# builds no Python rows — that is its entire reason to exist — so there is
# nothing to call `model_validate` on. Running the rows through pydantic anyway
# would make the columnar path exactly as slow as `infer`, which is to say
# pointless; silently skipping it gave two answers from one function. So it
# now refuses when an `output_model` was SUPPLIED. A synthesized one (the
# default) carries no validators, defaults or coercion, so the columnar path
# stays available for it, which is the common case.
#
# TASK-72 is resolved by matching DuckDB: `pa.string()`, 32-bit offsets. The
# 2 GiB-per-batch ceiling that comes with them is refused by name rather than
# wrapped.
#
# MIGRATION-NOTE (2026-08-13, arrow-schema-api): TASK-71's refusal fired only
# when an `output_model` was SUPPLIED to a pydantic-surface build — that whole
# kwarg, and the synthesized-model machinery it stood next to, is deleted by
# this migration (spec 2026-08-13-arrow-schema-api-design.md: "the
# `output_model` refusal on `infer_arrow` ... exist[s] only to serve pydantic
# out. Dict-out deletes the machinery and the limitation."). The test that
# pinned it (`test_infer_arrow_refuses_a_supplied_output_model`) has no
# construction left to express — `output_model=` now raises TypeError at
# `DuckDBInferFn.__init__` itself, for every entry point, which is already
# covered by `test_arrow_schema_api.py::test_infer_and_output_model_are_gone`.
# Removed rather than mistranslated.


def test_infer_arrow_without_an_output_model_still_works():
    """infer_arrow needs nothing extra supplied to serve — the fast path
    that TASK-71/72 protected is exercised directly here."""
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
def test_infer_arrow_string_type_matches_duckdb(sql):
    schema = pa.schema([pa.field("s", pa.string(), nullable=False)])
    fn = DuckDBInferFn(sql, row_tables={"__THIS__": schema}, static_tables={})
    got = fn.infer_arrow(pa.table({"s": ["a", "bb", "ccc"]}))
    con = duckdb.connect()
    con.execute("CREATE TABLE __THIS__ (s VARCHAR)")
    con.execute("INSERT INTO __THIS__ VALUES ('a'), ('bb'), ('ccc')")
    want = con.execute(sql).to_arrow_table()
    assert got.schema == want.schema
    assert got.to_pylist() == want.to_pylist()
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
