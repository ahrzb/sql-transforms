# Verification

This page says how the claims in this folder are checked. The gate
(`scripts/gate.py`) runs every check below.

## Evidence lines

Each claim ends with an *Evidence:* line. It names the tests that fail if
the rule breaks. A claim that no test pins says so, and the
[open work](../plan/README.md#open-work) lists it.

## Executable examples

`sql_transform/_docs_test.py` runs each `pycon` block in this folder with
Python's `doctest` module and its `ELLIPSIS` option. Each page runs in its
own namespace, so each example imports what it uses.

## The original-query differential

`sql_transform/_marginal_projection_test.py::test_fuzz_differential` draws
1,500 random window projections from seed 20260729. The environment
variable `MARGINALIZE_FUZZ_N` changes the count. It fits each projection
with `SQLProjection.marginalize` and compares the answers with the original
window query. Schemas, signed zero and NaN must match exactly.

## The corpus

`sql_transform/_corpus_test.py` keeps the original corpus SQL. That SQL is
the numerical oracle for each kept or migrated computation.

## The native catalog

The tests in `sql_transform/native/` compare each catalog entry with its
twin. `uv run --no-sync python -m sql_transform.native.coverage --write`
regenerates [native/coverage.md](native/coverage.md), and
`native/coverage_test.py` fails while that page is stale.
