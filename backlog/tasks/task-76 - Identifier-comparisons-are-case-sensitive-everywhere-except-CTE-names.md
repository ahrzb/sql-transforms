---
id: TASK-76
title: Identifier comparisons are case-sensitive everywhere except CTE names
status: Done
assignee: []
created_date: '2026-08-08 00:25'
updated_date: '2026-08-08 09:40'
labels: []
dependencies: []
ordinal: 67000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-71 folded CTE names to match DuckDB's case-insensitive binder. It did not
touch the other places the walk compares identifiers, and round two raised five
claims there — none verified when the ticket was written. Every one was
reproduced by hand before anything was changed; the results are below.

## The trap, checked first — and it is not a trap

The ticket warned that a blanket `.lower()` would be wrong for quoted
identifiers. **It is not: DuckDB folds quoted identifiers too**, unlike
Postgres. `CREATE TABLE "Weird"` is reachable as `weird`, `"weird"` and
`"WEIRD"`. So folding everywhere the *binder* would is right, and the only
boundary that matters is the caller's frame — Python's namespace, which is
case-sensitive and is looked up rather than bound. Pinned in `_case_test.py`.

## Confirmed, and fixed

**`_catalog` compared exact strings — silent.** A connection holding
`Customers` did not answer to `customers`, so the name was refused as unknown;
worse, with a frame object of that spelling in reach the frame answered instead
of the connection's own table. Measured 999.0 where the connection said 10.0,
no error.

**`_catalog` claimed names the connection cannot bind.** All 47 rows
`duckdb_views()` returns on a fresh connection are `internal` — `tables`,
`columns`, `schemata` — and none is reachable unqualified. Found while fixing
it: an ATTACHed database's tables are listed too and are equally unbindable.
Either one cost the caller their own object of that name.

**`_correlation` compared qualifiers exactly — both directions.** An inner
alias `T` did not shadow an outer `t`, so
`SELECT x.z FROM __THIS__ t, (SELECT avg(t.v) AS z FROM __FIT__ T) x` raised
`CorrelatedFit` on a query DuckDB binds inward. The mirror shape — outer `T`,
reference `t` — missed a real correlation, froze the subtree and died at fit
inside DuckDB's own message, which is a P7 violation rather than a wrong
answer.

## Confirmed as a coverage gap only

**`_reads`' CTE lookup was already folded** and correct. Nothing held it down:
the one case gate went through `_resolve`, so deleting the fold stayed green.
It now has its own gate, asked of `_reads` directly. The shape matters — the
subtree has to *reference* the CTE without containing it, because `_reads`
walks deep and a node carrying the definition answers from the definition
whatever the lookup does. That is exactly how `_plan` asks.

## The fix

Fold at every comparison DuckDB would bind: `_catalog` (both the stored name
and the reference), `_names_in`, `_bindings_at`, `_correlation`'s qualifier,
and `_resolve`'s catalog test. Restrict `_catalog` to
`NOT internal AND database_name IN (current_database(), 'temp')`. Nothing folds
on the `scope`/`captured` side.

## A regression the narrowing caused, caught before merge

Narrowing `_catalog` broke *qualified* names: `side.main.far` and
`information_schema.tables` both worked before, precisely because the
unfiltered listing contained `far` and `tables`. The catalog was never the
right test for them — a captured object is registered under a bare name, so a
qualified reference can never mean one, whatever the listing holds. `_resolve`
now takes a qualified name as the connection's own without consulting
anything, and refuses at construction when there is no connection to own it,
since `fit` makes a fresh one per call and nothing qualified could ever be
there. Gated in `_catalog_test.py`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [x] #1 Each reported case is reproduced by hand first, or struck from the list with a note saying it did not reproduce
- [x] #2 Identifier comparison folds case wherever the walk compares a name DuckDB would bind case-insensitively
- [x] #3 A quoted identifier that IS case-sensitive in DuckDB keeps that behaviour — folding must not overreach
- [x] #4 `_catalog` returns the names the supplied connection can actually bind, not every row of duckdb_tables/duckdb_views
- [x] #5 The case gates exercise `_reads` and `_correlation`, not only `_resolve`
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Landed as `_case_test.py` (23 gates) plus the folds in `_ast.py`,
`_analysis.py` and `_transform.py`. Full suite green: 949 passed, 3 xfailed.

AC#3 reads oddly now because its premise is false — no DuckDB identifier is
case-sensitive, quoted or not. Kept and pinned as a gate rather than deleted,
because the *reason* the blanket fold is safe is worth holding down: if a
future DuckDB starts honouring quotes, `test_duckdb_folds_quoted_identifiers_too`
goes red before any of this does. It lives in `_case_test.py` rather than
`_shapes_test.py` as the ticket suggested — it is a binder behaviour, not a
serialization shape, and it belongs next to the folds that depend on it.

Mutation-checked, all nine fix sites, each reverted in turn with the file
restored byte-exact: catalog fold, catalog filter, `_resolve`'s compare,
`_names_in`, `_bindings_at`, `_correlation`, `_reads`, and the two halves of
the qualified-name arm. Three survived the first pass — every one a weak gate
of mine, not dead code, because the spellings in my tests were already
lowercase. Parametrising both sides of each comparison caught them. That is
the third time in this ticket series a gate was shaped like the fix instead of
like the usage, and the qualified-name regression above is the fourth: no gate
covered it, and it only surfaced because the guide sentence "pass yours, and
it uses your catalog" sent me to try the shapes a user would.
<!-- SECTION:NOTES:END -->
