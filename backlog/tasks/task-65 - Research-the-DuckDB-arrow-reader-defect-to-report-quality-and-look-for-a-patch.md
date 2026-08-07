---
id: TASK-65
title: >-
  Research the DuckDB arrow-reader defect to report quality, and look for a
  patch
status: To Do
assignee: []
created_date: '2026-08-07 00:00'
labels: []
dependencies: []
documentation:
  - packages/sql-transform/sql_transform/_duckdb_arrow_test.py
type: chore
ordinal: 58000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Registering a lazy Arrow reader back into the connection that produced it does
not work on DuckDB 1.5.5, and the way it fails is not stable. Nothing has been
reported upstream and nothing should be until this ticket is done — a report
that is wrong about the mechanism costs more than no report.

```python
con = duckdb.connect()                     # one connection, not two
con.register("src", pa.table({"price": [10.0, 20.0, 30.0]}))
con.register("out", con.execute("SELECT price * 2 AS z FROM src").arrow())
con.execute("SELECT * FROM out ORDER BY 1").fetchall()
```

Measured 2026-08-07, duckdb 1.5.5 / pyarrow 25.0.0 / CPython 3.14 / Windows,
five runs of that exact script: **three hung with no output and no error, two
returned zero rows and reported success.** `con.sql(...).arrow()` and
`fetch_record_batch()` hung on every attempt. `to_arrow_table()` works, and so
does draining the reader into a table first.

The zero-rows outcome is what makes this worth reporting: it neither serves nor
refuses, which is C5's one unrecoverable state at the API level. A hang is at
least loud.

An earlier write-up of this called it a deadlock. That was overstated from a
run of three that happened to hang each time, and is exactly the failure this
ticket exists to prevent: we do not report until the mechanism is understood,
not merely observed.

The workaround is in place and not blocked on any of this
(`_duckdb_arrow_test.py`, xfail-strict, plus the two working spellings pinned).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Reproduced outside this repo, on a clean env, with versions and OS recorded — and checked on at least one non-Windows platform, since the nondeterminism may be scheduler-dependent
- [ ] #2 Established whether it is version-specific: which DuckDB releases show it, and whether `.arrow()` returning a reader rather than a table was an intentional API change (it materialised in older versions)
- [ ] #3 Mechanism identified, not just observed — why zero rows sometimes and a hang other times. Read the replacement-scan / arrow-scan path and the pending-result lifetime; name the code that decides it
- [ ] #4 Searched duckdb/duckdb issues and PRs for an existing report; if one exists, link it and stop
- [ ] #5 Decided whether it is a defect or documented behaviour. "Do not register a connection's own undrained result" may be the correct answer — in which case the ask is that it *raise* instead of scanning empty
- [ ] #6 A minimal repro that does not depend on our code, small enough to paste, with the observed distribution over N runs rather than a single outcome
- [ ] #7 If a fix is small and localised, draft a patch against duckdb/duckdb with a regression test; otherwise write up why not
- [ ] #8 Report text drafted and shown to AmirHossein in chat. NOTHING is filed upstream without his explicit go
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Not urgent and not blocking: the model never calls `.arrow()`, only
`to_arrow_table()`, and the pin flips to XPASS if DuckDB changes behaviour.

Start from `_duckdb_arrow_test.py`, which already carries the measurements and
both working spellings. `faulthandler.dump_traceback_later` gives the Python
stack on the hang; it will show the scan, so the interesting part is below
Python and needs the C++ side — the arrow replacement scan and whatever owns
the reader's lifetime once `register` has taken it.

Worth checking while in there: whether the same self-reference through
`duckdb.execute` (the module-level default connection) behaves differently
from an explicit `duckdb.connect()`, since the model uses both — the default
one to parse via `json_serialize_sql`, and another to execute.
<!-- SECTION:NOTES:END -->
