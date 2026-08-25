## 1. What the oracle is

### 1.1 It is a pseudo-oracle

**ORC-01.** The oracle answers exactly one question: *does confit produce the same
answer as this DuckDB build, run this way?* It never answers *is this correct?*
Authority is delegated to a reference implementation, so DuckDB's bugs are inherited
by construction and reproducing one is conformance, not a defect. An argument that
DuckDB's answer is wrong is out of scope of this document by design; it is an
upstream matter, and it changes nothing here until the pinned build changes.
*Verified-by:* the two-outcome contract, `packages/confit/docs/reports/pins-first-methodology.md:5-16`;
P18 in `packages/confit/docs/properties.md:231-235`.

**ORC-02.** The oracle is one constant, and both halves are load-bearing:

```
ORACLE = DuckDB 1.5.5, PRAGMA disable_optimizer, all other settings default
```

Neither half is a per-test choice, a per-campaign flag, or a thing a caller may
vary. A comparison run against anything else is not a comparison against the oracle.
*Verified-by:* the **pragma** half at `packages/confit/tests/conftest.py:7-12`
(decided 2026-08-17), `packages/confit/fuzz/oracle.py:10-16`,
`packages/confit/docs/known-limitations.md:20-30`. The **version** half is stated in
those documents' prose and enforced nowhere — see ORC-09. The **"all other settings
default"** half is `Unverified` and is measurably weaker than it reads: DuckDB's
`threads` default is derived from core count (measured 12 on this machine), and it
changes answers — see ORC-75 and ASK-13.

### 1.2 Why optimizer-off

**ORC-03.** The optimizer-on reading is not a function of the query, so it cannot be
a target. `statistics_propagation` reads a column's stored null statistic, so DuckDB
answers the same query over the same rows differently depending on the table's insert
history — measured: a table built as `[-128, NULL]` with the NULL then deleted answers
differently from one built as `[-128]`, with identical contents. confit compiles once
against a *schema* and serves many batches; it never sees a table, let alone its
history.
*Verified-by:* `packages/confit/tests/known_divergences/test_trap_elision.py`;
`packages/confit/docs/known-limitations.md:32-39`.

**ORC-04.** Optimizer-off is a *sanctioned* reference leg upstream, not our
invention. `PRAGMA enable_verification` registers an `UNOPTIMIZED` statement verifier
alongside COPIED / DESERIALIZED / NO_OPERATOR_CACHING and compares its result against
the original's; the function's own comment states the purpose as "Correctness of plans
both with and without optimizers". DuckDB itself treats disagreement between the two
legs as a bug in DuckDB.
*Verified-by:* DuckDB v1.5.5 source, `src/function/pragma/pragma_functions.cpp:134`
(`enable_verification` pragma), `src/main/client_verify.cpp:45` (the comment),
`:55` (the UNOPTIMIZED verifier), `:187` -> `src/verification/statement_verifier.cpp:155`
(`CompareResults`).

**ORC-05.** What `PRAGMA disable_optimizer` removes, precisely. It disables the
33 named optimizer passes (`OptimizerType` enumerates `EXPRESSION_REWRITER = 1`
through `WINDOW_SELF_JOIN = 33`), **and additionally** changes behavior at **twelve**
sites that read `enable_optimizer` directly, outside the pass list, in three groups:

| group | sites | what changes with the optimizer off |
|---|---|---|
| physical-operator selection | `plan_distinct.cpp:66`, `plan_window.cpp:31` and `:35`, `sorted_aggregate_function.cpp:686` and `:744` (both reached from `plan_aggregate.cpp:318`) | the DISTINCT-ON ordered-aggregate rewrite, streaming-vs-blocking window operators, sorted-aggregate `ORDER BY` simplification in two places |
| window-function execution | `window_aggregate_function.cpp:32` and `:56`, `window_rank_function.cpp:24`, `window_rownumber_function.cpp:26`, `window_value_function.cpp:207` and `:849` | window aggregation strategy and the rank/rownumber/value fast paths |
| logical-plan construction inside the binder | `plan_subquery.cpp:255`, `plan_joinref.cpp:411` | with the optimizer off, correlated subqueries **always** take a delim join; and a `RIGHT` outer join is **not** flipped to a `LEFT` with the sides swapped |

What is genuinely untouched is **expression binding**: nothing in name resolution,
overload selection or type inference reads the flag, so output types and bind-time
errors are identical, and so is execution-level laziness (an untaken `CASE` arm,
`AND`/`OR` short-circuit in a filter). The two `src/planner/binder/` sites are logical
*plan* construction performed by the binder, not binding, which is why types survive
them — but "none in binding" is the wrong sentence for them, and the join flip is the
axis that ORC-19 says decides hash-join output order.
*Verified-by:* DuckDB v1.5.5 source,
`src/include/duckdb/common/enums/optimizer_type.hpp:16-50` (33 members;
`EXPRESSION_REWRITER = 1` at `:18`, `WINDOW_SELF_JOIN = 33` at `:50`); the twelve
sites above, enumerated by grep over `src/` 2026-08-25, each line read.
*Correction, three parts.* (a) `conftest.py:20` ("`PRAGMA disable_optimizer` ==
disabling all 33 named optimizers") and `known-limitations.md:30` ("What it removes is
the 33 plan-rewrite passes") are incomplete by these twelve sites. (b) An earlier
version of this claim said "four sites ... none in binding" and that their reach was
"the static-tables-only path"; both are corrected above — the join flip and the delim
join are reachable from the row path. (c) `conftest.py:22`'s "constant folding still
happens (`1 + 2` is int32 3)" is true as an **observation** and wrong as a
**mechanism**: DuckDB's only constant folder is `src/optimizer/rule/constant_folding.cpp`,
an `EXPRESSION_REWRITER` rule, so the pragma removes it — `1 + 2` is still int32 3
because the expression is evaluated at run time instead of folded into the plan. Where
that distinction is observable, it is observable: `SELECT 2147483647 + 1 ... LIMIT 0`
errors with the optimizer off and serves `[]` with it on, because with the optimizer
on the plan collapses to `EMPTY_RESULT`. ORC-45's own source says the same thing.
See proposed ticket T-1.

**ORC-06.** The user-facing contract still names what a user's DuckDB returns, which
is optimizer-*on*. The gap between the two readings is therefore user-visible and
stays a reported finding (`DIVERGE_OPT`) rather than an accepted class. The oracle
and the contract surface are deliberately not the same thing.
*Verified-by:* `packages/confit/fuzz/oracle.py:43-46`;
`packages/confit/fuzz/runner.py:28-29` (`DIVERGE_OPT` is in `INTERESTING`).

### 1.3 How the identity is enforced

**ORC-07.** The specialization half is enforced mechanically across
`packages/confit/tests/`, and **deliberately not further**. An autouse fixture
monkeypatches `duckdb.connect` so every connection in the confit test suite comes back
with `PRAGMA disable_optimizer` already applied: the oracle is a property of the
package, not a per-test choice, and a new test that reaches for DuckDB gets the oracle
by construction. Measured today: **62 `duckdb.connect(` call sites across 23 files**
are covered by that one fixture (a grep for the bare name returns 66 lines in 24 files;
the four extra are inside `conftest.py` itself — three in its docstring and the
fixture's own `raw_connect = duckdb.connect` — and none of them is a covered call site).
The scoping is a decision with a measured ground, not an oversight: an import-time
assignment leaks into every other package for the rest of the session, and it did —
`sql_transform`'s single-evaluation tests count sklearn calls made through DuckDB, and
losing CSE doubled them. A `monkeypatch` fixture keeps the oracle inside this directory.
*Verified-by:* `packages/confit/tests/conftest.py:62-71` (the fixture), `:45-50` (the
scoping decision and its ground); counts measured 2026-08-25 by grep over
`packages/confit/tests/`.
*Correction:* `conftest.py:14` still says "the 42 call sites". See proposed ticket T-1.

**ORC-08.** A test that *wants* the optimizer says so in its own body
(`con.execute("PRAGMA enable_optimizer")`), which reads as the deliberate exception it
is. Exactly two such exceptions exist, both in the test that documents what the
optimizer does.
*Verified-by:* `packages/confit/tests/conftest.py:37-43`;
`packages/confit/tests/known_divergences/test_trap_elision.py:468` and `:566`.

**ORC-09.** **[FACT]** The version half of the identity is enforced **nowhere**.
**36 markdown files outside `backlog/`** name DuckDB 1.5.5 (48 counting `backlog/`) —
`known-limitations.md`, `properties.md`, three reports, both READMEs, both RFCs and
some twenty specs among them — but `pyproject.toml:15` and
`packages/sql-transform/pyproject.toml:10` declare `duckdb>=1.5.5` — a floor —
`packages/confit/pyproject.toml` declares no duckdb dependency at all, and only
`uv.lock` resolves 1.5.5 exactly. A `uv lock --upgrade` silently re-points the oracle
and no gate notices.
*Verified-by:* measured 2026-08-25 — `pyproject.toml:15`,
`packages/sql-transform/pyproject.toml:10`, `packages/confit/pyproject.toml`
(dependencies: `pyarrow>=19.0` only), `uv.lock:368-370`; document count by grep
2026-08-25. The fix is ASK-1.

> ### ASK-1 — pin or floor, and which line?
>
> The contract says 1.5.5; the manifests permit anything newer (ORC-09). Two
> sub-questions, and the second is time-boxed.
>
> **(a) How is the version enforced?** Options: hard-pin `duckdb==1.5.5` in the
> manifests; or keep the floor and add a loud assert in the fixture that already owns
> the oracle identity. The assert is one line in the file that already applies the
> pragma:
>
> ```python
> # packages/confit/tests/conftest.py, inside _duckdb_is_the_oracle
> assert duckdb.__version__ == "1.5.5", f"oracle is 1.5.5, got {duckdb.__version__}"
> ```
>
> Not applied here — this document is docs-only. Proposed ticket T-2.
>
> **(b) 1.5.5, or the LTS line?** DuckDB ships minor versions on a roughly 4-month
> cadence and semantics have already moved inside a patch release. v2.0 brings a new
> SQL parser. The bump protocol in section 9 is cheap to write now and expensive to
> write during a migration; which version it targets is your call.
>
> *Binds:* ORC-02, and every pin in the corpus by extension.

### 1.4 The three excluded neighbours

**ORC-72.** The **fit/serving oracle** is out of scope here. `sql_transform`'s
projection path runs DuckDB with the optimizer **on** and at `SET threads = 1`, and its
parity targets are KPI C1 (fit + serving over the training set is bit-exact against the
original SQL) and KPI C4 (transformer columns, which DuckDB cannot run at all, gate
against an independent clone-per-group sklearn reference). Fit reproducibility itself is
explicitly out of contract in v0. None of that is this document's oracle, and none of it
is a gap here.
*Verified-by:* `packages/sql-transform/sql_transform/_projection.py:188-189, :410`
(`SET threads = 1`); `packages/confit/docs/kpis.md:31-34` (C1), `:40-41` and `:76-87`
(C4); `packages/confit/docs/properties.md:118-121` (P11);
`docs/specs/2026-08-05-fit-transform-split-design.md:124` and P16 (fit reproducibility
is practice, not contract).

**ORC-73.** The **dialect gates** are out of scope here, and they run a *second* pinned
oracle: the L3 gate executes the printed query on Spark under a pinned configuration
(`ansi=true`, UTC, `local[1]`, pinned in `pins-dialect/spark-ansi.json`), compares
column names plus the row multiset, keeps an exact tier with a *designed* epsilon tier
reserved for float-accumulation aggregates ("extend `rows_of`, do not weaken the exact
tier"), skips loudly without BigQuery credentials, and carries a **ratchet**: "The match
floor is the measured count at introduction — raise it when the surface grows, never
lower it." Two of this document's open questions have their nearest precedent here —
a tiered comparison (ASK-6) and a ratchet (ASK-11).
*Verified-by:* `packages/confit/tests/test_dialect_cross_engine_gate.py:1-31`;
`packages/confit/docs/specs/2026-08-13-dialect-logical-plan-design.md:32-36, :244-248`
(the per-dialect oracle identity table, including BigQuery as "unversionable").

**ORC-74.** DuckDB fills **three** roles in this project and this document defines only
one of them. Besides the differential oracle it is (a) the **parser and printer** —
`json_serialize_sql` / `json_deserialize_sql`, pinned per DuckDB version in
`sql_transform/model/_shapes.json`, with the corollary that an identifier means what the
oracle binds; and (b) the **build-time evaluator** on the static-tables-only path, where
the query is handed to DuckDB once at build and the answer frozen. Role (b) is why
ORC-17 and ORC-22 exist at all: on that path parity is *identity*, not comparison.
*Verified-by:* `packages/confit/docs/properties.md:86-112` (P9, role (a));
`scripts/pin_ast_shapes.py`; `packages/confit/src/duckdb/mod.rs` `eval_static_only`,
`packages/confit/docs/known-limitations.md:109-112`, and the founding design
`packages/confit/docs/specs/2026-07-25-sql-specializer-design.md:97` (three roles).

---
