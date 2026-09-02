## 1. What the oracle is

### 1.1 It is a pseudo-oracle

**claim: pseudo-oracle.** The oracle answers exactly one question: *does confit produce
the same answer as this DuckDB build, run this way?* It never answers *is this correct?*
Authority is delegated to a reference implementation, so DuckDB's bugs are inherited by
construction and reproducing one is conformance, not a defect. An argument that DuckDB's
answer is wrong is out of scope of this document by design; it is an upstream matter,
and it changes nothing here until the pinned build changes.
*Verified-by:* the two-outcome contract,
`packages/confit/docs/reports/pins-first-methodology.md:5-16`; P18 in
`packages/confit/docs/properties.md:231-235`.

**claim: oracle-identity.** The oracle is one constant, and both halves are
load-bearing:

```
ORACLE = DuckDB 1.5.5, PRAGMA disable_optimizer, all other settings default
```

Neither half is a per-test choice, a per-campaign flag, or a thing a caller may
vary. A comparison run against anything else is not a comparison against the oracle.
*Enforced-by:* `confit.oracle.Oracle.__init__` — the constructor applies the pragma,
and it is the only way to build an oracle.
*Verified-by:* the **pragma** half at
`packages/confit/tests/test_oracle.py::test_construction_applies_the_pragma` (the probe
is plan-shaped, because `PRAGMA disable_optimizer` writes no setting `current_setting`
can read back). The **version** half is recorded as `Oracle.VERSION` and asserted
nowhere — see claim: oracle-version-constant. The **"all other settings default"** half
is `Unverified` and is measurably weaker than it reads: DuckDB's `threads` default is
derived from core count (measured 12 on this machine), and it changes answers — see
claim: threads-setting and ask: threads-and-value-order.

### 1.2 Why optimizer-off

**claim: optimizer-on-reading.** The optimizer-on reading is not a function of the
query, so it cannot be a target. `statistics_propagation` reads a column's stored null
statistic, so DuckDB answers the same query over the same rows differently depending on
the table's insert history — measured: a table built as `[-128, NULL]` with the NULL
then deleted answers differently from one built as `[-128]`, with identical contents.
confit compiles once against a *schema* and serves many batches; it never sees a table,
let alone its history.
*Verified-by:* `packages/confit/tests/known_divergences/test_trap_elision.py`;
`packages/confit/docs/known-limitations.md:32-39`.

**claim: unoptimized-verifier.** Optimizer-off is a *sanctioned* reference leg upstream,
not our invention. `PRAGMA enable_verification` registers an `UNOPTIMIZED` statement
verifier alongside COPIED / DESERIALIZED / NO_OPERATOR_CACHING and compares its result
against the original's; the function's own comment states the purpose as "Correctness of
plans both with and without optimizers". DuckDB itself treats disagreement between the
two legs as a bug in DuckDB.
*Verified-by:* DuckDB v1.5.5 source, `src/function/pragma/pragma_functions.cpp:134`
(`enable_verification` pragma), `src/main/client_verify.cpp:45` (the comment), `:55`
(the UNOPTIMIZED verifier), `:187` -> `src/verification/statement_verifier.cpp:155`
(`CompareResults`).

**claim: disable-optimizer-scope.** What `PRAGMA disable_optimizer` removes, precisely.
It disables the 33 named optimizer passes (`OptimizerType` enumerates
`EXPRESSION_REWRITER = 1` through `WINDOW_SELF_JOIN = 33`), **and additionally** changes
behavior at **twelve** sites that read `enable_optimizer` directly, outside the pass
list, in three groups:

| group | sites | what changes with the optimizer off |
|---|---|---|
| physical-operator selection | `plan_distinct.cpp:66`, `plan_window.cpp:31` (read into a local, used at `:35` — **one** site, two lines), `sorted_aggregate_function.cpp:686` and `:744` (both reached from `plan_aggregate.cpp:318`) | the DISTINCT-ON ordered-aggregate rewrite, streaming-vs-blocking window operators, sorted-aggregate `ORDER BY` simplification in two places |
| window-function execution | `window_aggregate_function.cpp:32` and `:56`, `window_rank_function.cpp:24`, `window_rownumber_function.cpp:26`, `window_value_function.cpp:207` and `:849` | window aggregation strategy and the rank/rownumber/value fast paths |
| logical-plan construction inside the binder | `plan_subquery.cpp:255`, `plan_joinref.cpp:411` | with the optimizer off, correlated subqueries **always** take a delim join; and a `RIGHT` outer join is **not** flipped to a `LEFT` with the sides swapped |

What is genuinely untouched is **expression binding**: nothing in name resolution,
overload selection or type inference reads the flag, so output types and bind-time
errors are identical, and so is execution-level laziness (an untaken `CASE` arm,
`AND`/`OR` short-circuit in a filter). The two `src/planner/binder/` sites are logical
*plan* construction performed by the binder, not binding, which is why types survive
them — but "none in binding" is the wrong sentence for them, and the join flip is the
axis that claim: join-output-order says decides hash-join output order. Both are
reachable from the **row path**, not only from the static-tables-only path.
*Verified-by:* DuckDB v1.5.5 source,
`src/include/duckdb/common/enums/optimizer_type.hpp:16-50` (33 members;
`EXPRESSION_REWRITER = 1` at `:18`, `WINDOW_SELF_JOIN = 33` at `:50`); the twelve
sites above, enumerated by grep over `src/` 2026-08-25, each line read — thirteen line
references, because `plan_window.cpp:31` and `:35` are one site.
*Correction, two parts, both against `confit/oracle.py`'s module docstring — which is
where the rationale now lives.* (a) "`PRAGMA disable_optimizer` == disabling all 33
named optimizers" (and `known-limitations.md:30`'s "What it removes is the 33
plan-rewrite passes") is incomplete by these twelve sites. (b) "constant folding still
happens (`1 + 2` is int32 3)" is true as an **observation** and wrong as a
**mechanism**: DuckDB's only constant folder is
`src/optimizer/rule/constant_folding.cpp`, an `EXPRESSION_REWRITER` rule, so the pragma
removes it — `1 + 2` is still int32 3 because the expression is evaluated at run time
instead of folded into the plan. Where that distinction is observable, it is observable:
`SELECT 2147483647 + 1 ... LIMIT 0` errors with the optimizer off and serves `[]` with
it on, because with the optimizer on the plan collapses to `EMPTY_RESULT`.
claim: phase-separated-probes' own source says the same thing. See proposed
ticket: oracle-docstring-corrections.

**claim: contract-surface-gap.** The user-facing contract still names what a user's
DuckDB returns, which is optimizer-*on*. The gap between the two readings is therefore
user-visible and stays a reported finding (`DIVERGE_OPT`) rather than an accepted class.
The oracle and the contract surface are deliberately not the same thing.
*Enforced-by:* `fuzz.runner.INTERESTING` (which contains `DIVERGE_OPT`).
*Verified-by:* the *kind* exists and is reachable —
`packages/confit/tests/test_fuzz_smoke.py::test_verdicts_cover_the_contract_and_reproduce`
(`v.kind in oracle.KINDS`). Its **membership in `INTERESTING`** is `Unverified`: no test
in `packages/confit/tests/` imports `fuzz.runner` at all (measured 2026-09-02), so
deleting `DIVERGE_OPT` from that tuple breaks nothing. Proposed
ticket: verdict-tuple-test.

### 1.3 How the identity is enforced

**claim: no-raw-connections.** Everything that compares against DuckDB gets its
connection from `confit.oracle.Oracle`, and a raw `duckdb.connect(` anywhere in `tests/`
or `fuzz/` is a gate failure. The oracle is a property of the repo rather than a
per-call-site choice that can be forgotten, and a new comparison site gets the oracle by
construction.
*Enforced-by:* `confit.oracle.Oracle.__init__`; the ban itself is read off the
**sources** rather than applied at run time, because the door is shared — see
claim: one-door-bypass.
*Verified-by:*
`packages/confit/tests/test_oracle.py::test_no_raw_connections_in_the_sources`, which
walks `tests/**/*.py` and `fuzz/**/*.py` and so also covers the tests a run never
reaches.

**claim: one-door-bypass.** **[FACT]** On the **comparison** path the one-door property
has exactly **one** known bypass, and it is the engine's own. `eval_static_only`
(`packages/confit/src/duckdb/mod.rs:1178`) folds a static-tables-only query at build
time by calling `duckdb.connect()` itself, with the optimizer **on**, which is what
production does; the oracle it is then compared against is optimizer-off. Pin *capture*
is a second, separate family and is outside claim: no-raw-connections' `tests/` and
`fuzz/` scope entirely — claim: capture-outside-the-oracle counts four `scripts/`
connects there, and none of them is covered by this claim.

The bypass is why the ban on raw connections cannot be a runtime patch of
`duckdb.connect`: a Python frame cannot tell the engine's fold from a test's connection,
so a patch either refuses the engine's own fold or silently folds those queries
optimizer-off while production folds them optimizer-on. The second is what the deleted
autouse fixture did.
*Verified-by:* `packages/confit/src/duckdb/mod.rs`, `eval_static_only` — the bare
`connect` at `:1184` with no pragma, which is the fact itself. The
*frame-indistinguishability* measurement behind the second paragraph is recorded only in
`packages/confit/tests/conftest.py`'s module docstring (`:7-19`), so by the front
matter's rule that half reads `Unverified`: nothing fails if it stops being true.
*Measured in part, and the half a ruling turns on is still the missing one.* One bounded
measurement of observability exists, and it is the campaign's static-only leg: that leg
**is** this comparison, grading the engine's optimizer-on fold against both oracle
readings, and over seeds 0-1999 all **35** static-only cases that build agree under both
readings with 0 schema deltas (the other 28 refuse at build). Its limits are the point:
that is one generated grammar — aggregates over static columns, no literals — not a run
of the suite. **No run of the suite under both readings is recorded anywhere in the
tree** — searched 2026-09-02 across `packages/confit/docs/`, `backlog/` and the tests. So
observability beyond the campaign grammar is still an assumption, and
ask: engine-fold-reading should not be answered on it.
*Verified-by (the campaign half):* the leg's nature is pinned by
`packages/confit/tests/test_fuzz_smoke.py::test_the_static_only_leg_has_no_unshipped_width_to_classify`;
the 35/35 count is a dated measurement (2026-09-02, seeds 0-1999), not a gate.
*Status:* stated, not ruled. See ask: engine-fold-reading.

> ### ask: engine-fold-reading — does the engine's build-time fold move to the oracle's reading?
>
> claim: one-door-bypass is a fact, not a disposition. The engine folds a static-only
> query with the optimizer ON (production's reading), and every gate then compares that
> frozen answer against an optimizer-OFF oracle. This is a question about which reading
> the frozen artifact is *supposed* to be. Three shapes it could take: the fold stays
> optimizer-on and the spec says the constant path deliberately freezes the user-visible
> reading (claim: contract-surface-gap's surface, not claim: oracle-identity's oracle);
> the fold moves to optimizer-off so that one identity covers both paths; or the
> difference is declared unobservable and gated by a test that says so.
>
> **First, the measurement — taken in part, and the cheap remainder is still cheap.**
> The campaign's static-only leg already runs this comparison, and it bounds the answer
> inside the generated grammar: 35 of 35 static-only cases that build over seeds 0-1999
> agree across both readings, with 0 schema deltas (claim: one-door-bypass). That grammar
> reaches aggregates over static columns and nothing else, so it is evidence that the
> difference is hard to observe, not evidence that it is unobservable. Whether the two
> readings differ on today's **suite** is still unrecorded, and the third option above
> cannot be chosen without it — a test asserting unobservability is only writable once
> someone has run the suite both ways. It is one pragma in `eval_static_only` and one
> run.
>
> Whichever way it goes, the answer belongs in claim: oracle-identity, because today the
> constant is stated as though it had no exceptions.
>
> *Binds:* claim: oracle-identity, claim: contract-surface-gap,
> claim: no-raw-connections, claim: row-limit-refusal,
> claim: build-vs-build-repeatability, claim: duckdb-three-roles' role (b),
> claim: one-door-bypass.

**claim: optimizer-flip-in-place.** A caller that *wants* the optimizer says so in its
own body, which reads as the deliberate exception it is. The flip happens in place, on
the same connection, so that the two readings of a differential comparison cannot differ
because `statistics_propagation` read different per-column statistics.
*Enforced-by:* `confit.oracle.Oracle.optimizer_on`.
*Verified-by:*
`packages/confit/tests/test_oracle.py::test_optimizer_on_flips_the_same_connection`;
`packages/confit/tests/known_divergences/test_trap_elision.py:458, :560` holds exactly
two such exceptions and is the test that documents what the optimizer does. Measured
2026-09-02, `optimizer_on` has exactly four call sites: those two, the campaign's
`fuzz.oracle._duck_run:433`, and the flip's own test above.

**claim: oracle-version-constant.** **[FACT]** The version half of the identity is
**recorded, not asserted**. `confit.oracle.Oracle.VERSION` is `"1.5.5"` and nothing
compares it to `duckdb.__version__`; the manifests declare a floor (`duckdb>=1.5.5` at
`pyproject.toml:15` and `packages/sql-transform/pyproject.toml:10`;
`packages/confit/pyproject.toml` declares no duckdb dependency at all), and only
`uv.lock` resolves 1.5.5 exactly. A `uv lock --upgrade` silently re-points the oracle
and no gate notices. **36 markdown files outside `backlog/`** name DuckDB 1.5.5 in prose
(48 counting `backlog/`), measured 2026-08-25.
*Verified-by:* `packages/confit/confit/oracle.py:74-76` (the constant and the comment
reserving the assert — source, not prose); measured 2026-08-25 — `pyproject.toml:15`,
`packages/sql-transform/pyproject.toml:10`, `packages/confit/pyproject.toml`
(dependencies: `pyarrow>=19.0` only), `uv.lock:368-370`. No test in
`packages/confit/tests/` reads `Oracle.VERSION`. The fix is ask: version-pin.

> ### ask: version-pin — pin `==1.5.5` or keep the floor; and 1.5.5 or the LTS line?
>
> **(a)** Hard-pin `duckdb==1.5.5` in the manifests, or keep the floor and assert. The
> landing spot is reserved and is one line in `Oracle.__init__`, beside the pragma it
> already applies:
>
> ```python
> assert duckdb.__version__ == VERSION, f"oracle is {VERSION}, got {duckdb.__version__}"
> ```
>
> **(b)** 1.5.5, or the LTS line? DuckDB ships minor versions on a roughly 4-month
> cadence, semantics have already moved inside a patch release, and v2.0 brings a new
> SQL parser. Section 9's bump protocol is cheap now and expensive during a migration;
> which version it targets is your call.
>
> *Binds:* claim: oracle-identity, claim: oracle-version-constant,
> claim: capture-outside-the-oracle, and every pin in the corpus.

### 1.4 The three excluded neighbours

**claim: fit-serving-oracle.** The **fit/serving oracle** is out of scope here.
`sql_transform`'s projection path runs DuckDB with the optimizer **on** and at `SET
threads = 1`, and its parity targets are KPI C1 (fit + serving over the training set is
bit-exact against the original SQL) and KPI C4 (transformer columns, which DuckDB cannot
run at all, gate against an independent clone-per-group sklearn reference). Fit
reproducibility itself is explicitly out of contract in v0. None of that is this
document's oracle, and none of it is a gap here.
*Verified-by:* `packages/sql-transform/sql_transform/_projection.py:188-189, :410`
(`SET threads = 1`); `packages/confit/docs/kpis.md:31-34` (C1), `:40-41` and `:76-87`
(C4); `packages/confit/docs/properties.md:118-121` (P11);
`docs/specs/2026-08-05-fit-transform-split-design.md:124` and P16 (fit reproducibility
is practice, not contract).

**claim: dialect-gate-oracle.** The **dialect gates** are out of scope here, and they
run a *second* pinned oracle: the L3 gate executes the printed query on Spark under a
pinned configuration (`ansi=true`, UTC, `local[1]`, pinned in
`pins-dialect/spark-ansi.json`), compares column names plus the row multiset, keeps an
exact tier with a *designed* epsilon tier reserved for float-accumulation aggregates
("extend `rows_of`, do not weaken the exact tier"), skips loudly without BigQuery
credentials, and carries a **ratchet**: "The match floor is the measured count at
introduction — raise it when the surface grows, never lower it." Two of this document's
open questions have their nearest precedent here — a tiered comparison
(ask: float-tolerance-list) and a ratchet (ask: match-count-ratchet).
*Verified-by:* `packages/confit/tests/test_dialect_cross_engine_gate.py:1-31`;
`packages/confit/docs/specs/2026-08-13-dialect-logical-plan-design.md:32-36, :244-248`
(the per-dialect oracle identity table, including BigQuery as "unversionable").

**claim: duckdb-three-roles.** DuckDB fills **three** roles in this project and this
document defines only one of them. Besides the differential oracle it is (a) the
**parser and printer** — `json_serialize_sql` / `json_deserialize_sql`, pinned per
DuckDB version in `sql_transform/model/_shapes.json`, with the corollary that an
identifier means what the oracle binds; and (b) the **build-time evaluator** on the
static-tables-only path, where the query is handed to DuckDB once at build and the
answer frozen. Role (b) is why claim: row-limit-refusal and
claim: build-vs-build-repeatability exist at all: on that path parity is *identity*, not
comparison — and it is the one place the engine opens its own DuckDB connection, which
is claim: one-door-bypass.
*Verified-by:* `packages/confit/docs/properties.md:86-112` (P9, role (a));
`scripts/pin_ast_shapes.py`; `packages/confit/src/duckdb/mod.rs` `eval_static_only`,
`packages/confit/docs/known-limitations.md:109-112`, and the founding design
`packages/confit/docs/specs/2026-07-25-sql-specializer-design.md:97` (three roles).

---
