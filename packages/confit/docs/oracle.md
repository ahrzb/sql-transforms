# The oracle spec

**What this document is.** The definition of *correct* for the confit engine: what
the oracle is, what it decides, what it declines to decide, and how to compare
against it. It is a consolidation, not a proposal — where a decision is already in
force it is written down here so the next reader stops re-deriving it.

**Governance.** The oracle spec states what is considered correct. Every
contradiction goes through the owner. This document therefore keeps three kinds of
content strictly apart:

- **Normative claims** — settled decisions in force, each pointing at the decision
  that made it. Each carries a stable id `ORC-NN` and a `Verified-by` pointer.
  Nothing unmarked here is new.
- **ASK blocks** — questions only the owner can answer, placed at the point in the
  document where the answer would bind. An ASK is never phrased as decided, and no
  normative claim depends on one.
- **Editorial** — framing, indexes, corrections to other documents. Unnumbered.

**Two markers keep the first category honest.** A claim carrying either marker is
*not* a decision in force:

- **`[PROPOSED]`** — a rule this document would like, which nobody has ruled on. It
  holds an id only so a pin or a ticket can reference it. Several of these are rules
  this document originally stated as though they were already in force; they are
  marked now, and adopting them is ASK-15. **15 claims** carry this marker.
- **`[FACT]`** — a measured statement of current state with no decision attached.
  It is here because the state is load-bearing, not because anyone ruled on it.
  **8 claims** carry this marker.

Of 89 claims, 66 are decisions in force.

Every other `ORC-NN` is a decision in force, made somewhere outside this document,
and its `Verified-by` names where.

**How to read a claim.** One claim per paragraph block, so a content-hash field can
be added later by tooling without re-cutting the text. `Verified-by` names a test
path, a pin file, a source line, or a measurement; where nothing verifies a claim it
says `Unverified` and says so plainly. "Derived" and "normative here" are not
verification and no longer appear: a claim that only this document asserts is
`[PROPOSED]`. Behaviors carry a status:

| status | meaning |
|---|---|
| `PINNED` | the oracle's answer is stable and it is the contract |
| `IMPL-DEFINED` | stable for the pinned build and configuration, fragile across versions or platforms; the pin names the discriminator |
| `UNSPECIFIED` | not stable; we refuse, normalize, or exclude, and never claim bit-for-bit |

**Completeness** here means *decision coverage*: every oracle fact that was decided
appears in this document. Emergent behavior owes nothing. A gap is a decision that
was made somewhere and is not written here — not a behavior nobody has ruled on.

**Scope, so that "complete" has edges.** This document covers **confit's DuckDB
oracle**: the identity in ORC-02, everything compared against it, and every gate that
does the comparing (`packages/confit/tests/`, `packages/confit/fuzz/`, the pins
corpus). Three neighbouring oracles exist in this repo and are **explicitly excluded**,
each with a pointer rather than a silent omission — ORC-72, ORC-73 and ORC-74 name
them. Their decisions are real decisions; they are simply not this document's.
Anything else that is an oracle fact and is not here is a gap.

**Doc homes.** confit's docs live under `packages/confit/docs/`. A spec, ticket or
comment citing a bare `docs/known-limitations.md` is a stale path (the move merged as
master `85b4739`).

---

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

## 2. Inherited quirks

**ORC-10.** Where DuckDB's behavior is a quirk, the quirk is reproduced, not fixed.
Because the contract is bit-for-bit DuckDB rather than the SQL standard, DuckDB's
oddities *are* our normative behavior, including ones DuckDB would call bugs.
*Verified-by:* `packages/confit/docs/reports/pins-first-methodology.md:39` ("pins are
engine==oracle contracts").

**ORC-11.** The enumerated quirks — *the oddities whose disposition needed a decision*,
not every descoped construct. This list exists so a future reader who meets an inherited
oddity has something to check it against before "fixing" it. It is short today and cheap
to write; reconstructing it later is not. The fuller descope list, including `#`,
`NOT GLOB` and the twelve fuzzer-found regex reject classes, is
`known-limitations.md:190-207`; a construct there but not here was descoped without an
inherited-oddity ruling to record.

| quirk | DuckDB's behavior | what we do | evidence |
|---|---|---|---|
| `^` | is `pow`, not bit-xor | descoped: sqlparser's precedence differs, so mapping it computes a wrong tree silently. Use `pow()`. | `known-limitations.md:197` |
| `~` | full-match, not search | reproduced | `pins-first-methodology.md:39` |
| `SIMILAR TO` | no wildcard translation | reproduced | `pins-first-methodology.md:39` |
| `SIMILAR TO ... ESCAPE` | not implemented in DuckDB itself | refused | `known-limitations.md:201` |
| `reverse()` | byte-reverses all-ASCII input, splitting CRLF (`'a\r\nb'` -> `'b\n\ra'`), violating UAX-29; only non-ASCII takes the grapheme path | both paths reproduced | `pins-waveA/reverse-graphemes.json`; `specs/2026-07-28-waveA-structural-tails.md` section 4 |
| paren-less `* REPLACE e AS c` | consumes exactly one item; a following comma starts a new select item, yielding a duplicate name | reproduced | `pins-waveA/columns-replace.json` |
| double-quoted identifiers in struct `EXCLUDE` | still case-insensitive: `a.* EXCLUDE("J")` removes field `j` | reproduced | `pins-waveA/struct-star.json` |
| `* EXCLUDE (t.key)` on a `USING` join | UNMERGES the coalesced column, which reappears at the right table's position | descoped (measured, not modeled) | `known-limitations.md:202` |
| `BETWEEN`/`IN` mixing non-numeric strings with numbers | converts at EXECUTION time, so an empty input succeeds | conservatively refused | `known-limitations.md:203` |
| `repeat(NULL, n)` on a bare NULL | picks the **BLOB** overload | refused; `CAST(NULL AS VARCHAR)` types identically on both | `known-limitations.md:207` |
| `\B` in a regex | crashes DuckDB at runtime on non-ASCII | reject-listed | `known-limitations.md:199` |
| `$` anchor in non-final position | the row path literal-optimizes `$`+literal into a PREFIX match while DuckDB's own constant fold matches normally — the oracle disagrees with itself | rejected by name (section 3.6) | `pins-waveB/fuzzer-20260728.json` |

*Verified-by:* each row cites its own evidence; the reproduce-don't-fix rule is ORC-10.

**ORC-12.** Meeting an inherited oddity that is not in the table above is a report,
not a fix. Adding a row is a decision and goes through the owner, because "this looks
wrong" and "this is a divergence" are the same observation until somebody measures.
*Verified-by:* the owner's standing governance rule — the oracle spec states what is
considered correct, and every contradiction goes through the owner — which is what makes
adding a row a decision rather than an edit; `known-limitations.md:284-285` ("If a
message you hit isn't in this document or the tests, that's a bug in our bookkeeping —
file it") is the existing half that makes it a report.

---

## 3. Nondeterminism

This is the section that converts future arguments into lookups. Read 3.1 and 3.2
before ruling on anything new.

### 3.1 The axiom

**ORC-13.** *An answer that is not a function of the query — plus the frozen statics —
is not a target.* This is the single axiom under every nondeterminism ruling in the
project. It is why the oracle is optimizer-off (ORC-03), why row limits on the
constant path refuse (ORC-17), why row order is compared per mode rather than by
byte-equality (ORC-18), and why statistics-dependent behavior is excluded (ORC-20).
*Verified-by:* three existing phrasings of the same rule —
`packages/confit/docs/known-limitations.md:39`, `packages/confit/fuzz/oracle.py:40-41`,
`backlog/tasks/task-128 ...md` description ("The doctrine already exists").
*Note:* `packages/confit/docs/properties.md` ends at P20, so **P21** is the free number
if the owner wants this numbered as a project property and the three sites made to cite
it. Proposed ticket T-3.

### 3.2 The disposition table

**ORC-14.** Five kinds of nondeterminism, five dispositions, all five already in
force. The last column is the operating instruction: what a *new* case of that kind
gets, without re-litigating from the axiom.

| what is nondeterministic | disposition | in force as | a NEW case gets |
|---|---|---|---|
| **which rows exist** | REFUSE by name at build | TASK-128 row limits (ORC-17) | a named build-time refusal, `ORDER BY` or not |
| **row order** | declare unspecified, compare per mode | TASK-129 compare modes (ORC-18) | a compare mode plus a DuckDB-free self-leg, never byte-equality |
| **values, from data outside the query** | structurally removed where possible, otherwise excluded by source name | optimizer-off structurally (ORC-03); the three named exclusion sets (ORC-20) | exclusion by name, carrying a measured reason |
| **the oracle disagrees with itself, across evaluation paths** | reject the construct by name | regex anchor families (ORC-21) | a named refusal — there is no behavior to match |
| **the oracle disagrees with itself, across its own builds** | a bounded, named tolerance | `cbrt` at <= 1 ulp (ORC-76) | a declared bound, named in the pin as the only such exception |

Two things this table does not yet do, both real and both owner-reserved. It has **no
tiebreaker** when more than one row applies — the optimizer gap fits rows 3 and 4 and is
dispositioned as neither (ORC-06 reports it as a finding). And it has **no row for
nondeterminism inside a single value** — order within a `string_agg` or `list`, which
`sort-at-freeze` cannot fix because sorting rows does not sort inside a string. Both are
ASK-13.
*Verified-by:* each row's own claim below.

**ORC-15.** **[PROPOSED]** Not in force. Every comparison target carries one status from
the vocabulary in the front matter (`PINNED` / `IMPL-DEFINED` / `UNSPECIFIED`), and a
target with no status is not yet a target. Today the vocabulary is applied to eight
claims and to the section 7 ledger's `proposed status` column; ORC-16, ORC-32, ORC-33,
ORC-36, ORC-37, ORC-38, ORC-39 and every ORC-11 quirk carry no status, so under this
rule as written they are not targets — which they plainly are. Adopting the rule means
either statusing them or narrowing the rule to the ledger.
*Verified-by:* Unverified — no rule outside this document requires a status. Part of
ASK-15.

### 3.3 Which rows exist

**ORC-16.** Row order on the *serving* path is not nondeterministic and is part of the
contract: output rows follow input rows — `map` exactly (`out[i] <-> in[i]`), `filter`
as a subsequence, `many` as per-input-row blocks in input order. That order comes from
the serving contract, not from SQL, and is checked by DuckDB-free self-legs.
*Verified-by:* `packages/confit/docs/known-limitations.md:41-51`;
`packages/confit/fuzz/oracle.py:799-836` (batch-vs-single sequence leg and reversal
leg); `packages/confit/tests/test_fuzz_order_legs.py` (the capability test that proves
the legs catch a scrambler).

**ORC-17.** A row limit on a static-tables-only query refuses at build, by name,
`ORDER BY` or not: `LIMIT`, `OFFSET`, `FETCH`, `TOP`, anywhere in the statement
including CTEs, derived tables and both sides of a set operation. Which rows survive a
limit is not a function of the query — measured, the same
`GROUP BY ... FETCH FIRST 1 ROWS ONLY` over the same four rows answered **four
different ways across twelve fresh connections**, and `ORDER BY` does not fix ties (a
tie fed from a `GROUP BY` flipped in 20 runs, while a unique sort key was stable at
1/20). Freezing whichever answer the build-time run happened to get would make two
builds of the same function disagree. Status: `UNSPECIFIED`, refused.
*Verified-by:* `packages/confit/docs/known-limitations.md:109-120`;
`backlog/tasks/task-128 ...md` (Done, decision (a), AC #1-#5);
`packages/confit/tests/test_arrow_schema_api.py:614-639`.
*Residual hole, named:* a query sqlparser cannot parse cannot be inspected and falls
through as before. Believed tiny — every row-limit spelling sqlparser knows does parse.

### 3.4 Row order

**ORC-18.** Order sensitivity is determined per query, and there are exactly three
compare modes. Values are bit-for-bit in all three; only the *sequence* rule differs.

| mode | when | the check |
|---|---|---|
| `row-path` | any query with a dynamic table | multiset against DuckDB (its order is not a function of the query even here), plus the sequence self-legs of ORC-16 |
| `constant-ordered` | static-only with a top-level `ORDER BY` | multiset equality **plus** our-side sortedness on the key — never byte-equality, because ties make DuckDB's sequence one of several valid answers |
| `constant-unordered` | static-only, no `ORDER BY` | multiset; SQL defines no order and neither do we |

Status: `UNSPECIFIED` for the sequence outside the row path and outside a total
`ORDER BY`; `PINNED` for values in all three modes.
*Verified-by:* `packages/confit/fuzz/oracle.py:432-449` (`compare_mode`), `:452-464`
(`_sorted_by`, DuckDB defaults: ASC, NULLS LAST, NaN above every number),
`:676-698` (the static-only branch); `backlog/tasks/task-129 ...md` (Done, AC #1-#7);
`packages/confit/docs/known-limitations.md:41-51`.

**ORC-19.** Join output order is a measured hash-join accident on three independent
axes (cost-chosen streamed side, LIFO chains emitting duplicate-key matches in reverse
build-insertion order in per-2048-row lockstep passes, and run-to-run variation at
multiple threads with ~500k+ rows). Parity for `shape='many'` is therefore
**multiset**, and the engine defines its own documented deterministic order: probe rows
in input order, matches contiguous in build-insertion order. Chasing byte-order parity
here would mean chasing a nondeterministic target. Status: `UNSPECIFIED` upstream,
`PINNED` on our side as our own published order.
*Verified-by:* `packages/confit/docs/specs/pins-stageB/order-contract.json`;
`packages/confit/docs/specs/2026-07-28-stageB-multiplicity-pins.md`;
`packages/confit/docs/reports/pins-first-methodology.md:35`.

### 3.5 Values from outside the query

**ORC-20.** Statistics-dependent behavior is excluded from the oracle, by source name,
each exclusion citing a measured reason it is irreproducible row-locally. It is **one of
three** exclusion sets in the corpus gate, not the only one — see ORC-77. The measured
exemplar: DuckDB's `ILIKE` result for a NUL-containing row depends on *sibling* rows —
pure-ASCII column statistics select a NUL-safe ASCII kernel (the row matches itself,
TRUE) while any non-ASCII sibling selects the generic kernel whose fold NUL-truncates
(same row, FALSE). A row-at-a-time engine cannot reproduce this even in principle; the
engine is NUL-transparent (the ASCII-kernel behavior). Status: `UNSPECIFIED`, excluded.
*Verified-by:* `packages/confit/tests/test_corpus_replay.py:38-49`
(`_KNOWN_DIVERGENT_SOURCES`, with the reason in-file);
`packages/confit/docs/specs/pins-wave1/pins_like.json` (measured 2026-07-26);
`packages/confit/docs/known-limitations.md:225-230`.
*Correction:* the exclusion set holds **one** source file, contributing two corpus
statements (measured 2026-08-25 over `tests/corpus/duckdb_mined.jsonl`).
`pins-first-methodology.md:89` says "Two such sources exist" — false;
`known-limitations.md:225` says "Two known oracle divergences", which is true only if
read as statements, not sources. See proposed ticket T-4.

**ORC-77.** The corpus gate carries **three** decided exclusion sets, and only the first
is about statistics. (a) `_KNOWN_DIVERGENT_SOURCES` — irreproducible-row-locally, one
source, two statements (ORC-20). (b) The **f32 base-table blanket rule**: any mined case
whose input table has a FLOAT column is clean-unsupported, because widening to f64 is
value-exact but every f32-*grid*-sensitive operation (`nextafter`'s ulp steps,
`FLOAT`->`VARCHAR` shortest round-trip, FLOAT rounding) then computes on the wrong grid;
the blanket rule was chosen over a per-operation one on the measured ground that
"comparisons happen to survive" (wave-3: 3 sources, 5 cases). (c) `_INEXPRESSIBLE_INPUTS`
— the SQL is fine, the *declared* input schema is not; one entry, whose in-file comment
records that its original ground (the width-less pydantic row surface) has since gone
away.
*Verified-by:* `packages/confit/tests/test_corpus_replay.py:38-49` (a), `:111-117` (b),
`:60-65` (c). Sets (b) and (c) were absent from an earlier version of this document; (c)
is a live instance of the ORC-53 REASON rule and is named in ASK-15.

### 3.6 The oracle disagreeing with itself

**ORC-21.** Where DuckDB's own evaluation paths return different answers for the same
input, there is no behavior to be bit-exact *with*, so the construct is rejected by
name with the measurement recorded. The two measured families are anchor-only
multi-anchor patterns and `$` anchors in non-final position, where the row path
literal-optimizes into a PREFIX match while DuckDB's own constant fold matches
normally. Status: `UNSPECIFIED`, refused.
*Verified-by:* `packages/confit/docs/reports/pins-first-methodology.md:78, :87`;
`packages/confit/docs/specs/pins-waveB/fuzzer-20260728.json`;
`packages/confit/docs/known-limitations.md:200`.
*Scope:* this covers the oracle disagreeing across its own **evaluation paths** within
one build. The oracle disagreeing across its own **builds** is a different species with
a different, already-decided disposition — ORC-76.

**ORC-76.** Where DuckDB's own wheels disagree with each other, the disposition is a
**bounded, named tolerance**, not refusal. The measured instance: `cbrt`. The Windows
wheel matches Rust/ucrt bit-exactly while the Linux wheel's bundled `std::cbrt` is one
ulp off (`cbrt(27)` = `3.0000000000000004`), CI-discovered 2026-07-26. The oracle is
platform-inconsistent there, so repr-exact parity is unpinnable; oracle parity for
`cbrt` is pinned at **<= 1 ulp**, and the wave-1 pins spec records it as "the only such
exception". The engine itself stays deterministic (Rust `cbrt`). Status:
`IMPL-DEFINED`, discriminator = the oracle's own build.
*Verified-by:* `packages/confit/tests/test_duckdb_interpreter.py:918-952`
(`duck_check_ulp`, `max_ulp=1`), used by `test_sqrt_cbrt_bigint` and
`test_cbrt_total_function` — both in the normal test gate;
`packages/confit/docs/specs/2026-07-26-wave1-builtin-pins.md:47-52` (the ground and the
"only such exception" wording).

### 3.7 Two holes that are not ruled

**ORC-22.** **[FACT]** Build-vs-build repeatability of a frozen artifact is **explicitly
undecided**, and recorded as such rather than as a guarantee. Two builds of the same
function disagreeing on frozen row order would flake downstream golden tests. Raw
DuckDB gave 12 distinct row orders over 12 fresh connections on an unordered 200-group
`GROUP BY`; our arrow materialization measured stable over 6 builds, which the ticket
records honestly as "luck, not contract". `sort-at-freeze` is the named artifact-level
fix and is orthogonal to the oracle definition.
*Verified-by:* `backlog/tasks/task-129 ...md` (the SEPARATE-but-adjacent paragraph and
AC #3's probe carried over from `task-128`).
*Scope, named:* this hole is about frozen **row** order. The adjacent hole about order
*inside a frozen value* is ORC-75, and `sort-at-freeze` does not reach it. See ASK-2 and
ASK-13.

**ORC-75.** **[FACT]** The `threads` setting changes frozen answers, and the oracle
constant leaves it at DuckDB's core-derived default. Measured 2026-08-25 under
`PRAGMA disable_optimizer`, DuckDB 1.5.5: `current_setting('threads')` is **12** on this
machine, and the within-group element order of `string_agg(v, ',') ... GROUP BY g`
over a 400k-row table is a function of it — stable per setting, different between 1/2 and
4/8. The construct is live rather than hypothetical: a `string_agg` over a static table
builds today on the constant path (backend `constant`) and freezes whatever order that
machine produced. So "all other settings default" (ORC-02) makes the oracle
machine-dependent for thread-sensitive aggregates, and a 4-core runner and a 12-core dev
box are different oracles for them. The project already has the counter-setting in force
on its other oracle — `SET threads = 1`, on the measured ground that DuckDB's parallel
window aggregation is not bit-deterministic for floats (1/500 fuzz drift) — but on the
fit side only (ORC-72).
*Verified-by:* measured 2026-08-25 as described; `packages/confit/docs/properties.md:118-121`
(P11) and `packages/confit/docs/kpis.md:31-34` (C1) for the existing `threads = 1`
decision; `packages/confit/tests/conftest.py:62-71` sets no `threads`.
*Status:* stated, not ruled. See ASK-13.

> ### ASK-13 — does `threads` join the oracle constant, and what disposition covers
> order *inside* a value?
>
> Two halves, both measured (ORC-75), neither ruled.
>
> **(a) The constant.** ORC-02 says "all other settings default", and DuckDB's `threads`
> default is core-count-derived. Either the constant names `threads` explicitly — the
> obvious value is `1`, which is what the fit side already runs (ORC-72) and is one line
> in the fixture that already applies the pragma — or the document says that
> hardware-derived defaults are part of the oracle identity and that thread-sensitive
> constructs are therefore `IMPL-DEFINED` with the machine as discriminator.
>
> ```python
> # packages/confit/tests/conftest.py, inside _duckdb_is_the_oracle, if (a) is taken
> con.execute("PRAGMA disable_optimizer")
> con.execute("SET threads = 1")
> ```
>
> Not applied here — docs-only. Proposed ticket T-16.
>
> **(b) The disposition.** ORC-14's table has four keys that can all fire on one case and
> no tiebreaker, and no row at all for nondeterminism *inside* a value. `string_agg`
> element order fits "which rows exist" (refuse), "row order" (compare per mode — the
> mode assigned, `constant-ordered`, cannot see it, because the multiset differs in the
> *values*), "values from outside the query" (exclude by source name — but a frozen
> build-time artifact has no corpus source file to name), and "the oracle disagrees with
> itself" (same build, same pragma, different machine). Ruling (b) means either an
> ordering rule for the table's keys, or a fifth row for intra-value order, or both.
>
> *Binds:* ORC-02, ORC-14, ORC-22, ORC-75, and every static-only query with an
> order-sensitive aggregate.

> ### ASK-2 — build-vs-build repeatability: decide it or park it formally
>
> This is the last live hole in the frozen-artifact story. Three options:
>
> - **adopt sort-at-freeze** — the artifact sorts its frozen rows, so two builds
>   agree by construction. Cost: a total order must exist and be cheap.
> - **declare it out of contract** — an unordered constant result is unspecified
>   between builds too, and downstream golden tests must not depend on it.
> - **park it in a `tentative` bucket with a review trigger** (see ASK-9) — measured,
>   not ruled, with a named condition that reopens it.
>
> Doing nothing keeps a measured-by-luck property carrying downstream weight.
>
> *Binds:* ORC-22, and any golden-file test over a static-only result.

---

## 4. Verdicts: agreement, abstention, refusal

### 4.1 The taxonomy

**ORC-23.** One case in, one verdict out. Every outcome — refusal, trap, disagreement,
and the oracle's own failure — comes back *as* a verdict rather than as an exception, so
nothing is classified by a human reading a stack trace. The oracle module emits ten
kinds; a **campaign** emits twelve, because the runner synthesizes two more for a worker
that never answered.

| kind | meaning | emitted by |
|---|---|---|
| `AGREE` | ours == off == on | oracle |
| `AGREE_TRAP` | both sides error at run time | oracle |
| `DIVERGE_VALUE` | wrong value, wrong schema, a self-leg failure, **or a cranelift-vs-interpreter split** (klass `backend-values` / `backend-trap-split`, no DuckDB involved) | oracle |
| `DIVERGE_BUILD` | confit builds what DuckDB refuses, **or the two backends disagree about whether the build succeeds** (klass `backend-split`, no DuckDB involved) | oracle |
| `DIVERGE_TRAP` | one side traps where the other serves rows | oracle |
| `DIVERGE_OPT` | we match the optimizer-off baseline; an optimizer pass changes what the user sees | oracle |
| `OPT_EMULATED` | we match optimizer-ON against a baseline that disagrees: a plan-rewrite pass we are reproducing, which is a bug | oracle |
| `BUILD_EXC` | a build raised something other than the contract's `ValueError` | oracle |
| `REFUSED` | confit refused at build | oracle |
| `SKIP` | the oracle harness itself raised | oracle |
| `TIMEOUT` | the worker did not answer inside the per-case budget; the detail is an 800-byte stderr tail | **runner** |
| `PANIC` | the worker died without answering; same detail shape | **runner** |

`TIMEOUT` and `PANIC` are in `INTERESTING` and reach `findings.jsonl` by the same path
as every other verdict, so any statement about "what a campaign reports" has to include
them — ORC-26's abstention story and ORC-68's blind-spot table both do now.
*Verified-by:* `packages/confit/fuzz/oracle.py:65-80` (`KINDS`, ten), `:538-745`
(`run_case`), `:577-583` and `:625-639` (the backend klasses), `:884-891`
(`run_case_json`); `packages/confit/fuzz/runner.py:101` (the synthesized kind), `:24-36`
(both in `INTERESTING`), `:183-189` (written to the findings file).

**ORC-24.** Every case is run against DuckDB twice on **one** connection, off then on.
Sharing the connection is not just a saving: `statistics_propagation` reads per-column
statistics, so two separate connections could differ for reasons that have nothing to
do with the optimizer. The pair therefore brackets the answer and a finding classifies
itself.
*Verified-by:* `packages/confit/fuzz/oracle.py:402-419` (`_duck_run` and its rationale).

**ORC-25.** `OPT_EMULATED` is a bug, not an accepted class, and it is excluded from
coverage. Counting it as agreement would hide it twice: once as a finding and once as
coverage.
*Verified-by:* `packages/confit/fuzz/runner.py:30-31` (in `INTERESTING`), `:38-41`
(`COVERED = ("AGREE",)`, with the reasoning in-file);
`packages/confit/fuzz/oracle.py:75-77`.

### 4.2 Abstention is a verdict

**ORC-26.** Abstention is reported, never silently downgraded to a pass. `SKIP` — the
oracle harness's own failure — is a finding and reaches `findings.jsonl`; it is not
allowed to look like agreement, because an error bucket that quietly grows is how a
suite hides real bugs behind a green bar. `TIMEOUT` and `PANIC` (ORC-23) are the same
species and are treated the same way.
*Verified-by:* `packages/confit/fuzz/oracle.py:884-891` (an exception escaping
`run_case` becomes `SKIP`, blaming the oracle rather than the engine);
`packages/confit/fuzz/runner.py:35` (`SKIP` is in `INTERESTING`), `:33-34`
(`PANIC`/`TIMEOUT` likewise).

**ORC-78.** A timeout is attributed before it is counted: **an oracle-side timeout and
an engine-side timeout mean opposite things.** Measured 2026-08-14 on seed 4395 —
`lpad(c1, 2147483647, 'NULL') LIKE '...'` — we refuse in 0.00s at bind ("lpad count
2147483647 exceeds the 1 GiB string-builder budget") while DuckDB takes 9.0s actually
building the 2 GiB pad and answering `false`. Under eight workers that exceeds the
per-case budget, which *is* the finding: no engine hang, no liveness bug. Three further
seeds are the same story. The recorded follow-ups are that the runner must record the
SQL *before* executing (a timeout currently loses it, so the case has to be recovered
from the generator by seed) and that oracle-side timeouts must classify apart from
engine-side ones.
*Verified-by:* `packages/confit/docs/2026-08-13-fuzz-triage.md:124-149`.

**ORC-27.** A comparison the checker cannot evaluate falls back to the weaker check
**with a logged tag**, never silently. The one instance today: an `ORDER BY` over an
expression that is not an output column cannot have its key evaluated, so the multiset
check stands and the case carries an `order-by-unevaluated` tag.
*Verified-by:* `packages/confit/fuzz/oracle.py:685-690`;
`backlog/tasks/task-129 ...md` AC #4.

**ORC-28.** `AGREE` is the only kind counted as coverage. A construct-coverage
histogram runs over agreeing cases only, so a grammar hole is visible rather than
absorbed by refusals.
*Verified-by:* `packages/confit/fuzz/runner.py:38-41` (`COVERED`), `:159-164` (the
`AGREE`-only histogram). Not `:145-157` — that range is `report()`'s docstring, the
verdict counter and the *refusal-class* histogram, which is a different block.

### 4.3 Refusals

**ORC-29.** A build-time refusal is the engine's second legal outcome and is always a
named `ValueError` at `DuckDBInferFn(...)` construction. Three documented message
prefixes classify it: `unsupported:` (real SQL, deliberately not served), `parse error:`
(the dialect surface ends here), `bind error:` (the query is wrong against your schema).
Refusal is cheap, named and testable by construction.
*Verified-by:* `packages/confit/docs/known-limitations.md:274-285`; P7 and P18 in
`packages/confit/docs/properties.md:63, :231-235`.
*Correction:* the prefix set is not exhaustive in code. The corpus gate's clean set is
`_CLEAN = ("unsupported:", "parse error:", "duplicate map key", "NULL in value column")`
(`test_corpus_replay.py:36`) — two real engine messages that carry none of the three
prefixes (`interp.rs` "@{i}: duplicate map key"; `duckdb/mod.rs` "static table '...' has
a NULL in value column '...'"), and `bind error:` is absent from that set entirely.
Either the two messages gain a prefix or the documented set gains them. See proposed
ticket T-17.

**ORC-79.** Refusal *grounds* are a decided three-way taxonomy, orthogonal to the
message prefix: **specialization-inherent** (the engine model cannot express it),
**scope-by-product-decision** (it could be served and we chose not to), and **resource**
(it would cost more than a serving engine may spend per row). The split is what makes
"we refuse" auditable — the first is permanent, the second is reversible by a decision,
the third is a judgement with a number attached.
*Verified-by:* `backlog/milestones/m-8 - duckdbs-type-lattice.md:30-36`; mirrored by
`known-limitations.md` section 1 ("The specialization bargain (inherent to the engine
model)", `:65`) vs section 2 ("Out of scope for row-serving (by decision, not
difficulty)", `:80`), with the resource class at `known-limitations.md:205` and
`known_divergences/test_arrow_boundary.py:34-36` (ledger rows D14 and D15).

**ORC-30.** **[FACT]** Current behavior of the campaign's refusal path: both
DuckDB readings are computed and then **discarded unconditionally** when confit
refused. `REFUSED` carries only a class derived from the first six words of the
message, and it is not in `INTERESTING`, so it never reaches `findings.jsonl` — only a
"top refusal classes" histogram at the end of a run.

```python
# packages/confit/fuzz/oracle.py:585-589
    duck_off, duck_on = _duck_run(sql, case, udf_objs)

    if fn_cl is None:
        klass = _refusal_class(cl_err)
        return Verdict("REFUSED", klass, cl_err, tags)
```

*Verified-by:* `packages/confit/fuzz/oracle.py:585-589` and `:876-877`;
`packages/confit/fuzz/runner.py:24-36` (the `INTERESTING` tuple, which does not contain
`REFUSED`), `:152-157` (the histogram).

**ORC-31.** **[PROPOSED]** Not in force. The general rule this exposes, and the one this
document would like written down: **an accepted cost must be countable, and the counting
mechanism is named in the decision that accepts it.** Without that, "deliberate
strictness" and "unnoticed over-refusal" are the same observation. The rule is not in
force anywhere today; the live instance is ASK-3, and adopting the rule itself is part of
ASK-15.
*Verified-by:* Unverified — no decision outside this document states it. The nearest
existing practice is the m-8 phase rule that each phase's markers must be deleted in the
phase's own PR and certified by a campaign (ORC-80), which is a counting mechanism for a
different kind of cost.

> ### ASK-3 — the accepted severity-4 cost is currently uncountable. Which way?
>
> You accepted the bind-time constant refusals twice (2026-08-24, re-affirmed
> 2026-08-25 on corrected facts). The RFC justifies the accepted cost three times by
> asserting the campaign will measure it:
>
> - `rfcs/2026-08-19-keep-the-bind-time-refusals.md:95-96` — "Under the fuzzer's
>   refusal-absorb rule (a refusal is acceptable **where the oracle traps**)"
> - `:100-102` — "A campaign that generates those shapes will (correctly) log
>   refuse-where-oracle-serves findings"
> - `:146-149` — "a campaign that generates them will log severity-4 findings, which
>   are attributed to this RFC"
>
> Verified: it cannot. The absorb rule in code is **total, not conditional** (ORC-30) —
> a refusal is absorbed even when both oracle readings serve rows — and `REFUSED` never
> reaches the findings file. The decision itself is not in question; only whether its
> price is observable.
>
> **(a) Split the verdict** so the cost becomes measurable:
>
> ```python
> # packages/confit/fuzz/oracle.py, replacing the refusal return at :587-589
> if fn_cl is None:
>     klass = _refusal_class(cl_err)
>     oracle_serves = duck_off[0] is not None
>     kind = "REFUSED_ORACLE_SERVES" if oracle_serves else "REFUSED_ORACLE_TRAPS"
>     return Verdict(kind, klass, cl_err, tags)
> ```
>
> with `REFUSED_ORACLE_SERVES` added to `runner.INTERESTING`. Three lines. It may
> reveal the class is larger than assumed, which is the point.
>
> **(b) Amend the RFC** to say the cost is accepted unmeasured. Honest and free.
>
> Not applied here — docs-only. Proposed ticket T-5.
>
> *Binds:* ORC-30, ledger row D9, and the severity ladder's rung 4 in section 8.

> ### ASK-4 — `OPT_EMULATED` gets `AGREE` treatment at one line
>
> Behaviorally the code is on the "bug" side everywhere (ORC-25) except one branch,
> where `OPT_EMULATED` is grouped with `AGREE` for the purpose of running the boundary
> legs:
>
> ```python
> # packages/confit/fuzz/oracle.py:740
> if v.kind not in ("AGREE", "OPT_EMULATED"):
>     return v
> ```
>
> Deliberate — run the extra legs anyway, since the values matched *something* — or a
> survivor of the pre-2026-08-17 doctrine, when `OPT_EMULATED` meant expected? This one
> changes what runs; the stale comments elsewhere are editorial, and the one that matters
> is named in section 11 (`runner.py:166-168` sits directly above the block that
> contradicts it).
>
> One recorded fact the ruling should have: the **only** observed `OPT_EMULATED` instance
> outside regex was a **mislabel, not an emulation**. Seed 1784's `FETCH FIRST 1 ROWS ONLY`
> without an `ORDER BY` had the two DuckDB reads lawfully pick different groups, and the
> triage records it as "mislabelled by this category because the readings disagree with
> each other" — filed as fuzzer QoL for TASK-94, not as a bug in the engine
> (`packages/confit/docs/2026-08-17-fuzz-triage.md:89-94`). ORC-25's "a bug, not an
> accepted class" is the right rule and has never yet had a true positive.
>
> *Binds:* ORC-25.

> ### ASK-5 — do abstention and refusal reason codes become user-visible?
>
> If a reason-code vocabulary (`unspecified-order`, `tie-break`, `fp-association`,
> `session-dependent`, `oracle-errored`) is adopted for campaign reporting and the
> ledger, it must not leak into build-error text without your explicit approval —
> refusal messages are product surface and fall under the API-change rule. Cleanest
> split, and my recommendation: internal codes for the ledger and reports, existing
> prose refusal messages unchanged.
>
> *Binds:* ORC-29.

---

## 5. The comparison contract

This is what "same as the oracle" means, mechanically. It has been the strictest
contract in the project from the start and has never been written down in one place.

**ORC-32.** Floats compare by **bit pattern**, and the pin records the bits, not a
rendering. No rounding, no `%.3f`. **Three exceptions are in force**, all named and all
narrow; there are no others, and a fourth would be a decision:

| exception | bound | why |
|---|---|---|
| `cbrt` (ORC-76) | `<= 1` ulp | the oracle's own wheels disagree by one ulp across platforms; repr-exact is unpinnable |
| the fuzzer's sklearn second-ground-truth leg | `1e-9` absolute | sklearn is a second reference, not the oracle; the leg exists where the oracle abstains (ORC-69) |
| matvec-tier parity (DRAFT-23, when native families land) | a declared per-family ulp bound | and the governance rule with it: "Loosening a control is a design decision, never a fix ... the new bound named — through review, not through a failing test" |

Two mechanical limits are worth naming beside them, because they are *not* exceptions —
they are places where the contract says "bits" and the gate compares something else.
Both differential gates canonicalize rows through `repr`, which makes every NaN
self-equal and its **sign and payload invisible**; only the explicit bit pins see those.
And the campaign's schema normalization can move a value before the comparison — ORC-38
and ASK-12.
*Verified-by:* `packages/confit/tests/test_duckdb_wave3_mathtail.py:205-235`
(explicit bit pinning, since `repr` collapses every NaN to `nan`); float bit patterns are
a recorded field across the pins corpus;
`packages/confit/tests/test_duckdb_interpreter.py:918-952` (the cbrt ulp bound);
`packages/confit/fuzz/oracle.py:851` (the sklearn `1e-9` leg), `:422-429` (`_key`,
repr-based) and `packages/confit/tests/test_corpus_replay.py:70-72` (`_norm_row`, whose
own comment says repr "makes NaN self-equal");
`packages/confit/docs/kpis.md:17-20` (the loosening rule) and `:85-89` (DRAFT-23's
declared per-family ulp bound).

**ORC-33.** `-0.0` is distinguished from `+0.0`, and this is **fixed, not tolerated**.
Unary minus was lowered as `0 - x`; IEEE `0.0 - 0.0` is `+0.0`, so the sign vanished
everywhere it could arise — 113 of 963 findings in the 2026-08-11 campaign. The fix
subtracts from `-0.0` for FLOAT operands (exact IEEE negation for every double) and
keeps `0 - x` with its `i64::MIN` trap on the integer path, matching DuckDB. It is now
a passing regression pin over both backends.
*Verified-by:* `packages/confit/tests/known_divergences/test_literal_typing.py:133-165`;
`backlog/tasks/task-80 ...md:46` (the 113/963 measurement), `:75` (the class measured
empty after).
*Scope, and it is narrower than it reads.* All five parametrizations of the pin use the
**`e0` (DOUBLE) spelling** — `-0.0e0`, a DOUBLE column, `-1.5e0`. A bare `-0.0` is
`DECIMAL(2,1)` in DuckDB, and a decimal zero has no sign, so `SELECT (c * (- 0.0))`
answers `0.0` on DuckDB and `-0.0` here. That is not a regression of the fixed class —
it is the **opposite direction**: the fix now *keeps* a sign DuckDB's decimal path
discards, and it is the D7 literal-typing mechanism wearing a third face. The
2026-08-17 campaign residual seed 998 is exactly this. Also: the "class measured empty
after" measurement is the **2026-08-13** campaign, one campaign earlier than the
committed baseline.

**ORC-34.** `%`-by-zero produces a NaN whose **sign bit is platform-libm** (`7ff8...`
on Windows ucrt, `fff8...` on Linux glibc), so the pin is *engine == oracle bit
agreement per platform*, not a constant. `fmod`'s NaN, by contrast, comes from hardware
arithmetic and is `fff8...` on every x86 platform, so it is pinned as a constant.
Status: `IMPL-DEFINED`, platform is the discriminator.
*Verified-by:* `packages/confit/tests/test_duckdb_wave3_mathtail.py:205-235` (the
engine == oracle assertion is `assert bits(got["m"]) == bits(m)` at `:235`);
`packages/confit/docs/specs/pins-wave3/math_tail.json` (the wave-3 correction);
`packages/confit/docs/known-limitations.md:258-259`.

**ORC-81.** Platform is a discriminator for **every libm-backed function**, not only for
`%`-by-zero. The wave-1 trig pins say so in-file — "the oracle is platform-libm-dependent,
so cross-OS bit-identity of pins must be re-verified on the CI/serving platform" — and the
`pow10` modifier behind `floor`/`ceil`/`trunc`/`round` is the DuckDB binary's own
`std::pow`, "neither correctly-rounded nor ucrt pow", which "must be extracted from the
oracle binary" and is named in-file as a cross-platform divergence hazard for CI. Those
pins record a single value today while being per-platform in substance; if ORC-35 or
ORC-63 lands they are the next `IMPL-DEFINED` candidates.
*Verified-by:* `packages/confit/docs/specs/pins-wave1/pins_trig-sin-x-cos-x-tan-x-p.json`
(the platform-libm note); `pins-wave1/pins_floor-ceil-trunc-round.json` (the `pow10`
extraction rule).

**ORC-82.** Data tables that stand in for oracle behavior are **extracted from the
oracle**, never from a host library. `strip_accents`' per-codepoint map, the case map and
the `pow10` table are all generated by querying DuckDB; DuckDB's own Unicode tables lag
Unicode 16 by 57 codepoints, so a host `unicodedata` would be a different oracle wearing
the same name. This is ORC-10 (reproduce the quirk) at the level of data rather than
behavior, and it is the strongest form of pins-first in the repo.
*Verified-by:* `packages/confit/docs/specs/2026-07-26-wave3-builtin-pins.md:129-142`;
`scripts/gen_strip_accents.py`, `scripts/gen_casemap.py`, `scripts/gen_pow10.py`.

**ORC-35.** **[PROPOSED]** Not in force. The multi-answer rule: a set of accepted answers
is legitimate **only when every member is acceptable in every context**, and every
set-valued pin names the predicate that selects a member (platform, profile, oracle
version). Shortest-diff or best-match selection is never legitimate: it makes the suite
pick the answer that happens to be closest, which is indistinguishable from picking the
answer that hides the bug. ORC-34 is the nearest existing practice — platform is a real
discriminator, evaluable before the comparison runs — though ORC-34 is strictly a *pinned
agreement relation* (engine == oracle, per platform) rather than an enumerated set of
accepted answers, so no pin in force today is set-valued in the sense this rule
describes. ORC-76's bounded tolerance is a second shape the rule as written does not
cover.
*Verified-by:* Unverified — no decision outside this document states the rule. The
survey it comes from is external. Part of ASK-15.

**ORC-36.** Rows compare as a **multiset** except where a compare mode says otherwise
(ORC-18). Row-order claims on the serving path are checked by DuckDB-free self-legs
(ORC-16), never by matching DuckDB's sequence.
*Verified-by:* `packages/confit/fuzz/oracle.py:422-429` (`_key` multiset form, `_seq`
sequence form), `:432-449`.

**ORC-37.** Duplicate output names are normalized through the **same rule on both
sides** before comparison, which is what makes the rename a contract rather than a
divergence: duplicates rename left-to-right to `<name>_N`, smallest free N,
case-insensitive, generated candidates included. DuckDB itself applies exactly this at
every subquery / CTE / CTAS boundary and in `.df()`; only its top-level arrow export
keeps duplicates, so the oracle leg is renamed before comparing.
*Verified-by:* `packages/confit/fuzz/oracle.py:467-488` (`_dedup_names`), `:646-651`
(applied to the DuckDB side);
`packages/confit/docs/specs/pins-wave5/dup-names-client-contract.json`;
`packages/confit/tests/test_known_limitations.py:255` (the twin).

**ORC-38.** Output **schemas** are compared, not just values. A name mismatch or a
type mismatch is a `DIVERGE_VALUE` in its own right; the only exception is the
enumerated unshipped-feature width class — today exactly one arm, decimal-vs-float64 —
which is *tagged* and cast before the value comparison so the tag stays visible in the
report rather than becoming an accepted equality. **The tag suppresses the schema delta
only. The value comparison still runs on the cast result**, and a tagged case that then
differs in value is still reported `DIVERGE_VALUE`; three such findings sit in the
committed baseline.
*Verified-by:* `packages/confit/fuzz/oracle.py:491-529` (`_schema_delta`, `_type_delta`
— the decimal arm at `:512-513` is the only unshipped-feature arm), `:700-710` (tag,
then `cast_to = sch_cl`), `:531-535` (`_norm` applies `table.cast(to)`), `:706-712`
(the value comparison after the cast); `packages/confit/fuzz/runner.py` contains no tag
filter of any kind.
*Measured, and it is the reason for ASK-12:* the cast is not value-preserving. Measured
2026-08-25 under the oracle, `SELECT -14.665` comes back `decimal128(5,3)`; the harness's
`table.cast(float64)` yields `-14.665000000000001`, while DuckDB's own `::DOUBLE` and
Python's `float()` both yield `-14.665` — which is also what confit returns. So for that
shape the normalization step, not either engine, is what produces the reported delta.

> ### ASK-12 — is the comparison harness's own normalization part of the oracle's answer?
>
> ORC-38 authorizes casting DuckDB's leg to our schema before comparing, and does not
> require the cast to preserve values. Measured, it does not: for a decimal literal the
> harness's `pyarrow` `decimal128 -> float64` cast lands one ulp away from the double that
> DuckDB's own `::DOUBLE` produces (measured above; the same shape reproduces for
> `94.579` and `-42.602`). Three of the four `DIVERGE_VALUE` residuals in the committed
> baseline — seeds 869, 1554, 3269, all `decimals`-tagged — are that shape, and they are
> currently reported as rung-2 contract violations.
>
> The **schema** divergence in those cases is real and is ledger row D7. What is at issue
> is only the value delta reported on top of it. Three ways:
>
> - **normalize through the oracle** — ask DuckDB for the cast (`::DOUBLE` in the emitted
>   SQL, or a second reading) rather than casting its output in the harness, so the
>   comparison never invents a value;
> - **compare on the decimal side** — cast *our* f64 up instead of casting DuckDB's
>   decimal down, and accept that this changes what "equal" means for the class;
> - **declare the normalization part of the harness contract** — the cast is what it is,
>   and a tagged case's value comparison is `UNSPECIFIED` until the feature lands.
>
> Not applied here — docs-only, and the choice is yours because it changes what the
> campaign reports. Proposed ticket T-18.
>
> *Binds:* ORC-32, ORC-38, ORC-26, ledger rows D7 and D12.

**ORC-39.** Error **texts** are not compared. Runtime traps reproduce DuckDB's message
bodies verbatim; some bind-time rejections use our own wording with the same error
class. The corpus compares successful results only, so texts never affect parity. Error
text is therefore *not oracle-decided output* — which is also a named blind spot
(ORC-57). Upstream does the same thing: DuckDB's own test infrastructure matches error
text by substring containment.
*Verified-by:* `packages/confit/docs/known-limitations.md:219-224`;
`packages/confit/tests/test_corpus_replay.py:173-176` (only successful rows compared).

**ORC-40.** Backend agreement is settled **before either reading is compared against**:
cranelift vs interpreter is a question about us, not about the oracle, so it is checked
once and short-circuits. A split there carries **its own `klass`** — `backend-split`,
`backend-values`, `backend-trap-split` — so it is never confused with a DuckDB
disagreement when reading a finding. Its *kind* is still a divergence kind
(`DIVERGE_BUILD` or `DIVERGE_VALUE`), which is what `findings.jsonl` and the ledger
census key on, so a backend split does count into those totals; ORC-23's table now says
so on both rows.
*Verified-by:* `packages/confit/fuzz/oracle.py:625-639` (the check and its in-code
comment), `:577-583` (the build-side split); P19 in
`packages/confit/docs/properties.md:240-245`.
*Precision:* the check is settled before either reading is *compared*, not before either
is *executed* — `_duck_run` runs at `:585`, upstream of the backend checks at `:625-639`.
The in-code comment says "before either reading" and means the comparison.

**ORC-83.** The interpreter is the **internal oracle backend** for the engine's own
two-backend differential: correctness and coverage over speed, never optimized, and
cranelift is checked against it byte-for-byte over a 500-seed random-IR sweep. It is why
ORC-40 can settle backend agreement without DuckDB at all.
*Verified-by:* P19 in `packages/confit/docs/properties.md:240-245`;
`packages/confit/docs/kpis.md:62-64` (the 500-seed random-IR sub-invariant, in
`packages/confit/src/specializer/exec/tests.rs`);
`backlog/tasks/task-42 - Specializer-M-interp-closure-compiled-IR-interpreter-the-oracle-backend.md`.

**ORC-41.** Mechanisms other cross-engine suites use that **this project has never
adopted**, and the reason each one is a bad fit here. This is a survey with an argument,
not a list of past decisions: only the first row corresponds to a rule in force (ORC-32),
and none of the other four was ever proposed here, so none was ever rejected here.
Turning the four into standing rejections would itself be a decision — that is ORC-84.

| mechanism | where it comes from | why it is a bad fit here |
|---|---|---|
| float rendering at `%.3f` | sqllogictest's cross-engine rendering contract | directly destroys bit-for-bit, which **is** in force (ORC-32); it trades float fidelity for cross-engine agreement, and we have exactly one engine to agree with |
| MD5 hashing result streams above a threshold | sqllogictest | hashes make a failure undebuggable; pins-as-data (reprs, float bits, verbatim error heads, the exact SQL) is strictly better evidence, and DuckDB's own docs advise using the hash form sparingly |
| shortest-diff variant matching | Postgres's `resultmap` driver, which admits it "cannot tell which variant is actually correct" | picks the closest answer, which is the answer most likely to hide the defect; ORC-35 is the proposed replacement |
| cross-engine agreement or majority vote as the oracle | sqllogictest, Csmith | creates a second authority; the contract delegates to exactly one engine on purpose (ORC-01). Note the dialect gates (ORC-73) *do* run a second engine — as a target for a printed query, never as an authority over DuckDB |
| a growing expected-errors allowlist | SQLancer's `ExpectedErrors` | every entry added to silence a false positive is a place a real bug can hide; structurally the same shape as ORC-30. The nearest thing we have is `_CLEAN` (ORC-29), which is four entries and has not grown |

*Verified-by:* ORC-32 for the first row. The other four are `Unverified` as decisions —
no spec, ticket or review in this repo records adopting or rejecting them (searched
2026-08-25). They are here as an argument, and the argument is what ORC-84 asks about.

**ORC-84.** **[PROPOSED]** Not in force. The four unadopted mechanisms in ORC-41 become
**standing rejections**, so that proposing one later is a contradiction of the spec
rather than a fresh idea. The cost of adopting this is real: a standing rejection of
"tolerance" has to be written so that it does not contradict the three tolerances
already in force (ORC-32's exception table) or the dialect gate's designed epsilon tier
(ORC-73).
*Verified-by:* Unverified. Part of ASK-15, and the substance is ASK-6.

> ### ASK-6 — is bit-for-bit float equality the contract, and what governs its exceptions?
>
> Every cross-engine suite surveyed quietly relaxes floats. The question is *not* whether
> this project has exceptions — it has three, all named in ORC-32's table — but whether
> the rule is "bit pattern, with a closed list of declared bounds" and what it takes to
> add a fourth entry to that list.
>
> **What is already decided, so that the ruling is about the open part.** An earlier
> version of this block asked about D7/D8 as if their classification were open. Two of
> its premises were wrong and are corrected here:
>
> - "The fuzzer's `decimals` tag suppresses it" — **it does not.** The tag suppresses the
>   *schema* delta only; the value comparison runs on the cast result and still reports
>   `DIVERGE_VALUE` (ORC-38). Three such findings are in the committed baseline.
> - "D7/D8 have no home" — the **feature-in-flight rule (ORC-80), decided 2026-08-11**,
>   already governs them: anything with an m-8 phase is a feature in flight, not a known
>   divergence, its markers are scaffolding, and each phase's definition of done includes
>   deleting them in all three homes in the feature's own PR. Two of the three homes are
>   enforced. The third — the fuzzer's suppression tag — is explicitly recorded as
>   unenforced, which is what `oracle.py:126-131` means by "nothing rings".
>
> **So the live question narrows to three parts:**
>
> **(a) The rule.** Is it "bit pattern, no exceptions" — in which case ORC-76's cbrt
> tolerance, the sklearn leg's `1e-9` and DRAFT-23's declared bound are three
> contradictions that need re-ruling — or is it "bit pattern, with a closed list of
> declared bounds, each naming its discriminator", in which case ORC-32's table *is* the
> list and adding to it is a decision like loosening any other control?
>
> **(b) Future float accumulation.** If parallel float accumulation ever lands, is its
> `UNSPECIFIED` region **refused** (the strict reading) or given a declared bound the way
> the dialect gate's epsilon tier already is (ORC-73)?
>
> **(c) The unenforced third home.** Does the `decimals` tag get an enforcement now — the
> shape TASK-95 would give it, or a strict-xfail twin standing behind the
> known-limitations row — or does it stay unenforced until the lattice phase, on the
> record?
>
> *Verified-by (the facts, not the ruling):*
> `packages/confit/docs/known-limitations.md:166-174`;
> `packages/confit/fuzz/oracle.py:113-131` (the tag's own scope statement and its
> "no strict-xfail twin" note), `:512-513` (the one arm), `:700-712` (the cast, then the
> value comparison); `packages/confit/docs/specs/2026-08-11-duckdb-type-lattice-design.md:110-131`
> (the feature-in-flight rule and the three homes).
>
> *Binds:* ORC-32, ORC-76, ORC-80, ORC-84, ledger rows D7 and D8, and every future
> float-accumulation feature.

---

## 6. Pins

### 6.1 What a pin is

**ORC-42.** A pin is **one measured oracle fact**: a behavioral claim backed by the
exact SQL that was run and the exact result that came back — query text, input reprs,
result reprs, float bit patterns, verbatim error heads. A pin is spec-as-data. It is
not a test of our code; it is a recording of the oracle's answer, which our code is
then written to.
*Verified-by:* `packages/confit/docs/reports/pins-first-methodology.md:20-22`;
the corpus at `packages/confit/docs/specs/pins-*/` (53 files measured 2026-08-25).

**ORC-43.** Pins-first: **no semantics are implemented from memory, documentation, or
intuition — only from executed queries against the oracle, recorded verbatim.**
Implementation starts only after the pins exist. A summary sentence with no query
behind it is treated as a guess.
*Verified-by:* `packages/confit/docs/reports/pins-first-methodology.md:20-22, :28`.

**ORC-44.** The rule was bought, not designed. During wave 3 a fleet summary claimed
`%`-by-zero returns NULL, generalizing from integer probes; the DOUBLE case was never
run and returns NaN. The correction is appended to the pin file with the honest note
that "raw probes never covered this cell ... the summary over-generalized", and every
wave dispatched since carries the rule explicitly.
*Verified-by:* `packages/confit/docs/specs/pins-wave3/math_tail.json` (the
`corrections` key); `packages/confit/docs/reports/pins-first-methodology.md:24-28`.

**ORC-45.** Phase separation is required for any claim about *when* DuckDB does
something. `con.execute` conflates prepare and execute, so a bind-time claim needs a
PREPARE/EXECUTE split, a zero-row leg, and the pinned source. This is the same genus as
the wave-3 incident with a larger blast radius: it killed the stated premise of an
already-accepted RFC.
*Verified-by:* `packages/confit/docs/rfcs/2026-08-19-keep-the-bind-time-refusals.md:29-58`
(the corrected facts, and the phase-confusion admission at `:31-35`).
*Note:* this rule lives in memory and in one RFC's body. The methodology report owns
"how we measure DuckDB" and does not carry it. Proposed ticket T-6.

### 6.2 Provenance

**ORC-46.** A pin's provenance is what makes a disagreement re-verifiable: without the
oracle version that produced a recorded answer, a future disagreement cannot be
re-run, only argued about. Measured state of the corpus today: of 53 pin files, **41
carry a `duckdb_version` field, 10 mention a capture date anywhere, and 3 mention a
harness or commit**; the version field itself is free text with at least four spellings
in use (`1.5.5`, `v1.5.5`, `v1.5.5 (python pkg 1.5.5)`, and a sentence).
*Verified-by:* measured 2026-08-25 over `packages/confit/docs/specs/pins-*/*.json`.
Best existing examples: `pins-dialect/joins.json` `_meta` (date, engine, task, spec,
how) and `pins-waveB/fuzzer-task54.json` `meta` (task, measured, method, contract).
*Proposed:* a uniform header — oracle version, settings profile, capture date, capture
harness commit — on every pin file. Not applied here. Proposed ticket T-7.

**ORC-47.** Generation scripts already stamp the oracle version into their outputs,
and two already say to regenerate after a duckdb bump — the right instinct, without a
uniform shape and without covering the pins corpus.
*Verified-by:* `scripts/pin_ast_shapes.py:29, :36`; `scripts/gen_casemap.py:152, :159`
("regenerate after a duckdb bump"); `scripts/gen_strip_accents.py:135` (same).

**ORC-48.** **[PROPOSED]** Not in force. Every pin carries a **decision back-reference** —
the `ORC-NN` id of the claim it evidences. This is the mechanical instrument for this
project's own definition of completeness: with back-references, "decisions with zero
pins" is the uncovered set and is computable in one query; without them, decision
coverage is a promise nobody can audit. Ids in this document are hand-assigned and
stable precisely so a pin can point at one.
*Verified-by:* Unverified — no pin carries such a field today (measured 2026-08-25).
Proposed ticket T-8.

**ORC-49.** **[PROPOSED]** Not in force. The pin format gains an inline token for an
under-determined field, so "this field is not part of the contract" or "this field is
contract per-platform" is written *in the pin* rather than in prose beside it. ORC-34 is the
existing instance and is currently a special case explained in a comment. Without a
token, every re-measurement pass must re-derive which fields were deliberate.
*Verified-by:* Unverified — no such token exists. Proposed ticket T-9.

---

## 7. The divergence ledger

### 7.1 The split is by intent, and it stays

**ORC-50.** The record is split by INTENT, not by severity:
`packages/confit/tests/known_divergences/` holds behavior we decided to **KEEP** — all
passing, each entry owing a measured REASON — and
`packages/confit/tests/test_open_divergences.py` holds behavior we decided to
**CHANGE** — one `xfail(strict=True)` pin each, ticket named, deleted rather than
edited when it closes. The ground for the split is a measured census, not a
preference: it found readers both implementing something we chose not to have, and
walking past a real bug because the paragraph above it sounded like a rationale.
*Verified-by:* `packages/confit/tests/known_divergences/README.md:19, :35-46`;
`packages/confit/tests/test_open_divergences.py:9-25`.

**ORC-51.** `strict=True` is the load-bearing part of the CHANGE ledger: a pin that
silently starts passing is worse than no pin, because it certifies work nobody did.
The marker self-expires — closing a divergence makes its pin fail loudly.
*Verified-by:* `packages/confit/tests/test_open_divergences.py:27-28`.

**ORC-52.** The CHANGE ledger is **empty as of 2026-08-25**, deliberately, with named
successor tickets. An empty CHANGE ledger with a named successor is the mechanism
working, not an absence of divergences. It has emptied and refilled inside a single day
before; that rhythm is intended.
*Verified-by:* `packages/confit/tests/test_open_divergences.py:30-35`;
`backlog/tasks/task-134 ...md` (To Do).

**ORC-53.** A KEEP entry owes a REASON, not just a description, and where the reason is
a claim about DuckDB it must be measured and must stay true. One had already gone false
and was propagating into a user-facing message when the census caught it.
*Verified-by:* `packages/confit/tests/known_divergences/README.md:44-46`.
*Live instance, uncaught:* the string-budget entry's ground was restated 2026-08-16
because the old one was false — measured, DuckDB is entirely deterministic there ("no
coin flip, no spelling-dependence"), so the honest ground is ours and is a judgement.
`known-limitations.md:205` still carries the **corrected-false** claim that "DuckDB's
own behaviour is spelling-dependent". See proposed ticket T-19 and ledger row D14.

**ORC-80.** "Known divergence" is reserved for **decided-and-unscheduled** differences —
the rows in `known-limitations.md` and in section 7.3 below. Anything carrying an m-8
phase or a ticket is a **feature in flight**, its tests live with the feature, and its
markers (xfail pins, fuzzer suppression tags) are **scaffolding**. Each phase's
definition of done includes deleting that phase's markers in all three homes, in the
same PR as the feature: (1) the xfail-strict pin flips to a real parity test — enforced,
because strict xfail turns XPASS-loud the moment the feature lands; (2) the
known-limitations row goes — enforced by its executable twin; (3) the fuzzer's
suppression tag is removed — **unenforced**, stated in the design precisely because a tag
that outlives its phase would silently swallow regressions in the code the phase just
changed. The certification campaign *after* the tag removal is what proves the class is
gone rather than hidden. Decided with the owner 2026-08-11.
*Verified-by:* `packages/confit/docs/specs/2026-08-11-duckdb-type-lattice-design.md:110-131`.
This rule is what classifies D7, D8 and D11; the unenforced third home is ASK-6(c).

### 7.2 Doc-twin totality is partial

**ORC-54.** **Five sites across four documents** assert that every limitation has an
executable twin. The code denies it in-file, TASK-95 is open with both acceptance
criteria unchecked, and the measured shape is: `test_known_limitations.py` holds 14 test
functions (two parameterized) against roughly three dozen enumerated limitations, with
several sections' twins living elsewhere rather than in the named twin file. Where they
actually live, per the ledger's own evidence column: `test_arrow_schema_api.py` for the
row-limit rule; `test_corpus_replay.py` for D2 and D3; `test_duckdb_wave3_mathtail.py`
for D5; **no test at all** for D6; and of D2-D6 only D4 is in `known_divergences/`
(D8 and D9, outside that range, are).
*Verified-by:* the claims at `packages/confit/docs/known-limitations.md:5-7` and
`:294-296` (two sites, one document), `packages/confit/docs/reports/pins-first-methodology.md:66`,
`packages/confit/docs/kpis.md:58-61`, and
`packages/confit/docs/reports/confit-architecture.md:150` ("each is a named build-time
rejection with an executable twin" — the fifth site, missed by an earlier version of this
claim); the denial at
`packages/confit/fuzz/oracle.py:126-131` ("doc-twin accounting — a strict-xfail pin
standing behind every known-limitations row — does not exist yet");
`backlog/tasks/task-95 ...md` (To Do, ACs `[ ]`); counts measured 2026-08-25.
*Status:* the mechanism works for the rows it covers. The overstatement is the problem,
because it stops readers checking. See ASK-7.

> ### ASK-7 — close TASK-95, or downgrade the four claims to what is true?
>
> An overstated guarantee is worse than a stated partial one. Two ways:
>
> - **close TASK-95** — a known-limitations row without a named twin fails a unit test,
>   and missing twins get written. This is the sig.rs totality-test pattern applied to
>   prose; it was approved 2026-08-13 and never built.
> - **downgrade the claim** at all five sites to "every limitation with a twin is
>   asserted; the twins are enumerated here", and drop the totality word. Note the fifth
>   site (`confit-architecture.md:150`) was missed by the original version of this ASK,
>   so a remedy scoped to "four documents" would have left it standing.
>
> Recommendation, if you want the cheap one: downgrade now, keep TASK-95 open. The
> claim costs nothing to correct and currently buys false confidence.
>
> *Binds:* ORC-54, and the four documents named in its Verified-by.

### 7.3 The ledger

**ORC-55.** The table below is the enumeration of currently-tolerated divergences —
engine-vs-oracle, plus two rows (D4, D16) that are kept here because readers look for
them and the row says plainly what each one actually is. Each row is a decision, so each
carries a proposed status from the section 3 vocabulary plus a permanence, and each
awaits an owner ruling. **This table is the adjudication surface: `accept` writes the
proposed status into force, `reject` sends the row back as a defect owing a ticket.**
Nothing in the `proposed status` column is in force until the ruling column is filled —
which means that today no row's status is in force, so under ORC-15's rule as written
none of them is yet a comparison target. That tension is real and is part of ASK-15;
until it is resolved, read the `proposed status` column as this document's
recommendation, not as the vocabulary's application.

*Severity* is the ladder of section 8: 1 = trap where DuckDB serves, 2 = wrong value,
3 = serve where DuckDB refuses, 4 = refuse where DuckDB serves.

| id | behavior | measured evidence | sev | proposed status | OWNER RULING |
|---|---|---|---|---|---|
| **D1** | duplicate output names renamed by DuckDB's own boundary algorithm; applied to *both* sides before comparison | `pins-wave5/dup-names-client-contract.json`; `oracle.py:467-488, :646-651`; twin `test_known_limitations.py:255` | n/a (contract) | `PINNED` / permanent | |
| **D2** | error texts approximate where noted; the corpus compares successes only | `known-limitations.md:219-224`; `test_corpus_replay.py:173-176` | n/a | `UNSPECIFIED` / permanent | |
| **D3** | `ILIKE` with embedded NUL is statistics-dependent; engine is NUL-transparent; source excluded by name | `test_corpus_replay.py:40-49`; `pins-wave1/pins_like.json` | n/a | `UNSPECIFIED` / permanent | |
| **D4** | a trapping subexpression the optimizer deletes, we still evaluate — the standing price of optimizer-off. **Not an engine-vs-oracle divergence**: against the oracle we agree exactly, so the gap is engine-vs-the-user's-optimizer-on-DuckDB, and ORC-06 reports it (`DIVERGE_OPT`) rather than accepting it. Listed here because it is the tolerated cost readers look for | `known-limitations.md:231-257`; `known_divergences/test_trap_elision.py`; campaign snapshot `packages/confit/findings.jsonl` (see D16) | 1 **against the contract surface**, n/a against the oracle | `PINNED` / permanent | |
| **D5** | `%`-by-zero NaN sign bit is platform-libm; the pin is per-platform bit agreement, not a constant | `test_duckdb_wave3_mathtail.py:205-235`; `pins-wave3/math_tail.json` | n/a | `IMPL-DEFINED` / permanent, discriminator = platform | |
| **D6** | schema qualifiers are registry-noise: `s1.t1` resolves on the bare table name; DuckDB's schema-existence errors are not reproduced; `w.w.w` binds the longer schema-ish parse | `known-limitations.md:260-272` | **3 for the first two clauses** (`s1.t1` on a non-existent schema serves here and gets `schema "x" does not exist` on DuckDB), **4 for `w.w.w`**. The source's "always a loud build-time rejection, never a different served value" closes the `w.w.w` sub-case only and was mis-lifted to the whole row | `PINNED` / permanent | |
| **D7** | DECIMAL **literals** are f64; exact-decimal accumulation is not reproduced. Three visible faces: a schema delta (decimal128 vs float64), a rounding-mode delta (D8), and a signed-zero delta (`- 0.0` has no sign as DECIMAL) | `known-limitations.md:166-174`; `oracle.py:113-131, :512-513`; faces measured 2026-08-25 over `findings.jsonl` seeds 869/998/1554/3269 | 2 | *unruled* — see ASK-6; the reported value delta also depends on ASK-12 | |
| **D8** | **D7's named consequence, not an independent divergence.** `CAST(-2.5 AS BIGINT)`: DuckDB types the bare `-2.5` as `DECIMAL(2,1)` and casts DECIMAL->BIGINT half away from zero (`-3`); we type it f64 and cast DOUBLE->BIGINT half to even (`-2`). Both engines agree on both casts *given the type* — the divergence is entirely the literal's type | `known-limitations.md:166-174` treats it as one limitation with "one visible consequence". **Not** `known_divergences/test_cast_semantics.py`: that file records the DOUBLE cast as FIXED 2026-08-08, contains no bare-literal test, and warns "Measure a DOUBLE cast with a DOUBLE column or an explicit `::DOUBLE`, never a literal", assigning the mechanism to D7 | 2, **the same instance as D7** | *unruled with D7* — see ASK-6 | |
| **D9** | bind-time constant refusals: `WHERE FALSE` and empty-input shapes refuse here and serve on DuckDB | RFC `2026-08-19-keep-the-bind-time-refusals.md:29-58` (the phase-separated measurement) and the decision ACCEPTED twice. **No executable twin exists**: `known_divergences/test_literal_typing.py:77-131` pins refusal where DuckDB *traps* (its DuckDB leg asserts `pytest.raises(..., "[Oo]verflow")`), which is the absorbed case, and `WHERE FALSE` appears in the suite only as a DuckDB probe in `test_trap_elision.py`. The missing twin is the same absence ASK-3 is about | 4 | `PINNED` / permanent — **but its cost is uncountable and unpinned, see ASK-3** | |
| **D10** | one-sided regex program-size guard: always fires before DuckDB's real RE2 budget, so it may over-refuse and can never serve where DuckDB errors | `pins-waveB/fuzzer-20260728.json`; `pins-first-methodology.md:79` ("the asymmetry is the contract") | 4 by construction | `PINNED` / permanent | |
| **D11** | narrow-lane overflow trap threshold not yet shipped: an overflowing narrow lane serves the i64 value on the row path and refuses by name at the `infer_arrow` boundary | `known-limitations.md:177-188`; catalogue pinned in `test_integer_widths.py` | 3 (row path) | `PINNED` / **until-fixed** (m-8 phase 3) — owes a strict-xfail twin if until-fixed is accepted | |
| **D12** | ~~unattributed~~ **attributed** campaign residuals in the committed 2026-08-17 baseline. Re-measured 2026-08-25, four of the five are attributed and the row's original premise was wrong: seeds **869, 1554, 3269** carry the `decimals` tag and are bare-DECIMAL-literal cases, i.e. **D7**, and their reported 1-ulp delta is produced by the harness's own cast (ASK-12); seed **998** is `(c3.f1 * (- 0.0))`, the same D7 mechanism in its signed-zero face (ORC-33); seed **2668** (`-2147483648 / -1`) is **TASK-122, Done**, whose AC #5 records "the campaign's seed 2668 class is gone at 4000 seeds". What survives as genuinely unruled is not a residual set but a question about the file: see D16 | `packages/confit/findings.jsonl`, counted 2026-08-25: 28 findings = 16 `DIVERGE_BUILD` + 7 `DIVERGE_OPT` + 4 `DIVERGE_VALUE` + 1 `DIVERGE_TRAP`, tags and SQL read per seed; `2026-08-17-fuzz-triage.md:62-63`; `backlog/tasks/task-122 ...md:63, :86-88` | 2, **as D7** | *ruled by attribution* — D7's ruling covers 869/998/1554/3269; 2668 is closed. See ASK-6 and ASK-12, not ASK-9 | |
| **D13** | the phase-2 width residuals, quoted from memory as 79 of 84, treated as a defect count | **Unverified** — the numbers appear nowhere in the tree (searched `packages/confit/docs/`, `backlog/`, `findings.jsonl`, 2026-08-25) | unknown | *unruled* — see ASK-10 | |
| **D14** | the **1 GiB string-builder budget**: a literal pad/repeat count that can exceed it refuses at build by name. Measured 2026-08-16, DuckDB is entirely deterministic here (`repeat` serves to n <= 4294967295 and errors above; `lpad`/`rpad` binder-error above INT32), so the ground is ours and is a judgement: "a serving engine does not allocate a gigabyte per row. **We refuse where DuckDB would serve**" | `known_divergences/test_string_budget.py:108-132` (a passing KEEP entry with a restated measured REASON); `known-limitations.md:205`, which **still carries the corrected-false ground** (ORC-53) | 4 | `PINNED` / permanent, resource class (ORC-79) | |
| **D15** | the **2 GiB-per-arrow-batch ceiling** that comes with matching DuckDB's `pa.string()` 32-bit offsets: refused by name rather than wrapped | `known_divergences/test_arrow_boundary.py:34-36` | 4 | `PINNED` / permanent, resource class (ORC-79) | |
| **D16** | **the baseline file's standing as evidence.** `packages/confit/findings.jsonl` is cited as evidence by D4 and by section 11's corrections, and it is a **snapshot of the 2026-08-17 campaign, not a live artifact**: TASK-129's notes record "the 8% draw shifts rng for static-bearing seeds, so residue seed IDs moved ... The committed findings.jsonl baseline is a historical artifact and was not regenerated here", and at least two of its classes are since closed (TASK-121's 16 `DIVERGE_BUILD`, TASK-122's seed 2668) while still sitting in the file. Any instruction phrased as "re-run seed N" against it is not executable as written | `backlog/tasks/task-129 ...md:148-150`; `backlog/tasks/task-121 ...md:84` (78/78 re-classify `REFUSED`); `backlog/tasks/task-122 ...md:86-88` | n/a (evidence hygiene) | *unruled* — see ASK-14 | |

*Verified-by:* each row cites its own evidence. The enumeration is complete against
`packages/confit/docs/known-limitations.md` section 5 plus section 3's value-family
rows, `packages/confit/tests/known_divergences/` (all ten modules, re-swept 2026-08-25 —
which is how D14 and D15 were added), and the committed campaign snapshot, all as of
master `85b4739`. Note D8 is D7's consequence and D4 is not an engine-vs-oracle
divergence at all; counting *distinct tolerated engine-vs-oracle divergences* the table
holds fewer rows than it has ids, and the ids are kept stable rather than renumbered.

**Closed, deliberately not a row:** the 16-seed `DIVERGE_BUILD` ambiguous-reference
class (the largest single class the 2026-08-17 campaign saw, 57% of findings) is
**TASK-121, status Done**. Its closure evidence is the implementation note "All 78
ambiguous findings of the 20k campaign now classify `REFUSED` (78/78 re-run
individually)" at `backlog/tasks/task-121 ...md:84` — **not** its acceptance criteria,
which are all five unticked, including "#5 the campaign's `DIVERGE_BUILD` ambiguity class
is gone at 4000 seeds". Stated plainly because ORC-54 uses unticked ACs to convict
TASK-95, and the same standard has to apply here: the class is closed on a different
campaign than AC #5 names, and the snapshot file still holds all 16 seeds (D16).
`packages/confit/docs/2026-08-17-fuzz-triage.md:56-58` and `:70-72` still say it is "not
yet ticketed" and "has no ticket". See proposed ticket T-10.

### 7.4 Placement

**ORC-56.** **[PROPOSED]** Not in force. Where confit deliberately does not match DuckDB,
the note belongs **at the requirement it violates**, with this ledger as the index; a
divergence filed only in an appendix stops being read. D9 (bind-time refusals) and D10
(one-sided guards) are the two rows this would apply to today. The honest obstacle: there
is no engine spec to place them in, so adopting this means either naming the destination
document or accepting that "the requirement it violates" is `known-limitations.md`'s own
section, which is where they already are.
*Verified-by:* Unverified — the anti-pattern is real (it is the reason ORC-53's census
found readers walking past entries), but no decision outside this document states the
placement rule. Part of ASK-15.

> ### ASK-8 — does "an unlisted divergence is a bug by definition" go in the spec?
>
> This one sentence converts "bit-for-bit or refuse" from an aspiration into a
> falsifiable promise. It also binds us: it makes the section 7.3 table's completeness
> a contract, and any divergence found in the wild becomes automatically a defect
> rather than a discussion.
>
> Recommendation: adopt the sentence, without an SLA. The SLA form exists to serve
> external implementors on a clock; this project has no external claimants.
>
> *Binds:* ORC-55 and every future divergence.

> ### ASK-9 — do we admit a "measured but not yet ruled" bucket?
>
> KEEP and CHANGE both presuppose a ruling (ORC-50). Things that are measured facts
> awaiting your call today live only in prose: build-vs-build repeatability (ORC-22),
> intra-value order under `threads` (ORC-75), the width residuals (ledger D13), and the
> baseline's standing as evidence (ledger D16). A `tentative` **tag** — not a third
> directory — would give them a home without promoting them to contract.
>
> The stricter alternative is that everything measured gets ruled at measurement time.
> That is slower, and it is a real option: it means a campaign cannot end until its
> residuals are classified. The repo already leans this way — the 2026-08-13 triage
> closes with "Every family is now mapped to a ticket", and the 2026-08-17 triage
> attributes every family it found. D12's original "no decision, no ticket" framing was
> wrong precisely because that practice is already in force.
>
> *Correction to this block's original premise:* it named D12's residuals as the second
> of three unruled things. They are attributed (see D12), so the open set is the four
> named above.
>
> *Verified-by (the facts):* `packages/confit/docs/2026-08-13-fuzz-triage.md:150-165`
> ("Every family is now mapped to a ticket").
>
> *Binds:* ORC-22, ORC-75, ledger rows D13 and D16.

> ### ASK-14 — is a campaign baseline evidence, or a snapshot?
>
> `packages/confit/findings.jsonl` is cited across this document as standing evidence,
> and it is a snapshot of one campaign that the tree itself records as superseded (ledger
> D16). Two of its classes are closed by Done tickets and still sit in the file, and its
> seeds are not re-addressable: TASK-129 records that the 8% static draw shifted the rng
> for static-bearing seeds, so a seed id no longer names the case it named in the file.
>
> That makes any instruction of the form "re-run seed N and classify it" — which is what
> ASK-10's remedy would ask for — unexecutable as written; the cases are recoverable only
> from the `sql` field stored in the file.
>
> Three options:
>
> - **regenerate on a cadence and commit the result**, so the file is always the current
>   campaign and "re-run seed N" means something;
> - **freeze it as a dated artifact** and rename it accordingly, so it is read as history
>   and never cited as current state;
> - **stop committing it** and cite triage documents (which carry dates and attributions)
>   as the evidence instead.
>
> Whichever way it goes, the operative sentence this document owes is: *a finding is
> addressed by its stored SQL, not by its seed, unless the generator is unchanged.*
>
> *Verified-by (the facts):* `backlog/tasks/task-129 ...md:148-150`;
> `backlog/tasks/task-121 ...md:84`; `backlog/tasks/task-122 ...md:86-88`.
>
> *Binds:* ORC-66, ledger rows D4, D12, D16, and ASK-10.

---

## 8. The severity ladder

**ORC-57.** One definition, cited everywhere, restated nowhere:

| rung | meaning | direction |
|---|---|---|
| **1** | we trap where DuckDB serves | contract violation |
| **2** | we serve a wrong value | contract violation |
| **3** | we serve where DuckDB refuses | directional — the query cannot be run against DuckDB at all |
| **4** | we refuse where DuckDB serves | directional — the safe side, sometimes chosen on purpose |

*Verified-by:* the two existing parenthetical definitions, which are **compatible but not
identical** — `packages/confit/docs/specs/2026-08-25-task-114-design.md:140-142` defines
rungs 1-4, `packages/confit/docs/specs/2026-08-25-task-127-remainders-design.md:154-156`
defines only 2-4 — plus use by name in the Rust source at
`packages/confit/src/specializer/frontend.rs:64-73` ("refusing is the severity ladder's
own preference") and in both RFCs.
*Note:* two partial definitions that agree today is one definition away from drift, which
is the case for consolidating them here. Proposed ticket T-11 replaces the parentheses
with a citation of this claim.

**ORC-58.** The ladder is also the scope tiering, and rungs 3 and 4 are **directional**:
rung 4 is the safe direction, and choosing it deliberately is legitimate — D10's
one-sided guard states it outright, "it may over-refuse; it can never serve where DuckDB
errors. The asymmetry is the contract." Rung 3 is the unsafe direction and is chosen
only where DuckDB cannot be run at all (D6's schema qualifiers, D11's row path).
*Verified-by:* `packages/confit/docs/reports/pins-first-methodology.md:79`; ledger rows
D6, D9, D10, D11.
*Correction:* an earlier version added "Rungs 1 and 2 are contract violations and are
never accepted." No source in the tree states that absolute, and the ledger contradicts
it on its face — D7/D8 are sev 2 and tolerated (as a feature in flight, ORC-80), and
D4 is sev 1 against the contract surface and `PINNED` / permanent. The true sentence is
weaker and is what the tree supports: *a rung-1 or rung-2 divergence against the oracle
is a defect unless a named decision places it under ORC-80 or ORC-72.* Whether the
stronger absolute is adopted is part of ASK-15.

**ORC-59.** **[PROPOSED]** Not in force, because ORC-31 is not. A deliberate rung-4
choice must be countable: rung 4's whole defense is that it is the safe direction, and
that defense is only inspectable if the class size is observable. D9 is the live instance
and the mechanism it needs is ASK-3.
*Verified-by:* Unverified — it follows from ORC-31, which is itself `[PROPOSED]`.

---

## 9. Version bumps and mutability

The urgency is external: DuckDB ships minor versions roughly every four months,
semantics have already moved inside a patch release, and v2.0 brings a new SQL parser.
This protocol is cheap to write now and expensive to write during a migration.

**ORC-60.** **[FACT]** The bump protocol's two preconditions — the pinned oracle version
is **one named constant** (ORC-02) and appears in **every pin file** (ORC-46) — are
today unenforced and partial respectively. Everything below depends on both.
*Verified-by:* ORC-02, ORC-09, ORC-46.

**ORC-85.** **[FACT]** Step one of a bump is not re-recording; it is making the corpus
re-runnable, and that work is not ticketed anywhere. Measured 2026-08-25 over the 53 pin
files: **21 distinct top-level key sets**; the claim body is spelled `probes` / `pins` /
`findings` / bare domain keys; the query field is spelled four ways (`query` in 20 files,
`sql` in 12, `q` in 13, `expr` in 7); a header key exists in **2** files (`_meta` in one,
`meta` in one); inputs are frequently prose rather than data
(`"input_repr": "t(a BIGINT, b BIGINT); rows=[(7, 2)]"`, or an `expr` string holding
three queries at once, or setup buried in a free-text `note`); and exactly **one** pin
ships a re-runnable capture harness beside it (`pins-dialect/probe_joins.py`). The
operative sentence this document owes the next migration: **a pin that cannot be re-run
mechanically is not re-recordable, and enumerating and converting those is the bump's
first task** — before ORC-61's diff report has anything to run against.
*Verified-by:* measured 2026-08-25 over `packages/confit/docs/specs/pins-*/*.json` and
`pins-*/`. Proposed ticket T-20.

**ORC-86.** **[FACT]** Pin *capture* runs outside the oracle fixture, and at least one
pin file was demonstrably captured with the optimizer **on**. The autouse fixture covers
`packages/confit/tests/` (ORC-07); capture scripts are not tests. Measured: no capture
script applies `PRAGMA disable_optimizer` — `scripts/gen_casemap.py`,
`scripts/gen_pow10.py`, `scripts/gen_strip_accents.py` and `scripts/mine_duckdb_corpus.py`
all use a bare `duckdb.connect()` — and `pins-stageB/order-contract.json`'s own notes
("The optimizer picks stream/build sides by cost", "Chained-join nesting is
optimizer-chosen") describe an optimizer-on capture. Only 4 of 53 pin files mention the
optimizer at all. By ORC-02's own sentence — a comparison run against anything else is
not a comparison against the oracle — a real part of the corpus is not a recording of
*this* oracle, and this document does not currently say whether that matters. It is
recorded here as state, not ruled: see ASK-1 and proposed ticket T-7.
*Verified-by:* measured 2026-08-25; `scripts/mine_duckdb_corpus.py:111`;
`packages/confit/docs/specs/pins-stageB/order-contract.json` (its own notes).

**ORC-87.** **[FACT]** The mined corpus has no provenance and its expectations are
optimizer-on. `scripts/mine_duckdb_corpus.py` recomputes every case's expected rows by
running it in a fresh `duckdb.connect()` at mining time — deliberately ignoring the
sqllogictest file's own expected block, "and DuckDB itself is our oracle anyway" — at an
unrecorded version, on an unrecorded date, with the optimizer on, and
`tests/corpus/duckdb_mined.jsonl` carries no provenance field. The 550-of-678 headline
(ORC-67) rests on that artifact, and ORC-46's provenance measurement scopes to `pins-*/`
only, so the largest single recorded artifact in the project sits outside it.
*Verified-by:* `scripts/mine_duckdb_corpus.py:1-12` (the recompute rule) and `:111` (the
bare connect); `packages/confit/tests/corpus/duckdb_mined.jsonl` (678 lines, no
provenance field). Proposed ticket T-21.

**ORC-61.** **[PROPOSED]** Partly in force; the generalization is not. Re-record means one
command that re-runs a pinned artifact against a new build and emits a **diff report,
never a silent rewrite** — a silent re-record is indistinguishable from adopting
whatever the new build does, which is the opposite of pinning. The pattern already
exists, correctly, for exactly one artifact: `scripts/pin_ast_shapes.py` re-pins the
AST shape manifest against the installed DuckDB "so that a version bump surfaces as a
reviewable diff instead of as a wrong answer three layers down", with the usage note
`git diff  # this diff IS the drift report`. What does not exist is the same command
over the 53-file pins corpus.
*Verified-by:* `scripts/pin_ast_shapes.py:1-12` (the prototype, for
`sql_transform/model/_shapes.json`); the pins corpus has no equivalent (measured
2026-08-25). Proposed ticket T-12.

**ORC-62.** **[PROPOSED]** Not in force. Every diff row from a re-record is triaged into
exactly one of four classes, and each class has a different consequence:

| class | consequence |
|---|---|
| DuckDB bug fixed upstream | update the pin, note the change, keep the evidence |
| DuckDB behavior changed | owner decision; becomes a ledger row |
| confit bug newly exposed | a ticket, and a strict-xfail pin |
| now-abstaining | we were passing on luck; move it into the section 3.2 disposition table |

*Verified-by:* Unverified — proposed with ORC-61.

**ORC-63.** **[PROPOSED]** Not in force. Each decision carries a **mutability class**, so
a bump's diff can be triaged mechanically rather than argued case by case. Three
classes, and the assignment for the decisions in this document:

| class | meaning | which decisions |
|---|---|---|
| `frozen` | a change means the oracle is wrong; we file upstream and keep the pin | ORC-01 delegation, ORC-13 axiom, ORC-32 bit-for-bit floats, ORC-35 multi-answer rule, ORC-41 rejected tolerances, ORC-50 ledger split, ORC-57 severity ladder |
| `follows-oracle` | support widens monotonically as the oracle's does; a bump may legitimately move it | ORC-10/ORC-11 inherited quirks, ORC-37 name dedup, ORC-39 error texts, D1, D6 |
| `may-change-on-bump` | expected to move; the bump protocol is where it gets re-decided | ORC-02 the version itself, ORC-05 what the pragma removes, ORC-34 / D5 platform NaN, D7/D8 decimal family, D11 narrow widths, ORC-18 compare modes if the parser changes |

*Verified-by:* Unverified — the classification is proposed here for ratification. Note
ORC-81's libm-backed pins and ORC-76's cbrt tolerance are `may-change-on-bump` if this
lands, and no claim covers them today.

**ORC-88.** The **live-oracle remeasure guard** is the in-force partial form of what
ORC-61 proposes: a divergence or parity test asserts the *oracle's* answer first, in the
same test, so an oracle that moves under us fails loudly at that assertion rather than
silently changing what parity means. It covers the tests that use it; it does not cover
the pins corpus, which is exactly the gap ORC-61 names.
*Verified-by:* `packages/confit/docs/specs/2026-08-19-cast-semantics-design.md:25`;
`packages/confit/docs/specs/2026-08-25-task-120-design.md:374`;
`packages/confit/docs/specs/2026-08-25-task-133-join-keys-design.md:625`; the pattern is
visible in every `known_divergences/` module that opens a connection.

**ORC-64.** **[PROPOSED]** Not in force. A pin whose value changed across a bump is itself
a decision and earns a line in this document, not just a new JSON blob. A changed pin
that leaves no trace here makes the next reader unable to tell a fix from a regression.
*Verified-by:* Unverified — proposed with ORC-61.

---

## 10. Campaign validity and blind spots

**ORC-65.** **[PROPOSED]** Not in force. A campaign residual that lands in an
`UNSPECIFIED` region is **not a parity defect**: classify before counting, because a
number that mixes determined-and-wrong with under-determined is not a defect count and
must not be treated as a backlog. The rule follows from ORC-13, but nothing outside this
document states it, and its own basis ORC-15 is `[PROPOSED]` too. Its immediate
application is ledger row D13 and ASK-10.
*Verified-by:* Unverified. Part of ASK-15.

**ORC-66.** **One** standing differential gate runs in the normal test gate: the regexp
fuzzer at N=250 with a fixed seed (`REGEXP_FUZZ_SEED` / `REGEXP_FUZZ_N` for deep runs).
Its first deep run found 122 divergences distilled to 12 reject classes, then re-swept to
**zero divergences over 40k cases across 8 seeds**. Its four-outcome rule is decided and
worth stating, because it is a precedent: duckdb-ok + engine-rejects is declared "fine
(conservative bind-time reject)" — a designed, unconditional rung-4 absorb — and a finding
is dispositioned as a reject-list entry plus a pin note plus a limitations row.
*Verified-by:* `packages/confit/tests/test_duckdb_regexp_fuzz.py:13-20` (the four-outcome
rule and the finding-disposition protocol), `:35-37` (the seed and N defaults);
`packages/confit/docs/specs/pins-waveB/fuzzer-task54.json`;
`packages/confit/docs/reports/pins-first-methodology.md:70-74`.
*Correction:* an earlier version of this claim, and `known-limitations.md:301-307`, call
the **campaign fuzzer** a second standing gate. It is not. `packages/confit/fuzz/` is a
manual CLI (`python -m fuzz.runner`) and is not collected — `packages/confit`'s
`testpaths = ["tests"]`. What runs in the gate is `packages/confit/tests/test_fuzz_smoke.py`,
whose docstring says the opposite of a differential gate: "The fuzzer exists to find live
bugs, so 'no findings over N seeds' **cannot** be the CI invariant ... What CI pins
instead: generation is deterministic, the oracle produces verdicts across the seed range,
verdicts are reproducible" — machinery, not zero findings. That distinction is what §10's
continuity claim actually rests on. See proposed ticket T-22.

**ORC-89.** A campaign is an **acceptance test, not a formality**: each m-8 phase ends
with a fuzz campaign certifying it before the next starts, and for a feature whose
scaffolding included a fuzzer suppression tag, the certification campaign *after* the tag
removal is what proves the class is gone rather than hidden (ORC-80).
*Verified-by:* `backlog/milestones/m-8 - duckdbs-type-lattice.md:40-41`;
`packages/confit/docs/specs/2026-08-11-duckdb-type-lattice-design.md:110-131`.

**ORC-67.** The corpus replay gates **zero FAILs, always**. Three outcomes — match,
clean-unsupported, FAIL — where a rejection is only clean if it is one of the
*documented* rejection classes, so an undocumented error is a FAIL like any wrong
answer. The match count is deliberately **ungated**: it is the growth ladder
(53 -> 395 -> 505 -> 511 -> 529 -> 546 -> 550 of 678), and every construct learned flips
cases from clean-unsupported to match, never into FAIL.
*Verified-by:* `packages/confit/tests/test_corpus_replay.py:17-20, :179-198`;
`packages/confit/docs/reports/pins-first-methodology.md:41-62`.
*Correction:* the number 550/678 is quoted **without its hedge at six sites**, and tasks
have landed since that flip cases. `known-limitations.md:12` is the one place that
carries the "as of stage B" hedge; the unhedged sites are
`reports/confit-architecture.md:3`, `reports/performance-report.md:9`,
`reports/pins-first-methodology.md:3` (the abstract) and `:124` (a second quote in the
same document), `packages/confit/README.md:109`, and the repo-root `README.md:112`. A
remedy scoped to "the abstract and two reports", or to "exactly one place", would leave
three of them standing. See ASK-11 and proposed ticket T-13.
*Second correction, structural:* the corpus's expected rows are themselves an
optimizer-on recording with no provenance (ORC-87), so this ladder is measured against a
different reading of DuckDB than the one ORC-02 names. That does not make the zero-FAILs
gate wrong — a match is still a match — but it does mean the ladder is not evidence about
the oracle in ORC-02's sense, and section 9 cannot re-record what has no provenance.

**ORC-68.** Our full-result diff is **stronger** than any partial oracle wherever it is
total. Wherever it is partial we have re-created a partial oracle and we own its blind
spot. The blind spots, named:

| blind spot | what the oracle does not pin there | mitigation in force |
|---|---|---|
| row order **in every mode** except a total `ORDER BY` — including the row path, where DuckDB's order is not a function of the query either (ORC-18) | the sequence, against DuckDB | multiset against DuckDB, plus **our-side-only** self-legs (ORC-16); the self-legs pin our contract, not agreement with the oracle |
| order **inside a value** (`string_agg`, `list`) on the frozen path | element order, which is a function of `threads` (ORC-75) | **none today** — the assigned compare mode cannot see it. This is ASK-13 |
| `order-by-unevaluated` fallback | sortedness on a non-output-column key | logged tag, not silent (ORC-27) |
| error texts (D2) | message bodies for bind-time rejections | error *class* is compared; texts are not oracle-decided output (ORC-39) |
| the excluded ILIKE-NUL source (D3), the f32 blanket rule and `_INEXPRESSIBLE_INPUTS` (ORC-77) | statistics-dependent kernel selection; every f32-grid-sensitive operation; declared-schema-inexpressible inputs | exclusion by name with a measured reason (ORC-20, ORC-77) |
| refusals | whether the oracle would have served (ORC-30) | **none today** — this is ASK-3 |
| NaN sign and payload, in both differential gates | which NaN | `repr` makes every NaN self-equal in `_key` and `_norm_row`; only the explicit bit pins see the difference (ORC-32) |
| the harness's own normalization | whether the cast that makes the two legs comparable preserves the value | **none today** — measured not to, for decimal literals. This is ASK-12 |
| a worker that never answered | everything about that case | `TIMEOUT` / `PANIC` are findings, not silence (ORC-23, ORC-26), and are attributed oracle-side vs engine-side by hand (ORC-78) |

*Verified-by:* each row's cited claim.

**ORC-69.** Metamorphic self-legs are the oracle *substitute* in the abstention region,
and the set is capped rather than grown: batch-vs-single sequence equality, input
reversal reversing the output blocks, hostile-arrow invariance (sliced, chunked,
empty), `infer_rows` vs `infer_arrow` agreement, cranelift vs interpreter agreement,
and sklearn as a second ground truth on plain tree cases. They involve no DuckDB, which
is what makes them usable exactly where the oracle abstains. Five of the six compare
exactly; the sklearn leg compares within `1e-9` (ORC-32), because sklearn is a second
reference and not the oracle.
*Verified-by:* `packages/confit/fuzz/oracle.py:748-858` (`_extra_legs`), `:799-836` (the
sequence and reversal legs), `:851` (the sklearn `1e-9` bound), `:625-639`;
`packages/confit/tests/test_fuzz_order_legs.py`; P1-P20 in
`packages/confit/docs/properties.md`.

**ORC-70.** **[PROPOSED]** Not in force. A campaign declares a coverage signal, because
"20k queries, N residuals" is unanchored without a denominator that means something. We have no query
plans to diversify over, so the analogue is distinct **(operator, argument-type,
edge-class)** triples reached per campaign — which extends the existing `AGREE`-only
construct histogram's axis rather than replacing it. That same triple is the right unit
for decision coverage: one operator x one type x one edge class, not one feature.
*Verified-by:* the histogram exists at `packages/confit/fuzz/runner.py:38-41, :159-164`;
the triple axis does not. Proposed ticket T-14.

**ORC-71.** **[PROPOSED]** Not in force. A campaign reports an **abstention rate per
kind** alongside the coverage histogram, and a rising rate is read as generator drift
rather than as good news. A generator that has drifted out of the answerable region
measures nothing while still printing a green bar. The **kinds** it would report over all
exist and are all findings today — `SKIP` (ORC-26), `TIMEOUT` and `PANIC` (ORC-23), the
`order-by-unevaluated` tag (ORC-27) — and the rule that oracle-side and engine-side
timeouts are opposite things is **already decided** (ORC-78). What does not exist is the
rate, and the machine-readable separation ORC-78 asks for by hand.
*Verified-by:* the kinds exist per ORC-23, ORC-26, ORC-27; ORC-78 for the attribution
rule; the rate does not exist (`runner.py:147-150` prints raw verdict counts only).
Proposed ticket T-15, and see ASK-5 for the user-visibility boundary.

> ### ASK-10 — the width residuals: defect count or mixed bag?
>
> ORC-65 says residuals in the unspecified region are not parity defects, so before the
> width-residual number is treated as a backlog, each residual needs classifying as
> determined-and-wrong vs under-determined. That classification is work, and
> authorizing it is yours; the answer changes what the number means.
>
> One honesty note first: **the 79-of-84 figure is not reconstructible from this tree.**
> Searched 2026-08-25 across `packages/confit/docs/`, `backlog/`, and the committed
> `findings.jsonl` — the numbers appear in none of them. Whatever is authorized should
> begin by re-running the campaign that produced them, because a count nobody can
> re-derive is not evidence.
>
> A second, practical one: the committed baseline **cannot** serve as the starting point
> for that re-run. Its seeds no longer address the cases they addressed (ASK-14 / ledger
> D16), so any classification pass has to start from a fresh campaign, not from the file.
>
> *Binds:* ledger rows D13 and D16.

> ### ASK-11 — does the corpus match count become a ratchet?
>
> Zero FAILs is the gate and should stay the gate (ORC-67). The match count is
> deliberately ungated as a growth ladder, and it has now gone stale in three documents
> because nothing watches it. Two options:
>
> - **leave ungated**, and generate the number with a date stamp in exactly one place
>   so drift cannot recur silently — note the drift is at **six** sites, not three
>   (ORC-67's correction);
> - **add a never-decreases ratchet.** Real cost: any deliberate scope reduction
>   becomes a gate failure, and a scope reduction is sometimes the right call.
>
> The precedent for the second option already exists inside this package and is worth
> reading before ruling: the dialect L3 gate carries exactly that ratchet — "The match
> floor is the measured count at introduction — raise it when the surface grows, never
> lower it" (ORC-73) — so the question is whether it generalizes from a printed-surface
> gate to a mined-corpus one, not whether it is workable.
>
> *Binds:* ORC-67, ORC-73.

> ### ASK-15 — do this document's own rules become normative?
>
> Eight rules in this document (across nine claim ids) were originally written as
> decisions in force with a `Verified-by` of "derived" or "normative here", or with no
> source at all. They are not decisions in force: nothing
> outside this document states them, and re-checking found no spec, ticket or review that
> adopted them. They are marked `[PROPOSED]` now rather than deleted, because each one is
> a real answer to a real question — but adopting a rule is your call, not a document's.
>
> | claim | the rule | the cost of adopting it |
> |---|---|---|
> | ORC-15 | every comparison target carries a status | eight claims have one today; the rest must be statused or the rule narrowed to the ledger |
> | ORC-31 | an accepted cost must be countable, with the mechanism named in the decision | makes ASK-3's split mandatory rather than optional, and applies retroactively to D10 |
> | ORC-35 | multi-answer sets are legitimate only with a selecting predicate | no pin in force is set-valued in that sense; ORC-76's bounded tolerance is a shape the rule does not cover |
> | ORC-41's four unadopted mechanisms / ORC-84 | they become standing rejections | must be written so as not to contradict ORC-32's three in-force tolerances or ORC-73's designed epsilon tier |
> | ORC-56 | a divergence note sits at the requirement it violates | there is no engine spec to put them in |
> | ORC-58's absolute | "rungs 1 and 2 are never accepted" | contradicted on its face by D4, D7 and D8; adopting it means re-ruling those |
> | ORC-59 | a deliberate rung-4 choice must be countable | follows ORC-31 and stands or falls with it |
> | ORC-65 | an `UNSPECIFIED` residual is not a parity defect | needs ORC-15 first, since "unspecified" has to be a recorded status to be checkable |
>
> The cheap option is to adopt none of them and let the document record only what was
> decided elsewhere. The expensive-but-useful option is to adopt them individually. The
> option to avoid is leaving them marked forever, because a `[PROPOSED]` rule that
> everyone cites is a decision that was never made.
>
> *Binds:* ORC-15, ORC-31, ORC-35, ORC-41, ORC-56, ORC-58, ORC-59, ORC-65, ORC-84.

---

## 11. Proposed tickets

Everything this document wants changed in code or in another document. **None of it is
applied here** — this deliverable is one markdown file.

| id | change | claim | blocked on |
|---|---|---|---|
| **T-1** | correct four statements in `conftest.py` and `known-limitations.md`: (a) `:20` / `:30`, `disable_optimizer` also changes behavior at **twelve** sites outside the pass list, two of them in `src/planner/binder/`; (b) `conftest.py:22`, "constant folding still happens" is an observation, not a mechanism — the folder is an optimizer rule; (c) `conftest.py:14`'s "the 42 call sites" is 62 | ORC-05, ORC-07 | editorial, no ruling needed |
| **T-2** | assert `duckdb.__version__` in the fixture that already applies the pragma | ORC-09 | ASK-1 |
| **T-3** | number the axiom as P21 in `properties.md`; make its three existing sites cite it | ORC-13 | owner's go |
| **T-4** | correct the source count: `pins-first-methodology.md:89` says two sources, the set holds one (two statements) | ORC-20 | editorial |
| **T-5** | split `REFUSED` into `REFUSED_ORACLE_TRAPS` / `REFUSED_ORACLE_SERVES`, the latter in `INTERESTING` | ORC-30 | ASK-3 |
| **T-6** | put phase-separated probing into the methodology report, where "how we measure DuckDB" lives | ORC-45 | owner's go |
| **T-7** | uniform provenance header on every pin file | ORC-46 | owner's go |
| **T-8** | decision back-reference (`ORC-NN`) field on every pin | ORC-48 | owner's go |
| **T-9** | an inline under-determined-field token in the pin format | ORC-49 | owner's go |
| **T-10** | correct `2026-08-17-fuzz-triage.md:56-58, :70-72`: the ambiguous-reference class is TASK-121, Done | section 7.3 | editorial |
| **T-11** | delete the two parenthetical severity-ladder definitions; cite ORC-57 | ORC-57 | editorial |
| **T-12** | generalize `pin_ast_shapes.py`'s drift-report pattern to the pins corpus | ORC-61 | owner's go — **not** ASK-1(b), which an earlier version listed. Which version the corpus targets has no bearing on whether a re-record tool exists, and that spurious dependency is what made section 9 read as having no first step |
| **T-13** | generate the corpus match count with a date stamp in exactly one place, and correct the **six** unhedged sites ORC-67 enumerates | ORC-67 | ASK-11 |
| **T-14** | extend the coverage histogram to (operator, arg-type, edge-class) triples | ORC-70 | owner's go |
| **T-15** | report an abstention rate per kind per campaign, and classify oracle-side vs engine-side timeouts in the runner rather than by hand (ORC-78's recorded follow-up, plus recording the SQL before executing) | ORC-71, ORC-78 | ASK-5 |
| **T-16** | `SET threads = 1` in the fixture that already applies the pragma, if ASK-13(a) takes that option | ORC-75 | ASK-13 |
| **T-17** | reconcile `_CLEAN`'s two unprefixed messages with the documented three-prefix rule, either way | ORC-29 | editorial |
| **T-18** | make the campaign's schema normalization value-preserving, whichever way ASK-12 rules | ORC-38 | ASK-12 |
| **T-19** | correct `known-limitations.md:205`: DuckDB is deterministic on pad/repeat budgets; the "spelling-dependent" ground was measured false and restated 2026-08-16 | ORC-53, D14 | editorial |
| **T-20** | enumerate the pins that cannot be re-run mechanically and convert them; this is the bump's actual first task | ORC-85 | owner's go |
| **T-21** | stamp provenance onto `duckdb_mined.jsonl` at mining time (version, date, settings profile) | ORC-87 | owner's go |
| **T-22** | correct `known-limitations.md:301-307`: the campaign fuzzer is a manual CLI, not a gate mechanism; what runs is `test_fuzz_smoke.py`, and its invariant is machinery | ORC-66 | editorial |

Two further editorial corrections found while writing, no ticket needed if fixed in
place: `known-limitations.md:248-249` says a 4000-seed campaign puts trap elision at
"8 seeds in 28 findings; all eight are labelled `DIVERGE_OPT`" — the committed snapshot
holds **7** `DIVERGE_OPT` seeds (312, 812, 1196, 1563, 1564, 2174, 2805) of 28 findings
(counted 2026-08-25 over `packages/confit/findings.jsonl`), and
`2026-08-17-fuzz-triage.md:102` names only five of them in prose, while its table at
`:87` says `OPT_EMULATED = 0` and its prose at `:89-94` says one remains.

And one comment that is *not* merely editorial, because it states the opposite of a claim
in force: `packages/confit/fuzz/runner.py:166-168` still reads "The passes we reproduce
**on purpose** ... Empty here means no emulation was exercised at all, which is **a
coverage hole rather than good news**", sitting directly above the block at `:171-175`
that corrects it ("Since the oracle became optimizer-off DuckDB these are **BUGS**, not
notes") and above a printed header that says "(each one is a bug)". A reader who stops at
the first comment gets ORC-25 backwards. Fold into T-1's editorial pass.

---

## 12. ASK index

Every ASK, in document order, with what it binds. All fifteen are open.

| ask | question | at | binds |
|---|---|---|---|
| **ASK-1** | pin `==1.5.5` or floor-plus-assert; and 1.5.5 or the LTS line | 1.3 | ORC-02, ORC-09, ORC-86, every pin |
| **ASK-2** | build-vs-build repeatability: sort-at-freeze, out of contract, or tentative | 3.7 | ORC-22 |
| **ASK-13** | does `threads` join the oracle constant, and what disposition covers order *inside* a value | 3.7 | ORC-02, ORC-14, ORC-22, ORC-75 |
| **ASK-3** | make the accepted severity-4 refusal cost countable, or amend the RFC to say it is unmeasured | 4.3 | ORC-30, ORC-59, D9 |
| **ASK-4** | is `OPT_EMULATED`'s `AGREE` treatment at `oracle.py:740` deliberate | 4.3 | ORC-25 |
| **ASK-5** | do reason codes stay internal, or become user-visible refusal text | 4.3 | ORC-29 |
| **ASK-12** | is the comparison harness's own normalization part of the oracle's answer | 5 | ORC-32, ORC-38, ORC-26, D7, D12 |
| **ASK-6** | bit-for-bit floats and what governs the three exceptions already in force; and the unenforced third home | 5 | ORC-32, ORC-76, ORC-80, ORC-84, D7, D8 |
| **ASK-7** | close TASK-95 or downgrade the five doc-twin totality sites | 7.2 | ORC-54 |
| **ASK-8** | adopt "an unlisted divergence is a bug by definition" | 7.4 | ORC-55 |
| **ASK-9** | admit a `tentative` (measured, not ruled) bucket | 7.4 | ORC-22, ORC-75, D13, D16 |
| **ASK-14** | is a campaign baseline evidence, or a snapshot | 7.4 | ORC-66, D4, D12, D16, ASK-10 |
| **ASK-10** | classify the width residuals before treating them as a defect count | 10 | D13, D16 |
| **ASK-11** | corpus match count: dated generated number, or a ratchet | 10 | ORC-67, ORC-73 |
| **ASK-15** | do this document's own nine `[PROPOSED]` rules become normative | 10 | ORC-15, 31, 35, 41, 56, 58, 59, 65, 84 |

Plus the section 7.3 ledger: **16 rows awaiting a ruling** (D1-D16), of which D13 and
D16 are marked *unruled*, D7 and D8 are ruled together under ASK-6, D12 is ruled by
attribution to D7, and the remainder carry a proposed status that is not in force until
you fill the column.

**What changed since the first draft, in one place.** This revision was written against
four independent audits of the first draft. Every substantive correction is marked in
place with the word *Correction* and the measurement that forced it; the ones that change
what a reader should do are: ORC-05 (four sites -> twelve, two of them under
`src/planner/binder/`, and constant folding *is* removed), ORC-38 / ASK-12 (the
`decimals` tag does not suppress the value comparison, and the normalization cast is not
value-preserving), ledger D12 (four of five "unattributed" residuals were attributed all
along), ORC-32 (three float tolerances are already in force, so ASK-6's question was
mis-posed), ORC-66 (the campaign fuzzer is not a standing gate), and the eight rules this
document had been stating as decided that nobody ever decided (ASK-15). Two ledger rows
were added from a re-sweep of `known_divergences/` (D14, D15) and one from the audit of
the baseline file itself (D16).
