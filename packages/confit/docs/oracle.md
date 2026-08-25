# The oracle spec

**What this document is.** The definition of *correct* for the confit engine: what
the oracle is, what it decides, what it declines to decide, and how to compare
against it. It is a consolidation, not a proposal — where a decision is already in
force it is written down here so the next reader stops re-deriving it.

**Governance.** The oracle spec states what is considered correct. Every
contradiction goes through the owner. This document therefore keeps three kinds of
content strictly apart:

- **Normative claims** — settled decisions in force. Each carries a stable id
  `ORC-NN` and a `Verified-by` pointer. Nothing here is new.
  A claim marked **`[PROPOSED]`** is the exception and is **not in force**: it holds
  an id only so a pin or a ticket can reference it, and it becomes normative only when
  the owner says so. Eight claims carry that marker; every other `ORC-NN` is decided.
- **ASK blocks** — questions only the owner can answer, placed at the point in the
  document where the answer would bind. An ASK is never phrased as decided, and no
  normative claim depends on one.
- **Editorial** — framing, indexes, corrections to other documents. Unnumbered.

**How to read a claim.** One claim per paragraph block, so a content-hash field can
be added later by tooling without re-cutting the text. `Verified-by` names a test
path, a pin file, a source line, or a measurement; where nothing verifies a claim it
says `Unverified` and says so plainly. Behaviors carry a status:

| status | meaning |
|---|---|
| `PINNED` | the oracle's answer is stable and it is the contract |
| `IMPL-DEFINED` | stable for the pinned build and configuration, fragile across versions or platforms; the pin names the discriminator |
| `UNSPECIFIED` | not stable; we refuse, normalize, or exclude, and never claim bit-for-bit |

**Completeness** here means *decision coverage*: every oracle fact that was decided
appears in this document. Emergent behavior owes nothing. A gap is a decision that
was made somewhere and is not written here — not a behavior nobody has ruled on.

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
*Verified-by:* `packages/confit/tests/conftest.py:7-12` (decided 2026-08-17);
`packages/confit/fuzz/oracle.py:10-16`; `packages/confit/docs/known-limitations.md:20-30`.

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
through `WINDOW_SELF_JOIN = 33`), **and additionally** changes physical-operator
selection at four sites that read `enable_optimizer` directly outside the pass list:
the DISTINCT-ON ordered-aggregate rewrite, streaming-vs-blocking window operator
selection, and sorted-aggregate `ORDER BY` simplification in two places. The binder is
genuinely untouched — all four sites run in physical planning or execution, none in
binding — so output types, constant folding and bind-time errors are unaffected, and
execution-level laziness (an untaken `CASE` arm, `AND`/`OR` short-circuit in a filter)
is unaffected. All four extra sites sit under aggregation, windows and `DISTINCT`,
which the row path refuses outright, so their reach is the static-tables-only path.
*Verified-by:* DuckDB v1.5.5 source,
`src/include/duckdb/common/enums/optimizer_type.hpp:16-49` (33 members),
`src/execution/physical_plan/plan_distinct.cpp:66`,
`src/execution/physical_plan/plan_window.cpp:31-35`,
`src/function/aggregate/sorted_aggregate_function.cpp:686` and `:744`, both reached
from `src/execution/physical_plan/plan_aggregate.cpp:318`.
*Correction:* `conftest.py:20` ("`PRAGMA disable_optimizer` == disabling all 33 named
optimizers") and `known-limitations.md:30` ("What it removes is the 33 plan-rewrite
passes") are incomplete by these four sites. The binder half of both statements is
exactly right. See proposed ticket T-1.

**ORC-06.** The user-facing contract still names what a user's DuckDB returns, which
is optimizer-*on*. The gap between the two readings is therefore user-visible and
stays a reported finding (`DIVERGE_OPT`) rather than an accepted class. The oracle
and the contract surface are deliberately not the same thing.
*Verified-by:* `packages/confit/fuzz/oracle.py:43-46`;
`packages/confit/fuzz/runner.py:28-29` (`DIVERGE_OPT` is in `INTERESTING`).

### 1.3 How the identity is enforced

**ORC-07.** The specialization half is enforced mechanically and repo-wide. An
autouse fixture monkeypatches `duckdb.connect` so every connection in the confit test
suite comes back with `PRAGMA disable_optimizer` already applied: the oracle is a
property of the repo, not a per-test choice, and a new test that reaches for DuckDB
gets the oracle by construction. Measured today: 66 `duckdb.connect` call sites across
24 files are covered by that one fixture.
*Verified-by:* `packages/confit/tests/conftest.py:62-71`; call-site count measured
2026-08-25 by grep over `packages/confit/tests/`.

**ORC-08.** A test that *wants* the optimizer says so in its own body
(`con.execute("PRAGMA enable_optimizer")`), which reads as the deliberate exception it
is. Exactly two such exceptions exist, both in the test that documents what the
optimizer does.
*Verified-by:* `packages/confit/tests/conftest.py:37-43`;
`packages/confit/tests/known_divergences/test_trap_elision.py:468` and `:566`.

**ORC-09.** The version half of the identity is enforced **nowhere**. Seven documents
plus the manifests name DuckDB 1.5.5, but `pyproject.toml:15` and
`packages/sql-transform/pyproject.toml:10` declare `duckdb>=1.5.5` — a floor —
`packages/confit/pyproject.toml` declares no duckdb dependency at all, and only
`uv.lock` resolves 1.5.5 exactly. A `uv lock --upgrade` silently re-points the oracle
and no gate notices.
*Verified-by:* measured 2026-08-25 — `pyproject.toml:15`,
`packages/sql-transform/pyproject.toml:10`, `packages/confit/pyproject.toml`
(dependencies: `pyarrow>=19.0` only), `uv.lock:368-370`.
*Status:* this is a stated fact, not a decision. The fix is ASK-1.

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

---

## 2. Inherited quirks

**ORC-10.** Where DuckDB's behavior is a quirk, the quirk is reproduced, not fixed.
Because the contract is bit-for-bit DuckDB rather than the SQL standard, DuckDB's
oddities *are* our normative behavior, including ones DuckDB would call bugs.
*Verified-by:* `packages/confit/docs/reports/pins-first-methodology.md:39` ("pins are
engine==oracle contracts").

**ORC-11.** The enumerated quirks. This list exists so a future reader who meets an
inherited oddity has something to check it against before "fixing" it. It is short
today and cheap to write; reconstructing it later is not.

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
*Verified-by:* the governance rule in this document's front matter;
`known-limitations.md:284-285` ("If a message you hit isn't in this document or the
tests, that's a bug in our bookkeeping — file it").

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

**ORC-14.** Four kinds of nondeterminism, four dispositions, all four already in
force. The last column is the operating instruction: what a *new* case of that kind
gets, without re-litigating from the axiom.

| what is nondeterministic | disposition | in force as | a NEW case gets |
|---|---|---|---|
| **which rows exist** | REFUSE by name at build | TASK-128 row limits (ORC-17) | a named build-time refusal, `ORDER BY` or not |
| **row order** | declare unspecified, compare per mode | TASK-129 compare modes (ORC-18) | a compare mode plus a DuckDB-free self-leg, never byte-equality |
| **values, from data outside the query** | excluded from the oracle | optimizer-off structurally (ORC-03); residuals excluded by source name (ORC-20) | exclusion by name, carrying a measured reason |
| **the oracle disagrees with itself** | reject the construct by name | regex anchor families (ORC-21) | a named refusal — there is no behavior to match |

*Verified-by:* each row's own claim below.

**ORC-15.** Every comparison target carries one status from the vocabulary in the
front matter (`PINNED` / `IMPL-DEFINED` / `UNSPECIFIED`). A target with no status is
not yet a target. The divergence ledger in section 7 carries the status column for
every currently-tolerated divergence.
*Verified-by:* the vocabulary is defined in this document; its application is the
`proposed status` column of the section 7 table (which awaits ruling).

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
each exclusion citing a measured reason it is irreproducible row-locally. The measured
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

### 3.7 The one hole that is not ruled

**ORC-22.** Build-vs-build repeatability of a frozen artifact is **explicitly
undecided**, and recorded as such rather than as a guarantee. Two builds of the same
function disagreeing on frozen row order would flake downstream golden tests. Raw
DuckDB gave 12 distinct row orders over 12 fresh connections on an unordered 200-group
`GROUP BY`; our arrow materialization measured stable over 6 builds, which the ticket
records honestly as "luck, not contract". `sort-at-freeze` is the named artifact-level
fix and is orthogonal to the oracle definition.
*Verified-by:* `backlog/tasks/task-129 ...md` (the SEPARATE-but-adjacent paragraph and
AC #3's probe carried over from `task-128`).
*Status:* undecided. See ASK-2.

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

**ORC-23.** One case in, one verdict out, from a closed set of ten kinds. Every
outcome — refusal, trap, disagreement, and the oracle's own failure — comes back *as*
a verdict rather than as an exception, so nothing is classified by a human reading a
stack trace.

| kind | meaning |
|---|---|
| `AGREE` | ours == off == on |
| `AGREE_TRAP` | both sides error at run time |
| `DIVERGE_VALUE` | wrong value, wrong schema, or a self-leg failure |
| `DIVERGE_BUILD` | confit builds what DuckDB refuses |
| `DIVERGE_TRAP` | one side traps where the other serves rows |
| `DIVERGE_OPT` | we match the optimizer-off baseline; an optimizer pass changes what the user sees |
| `OPT_EMULATED` | we match optimizer-ON against a baseline that disagrees: a plan-rewrite pass we are reproducing, which is a bug |
| `BUILD_EXC` | a build raised something other than the contract's `ValueError` |
| `REFUSED` | confit refused at build |
| `SKIP` | the oracle harness itself raised |

*Verified-by:* `packages/confit/fuzz/oracle.py:65-80` (`KINDS`), `:538-745`
(`run_case`), `:884-891` (`run_case_json`).

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
suite hides real bugs behind a green bar.
*Verified-by:* `packages/confit/fuzz/oracle.py:884-891` (an exception escaping
`run_case` becomes `SKIP`, blaming the oracle rather than the engine);
`packages/confit/fuzz/runner.py:35` (`SKIP` is in `INTERESTING`).

**ORC-27.** A comparison the checker cannot evaluate falls back to the weaker check
**with a logged tag**, never silently. The one instance today: an `ORDER BY` over an
expression that is not an output column cannot have its key evaluated, so the multiset
check stands and the case carries an `order-by-unevaluated` tag.
*Verified-by:* `packages/confit/fuzz/oracle.py:685-690`;
`backlog/tasks/task-129 ...md` AC #4.

**ORC-28.** `AGREE` is the only kind counted as coverage. A construct-coverage
histogram runs over agreeing cases only, so a grammar hole is visible rather than
absorbed by refusals.
*Verified-by:* `packages/confit/fuzz/runner.py:38-41`, `:145-157`.

### 4.3 Refusals

**ORC-29.** A build-time refusal is the engine's second legal outcome and is always a
named `ValueError` at `DuckDBInferFn(...)` construction, classified by its message
prefix: `unsupported:` (real SQL, deliberately not served), `parse error:` (the
dialect surface ends here), `bind error:` (the query is wrong against your schema).
Refusal is cheap, named and testable by construction.
*Verified-by:* `packages/confit/docs/known-limitations.md:274-285`; P7 and P18 in
`packages/confit/docs/properties.md:63, :231-235`.

**ORC-30.** Current behavior of the campaign's refusal path, stated as fact: both
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

**ORC-31.** The general rule this exposes, and the one worth writing down: **an
accepted cost must be countable, and the counting mechanism is named in the decision
that accepts it.** Without that, "deliberate strictness" and "unnoticed over-refusal"
are the same observation.
*Verified-by:* derived; the instance is ASK-3 below.

> ### ASK-3 — the accepted severity-4 cost is currently uncountable. Which way?
>
> You accepted the bind-time constant refusals twice (2026-08-24, re-affirmed
> 2026-08-25 on corrected facts). The RFC justifies the accepted cost three times by
> asserting the campaign will measure it:
>
> - `rfcs/2026-08-19-keep-the-bind-time-refusals.md:96-97` — "Under the fuzzer's
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
> survivor of the pre-2026-08-17 doctrine, when `OPT_EMULATED` meant expected? The
> stale comments elsewhere are editorial; this one changes what runs.
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

**ORC-32.** Floats compare by **bit pattern**. The pin records the bits, not a
rendering. No tolerance, no rounding, no `%.3f`.
*Verified-by:* `packages/confit/tests/test_duckdb_wave3_mathtail.py:205-231`
(explicit bit pinning, since `repr` collapses every NaN to `nan`); float bit patterns
are a recorded field across the pins corpus.

**ORC-33.** `-0.0` is distinguished from `+0.0`, and this is **fixed, not tolerated**.
Unary minus was lowered as `0 - x`; IEEE `0.0 - 0.0` is `+0.0`, so the sign vanished
everywhere it could arise — 113 of 963 findings in the 2026-08-11 campaign. The fix
subtracts from `-0.0` for FLOAT operands (exact IEEE negation for every double) and
keeps `0 - x` with its `i64::MIN` trap on the integer path, matching DuckDB. It is now
a passing regression pin over both backends.
*Verified-by:* `packages/confit/tests/known_divergences/test_literal_typing.py:133-165`;
`backlog/tasks/task-80 ...md:46` (the 113/963 measurement), `:75` (the class measured
empty after).

**ORC-34.** `%`-by-zero produces a NaN whose **sign bit is platform-libm** (`7ff8...`
on Windows ucrt, `fff8...` on Linux glibc), so the pin is *engine == oracle bit
agreement per platform*, not a constant. `fmod`'s NaN, by contrast, comes from hardware
arithmetic and is `fff8...` on every x86 platform, so it is pinned as a constant.
Status: `IMPL-DEFINED`, platform is the discriminator.
*Verified-by:* `packages/confit/tests/test_duckdb_wave3_mathtail.py:205-231`;
`packages/confit/docs/specs/pins-wave3/math_tail.json` (the wave-3 correction);
`packages/confit/docs/known-limitations.md:258-259`.

**ORC-35.** The multi-answer rule. A set of accepted answers is legitimate **only when
every member is acceptable in every context**, and every set-valued pin names the
predicate that selects a member (platform, profile, oracle version). Shortest-diff or
best-match selection is never legitimate: it makes the suite pick the answer that
happens to be closest, which is indistinguishable from picking the answer that hides
the bug. ORC-34 is the model instance — platform is a real discriminator, evaluable
before the comparison runs.
*Verified-by:* ORC-34 is the only set-valued pin in force today (measured 2026-08-25
over `packages/confit/docs/specs/pins-*/`); the rule itself is normative here.

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
type mismatch is a `DIVERGE_VALUE` in its own right; the only exceptions are the
enumerated unshipped-feature width classes, which are *tagged* and cast before the
value comparison so the tag stays visible in the report rather than becoming an
accepted equality.
*Verified-by:* `packages/confit/fuzz/oracle.py:491-529` (`_schema_delta`,
`_type_delta`), `:700-710`.

**ORC-39.** Error **texts** are not compared. Runtime traps reproduce DuckDB's message
bodies verbatim; some bind-time rejections use our own wording with the same error
class. The corpus compares successful results only, so texts never affect parity. Error
text is therefore *not oracle-decided output* — which is also a named blind spot
(ORC-57). Upstream does the same thing: DuckDB's own test infrastructure matches error
text by substring containment.
*Verified-by:* `packages/confit/docs/known-limitations.md:219-224`;
`packages/confit/tests/test_corpus_replay.py:173-176` (only successful rows compared).

**ORC-40.** Backend agreement is settled **before** either DuckDB reading: cranelift
vs interpreter is a question about us, not about the oracle, and a split there is
reported as its own class rather than being attributed to a divergence.
*Verified-by:* `packages/confit/fuzz/oracle.py:625-639`; P19 in
`packages/confit/docs/properties.md:240-245`.

**ORC-41.** Tolerances we refuse, and why the refusal is itself a decision:

| rejected mechanism | where it comes from | why we refuse it |
|---|---|---|
| float rendering at `%.3f` | sqllogictest's cross-engine rendering contract | directly destroys bit-for-bit (ORC-32); it trades float fidelity for cross-engine agreement, and we have exactly one engine to agree with |
| MD5 hashing result streams above a threshold | sqllogictest | hashes make a failure undebuggable; pins-as-data (reprs, float bits, verbatim error heads, the exact SQL) is strictly better evidence, and DuckDB's own docs advise using the hash form sparingly |
| shortest-diff variant matching | Postgres's `resultmap` driver, which admits it "cannot tell which variant is actually correct" | picks the closest answer, which is the answer most likely to hide the defect; ORC-35 is the replacement |
| cross-engine agreement or majority vote as the oracle | sqllogictest, Csmith | creates a second authority; the contract delegates to exactly one engine on purpose (ORC-01). Whether Postgres agrees is irrelevant |
| a growing expected-errors allowlist | SQLancer's `ExpectedErrors` | every entry added to silence a false positive is a place a real bug can hide; structurally the same shape as ORC-30 |

*Verified-by:* ORC-32 for the first row; the rest are normative rejections recorded
here so they are decisions rather than omissions. Status of the whole table:
in force by construction today, ratified by ASK-6.

> ### ASK-6 — is bit-for-bit float equality the contract with no exceptions?
>
> Every cross-engine suite surveyed quietly relaxes floats. If the answer is "bit
> pattern, no exceptions", that is a stronger claim than any prior art and should be
> stated as such — and it means any future parallel float accumulation is an
> `UNSPECIFIED` region we must **refuse** rather than tolerate.
>
> The counter-pressure is ledger rows D7/D8. DECIMAL literals compute as f64
> (`1.5` is `DECIMAL(2,1)` in DuckDB), so `CAST(-2.5 AS BIGINT)` is `-3` there (half
> away from zero) and `-2` here (half to even). It is the **only tolerated value-family
> divergence in the ledger**, and it has no strict-xfail twin — the fuzzer's `decimals`
> tag suppresses it and the code says in-file that when decimal arithmetic lands and
> the tag must be deleted, nothing will ring.
>
> Two readings, and they lead different places:
>
> - **scope carve-out** — exact decimal arithmetic is out of contract until the lattice
>   phase lands; D7/D8 are `permanent` until then and the ledger says so.
> - **defect on a clock** — D7/D8 are a `until-fixed` row owing a strict-xfail pin
>   today, so the tag's deletion is forced by a failing test rather than by memory.
>
> *Verified-by (the facts, not the ruling):*
> `packages/confit/docs/known-limitations.md:166-174`;
> `packages/confit/fuzz/oracle.py:113-131` (the tag and its own statement that it has
> no twin), `:512-513`.
>
> *Binds:* ORC-32, ledger rows D7 and D8, and every future float-accumulation feature.

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

### 7.2 Doc-twin totality is partial

**ORC-54.** Four documents assert that every limitation has an executable twin. The
code denies it in-file, TASK-95 is open with both acceptance criteria unchecked, and
the measured shape is: `test_known_limitations.py` holds 14 test functions (two
parameterized) against roughly three dozen enumerated limitations, with several
sections' twins living elsewhere (`test_arrow_schema_api.py` for the row-limit rule,
`known_divergences/` for D2-D6) rather than in the named twin file.
*Verified-by:* the claims at `packages/confit/docs/known-limitations.md:5-7` and
`:294-296`, `packages/confit/docs/reports/pins-first-methodology.md:66`,
`packages/confit/docs/kpis.md:58-61`; the denial at
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
> - **downgrade the claim** in all four documents to "every limitation with a twin is
>   asserted; the twins are enumerated here", and drop the totality word.
>
> Recommendation, if you want the cheap one: downgrade now, keep TASK-95 open. The
> claim costs nothing to correct and currently buys false confidence.
>
> *Binds:* ORC-54, and the four documents named in its Verified-by.

### 7.3 The ledger

**ORC-55.** The table below is the complete enumeration of currently-tolerated
engine-vs-oracle divergences. Each row is a decision, so each row carries a proposed
status from the section 3 vocabulary plus a permanence, and each row awaits an owner
ruling. **This table is the adjudication surface: `accept` writes the proposed status
into force, `reject` sends the row back as a defect owing a ticket.** Nothing in the
`proposed status` column is in force until the ruling column is filled.

*Severity* is the ladder of section 8: 1 = trap where DuckDB serves, 2 = wrong value,
3 = serve where DuckDB refuses, 4 = refuse where DuckDB serves.

| id | behavior | measured evidence | sev | proposed status | OWNER RULING |
|---|---|---|---|---|---|
| **D1** | duplicate output names renamed by DuckDB's own boundary algorithm; applied to *both* sides before comparison | `pins-wave5/dup-names-client-contract.json`; `oracle.py:467-488, :646-651`; twin `test_known_limitations.py:255` | n/a (contract) | `PINNED` / permanent | |
| **D2** | error texts approximate where noted; the corpus compares successes only | `known-limitations.md:219-224`; `test_corpus_replay.py:173-176` | n/a | `UNSPECIFIED` / permanent | |
| **D3** | `ILIKE` with embedded NUL is statistics-dependent; engine is NUL-transparent; source excluded by name | `test_corpus_replay.py:40-49`; `pins-wave1/pins_like.json` | n/a | `UNSPECIFIED` / permanent | |
| **D4** | a trapping subexpression the optimizer deletes, we still evaluate — the standing price of optimizer-off | `known-limitations.md:231-257`; `known_divergences/test_trap_elision.py`; committed campaign baseline `packages/confit/findings.jsonl` | 1 (direction) | `PINNED` / permanent | |
| **D5** | `%`-by-zero NaN sign bit is platform-libm; the pin is per-platform bit agreement, not a constant | `test_duckdb_wave3_mathtail.py:205-231`; `pins-wave3/math_tail.json` | n/a | `IMPL-DEFINED` / permanent, discriminator = platform | |
| **D6** | schema qualifiers are registry-noise: `s1.t1` resolves on the bare table name; DuckDB's schema-existence errors are not reproduced; `w.w.w` binds the longer schema-ish parse | `known-limitations.md:260-272` | 3 and 4 (always a loud build-time rejection, never a different served value) | `PINNED` / permanent | |
| **D7** | DECIMAL **literals** are f64; exact-decimal accumulation is not reproduced | `known-limitations.md:166-174`; `oracle.py:113-131, :512-513` | 2 | *unruled* — see ASK-6 | |
| **D8** | `CAST(-2.5 AS BIGINT)` on a bare literal: DuckDB rounds half away from zero (`-3`), we round half to even (`-2`) | `known-limitations.md:169-174`; `known_divergences/test_cast_semantics.py` | 2 | *unruled* — see ASK-6 | |
| **D9** | bind-time constant refusals: `WHERE FALSE` and empty-input shapes refuse here and serve on DuckDB | RFC `2026-08-19-keep-the-bind-time-refusals.md` (ACCEPTED twice); `known_divergences/test_literal_typing.py:77-131` | 4 | `PINNED` / permanent — **but its cost is uncountable, see ASK-3** | |
| **D10** | one-sided regex program-size guard: always fires before DuckDB's real RE2 budget, so it may over-refuse and can never serve where DuckDB errors | `pins-waveB/fuzzer-20260728.json`; `pins-first-methodology.md:79` ("the asymmetry is the contract") | 4 by construction | `PINNED` / permanent | |
| **D11** | narrow-lane overflow trap threshold not yet shipped: an overflowing narrow lane serves the i64 value on the row path and refuses by name at the `infer_arrow` boundary | `known-limitations.md:177-188`; catalogue pinned in `test_integer_widths.py` | 3 (row path) | `PINNED` / **until-fixed** (m-8 phase 3) — owes a strict-xfail twin if until-fixed is accepted | |
| **D12** | unattributed campaign residuals in the committed 2026-08-17 baseline: 4 x `DIVERGE_VALUE` (seeds 869, 998, 1554, 3269) and 1 x `DIVERGE_TRAP` (seed 2668, `-2147483648 / -1` where we serve and both readings error) | `packages/confit/findings.jsonl`, counted 2026-08-25: 28 findings = 16 `DIVERGE_BUILD` + 7 `DIVERGE_OPT` + 4 `DIVERGE_VALUE` + 1 `DIVERGE_TRAP` | 2 and 3 | *unruled* — measured, no decision, no ticket. See ASK-9 | |
| **D13** | the phase-2 width residuals, quoted from memory as 79 of 84, treated as a defect count | **Unverified** — the numbers appear nowhere in the tree (searched `packages/confit/docs/`, `backlog/`, `findings.jsonl`, 2026-08-25) | unknown | *unruled* — see ASK-10 | |

*Verified-by:* each row cites its own evidence; the enumeration is complete against
`packages/confit/docs/known-limitations.md` section 5 plus section 3's value-family
rows, `packages/confit/tests/known_divergences/`, and the committed campaign baseline,
all as of master `85b4739`.

**Closed, deliberately not a row:** the 16-seed `DIVERGE_BUILD` ambiguous-reference
class (the largest single class the 2026-08-17 campaign saw, 57% of findings) is
**TASK-121, status Done**. `packages/confit/docs/2026-08-17-fuzz-triage.md:56-58` and
`:70-72` still say it is "not yet ticketed" and "has no ticket". See proposed ticket
T-10.

### 7.4 Placement

**ORC-56.** Where confit deliberately does not match DuckDB, the note belongs **at the
requirement it violates** in the engine spec, with this ledger as the index. A
divergence filed only in an appendix stops being read. D9 (bind-time refusals) and D10
(one-sided guards) are the two rows this applies to today.
*Verified-by:* normative here; the anti-pattern is the reason ORC-53's census found
readers walking past entries.

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
> KEEP and CHANGE both presuppose a ruling (ORC-50). Three things are measured facts
> awaiting your call and today live only in prose: build-vs-build repeatability
> (ORC-22), the unattributed campaign residuals (ledger D12), and the width residuals
> (ledger D13). A `tentative` **tag** — not a third directory — would give them a home
> without promoting them to contract.
>
> The stricter alternative is that everything measured gets ruled at measurement time.
> That is slower, and it is a real option: it means a campaign cannot end until its
> residuals are classified.
>
> *Binds:* ORC-22, ledger rows D12 and D13.

---

## 8. The severity ladder

**ORC-57.** One definition, cited everywhere, restated nowhere:

| rung | meaning | direction |
|---|---|---|
| **1** | we trap where DuckDB serves | contract violation |
| **2** | we serve a wrong value | contract violation |
| **3** | we serve where DuckDB refuses | directional — the query cannot be run against DuckDB at all |
| **4** | we refuse where DuckDB serves | directional — the safe side, sometimes chosen on purpose |

*Verified-by:* the two existing parenthetical definitions, which agree:
`packages/confit/docs/specs/2026-08-25-task-114-design.md:140-142` and
`packages/confit/docs/specs/2026-08-25-task-127-remainders-design.md:154-156`; used by
name in the Rust source at `packages/confit/src/specializer/frontend.rs:64-73`
("refusing is the severity ladder's own preference") and by both RFCs.
*Note:* two definitions that agree today is one definition away from drift. Proposed
ticket T-11 replaces the parentheses with a citation of this claim.

**ORC-58.** The ladder is also the scope tiering. Rungs 1 and 2 are contract
violations and are never accepted. Rungs 3 and 4 are directional: rung 4 is the safe
direction, and choosing it deliberately is legitimate — D10's one-sided guard states it
outright, "it may over-refuse; it can never serve where DuckDB errors. The asymmetry is
the contract."
*Verified-by:* `packages/confit/docs/reports/pins-first-methodology.md:79`; ledger rows
D9 and D10.

**ORC-59.** A deliberate rung-4 choice must be countable (ORC-31). Rung 4's whole
defense is that it is the safe direction; that defense is only inspectable if the class
size is observable. This is exactly what ASK-3 is about.
*Verified-by:* derived from ORC-31 and ORC-30.

---

## 9. Version bumps and mutability

The urgency is external: DuckDB ships minor versions roughly every four months,
semantics have already moved inside a patch release, and v2.0 brings a new SQL parser.
This protocol is cheap to write now and expensive to write during a migration.

**ORC-60.** The pinned oracle version is **one named constant** (ORC-02) and appears in
every pin file (ORC-46). Everything below depends on both being true; today the first
is unenforced (ORC-09) and the second is partial.
*Verified-by:* ORC-02, ORC-09, ORC-46.

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

*Verified-by:* Unverified — the classification is proposed here for ratification.

**ORC-64.** **[PROPOSED]** Not in force. A pin whose value changed across a bump is itself
a decision and earns a line in this document, not just a new JSON blob. A changed pin
that leaves no trace here makes the next reader unable to tell a fix from a regression.
*Verified-by:* Unverified — proposed with ORC-61.

---

## 10. Campaign validity and blind spots

**ORC-65.** A campaign residual that lands in an `UNSPECIFIED` region is **not a parity
defect**. Classify before counting: a number that mixes determined-and-wrong with
under-determined is not a defect count and must not be treated as a backlog.
*Verified-by:* normative here; the rule is the direct consequence of ORC-13 and
ORC-15. Immediate application: ledger rows D12 and D13, and ASK-10.

**ORC-66.** Two standing differential gates run in the normal test gate: the regexp
fuzzer at N=250 with a fixed seed (`REGEXP_FUZZ_SEED` / `REGEXP_FUZZ_N` for deep runs),
and the seed-addressed campaign fuzzer. The regexp gate's first deep run found 122
divergences distilled to 12 reject classes, then re-swept to **zero divergences over
40k cases across 8 seeds**.
*Verified-by:* `packages/confit/tests/test_duckdb_regexp_fuzz.py`;
`packages/confit/fuzz/`; `packages/confit/docs/specs/pins-waveB/fuzzer-task54.json`;
`packages/confit/docs/reports/pins-first-methodology.md:70-74`.

**ORC-67.** The corpus replay gates **zero FAILs, always**. Three outcomes — match,
clean-unsupported, FAIL — where a rejection is only clean if it is one of the
*documented* rejection classes, so an undocumented error is a FAIL like any wrong
answer. The match count is deliberately **ungated**: it is the growth ladder
(53 -> 395 -> 505 -> 511 -> 529 -> 546 -> 550 of 678), and every construct learned flips
cases from clean-unsupported to match, never into FAIL.
*Verified-by:* `packages/confit/tests/test_corpus_replay.py:17-20, :179-198`;
`packages/confit/docs/reports/pins-first-methodology.md:41-62`.
*Correction:* the number 550/678 is quoted without its "as of stage B" hedge in the
methodology abstract and in two reports, and tasks have landed since that flip cases.
See ASK-11 and proposed ticket T-13.

**ORC-68.** Our full-result diff is **stronger** than any partial oracle wherever it is
total. Wherever it is partial we have re-created a partial oracle and we own its blind
spot. The blind spots, named:

| blind spot | what the oracle does not pin there | mitigation in force |
|---|---|---|
| row order outside the row path and outside a total `ORDER BY` | the sequence | multiset plus self-legs (ORC-16, ORC-18) |
| `order-by-unevaluated` fallback | sortedness on a non-output-column key | logged tag, not silent (ORC-27) |
| error texts (D2) | message bodies for bind-time rejections | error *class* is compared; texts are not oracle-decided output (ORC-39) |
| the excluded ILIKE-NUL source (D3) | statistics-dependent kernel selection | exclusion by name with a measured reason (ORC-20) |
| refusals | whether the oracle would have served (ORC-30) | **none today** — this is ASK-3 |

*Verified-by:* each row's cited claim.

**ORC-69.** Metamorphic self-legs are the oracle *substitute* in the abstention region,
and the set is capped rather than grown: batch-vs-single sequence equality, input
reversal reversing the output blocks, hostile-arrow invariance (sliced, chunked,
empty), `infer_rows` vs `infer_arrow` agreement, cranelift vs interpreter agreement,
and sklearn as a second ground truth on plain tree cases. They involve no DuckDB, which
is what makes them usable exactly where the oracle abstains.
*Verified-by:* `packages/confit/fuzz/oracle.py:748-858` (`_extra_legs`), `:625-639`;
`packages/confit/tests/test_fuzz_order_legs.py`; P1-P20 in
`packages/confit/docs/properties.md`.

**ORC-70.** **[PROPOSED]** Not in force. A campaign declares a coverage signal, because
"20k queries, N residuals" is unanchored without a denominator that means something. We have no query
plans to diversify over, so the analogue is distinct **(operator, argument-type,
edge-class)** triples reached per campaign — which extends the existing `AGREE`-only
construct histogram's axis rather than replacing it. That same triple is the right unit
for decision coverage: one operator x one type x one edge class, not one feature.
*Verified-by:* the histogram exists at `packages/confit/fuzz/runner.py:38-41, :145-150`;
the triple axis does not. Proposed ticket T-14.

**ORC-71.** **[PROPOSED]** Not in force. A campaign reports **abstentions per kind**
alongside the coverage histogram, and a rising abstention rate is read as generator
drift rather than as good news. A generator that has drifted out of the answerable
region measures nothing while still printing a green bar. `oracle-errored` must stay a
separate code from the rest.
*Verified-by:* `SKIP` and `order-by-unevaluated` exist (ORC-26, ORC-27); the rate does
not. Proposed ticket T-15, and see ASK-5 for the user-visibility boundary.

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
> *Binds:* ledger row D13.

> ### ASK-11 — does the corpus match count become a ratchet?
>
> Zero FAILs is the gate and should stay the gate (ORC-67). The match count is
> deliberately ungated as a growth ladder, and it has now gone stale in three documents
> because nothing watches it. Two options:
>
> - **leave ungated**, and generate the number with a date stamp in exactly one place
>   so drift cannot recur silently;
> - **add a never-decreases ratchet.** Real cost: any deliberate scope reduction
>   becomes a gate failure, and a scope reduction is sometimes the right call.
>
> *Binds:* ORC-67.

---

## 11. Proposed tickets

Everything this document wants changed in code or in another document. **None of it is
applied here** — this deliverable is one markdown file.

| id | change | claim | blocked on |
|---|---|---|---|
| **T-1** | correct `conftest.py:20` and `known-limitations.md:30`: `disable_optimizer` also changes physical-operator selection at four sites; the binder half stands | ORC-05 | editorial, no ruling needed |
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
| **T-12** | generalize `pin_ast_shapes.py`'s drift-report pattern to the pins corpus | ORC-61 | ASK-1(b) |
| **T-13** | generate the corpus match count with a date stamp in exactly one place | ORC-67 | ASK-11 |
| **T-14** | extend the coverage histogram to (operator, arg-type, edge-class) triples | ORC-70 | owner's go |
| **T-15** | report abstentions per kind per campaign | ORC-71 | ASK-5 |

Two further editorial corrections found while writing, no ticket needed if fixed in
place: `known-limitations.md:249-250` says a 4000-seed campaign puts trap elision at
"8 seeds in 28 findings; all eight are labelled `DIVERGE_OPT`" — the committed baseline
holds **7** `DIVERGE_OPT` seeds (312, 812, 1196, 1563, 1564, 2174, 2805) of 28 findings
(counted 2026-08-25 over `packages/confit/findings.jsonl`), and
`2026-08-17-fuzz-triage.md:102` names only five of them in prose, while its table at
`:87` says `OPT_EMULATED = 0` and its prose at `:89-94` says one remains.

---

## 12. ASK index

Every ASK, in document order, with what it binds. All eleven are open.

| ask | question | at | binds |
|---|---|---|---|
| **ASK-1** | pin `==1.5.5` or floor-plus-assert; and 1.5.5 or the LTS line | 1.3 | ORC-02, ORC-09, every pin |
| **ASK-2** | build-vs-build repeatability: sort-at-freeze, out of contract, or tentative | 3.7 | ORC-22 |
| **ASK-3** | make the accepted severity-4 refusal cost countable, or amend the RFC to say it is unmeasured | 4.3 | ORC-30, ORC-59, D9 |
| **ASK-4** | is `OPT_EMULATED`'s `AGREE` treatment at `oracle.py:740` deliberate | 4.3 | ORC-25 |
| **ASK-5** | do reason codes stay internal, or become user-visible refusal text | 4.3 | ORC-29 |
| **ASK-6** | bit-for-bit floats with no exceptions; is D7/D8 a carve-out or a defect on a clock | 5 | ORC-32, D7, D8 |
| **ASK-7** | close TASK-95 or downgrade the four doc-twin totality claims | 7.2 | ORC-54 |
| **ASK-8** | adopt "an unlisted divergence is a bug by definition" | 7.4 | ORC-55 |
| **ASK-9** | admit a `tentative` (measured, not ruled) bucket | 7.4 | ORC-22, D12, D13 |
| **ASK-10** | classify the width residuals before treating them as a defect count | 10 | D13 |
| **ASK-11** | corpus match count: dated generated number, or a ratchet | 10 | ORC-67 |

Plus the section 7.3 ledger: **13 rows awaiting a ruling** (D1-D13), of which D7, D8,
D12 and D13 are marked *unruled* and the other nine carry a proposed status that is not
in force until you fill the column.
