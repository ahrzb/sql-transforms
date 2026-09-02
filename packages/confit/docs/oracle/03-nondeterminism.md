## 3. Nondeterminism

This is the section that converts future arguments into lookups. Read 3.1 and 3.2
before ruling on anything new.

### 3.1 The axiom

**ORC-13.** *An answer that is not a function of the query — plus the frozen statics —
is not a target.* This is the single axiom under every nondeterminism ruling in the
project. It is why the oracle is optimizer-off (ORC-03), why row limits on the
constant path refuse (ORC-17), why row order is compared per mode rather than by
byte-equality (ORC-18), and why statistics-dependent behavior is excluded (ORC-20).
*Verified-by:* two existing phrasings of the same rule outside the enforcement modules —
`packages/confit/docs/known-limitations.md:39` and `backlog/tasks/task-128 ...md`'s
description ("The doctrine already exists"). `confit/oracle.py`'s module docstring phrases
it a third time ("the optimizer-on reading is NOT matchable in principle"); that is the
module mirroring this claim, not evidence for it, and is named here only so the next
editor keeps the three in step.
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
target with no status is not yet a target. Today the vocabulary is applied to **seven**
claims and to the section 7 ledger's `proposed status` column; ORC-16, ORC-32, ORC-33,
ORC-37, ORC-38, ORC-39, ORC-90, ORC-92, ORC-93 and every ORC-11 quirk carry no status, so
under this rule as written they are not targets — which they plainly are. Adopting the
rule means either statusing them or narrowing the rule to the ledger. (The seven, counted
2026-09-02: ORC-17, ORC-18, ORC-19, ORC-20, ORC-21, ORC-34, ORC-76. Every other
`PINNED` / `IMPL-DEFINED` / `UNSPECIFIED` outside the section 7 ledger is a reference to
the vocabulary, not an application of it.)
*Verified-by:* Unverified — no rule outside this document requires a status. Part of
ASK-15.

### 3.3 Which rows exist

**ORC-16.** Row order on the *serving* path is not nondeterministic and is part of the
contract: output rows follow input rows — `map` exactly (`out[i] <-> in[i]`), `filter`
as a subsequence, `many` as per-input-row blocks in input order. That order comes from
the serving contract, not from SQL, and is checked by DuckDB-free self-legs.
*Enforced-by:* `fuzz.oracle._extra_legs` (the batch-vs-single and reversal legs, both
compared with `confit.compare.sequence`).
*Verified-by:* `packages/confit/tests/test_fuzz_order_legs.py` (the capability test that
proves the legs catch a scrambler); `packages/confit/docs/known-limitations.md:41-51`.

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
*Enforced-by:* `fuzz.oracle.compare_mode` (which mode a case is owed) and
`fuzz.oracle._sorted_by` (DuckDB defaults: ASC, NULLS LAST, NaN above every number);
`confit.compare.assert_rows`'s `ordered` flag is the same axis for a test.
*Verified-by:* `packages/confit/tests/test_compare.py::test_assert_rows_default_accepts_reordered_rows`
and `::test_assert_rows_ordered_rejects_reordered_rows`;
`backlog/tasks/task-129 ...md` (Done, AC #1-#7);
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
*Verified-by:* `packages/confit/tests/test_corpus_replay.py:38-49` (a), `:103-110` (b —
the measured comment and the `is_float32` check it guards), `:60-65` (c).
Sets (b) and (c) were absent from an earlier version of this document; (c)
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
one build, where neither answer can be preferred. Where one *can* be — because the
engine follows one of them — the identity names it instead of refusing: that is ORC-93.
The oracle disagreeing across its own **builds** is a third species with a third,
already-decided disposition — ORC-76.

**ORC-93.** The oracle's tables are **native tables**, never registered arrow relations,
and the difference is semantic rather than a convenience. DuckDB pushes constant filters
into a registered-arrow scan with IEEE NaN semantics, which disagrees with its own
native-table comparison order; the engine follows the native-table reading, so a bare
`register` is a *different* oracle wearing the same name. Column widths survive the
materialization; `NOT NULL` does not, which is why a fixture that needs a constraint
declares it in SQL instead.
*Enforced-by:* `confit.oracle.Oracle.load` (register, CTAS, unregister) and
`confit.oracle.Oracle.table` (the SQL declaration is kept verbatim).
*Verified-by:* the **materialization** —
`packages/confit/tests/test_oracle.py::test_load_materializes_a_native_table_with_widths_intact`,
`::test_load_unregisters_its_alias`, `::test_table_keeps_the_declaration_including_not_null`
(catalogue shape, TINYINT survival, the `NOT NULL` flag). The **semantic ground** is
recorded at `packages/confit/docs/specs/2026-07-26-stretch4-builtin-pins.md:114-117`
("duckdb-python pushes constant filters into REGISTERED-ARROW scans with IEEE NaN
semantics, disagreeing with its own native-table order ... duck_check now materializes
native tables") — the stretch4 file, not wave1 or wave3. No test in
`packages/confit/tests/` runs that NaN comparison against a registered relation, so the
semantic half is pinned by the recorded measurement rather than by a gate.

**ORC-76.** Where DuckDB's own wheels disagree with each other, the disposition is a
**bounded, named tolerance**, not refusal. The measured instance: `cbrt`. The Windows
wheel matches Rust/ucrt bit-exactly while the Linux wheel's bundled `std::cbrt` is one
ulp off (`cbrt(27)` = `3.0000000000000004`), CI-discovered 2026-07-26. The oracle is
platform-inconsistent there, so repr-exact parity is unpinnable; oracle parity for
`cbrt` is pinned at **<= 1 ulp**, and the wave-1 pins spec records it as "the only such
exception". The engine itself stays deterministic (Rust `cbrt`). Status:
`IMPL-DEFINED`, discriminator = the oracle's own build.
*Verified-by:* `packages/confit/tests/test_duckdb_interpreter.py:913-946`
(`duck_check_ulp`, `max_ulp=1` by default), used by `test_sqrt_cbrt_bigint` (`:949`) and
`test_cbrt_total_function` (`:958`) — both in the normal test gate;
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
decision; `confit.oracle.Oracle.__init__` sets no `threads`.
*Status:* stated, not ruled. See ASK-13.

> ### ASK-13 — does `threads` join the oracle constant, and what disposition covers
> order *inside* a value?
>
> **(a) The constant.** ORC-02 says "all other settings default" and DuckDB's `threads`
> default is core-count-derived, so a 4-core runner and a 12-core dev box are different
> oracles for thread-sensitive aggregates. Either the constant names `threads` (the
> obvious value is `1`, which the fit side already runs — ORC-72) or the document says
> hardware-derived defaults are part of the identity and those constructs are
> `IMPL-DEFINED` with the machine as discriminator. The landing spot is one line in
> `Oracle.__init__`, beside the pragma: `self.con.execute("SET threads = 1")`.
>
> **(b) The disposition.** ORC-14's table has four keys that can all fire on one case and
> no tiebreaker, and no row at all for nondeterminism *inside* a value. `string_agg`
> element order fits all four and is seen by none: the assigned mode
> (`constant-ordered`) cannot see it because the multiset differs in the *values*, and a
> frozen build-time artifact has no corpus source file to exclude by name. Ruling (b)
> means an ordering rule for the keys, a fifth row for intra-value order, or both.
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
