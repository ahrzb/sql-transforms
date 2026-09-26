# Goal yardsticks, reading N=3 (2026-09-26)

**What this is.** The third reading of the yardsticks in
[success measures](../specs/success-measures.md), taken on master `bc167f4` on
2026-09-26, the same day as [reading N=2](2026-09-26-goal-reading.md) (master `24aafe8`).
It does not edit N=2: it reads the same measures, with the same commands, and the column
that matters is the delta. It also reads the campaign twice: once with N=2's generator, so
that the delta over identical input is the engine's alone, and once with the generator
widened to the forms the engine learned since N=2.

`gap:` and `finding:` slugs keep the names of the earlier readings.

---

## 1. What moved {#what-moved}

**All three findings of N=2 are closed, and no defect is known to be live.** CAST to a
target without an exact lane refuses by name (finding: cast-target-collapse), and so does
EXCLUDE of either qualifier of a USING key (finding: using-exclude-left-key). The
transformer-path bench runs (finding: transform-bench-broken). Neither campaign found a
value divergence: 0 of 2000 on N=2's grammar and 0 of 2000 on the widened one. Three
further defects that served where DuckDB does not were found on master and fixed, and a
fourth was caught before it merged; the findings section lists them.

**More SQL serves, almost none of it on N=2's grammar.** On N=2's 2000 queries, `AGREE`
went from 1079 to 1081 and refusals from 875 to 872. The work since N=2 served forms that
grammar never emits: VARCHAR/BOOLEAN comparison and CAST, `IS [NOT] DISTINCT FROM` as an
expression, all-NULL `CASE`/`COALESCE`/`least`/`greatest`, a number joining a VARCHAR key,
schema-qualified columns and keys, dotted struct field names, struct leaves as join keys,
and struct fields read by `s['f']`, `struct_extract` and `v.*`. The widened generator now
reaches the struct and CAST forms (the coverage section); N=2's did not.

**One downstream product outcome.** sql-transform's marginalized projection scopes read
their frozen θ with `struct_extract`, so `compile()` refused all of them. All three lawful
shapes now compile, and the row path answers row-for-row what batch does
(`_marginal_test.py::test_a_projection_scope_serves_on_the_row_path`).

**Every refusal names its construct.** 872 of 872, up from 692 of 875 (79.1%). The 183
echoes N=2 counted are gone, which closes gap: refusal-naming. The share that says what to
do rose from 31.7% to 53.3%. Naming also exposed the largest coverage class, which the
echoes had hidden: derived tables (`FROM (SELECT …)`), 148 refusals of queries DuckDB
answers.

**The ladders did not move.** Corpus 548, L2 288, L3 260, and the authoring ladder
unchanged. Each sits exactly on its floor, as at N=2. Nothing served since N=2 appears in
the mined corpus.

**Latency: the regime holds, and the transformer path has its first number.** `spec`
serves one row in 2.7–10.8 µs, against 5.2–15.3 ms for DuckDB per call. It is still
1.19–1.70× slower than the handwritten Python twin at n=64. A fitted sklearn transformer
served through the UDF slot costs about 108 µs per row, against 1.4 µs for the same query
without it: roughly 80 times the SQL (gap: native-transform-families).

---

## 2. KPI status {#kpi-status}

| kpi | N=2, `24aafe8` | N=3, `bc167f4` | delta |
|---|---|---|---|
| kpi: training-round-trip (C1) | passes at depth 25 (80 tests, 7.6 s) and at 1,500 (58.8 s); default 25 | passes at depth 25 (80 tests, 5.5 s) and at 1,500 (67.3 s); default 25 | unchanged; finding: c1-depth still open |
| kpi: engine-parity (C2) | 0 `DIVERGE_VALUE`; corpus 0 FAIL; 7 `DIVERGE_OPT`; 2 live defects outside the grammar | 0 `DIVERGE_VALUE` on both grammars; corpus 0 FAIL; 7 `DIVERGE_OPT` (N=2 grammar), 6 (widened); no known live defect | both live defects closed |
| kpi: binding-parity (C3) | `_serving_test.py`: 11 passed | 11 passed | unchanged |
| kpi: transformer-parity (C4) | 76 passed | 76 passed | unchanged |
| kpi: no-third-mode (C5) | FAILED empty; 100% prefixed; 79.1% name the construct | FAILED empty (corpus and both campaigns); 100% prefixed; 100% name the construct | naming closed |
| kpi: coverage-ladder (D1) | corpus 548; L2 288; L3 260; authoring 11/11, 39/17/5 | same | unchanged, all on floor |
| kpi: serving-latency (D2) | `spec` 2.5–6.6 µs at n=1; 1.20–1.80× the twin at n=64; transformer path unmeasured | `spec` 2.7–10.8 µs at n=1; 1.19–1.70× the twin at n=64; transformer ≈ 108 µs/row | transformer path read for the first time |

---

## 3. Findings {#findings}

### 3.1 N=2's findings

| finding | N=2 | N=3 |
|---|---|---|
| cast-target-collapse | live | **closed** (`47aad0e`). `cast_target` accepts exactly the served spellings of TINYINT, SMALLINT, INTEGER, BIGINT, DOUBLE, VARCHAR and BOOLEAN; every other target refuses naming the type. The widened generator renders the served spellings, so a regression is reachable |
| using-exclude-left-key | live | **closed** (`47aad0e`). EXCLUDE of either qualifier of a USING key refuses by name |
| transform-bench-broken | live | **closed** (`5b236c4`). The bench declares its UDF with an Arrow schema and runs; its numbers are in the latency section |
| c1-depth | open, priced at about 50 s | **open.** The written depth passes in 67.3 s; the default is still 25 |

### 3.2 Defects found and fixed since N=2

Each of these served an answer, or accepted an input, where DuckDB 1.5.5 does something
else. None was reachable by N=2's generator.

| defect | fixed in | what it did |
|---|---|---|
| bind-time folding beyond DuckDB's binder | `d74a970` | an operand fold dead-arm-eliminated a CASE holding a column, which DuckDB's binder never folds: `- (CASE WHEN false THEN x END) \|\| 'y'` typed as an INTEGER NULL where DuckDB answers VARCHAR, and `abs()` or unary minus over it served where DuckDB refuses. The six xfail pins now pass as regression tests; `test_open_divergences.py` holds no pins |
| presence lane fed a non-struct | `25c0405` | a struct join key with no leaf lanes is read through its presence lane, which counted any non-NULL value as present: an int64 arrow column, or `5`, `"x"` or `[1]` in a row dict, joined as a present struct. Both boundaries now refuse it |
| wrong-schema column qualifier | `682f52e` | `other.__THIS__.a` served where DuckDB errors (`Referenced table "other.__THIS__" not found`) |
| VARCHAR key vs a narrower probe | `ec32249` | found during that change, before merge: a lane-erased map key served `'3000000000'` against an INTEGER probe, where DuckDB's cast fails |

### 3.3 New: gap: generator-name-collisions

*Target:* the generator's struct-leaf ON keys exercise the key path.
*Today:* the generator names row and static columns alike (`c0`, `c1`, …). Of the 34
widened-grammar cases that join on a static struct leaf, 29 are queries DuckDB rejects,
nearly all for an ambiguous unqualified key (`c0` in both relations), and 4 refuse for an
unrelated construct (a RIGHT JOIN, a WHERE under `shape='map'`, an unclassified ON
residual, a width-1 UDF). One agrees. The engine is right in every case; the generator mostly
spends this construct on invalid SQL.
*Closes when:* the generator qualifies the row side of an ON key, or draws static column
names from a disjoint pool, and the campaign shows struct-leaf keys agreeing.

---

## 4. Gap ledger, delta {#gap-ledger}

| gap | N=3 status | reading |
|---|---|---|
| refusal-naming | **closed** | 872 of 872 refusals name the construct; the runner reads no echoes (`26b02ab`) |
| wide-integer-lanes | **narrowed** | CAST targets without a lane refuse by name. The literal 9223372036854775808 still refuses: 34 refusals of queries DuckDB answers, as at N=2 |
| non-scalar-values | open | 7 refusals name a non-scalar column (N=2: 12; different queries, same class) |
| unshipped-decimal-arithmetic | open | `UNSHIPPED` 15 of 2000 on N=2's grammar, 19 on the widened one |
| join-composition-limits | open, half closed | unchanged since N=2 |
| static-only-fold | open, **waiting on the owner** | removal is ruled in `docs/decisions/closed/static-only-queries.md`; it changes user-visible behavior, so it waits for the owner's go-ahead |
| native-transform-families | open, **now measured** | ≈ 108 µs per transformer call per row, 80× the SQL row (latency section) |
| bench-baseline-flip | narrowed | unchanged since N=2: (a) the baseline's change of identity or (b) a real regression; (b) needs a bisect against `a6fa318` |
| admission-ladder-headroom | open, unchanged | 11/11 of 22 mined; 39/17/5 curated |
| parse-divergence-guards | open | not re-read |
| generator-name-collisions | **new** | the findings section |

---

## 5. Campaign verdicts and acceptance {#acceptance-reading}

Seeds 0–1999, two generators on the same engine.

- **N=2's grammar.** `fuzz/gen.py` at master `bc167f4`. Since N=2 it changed only in its
  module docstring (`4458616`), so it renders the same 2000 queries N=2 scored.
- **Widened grammar.** The same file plus this reading's change: CAST gains TINYINT and
  SMALLINT targets and renders every served spelling of each target (`INT4`, `TEXT`,
  `BOOL`, `DOUBLE PRECISION`, …); a struct lane read is spelled `w['f']`, `(w).f` or
  `struct_extract(w, 'f')` a third of the time. Any change to the generator re-deals every
  seed, so these are different queries, not a superset.

| verdict | N=2 | N=3, N=2's grammar | N=3, widened |
|---|---|---|---|
| `AGREE` | 1079 | 1081 | 1073 |
| `REFUSED` | 875 | 872 | 875 |
| `AGREE_TRAP` | 24 | 25 | 27 |
| `UNSHIPPED` | 15 | 15 | 19 |
| `DIVERGE_OPT` | 7 | 7 | 6 |
| `DIVERGE_VALUE` | 0 | 0 | 0 |
| `TIMEOUT` / `PANIC` / `SKIP` | 0 | 0 | 0 |

**Acceptance.** On N=2's grammar, 1128 of 2000 built: 56.4% (N=2: 56.3%). Taking out the
288 refusals of queries DuckDB rejects too, it is 1128 of 1712, or 65.9% (N=2: 65.7%). On
the widened grammar, 1125 of 2000 (56.3%), and 1125 of 1715 (65.6%) without the 285
DuckDB rejects.

**The `DIVERGE_OPT` cases** are N=2's: an INT64 overflow, a sqrt of a negative, or an
`ln(0)` inside an expression that optimizer-on DuckDB folds away (`… OR TRUE`). They agree
with the optimizer-off oracle. The widened grammar re-deals seed 1196, so it has 6.

---

## 6. Refusals {#refusals}

| DuckDB on the refused query | N=2 | N=3, N=2's grammar | N=3, widened |
|---|---|---|---|
| answers | 526 | 524 | 525 |
| rejects (parse or bind error) | 288 | 288 | 285 |
| traps on this data | 61 | 60 | 65 |

**What the refusals of queries DuckDB answers are**, on the widened grammar, grouped:

| class | refusals | owner |
|---|---|---|
| derived table, `FROM (SELECT …)` | 144 | whole-relation shapes; hidden at N=2 behind echoed SQL |
| `WITH` (common table expressions) | 87 | whole-relation shapes; row-local CTEs are a candidate class |
| UDF declared with a width-1 list return | 56 | the caller's declaration, emitted on purpose by the generator |
| `shape='map'`: a WHERE clause can drop the row | 52 | permanent: multiplicity only under `shape='many'` |
| row limits: `LIMIT`/`OFFSET`, `FETCH`, `TOP` | 49 | the row-limit restriction and gap: static-only-fold |
| integer literal 9223372036854775808 | 35 | gap: wide-integer-lanes (i128) |
| `DISTINCT` | 23 | whole-relation shapes |
| `ORDER BY` | 16 | whole-relation shapes |
| `GROUP BY` / `HAVING` | 15 | whole-relation shapes |
| non-scalar column | 13 | gap: non-scalar-values |
| RIGHT / FULL / CROSS join type | 8 | join-composition-limits |
| static table as the driving relation | 9 | gap: static-only-fold |

**Quality**, N=2's grammar:

| property | N=2 | N=3 |
|---|---|---|
| documented prefix | 875 (100%) | 872 (100%) |
| names the construct | 692 (79.1%) | 872 (100%) |
| actionable (says what to do) | 277 (31.7%) | 465 (53.3%) |

---

## 7. Coverage {#coverage}

Distinct (operator, argument type, edge class) triples: on N=2's grammar 636 reached and
557 agreed, exactly N=2's; on the widened grammar 655 and 575.

What the widening reaches, counted from each case's AST:

| construct | cases | agree | refused | other |
|---|---|---|---|---|
| lane spelled `struct_extract(w, 'f')` | 101 | 43 | 51 | 7 |
| lane spelled `(w).f` | 92 | 37 | 52 | 3 |
| lane spelled `w['f']` | 75 | 32 | 40 | 3 |
| CAST synonym spelling | 65 | 28 | 34 | 3 |
| CAST AS TINYINT / SMALLINT | 35 | 15 | 16 | 4 |
| static struct leaf as ON key | 34 | 1 | 33 (29 of them queries DuckDB rejects) | 0 |

The refusals in the first five rows come from the rest of the query, not from the new
construct: the leading causes are derived tables, CTEs, `shape='map'` WHERE clauses and the
2⁶³ literal, in the same proportions as the whole campaign. The last row is gap:
generator-name-collisions.

This is still a reading of the generator, not of the SQL people write.

---

## 8. Ladders {#ladders}

| ladder | N=2 | N=3 | gate |
|---|---|---|---|
| mined corpus replay | 548 / 130 / 0 of 678 | 548 / 130 / 0 | 0 FAIL and `MATCH_FLOOR = 548` |
| dialect L2 | 288 / 390 / 0 | 288 / 390 / 0 | `SUPPORTED_FLOOR = 288` |
| dialect L3, Spark | 260 / 418 / 0 | 260 / 418 / 0 | `SPARK_MATCH_FLOOR = 260`; BigQuery skips without credentials |
| authoring, mined | 11 marginalized / 11 refused of 22 | same | `_corpus_test.py::test_mined_corpus_scoreboard` |
| authoring, curated | 39 / 17 / 5 | same | `_corpus_test.py::test_progression_totals` |

---

## 9. Serving latency {#latency-reading}

`benchmarks/bench_serving.py`, p50 per call, release build of the reading's tree
(`confit.BUILD_PROFILE == "release"`). The three-way parity gate passed on all five
scenarios before timing. Two consecutive runs:

| scenario | `spec` n=1 (run 1 / 2) | `python_dict` n=1 (run 1 / 2) | DuckDB per call, n=1 | `spec` / `python_dict` at n=64 (run 1 / 2) |
|---|---|---|---|---|
| titanic | 4,250 / 4,204 ns | 2,404 / 2,428 ns | 7.9 / 7.9 ms | 1.65× / 1.66× |
| house_prices | 6,509 / 6,489 ns | 4,687 / 4,632 ns | 12.4 / 12.4 ms | 1.19× / 1.24× |
| fraud_txn | 6,991 / 6,870 ns | 5,203 / 5,331 ns | 15.2 / 15.3 ms | 1.48× / 1.39× |
| store_sales | 10,756 / 6,918 ns | 4,485 / 4,588 ns | 14.2 / 14.5 ms | 1.65× / 1.63× |
| feature_bundle | 2,670 / 2,712 ns | 1,503 / 1,501 ns | 5.2 / 5.3 ms | 1.61× / 1.70× |

The regime holds: three orders of magnitude under DuckDB per call. The n=64 ratio against
the twin repeats between runs and sits in N=2's band. store_sales at n=1 read 10.8 µs once
and 6.9 µs the next run; the other `spec` cells repeat within 2%.

**The transformer path, first reading** (`benchmarks/bench_transforms.py`, one run,
ns per row):

| variant | n=1 | n=64 | batch | cost over `sql_only` at n=1 |
|---|---|---|---|---|
| `sql_only` (marginalized aggregates, no UDF) | 1,372 | 629 | 4,257 | — |
| `udf_plain` (+ one Python UDF) | 2,040 | 1,069 | 6,686 | +668 ns |
| `tf_width1` (+ one fitted StandardScaler) | 108,968 | 109,937 | 132,906 | +107,596 ns |
| `tf_fields2` (two fields of one width-2 PCA) | 109,673 | 112,245 | 123,103 | +108,300 ns |

The UDF trampoline itself costs 0.7 µs. The sklearn call on a 1×k block costs about
107 µs, and it does not amortize with batch size, because the row path calls it once per
row. `tf_fields2` costs the same as `tf_width1`: two fields of one transformer share one
call, as pinned. This is the measurement gap: native-transform-families was waiting for.

---

## 10. What this reading leaves open {#left-open}

| item | where it goes |
|---|---|
| gap: static-only-fold | waiting on the owner's go-ahead: the removal changes user-visible behavior |
| finding: c1-depth | the owner's choice: raise the default to 1,500 (about 60 s more on this machine) or correct the text |
| gap: generator-name-collisions | the generator's ON-key production |
| gap: native-transform-families | 107 µs of the 109 µs row is the sklearn call; native families (a scaler is an affine map) would remove it |
| gap: bench-baseline-flip | a bisect against `a6fa318` |
| ask: next-query-classes | the evidence below |

### 10.1 ask: next-query-classes, with this reading's evidence

Recoverable refusals of queries DuckDB answers, widened grammar (of 525):

| candidate | recoverable | note |
|---|---|---|
| derived tables, `FROM (SELECT …)` | up to 144 | new at this reading: N=2 could not count them, because the refusal echoed the SQL |
| row-local CTEs | up to 87 | only the row-local ones; the grammar does not separate them |
| HUGEINT and the unsigned family (the i128 lane) | 35 | |
| nested and struct-valued outputs | 13 | |
| native transform families | not ranked by coverage | an 80× latency lever on the transformer path |

The ranking describes the generator, not demand, as N=1 and N=2 said. It remains the
owner's question.

---

## 11. Environment and reproduction {#environment-and-repro}

| | |
|---|---|
| worktree | master `bc167f4`; the widened campaign adds this reading's `fuzz/gen.py` change |
| DuckDB | 1.5.5, asserted by `confit.oracle.Oracle` on open |
| build | `uv sync --group spark`; `confit.BUILD_PROFILE == "release"` |
| machine | Linux x86_64, 4 cores, Python 3.14.7 |
| campaign flags | `--workers 4 --timeout 20`; generators `sha256:33ad33f8946488e7` (N=2's grammar) and `sha256:7bfd33c60d3745e8` (widened) |

| reading | command |
|---|---|
| campaign | `python -m fuzz.runner --seed 0 --n 2000 --workers 4 --timeout 20` (from `packages/confit/`); 16 s wall |
| corpus replay | `pytest -s packages/confit/tests/test_corpus_replay.py` |
| dialect L2 / L3 | `pytest -s packages/confit/tests/test_dialect_corpus_gate.py` / `test_dialect_cross_engine_gate.py` |
| C1 default / written depth | `pytest packages/sql-transform/sql_transform/_projection_test.py`; the same with `MARGINALIZE_FUZZ_N=1500 -k test_fuzz_differential` |
| C3 / C4 / D1 | `pytest` on `_serving_test.py` / `_transformers_test.py _trees_test.py` / `_corpus_test.py` |
| serving latency | `python -m benchmarks.bench_serving` (twice), then `python -m benchmarks.bench_transforms` |
| widened-construct table | each seed's AST walked with `fuzz.coverage._nodes`, the case run through `fuzz.oracle.run_case` |

**Reading N=4 replaces none of this.** It is a new file next to this one.
