# Goal yardsticks, reading N=2 (2026-09-26)

**What this is.** The second reading of the yardsticks in
[success measures](../specs/success-measures.md), taken on master `24aafe8` on
2026-09-26. The [first reading](2026-09-02-goal-baseline.md) (N=1) was taken on
`2ba96e5` on 2026-09-02. This file does not edit that one: it reads the same measures,
and the column that matters is the delta. Every number here was produced in the
environment in the environment-and-repro section, unless the text says it was not
re-read.

`gap:` and `finding:` slugs keep N=1's names, so an entry can be followed from one
reading to the next. `gap: static-only-fold` is defined in the
[2026-09-21 reading](2026-09-21-per-row-aggregation-and-the-fold.md).

---

## 1. What moved {#what-moved}

**Parity on the generated grammar is clean. Parity overall is not met: two defects are
live.** Over seeds 0–1999 the campaign found no value divergence. N=1 found one
(seed 1804), which is now fixed and pinned. Two defects that the generator cannot reach
serve wrong answers on master today: finding: cast-target-collapse and
finding: using-exclude-left-key. Both break kpi: engine-parity and kpi: no-third-mode.

**More SQL serves.** On the same 2000 queries, acceptance rose from 52.8% to 56.3% (1056
to 1125 built), and `AGREE` rose from 1013 to 1079. BOOLEAN comparison accounted for 86
refusals at N=1 and accounts for none now. USING/NATURAL self-joins now serve under
`shape='many'`.

**Every refusal is documented, and refusal quality is measured.** 875 of 875 refusals
carry a documented prefix, up from 837 of 944 (88.7%). N=1 left the naming half
unmeasured. It reads 692 of 875 (79.1%); the other 183 echo source text. 277 (31.7%) say
what to do.

**The cost of refusing can now be separated out.** Of the 875 refusals, 288 are queries
DuckDB rejects too, so refusing them costs nothing. 526 are queries DuckDB answers. Those
526 are the real coverage gap on this grammar, and their classes rank the next work (the
refusals section).

**Every ladder sits exactly on its floor. The corpus gained a floor, and L3 was read for
the first time.** Corpus replay reads 548 of 678 (N=1: 547), now gated by
`MATCH_FLOOR = 548`, which closes gap: corpus-match-slip. The L3 Spark gate reads 260 of
678, exactly on its floor, and N=1 could not read it. L2 and the authoring ladder did not
move.

**C1 passes at its written depth, and doing so costs about 50 s.** The seeded round-trip
differential passes at 1,500 cases in 58.8 s. The whole file at the default depth of 25
takes 7.6 s. finding: c1-depth now has a price attached.

**Latency.** The regime holds: `spec` serves one row in 2.5–6.6 µs, against 4.7–13.9 ms
for DuckDB per call. Against the handwritten Python twin the engine is still slower,
1.20–1.80× at n=64, in two runs on a fresh release wheel. So a stale wheel is ruled out as
the cause of N=1's sign flip. The transformer-path harness does not run
(finding: transform-bench-broken).

---

## 2. KPI status {#kpi-status}

| kpi | N=1, 2026-09-02 | N=2, 2026-09-26 | delta |
|---|---|---|---|
| kpi: training-round-trip (C1) | pointer resolves; standing gate at depth 25 | passes at depth 25 (80 tests, 7.6 s) and at the written depth of 1,500 (58.8 s); the default is still 25 | first reading at the written depth |
| kpi: engine-parity (C2) | 1 `DIVERGE_VALUE` (seed 1804); corpus 0 FAIL; 7 `DIVERGE_OPT` | 0 `DIVERGE_VALUE`; corpus 0 FAIL; 7 `DIVERGE_OPT`; **2 live defects outside the grammar** | seed 1804 fixed; two new findings |
| kpi: binding-parity (C3) | pointer resolves; not read | `_serving_test.py`: 11 passed | read |
| kpi: transformer-parity (C4) | pointer resolves; bounds not re-read | `_transformers_test.py` and `_trees_test.py`: 76 passed | read |
| kpi: no-third-mode (C5) | FAILED empty; 88.7% of refusals prefixed; naming unmeasured | FAILED empty (corpus and campaign); 100% prefixed; 79.1% name the construct; the two findings are third-mode cases | prefix gap closed; naming measured |
| kpi: coverage-ladder (D1) | mined 11/11 of 22; curated 39/17/5; corpus 547; L2 288; L3 unread | authoring ladder unchanged; corpus 548 (floor 548); L2 288 (floor 288); L3 260 (floor 260) | corpus +1 and gated; L3 read |
| kpi: serving-latency (D2) | `spec` 2.3–8.8 µs at n=1; 1.13–2.05× slower than `python_dict` | `spec` 2.5–6.6 µs at n=1, against 4.7–13.9 ms for DuckDB; 1.20–1.80× slower than `python_dict` at n=64 (two runs); `bench_transforms.py` fails before timing | regime unchanged; slower-than-twin confirmed on a fresh wheel; transformer path unmeasured |

---

## 3. Findings {#findings}

A finding is the engine, or a bar, failing on its own terms. It is not distance from the
target, and it is never something to live with.

### 3.1 finding: cast-target-collapse

`CAST` and `TRY_CAST` to a type the engine has no lane for serve a different answer
instead of refusing. `cast_target` (`src/specializer/frontend.rs:8116`) maps every target
whose name contains `INT` and is not TINYINT, SMALLINT or INTEGER to i64. It maps FLOAT,
REAL, DECIMAL and NUMERIC to f64. Measured on master against DuckDB 1.5.5:

| query (row value) | confit | DuckDB |
|---|---|---|
| `CAST(a AS UINTEGER)` (−1) | −1 | Conversion Error |
| `CAST(a AS UTINYINT)` (300) | 300 | Conversion Error |
| `TRY_CAST(a AS UINTEGER)` (−1) | −1 | NULL |
| `CAST(a AS HUGEINT) + 1` (2⁶³−1) | overflow error | 9223372036854775808 |
| `CAST(a AS FLOAT)` (16777217) | 16777217.0, double | 16777216.0, float |
| `CAST(f AS DECIMAL(3,1))` (1.25) | 1.25, double | 1.3, decimal128(3,1) |
| `CAST(a AS DECIMAL(4,1))` (100000) | 100000.0 | Conversion Error |
| `CAST(s AS INTERVAL)` ('1 day') | conversion error, string to INT64 | 1 day |

The last row comes from a substring match: `INTERVAL` contains `INT`.

The documents disagree about this behavior. `known-limitations.md` (type-system
boundaries) says HUGEINT and the unsigned family "refuse by name rather than collapse to
i64". That holds for declared columns and statics, and not for CAST targets. The
frontend's module comment (`frontend.rs:21-23`) calls the integer half a deliberate
divergence. The success measures leave no room for one: "Neither a known implementation
gap nor an unruled ledger entry permits a further deviation."

*Why nothing caught it.* The generator renders CAST targets as BIGINT, INTEGER, DOUBLE,
VARCHAR and BOOLEAN only (`fuzz/gen.py`, the `Cast` arm of `rexpr`). No mined corpus
statement reaches it while serving. If the generator did reach the DECIMAL rows, it would
classify them `UNSHIPPED`, and that verdict skips the value comparison, so 1.25 against
1.3 would still go unseen.
*Closes when:* `cast_target` refuses, by name, every target it cannot represent exactly,
and the generator gains those targets so the campaign catches a regression. Serving them
is the work of gap: wide-integer-lanes and gap: unshipped-decimal-arithmetic.

### 3.2 finding: using-exclude-left-key

`SELECT * EXCLUDE (__THIS__.k) FROM __THIS__ JOIN dim USING (k)`, with `dim` static, gives
columns `(w, v)` in confit. DuckDB gives `(w, k, v)`: the key comes back unmerged, at the
right table's position and with the right table's values. The mirror form,
`EXCLUDE (dim.k)`, refuses by name (`EXCLUDE of a USING-merged column (DuckDB unmerges
it)`), but the left-qualified form is not caught. At `3204ff3`, before the USING
self-join work, the check already covered only the right qualifier, so this defect
predates that work.
*Closes when:* the left-qualified form refuses too, or both forms are modeled.

### 3.3 finding: transform-bench-broken

kpi: serving-latency names two harnesses. The second, `benchmarks/bench_transforms.py`,
does not run on master. It fails before any timing, while declaring its first UDF:
`AttributeError: 'tuple' object has no attribute 'types'`
(`sql_transform/_udf.py:132`, `take_types`). The bench still passes a UDF's types as a
tuple of strings, where the declaration has taken an Arrow schema since `2113cc7`
(2026-08-08). That predates N=1, which did not run this harness. So the transformer-path
latency, which is the measurement behind gap: native-transform-families, has no working
method today.
*Closes when:* the bench declares its UDFs in the current form and runs to completion.

### 3.4 The first reading's findings

| finding | N=1 | N=2 |
|---|---|---|
| seed-1804 | live: a NaN sign reaching a string | **closed.** Fixed; the found case and a NaN-sign pin are in `tests/test_double_to_varchar.py`, and seed 1804 no longer diverges |
| c1-depth | standing gate at 25 against a written 1,500–2,000 | **open, priced.** The default is still 25 (`_projection_test.py:375`). At 1,500 the test passes in 58.8 s. The success measures now say that a default run is not evidence the widening requirement ran, which names the gap without closing it |
| static-only-tie-order | a tie-producing static-only `ORDER BY` freezes one order | **open, subsumed** by gap: static-only-fold. `SELECT v FROM s ORDER BY g` with a tie in `g` still builds on master |

---

## 4. Gap ledger, delta {#gap-ledger}

| gap | N=2 status | reading |
|---|---|---|
| undocumented-boolean-comparison | **closed** | BOOLEAN comparisons serve, and the class (86 of 944 at N=1) is gone from the histogram |
| undocumented-refusal-prefixes | **closed** | 875 of 875 refusals carry a documented prefix. The corpus gate's `_CLEAN` (`test_corpus_replay.py:56`) is now `unsupported:` and `parse error:`; the two substring families N=1 found in it are gone |
| corpus-match-slip | **closed** | 548 of 678, gated by `MATCH_FLOOR = 548`; the dated count lives in `corpus-counts.json`. The old 550 survives in the 2026-08-13 dialect design spec (`:265`, `:298`) and in the narrative reports |
| join-composition-limits | **half closed** | USING/NATURAL self-joins serve under `shape='many'` (`test_duckdb_stageb_many.py::test_using_and_natural_self_joins_vs_oracle`). More than one join under `'many'` still refuses (`lower.rs:147`) |
| wide-integer-lanes | **narrowed, with a finding against it** | Narrow-lane overflow traps shipped (`known-limitations.md`, type-system boundaries). Unsigned and float32 columns refuse. A static column's refusal names the type (`static table 's' column 'u' has a non-scalar type: uint32`); a row column's (`row column 'a' has a non-scalar type`) calls a scalar non-scalar and omits the type. CAST targets do not refuse at all (finding: cast-target-collapse). The i128 lane would also admit the literal 9223372036854775808: 41 refusals, 34 of them on queries DuckDB answers |
| unshipped-decimal-arithmetic | open | `UNSHIPPED` holds 15 of 2000 (N=1: 14). All 15 are bare decimal literals, and all 15 agree in value once DuckDB's decimal is read as a double. Decimal reach is 109 cases: 56 refused, 37 agree, 15 unshipped, 1 agree-trap |
| non-scalar-values | open, now sized | 12 refusals name a non-scalar column, all on queries DuckDB answers |
| static-only-fold | open | unchanged since 2026-09-21; the campaign's `static_agg` arm still gets 31 agreements through it |
| parse-divergence-guards | open | not re-read |
| admission-ladder-headroom | open, unchanged | 11/11 of 22 mined; 39/17/5 curated |
| native-transform-families | open | not re-measured: the harness fails (finding: transform-bench-broken) |
| bench-baseline-flip | **narrowed** | Causes (c) machine noise and (d) stale wheel cannot explain the sign: two runs on a wheel built fresh from `24aafe8` agree at n=64 on all five scenarios. (a) the baseline's change of identity and (b) a real regression remain; (b) needs a bisect against `a6fa318` |

**New: gap: refusal-naming.**
*Target:* a refusal names the construct (the second outcome of kpi: no-third-mode, and
what kpi: named-refusal-share measures).
*Today:* 183 of 875 refusals (20.9%) echo source text instead. 168 are
`unsupported: FROM (SELECT …`, which prints a derived table back. 9 are
`join type Right(On(…` and 6 are `join type FullOuter(On(…`, which are Rust debug dumps of
the AST.
*Closes when:* those three refusal sites name their construct, and the runner's
refusal-quality section reads no echoes.

---

## 5. Campaign verdicts and acceptance {#acceptance-reading}

Seeds 0–1999. `fuzz/gen.py` is byte-identical between the two readings, so both score the
same 2000 queries. `fuzz/oracle.py` did change: nullability is now checked for soundness
against our own rows, and a refusal keeps DuckDB's outcome. The delta is therefore engine
changes plus classification changes, over identical input.

| verdict | N=1 | N=2 |
|---|---|---|
| `AGREE` | 1013 | 1079 |
| `REFUSED` | 944 | 875 |
| `AGREE_TRAP` | 21 | 24 |
| `UNSHIPPED` | 14 | 15 |
| `DIVERGE_OPT` | 7 | 7 |
| `DIVERGE_VALUE` | 1 | 0 |
| `TIMEOUT` / `PANIC` / `SKIP` | 0 | 0 |

**Acceptance.** 1125 of 2000 built rather than refused: 56.3%. As the success measures
require, the population is stated. It is the generated grammar, with 0 unknown outcomes.
It includes at least 55 refusals of UDFs the generator declares wrongly on purpose (a
width-1 list return), and 288 refusals of queries DuckDB rejects as well. With those 288
taken out of the denominator, the rate is 1125 of 1712, or 65.7%.

**The 7 `DIVERGE_OPT`.** All 7 agree with optimizer-off DuckDB and differ only from
optimizer-on DuckDB. In five, INT64 addition overflows inside an expression the optimizer
folds away (`… OR TRUE`). In one, sqrt of a negative traps the same way. In the
seventh, the optimizer-on run errors on `ln(0)` and we return rows. They are the standing cost of
exclusion: optimizer-on-answers: reported, not accepted.

---

## 6. Refusals {#refusals}

**By DuckDB's outcome on the same query.** This is new with this reading: a refusal now
keeps what DuckDB would have done.

| DuckDB on the refused query | refusals | share |
|---|---|---|
| answers | 526 | 60.1% |
| rejects (parse or bind error) | 288 | 32.9% |
| traps on this data | 61 | 7.0% |

**What the 526 are.** These are the refusals that cost coverage. The runner lists the top
15 classes per outcome. Grouped, the ten classes below cover 350 of the 526.

| class | refusals of queries DuckDB answers | owner |
|---|---|---|
| `WITH` (common table expressions) | 86 | whole-relation shapes; row-local CTEs are a candidate class |
| UDF declared with a width-1 list return | 55 | the caller's declaration, emitted on purpose by the generator |
| row limits: `LIMIT`/`OFFSET`, `FETCH`, `TOP` | 50 | the row-limit restriction and gap: static-only-fold |
| `shape='map'`: a WHERE clause can drop the row | 48 | permanent: multiplicity only under an explicit `shape='many'` |
| integer literal 9223372036854775808 | 34 | gap: wide-integer-lanes (i128) |
| `DISTINCT` | 25 | whole-relation shapes |
| `ORDER BY` | 19 | whole-relation shapes |
| `GROUP BY` / `HAVING` | 15 | whole-relation shapes; per-row aggregation is scoped in the 2026-09-21 reading |
| non-scalar column | 12 | gap: non-scalar-values |
| static table as the driving relation | 6 | gap: static-only-fold |

**Quality.** The runner scores each refusal (`fuzz/runner.py::refusal_quality`).

| property | refusals | share | N=1 |
|---|---|---|---|
| documented prefix | 875 | 100% | 88.7% |
| names the construct (no echoed SQL or AST dump) | 692 | 79.1% | unmeasured |
| actionable (says what to do) | 277 | 31.7% | unmeasured |

---

## 7. Coverage {#coverage}

The campaign now reports coverage as distinct (operator, argument type, edge class)
triples. It reached 636 and agreed on 557. Per operator, the widest gaps between reached
and agreed are `ln` (7 reached, 2 agreed), `trunc` (6, 3), tree UDFs (11, 6) and
`CAST AS BIGINT` (15, 9). "Reached, not agreed" means the case refused, trapped or
diverged; on this run no case diverged in value.

This is a reading of the generator, not of the SQL people write. The CAST row shows the
limit: the generator reaches five CAST targets, and the defect lives in the others.

---

## 8. Ladders {#ladders}

| ladder | N=1 | N=2 | gate |
|---|---|---|---|
| mined corpus replay | 547 match / 131 clean / 0 FAIL of 678 | 548 / 130 / 0 | 0 FAIL and `MATCH_FLOOR = 548` (`test_corpus_replay.py`) |
| dialect L2 | 288 / 390 / 0 | 288 / 390 / 0 | `SUPPORTED_FLOOR = 288` |
| dialect L3, Spark | unread (no `pyspark`) | 260 / 418 / 0 | `SPARK_MATCH_FLOOR = 260`; the BigQuery leg skips without credentials |
| authoring, mined | 11 marginalized / 11 refused of 22 | same | `_corpus_test.py::test_mined_corpus_scoreboard` |
| authoring, curated | 39 / 17 / 5 | same | `_corpus_test.py::test_progression_totals` |

Every ladder sits exactly on its floor. Between the readings the corpus gained one match
and a floor to hold it, and nothing else moved.

---

## 9. Serving latency {#latency-reading}

`benchmarks/bench_serving.py`, p50 per call, release wheel built from `24aafe8` for this
reading. The three-way parity gate passed on all five scenarios before timing. Two
consecutive runs:

| scenario | `spec` n=1 (run 1 / 2) | `python_dict` n=1 (run 1 / 2) | DuckDB per call, n=1 | `spec` / `python_dict` at n=64 (run 1 / 2) |
|---|---|---|---|---|
| titanic | 3,617 / 3,949 ns | 2,042 / 3,972 ns | 7.1 / 7.4 ms | 1.75× / 1.57× |
| house_prices | 6,108 / 6,161 ns | 4,459 / 3,940 ns | 11.7 / 11.6 ms | 1.20× / 1.20× |
| fraud_txn | 5,958 / 5,975 ns | 4,833 / 4,368 ns | 13.9 / 13.6 ms | 1.34× / 1.43× |
| store_sales | 6,624 / 5,973 ns | 3,887 / 3,920 ns | 13.0 / 12.3 ms | 1.64× / 1.68× |
| feature_bundle | 2,539 / 2,661 ns | 2,726 / 1,407 ns | 4.7 / 4.8 ms | 1.37× / 1.80× |

**The regime holds.** One row costs 2.5–6.6 µs in `spec` and 4.7–13.9 ms in DuckDB:
three orders of magnitude, as at N=1. No gate bounds this number, so a regression would
still fail nothing.

**The engine is slower than the twin, and n=1 is the wrong place to read the ratio.** The
`spec` cells repeat within 11% between runs. The `python_dict` cells at n=1 do not:
titanic reads 2,042 and then 3,972 ns, and feature_bundle 2,726 and then 1,407 ns. A
single-call p50 of a 2 µs Python function is noise-dominated here. At n=64 both runs agree
in direction on all five scenarios, 1.20–1.80× slower. N=1's n=1 ratios (1.13–2.05×) sit
in a similar band. When kpi: bench-refresh-cadence records a ratio, as N=1 proposed, the
n=64 ratio is the one that repeats.

**What this does to gap: bench-baseline-flip.** N=1 listed four candidate causes. (d), a
stale or debug wheel, is ruled out: the wheel was built for this reading and the harness
refuses debug builds. (c), machine noise, does not explain a direction that two runs
agree on, on a different machine. (a), the twin changing identity from pydantic models to
plain dicts, and (b), a real regression since `a6fa318`, remain. (b) costs one bisect.

The transformer path was not measured: finding: transform-bench-broken.

---

## 10. What this reading leaves open {#left-open}

| item | where it goes |
|---|---|
| finding: cast-target-collapse | a named refusal in `cast_target` for every target without an exact lane, and the missing targets in the generator; before any other widening |
| finding: using-exclude-left-key | a refusal for the left-qualified form, pinned beside the right-qualified one |
| finding: transform-bench-broken | the bench's UDF declarations, in the current form; then a transformer-path reading |
| finding: c1-depth | the owner's choice: raise the default to 1,500 (about 50 s more on this machine) or correct the text |
| gap: bench-baseline-flip | a bisect against `a6fa318` settles (b); if it is clean, (a) stands and the recorded prose is simply obsolete |
| gap: refusal-naming | three refusal sites |
| gap: static-only-fold | as routed by the 2026-09-21 reading |
| ask: next-query-classes | the evidence below |

### 10.1 ask: next-query-classes, with this reading's evidence

The candidates are those in N=1's left-open section, minus two that have shipped: the
narrow-lane overflow traps and USING/NATURAL self-joins. The new evidence is how many
refusals of queries DuckDB answers each candidate would recover on this grammar:

| candidate | recoverable refusals (of 526) | note |
|---|---|---|
| row-local CTEs | up to 86 | only the CTEs that are row-local; the grammar does not separate them |
| HUGEINT and the unsigned family (the i128 lane) | 34 | also removes the integer half of finding: cast-target-collapse by serving it |
| nested and struct-valued outputs | 12 | |
| decimal arithmetic | not ranked | reaches 109 cases, of which 56 refused; this reading does not split those refusals by cause |
| more than one join under `'many'` | not ranked | the generator emits one join per query |
| native transform families | not ranked | a latency lever, not a coverage one |

The ranking describes the generator and not demand, as N=1 said. It remains the owner's
question.

---

## 11. Environment and reproduction {#environment-and-repro}

| | |
|---|---|
| worktree | master `24aafe8`, clean |
| DuckDB | 1.5.5, asserted by `confit.oracle.Oracle` on open |
| build | `uv sync --group spark` (PEP 517 build through maturin); `confit.BUILD_PROFILE == "release"` |
| machine | Linux x86_64, 4 cores, 15 GB, Python 3.14.7 |
| campaign flags | `--workers 4 --timeout 20`; generator `sha256:c33c45430d488c19` |

N=1 ran on Windows 11 with 12 cores and `--workers 8`. Linux is the reference platform.

| reading | command |
|---|---|
| campaign | `python -m fuzz.runner --seed 0 --n 2000 --workers 4 --timeout 20` (from `packages/confit/`); 15 s wall |
| corpus replay | `pytest -s packages/confit/tests/test_corpus_replay.py` |
| dialect L2 / L3 | `pytest -s packages/confit/tests/test_dialect_corpus_gate.py` / `test_dialect_cross_engine_gate.py` |
| C1 default / written depth | `pytest packages/sql-transform/sql_transform/_projection_test.py`; the same with `MARGINALIZE_FUZZ_N=1500 -k test_fuzz_differential` |
| C3 / C4 / D1 | `pytest` on `_serving_test.py` / `_transformers_test.py _trees_test.py` / `_corpus_test.py` |
| serving latency | `python -m benchmarks.bench_serving`, then `python -m benchmarks.bench_transforms` |
| `UNSHIPPED` value check | the campaign's cases rerun with the unshipped schema delta suppressed and decimals read as doubles; 15 of 15 agree |
| the findings | the queries in the findings section, each run through `DuckDBInferFn` and `confit.oracle.Oracle` |

**Reading N=3 replaces none of this.** It is a new file next to this one.
