---
loop: confit
date: 2026-10-14
master: b4b8d03
previous: mandates/confit/reports/2026-10-06-short-reading-2.md
kind: milestone
trigger: cross-loop delivery
kpis:
  training_round_trip: "pass: C1 passes with its default of 1,500 random window queries on b4b8d03 (145 tests)"
  engine_parity: "fail: 5 open classes of DIVERGE_VALUE or DIVERGE_TRAP in the 8 nightly campaigns on b4b8d03 (3,200,000 seeds)"
  binding_parity: "pass: _serving_test.py, 12 tests pass on b4b8d03"
  transformer_parity: "pass: _transformers_test.py and _trees_test.py, 94 tests pass on b4b8d03"
  no_third_mode: "pass: _corpus_test.py, 83 tests pass; corpus 0 FAIL"
  coverage_ladder: "pass: corpus 543 of 678 (floor 539), L2 288, L3 260; the authoring cases have new categories since 644d2ee"
  serving_latency: "open: one request in 4.5 to 20 µs (DuckDB 9 to 34 ms a call); 1.0 to 1.7 times the handwritten Python function at 64 rows"
  corpus_match: 543
  corpus_total: 678
  corpus_fail: 0
  gated_campaign_classes: 9
  catalog_needs_open: 2
  catalog_needs_served_since_last: 0
needs_owner:
  - questions/README.md
  - questions/README.md
  - questions/README.md
needs_from_other_loop: []
---

# Confit: milestone report, 2026-10-14

This report names master b4b8d03. It follows the [second short reading of
2026-10-06](2026-10-06-short-reading-2.md), which named master 4865d9f.

The reporting policy ([`loops/reporting.md`](https://github.com/ahrzb/sql-transforms/blob/40f6ab17/loops/reporting.md)) gives two
forms of a report. A short reading is the short form. The loop writes one
when 24 hours of work have passed, or 15 PRs have merged, since the last
report. A milestone report, such as this one, is the full form. The policy
also lists the events that start a milestone report, and the keys of the
KPI block at the top of this report.

This report starts from one of these events, a cross-loop delivery: a
change of confit serves a need of the native loop. The native loop is the
loop that builds the native catalog. Confit now builds the input guard of a
catalog entry once, and so serves part of one need (#430). The nightly
campaigns of the pause also found new classes, and so the KPI
`engine_parity` changed from pass to fail. These are two more events of the
list.

The owner paused both loops from 2026-10-06 at 16:11Z to 2026-10-14 at
10:07Z. A paused loop finishes the work that it started, but it starts no
new work. So the loop merged #430 on 2026-10-06 at 17:44Z, as the last work
that it had started before the pause.

The owner's own changes followed, 644d2ee included, up to b4b8d03 on
2026-10-07 at 03:56Z. So the eight nightly campaigns of the pause ran on
b4b8d03. On 2026-10-14, the loop measured all the other numbers again on
b4b8d03. It used a release build of confit (compiled with optimizations), on
one machine with 4 cores.

## What moved

**Confit builds the input guard of a catalog entry once, not again at each
field read (#430).** A catalog entry returns a struct. Field 0 of the
struct holds the input guard, a set of checks that traps where the twin of
the entry raises an error. A query can read the fields of the struct one at
a time. At each read, the oracle evaluates the whole struct, and so it runs
the guard again. So when the guard traps, a read of any field traps too.

Confit must trap in the same way, so before #430 it built one copy of the
guard for each field read. Now, when confit builds the query, it finds each
copy that an earlier copy already ran for the same request. It removes that
copy. A request that fails the guard still traps with the same error as
before. This table gives the build time of a query that reads each field of
a struct:

| struct | confit before #430 | b4b8d03 |
|---|---|---|
| 500 fields, a guard of 64 checks | 1.72 s | 0.32 s |
| 2,000 fields, a guard of 64 checks | 8.56 s | 1.44 s |
| 2,000 fields, no guard | 1.69 s | 1.60 s |

"Confit before #430" is a release build of 24674cd, the last change of
confit before #430. Both columns use the Python code of b4b8d03, and only
the build of confit differs. On b4b8d03, a query that reads the struct
whole, as one value, builds in 0.14 s at 2,000 fields with the guard of 64
checks.

**#430 makes one more catalog configuration native, and confit builds the
other six 2 to 31 times faster.** The table gives the result of
`to_native`, the call that turns a fitted sklearn transformer into a
transform that confit serves. It also gives the build time of a query that
reads each field of the catalog entry. This time is the best of two builds
after `to_native`, in the same process.

| configuration | fields | confit before #430 | b4b8d03 |
|---|---|---|---|
| OneHotEncoder, 16 string features of 125 categories | 2,000 | `NotNative` after a Cranelift verifier error | native, build 2.0 s |
| OneHotEncoder, 8 string features of 50 categories | 400 | build 8.5 s | build 0.27 s |
| RandomTreesEmbedding, 100 trees of depth 5, 8 features | 2,146 | build 11.1 s | build 2.5 s |
| PolynomialFeatures, degree 2, 64 features | 2,145 | build 8.2 s | build 2.6 s |
| SplineTransformer, 8 knots, degree 3, 64 features | 640 | build 7.9 s | build 3.2 s |
| StandardScaler, 256 features | 256 | build 2.1 s | build 0.66 s |
| OrdinalEncoder, 32 string features of 125 categories | 32 | build 2.1 s | build 1.0 s |

The first row is the configuration that is now native. sklearn is the
Python library scikit-learn. Its transformers in the table give these
output fields:

- OneHotEncoder: one field for each category of each feature. The field is
  1 for the category of the row and 0 for the other categories.
- OrdinalEncoder: one field for each feature, with a number for each
  category.
- RandomTreesEmbedding: one field for each leaf of each tree of a random
  forest. The field is 1 for the leaf that the row reaches.
- PolynomialFeatures: the products of the features, up to degree 2 here.
- SplineTransformer: the values of spline functions of each feature.
- StandardScaler: each feature minus its mean, divided by its standard
  deviation.

Cranelift is the code generator of the Cranelift backend. It checks the
code of each function while it compiles the function. A verifier error
means that this check failed. The error is an internal error of confit. It
is not a refusal, which names the construct that confit does not serve.

**The estimates of the catalog still refuse two configurations that confit
now builds.** Each catalog entry estimates the build time of a
configuration. It raises `NotNative` when the estimate passes 7 s. The
native loop fitted the formulas of these estimates on builds that it
measured before #430.

The formulas are `_case_seconds` and `_paths_seconds` in `native/trees.py`,
and `_build_estimate` in `native/spline.py`. A forest builds as one CASE
expression for each tree, or as the paths from the root of each tree to its
leaves. `_case_seconds` estimates the first form, and `_paths_seconds` the
second. For L output fields and F features, `_case_seconds` gives
`5.32e-4*L + 7.67e-7*L**2 + 5.32e-5*L*F` seconds. Each of the three
formulas has a term in L × F. This term is the cost of the guard at each
field read, which #430 removed.

So `to_native` raises `NotNative` for these two configurations, with
confit before #430 and with b4b8d03. The column "estimate" gives the
estimate of the catalog. The last two columns give the build time that the
loop measured without the limit of 7 s:

| configuration | estimate | build, confit before #430 | build, b4b8d03 |
|---|---|---|---|
| RandomTreesEmbedding, 100 trees of depth 5, 64 features | 10 s | 20.9 s | 2.8 s |
| SplineTransformer, 8 knots, degree 3, 128 features | 16 s | refused after 17.7 s | 6.7 s |

For these builds, the loop removed the limit of 7 s in the process that
measured them. Confit before #430 refuses the SplineTransformer with the
error "the compiled query is past the code generator's size limit (Code
for function is too large)". On 2026-10-06 at 17:45Z, after the merge of
#430, the loop sent its build times to the native loop. The native loop can
then fit its estimates again.

**The nightly campaign found cases with a gated verdict on each of the
eight nights.** From 2026-10-07 to 2026-10-14, it ran 3,200,000 seeds on
b4b8d03, and 65 cases got a gated verdict. So each of the eight runs
failed.

The issue of the nightly findings and the logs of the runs give the seeds
of 46 of these cases. The loop ran each of the 46 cases again on b4b8d03.
The 19 cases of DIVERGE_VALUE, DIVERGE_TRAP and OPT_EMULATED among them
gave the same verdicts again. A release build of 24674cd, the last change
of confit before #430, gave the same 19 verdicts. So #430 did not cause
these cases.

The 46 cases form 9 classes:

- Five classes are defects of confit. The loop fixes them next.
- In two classes, a run passes the time limit of the campaign, but confit
  has no defect. In one class, confit and the oracle are both slow. In the
  other class, only DuckDB with its optimizer on is slow. PLANS holds both
  classes.
- Two classes wait on the owner's open decision records.

The section "The nightly campaigns of the pause" gives each class.

**Six other changes of confit merged before the pause.**

- A struct is now one output value (#417). The owner [ruled the
  order](../../../proposals/closed/next-query-classes.md) in which confit takes
  the next kinds of query, and this kind is the third.
- An output item without an alias gets the name that the oracle gives it
  (#419).
- A list or struct output inside `CASE WHEN g IS NULL THEN NULL ELSE ... END`,
  where g is a column, builds about as fast as before #417 again (#422).
  #417 gave each element of the list, and each field of the struct, a CASE
  of its own. This made such builds up to 5 times slower. Now only an
  element that can trap keeps its CASE.
- Three changes fix the classes of the nightly campaign of 2026-10-06.
  #424 fixes five classes. For example, a LEFT JOIN over a static table
  without rows no longer evaluates its ON clause. #425 makes confit refuse
  by name a join without keys to a static table whose columns the query
  does not read. Before, it stopped with an internal error. #426 makes an
  operator over a subquery column that is the NULL literal evaluate its
  other operands, as the oracle does.

#417 and #419 also extend the generator of the campaign. It now generates
struct values (#417) and SQL functions (#419). Of the 46 cases with seeds,
10 have the verdict DIVERGE_VALUE. Four of these 10 cases read a field of a
SQL function.

**The corpus and the dialect levels did not move.** The coverage ladder
(D1) counts the SQL that is accepted and answered correctly. It has three
parts: the corpus, the two dialect levels and the authoring cases. The
corpus matches 543 of 678 statements with 0 FAIL, as on 2026-10-05. Its
floor, the minimum that the gate accepts, is 539.

The dialect support of confit prints a parsed statement back as SQL. Its
level L2 counts 288 statements whose printed SQL gives the oracle the same
answer as the original SQL. Its level L3 counts 260 statements whose SQL
printed for Apache Spark gives the same answer in Spark. Both counts are as
on 2026-10-05.

**The owner's change of 2026-10-06 (644d2ee) gives the authoring cases new
categories.** The authoring cases are SQL statements for sql-transform. The
mined cases come from the window tests of DuckDB, and the curated cases are
written by hand. Each case is now kept, migrated or refused:

- sql-transform accepts the original SQL of a kept case.
- A migrated case uses explicit replacement SQL. The test compares its
  values with the values of the original SQL.
- A refused case stops with an error that names its construct.

Of the 61 curated cases, 12 are now migrated:

- 7 chains, which are queries over a WITH clause or a subquery. The change
  stopped the acceptance of chains on purpose.
- 5 cases that needed a declared schema, in which the caller gave the names
  and types of the input columns. An example is `SELECT * RENAME (depname
  AS dep)`.

One case that was refused is now kept: `SELECT salary + 1 AS s, s * 2 FROM
__THIS__`, where the second item reads the alias of the first. So 33
curated cases are kept, 12 are migrated, and 16 are refused.

**Serving latency: the ratios hold on a slower machine.** Confit serves one
request in 4.5 to 20 µs, and DuckDB, with its default settings, takes 9 to
34 ms for one call. At 64 rows, confit takes 1.0 to 1.7 times the time of a
Python function that an engineer writes by hand for the same transform. The
reading of 2026-09-26 gave 1.2 to 1.7 times.

This machine is slower than the machine of the reading of 2026-09-26. On
this machine, DuckDB takes about 2 times as long, and the handwritten
function takes 1.8 to 3.3 times as long. So the times of the two readings
do not compare, and the loop compares only the ratios.

## The success measures

The KPI block gives each measure the state pass, fail or open. Serving
latency (D2) has no threshold to pass, so its state is open.

| KPI | before (the latest reading that measured it) | b4b8d03 | delta |
|---|---|---|---|
| C1: fit, then serving on the fit data, gives the original values bit-exact (`training_round_trip`) | passes with the default of 25 random cases (2026-10-05, 35617d7) | passes with the default of 1,500 random cases (145 tests) | the default is now 1,500 cases, as [ruled](../../../proposals/closed/c1-depth.md) |
| C2: confit agrees with the oracle (`engine_parity`) | 0 gated classes in 10,000 seeds (2026-10-06, 4865d9f) | 5 open classes of DIVERGE_VALUE or DIVERGE_TRAP in 3,200,000 seeds | 5 new classes, in 320 times as many seeds |
| C3: DuckDB batch execution and the two interfaces of confit give the same values for a fitted transform (`binding_parity`) | `_serving_test.py` passes (2026-10-05) | 12 tests pass | unchanged |
| C4: transformer columns agree with sklearn (`transformer_parity`) | `_transformers_test.py` and `_trees_test.py` pass (2026-10-05) | 94 tests pass | unchanged |
| C5: no third mode (`no_third_mode`) | corpus 0 FAIL (2026-10-05) | corpus 0 FAIL; `_corpus_test.py` 83 tests pass | unchanged |
| D1: coverage ladder, corpus and dialects (`coverage_ladder`) | corpus 543; L2 288; L3 260 (2026-10-05) | corpus 543; L2 288; L3 260 | unchanged |
| D1: coverage ladder, authoring cases | mined 22: 11 accepted, 11 refused; curated 61: 39 accepted, 17 refused, 5 with a declared schema (2026-09-26) | mined 22: 11 kept, 11 refused; curated 61: 33 kept, 12 migrated, 16 refused | new categories (644d2ee) |
| D2: serving latency (`serving_latency`) | 2.7 to 10.8 µs a request; 1.2 to 1.7 times the handwritten function at 64 rows (2026-09-26) | 4.5 to 20 µs a request; 1.0 to 1.7 times | the ratio holds on a slower machine |

A random case of C1 is a random query of windows over a random table of 1
to 39 rows, from a fixed seed. For C3, DuckDB batch execution runs the
fitted transform in DuckDB over a whole table. Of the two interfaces of
confit, `infer_rows` takes rows as Python dicts, and `infer_arrow` takes an
Arrow table. The test files of C3, C4 and C5 are in
`packages/sql-transform/sql_transform/`.

The gate passes on b4b8d03: 326 Rust tests (1 ignored) and 7,749 Python
tests (12 skipped, 3 expected failures). The gate does not run the nightly
campaign, so it passes while C2 fails.

## The nightly campaigns of the pause

The nightly campaign runs 400,000 new seeds each night, in four parallel
parts. It gives each case 20 s for its three runs together. Confit runs
first, then the oracle, then DuckDB with its optimizer on. When the 20 s
run out, the verdict is TIMEOUT, and it names the run that was in progress.
The eight nights of the pause found these cases with a gated verdict:

| verdict | cases | nights with cases |
|---|---|---|
| DIVERGE_VALUE: confit answers with other values or types than the oracle | 12 | 6 |
| DIVERGE_TRAP: confit traps and the oracle returns rows, or the reverse | 7 | 4 |
| OPT_EMULATED: confit agrees with DuckDB with its optimizer on, but not with the oracle | 2 | 2 |
| TIMEOUT in the run of the oracle or of DuckDB with its optimizer on | 43 | 8 |
| TIMEOUT in the run of confit | 1 | 1 |

DIVERGE_OPT is the reverse of OPT_EMULATED: confit agrees with the oracle,
and only DuckDB with its optimizer on gives another answer. It is not
gated.

The issue "Nightly campaign findings" (#435) and the logs of the 32 parts
give the seeds of 46 of the 65 cases. GitHub keeps the seeds of the other
cases in the files of each run. The network of the loop's machine does not
allow the download of these files.

This table gives the 9 classes of the 46 cases, one row for each cause. The
column "cases" gives the number of cases and their seeds.
`python -m fuzz.runner --seed N --n 1` runs the case of seed N again.

| cause | verdict | cases | next |
|---|---|---|---|
| A field read of a SQL function whose body is NULL under a constant condition, such as `CASE WHEN TRUE THEN NULL ELSE struct_pack(...) END`. The oracle types the field as NULL. Confit gives the field the type that the struct declares. | DIVERGE_VALUE | 4 (5397716, 6212474, 7430849, 7843947) | fix |
| A UDF (a user-defined function, in Python) over an argument that names a column but has the same value on each row. An example is `udf0(5, 91.693 - -1.75, coalesce(8, c1))`, where `coalesce(8, c1)` is always 8. Confit calls the UDF once, at build time, and types its NULL result as NULL. The oracle calls the UDF for each row and keeps the declared type. | DIVERGE_VALUE | 2 (5144477, 6949338) | fix |
| A function or operator over an argument that is always NULL but is not the NULL literal, such as `trunc(CASE WHEN c1 THEN 53.832 * NULL END)`. The oracle gives the argument the type NULL, and the call the type BIGINT. Confit gives the argument the type INTEGER, and so the call the type INTEGER. | DIVERGE_VALUE | 4 (5248694, 5923305, 6186379, 6686677) | fix |
| A product in HUGEINT arithmetic (128-bit integers) that is exactly −2^127, such as `18446744073709551616 * -9223372036854775808`. The oracle traps on this product, and confit returns it. | DIVERGE_TRAP | 6 (6065493, 6112137, 6263288, 6349920, 6917565, 7511938) | fix |
| BETWEEN in the ON clause of a join, where the static side of the BETWEEN has arithmetic that can trap. Before the join, the oracle computes this side for each row of the static table. One static row traps there, although no request matches it. Confit computes the BETWEEN only for the pairs of rows that match, so it returns rows. | DIVERGE_TRAP | 1 (6738617) | fix |
| `levenshtein` over two strings of 131,068 characters. Measured alone, the oracle needs 73 s for the call, and confit needs 55 s. The work grows with the product of the two lengths. Confit runs first, so the limit stops the case in the run of confit. | TIMEOUT, confit | 1 (6154334) | PLANS |
| A constant `lpad` of 2,147,483,647 characters, under a WHERE that is always false. DuckDB with its optimizer on computes the constant before it runs the query, and the limit stops it. The oracle returns no rows in 0.45 s, and confit also returns no rows. | TIMEOUT, DuckDB with the optimizer on | 2 (5068704, 7731793) | PLANS |
| A query with `lpad` or `repeat`. Confit traps on both backends, and the oracle runs past the time limit. | TIMEOUT, oracle | 24 | [the open record on a TIMEOUT of the oracle](../../../questions/README.md) |
| An INNER join whose static keys are all NULL. | OPT_EMULATED | 2 (7242549, 7829571) | [the open record on all-NULL static keys](../../../questions/README.md) |

## The needs of the native loop

The native loop lists three needs from confit in its PLANS. The third need
concerns an expression with two CASE trees, such as `0.5 * (tree(x) -
tree(-x))`. A CASE tree is a balanced tree of nested CASE expressions. One
CASE tree builds in a time linear in its size, but such an expression does
not. The native loop says that no catalog entry needs it now, so the KPI
block does not count it as open (`catalog_needs_open`).

This table gives the other two needs, which the KPI block counts as open:

| need | state on b4b8d03 |
|---|---|
| A struct output read field by field, built once for the call | Open. #430 removed the cost of the guard. At 2,000 fields, a struct read field by field still builds 10 to 13 times slower than the struct read whole. The build takes 1.44 s against 0.14 s with the guard of 64 checks, and 1.60 s against 0.12 s with no guard. |
| A named refusal, not a Cranelift verifier error | Open. The configuration that showed the verifier error now builds, because #430 makes its function smaller. The cause stays. Confit checks the size of a function before the optimizer of Cranelift runs, and this optimizer can make the function larger. So a function can pass the check of confit and then stop with a verifier error. |

Neither need is complete, so the KPI block counts 0 needs served since the
last report (`catalog_needs_served_since_last`).

## Serving latency

`benchmarks/bench_serving.py` measures the median time of one call for five
scenarios. Each scenario is the transform of a known tabular problem:

- titanic: the passengers of the Titanic.
- house_prices: the sale prices of houses, with 43 input columns and 54
  outputs.
- fraud_txn: card transactions, checked for fraud.
- store_sales: the sales of stores.
- feature_bundle: features that arrive in one struct column.

Before each run, the bench checks that confit, DuckDB with its default
settings and the handwritten function give the same values in all five
scenarios. Each cell gives run 1 and run 2.

| scenario | confit, 1 row (µs) | handwritten Python function, 1 row (µs) | DuckDB, 1 call (ms) | confit ÷ handwritten function, 64 rows |
|---|---|---|---|---|
| titanic | 6.3 / 11.8 | 6.6 / 4.3 | 16.6 / 15.1 | 1.59 / 1.51 |
| house_prices | 18.7 / 20.1 | 15.4 / 8.7 | 22.7 / 25.7 | 1.12 / 1.72 |
| fraud_txn | 17.4 / 19.0 | 9.4 / 9.5 | 34.0 / 34.1 | 1.14 / 1.14 |
| store_sales | 9.4 / 14.7 | 8.2 / 14.7 | 26.5 / 30.6 | 1.22 / 1.04 |
| feature_bundle | 4.5 / 7.5 | 2.9 / 2.9 | 9.3 / 11.2 | 1.14 / 1.54 |

A cell for one row can change by up to 1.9 times between two runs, so read
these cells as a range. On 2026-09-26, the handwritten function took 1.5 to
5.3 µs for one row, and DuckDB took 5.2 to 15.3 ms for one call.

`benchmarks/bench_transforms.py` measures, in one run, the serving of a
transform that calls a fitted sklearn transformer. During the fit, a window
aggregate (`... OVER (PARTITION BY grp)`) fits the transformer on each
group of rows of the fit data. For each request, the transform calls the
fitted transformer of the group of the request. The table gives µs for each
row:

| variant | 1 row | 64 rows | added to 1 row, against no UDF |
|---|---|---|---|
| no UDF | 1.7 | 1.0 | |
| one Python UDF | 2.7 | 2.7 | 1.0 |
| one field of a fitted StandardScaler | 249.9 | 231.2 | 248.2 |
| two fields of one fitted PCA | 265.7 | 247.5 | 264.0 |

PCA (principal component analysis) gives the projections of the features
on their principal components. The fitted StandardScaler now adds 248.2 µs
to a row. On 2026-09-26, it added 107.6 µs, so the cost is 2.3 times as
large. On this machine, the handwritten function is also 1.8 to 3.3 times
as slow. So the slower machine can explain this increase, but the loop did
not measure the two commits on the same machine.

## Needs the owner

- [An INNER join whose static keys are all NULL](../../../questions/README.md):
  in the nightly cases 7242549 and 7829571, confit traps, and the oracle
  returns no rows.
- [Empty-static witness: must the padded trap match confit's category?](../../../questions/README.md):
  for seed 4313391 with an empty static table, confit traps with a
  conversion error. The witness, a run of the oracle with one row in the
  static table, traps with an overflow.
- [The oracle times out where confit traps first](../../../questions/README.md):
  in 24 nightly cases with `lpad` or `repeat`, confit traps on both
  backends. The oracle runs past the time limit.

## Next

1. Fix the five defects of confit that the nightly campaigns found, each
   with a parity test. Then run a campaign of at least 10,000 seeds.
2. Then take the items of PLANS, correctness first:
   - A regexp function with the NULL literal as its pattern, such as
     `s ~ NULL`. Confit replaces the call with a NULL constant, but the
     oracle calls the function for each row. So when an operator over the
     call has another operand that traps, the oracle traps and confit
     returns rows.
   - A field read over `struct_pack`, such as
     `struct_pack(f := a, g := CAST(NULL AS BIGINT)).g`. Confit replaces
     this read with a NULL constant, but the oracle evaluates every field
     for each row. So the oracle can trap where confit returns rows, as
     for the regexp function.
   - The two TIMEOUT classes above.
3. Then take the two open needs of the native loop.

## How the loop measured

The scripts in `/mnt/project-files/` are in the shared folder of the
project.

| reading | command |
|---|---|
| gate | `uv run --no-sync python scripts/gate.py` |
| corpus | `uv run --no-sync python scripts/corpus_counts.py` |
| dialects L2 and L3 | `pytest -s packages/confit/tests/test_dialect_corpus_gate.py packages/confit/tests/test_dialect_cross_engine_gate.py` |
| C1 | `pytest packages/sql-transform/sql_transform/_marginal_projection_test.py`, the test file of C1 |
| C3, C4, C5 and the authoring cases | `pytest` on `_serving_test.py`; on `_transformers_test.py` and `_trees_test.py`; on `_corpus_test.py` |
| serving latency | `python -m benchmarks.bench_serving`, two times, then `python -m benchmarks.bench_transforms` |
| builds with and without a guard | `/mnt/project-files/transforms-loop/confit-trap-size.py 500 2000`, for 500 and 2,000 fields: the best of three builds in one process |
| catalog configurations | `/mnt/project-files/confit-loop/catalog-reach.py`, one process for each configuration. For the column "confit before #430", the loop installed the release build of 24674cd. |
| the two configurations without the limit | `/mnt/project-files/confit-loop/catalog-nocap.py rte 64` (RandomTreesEmbedding with 64 features) and `catalog-nocap.py spline 128 3` (SplineTransformer with 128 features, degree 3) |
| the nightly cases | the reports on issue #435, and the logs of the 32 parts of the eight runs. Each case ran again with `fuzz.oracle.run_case_json`, or with `python -m fuzz.runner --seed N --n 1 --timeout 20` for a TIMEOUT. |
| `levenshtein` | one call over two strings of 131,068 characters, in the oracle and in confit |
