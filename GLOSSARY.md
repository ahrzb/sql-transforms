# Glossary

The terms of this project. Each entry says what a thing is. The words under
_Avoid_ are other names for the same thing. Do not use them for that thing.
They are not forbidden in other meanings. The rules for this file are in
the `simple-english` skill
(`.claude/skills/simple-english/SKILL.md`).

## Product

**Confit**:
The serving engine (`packages/confit`). It turns a transform and its static
tables into a native function that answers one request at a time.
_Avoid_: the compiler, the specializer (as the product name)

**sql-transform**:
The authoring package (`packages/sql-transform`). It authors, composes and fits
transforms.

**Transform**:
An authored computation `(F, T) -> R` over fit data `F` and request data `T`.
A row-local transform answers each request independently after fit.
_Avoid_: feature query, pipeline

**Request**:
One row of the request table. For row-local serving, its answer does not
depend on other requests in the same batch.

**Request table**:
The table that holds the request rows, `__THIS__` by default. The caller
declares it in `row_tables`.
_Avoid_: input table, row table

**Static table**:
A table whose content is fixed when the function is built. The caller
declares it in `static_tables`.
_Avoid_: frozen table, lookup table

**Params table**:
A static table that fit writes. It can hold fitted values or explicitly
retained fit data.
_Avoid_: parameter table, stats table

**Fit**:
The step that binds fit data `F` and writes params tables for later
execution on request data `T`.
_Avoid_: train, prepare

**Decorrelation**:
Rewriting supported correlated fit subqueries into fit queries and queries
over params tables.

**Marginalize**:
To replace a window aggregate over the request table with a join to a
params table that holds its value for each partition.
_Avoid_: precompute

**Serving SQL**:
The fitted SQL over request data and static tables. It does not read the
fit relation.
_Avoid_: rewritten query

**Build**:
The one-time step that parses, binds, partially evaluates and compiles a
transform. A build gives a function or a refusal.
_Avoid_: construction (in new text), prepare

**Refusal**:
A build that stops with a `ValueError` that names the construct confit does
not serve.
_Avoid_: rejection, bail-out, unsupported error

**Trap**:
An error that a function raises while it serves a request, because of the
values in that request. A trap is correct when the oracle raises the same
error for the same input.
_Avoid_: crash, exception (for this case), panic

**Shape**:
The number of output rows for each request that a build proves: `map` (one
row), `filter` (zero or one row) or `many` (any number).
_Avoid_: multiplicity mode

**Backend**:
One of the two executors of a built function: Cranelift (native code) or the
interpreter. The two backends must agree byte for byte.
_Avoid_: JIT (for both)

**Lane**:
The machine type that holds a value in a built function: i64, f64, i128 or
a string. A field of a struct output is a field, not a lane.

**Native catalog**:
The entries in `sql_transform.native` that translate a fitted sklearn
transformer into a transform that confit serves.
_Avoid_: native zoo, native transforms (for the catalog)

**Catalog entry**:
One translation in the native catalog. It turns a fitted sklearn
transformer of one class into a transform that confit serves. In the native
loop's records, "entry" alone means a catalog entry.

**Twin**:
The reference for a catalog entry: `PythonTransform`, which calls the fitted
sklearn estimator.
_Avoid_: baseline, python path

**Family**:
A group of sklearn transformers in one catalog module, for example
`scalers.py`.

**`NotNative`**:
The error that a catalog entry raises for a configuration that it does not
serve.

**Parity bound**:
The maximum distance that a family allows between a catalog entry and its
twin, for each output field. The distance is K·eps·S + τ, with these terms:
S is the error scale of the field. K is a constant that the family derives.
eps is 2^-52. τ is a small floor for underflow. A bound with K = 0 is
bit-exact.
_Avoid_: tolerance

**Error scale**:
The quantity S in a parity bound. It is a formula over the fitted state and
the input row, and the family declares it. It adds the sizes of the
operations whose rounding can differ between the entry and its twin.

**Ulp bound**:
A parity bound whose error scale is the size of the result itself. The
family states it in units in the last place (ulps). A bound of 0 is
bit-exact.
_Avoid_: tolerance

**Input guard**:
A test on the input row that a catalog entry runs, so that the entry traps
where the validation of its twin raises an error.

## Correctness

**Oracle**:
The fixed reference for confit's SQL behavior: DuckDB 1.5.5 with
`PRAGMA disable_optimizer`, all other settings at their defaults, and the
same UDFs registered.
_Avoid_: ground truth, DuckDB (without "optimizer off")

**Optimizer-off reading**, **optimizer-on reading**:
The answers of DuckDB with the optimizer disabled and enabled, on one
connection. The optimizer-off reading is the oracle's answer.

**Bit-exact**:
Equal in type and in the bits of each value, signed zero included.
_Avoid_: identical, matching

**Parity**:
Agreement between confit and a reference, bit-exact unless the comparison
contract approves a bound.
_Avoid_: equivalence, compatibility

**Pin**:
A recorded oracle fact: the SQL and inputs sent to DuckDB, and the exact
answer or error that it returned.
_Avoid_: golden value, snapshot

**Success measure**:
One of the seven measures in
`packages/confit/docs/specs/success-measures.md`: five correctness controls
(C1 to C5) and two optimization measures (D1, D2). Each has a KPI key.
_Avoid_: yardstick, goal metric

**Correctness control**:
A success measure that a change must not make worse to improve coverage or
latency (C1 to C5).
_Avoid_: invariant, guarantee

**Third mode**:
The outcome that the contract forbids: a build that succeeds and a function
that answers differently from the reference.
_Avoid_: silent divergence

**Divergence ledger**:
The index of measured differences between confit and the oracle, and the
evidence for each difference.
_Avoid_: known-bugs list

## Testing

**Gate**:
The full test run in `scripts/gate.py`. It must pass before a push. CI runs
it on each PR.
_Avoid_: test suite (for this run), checks

**Corpus**:
SQL statements taken from DuckDB's own test suite. The corpus run counts the
statements that confit matches, refuses or fails.
_Avoid_: test set, benchmark

**Campaign**:
A run of the random SQL generator (`fuzz.runner`) over a range of seeds. Each
seed gives one case and one verdict.
_Avoid_: fuzz run, sweep

**Seed**:
The integer that selects one generated case. The same seed always gives the
same case.

**Verdict**:
The class of one campaign case, for example `AGREE`, `REFUSED` or
`DIVERGE_VALUE`. `packages/confit/docs/oracle/04-verdicts-agreement-abstention-refusal.md`
defines each verdict.
_Avoid_: result, status

**Gated verdict**:
A verdict that fails a strict campaign (`fuzz.runner.GATED`). `DIVERGE_OPT`
is not gated, because it disagrees only with the optimizer-on reading.

**Finding**:
A measured result that disagrees with a requirement. A finding is a confit
defect until evidence shows a different cause.
_Avoid_: issue, bug (before the cause is known)

**Class**:
A group of findings with one cause. A gated class is a class whose verdict is
gated.
_Avoid_: bucket, cluster

**Nightly**:
The campaign that CI runs each night. It posts its findings to the open issue
titled "Nightly campaign findings", and creates that issue if none is open.

**Exclusion**:
A rule in `fuzz/exclusions.py` that removes a generated case from the
comparison. Each rule has a witness: a case that shows why the rule is
necessary.
_Avoid_: skip list, allowlist

**Metamorphic test**:
A test that compares two spellings of the same query, which must give the
same answer.

## Loops

**Loop**:
An agent session that turns a goal into merged PRs, one verified change at a
time. It stops when the remaining work needs the owner.
_Avoid_: bot, agent run

**Owner**:
The person who owns the repository and rules on decisions.
_Avoid_: user, human

**Supervisor**:
A loop session that splits work into tickets, starts workers, reviews their
PRs and merges them.
_Avoid_: lead, orchestrator

**Worker**:
A session that does one ticket on its own branch and opens one PR.
_Avoid_: subagent, helper

**Ticket**:
One item of work, for one branch and one PR. The loop's `tickets.md` lists
the tickets in progress.
_Avoid_: task, job

**Wave**:
A set of tickets that the supervisor starts at the same time.
_Avoid_: batch, round

**Decision record**:
A file in a loop's `decisions/` folder. An open record asks the owner a
question. A closed record holds the owner's ruling.
_Avoid_: ADR, owner question (without a record)

**Need**:
A capability that one loop asks the other loop to make.
_Avoid_: ask, blocker, dependency

**Report**:
A file in a loop's `reports/` folder that starts with the KPI block. It is a
milestone report or a short reading.
_Avoid_: update, status

**Reading**:
A report that measures each success measure on one commit of master.
_Avoid_: snapshot, yardstick reading

**KPI block**:
The front matter of a report, with fixed keys that the owner's tracker
reads. `loops/reporting.md` lists the keys.

**PLANS**:
The loop's list of open work in `PLANS.md`, with the most valuable item
first.
_Avoid_: roadmap, todo list
