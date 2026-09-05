# The confit goal

**What this document is.** The **target**: what confit is *for*, how much of it we intend to
serve, where the target's edge is drawn and why, and by which yardsticks it is measured. It
sits above the oracle spec (`packages/confit/docs/oracle/`) and the engine and testing specs
that follow it: the oracle spec defines *correct*, this document says *how much correct, over
which queries, and why the rest is out*.

**Where the distance is.** This document is the destination and holds **definitions, methods
and enforcement pointers, and no dated reading**. How close today's engine is to the
destination — every measured value, and every way the current engine falls short of what is
written here — lives in a dated report under `packages/confit/docs/reports/`; the first is
`2026-09-02-goal-baseline.md`, and a later reading is a **new dated file**, never an edit to
this one. Each shortfall is stated there as a **gap** — a divergence of the current state
from this document's target — under the same slug it would have here. So a construct that is
out of scope *by decision* is in the scope-edge section below; a construct that is out only
because it is not built is a gap in the report, and the scope-redirects table says where each
one went. Facts are still measured or read from a pin and never recalled; what changed is
that a *reading* no longer lives in prose here, which is how a bar goes a month unre-run and
nobody notices. Current tasks, tickets and defects are the implementation loop's, and the
loop reports into those files.

**Non-circularity, the same rule the oracle spec runs on.** This document is the
authority. Code, tests, pins and gate floors are its **enforcement**, never its
definition. A goal is not "whatever the gates currently pass" — if it were, every
regression would redefine the goal the moment it landed. `Enforced-by:` names the thing
that makes a statement hold; `Verified-by:` names the test, pin, gate line or dated
measurement that would catch it not holding. A statement nothing checks says `Unverified`
and says so plainly.

**How an item is named.** Nothing is named by a number. Every item carries a **slug** —
a short kebab-case noun phrase naming its *subject*, so the name still reads true when
the ruling changes. Slugs are assigned once; renaming one is a tombstone line naming
both. Moving an item to a dated report is **not** a rename and not a retirement: it keeps
its slug there. The family is carried by the citation, not by the slug:

| kind | written as | example | defined in |
|---|---|---|---|
| goal | `goal: <slug>` | goal: two-outcome-contract | here |
| exclusion-ledger row | `exclusion: <slug>` | exclusion: whole-relation-shapes | here |
| KPI | `kpi: <slug>` | kpi: acceptance-rate | here |
| claim of fact | `claim: <slug>` | claim: model-surface-split | here, or the oracle spec |
| ASK block | `ask: <slug>` | ask: acceptance-target | here |
| gap from the target | `gap: <slug>` | gap: unshipped-decimal-arithmetic | the dated report |
| finding | `finding: <slug>` | finding: seed-1804 | the dated report |

Slugs are unique across every family and across the oracle spec, and the form above is
used at the definition and at every reference, so `grep -rn "<slug>"` finds an item and
everything that cites it. A row that leaves this document for the report keeps its slug and
changes only its family, and the scope-redirects table records the pair so an old citation
still lands.

**Sections are cited the same way.** Every numbered heading carries a kebab-case slug anchor
— `## 4. What is out of scope, by decision {#scope-edge}` — and every cross-reference, here
or into the dated report, names that slug ("the scope-edge section"). The number in a heading
is reading order; nothing cites it. A heading whose subject is already a slugged item
(`### 4.1 finding: seed-1804`, in the report) is cited by that item and takes no anchor.

**Every example is executed.** The API examples below were run against the shipped wheel
(`BUILD_PROFILE == "release"`) and behave exactly as printed; refusal texts are quoted
verbatim from the raised exception. A line is labelled `SERVES` or `REFUSES` for what the
call actually did. Where the **target** requires an outcome today's engine does not produce,
the line says `target: refuses` and cites the finding that carries the mismatch — this
document may state a rule the engine has not met, but it may not print a result that did not
happen.

**Two markers keep the normative half honest.** `[PROPOSED]` — a statement this document
would like, which nobody has ruled on; it holds a slug only so a ticket can cite it.
`[FACT]` — a measured statement of current state with no decision attached. Every
decision that belongs to the owner is an ASK block or carries `[PROPOSED]`; none of them
is written as settled. That includes every acceptance-rate target, ratification of the
permanent scope set in the scope-edge section, and any change to the KPI set — the set
itself is in force in the measurement-and-kpis section, which is a ruling and not a proposal.

---

## 1. The public surface this document is about {#public-surface}

**Everything that follows is a statement about one class.** The package `__init__` exports
exactly two names, and one of them is a string:

```python
from confit import BUILD_PROFILE, DuckDBInferFn
```

`BUILD_PROFILE` is `"debug"` or `"release"` — the benchmarks refuse a debug build, so a
serving number carries the profile it was taken under. `confit.oracle` and `confit.compare`
ship in the wheel and are deliberately *outside* `__init__`: they are the **measurement**
surface — the apparatus the four-yardsticks section runs — and not the engine anyone serves
with. That boundary is part of the target, not an accident of packaging.

The engine is one constructor and six members (`packages/confit/confit/_engine.pyi`, quoted
exactly and in its order):

```python
class DuckDBInferFn:
    def __init__(
        self,
        sql: str,
        row_tables: dict[str, pa.Schema],
        static_tables: dict[str, pa.Table],
        udfs: list[Any] | None = None,
        shape: str | None = None,
    ) -> None: ...

    @property
    def shape(self) -> str: ...
    @property
    def backend(self) -> str: ...
    @property
    def boundary(self) -> str: ...
    @property
    def output_schema(self) -> pa.Schema: ...
    def infer_rows(self, rows: list[Any]) -> list[dict[str, Any]]: ...
    def infer_arrow(self, batch: pa.Table) -> pa.Table: ...
```

Every schema is arrow, so a declared width is real: `pa.int32()` binds INTEGER exactly as
DuckDB DDL does. `shape` is the output-multiplicity contract — `"map"` / `"filter"` (the
default) / `"many"` — proven at build, never checked at runtime. `output_schema` is the
output contract: the field names, arrow types and order of `infer_arrow`'s table and the
keys of every `infer_rows` dict. `backend` is `"cranelift"`, `"interpreter"` or
`"constant"`, and the last of those is the static-tables-only carve-out the scope-edge
section turns on. `boundary` is how rows cross into Python — `"marshaller"` (generated at
prepare), `"generic"` (the env-pinned baseline) or `"constant"`; it is a second axis, not a
restatement of `backend`, and both are readable state, never a mode anyone selects.

**One build, both call paths.** SERVES:

```python
ROW = pa.schema([("price", pa.float64()), ("city", pa.string())])
fn = DuckDBInferFn(
    "SELECT price * 1.2 AS gross, upper(city) AS c FROM __THIS__",
    row_tables={"__THIS__": ROW}, static_tables={}, shape="map",
)
fn.output_schema.names                          # ['gross', 'c']
fn.shape                                        # 'map'
fn.backend                                      # 'cranelift'
fn.boundary                                     # 'marshaller'
fn.infer_rows([{"price": 10.0, "city": "de"}])  # [{'gross': 12.0, 'c': 'DE'}]
fn.infer_arrow(pa.table({"price": [10.0, 20.0], "city": ["de", "fr"]})).to_pydict()
                            # {'gross': [12.0, 24.0], 'c': ['DE', 'FR']}
```

**Refusal is the constructor raising.** There is no validate step, no error mode, no
degraded result: the second outcome of goal: two-outcome-contract is a `ValueError` out of
`DuckDBInferFn(...)` naming the construct, before any row exists. REFUSES:

```python
DuckDBInferFn("SELECT sum(price) AS total FROM __THIS__",
              row_tables={"__THIS__": ROW}, static_tables={})
# ValueError: unsupported: aggregate function sum (no aggregation in v0)
```

**The `udfs=` protocol is the model surface**, and it is duck-typed: no base class, no
registration call. An object carries `name: str`, `takes: pa.Schema` (one field per
argument, names and types together, in call order), `returns: pa.DataType`, a scalar
`__call__` returning a tuple of output lanes (or `None` for an all-NULL result), and
optionally `instances`, which marks it a fitted transformer and adds an implicit leading
nullable-BIGINT instance id never written in `takes`. `returns` also declares the call's
width: `pa.struct([...])` is width-k with addressable field names, `pa.list_(t, k)` is
width-k unnamed. An object that additionally exposes `tree_tables()` is a fitted ensemble
scored by the native kernel, with no Python on the row path. SERVES:

```python
class Shout:                      # the contract is the protocol, not a class
    name = "shout"
    takes = pa.schema([("s", pa.string())])
    returns = pa.string()

    def __call__(self, s):
        return (None if s is None else s.upper(),)

fn = DuckDBInferFn("SELECT shout(city) AS c FROM __THIS__",
                   row_tables={"__THIS__": ROW}, static_tables={},
                   udfs=[Shout()], shape="map")
fn.infer_rows([{"price": 1.0, "city": "de"}])   # [{'c': 'DE'}]
```

The rest of this document says what that constructor must return, what it must refuse, and
which half of SQL sits on each side of the line.

---

## 2. What confit is for {#engine-purpose}

The repository has **two** goals, and confit is one half of one of them: ergonomic
SQL-to-transformer authoring, and fast inference. `sql_transform` owns authoring and fit;
confit owns the serving engine. This document is only about confit's half
(goal: engine-half-only).

**goal: serving-without-skew.** A feature transform written once, in SQL, produces the
same values at request time that it produced over the training table — bit for bit, not
approximately. Train/serve skew is the failure this engine exists to make impossible, and
"bit-exact" rather than "close" is the whole point: an engine that is 99% compatible does
not fail on 1% of queries, it silently corrupts a fraction of rows on queries it appears
to support.
*Enforced-by:* the two-outcome contract below, and the fit-side gates in
`packages/sql-transform` (kpi: training-round-trip).
*Verified-by:* `packages/confit/README.md:11-22` (the contract and this argument);
kpi: training-round-trip and kpi: engine-parity (the controls-in-force section).

**goal: two-outcome-contract.** For any SQL handed to `DuckDBInferFn`, exactly one of two
things happens: it serves bit-for-bit identical to the oracle, or it refuses at build with
a `ValueError` naming the construct. Nothing is approximated, silently dropped, or widened
at inference time. This is the load-bearing goal, and everything in the acceptance-frame
section follows from it.

**The two outcomes are the two things `DuckDBInferFn(...)` can do**, which is why the
contract is testable one call at a time. It returns a servable function — SERVES:

```python
fn = DuckDBInferFn(
    "SELECT t.price * r.rate AS gross FROM __THIS__ AS t "
    "LEFT JOIN rates AS r ON t.city = r.city",
    row_tables={"__THIS__": ROW},
    static_tables={"rates": pa.table({"city": ["de", "fr"], "rate": [1.1, 1.2]})},
    shape="map",
)
fn.infer_rows([{"price": 10.0, "city": "fr"}])  # [{'gross': 12.0}]
```

or it raises, naming what it could not take — REFUSES:

```python
class Width1:                          # a width-1 list return is a scalar
    name = "emb"
    takes = pa.schema([("x", pa.float64())])
    returns = pa.list_(pa.float64(), 1)

    def __call__(self, x):
        return (x,)

DuckDBInferFn("SELECT emb(price) AS e FROM __THIS__",
              row_tables={"__THIS__": ROW}, static_tables={}, udfs=[Width1()])
# ValueError: udf 'emb': a width-1 list return is a scalar — declare the element
# type rather than pa.list_(t, 1)
```

That second one is deliberately not a SQL construct: it is the caller declaring their own
UDF wrong — the class ask: exclusion-ratification (2) has no ground for, and which lands in
the same `REFUSED` bucket as everything else.

**The third mode exists and is enumerated — that is the whole difference.** The absolute
above is "no *unenumerated* third mode", not "no third mode", and writing it the strong
way would make this document false on its own evidence. The enumeration is the oracle
spec's divergence ledger
(`packages/confit/docs/oracle/07-the-divergence-ledger.md`), whose rows *serve* with a
consciously different surface. **Which rows are in force, and at what severity, is the
ledger's status column to say, never this document's** (the document-set section) — so the
ledger is pointed at here and never copied. The pointer is the structural part: a contract
that admits enumerated exceptions is only honest while the enumeration has a single home.
*Enforced-by:* build-time refusal at `DuckDBInferFn(...)` construction throughout
`packages/confit/src/specializer/`; the ledger for the enumerated exceptions.
*Verified-by:* kpi: no-third-mode (the controls-in-force section, the corpus's
FAILED bucket pinned empty); `packages/confit/tests/test_corpus_replay.py:171-190` (three
outcomes, zero FAILs); `packages/confit/docs/properties.md:231-235` (P18);
`packages/confit/docs/oracle/` claim: oracle-identity for which DuckDB.

**goal: pack-time-only-work.** All general work happens once, at pack time: parse once,
bind once, compile once, freeze the statics into the code. Nothing general remains at call
time. The ceiling this doctrine tolerates is a single n-dispatch branch; anything that
would re-do general work per row is refused rather than served slowly.
*Enforced-by:* the specializer's partial-evaluation model; refusals for every construct
that would need per-row compilation (exclusion: per-row-general-work).
*Verified-by:* `packages/confit/docs/known-limitations.md:65-79` ("the specialization
bargain"); `packages/confit/docs/properties.md:246` (P20, statics frozen at
build); `packages/confit/docs/specs/2026-08-25-task-133-join-keys-design.md:520` ("an
engine whose whole doctrine is compile-once with no runtime ...").

**goal: request-latency-budget.** Serving cost is a single-digit-microsecond,
in-process, per-request number — the regime where a model server calls a feature
transform inside its own request, not the regime where a warehouse scans a batch.
DuckDB-per-call is three orders of magnitude off that and is not a competitor for it.
*Enforced-by:* **nothing — `Unverified`.** No gate, floor or pin bounds serving latency;
`benchmarks/` prints a table and asserts a three-way *parity* gate, not a time. Read the
front matter's rule against goals-defined-by-current-measurement as binding here: a
measured gap is evidence for the regime, and a regression that moved it would not fail
anything.
*Verified-by:* the serving bench, read into the dated baseline report. **No target number
is in force** — the budget is a regime, not a bound anyone has ruled on. See
ask: kpi-set-change.

**goal: engine-half-only.** Confit's scope stops at the engine: SQL plus frozen tables
plus declared UDF objects in, a native function out. Authoring sugar, window
marginalization, transformer fitting, and parity against sklearn belong to
`packages/sql-transform` and are not confit's to own or to claim
(claim: model-surface-split).
*Verified-by:* `README.md:12-13` (the two-package split);
`packages/confit/tests/test_tree_predict.py:9-11` ("Parity against sklearn is a separate
gate that lives in sql-transform").

**goal: growing-accepted-surface.** The quantity that is actually optimized is not parity
— parity outside the enumerated divergence ledger is fixed at 100% by
goal: two-outcome-contract — it is **acceptance**: how much real SQL builds instead of
refusing. Progress is queries moving from REFUSED to served,
never into a wrong answer. The acceptance-frame section is this goal's frame and its
yardsticks.
*Enforced-by:* **two of the four-yardsticks table's rows, and only two.** The dialect
floors gate ratchets its two constants (`test_dialect_corpus_gate.py:37`,
`test_dialect_cross_engine_gate.py:50`). The corpus match count is *printed and never
asserted* — `test_corpus_replay.py:180-184` prints it, `:186` asserts only `not fails` —
and the campaign acceptance rate has no gate at all. A count records a goal; it does not
make one hold. That gap is what ask: acceptance-target and kpi: ladder-ratchet are about.
*Verified-by:* kpi: coverage-ladder (the drives-in-force section, "Progress = moving queries
from REFUSED to MARGINALIZED (never to FAILED)"); the four line reads above.

---

## 3. Parity is a control; acceptance is the goal {#acceptance-frame}

**The yardstick is the constructor returning.** Acceptance is the share of queries for which
`DuckDBInferFn(...)` hands back a function instead of raising — one call, one bit, counted
over a corpus — so every number in this section is a count of successful constructions, and
nothing in it is a judgement about the values that function then serves.

The reframe this document exists to make explicit. Because the contract is
bit-exact-or-refuse-by-name (goal: two-outcome-contract), **parity on the accepted surface
outside the divergence ledger is a control fixed at 100%**: a parity violation there is a
bug to fix, never a dial to trade. Asking "how much oracle parity do we want" has exactly
one answer, and it is not interesting. The goal-shaped question is **which queries are
accepted at all**, and how fast that set grows.

Two verdicts carry the distinction, and the oracle spec defines them
(`packages/confit/docs/oracle/04-verdicts-agreement-abstention-refusal.md`):

| verdict | means | counts as |
|---|---|---|
| `AGREE` | ours == optimizer-off DuckDB == optimizer-on DuckDB | the only thing counted as coverage (claim: coverage-accounting) |
| `REFUSED` | confit refused at build; the case never entered the comparison | **not** a finding and **not** coverage — absorbed into a histogram (claim: refusal-absorb). This is the number this section is about |

Two of eleven, and the two are not a partition. The full verdict space is
`fuzz.oracle.KINDS` (eleven), plus two the *runner* produces rather than the oracle —
`TIMEOUT` and `PANIC`, both in `fuzz.runner.INTERESTING`, both reaching `findings.jsonl`
by the same path as every other verdict. The oracle spec is explicit that any statement
about what a campaign reports has to include them
(`04-verdicts-agreement-abstention-refusal.md:27-29`), so a reading of a campaign says when
they were zero rather than leaving them out.

`REFUSED` is the shape of "not accepted", and it is the only shape the target has: the
contract in the engine-purpose section admits exactly two outcomes, so a case either built
or it refused by name, and a refusal retires by a decision to serve the construct.

**Where the rest of the vocabulary lives.** A measurement apparatus needs more words than
the target does — including what it does with a case whose answer exercises a currently-open
gap, which is classified rather than value-compared and so lands in neither the coverage
count nor the findings, and how such a case is counted. Those classifications are the
measurement layer's:
defined in the oracle spec's verdict chapter
(`packages/confit/docs/oracle/04-verdicts-agreement-abstention-refusal.md`, and
claim: unshipped-verdict in `05-the-comparison-contract.md`), and read on a date in
`packages/confit/docs/reports/2026-09-02-goal-baseline.md`, where
gap: unshipped-decimal-arithmetic carries the whole story and the acceptance-reading section
carries what today's acceptance numerator contains. This document points at both and
restates neither.

**Acceptance is defined as cases that built rather than refused** — on this definition a
parity defect counts as accepted, which is correct for a *scope* metric and is exactly the
reason acceptance can never stand in for parity.

### 3.1 The four yardsticks {#four-yardsticks}

What the accepted surface is read with. Each row is a **method**, not a number; the numbers
are the dated report's.

| yardstick | what it reads | what runs it | ratcheted? |
|---|---|---|---|
| mined-corpus replay | of the statements mined from DuckDB's own test suite, how many replay bit-exact, how many refuse cleanly, how many FAIL | `test_corpus_replay.py` | no — the match count is printed, only zero-FAIL is asserted |
| dialect L2 floor | how many mined statements survive parse-then-print invisibly to the oracle | `test_dialect_corpus_gate.py` | **yes**, an asserted floor |
| dialect L3 cross-engine floor | how many match through a second engine's dialect | `test_dialect_cross_engine_gate.py` | **yes**, an asserted floor — but only in an environment that has `pyspark`, and the fixture fails loudly rather than skipping |
| campaign verdict census | over a seed range of the generated grammar, the verdict distribution and the acceptance rate that falls out of it | `python -m fuzz.runner` (a manual CLI, **not** a standing gate) | no |

A fifth number is cited from across the package boundary: kpi: coverage-ladder (the
drives-in-force section) pins the `sql_transform` admission ladder (mined
marginalized/refused, and the curated three-way split). That ladder is `sql_transform`'s,
not confit's (goal: engine-half-only); it is cited because acceptance growth is measured on
both sides of the boundary, and it is the one ladder whose *split* has a pin test that fails
when the number moves. It takes two tests to say so —
`_corpus_test.py::test_progression_totals` pins the totals and
`::test_mined_corpus_scoreboard` pins the split — so citing the totals test alone would name
a gate that cannot catch a drift in the split.

**Campaign-validity caveat, and it bounds every acceptance number this document's methods
produce.** Those rates are over the **generated grammar**, which is not query space. The
oracle spec records the caveats directly: the campaign fuzzer is a manual CLI and *not* a
standing gate — only the regexp fuzzer is (claim: regexp-fuzz-gate's correction); there is
no coverage denominator that means anything yet (claim: coverage-denominator, `[PROPOSED]`);
a rising abstention rate would read as generator drift and is not reported
(claim: abstention-rate, `[PROPOSED]`); and the mined corpus's expected rows are an
optimizer-**on** recording with no provenance, so the mined ladder is measured against a
different reading of DuckDB than claim: oracle-identity names (claim: zero-fails-gate's
second correction). A share of a grammar is a real measurement of a synthetic population,
and nothing converts it into a statement about the SQL people write.

> ### ask: acceptance-target — is there an acceptance-rate target, and what is it?
>
> Neither the campaign acceptance rate nor the mined-corpus match count has a target, and
> neither has a ratchet. Three shapes, and the choice is yours:
>
> **(a) No target, dated reporting only.** Cheapest, and honest about the denominator
> problem: a grammar rate is not a query-space rate, so a target on it optimizes the
> generator as readily as the engine. Cost: nothing notices a slip, and an ungated ladder
> has already slipped unnoticed once.
>
> **(b) Ratchet, no target.** "Never decreases", the rule the dialect floors already run.
> Cost, and it is real: a deliberate scope *reduction* becomes a gate failure. The oracle
> spec's ask: match-count-ratchet is the same question for the mined corpus and should be
> answered with this one, not separately.
>
> **(c) A named target per surface**, e.g. "N% acceptance over the campaign grammar, by a
> named date". Cost: it needs the denominator work
> (claim: coverage-denominator) before the number means anything.
>
> My read: (b) for the corpus and dialect ladders, which have stable denominators, and (a)
> for the campaign rate until a denominator exists. Not applied — this is your call.
>
> *Context:* the current readings, and the slip, are in
> `packages/confit/docs/reports/2026-09-02-goal-baseline.md`.
>
> *Binds:* goal: growing-accepted-surface, kpi: acceptance-rate, kpi: ladder-ratchet, and
> the oracle spec's ask: match-count-ratchet.

**ask: next-query-classes is not here.** Which unbuilt class ships next is a question about
*distance*, and every candidate it ranks is a gap entry rather than a scope decision, so the
block lives in the dated report next to the ledger it ranks
(`packages/confit/docs/reports/2026-09-02-goal-baseline.md`). It still binds
goal: growing-accepted-surface, and the ASK index below records where it went.

---

## 4. What is out of scope, by decision {#scope-edge}

**This is the target's edge, not its distance.** Every row here is out of scope **by
decision**: the engine model cannot express it, or it could be served and we chose not to,
or it costs more per row than a serving engine may spend. **None of these retires** — a
permanent decision does not retire, it gets re-decided, and re-deciding one is
ask: exclusion-ratification's business, not a feature's. Anything that is out today only
because it is not built yet is *distance from this target*, and the redirects at the end of
this section say where each such row went.

**Every row in this section is `[PROPOSED]`.** The marker is stated once here rather than
six times below, and it is not decoration: the *behaviour* each row describes is in force
and measured, but the **ground assigned** and the **set's completeness** are exactly what
ask: exclusion-ratification puts to the owner. Read the declarative voice below as "this is
what the evidence says the rule is", never as "this is ruled". A reader grepping for
`[PROPOSED]` should find this section.

Each row: **what** is out of scope, **why**, what would have to happen to **re-decide** it
where that is worth stating, and **how it rings** if the decision is ever violated or
changed — or an honest flag that nothing rings.

Three grounds classify every exclusion, and they are already decided:
**specialization-inherent** (the engine model cannot express it — permanent),
**scope-by-product-decision** (it could be served and we chose not to — reversible by a
decision), **resource** (it would cost more than a serving engine may spend per row — a
judgement with a number attached). See `packages/confit/docs/oracle/`
claim: refusal-grounds.

**Not in this ledger, deliberately:** the rows of `known-limitations.md:282-346` ("deliberate
contract choices") that *serve* with a consciously different surface (duplicate-column
rename, approximate error texts, schema-qualifier resolution, the platform-libm NaN bit
pattern). Those are divergences, not exclusions, and they belong to the oracle spec's
divergence ledger (`packages/confit/docs/oracle/07-the-divergence-ledger.md`). Entries below
whose *decision* is an exclusion even though the *symptom* is a divergence say so and point at
the row.

**exclusion: whole-relation-shapes.** `GROUP BY`/`HAVING`/aggregates, `ORDER BY`,
`LIMIT`/`OFFSET`/`FETCH`/`TOP`, `DISTINCT`, CTEs, `UNION`/`INTERSECT`/`EXCEPT`, subqueries,
multiple statements, table functions in `FROM`, `rowid`, `FULL OUTER JOIN`, **`QUALIFY`**
(`frontend.rs:284`), **named window definitions** (`WINDOW`, `:314`) and **window/aggregate
modifiers on a scalar call** — `OVER`, `FILTER`, `IGNORE NULLS`, `WITHIN GROUP`
(`:6289-6299`). The last three are not small: `QUALIFY` and the modifier class are both
constructs the campaign reaches often, and a modifier on a scalar call is the shape that
gets *dropped* rather than refused if nobody is watching for it. How often, and where they
rank, is a reading and the report's.
*Why:* scope-by-product-decision. Their output shape is not one-row-in / 0..N-out, so they
are not row-at-a-time feature transforms.
*The carve-out, which is part of the decision and not an exception to it:* a
**static-tables-only** query is evaluated once by DuckDB at build and frozen, so aggregation
and `ORDER BY` serve there. Its edge is the same decision applied twice — **what a
whole-relation construct selects is frozen only when it is a function of the query.** A row
limit is not (measured: four different answers across twelve connections) and refuses,
and neither is any other selection by position; a tie-producing `ORDER BY` is not either
(five sequences under five settings a build machine picks for itself,
`known-limitations.md:132-143`), and freezing one would let two builds of the same function
disagree — goal: serving-without-skew's failure in its build-to-build face rather than its
train-to-serve one. **Both must refuse**, and both now do: the tie half was
finding: static-only-tie-order in the dated report, and closing it is what makes this
sentence a statement rather than a target.
*Re-decided by:* nothing intended — the output-shape argument would have to change first.
*Rings:* nothing rings — there is no trigger. Lifting one breaks
`packages/confit/tests/test_known_limitations.py` — which is the executable twin of
`known-limitations.md`, **not of this row**. Its docstring says so (`:1-7`, "Section
numbers mirror the document"), and its whole-relation parameterization (`:98-117`) covers
nine of the constructs enumerated above and not the rest. The oracle spec's
claim: doc-twin-totality measured that gap across five other sites and
ask: doc-twin-overstatement is open on it; this document does not become a sixth site.
*At the constructor.* Over `ROW` — REFUSES, five shapes, five names:

```python
def build(sql, **kw):
    return DuckDBInferFn(sql, row_tables={"__THIS__": ROW}, static_tables={}, **kw)

build("SELECT sum(price) AS total FROM __THIS__")
# ValueError: unsupported: aggregate function sum (no aggregation in v0)
build("SELECT price FROM __THIS__ ORDER BY price")
# ValueError: unsupported: ORDER BY
build("SELECT DISTINCT city FROM __THIS__")
# ValueError: unsupported: DISTINCT
build("SELECT rank() OVER (ORDER BY price) AS r FROM __THIS__")
# ValueError: unsupported: modifier on scalar call rank (FILTER, OVER, IGNORE
# NULLS and WITHIN GROUP apply to aggregates and window functions, which this
# engine does not serve)
build("SELECT price FROM __THIS__ QUALIFY price > 1")
# ValueError: unsupported: QUALIFY
```

The same shapes over a **frozen static** are a different query: nothing dynamic remains, so
DuckDB evaluates them once at build and the result is the function. SERVES, and
`backend == "constant"` is how you can tell:

```python
S = pa.table({"v": pa.array([1, 2, 3], pa.int64())})
fn = DuckDBInferFn("SELECT max(v) AS top FROM s",
                   row_tables={"__THIS__": ROW}, static_tables={"s": S})
fn.backend, fn.infer_rows([])   # ('constant', [{'top': 3}])

fn = DuckDBInferFn("SELECT v AS o FROM s ORDER BY v DESC",
                   row_tables={"__THIS__": ROW}, static_tables={"s": S})
fn.backend, fn.infer_rows([])   # ('constant', [{'o': 3}, {'o': 2}, {'o': 1}])
```

**The not-a-function-of-the-query rule, shown.** A row limit picks *which* rows survive, and
that pick is not in the query — so it refuses even where the aggregation it sits on serves.
REFUSES:

```python
DuckDBInferFn("SELECT v AS o FROM s ORDER BY v LIMIT 1",
              row_tables={"__THIS__": ROW}, static_tables={"s": S})
# ValueError: unsupported: row limit (LIMIT/OFFSET) on a
#             static-tables-only query -- which rows survive
#             depends on scan order, not the query
```

The same rule reaches a tie: two groups with equal sort keys have no order in the query
either, so freezing whichever one this build's DuckDB run produced would let two builds of
the same function disagree. REFUSES — the ties are read off the frozen result by DuckDB
itself at build, so an `ORDER BY` whose keys separate every row still serves:

```python
TIES = pa.table({"g": ["x", "y", "z"], "v": pa.array([1, 1, 2], pa.int64())})
DuckDBInferFn("SELECT g AS o, min(v) AS t FROM ties GROUP BY g ORDER BY t",
              row_tables={"__THIS__": ROW}, static_tables={"ties": TIES})
# ValueError: unsupported: tie-producing ORDER BY on a static-tables-only
#             query -- which of the tied rows comes first depends on scan
#             order, not the query

fn = DuckDBInferFn("SELECT g AS o, v AS t FROM ties GROUP BY g, v ORDER BY g",
                   row_tables={"__THIS__": ROW}, static_tables={"ties": TIES})
fn.backend, [r["o"] for r in fn.infer_rows([])]   # ('constant', ['x', 'y', 'z'])
```

A tie in the ORDER BY is one instance of the rule, not the whole of it. Anything else that
lets scan order, thread scheduling or a random draw pick among valid answers refuses the
same way and by name — a function whose value is a draw or a clock, an aggregate whose
answer follows the arrival order, a window frame counted in rows rather than in key peers.
REFUSES:

```python
DuckDBInferFn("SELECT g AS o, v AS t FROM ties ORDER BY random()",
              row_tables={"__THIS__": ROW}, static_tables={"ties": TIES})
# ValueError: unsupported: the non-deterministic function random() on a
#             static-tables-only query -- its value is drawn when the query
#             runs, not fixed by the query

DuckDBInferFn("SELECT list(g) AS o FROM ties",
              row_tables={"__THIS__": ROW}, static_tables={"ties": TIES})
# ValueError: unsupported: order-sensitive aggregate list on a
#             static-tables-only query -- its answer follows scan order, and
#             an ORDER BY inside the aggregate is not read as a fix

fn = DuckDBInferFn("SELECT g AS o, max(v) OVER (ORDER BY v) AS w FROM ties ORDER BY g",
                   row_tables={"__THIS__": ROW}, static_tables={"ties": TIES})
fn.backend, [r["w"] for r in fn.infer_rows([])]   # ('constant', [1, 1, 2])
```

The classification is DuckDB's own, not a list kept here: `duckdb_functions().stability`
says which functions are a draw or a clock, and the aggregate source's
`SetOrderDependent` says which aggregates follow the scan — everything DuckDB does not
opt out of is refused, so `count`/`min`/`max`/`median` serve and `sum`/`avg`/`first`/`list`
do not.

*Verified-by:* `packages/confit/docs/known-limitations.md:95-229`;
`packages/confit/tests/test_known_limitations.py:1-7, :98-117`;
`packages/confit/tests/test_arrow_schema_api.py:604-630` (the row-limit refusal);
`packages/confit/tests/test_static_only_order.py` (the tie refusal, and the ORDER BY
shapes that still serve).

**exclusion: per-row-general-work.** Non-constant regex patterns, replacement strings,
regex options and extract-group indexes; anything that would compile or bind per row.
*Why:* specialization-inherent, and the direct negation of goal: pack-time-only-work.
DuckDB compiles regexes per row; we compile at prepare.
*Re-decided by:* a decision to abandon compile-once for this construct — which contradicts
goal: pack-time-only-work, so it is a goal change before it is a scope change.
*Rings:* nothing rings.
*At the constructor.* The line is where the pattern comes from, not what it says:

```python
PAT = pa.schema([("s", pa.string()), ("p", pa.string())])

DuckDBInferFn("SELECT regexp_matches(s, p) AS m FROM __THIS__",
              row_tables={"__THIS__": PAT}, static_tables={})     # REFUSES
# ValueError: unsupported: non-constant regex pattern (compiled at prepare in v0)

fn = DuckDBInferFn("SELECT regexp_matches(s, '^a.c$') AS m FROM __THIS__",
                   row_tables={"__THIS__": PAT}, static_tables={},
                   shape="map")                                   # SERVES
fn.infer_rows([{"s": "abc", "p": "x"}])   # [{'m': True}]
```

*Verified-by:* `packages/confit/docs/known-limitations.md:74-75`.

**exclusion: resource-ceilings.** Pad/repeat counts past a 1 GiB string-builder budget (a
literal count refuses at build; a data-driven count traps at runtime), the regex
program-size guard, and the Arrow batch ceiling.
*Why:* **resource** — no gigabyte allocations in a serving engine, by decision. This is
the ground whose whole point is that the accepted cost has a number attached.
*Re-decided by:* a review that raises the number. A ceiling moves by a decision, never by a
feature landing.
*Rings:* **for the string budget only.**
`known_divergences/test_string_budget.py::test_a_budget_breaking_literal_count_refuses`
(`:145-146`) asserts the refusal with `pytest.raises(ValueError, match="builder|GiB")`, so
that ceiling rings on a *change*. The **Arrow batch ceiling rings nothing**: no test touches
it, and its only mention in `test_arrow_boundary.py` is a prose comment (`:33-35`). A
`Verified-by` pointing at a comment names nothing that would fail, which is this document's
own definition of not-verified — so: **`Unverified` for the Arrow half.** This row is a live
instance of ask: exclusion-ratification (1), inherited from the same pointer in
`oracle/07-the-divergence-ledger.md:147`.
*At the constructor.* The number is in the message, which is what a ceiling with a decision
behind it looks like:

```python
SROW = pa.schema([("s", pa.string())])

DuckDBInferFn("SELECT lpad(s, 2000000000, 'x') AS o FROM __THIS__",
              row_tables={"__THIS__": SROW}, static_tables={})     # REFUSES
# ValueError: bind error: lpad count 2000000000 exceeds the 1 GiB string-builder
# budget — this engine will not allocate a gigabyte per row, so the result could
# never serve. DuckDB does serve it; refusing at build is our deliberate limit,
# not a DuckDB restriction

fn = DuckDBInferFn("SELECT lpad(s, 8, 'x') AS o FROM __THIS__",
                   row_tables={"__THIS__": SROW}, static_tables={},
                   shape="map")                                    # SERVES
fn.infer_rows([{"s": "ab"}])   # [{'o': 'xxxxxxab'}]
```

*Verified-by:* `packages/confit/docs/known-limitations.md:278`;
`packages/confit/tests/known_divergences/test_string_budget.py:145-146`;
`packages/confit/docs/oracle/` divergence: string-builder-budget,
divergence: arrow-batch-ceiling.

**exclusion: optimizer-on-answers.** We do not reproduce DuckDB's 33 plan-rewrite passes.
The user-visible cost: a trapping subexpression the optimizer would delete, we still
evaluate — so a query that returns a value in the reader's own DuckDB session can raise
here. Overlaps the divergence ledger; the *decision* is an exclusion, the *symptom* is
divergence: trap-elision.
*Why:* scope-by-product-decision, on a measured ground. The optimizer-on reading is not a
function of the query — `statistics_propagation` answers from a column's stored null
statistic, so the same query over the same rows differs by the table's insert history. A
target you cannot compute from the query is not a target, and confit compiles against a
schema and never sees a table.
*Decided:* 2026-08-17; no reversal intended.
*Rings:* **yes** — the campaign reads DuckDB twice per case and labels these `DIVERGE_OPT`,
so the class is counted rather than absorbed. It is a **reported finding, not an accepted
class** (the oracle spec's claim: contract-surface-gap), so its standing count is a cost the
report carries, never a bucket that quietly grows.
*At the constructor.* This one **SERVES** — that is the whole point of the row: the cost is
paid at call time, as a trap, never as a different value:

```python
fn = DuckDBInferFn("SELECT (i + 1) > 5 AS o FROM __THIS__",
                   row_tables={"__THIS__": pa.schema([("i", pa.int32())])},
                   static_tables={}, shape="map")
fn.backend                        # 'cranelift' — it built
fn.infer_rows([{"i": 1}])         # [{'o': False}]
fn.infer_rows([{"i": 2147483647}])
# ValueError: Out of Range Error: value out of range for INTEGER (arrow int32)
```

The reader's own DuckDB answers `[(True,)]` for that row, because
`expression_rewriter` turns `(i + 1) > 5` into `i > 4` and the addition never runs;
`PRAGMA disable_optimizer` reproduces confit's trap
(`Out of Range Error: Overflow in addition of INT32 (2147483647 + 1)!`).
*Verified-by:* `packages/confit/docs/known-limitations.md:20-39`, `:304-330`;
`packages/confit/docs/oracle/` claim: oracle-identity, claim: optimizer-bracket.

**exclusion: statistics-dependent-kernels.** Behaviors that depend on column *statistics*
— ILIKE's NUL handling selects a different kernel depending on sibling rows — are excluded
from the corpus by name, and the engine takes the ASCII-kernel (NUL-transparent) behavior.
*Why:* specialization-inherent. A row-at-a-time engine cannot reproduce statistics-
dependent semantics even in principle — permanent by construction, with nothing to
re-decide.
*Rings:* nothing rings, by design — the exclusion is a named source list, and a named list
does not fire.
*At the constructor.* Nothing refuses — the engine SERVES one of the two behaviours and
always the same one, which is the only thing a row-at-a-time engine can promise:

```python
fn = DuckDBInferFn("SELECT s ILIKE 'A\x00B' AS m FROM __THIS__",
                   row_tables={"__THIS__": SROW}, static_tables={}, shape="map")
fn.infer_rows([{"s": "a\x00b"}])   # [{'m': True}] — NUL-transparent, always
```

DuckDB answers `True` here only when the column's statistics are pure-ASCII; a single
non-ASCII sibling row selects its generic kernel, whose fold NUL-truncates, and the same
row answers `False`. There is no sibling row at this API — `infer_rows` sees one row's
values — so the statistic is unreachable by construction.
*Verified-by:* `packages/confit/docs/known-limitations.md:298-303`;
`packages/confit/tests/test_corpus_replay.py:150-151` (`_KNOWN_DIVERGENT_SOURCES`);
`packages/confit/docs/oracle/` claim: statistics-dependent-exclusion.

**exclusion: multiplicity-by-default.** Multiplicity never enters a serving path by default.
Duplicate-key joins, cross joins and inequality/constant `ON` joins build **only** under an
explicit `shape='many'`; `shape='map'` rejects anything that can drop a row.
*Why:* scope-by-product-decision — `map` is a build-time *proof* of exactly-one, not a
runtime check, and a shape that can multiply rows is something the caller asks for by name
or does not get.
*Re-decided by:* nothing intended — defaulting to multiplicity is the failure this row
exists to prevent.
*Rings:* the shape contract is pinned in
`packages/confit/tests/test_shape_contract.py`.
*At the constructor.* One query, one static table, three values of `shape` — the multiplicity
is in the data, and the keyword is the caller asking for it by name:

```python
DUP = pa.table({"city": ["de", "de"], "rate": [1.1, 1.3]})
J = ("SELECT t.price * r.rate AS gross FROM __THIS__ AS t "
     "JOIN dup AS r ON t.city = r.city")

DuckDBInferFn(J, row_tables={"__THIS__": ROW}, static_tables={"dup": DUP})
# REFUSES  ValueError: static data mismatch: @0: duplicate map key

DuckDBInferFn(J, row_tables={"__THIS__": ROW}, static_tables={"dup": DUP},
              shape="map")
# REFUSES  ValueError: shape='map': INNER JOIN 'dup' drops rows on a key miss
#          (use LEFT JOIN)

fn = DuckDBInferFn(J, row_tables={"__THIS__": ROW}, static_tables={"dup": DUP},
                   shape="many")                                    # SERVES
fn.infer_rows([{"price": 10.0, "city": "de"}])
# [{'gross': 11.0}, {'gross': 13.0}]   — one row in, two out, by request
```

The `map` refusal is the second half of the same decision: `map` is a build-time *proof* of
exactly-one, so it rejects the row-dropping direction as flatly as the default rejects the
row-multiplying one.
*Scope note:* the *composition* limits inside `'many'` — one join per query, and
`USING`/`NATURAL` self-joins refusing under every shape — are **not** this decision. They
are unbuilt work, and they are gap: join-composition-limits in the report.
*Verified-by:* `packages/confit/docs/known-limitations.md:76-93`;
`packages/confit/tests/test_shape_contract.py`.

### 4.1 Rows that left this section {#scope-redirects}

Each of these was here because something is **not built yet**, which is distance from the
target rather than the target's edge. They are gap entries in the dated report, under the
same slug, and a citation of the old name lands here:

| was | is now | carries |
|---|---|---|
| exclusion: unshipped-decimal-arithmetic | gap: unshipped-decimal-arithmetic | DECIMAL expressions, the served literal width, and how the campaign counts them |
| exclusion: wide-integer-lanes | gap: wide-integer-lanes | HUGEINT, the unsigned family, `float32`, the narrow-lane traps |
| exclusion: non-scalar-values | gap: non-scalar-values | lists, whole structs, bracket access, BLOB, `decimal256` |
| exclusion: parse-divergence-guards | gap: parse-divergence-guards | `^`, prefix `~`, `#`, `NOT GLOB`, the regex reject list |
| part of exclusion: multiplicity-by-default | gap: join-composition-limits | one join per query, `USING`/`NATURAL` self-joins |

The rule that produced this split is the front matter's: a construct we **chose** not to
serve is scope; a construct we have **not got to** is a gap. The refusal a user sees today is
identical either way — which is exactly why the two had to stop sharing a section.

> ### ask: exclusion-ratification — does this become the permanent scope set?
>
> Six rows above, all `[PROPOSED]`, assembled from `known-limitations.md`, the engine's
> refusal sites and the campaign's refusal histogram. Ratifying them binds two things: the
> **ground** assigned to each (specialization-inherent vs scope-by-product-decision vs
> resource, which is what makes "we refuse" auditable), and that each is a **decision rather
> than a delay** — nothing here is waiting on work, so nothing here retires.
>
> Two specific pieces I would like ruled on rather than assumed:
>
> **(1) Is silence acceptable where the decision is permanent?** Counted from the `Rings:`
> lines it is four of six: exclusion: whole-relation-shapes, per-row-general-work and
> statistics-dependent-kernels ring nothing at all, and exclusion: resource-ceilings rings for
> its string budget while its **Arrow half rings nothing** and is `Unverified`. Silence is
> defensible where the condition is "never" — nothing is waiting to fire. The Arrow half is
> the exception and is a real hole: a ceiling with a number and no check is a number nobody
> is holding. Ratify the silence, or ask for the Arrow test.
>
> **(2) Do the three grounds cover what actually refuses?** They classify *scope* decisions,
> and the campaign holds a class that is not one: a UDF whose declared return shape is
> wrong (`src/duckdb/mod.rs:606-609`) is neither specialization-inherent, nor
> scope-by-product-decision, nor resource — the **caller declared their UDF wrong**. Adding a
> message prefix would not classify it, because the taxonomy has no slot for
> caller-declaration errors. Such refusals nevertheless sit in the same `REFUSED` bucket that
> feeds the acceptance rate. Either a fourth ground, or a rule that they are excluded from
> the acceptance denominator — both are your call, and the second is the one
> kpi: acceptance-rate depends on.
>
> **What this ask no longer carries.** Whether the enumeration *covers* what the engine
> refuses is a question about today's code, not about the target: the refusal-site census,
> the undocumented classes and the missing site-to-row mapping are the report's open
> questions now. They still want an answer; they are just not answered by ratifying a
> destination.
>
> *Context:* the class sizes and the refusal-site census are in
> `packages/confit/docs/reports/2026-09-02-goal-baseline.md`.
>
> *Binds:* every `exclusion:` slug in this section, kpi: acceptance-rate, and
> `packages/confit/docs/known-limitations.md` as their source.

---

## 5. What we cover that DuckDB does not {#beyond-duckdb}

The model surface. DuckDB is the oracle for SQL; it has no opinion at all about a fitted
sklearn transformer or a gradient-boosted tree, so on this surface there is no differential
oracle and a different reference takes its place.

**claim: model-surface-split.** **[FACT]** The surface is owned by two packages and the
line is drawn at the artifact, not at the algorithm. **Confit owns**: the `udfs=` protocol
(a declared object with `name` / `takes` / `returns` / optional `instances` and a scalar
`__call__`), the extern call machinery, the STRUCT-valued return and its field access, and
the native tree kernel — a UDF exposing `tree_tables()` is scored by native code from a
pair of Arrow tables plus a grid, with no sklearn import anywhere in the package.
**sql-transform owns**: fitting, clone-per-group semantics, the packing of an sklearn
estimator into those tables, and **parity against sklearn**.
*At the constructor.* A fitted transform is an object in `udfs=` and a call in the SQL —
`instances` is the whole of what marks it fitted, and the instance id is a static column
joined in like any other. SERVES:

```python
class Scale:                     # a transformer: instances -> implicit leading id
    name = "scale"
    takes = pa.schema([("x", pa.float64())])
    returns = pa.float64()
    instances = {0: 10.0, 1: 100.0}

    def __call__(self, iid, x):
        if iid is None or x is None:
            return None
        return (x * self.instances[iid],)

PARAMS = pa.table({"city": ["de", "fr"], "est": pa.array([0, 1], pa.int64())})
fn = DuckDBInferFn(
    "SELECT scale(p.est, t.price) AS y FROM __THIS__ AS t "
    "LEFT JOIN params AS p ON t.city = p.city",
    row_tables={"__THIS__": ROW}, static_tables={"params": PARAMS},
    udfs=[Scale()], shape="map",
)
fn.infer_rows([{"price": 2.0, "city": "fr"}])   # [{'y': 200.0}]
fn.infer_rows([{"price": 2.0, "city": "de"}])   # [{'y': 20.0}]
```

Width-k is the same object with a wider `returns`, and a struct makes the lanes
addressable off **one** evaluation. SERVES:

```python
class Emb2:
    name = "emb"
    takes = pa.schema([("x", pa.float64())])
    returns = pa.struct([("a", pa.float64()), ("b", pa.float64())])

    def __call__(self, x):
        return (x + 1.0, x - 1.0)

fn = DuckDBInferFn("SELECT (emb(price)).a AS a, (emb(price)).b AS b FROM __THIS__",
                   row_tables={"__THIS__": ROW}, static_tables={},
                   udfs=[Emb2()], shape="map")
fn.infer_rows([{"price": 2.0, "city": "de"}])   # [{'a': 3.0, 'b': 1.0}]
```

A tree ensemble is that same protocol with one addition — `tree_tables()` returning
`(nodes, models, compare_grid)` — after which the engine scores it natively and never calls
`__call__` at all. Which is the sense in which the line is drawn at the artifact: confit
takes Arrow tables, and whoever packed them owns sklearn.
*Verified-by:* `packages/confit/tests/test_tree_predict.py:1-17` — "Nothing here imports
sklearn ... DuckDB has no native tree scoring, so there is no differential oracle here ...
Parity against sklearn is a separate gate that lives in sql-transform";
`packages/confit/tests/test_udfs.py:1-10` ("this package's contract is the protocol, not
the class").

**claim: udf-parity-is-still-the-oracle.** Where a UDF *can* be registered with DuckDB
(`con.create_function`), the contract does not weaken: confit serves bit-for-bit identical
to DuckDB **with the same UDFs registered**, or refuses. The UDF surface is the ordinary
contract with one parameter, not an exemption from it — which is why a UDF named after a
builtin is refused rather than resolved: DuckDB lets a registered function shadow its
builtin and we do not, so serving it would be two engines answering one SQL differently.
REFUSES, and the message says which two engines would disagree:

```python
class Round:
    name = "round"
    takes = pa.schema([("x", pa.float64())])
    returns = pa.float64()

    def __call__(self, x):
        return (x,)

DuckDBInferFn("SELECT round(price) AS p FROM __THIS__",
              row_tables={"__THIS__": ROW}, static_tables={}, udfs=[Round()])
# ValueError: udf 'round' collides with the builtin function 'round' — rename it.
# The builtin binds first here, while DuckDB binds the udf, so the two engines
# would answer differently.
```

*Enforced-by:* `packages/confit/tests/test_udfs.py::udf_check` (`:240`), the parameterized
form of the contract.
*Verified-by:* kpi: engine-parity's `Enforced-by:` line (the controls-in-force section),
which names `test_udfs.py` (`udf_check`) among its enforcing suites — its "Which DuckDB" line
names no suite; `test_udfs.py::test_a_udf_may_not_take_a_builtin_name`.

**claim: sklearn-is-the-reference.** On the surface DuckDB cannot run, an independent
sklearn reference plays the role optimizer-off DuckDB plays for SQL — and it is a
**reference**, not the oracle, which is why it comes with a named bound instead of bit
equality everywhere. The bound differs by family, and **there are three bounds in force,
not one** — naming only the loosest would be the quiet loosening the standing-law section
forbids:

| surface | bound | where |
|---|---|---|
| tree scoring | **bit-exact** against `sklearn.predict` | `_trees_test.py::test_matches_sklearn_bit_exactly` |
| bare `StandardScaler` transformer columns | **`rtol=1e-12`** | `_transformers_test.py:67, :81, :204, :271` |
| `Pipeline([StandardScaler, PCA])` columns | **`rtol=1e-9`** | `_transformers_test.py:109, :127` |
| the campaign's sklearn metamorphic leg | **absolute `1e-9`**, not a relative tolerance | `fuzz/oracle.py:845` — `abs(o - p) > 1e-9` |

Four of the seven `assert_allclose` sites in the transformer file are the tighter `1e-12`
and two are the pipeline's `1e-9`; the campaign's `1e-9` shares a numeral with the
transformer bound and not a meaning. The seventh (`_transformers_test.py:166`) is not a
bound in force here and is not one of the rows above: it compares a SQL window expression,
not a transformer column, against a numpy mean, and it passes no `rtol` at all — so it runs
at numpy's default `1e-7`, looser than every bound named above.
*Enforced-by:* the four rows above;
`packages/sql-transform/sql_transform/_transformers_test.py::_reference` (`:34`) is the
clone-per-group reference all the transformer rows compare against.
*Verified-by:* kpi: transformer-parity (the controls-in-force section, the control this is —
note its own text names **no** tolerance, so the bounds above are read from the tests, not
from the KPI); `packages/confit/docs/oracle/` claim: metamorphic-self-legs (the `1e-9` leg,
itself recorded as having **no test of its own** — `Unverified`).
*Note:* a fourth bound, for natively implemented transform families, is written but **not in
force** — it is written against work that has not landed, which makes it a gap rather than a
reference bound. It is gap: native-transform-families in the dated report, and adopting it is
the owner's through ask: kpi-set-change.

---

## 6. Measurement and KPIs {#measurement-and-kpis}

**The KPI set is in force here.** It lived in `packages/confit/docs/kpis.md` until the
owner ruled that this document owns it (ask: kpis-absorb-or-defer, the document-set
section); that file is deleted and its definitions, enforcing-suite pointers and standing law
are below, under slugs. What did **not** move is its dated readings — bench tables and ladder
counts are a report's under the front matter's rule, and the current ones are in
`packages/confit/docs/reports/2026-09-02-goal-baseline.md`.

Companion: `packages/confit/docs/properties.md` — KPIs measure; properties state what must
remain true.

**The old codes, kept as pointers**, because merged PRs, backlog tickets, drafts and the
oracle spec cite them: C1 = kpi: training-round-trip, C2 = kpi: engine-parity,
C3 = kpi: binding-parity, C4 = kpi: transformer-parity, C5 = kpi: no-third-mode,
D1 = kpi: coverage-ladder, D2 = kpi: serving-latency. The slug is the name; the code is a
pointer and nothing is named by it.

### 6.1 Two kinds, and the standing law {#standing-law}

Two kinds, optimized in opposite directions:

- **Control KPIs** are invariants. Their target is a fixed point; they are never
  "improved", only *defended* — adversarial work (fuzz, differentials, refusal pins) exists
  to hunt counterexamples. A control that can only be held at 99% is not a control, it is an
  unacknowledged trade-off.
- **Drive KPIs** are the ones actually optimized: coverage up, latency down. A drive gain
  only counts if it lands inside every control — the pins make the difference impossible to
  smuggle past.

**Loosening a control is a design decision, never a fix.** The only legitimate way a
control moves: explicitly, in a spec or draft, with the new bound named and reviewed — the
precedent being a *declared* per-family ulp tolerance written down before the code that
needs it, never a bar relaxed by a failing test. And the rule that governs every trade below:

> Never trade a control for a drive gain. If a bar seems in the way, the move is a
> written, named tolerance — or a refusal.

**Three of the six candidates in the proposed-kpis section are bars, not drives, and one of
them lands on kpi: no-third-mode.** Saying otherwise here would be an unforced error in the
one paragraph whose job is to show the standing law is respected, so it is said plainly
instead: kpi: named-refusal-share is proposed as a **control at 100%** over refusal message
quality, and kpi: no-third-mode's own text is "refuses at construction with an error naming the
construct" — so it is a second bar over that clause, over a property that demonstrably does not
hold at 100% today, and the two-kind rule above calls a control adopted below its own bar "not
a control, it is an unacknowledged trade-off". kpi: ladder-ratchet is a never-decreases rule
and kpi: bench-refresh-cadence a staleness bound; neither is a drive. **None of the six may be
adopted in a form that weakens the seven in force**, and the sequence that respects the law for
the one that touches kpi: no-third-mode is spelled out in its own entry: drive first, control
only after the gap is closed. Nothing in the proposed-kpis section is adopted; ask:
kpi-set-change is the door.

### 6.2 The controls in force (5) {#controls-in-force}

**kpi: training-round-trip.** `fit(train)` + serving, applied to the training set, is
**bit-exact** equal to running the original SQL with `__THIS__` = train (both at
`SET threads = 1` — DuckDB's parallel window aggregation is not bit-deterministic for
floats). Transformer columns are exempt from the *DuckDB* oracle (DuckDB cannot run them)
and gate against kpi: transformer-parity instead.
*Enforced-by:* `packages/sql-transform/sql_transform/_projection_test.py` (`gate()` across
the admitted surface, plus the seeded differential fuzz — `MARGINALIZE_FUZZ_N`, seed
20260729; 1,500-2,000-case runs at each widening loop).

**kpi: engine-parity.** Confit serves **bit-for-bit identical to DuckDB** — with the same
declared UDFs registered via `create_function`, when there are any — **or refuses at build
with a named error**. No third behavior.
*Which DuckDB:* the optimizer-off reading (`PRAGMA disable_optimizer`), decided 2026-08-17.
The ground and the user-visible cost are exclusion: optimizer-on-answers; the oracle spec's
claim: oracle-identity is the authority on the oracle's identity, and this entry does not
restate it.
*Enforced-by:* `packages/confit/tests/test_duckdb_*.py` (the wave suites),
`test_params_joins.py`, `test_udfs.py` (`udf_check` — the parameterized form of the
contract), `packages/confit/docs/known-limitations.md` (each refusal has an executable
twin). Internal sub-invariant: cranelift ≡ interpreter, byte-for-byte —
`packages/confit/src/specializer/exec/tests.rs` (500-seed random-IR differential; shared
helper code is the structural argument).

**kpi: binding-parity.** `infer` / `infer_batch` (Confit row path) equals `transform`
(DuckDB batch path) **value-for-value** on the same fitted artifact — one artifact
(serving_sql + params tables + UDF objects), two bindings, no divergence.
*Enforced-by:* `packages/sql-transform/sql_transform/_serving_test.py` (`serve_gate()`
across aggregates, transformers width-1/width-k, author UDFs, chains, unseen-group NULLs;
dict rows ≡ model rows).

**kpi: transformer-parity.** Transformer columns equal an **independent clone-per-group
sklearn reference** (fit and apply re-derived from scratch in the test, not via the library
code under test). The entry names no tolerance of its own; the three bounds actually in
force are read from the tests, in claim: sklearn-is-the-reference.
*Enforced-by:* `packages/sql-transform/sql_transform/_transformers_test.py`
(`_reference()`).
*Not in force:* an extension of this control to natively implemented transform families is
written against work that has not landed, so it is a gap and not a bound —
gap: native-transform-families in the dated report carries it.

**kpi: no-third-mode.** Every query either **serves** (under the four controls above) or
**refuses at construction with an error naming the construct**. Silent wrongness — a query
that builds and quietly computes something else — is the one unrecoverable state; the
corpus's FAILED bucket is pinned empty.
*Enforced-by:* `packages/sql-transform/sql_transform/_corpus_test.py` (three outcomes:
MARGINALIZED / REFUSED / FAILED-must-be-empty), the refusal tables in every test module,
`packages/confit/docs/known-limitations.md`.

### 6.3 The drives in force (2) {#drives-in-force}

**kpi: coverage-ladder.** How much of projection-SQL the marginalizer admits: the mined
scoreboard (queries lifted verbatim from DuckDB's own window test suite, with provenance)
and the curated corpus's three-way split. Pinned in
`packages/sql-transform/sql_transform/_corpus_test.py::test_progression_totals` ("the
metric, in one place — edit these pins when a loop widens support") for the totals, and in
`::test_mined_corpus_scoreboard` for the mined split — it takes both tests, so citing the
totals alone would name a gate that cannot catch a drift in the split.
Progress = moving queries from REFUSED to MARGINALIZED (never to FAILED) and growing the
corpus. The named headroom is unbuilt work and therefore a gap, not a definition:
gap: admission-ladder-headroom in the dated report lists it in order of value.
*Method for widening it:* add the query to the corpus first, watch it refuse, then implement
until it marginalizes — and extend kpi: training-round-trip's gate to the new family in the
same loop. Update the pins deliberately.
*Scope, stated because it is not confit's:* this ladder is `sql_transform`'s
(goal: engine-half-only); the four-yardsticks section says why it is cited from here.
*Current reading:* `packages/confit/docs/reports/2026-09-02-goal-baseline.md`, the
authoring-side-ladder section.

**kpi: serving-latency.** Row-at-a-time serving cost on the wide-table scenarios in
`benchmarks/`. Two harnesses: `bench_serving.py` (pure-SQL path) and `bench_transforms.py`
(the transformer/UDF path). Absolute numbers drift with machine load between runs —
**compare WITHIN a run**; the honest cross-run metric is a ratio to a baseline row measured
in the same run, which the transformer path has (a 2-field query against a 1-field one) and
the pure-SQL table does not.
*Method for moving it:* measure first (`benchmarks/`), then swap entries behind the extern
slots — the five controls are the safety net; if an optimization needs a control loosened,
that is a draft + review, not a code change. Which lever is worth pulling is a reading and
not a definition: the measured cost split, and the ranked levers that fall out of it, are
gap: native-transform-families in the dated report.
*No target number is in force* — goal: request-latency-budget is a regime, `Unverified` by
construction, and no gate, floor or pin bounds serving latency.
*Current reading:* `packages/confit/docs/reports/2026-09-02-goal-baseline.md`, the
latency-reading section, which also carries gap: bench-baseline-flip.

### 6.4 What enforces them {#enforcing-suites}

**claim: kpi-pointers-resolve.** Every `Enforced-by:` pointer in the controls-in-force
section resolves — the check itself is a dated reading and lives in
`packages/confit/docs/reports/2026-09-02-goal-baseline.md`, the enforcement-as-read section,
which carries the map suite by suite.

**Four of the five controls, and the coverage-ladder drive, are enforced in the other
package.** Only kpi: engine-parity is confit's own (`_projection_test.py`,
`_serving_test.py`, `_transformers_test.py` and `_corpus_test.py` all live in
`packages/sql-transform`). Absorbing the set did not move a gate: this document's
engine-purpose section says it is about confit's half (goal: engine-half-only), and it now
holds four bars its own package does not enforce. That is stated rather than hidden — it was
the strongest argument against absorbing, and the ruling in the document-set section took it
knowingly.

**A pointer that resolves is not a bar that holds at its written depth.** Where a control's
text names a run depth and its gate reads that depth from an environment variable, the
control holds at the default, not at the text — under the standing-law section that is an
unacknowledged trade-off, and the remedy is a decision (correct the text, or raise the
default and pay the runtime), not an edit. Which control that is today, and by how much, is the
dated report's; the decision routes through ask: kpi-set-change.

### 6.5 Proposed KPI candidates — none adopted {#proposed-kpis}

Six candidates, **separate from the seven in force above**. Each names the measurement that
would back it and what it costs. **None is adopted** — changing the set is
ask: kpi-set-change. Definitions and methods only: no candidate carries its current value,
which lives in the dated report.

**kpi: acceptance-rate.** **[PROPOSED]** A **drive**: the fraction of generated-grammar
cases that build rather than refuse, reported per campaign with its seed range.
*Measurement that backs it:* exists and runs today — `python -m fuzz.runner` over a named
seed range, deterministic and re-runnable; acceptance is the complement of the `REFUSED`
count over the case count.
*Cost:* the denominator is a grammar, not query space, so the number can be moved by
editing the generator — and that is not hypothetical. `fuzz/gen.py:1060` reads
`w = rng.choice([1, 2, 3])  # width-1 list must REFUSE`: the generator deliberately emits
UDF return shapes it knows are rejected, and changing that one `rng.choice` moves acceptance
by percentage points with no engine change at all. Adopting it as a *drive* without
claim: coverage-denominator's triple-axis work invites optimizing the wrong thing; adopting
it as a *reported statistic* costs nothing. Either way ask: exclusion-ratification (3) has
to settle first whether a caller-declaration error belongs in the denominator.

**kpi: findings-per-campaign.** **[PROPOSED]** An explicit **zero-control**: findings per
N seeds, counted **beside** `DIVERGE_OPT` rather than with it. The wording matters: the
oracle spec's claim: contract-surface-gap is explicit that `DIVERGE_OPT` "stays a reported
finding ... rather than an accepted class", so this KPI may hold it out of the *bar* — a
zero-control over a class we knowingly tolerate would be red on day one — but never out of
the *report*. A control also has to name `TIMEOUT`/`PANIC`, or a loaded machine reads as a
finding.
*Measurement that backs it:* `findings.jsonl` already carries exactly this, per kind and
per class.
*Cost, and it is the reason to think:* the campaign fuzzer is **not** a standing gate —
`packages/confit/fuzz/` is a manual CLI, `testpaths = ["tests"]` does not collect it, and
`test_fuzz_smoke.py`'s own docstring says "'no findings over N seeds' **cannot** be the CI
invariant". Making findings-per-N a control would mean either wiring the campaign into CI
(minutes per run, and the value of a fuzzer is finding live bugs, which a zero-control
punishes) or declaring a control nothing enforces. The honest middle is a control on the
**release** cadence, not the commit cadence.

**kpi: unshipped-burndown.** **[PROPOSED]** A **drive**: the set of widths the oracle emits
and we do not yet serve, with its class list, driven to empty.
*Measurement that backs it:* the campaign already isolates these cases and reports them in a
section of their own, and `fuzz.oracle._type_delta` carries one arm per open width — so the
set is exactly the divergence ledger's open widths, already machine-readable. The rules by
which the campaign classifies them are the measurement layer's (the acceptance-frame
section's pointer), and the reading is
`packages/confit/docs/reports/2026-09-02-goal-baseline.md`, gap: unshipped-decimal-arithmetic.
*Cost:* near zero; this is the one candidate whose machinery is already built and whose
closing bell already rings. Note the empty state is ambiguous — the runner's own comment
says an empty section means either the width shipped **or the grammar stopped reaching
it**, so burn-down to zero needs the generator checked, not just the number.

**kpi: named-refusal-share.** **[PROPOSED]** A **control** at 100% — and it is the one
candidate that is a *bar*, over kpi: no-third-mode's own clause; the standing-law section
says so rather than pretending otherwise. The property: every refusal carries a documented,
actionable prefix **and names the construct**.
*Measurement that backs it:* a script over the campaign's own verdicts measures the
**prefix half** — a refusal message either starts with one of the documented prefixes or it
does not — so any reading of it is a **floor** on this KPI, never a reading of it: the
frontend's catch-all `expression: {other}` site passes the prefix check while naming
nothing. The **construct-naming half has no measurement today** and would need one before
the bar could be set.
*Cost:* a control adopted while the property demonstrably does not hold is, by the
standing-law section's own definition, "not a control, it is an unacknowledged trade-off" — so
adopting it means **first** closing the undocumented refusals, which is the oracle spec's open
prefix-reconciliation work, and only then declaring the bar. Adopting it as a drive first,
then promoting it, is the sequence that respects the standing law. This is the candidate I
would rank first: it is the only one that measures whether goal: two-outcome-contract's
*second* outcome is actually usable, and nothing measures that today.

**kpi: ladder-ratchet.** **[PROPOSED]** kpi: coverage-ladder, extended: the corpus match
count, the dialect L2 and L3 floors, and the sql-transform admission ladder, each with a
never-decreases rule and a single dated home.
*Measurement that backs it:* all four already exist; two already ratchet (the dialect
floors), two do not.
*Two scope notes before this one is adopted, because both change what it costs.* The
sql-transform ladder is **not confit's** (goal: engine-half-only, and the four-yardsticks
section says so in the same breath as citing it) — a confit document proposing a gate over
the other package's admission ladder is annexation unless the owner rules the KPI
cross-package. And the L3 Spark floor **cannot be read in an environment without `pyspark`**,
where the fixture fails loudly rather than skipping, so ratcheting it means naming the
environment that checks it.
*Cost:* a deliberate scope reduction becomes a gate failure. That cost is not theoretical —
an ungated count has already slipped unnoticed, which is the argument *for*; and a scope
reduction is sometimes right, which is the argument *against*. Same question as the oracle
spec's ask: match-count-ratchet; answer them together.

**kpi: bench-refresh-cadence.** **[PROPOSED]** kpi: serving-latency carries a **staleness bound**: the
serving bench is re-run and re-recorded at a named cadence (per release, or per N commits),
and a number older than the bound is marked stale rather than quoted.
*Measurement that backs it:* the bench runs in a few minutes with a parity gate of its own,
so the cadence is affordable.
*Cost:* absolute numbers drift with machine load — the same cell has read a wide spread
across two runs on one machine on one day, and the size of that spread is the report's — so
a cadence produces noise unless what is
recorded is the **ratio** to a baseline row in the same run, which is what
kpi: serving-latency already has for the transformer path (a 2-field query against a
1-field one) and does *not* have for the pure-SQL table. Adopting this means picking
which ratio is the metric, and settling
`gap: bench-baseline-flip` in
`packages/confit/docs/reports/2026-09-02-goal-baseline.md` first — a cadence on a metric
whose baseline changed identity would re-record the confusion.

> ### ask: kpi-set-change — adopt any of the six, and as what?
>
> The KPI set is five controls and two drives, and changing it is yours alone. My ranking,
> with the reason, not as a recommendation to be rubber-stamped:
>
> 1. **kpi: named-refusal-share, as a drive now and a control later.** Nothing today
>    measures whether a refusal is usable, and a refusal that does not name its construct
>    fails goal: two-outcome-contract's promise as surely as a wrong value does. Adopting it
>    as a control at today's share would violate the standing law on its first day.
> 2. **kpi: unshipped-burndown, as a drive.** Its machinery already exists and already
>    rings; adopting it costs a line in the drives-in-force section.
> 3. **kpi: ladder-ratchet**, decided together with the oracle spec's
>    ask: match-count-ratchet, because they are one question.
> 4. **kpi: bench-refresh-cadence**, once the serving bench's baseline question is settled —
>    a cadence on a metric whose baseline changed identity would just re-record the
>    confusion.
> 5. **kpi: acceptance-rate**, as a *reported statistic* rather than a KPI, until a
>    denominator exists (claim: coverage-denominator).
> 6. **kpi: findings-per-campaign** last, and only at release cadence — a zero-control on a
>    fuzzer punishes the fuzzer for working.
>
> This ask also carries one decision that is not an adoption: where a control's written run
> depth and its enforced default disagree, **correct the text or raise the default**. Both
> are changes to the KPI set, so both land here.
>
> Adopting none is a legitimate answer and leaves the seven in force exactly as they are.
>
> *Context:* the readings behind all six, and the depth gap, are in
> `packages/confit/docs/reports/2026-09-02-goal-baseline.md`.
>
> *Binds:* all six `kpi:` slugs, and the measurement-and-kpis section's two-kind structure.

---

## 7. Where this document sits {#document-set}

Above the specs, below the owner. The intended shape of the set:

| document | answers |
|---|---|
| **this document** (`docs/goal.md`) | the target — what confit is for, how much of it we intend to serve, what is out of scope by decision, how it is measured |
| `docs/reports/<date>-goal-baseline.md` | the distance — what those yardsticks read on a date, and every way the engine diverges from the target; one file per reading, never an edit to an older one |
| `docs/oracle/` (merged) | what *correct* means — the oracle's identity, the verdict taxonomy, the comparison contract, pins, the divergence ledger |
| the engine spec | how the engine achieves it — lanes, slots, the specializer, the backends |
| the testing spec | how it is checked — gates, corpora, campaigns, what each suite is for |

The direction of citation runs downward: this document may cite an oracle-spec claim as
evidence for a goal, and the oracle spec does not cite goals. Where the two overlap the
oracle spec wins on *correctness* questions and this document wins on *scope* questions.
Concretely, and it is tested by two places above: the engine-purpose section **points at**
the divergence ledger as the enumerated exception to the two-outcome contract and takes the
rows and their status from the ledger's own column rather than copying them; and where a scope
decision overlaps a divergence row — exclusion: resource-ceilings and exclusion:
optimizer-on-answers here, gap: unshipped-decimal-arithmetic in the report — the entry
**names** the row it overlaps and leaves it `unruled` where the ledger leaves it. Citing a row
is deference; restating its verdict would be re-litigation, and the acceptance-frame section
does not redefine a verdict either. The same direction holds downward into the reports: a
report reads this document's yardsticks and states the distance from them; it never amends one.

**ask: kpis-absorb-or-defer — RULED: this document owns the KPIs.** Three options were on
the table: absorb, defer-and-split-by-kind, defer wholly. The owner ruled **absorb**.
`packages/confit/docs/kpis.md` is deleted; its five controls, its two drives and its
standing law are the measurement-and-kpis section, under `kpi:` slugs, and its dated readings
stayed out under the front matter's rule — they are the dated report's.

Two costs were known before the ruling and are taken, not discovered. **Citations:**
line-anchored references into that file — from the oracle spec (claim: fit-serving-oracle),
`properties.md`, drafts and merged PRs — no longer resolve; every live one was repointed in
the same commit, and the historical ones (backlog tickets, dated reports and decision
files) were left as the records they are. **Scope:** four of the five controls and
kpi: coverage-ladder are enforced in `packages/sql-transform`, so a document whose
engine-purpose section says it is about confit's half (goal: engine-half-only) now holds four
bars its own package does not enforce. The enforcing-suites section states that rather than
hiding it.

---

## ASK index {#ask-index}

### Ruled

An ASK leaves the open table only by being answered. The ruling text lives at the point in
the document where it binds, next to what it created.

| ask | ruling | where it landed |
|---|---|---|
| **ask: kpis-absorb-or-defer** | **absorb** — this document owns the KPIs; `packages/confit/docs/kpis.md` is deleted, its definitions and standing law move here under `kpi:` slugs, its dated readings stay in the reports | the document-set section (the ruling) and the measurement-and-kpis section (the set itself: kpi: training-round-trip, kpi: engine-parity, kpi: binding-parity, kpi: transformer-parity, kpi: no-third-mode, kpi: coverage-ladder, kpi: serving-latency) |

### Open here (3)

| ask | question | binds |
|---|---|---|
| ask: acceptance-target | is there an acceptance-rate target, and does the ladder ratchet? | goal: growing-accepted-surface, kpi: acceptance-rate, kpi: ladder-ratchet |
| ask: exclusion-ratification | does the six-row permanent scope set bind — its grounds, its silence, and the ground it has no slot for? | every `exclusion:` slug, kpi: acceptance-rate |
| ask: kpi-set-change | adopt any of the six proposed KPIs, and as what kind? | all six proposed `kpi:` slugs in the proposed-kpis section |

### Open in the dated report (1)

An ask whose subject is *distance* is asked where the distance is measured. It is still the
owner's to answer; it is just not a question about the destination.

| ask | question | where |
|---|---|---|
| ask: next-query-classes | which unbuilt query classes are next, in what order? | `packages/confit/docs/reports/2026-09-02-goal-baseline.md`, in its left-open section, ranking the gap-ledger section |

The report also carries the part of ask: exclusion-ratification that asked whether the
enumeration *covers* what the engine refuses today — a question about the current refusal
surface, not about the target.

**Current readings live in `packages/confit/docs/reports/`**, one dated file per reading,
starting with `2026-09-02-goal-baseline.md`. Anything this document once carried as a dated
number is there under the same slug, as is every way today's engine falls short of what is
written here — including the serving bench's baseline question, which is a **gap with
options** for the implementation loop rather than a question the owner must rule on, and the
loop-level findings each reading turns up. Those files also carry the reproduction commands
and their environment preconditions, which is where a fresh checkout starts.

Three questions this document deliberately does **not** ask, because they are already open
in the oracle spec and forking them would split the answer: the corpus match count's
ratchet (ask: match-count-ratchet), the undocumented refusal prefixes (the oracle spec's
claim: refusal-message-prefixes already records that the prefix set is not exhaustive and
routes the fix), and whether every documented limitation really has an executable twin
(ask: doc-twin-overstatement — exclusion: whole-relation-shapes states the partial truth
rather than becoming a sixth site for the overstatement).
