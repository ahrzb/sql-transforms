# The dialect logical plan — Query ⇄ Plan for DuckDB, Spark, BigQuery

**Date:** 2026-08-13. **Status:** proposal (initial design PR, pre-implementation).
**From:** the 2026-08-13 dialect discussion with AmirHossein — decisions
carried in verbatim, marked **[D]** below. Everything labeled *to pin* is a
Phase-0 measurement, not a claim.
**Builds on:** `2026-08-11-duckdb-type-lattice-design.md` (the typing
doctrine this design reapplies), `2026-08-13-function-signature-registry-design.md`
(calls as resolved signatures), `2026-08-08-typed-duckdb-ast-design.md`
(why lenient serialization boundaries are C5-dangerous),
`2026-08-11-differential-fuzzer-design.md` (the gate this extends),
`packages/confit/src/specializer/ir/` (the owned-IR recipe: verifier,
canonical text, round-trip).

## Goal

Push fit-time computation back to the OLAP engine where the training data
lives — before reading a row — while authoring stays one dialect **[D]**.
Concretely:

```
DuckDB-dialect SQL ──parse──► bound, typed logical plan ──print──► Spark SQL
                                        │                          BigQuery SQL
                                        └──print──► canonical DuckDB SQL
```

with every conversion **bit-exact where representable, ε-bounded where
floating-point accumulation makes bit-exact impossible, and refused by name
otherwise** — the three-outcome discipline, per dialect.

**Dialect choice for this epic: both Spark and BigQuery are designed here
(mapping tables below); Spark ships first.** The reason is oracle economics,
not preference: `pyspark` in local mode is a free per-commit oracle in CI,
exactly like DuckDB is today, so Spark keeps the repo's gate cadence.
BigQuery is remote and metered — its gate is a pinned corpus on a schedule,
a genuinely new risk class (the oracle can move under us between runs), and
that deserves its own phase, not a footnote in the first one.

### Non-goals

* **Printers are not general transpilers.** *(Amended 2026-08-13, with
  AmirHossein: representation IS general.)* The plan's goal is universal
  DuckDB **query** coverage — almost any query gets a plan, D2's doctrine
  ("representable is unconditional") applied to constructs, grown
  corpus-first through the L2 gate. What stays bought per consumer, with
  named refusals, is *printing* (per dialect) and *lowering* (confit —
  RFC-1). DDL and DML stay out. The frontend remains sqlparser + token
  pre-rewrites; DuckDB's own AST is plugin-extensible and not a stable
  contract.
* **Not an optimizer.** The plan preserves author structure (projection
  chains stay chains; nothing is fused or reordered). Printers print what
  is there. Canonicalization is spelling-level, never structural — the
  round-trip law below depends on this.
* **Not Substrait.** Substrait puts a third party's function semantics and
  a lossy producer inside the trust boundary; this plan keeps DuckDB — an
  engine we can execute — as the semantic spec. A Substrait *exporter* at
  the edge remains possible later and is out of scope.

## Decisions carried in from the 2026-08-13 discussion

* **[D1] DuckDB is the reference engine.** Plan semantics are "what DuckDB
  (the repo's pinned oracle version) does, as measured." DuckDB is also the
  executable spec: it runs locally, so every cross-dialect gate is
  "run the plan on DuckDB, run the printed SQL on the target, compare."
  The no-warehouse user is the degenerate pipeline: no printer at all.
* **[D2] The plan's type universe is the full DuckDB type lattice**, spoken
  in Arrow vocabulary at every data boundary (params tables, gate
  comparisons) via DuckDB's own Arrow export, pinned. Representable is
  unconditional; *printable* is bought per dialect by reachability, with
  named refusals — the type-lattice doctrine ("semantic typing is
  unconditional; compute machinery is bought per type, by reachability")
  reapplied with "printer" in place of "compute".
* **[D3] Nothing implicit in the plan.** Every dialect-divergent default is
  a mandatory explicit field (table below). Frontends fill in the *source*
  dialect's default; printers force the semantics in the *target* dialect.
* **[D4] Multiset semantics.** No relation in the plan has an order; any
  construct whose value depends on physical order must carry a total order
  or is refused (the marginalizer's multiplicity rule, generalized).
* **[D5] Tiers, not heroics, for floats.** Order-insensitive operations are
  the exact tier; float-accumulation aggregates are the ε tier;
  irreconcilable constructs (regexp flavors, etc.) refuse per dialect.
  Opt-in **quantization** is the bridge from ε back to exact. ε tolerance
  policy is deliberately deferred until a second engine exists to measure
  (decided 2026-08-13); the gate ships with a *provisional* tolerance.
* **[D6] A dialect is an engine + version + pinned session configuration.**
  Spark without `spark.sql.ansi.enabled` pinned is a family of dialects,
  not one.
* **[D7] Hub and spoke, printers first.** One frontend (DuckDB — the
  authoring dialect, the oracle grammar we already sit on) and N printers.
  Reverse frontends (Spark/BQ → plan) are demand-driven later phases.

## The laws

Four executable identities. Everything else in this document is machinery
for making them checkable.

```
L1  round-trip        parse_d(print_d(p)) == p            for every dialect d with a frontend
L2  invisibility      run_duck(sql) == run_duck(print_duck(parse_duck(sql)))     bit-exact
L3  cross-dialect     run_duck(print_duck(p), D) ≍ run_e(print_e(p), D)          per tier
L4  determinism       the meaning of a plan the verifier marks deterministic
                      is a function of its input multisets
```

* **L2** is what lets the plan sit inside the existing gates without
  weakening them: parsing and reprinting must be invisible to the oracle,
  statement by statement, on the corpus. It is this design's analogue of
  the marginalize law ("freezing must be invisible on the training data").
* **L3** compares per output column by tier: exact-tier columns byte-equal
  through Arrow (after pinned schema normalization); ε-tier columns within
  the provisional tolerance; and a plan containing anything the target
  printer hasn't bought **fails at print time, by name** — never at the
  gate, never silently.
* **L4** is a verifier *verdict*, not a frontend refusal *(amended
  2026-08-13: represent + mark)*: the verifier classifies every plan
  `deterministic | nondeterministic(named cause)` — `array_agg` without an
  inner `ORDER BY`, `LIMIT` without a total order, `random()`, `SAMPLE`.
  L1/L2 hold for every admitted plan either way; the cross-dialect gate
  (L3) and confit's lowering refuse nondeterministic plans by their named
  cause.

## The plan

Lives at `packages/confit/src/dialect/` — a sibling of `specializer`, not a
change to it. It reuses the crate's assets (token pre-rewrites from
`rewrite.rs`, `sqlparser`, the function signature registry, the growing `Ty`
lattice) and is exposed to Python through the existing PyO3 module:

```python
from confit.dialect import parse, print_sql, Dialect

p = parse(sql, catalog)                      # DuckDB dialect; bound + typed + verified
print_sql(p, Dialect.DUCKDB)                 # canonical DuckDB SQL (L2)
print_sql(p, Dialect.SPARK)                  # or raises PrintUnsupported("hugeint: no spark landing zone")
```

The module carries the `ir/` recipe wholesale — its three mandatory
properties are mandatory here too:

1. **Airtight verifier** — explicitness (D3), determinism (L4), and type
   correctness checked structurally; unverifiable plans are constructor
   errors, the `Internal` class.
2. **Canonical plan text** — the plan's own text form with bindings and
   types visible (not SQL), for fixtures and review, exactly as
   `ir/print.rs` does for the SSA IR.
3. **Round-trip** — `parse(print(p)) == p` for the plan text, so fixtures
   are executable and diffs are meaningful.

The lesson of the typed-AST spec applies to every boundary here: no lenient
readers, no defaulting writers. A printer that forgets a field must fail to
compile (exhaustive `match` on node kinds — "break the build, not the
answers"), because the alternative is C5's unrecoverable state: a different
query, silently.

### Node set (v0) and expressions

```
Rel    := Scan(table, schema)                      -- __THIS__, statics, params
        | Project(input, [(name, Expr)])
        | Filter(input, Expr)
        | Join(input, input, kind, on: Expr?)      -- on bound over left++right; None iff CROSS
                                                   -- (2026-08-13-dialect-join-node-design.md)
        | Window(input, [WindowDef])               -- partition keys, ORDER (total, explicit), frame (explicit)
        | Aggregate(input, [key: Expr], [agg])     -- the fit plan's GROUP BY / DISTINCT steps
        | Distinct(input)
```

Expressions: bound column refs (name + ordinal — names for rewrite
ergonomics, ordinals checked by the verifier), typed literals on the full
lattice, `Cast { strict | try, target }`, `CASE`, `IS [NOT] DISTINCT FROM`,
and **calls as resolved signature ids** from the registry — never bare
names. A printer maps *signatures* to target spellings (or expands to an
expression with identical measured semantics, e.g. a null-safe equality that
the target lacks) or refuses the signature by name. This is where the
per-dialect function surface lives, and it is a data table, not code paths.

Growth beyond v0 (set ops, `Unnest`, `Limit`+total-order, …) is
corpus-driven, each node arriving with its verifier rules, both text forms,
and its row in every printer's support table — support tables are exhaustive
by construction (a `match`, not a lookup with a default).

### Nothing implicit — the mandatory-field table (D3)

| surface construct | plan field (mandatory) | frontend fills (DuckDB default) | printers emit |
|---|---|---|---|
| sort key in window/agg `ORDER BY` | direction **and** null order | `ASC`, `NULLS LAST` *(to pin)* | both, always, every dialect |
| join equality | explicit via expression node kind (`Eq` vs `IsDistinct`) | `=` → `Eq`; `IS NOT DISTINCT FROM` → `IsDistinct` | each dialect's pinned spelling of that node |
| `CAST` | `strict` \| `try` + failure semantics | `CAST` → strict, `TRY_CAST` → try | dialect's checked/safe form *(to pin per dialect)* |
| window frame | explicit frame bounds | DuckDB's default frame *(to pin)* | explicit frame, always |
| numeric operators | resolved signature (int vs float division, overflow class) | registry | per-signature spelling *(to pin)* |
| temporal ops | explicit tz-relevance per signature | registry | under the pinned session tz (D6) |

The pattern is uniform: **disagreement between engines becomes data in the
plan instead of behavior in the engine.**

### Determinism (D4/L4), and the ε residue

With multiset semantics enforced, the only order-dependence left is float
accumulation inside `sum`/`avg`/`stddev`-family aggregates over
`FLOAT`/`DOUBLE`. Tier assignment is *measured per aggregate signature*,
not assumed; the a-priori expectation:

* exact tier: `count`, `min`, `max`, `bool_and/or`, sums over
  integers/decimals — order-insensitive by algebra.
* ε tier: float-accumulation aggregates.

**Quantize node (D5):** `quantize(x, grid)` with `grid ∈ {f32,
round(s, mode)}` — an explicit, opt-in plan node whose per-dialect spelling
is pinned (e.g. DuckDB `CAST(CAST(x AS FLOAT) AS DOUBLE)` for the f32 grid,
the same trick `pack_trees` uses on thresholds). It moves a value from the
ε tier to the exact tier deterministically. A later analysis pass that
*warns* when an ε-tier value flows into a discontinuity (comparison, tree
call, bucketing) is named future work, not v0.

## Types (D2)

The plan represents the **full DuckDB lattice**: BOOLEAN, the eight integer
widths incl. HUGEINT/UHUGEINT, FLOAT, DOUBLE, DECIMAL(p,s), VARCHAR, BLOB,
DATE, TIME, TIMESTAMP(_S/_MS/µs/_NS), TIMESTAMPTZ, INTERVAL, UUID, ENUM,
LIST, STRUCT, MAP, UNION, BIT. At data boundaries the vocabulary is Arrow,
per DuckDB's own export, **pinned type-by-type in Phase 0** (the lattice
spec already caught `sum(BIGINT) → decimal128(38,0)` this way).

Printable is narrower than representable, per dialect, with named refusals.
Expected landing zones — **every row below is pinned by probe before the
printer ships it; an unprobed row is a refusal**:

| plan type | Spark (ANSI, to pin) | BigQuery (to pin) |
|---|---|---|
| BOOLEAN | BOOLEAN | BOOL |
| TINY/SMALL/INT/BIGINT | same widths | **refuse v0** — only INT64 exists; widening erases DuckDB's per-width overflow traps. Possible later: guard exprs via BQ `ERROR()` |
| unsigned ints | refuse | refuse |
| HUGEINT | **refuse** — DECIMAL(38) is too narrow for int128 | BIGNUMERIC (76 digits) |
| FLOAT | FLOAT | **refuse v0** — no float32; widening changes rounding |
| DOUBLE | DOUBLE | FLOAT64 |
| DECIMAL(p≤38,s) | DECIMAL(p,s) | NUMERIC / BIGNUMERIC by (p,s) fit |
| VARCHAR / BLOB | STRING / BINARY | STRING / BYTES |
| DATE / TIME | DATE / **refuse** (no TIME) | DATE / TIME |
| TIMESTAMP (µs, wall) | TIMESTAMP_NTZ | DATETIME |
| TIMESTAMPTZ | TIMESTAMP (session tz pinned UTC) | TIMESTAMP |
| TIMESTAMP_S/_MS/_NS | refuse v0 | refuse v0 |
| INTERVAL | refuse v0 (interval algebras differ) | refuse v0 |
| LIST / STRUCT | ARRAY / STRUCT | ARRAY (element rules to pin) / STRUCT |
| MAP / UNION / ENUM / UUID / BIT | MAP / refuse / refuse / refuse v0 | refuse / refuse / refuse / refuse v0 |

Decimal *arithmetic* (result-scale propagation) follows D1: the rules are
DuckDB's, measured — the lattice spec's phase 5 measures them once for both
consumers. A printer for an engine whose propagation differs rewrites with
explicit casts to force DuckDB's result type, or refuses that operator on
decimals; which one is a Phase-0/3 measurement.

## Dialects (D6)

| dialect | engine + version | pinned configuration | oracle | gate cadence |
|---|---|---|---|---|
| `duckdb` | the repo's pinned oracle (1.5.5 today) | defaults | itself | per-commit (existing) |
| `spark` | one pinned Spark 3.5/4.x, chosen in Phase 0 | `spark.sql.ansi.enabled=true`, `spark.sql.session.timeZone=UTC`, + any flag Phase 0 shows matters | `pyspark` local mode in CI | per-commit |
| `bigquery` | the service — **unversionable** | (few knobs; enumerate in its phase) | real BQ, scheduled | pinned corpus, scheduled; pins are date-stamped because the oracle moves |

The harness *sets* the pinned config; printed artifacts document that they
are only valid under it. Spark's ANSI flag flips overflow, cast, and
division semantics wholesale — it is the difference between "a dialect" and
"a family of dialects".

Known refuse-tier families going in (from `retrans.rs` precedent): the
regexp functions cross-dialect (DuckDB/BQ are RE2, Spark is `java.util.regex`
— per-pair translation like RE2→rust-regex is possible later, refusal
first); collation-sensitive string ordering beyond binary UTF-8 *(3-engine
probe in Phase 0)*.

## What this deliberately does not touch

* **`specializer/` is untouched.** Confit's serving frontend re-hosting on
  the dialect plan is a *named future epic*, gated on corpus replay staying
  bit-identical (550/678+, no outcome may change class).
* **The JSON-AST marginalizer is untouched.** Re-hosting `marginalize` on
  the plan is a later epic, gated on the marginalize law
  (`…marginalize(text).fit(F).transform(F) == run(SQLTransform(text), F)`)
  holding verbatim. Until then the two representations coexist, each behind
  its own gate — which is already true today.

## Phases (each certified before the next)

**Phase 0 — measurements, no code.** Probe and pin: DuckDB→Arrow export for
every lattice type; DuckDB's default sort null order and default window
frame; Spark version choice + ANSI semantics for division, overflow, cast
failure, `<=>`, `TIMESTAMP_NTZ` reachability from SQL; 3-engine string
ordering; the aggregate tier table (which aggregates are order-sensitive,
measured, per engine). Output: `pins-dialect/` fixtures in the house style.

**Phase 1 — the plan exists, DuckDB round-trips.** Node set v0 over the
specializer's *current* surface (projections, filters, static joins,
expressions), verifier, canonical plan text + text round-trip, DuckDB
frontend + printer. **Gate:** the 678-statement corpus — every statement
Confit prepares today satisfies L1 and L2 bit-exact; three-outcome
accounting (new `Unsupported` allowed, no statement changes class, zero
wrong answers).

**Phase 2 — widen to the fit surface.** Window, Aggregate, Distinct,
projection chains, scalar subqueries. **Gate:** the marginalizer's fixture
corpus replayed through L1+L2; the differential fuzzer taught to route
generated statements through parse→print (invisibility fuzzing).

**Phase 3 — Spark printer.** Signature spelling table, type table above
(probed rows only), forced-explicitness printing, `quantize` spellings.
**Gate:** L3 per-commit against `pyspark` local on corpus + fuzzer data;
`pins-spark/`; a published printed/refused KPI per construct class
(kpis.md), like 550/678 is today.

**Phase 4 — BigQuery printer.** Same shape; scheduled remote gate with a
small pinned corpus; decide the narrow-int guard-expression question here;
date-stamped `pins-bigquery/`.

**Phase 5 — reverse frontends (demand-driven).** `parse(sql, dialect=SPARK
| BIGQUERY)`: each must satisfy L1 and its own-engine L2 before it ships.
This is where "from BigQuery" becomes real; nothing earlier depends on it.

## Gates & accounting

* Print refusals are a first-class outcome: `PrintUnsupported`, named per
  dialect per construct — the corpus accounting grows a column per dialect,
  never a silent approximation.
* The ε tolerance is **provisional** until Phase 3 produces measurements;
  revisiting it is a standing agenda item of that phase (per the 2026-08-13
  deferral), and `quantize` is documented as the supported way to opt out
  of ε entirely.
* Anything in the mapping tables above that ships unprobed is a bug in this
  document; the tables are expectations, the pins are the truth.
