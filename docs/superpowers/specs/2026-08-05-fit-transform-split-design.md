# Fit/transform split — surface and semantics (v0)

Date: 2026-08-05. Status: designed with AmirHossein (this session);
settles DRAFT-25's open edges. Every DuckDB behavior cited below was
measured on 2026-08-04/05, not recalled.

## The governing rule (unchanged)

Any construct involving a transform means **what DuckDB would compute if
the pieces were registered UDFs/UDAFs**. We add only fit machinery and
error timing (P7 — "construction-time refusals": anything DuckDB would
reject at bind/run refuses at construction or fit, by name). No third
behavior (C2 — "confit ≡ DuckDB bit-for-bit or named refusal").

## The split, and why the `OVER` sugar dies

```
tfm_fit       : Agg[Struct<a,b,c>]  =>  Struct<type, id>    -- true window aggregate: rows of a scope → θ handle
tfm_transform : (Θ, Struct<a,b,c>) =>  Struct<f1,f2>        -- true scalar UDF
tfm(x)          ≡  tfm_transform(tfm_fit(x) OVER (), x)     -- the ONE sugar: global fit-transform
```

Today's spelling `tfm(x) OVER w` is **deleted** (v0, no compat shim). It
is the one construct in the design with no oracle reading: a real
aggregate over a window returns the *same value for every row of a
partition*; the sugar returned per-row values. Inside a larger
expression a reader cannot tell fit scopes from real window aggregates.
After the split, every `OVER`, `FILTER`, and frame clause in the
language is a true window construct — `tfm_fit` returns the same θ per
partition exactly like `avg` — and `tfm_transform` is per-row exactly
like `round`.

The bare sugar's residual license: a scalar UDF's value depends only on
its arguments; `tfm(x)`'s depends on the training stream. That is
precisely the "fit machinery" the rule declares as our one addition,
and at serving time the call literally is a scalar UDF (θ frozen,
joined).

## Surface

```sql
-- the common case (sugar): global fit-transform, field-addressed
SELECT sc(struct_pack(v := age)).v AS z FROM __THIS__

-- any non-trivial fit scope: the split, θ parked in a PRIVATE column
SELECT sc_fit(struct_pack(v := age)) OVER (PARTITION BY sex)   AS _th,
       sc_transform(_th, struct_pack(v := age)).v              AS z
FROM __THIS__

-- inline nesting is equally legal (measured: window app inside a scalar call)
SELECT sc_transform(sc_fit(x) OVER (), x).v AS z FROM __THIS__

-- leakage control is a spelling, not a policy (measured: FILTER on window aggs)
sc_fit(x) FILTER (WHERE split = 'train') OVER ()

-- order-sensitive fits name their order (measured: in-call ORDER BY on window aggs)
sgd_fit(x ORDER BY ts) OVER (PARTITION BY g)
```

Naming: a registered transformer `x` reserves `x_fit` and `x_transform`.
A user UDF colliding with a reserved name refuses at construction.

Refusals (all at construction, by name, unless marked):

| spelling | verdict |
|---|---|
| `sc(x) OVER w` | refuse — the deleted sugar; say "fit scope moved to sc_fit" |
| `sc_fit(x) OVER (ORDER BY ts)` (any ordered frame) | refuse — running fit (per-row θ); future feature |
| `sgd_fit(x) OVER ()` with `sgd` declared order-sensitive | refuse — order-sensitive fit needs in-call ORDER BY |
| `sc_transform({'type':'sc','id':3}, x)` | refuse — hand-written θ has no lawful provenance (graduates with composition's frozen members) |
| `sc_transform(pca_fit(x) OVER (), x)` | refuse — θ type mismatch, statically visible |
| θ type mismatch only visible at runtime (θ routed through CASE etc.) | traps at serving — the UDF's own error; our addition is only the earlier timing where visible |

`avg`-style in-call ORDER BY on an order-*blind* fit is accepted and
honored (DuckDB accepts it on `avg`; the oracle rule answers this one).

## θ — the handle

`Struct<type, id>` at value level; it IS the existing wire mechanism
(the `type` names the minted UDF binding to its instances store, the
`id` is the params-table `__cf_est` joined per group). Consequences:

- **θ is an ordinary value.** Private columns, public output columns,
  arbitrary expressions — it flows like any struct. Export is already
  paid for: a public `... AS theta` column serves as
  `struct_pack(type := '__cf_tf0', id := __cf_p0.__cf_est)` — pure
  existing parts, same value per group, lawful under the oracle.
- **Staleness is out of scope in v0 — unreachable.** `fit()` rebuilds
  instances + params atomically in one object; exported θ is inert data
  with no re-entry path (literals refuse; `tfm_transform` accepts only
  θs statically traced to a fit call in the same SQL). Non-reusing
  (generation-keyed) ids become the FIRST requirement of whichever
  future feature opens re-entry — composition's frozen members or
  artifact splitting — and are not built before then.
- The `type` tag is value-level because the value is the mechanism;
  nothing phantom-typed to design.

## Fit lawfulness and the determinism non-promise

1. **Default contract: a fit is a multiset aggregate** — order-blind,
   seed-fixed. P15's family ("purity/determinism as declared contract,
   not detected property"): the transform author signs it; a fit that
   silently violates it is a broken transform, like a nondeterministic
   UDF. Unenforced.
2. **Order-sensitive fits declare it** (wrapper in the `Named` pattern,
   e.g. `OrderSensitive(SGDRegressor(...))`) **and the query names the
   order** via in-call `ORDER BY` — DuckDB's own ordered-aggregate
   spelling. Marginalization stable-sorts each fit scope by the named
   key before `est.fit()`. Mechanism promise only: *we sort by what you
   name.* Nothing more.
3. **Running fits refuse by name.** Window-clause `ORDER BY` means
   per-prefix θ under the oracle (measured: default frame = running
   aggregate) — real feature someday (expanding-window backtests,
   DRAFT-21's territory), explosive cost, nobody asked. One meaning
   gets one spelling in v0: the unbounded-frame spelling also refuses
   until someone wants it.
4. **No determinism promise for fit reproducibility. Ever, in v0.**
   Definition (AmirHossein's): same DataFrame in twice → the order of
   intermediate invocations promised identical. Not promised: no tie
   checks, no total-order enforcement, no cross-run θ guarantee.
   DuckDB itself promises no row order for unordered queries
   (`threads=1` repeating is practice, not contract). A future opt-in
   `deterministic` flag is where the promise would live; its sketch —
   which needs no DuckDB cooperation — is: tag rows with their input
   frame position at ingestion, stable-sort every fit scope by the tag
   before fitting. Costs a sort per scope; build it when wanted.
   Serving determinism is untouched: frozen θ, and C1 ("round-trip
   bit-exact") / C2 compare the same fitted artifact.

## Serving

The serving text shape is unchanged for everything except θ export:
`sc_transform(sc_fit(x) OVER w, x).v` marginalizes to the same
`(__cf_tf0(__cf_p0.__cf_est, ...)).v` form served today. Confit is
untouched until the θ-export and nested-output slices.

## Measured facts (DuckDB 1.5.5)

- A window-valued alias may be referenced laterally in the same SELECT
  (`avg(x) OVER w AS m, x - m`). Window applications may sit inside
  scalar calls. `FILTER` composes with window aggregates.
- In-call `ORDER BY` works on window aggregates (`string_agg(x, ','
  ORDER BY ts) OVER (PARTITION BY g)`) — and is honored. (Postgres
  refuses this; DuckDB is the oracle.)
- Window-clause `ORDER BY` with the default frame = running aggregate;
  with an explicit unbounded frame = whole-partition in declared order.
- In-call `ORDER BY` is also accepted on order-blind aggregates
  (`avg(x ORDER BY ts)`).
- Tie order in any `ORDER BY` aggregate is unspecified in the oracle.

## Out of scope (graduates later, each judged against the oracle)

Running fits; θ literals / frozen application (composition's frozen
members); artifact serialization; `SELECT g, tfm_fit(x) ... GROUP BY g`
(fit artifact as data — also blocked on GROUP BY cardinality, which the
projection surface does not admit); the `deterministic` flag; work
deduplication (gated on purity/volatility typing — DRAFT-25 edge 6).

## Slice 2 addendum — private columns + θ laterals (2026-08-05)

Decisions required by TASK-64's acceptance criteria, recorded here.

**Privacy trigger: output-field-NAME based** (AC#1). Any select item
whose output name starts with `_` is private. Rejected alternative —
authored-alias-only — is non-uniform: a `_meta` table column arriving
via `*` would leak the boundary. Measured constraint that shrinks the
surface: pydantic treats `_`-leading names as private attrs and drops
them from `model_fields`, so a declared `this_model` can never carry a
`_` column — star expansion under a schema cannot produce one, and at
fit the table canonicalizes to the model (extras drop). The one leak
path left is a schema-free `*` passthrough, which refuses at fit by
name. A star `RENAME` to a `_` name refuses at construction.

**The lateral-in-window refusal is LIFTED via substitution** (AC#5).
Lateral aliases now β-reduce at the raw-AST level *before* any rewrite,
so every consumer — later items, window partition/order keys, in-call
aggregate args, transformer bundles — receives a closed expression in
level terms; fit-side SQL never mentions an alias. The old
"lateral alias inside a window function" refusal is deleted.

**A private column is a same-SELECT macro.** It is never rewritten as
an item, never enters the serving text, the output model, or the next
level's environment. Consequences, each a named refusal:

| spelling | verdict (measured DuckDB 1.5.5, 2026-08-05) |
|---|---|
| private column never read | refuse — dead code, and its errors would otherwise never surface (P7) |
| every output column private | refuse — nothing crosses the boundary |
| `_t.f` dotted read of a lateral | refuse — DuckDB binds `_t.f` as table.column ("Referenced table not found"); hint `struct_extract(_t, 'f')`, which DuckDB accepts and we support |
| window-valued lateral inside any window clause | refuse — DuckDB: "window function calls cannot be nested" |
| lateral referenced inside a scalar subquery | refuse — DuckDB accepts (correlated), but our subquery fit steps run verbatim where the alias does not exist; v0 refuses by name. The scan is name-based in both directions and conservatively also hits a subquery-internal binding of the same name |
| consumed lateral whose expression holds a subquery | refuse — measured DuckDB: "the expression has a subquery" |
| any reference before its definition (incl. into windows and subqueries, which ride raw into fit-side SQL) | refuse — DuckDB's own left-to-right rule |
| duplicate output name, any level | refuse — DuckDB's duplicate rules (last-wins after the last definition, refuse between) cannot be honored: the fit-side level table cannot carry two same-named columns |
| duplicate private name | refuse |
| private column read from a higher level | refuse — same-SELECT scope, by name (subqueries included) |
| lambda whose parameter shadows a lateral | substitution never enters a lambda; lambdas already refuse by name (unknown column) |
| authored private item without a declared schema | refuse — lateral resolution is undecidable against unknown table columns (same family as the schema-free star-modifier refusals) |
| schema-free `*` leaking a `_` table column | refuses at transform, per input table (the batch boundary check); `* EXCLUDE (_x)` serves; the row model cannot carry `_` fields at all |

**Public laterals** keep today's surface but ride the same substitution
(identical serving text), which also makes them legal inside windows.
The reference's own alias — a struct_pack field name, a named argument —
survives resolution on every path (review round: it used to be dropped,
so `struct_pack(a := age)` served the wrong field name under a schema).

**θ laterals**: `sc_fit(b) OVER w AS _th` is legal as a private item;
`sc_transform(_th, b)` β-reduces to the inline slice-1 spelling — same
serving SQL, same fit-step dedup. A *public* θ alias still refuses
(export is slice 6; the message now hints the private spelling).
Cross-level θ (a fit parked in a CTE, consumed above) stays refused —
that is θ-as-data, slice 6 territory.

## Slice 3 addendum — FILTER on tfm_fit (2026-08-05)

`tfm_fit(b) FILTER (WHERE pred) OVER (w)` fits each scope on
predicate-TRUE rows and transforms every row. Mechanics and measured
edges:

- The level table materializes `CAST(pred AS BOOLEAN)`; the fit step
  keeps rows where that column IS TRUE. DuckDB accepts non-boolean
  predicates (measured: nonzero is true) — the cast inherits exactly its
  semantics, and SQL's three-valued logic happens in the engine.
- A group with no passing rows gets no params row: an unseen group,
  NULL at serving — the same P14 story, and exactly DuckDB's answer for
  an all-filtered aggregate partition (measured: NULL).
- Everything filtered (no groups at all) refuses at fit by name — the
  fitted output shape is unlearnable.
- FILTER on any *scalar* call — the bare sugar, `tfm_transform`, plain
  functions — refuses at construction (measured: DuckDB binds FILTER
  only on aggregates; previously this crashed at serving or silently
  dropped the clause).
- The predicate is row-wise: transformer calls inside it refuse; the
  fit-step identity includes the filter (a filtered and an unfiltered
  fit never share a step); θ laterals compose (`... FILTER (...) OVER ()
  AS _th`).
- The predicate RESOLVES like any expression (review round): it has no
  serving side, so split_ref rewrites-and-discards it at construction —
  unknown columns and functions refuse by name, author UDFs register for
  the fit connection. Backward select-alias names bind laterally in the
  level table, matching DuckDB; a FORWARD name is undecidable
  schema-free (DuckDB refuses the text) and refuses with the
  qualify-or-rename hint. Fit-step identity includes the predicate's
  named-argument aliases (`_alias_sig`), like every other ident site.

## Slice 4 addendum — ordered fits (2026-08-05)

`OrderSensitive(est)` (the `Named` wrapper pattern: full delegation +
sklearn clone protocol, `order_sensitive = True`) flips the fit contract
from multiset to sequence; the query names the order with the in-call
spelling, `sm_fit(bundle ORDER BY key, ...) OVER (w)`. Mechanics:

- Order keys resolve like FILTER predicates (no serving side →
  rewrite-and-discard at construction; transformer calls inside refuse;
  they ride into the level table as `__cf_ord{j}_{n}` columns).
- The fit scope sorts before `est.fit`: DuckDB does the comparing (its
  collations, its NULL placement — one ORDER BY over the level table
  with a row-index tiebreak), so ties keep input order. That is the
  whole promise: *we sort by what you name.* Defaults are DuckDB's own,
  measured: ASC, NULLS LAST in both directions.
- An order-sensitive transformer without in-call ORDER BY refuses by
  name — on the split spelling and on the bare sugar (which cannot name
  an order). In-call ORDER BY on an order-*blind* fit is accepted and
  honored (the oracle accepts `avg(x ORDER BY ts)`).
- Identity: the θ node carries `arg_orders`, so direction, null
  placement, and key text all distinguish fit steps; `_alias_sig`
  covers named-arg aliases inside keys.
- Composes with FILTER (filter first, then sort the passing rows) and
  with θ laterals. Running fits (window-clause ORDER BY) stay refused.
- Review-round decisions: a top-level `COLLATE` on a key is carried by
  name and re-emitted in the fit-side sort (Arrow strips the annotation
  from the level column); unknown collations refuse at construction
  (probed against the oracle) and a `COLLATE` nested deeper in a key
  expression refuses. Non-integer literal keys refuse (DuckDB's binder
  rule); integer literals are constants, not positional (measured).
  In-call ORDER BY on any *scalar* call — the bare sugar, the transform
  half, plain and author functions — refuses at construction (measured:
  aggregates-only binding; it used to drop silently or crash at
  serving). Wrapper hygiene: `Named` forwards inner declarations like
  `OrderSensitive` does (nesting order cannot cancel a contract), and
  neither wrapper forwards dunders — protocol probes (`__sklearn_clone__`,
  pickle) must see the wrapper, or `clone()` strips it.

## Slice 5 addendum — nested struct outputs (2026-08-05)

**Bare call = struct output, both paths.** A bare `tfm(bundle)` or
`tfm_transform(θ, bundle)` as a WHOLE output item serves its output
struct — `{field: value, ...}`, NULL for an unseen group (the whole
struct is NULL, distinct from a struct of NULLs). Mechanics:

- Construction: whole-item routing in `rewrite_items` sends the item to
  the existing whole-value call (`__cf_tf{j}`, `UDFSpec.field=None`,
  TASK-63); embedded positions still refuse by name ("struct value
  inside an expression" — DuckDB's struct registration would
  binder-error there). An unaliased item keeps DuckDB's derived column
  name: the parse step stamps it as the alias (`AS "sc(age)"`), so the
  oracle's name survives the rewrite verbatim.
- Batch path: nothing to lift — the serving text has been final since
  TASK-63 and the arrow-typed struct registration already serves the
  column.
- Row path (engine): `WideOut`/`EmitField::Wide` carry the declared
  field names; a NAMED extern takes the wide-lane boundary at EVERY
  width (whole-validity lane + one lane per declared field, one ecall
  site); `wide_py` assembles a dict keyed by the names (the DRAFT-22
  list boundary stays for unnamed externs); the synthesized output
  model types the field as a nested struct model; arrow emit builds a
  `pa.struct_` column. A plan holding a named wide field routes through
  `model_validate` — the slot-fill fast path is sound only for plain
  scalar fields (a raw dict in a nested-model slot serializes with
  warnings and breaks attribute access).
- Measured law (2026-08-05): `call.*` / `(call).*` are DuckDB PARSER
  errors — star-over-expression does not exist in the oracle, so the
  spelling refuses at parse (pinned). `unnest(call)` IS the oracle's
  expansion spelling: one column per learned field, named by the FIELD
  names (an alias is ignored), expanded in place among the other items.
- Unnest LANDED as the slice's follow-up (measured rules): `unnest(tfm(...))`
  as a root select item expands in place to one column per LEARNED field,
  named by the field, **alias ignored**; every column reads a lane of the
  one shared ecall site, so unnest alongside a field read still fits and
  evaluates once. Unlawful positions refuse by name, matching the oracle's
  binder: inside an expression ("root element of a SELECT expression"),
  nested unnest, extra `recursive :=`/`max_depth :=` arguments, and a
  non-final level. Collisions refuse AT FIT (P7's learned-T carve-out):
  DuckDB emits duplicate result columns (measured `a, b, a`), which a row
  — a named struct — cannot carry, so a learned name colliding with any
  sibling output or another unnest refuses by name. θ export (slice 6)
  rides on the whole-value struct boundary landed in slice 5.
- Review-round decisions (2026-08-05): learned output names must survive
  the row-path model boundary WHEN SERVED WHOLE — a fit-time probe of
  the real pydantic model builder refuses by name (`_`-leading names
  become private attributes = silent drop; config/protected/dunder
  names crash raw); field-read-only fits are exempt, their names never
  become pydantic fields. DISTINCT on any scalar call — the bare sugar,
  the transform half, author UDFs and builtins — refuses at
  construction (measured: DuckDB binds DISTINCT only on aggregates; the
  flag used to drop silently). The whole-item lowering shares the field
  read's ecall site (P16 single-eval — the row path used to
  double-evaluate a call served both whole and by field). P14
  whole-struct NULL and the infer_arrow struct branch got their missing
  pins.

## Implementation slices (sequential standalone PRs)

1. **The split.** Recognize `x_fit`/`x_transform`, fit-scope clauses on
   the fit call, inline-nested consumption, bare-call sugar, delete the
   `OVER` sugar. Field reads only — a bare call as an output item still
   refuses (message updates to point at `.field` or the nested-output
   slice). Serving shape unchanged; confit untouched; all tests/docs/
   bench re-spell.
2. **Private columns (TASK-64) + θ laterals.** `_`-prefixed fields
   never cross the output boundary; `_th` becomes the ergonomic split
   spelling.
3. **`FILTER` on `tfm_fit`.** Small.
4. **Ordered fits.** `OrderSensitive` wrapper + in-call `ORDER BY` +
   the stable sort.
5. **Nested struct outputs.** Bare call as an output column, `s.*`,
   `unnest` — the output boundary learns struct columns on both paths
   (engine work; C2/C3 for struct-valued outputs).
6. **θ public export.** Rides on 5 — θ is just another struct output.

Order fixed only by dependency (2 needs TASK-64; 6 needs 5); 3 and 4
may slot anywhere after 1.
