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
