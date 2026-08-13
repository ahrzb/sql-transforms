# `SQLProjection.marginalize` — deriving the `__FIT__` half

**Date:** 2026-08-13. **Status:** approved design, pre-implementation.
**Builds on:** `2026-08-11-row-wise-projections-design.md` (the class, the
gates, the leaf role), `2026-08-05-fit-transform-split-design.md` (the split,
the one sugar, the deleted `OVER` sugar), and the 2026-07-29 marginalization /
window-widening specs (the admission rule, ported).

## What it is

A rewrite in front of the ordinary constructor. The author writes a
`__THIS__`-only text; `marginalize` derives the `__FIT__` half and hands the
result to `SQLProjection(...)`:

```python
per_store = SQLProjection.marginalize("""
    SELECT (price - avg(price) OVER (PARTITION BY store))
           / stddev_pop(price) OVER (PARTITION BY store) AS z
    FROM __THIS__
""")
```

Everything downstream of the constructor — the row-wise gate, the key probes,
`fit`, `params`, `compile` — is the shipped machinery and cannot tell how the
`__FIT__` half got there. That was the deferred promise in the row-wise spec;
this document is the design that keeps it.

## The law (RFC M2)

`SQLTransform` is a superset of `SQLProjection`: every projection text is a
valid transform text meaning the same thing — projection adds the row-wise
obligation and gets serving in exchange. `marginalize` maps the transductive
corner of the transform space into projections, and the mapping is *defined*
by an executable identity:

```python
SQLProjection.marginalize(text).fit(F).transform(F)  ==  run(SQLTransform(text), F)
```

compared on row content, both sides sorted by a shared key (`SQLProjection`
promises input order out; plain `SQLTransform` does not).

Read it as: **freezing must be invisible on the training data itself.** A fit
scope in the text always means "fit over the partitions of the data flowing
through" — `marginalize` changes only *which data flows through it and when*:
at fit time the fit table flows through the scopes once and the θs are kept;
at transform time only the residual runs. On F the two sides agree
mechanically (each row joins its own partition's θ; `GROUP BY` keys are unique
by construction; no misses on the fit set). They part ways exactly where
marginalization should: on new data, the transform side joins frozen θ
(unseen partition → NULL), where the transductive side would refit.

So the same spelling has one meaning in both constructors, and the identity is
also the feature's parity gate — the right-hand side runs today, through the
existing leaf splice, with zero new code.

## Surface: no new syntax (RFC M1, amended)

`marginalize` introduces **zero** spellings. A marginalize text uses exactly
the vocabulary that already exists, and every fit scope in it freezes over
`__FIT__`:

```sql
avg(price) OVER (PARTITION BY store)   -- plain window aggregate → frozen per store
avg(price) OVER ()                     -- global scope → frozen, keyless
tfm(x)                                 -- the ONE sugar ≡ tfm_transform(tfm_fit(x) OVER (), x)
tfm_transform(tfm_fit(x) OVER (PARTITION BY store), x)   -- explicit windowed fit
```

The deleted `tfm(x) OVER w` spelling from the split spec **stays deleted**,
here and everywhere — its refusal ("fit scope moved to tfm_fit") is untouched.
Authors who want a windowed fit scope write the explicit split; authors who
want the global scope write the bare sugar. The 2026-08-05 objection — a
reader cannot tell fit scopes from runtime windows — is answered by
uniformity: inside a marginalize text there are *no* runtime windows. Every
`OVER` is a fit scope, every aggregate freezes, one rule with no exceptions.

Ruled out along the way (recorded, not revisitable without new evidence):

- **Calls-only v1** (freeze `tfm` calls, refuse plain aggregates): rejected.
  It mislabels a coherent fit scope as "not row-wise", fails the spec's own
  motivating example, and taxes every frozen statistic with a leaf definition.
- **Reintroducing the combined-with-`OVER` sugar as marginalize's surface**:
  rejected by the author of the original sketch — explicit spellings only.

## The rewrite

Every fit scope lowers to the same target shape:

```sql
-- derived from the per-store z-score above:
SELECT (t.price - cf_m0.cf_w0) / cf_m0.cf_w1 AS z
FROM __THIS__ t
LEFT JOIN (SELECT (store) AS cf_k0,
                  avg(price)        AS cf_w0,
                  stddev_pop(price) AS cf_w1
           FROM __FIT__ GROUP BY (store)) AS cf_m0
  ON (store) IS NOT DISTINCT FROM cf_m0.cf_k0
```

- **Derived names live in author space** (measured 2026-08-13, slice 2): the
  ordinary constructor *reserves* `__`, so `__cf_` spellings in the derived
  text would refuse at its own gate. The rewrite gensyms a fresh prefix
  against the author's identifiers (`cf_`, stepping aside to `cf1_` etc. on
  collision) — which also sharpens M3: the derived text is an ordinary author
  text, re-feedable to the constructor verbatim.

- **Join predicate is `IS NOT DISTINCT FROM`, never `=`.** Window
  `PARTITION BY` groups NULL keys into one partition; an equality join would
  drop them. Measured 2026-08-13: the constructor's key probes already
  recognize it (`COMPARE_NOT_DISTINCT_FROM` is in `_EQUALITIES`), and the join
  fits, compiles, and serves bit-exact through confit — including the NULL-key
  row joining its own partition's params. Both ends of the pipeline accept the
  derived shape as-is.
- **Scopes sharing a key tuple share a subquery** (one join per distinct
  partition spelling, one `cf_wN` column per scope). Keyless scopes share a
  one-row subquery joined `LEFT JOIN ... ON 1 = 1` — never `CROSS JOIN`
  (measured: the printer re-emits CROSS JOIN as a comma, which binds looser
  than a following LEFT JOIN and regroups the tree into a correlated join);
  the same respelling the row path's `_flattened` already does.
- **Key expressions** (`PARTITION BY date_trunc('month', ts)`) are computed on
  both sides — as the `GROUP BY` expression in the params subquery and
  verbatim in the `ON` clause.
- **Uniqueness by construction:** `GROUP BY keys` makes the params side unique
  per key tuple, so `KeyNotUnique` cannot fire on a derived join (the probes
  still run; they always pass).
- **Unseen partition → join miss → NULL θ → NULL output.** That is P14,
  already pinned. The fallback ladder therefore needs **no machinery**: the
  author writes `coalesce(z_fine, z_coarse, z_global)` over scopes at
  different partitions, each freezing into its own join, and COALESCE walks
  the ladder — oracle-clean, no new semantics.
- **Params are ordinary (RFC M4).** The derived subqueries become ordinary fit
  steps: frozen θs appear in `fitted.params` as readable per-key tables,
  exactly like a hand-written keyed join; `instances == {}` for pure-SQL
  scopes.

## Admission: what may freeze

Ported from the window-widening rule, which old `_marginalize.py` implements:
**a window's value must be a function of row-visible values** — the partition
keys plus, when `ORDER BY` discriminates (`RANGE`/`GROUPS` peers share
values), the order values. Physical position is the one thing a join key
cannot carry.

| construct | verdict |
|---|---|
| plain aggregate `OVER ()` / `OVER (PARTITION BY ...)` | freeze (a *bare* aggregate beside row columns is DuckDB's own binder refusal — the law inherits it) |
| `RANGE`/`GROUPS` frames where peers share values | freeze (keys = partition + order values) |
| `ROWS` frames, `row_number`, `ntile`, `lag`, `lead` (positional) | refuse — position is not a joinable key |
| `tfm(x)` (bare sugar) | freeze, keyless |
| `tfm_fit(x) OVER (PARTITION BY ...)` | freeze per key |
| `tfm_fit(x) OVER (ORDER BY ...)` (any ordered frame) | refuse — running fit, exactly as the split spec already rules |
| `tfm_fit(x) OVER (PARTITION BY ...)` where `tfm` is internally keyed (its own text joins `__FIT__` through a `GROUP BY`) | freeze — keys compose; effective key = scope keys ⊕ internal keys (next section) |
| scalar subquery over `__THIS__` | freeze (provably uncorrelated: `FROM __THIS__` has no alias in scope) |
| top-level `WHERE` / `GROUP BY` / modifiers over `__THIS__` | refuse — same constructs the row-wise gate refuses, same reasons |

Note the asymmetry is inherited, not invented: plain aggregates get the
widened vocabulary (the fit plan reruns the original computation, so even
order-sensitive aggregates freeze the value the text produced), while
`tfm_fit` keeps the split spec's stricter rule (an ordered frame means per-row
θ — a running fit, still a future feature).

## Key composition (RFC M5)

An internally keyed projection composes with a partitioned fit scope by
**concatenating keys**:

```
effective key  =  scope keys  ⊕  the projection's internal keys
```

```sql
-- tfm internally: ... FROM __FIT__ GROUP BY store ... ON t.store = f.store
-- usage site:     tfm_fit(x) OVER (PARTITION BY city)

-- flat lowering: one params table per (city, store), θ stays columns
LEFT JOIN (SELECT city AS cf_k0, store, avg(price) AS m
           FROM __FIT__ GROUP BY city, store) f
  ON t.city IS NOT DISTINCT FROM f.cf_k0      -- scope half: window discipline
 AND t.store = f.store                        -- internal half: verbatim

-- cf_k0 is the scope-key EXPRESSION, computed over __FIT__ and frozen under
-- a derived name (keys can be expressions — date_trunc('month', ts) — and
-- the author's name could collide with the projection's own columns); the
-- ON clause evaluates the same expression on the batch row. Derived names
-- are gensym'd in author space — see the naming rule under "The rewrite".
```

The two halves keep **different equality disciplines**, deliberately:

- **Scope keys are window keys.** `PARTITION BY` groups NULL keys into one
  partition, so the derived predicate is `IS NOT DISTINCT FROM` — same rule
  as every other fit scope.
- **Internal keys keep the author's spelling.** Under the UDF oracle reading,
  θ for one scope of a keyed projection is a per-key *table*, and the
  transform half looks the row's key up in it — the projection's own
  `ON t.store = f.store` is that lookup, and a NULL store misses. Rewriting
  its `=` would change the projection's meaning behind its author's back.

This is also why the lowering is a join and never window-key concatenation:
`avg(price) OVER (PARTITION BY city, store)` would hand NULL-store rows a θ
that the projection's own text denies them.

**How it landed (2026-08-13, slice 4) — an amendment.** The paragraph this
replaces claimed the transductive splice and the freeze would share the
lowering. They cannot: the lowering is *structural* (it adds a join), and a
window expression cannot nest grouping, so θ-as-table has no expression-level
splice. The host-side keyed-leaf refusal therefore **stays**
(`test_a_keyed_projection_refuses_the_leaf_role_by_name`), and the M2 law
does not extend to keyed scopes — their gate is the definition itself, pinned
executable: *each scope's answer equals the keyed projection fitted
standalone on that scope's rows.* Consequences:

- Keyed scopes are admitted **in one piece only** — the bare sugar or
  `tfm_transform(tfm_fit(bundle) OVER (PARTITION BY ...), bundle)` — because
  θ never exists as a value to pass around; a keyed fit scope anywhere else
  refuses by name.
- The admitted keyed shape, v1: exactly one grouped fit step joined once
  onto `__THIS__` through an AND-tree of column equalities; anything wider
  refuses naming the stem.
- Exported params columns wear derived names (the leaf's own names would be
  ambiguous against the spine in the ON clause).
- A bonus the struct-θ path lacks: keyed params are flat columns, so keyed
  scopes **serve** — `compile()` works where the keyless projection scope
  still refuses on struct reads.

Corollaries: bare `tfm(x)` on a keyed projection has no scope keys — the
effective key is the internal key alone, and the splice is just the
projection's own join, once per scope. A scope key colliding with an internal
key name needs no special case: both conjuncts apply and their conjunction is
the meaning (the NULL-safe scope half and the `=` internal half agree
everywhere except NULL keys, where the stricter `=` wins — exactly the
lookup semantics). `fitted.params` carries one table keyed by the full
effective key; uniqueness holds by construction (`GROUP BY` the effective
key); a miss on *either* half is P14 NULL.

## What `marginalize` emits (RFC M3)

Text. The rewrite parses the author's text, lowers fit scopes to the join
shape, renders SQL, and calls the ordinary constructor:

```python
@classmethod
def marginalize(cls, sql: str) -> SQLProjection:
    return cls(_derive_fit_half(sql))   # one code path below this line
```

Old `_marginalize.py`'s *rule table* (admission, key extraction, level
validation) is ported into the rewrite; its emitter (`FitStep` / `UDFSpec` /
`ParamsSpec` plan structures) dies. Constructing a `Program` directly was
considered and rejected: it creates a second way to build a projection and
breaks "a text you can read".

`marginalize` reads the caller's frame exactly like the constructors do —
`tfm` stems in the text resolve lexically at the `marginalize` call site.

## Attribution (the D3 concern, again)

The derived text is machine-written, so a refusal escaping from it would cite
SQL the author never wrote. The rewrite therefore carries its own obligation:

> **`marginalize` refuses in the author's vocabulary, or succeeds.** Every
> refusal fires during the rewrite, against the author's own text — naming
> their window, their key expression, in their spelling. A refusal escaping
> from the derived text is *our* bug.

That is a testable gate: for every text the rewrite accepts, the constructor
must accept the derived text. M2 buys most of it — whatever the transductive
reading refuses, marginalize refuses identically, before rewriting — and the
admission table above covers the rest.

## Properties and gates

| | |
|---|---|
| **law** | `marginalize(text).fit(F).transform(F) == run(SQLTransform(text), F)`, content-compared, both sides sorted — differential against the shipped splice, per admitted construct |
| **freeze** | on X with unseen partitions: NULL out (P14), where the transductive side refits — the divergence is *only* at misses |
| **attribution** | every refusal fires pre-rewrite, in the author's spelling; accepted text ⇒ derived text constructs (fuzzable) |
| **composition** | keyed leaf under a partitioned scope: effective key = scope ⊕ internal; the NULL-key rows pin the two disciplines (scope half matches its partition, internal half misses) |
| **params** | frozen θs readable in `fitted.params`; `instances == {}` for pure-SQL scopes |
| **serving** | derived projections compile like hand-written ones; the `IS NOT DISTINCT FROM` parity probe becomes a pinned test |
| **ladder** | a COALESCE over fine→coarse scopes equals per-level marginalize with fallback-on-NULL (executable example, not machinery) |

## Slices (sequential standalone PRs — this section is the plan)

1. **The bare sugar for projection leaves.** `tfm(x)` ≡
   `tfm_transform(tfm_fit(x) OVER (), x)` where `tfm` resolves to a
   `SQLProjection` — the split spec's ONE sugar, currently implemented only
   for registered `Transform`s. Host-general (works in any `SQLTransform`
   text), and marginalize needs it. Small: one new route beside the existing
   `stem_fit`/`stem_transform` dispatch in `_program`.
2. **The rewrite, plain scopes.** `_derive_fit_half` for plain aggregates and
   `PARTITION BY` windows (the common case): lowering to the join shape,
   shared-key coalescing of subqueries, key expressions, the law gate wired as
   a parametrized differential, the attribution gate, refusal table for
   everything not yet admitted (positional windows, ordered frames, top-level
   clauses — refuse by name now, widen later).
3. **Projection scopes.** `tfm(x)` and `tfm_fit(x) OVER (PARTITION BY ...)`
   inside marginalize texts, freezing through the same lowering; the
   COALESCE-ladder executable example; `fitted.params` pins for θ-as-data.
4. **Key composition.** The flat keyed lowering inside marginalize — merged
   join predicates, the two equality disciplines, exported columns under
   derived names — gated definitionally (per-scope standalone fits) plus
   miss and serving pins. The host splice keeps its keyed refusal; see the
   amendment under RFC M5 for why the law cannot reach here.
5. **The widened window vocabulary.** `RANGE`/`GROUPS` order-discriminating
   frames (keys = partition + order values) and scalar subqueries, ported from
   old `_marginalize` with the law gate extended over them.

Slices 1–4 deliver the feature; slice 5 completes parity with the old class's
vocabulary and can trail.

## Deferred

- **Running fits** (`tfm_fit(x) OVER (ORDER BY ...)`) — refused here exactly
  as the split spec refuses them; a future feature with its own design.
- **Serving projection scopes** (measured 2026-08-13, slice 3): the frozen θ
  crosses the derived join as a struct column, and the spliced residual reads
  it with `struct_extract` — which confit's v0 catalogue lacks. Batch
  transform is complete and lawful; `compile()` refuses with confit's own
  message, by name, until the row path's vocabulary grows structs. (Plain
  scopes serve fully — their frozen values are scalars.)
- **`FILTER`/`DISTINCT` on a projection fit scope** — the split hooks pass
  children only, so the clause would drop silently; refused by name instead.
- **The port and the old class's deletion** — unchanged from the row-wise
  spec. `marginalize` closes the *authoring* gap (`__THIS__`-only texts);
  the registry/unnest gaps remain the deletion trigger.
