# Composition: projections as vocabulary members (DRAFT-24 loop 5)

Ruled 2026-08-04: second in the sequence (struct-valued calls → THIS →
private columns TASK-64 → the fit/transform-split epic DRAFT-25), WITH
partition composition as DRAFT-24 wrote it. Prior art: the July t-string
composition and fit-cascade specs were implemented and deleted in the
architecture reset; their surviving principles — the member is never
mutated, per-reference name scoping, inlining at one choke point — carry
into this design on today's chain/fit-plan machinery.

## Surface

```python
zscore = SQLProjection(          # authored against ITS schema: age, fare, country
    "SELECT sc(struct_pack(a := age)) OVER (PARTITION BY country).a AS z_age,"
    "       fare / avg(fare) OVER (PARTITION BY country) AS rel_fare"
    " FROM __THIS__",
    transformers={"sc": StandardScaler()},
)

risk = SQLProjection(
    "SELECT zs(struct_pack(age := years, fare := price, country := region))"
    " OVER (PARTITION BY store).z_age * weight AS r, name FROM __THIS__",
    transformers={"zs": zscore},          # ← an UNFITTED projection as a member
).fit(TRAIN)
```

A member is called exactly like a transformer: windowed, bundled,
**field-addressed** (bare member calls refuse — the struct-valued rule,
inherited). The bundle is the application: caller names map onto the
member's `__THIS__` columns, name-keyed.

## Semantics

- **T is authored, not learned.** A member's output struct is its select
  aliases — statically known. So an unknown addressed field
  (`zs(...).nope`) refuses at CONSTRUCTION, not fit: the P7 carve-out
  (fit-time errors for learned codomains) does not apply to members;
  only their INTERNAL transformers keep it.
- **Refit-through, member never mutated.** Fitting the caller fits the
  member's internals — through the adapter, on the caller's training
  stream — into CALLER-owned state (the member's transformer prototypes
  clone per group exactly as direct transformers do). Passing a FITTED
  projection refuses by name: frozen members are θ-as-argument, DRAFT-25's
  machinery.
- **Partition composition.** Call-site keys K prepend to every window and
  params join inside the member: zscore's `PARTITION BY country` becomes
  `PARTITION BY store, country`; a keyless internal window becomes
  `PARTITION BY store`; every member params table gains K key columns.
  `OVER ()` applies the member as authored. Call-site ORDER BY / frames /
  FILTER refuse (same rule as transformer windows).
- **Recursion refuses** at construction (object-identity cycle through the
  member graph), and member nesting is capped (a named refusal, cap 8 —
  well under the chain cap of 64 that bounds the spliced plan).
- **The oracle rule holds**: every construct inside the member means what
  it means; composition adds no expression semantics — only the fit
  machinery (plan splicing) and name hygiene.

## Lowering — namespace-born α-renaming, plan splice, β-reduction

No post-hoc rename exists or is needed. At the CALLER's construction, the
member's **authored SQL** is re-marginalized under a namespaced planner:
every minted family carries the member index — `__cf_m0_p{i}`,
`__cf_m0_tf{j}`, `__cf_m0_k{n}`, `__CF_M0_PARAMS_{i}__`,
`__CF_M0_LEVEL_{i}__` — so member and caller names are collision-free by
construction, and the resolver's existing rule (any `__cf_*` head is
opaque) protects them through every later rewrite. The `__cf_` reserved
gate on INPUT SQL is unaffected (the member's authored SQL is user SQL and
must not contain `__cf_`).

- **Plan splice**: the member's namespaced levels read an adapter
  projection of the caller's level table (bundle exprs materialized under
  the member's column names, K keys carried through); its fit and collapse
  steps join the caller's plan DAG in topological order; ParamsSpec and
  UDFSpec tuples concatenate.
- **Serving β-reduction**: `zs(...).z_age` substitutes the member's
  `z_age` item expression — its `__cf_t` column refs replaced by the
  caller's bundle serving expressions, its params references already
  namespaced — into the call site. This is the same substitution
  `rewrite_items` performs between chain levels. Extern calls inside the
  substituted expression share evaluation via the shared-site/CSE
  machinery; NO new dedup is introduced (per the ruling: further work
  deduplication waits on type-system purity tracking).
- A member output feeding ANOTHER transformer's bundle inside one level
  keeps today's rule (levels are the sequencing construct — spell it as a
  chain); unchanged by this loop.

## Refusals (all construction-time unless noted)

unknown member field (T is authored) · bare member call (struct value) ·
fitted member ("frozen members arrive with DRAFT-25") · recursion / depth
cap · call-site ORDER BY/frame/FILTER · bundle missing a column the
member references (fit-time when the member is schema-free; construction
when it declares `this_model`) · a member whose SQL the marginalizer
refuses reports the member's own refusal prefixed with the member name.

## Gates

- C4 extended: a composed projection compares against an INDEPENDENT
  reference that re-derives the member per (K × internal-keys) group —
  clone-fit-transform through the adapter by hand in the test.
- C3 serve_gate on composed projections (row ≡ batch, value for value).
- C2 rides for free (serving SQL is ordinary marginalized SQL after the
  splice); C1 round-trip and C5 corpus pins untouched.
- Name-hygiene test: caller and member each with joins + windows + a
  transformer; assert zero `__cf_` name collisions and member-object
  immutability after the caller's fit.
- Bench: one composition scenario (member with an internal window + a
  fitted transformer) vs its hand-inlined equivalent — the delta IS the
  composition overhead; expected ≈ 0 (symbolic inlining, no runtime
  boundary).
