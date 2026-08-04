# DRAFT-25 — Fit/transform split, θ handles, and the oracle rule for transform semantics

Status: **epic, parked by AmirHossein's sequencing (2026-08-04): finish
DRAFT-24 loop 5 (composition) and the entirety of that feature FIRST, then
come back and implement this.** Designed in-session 2026-08-04; every
DuckDB behavior below was measured that day, not recalled.

## The governing rule

For any construct involving a transform, the semantics is: **what would
DuckDB compute if the pieces were registered UDFs/UDAFs?** We add two
things only — the fit machinery, and error *timing* (anything DuckDB
would reject at bind/run refuses at construction or fit, by name — P7).
Accepted queries mean exactly what the oracle computes; we never invent
expression semantics. This is C2's serving contract lifted into the
authoring surface. (TASK-63's decisions were all instances of this rule
before it was named: CSE call-sharing, ASCII-case-insensitive field
matching + collision refusals, NULL struct → NULL fields.)

## The type semantics

A transform application is a UDF and a UDAF at the same time. Split them:

```
tfm_fit       : Agg[Struct<a,b,c>]  =>  Struct<type, id>       -- a true UDAF: rows of its scope → θ (a HANDLE)
tfm_transform : (Θ, Struct<a,b,c>) =>  Struct<f1,f2>           -- a true scalar UDF
tfm(x) OVER w   ≡   tfm_transform(tfm_fit(x) OVER w, x)        -- today's spelling = sugar
```

θ = `Struct<type, id>` — a handle, not the parameters. This is ALREADY
the wire format: today the id is the params column `__cf_est` (joined per
group) and the type lives statically in the minted UDF's name
(`__cf_tf0` binds to its own instances store). The rework fuses them into
one first-class value.

- The `type` tag earns its place exactly when θ becomes nameable: with
  bare ids, `sc_transform(pca_fit(x) OVER (), x)` silently indexes the
  wrong store — the positional-rewiring failure class, one level up.
  Tagged, it refuses at build where provenance is visible, traps as a
  broken artifact otherwise.
- Marginalization stops being transform-specific: it is the SAME rewrite
  as for `avg` — evaluate every aggregate over the training stream, park
  the result (params table = the materialized `tfm_fit` per group),
  reference it at serving. sklearn's own fit/transform API becomes the
  SQL type story; the registry entry provides both halves.

## Output semantics: nested by default, expansion is native

A call is ONE value (true of every UDF everywhere). Columns are
presentation, via DuckDB's own expansion constructs — we own NO
presentation rules:

```sql
SELECT tfm(__THIS__) AS features, name FROM __THIS__
-- Struct<features: Struct<f1,f2>, name>            nested: what a call IS

SELECT unnest(tfm(__THIS__)), name FROM __THIS__
-- Struct<f1, f2, name>                              measured: unnest expands, alias ignored

SELECT s.*, name FROM (SELECT tfm(__THIS__) AS s, name FROM __THIS__)
-- Struct<f1, f2, name>                              measured: a NAMED struct column
--                                                   star-expands; one call per row (counted);
--                                                   NULL struct row → NULL fields
```

Measured refusals that stay refusals: `(expr).*` is a parser error;
multi-column scalar subqueries refuse (a subquery becomes a struct only
via its single column, incl. `(SELECT o FROM o WHERE …)` — the row-struct
spelling; `SELECT t FROM t` is the row as a struct).

**This DELETES owned rules rather than adding them**: the bare-wide-item
expansion (loop 3's `AS e` → `e_pca0`, `e_pca1` flat alias-prefixing)
goes away entirely — a width-k call is not a special item, it is a value.
The alias question disappears (`AS e` names the struct). P16's list/flat
boundary clauses get rewritten around "struct crosses the OUTPUT boundary
where the author left it nested; nothing struct-shaped flows INTO
computation (bundles still destructure; θ passes by reference)".

Also on the deletion list (ruled 2026-08-04, and **pulled forward into
the immediate "struct-valued transform calls" loop** rather than waiting
for this epic): loop 1/4's width-1 scalar-valuedness — `sc(...) * 10`
treating a one-field struct as a bare scalar, and the fit-time collapse
of width-1 field reads to the bare call. Under the oracle rule a call is
struct-valued at every width; only field reads are scalars. Until this
epic lands the nested output boundary, a bare transformer call as a
select item refuses by name instead of expanding or unwrapping.

## What the split makes expressible (previously unspellable)

```sql
-- fit HERE, apply THERE:
tfm_transform(tfm_fit(struct_pack(v := a)) OVER (), struct_pack(v := b))

-- global and per-group θ of the same transform, simultaneously:
tfm_transform(tfm_fit(x) OVER (),               x) AS f_global,
tfm_transform(tfm_fit(x) OVER (PARTITION BY g), x) AS f_local

-- the leakage question becomes a SPELLING (native aggregate FILTER), not a policy mystery:
tfm_transform(tfm_fit(x) FILTER (WHERE split = 'train') OVER (), x)

-- frozen application = θ as an ordinary argument (composition's "frozen member" fork):
tfm_transform({'type': 'sc', 'id': 3}, x)

-- the fit artifact as data:
SELECT g, tfm_fit(x) AS theta FROM __THIS__ GROUP BY g
```

## Open edges (the real design work when this un-parks)

1. **Handle stability across fits.** A θ that outlives the fit that
   minted it (the frozen spelling) names a slot unless identity is
   fit-generation- or content-keyed; a dangling handle must refuse
   loudly, never resolve to whatever now occupies the id. This is where θ
   stops being an implementation detail and becomes the artifact
   contract (ties to parked artifact serialization — which this reshapes
   into "serialize the handle + the store it points into"; the
   params+instances pair is already the serving artifact).
2. **Fit as a lawful aggregate.** SQL aggregates are functions of the
   multiset (unordered). Order-sensitive fits (iterative solvers, sign
   conventions) violate that; today it's papered over operationally
   (threads=1). The native escape is ordered-set aggregates
   (`WITHIN GROUP (ORDER BY …)`) — DRAFT-21's order-keyed windows wearing
   their true name.
3. **θ across the output boundary** (`SELECT tfm_fit(x) …` as an output
   column): legal-looking under the rule; what serializes is the handle —
   decide what a consumer can DO with it.
4. **`type` tag: value-level (storable, as drafted) vs type-level
   (phantom-typed Θ<tfm>, static mixing errors)** — or both.
5. Whether `tfm_fit`/`tfm_transform` are user-visible surface or only the
   semantic ground truth under the `tfm(x)` sugar.
6. **Work deduplication is future work, gated on type-system support**
   (AmirHossein, 2026-08-04). Evaluation sharing beyond what already
   shipped (the shared-site ecall + DuckDB CSE for textually identical
   pure calls) — cross-member sharing in composition, cross-projection
   CSE, dedup of repeated private-column subexpressions — is only safe
   for provably pure computations; a stochastic or volatile transform
   shared across mentions is silently wrong. Before any further dedup,
   the type system needs a purity/volatility distinction so safety is
   checked, not assumed (P15 declares purity today but nothing marks the
   exceptions).

## Native-boundary facts (measured 2026-08-04, for the record)

- Python UDAFs do not exist in DuckDB (no aggregate-function API): the
  UDAF half is semantics-by-analogy, never executable — marginalization
  consumes it, exactly as it already consumes the OVER spelling.
- Table functions are name-parameterized natively (`query_table` +
  macros; bare table names work) but cannot chain (`f(g(t))` is a binder
  error) and cannot take subqueries — a value-parameterized transform
  pipeline has no native reading, which is why the expression/UDAF
  framing won over the table-function framing.

## Sequencing

Ruled 2026-08-04, in order:
1. **Struct-valued transform calls** (the pulled-forward deletions above:
   no width-1 auto-unwrap, no flat expansion; bare calls refuse until the
   nested boundary arrives).
2. **DRAFT-24 loop 5 — composition**, WITH partition composition
   (call-site keys prepending to member windows) as the draft wrote it.
   Refit-through members only; FROZEN members are this epic's machinery
   (θ as argument) and wait for it.
3. **Private columns** (TASK-64: `_`-prefixed output fields never cross
   the output boundary; lateral-alias sugar lowered by substitution).
4. This epic.
