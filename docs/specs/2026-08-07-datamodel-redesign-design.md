# SQLTransform: a two-argument relation function (v0)

Date: 2026-08-07. Status: designed with AmirHossein (this session).
Every DuckDB behavior cited was measured on 2026-08-06/07 against 1.5.5.

## What broke

The current model decides on its own which windows to freeze. Measured on a
projection fitted with `avg(price) OVER (ORDER BY ts ROWS 2 PRECEDING)`:

```
seen   ts=3, price=999  ->  ma = 20.0    the row's own value is ignored
unseen ts=9, price=50   ->  ma = None
```

Both wrong, neither raises — C5's one unrecoverable state ("no third mode":
every query either serves or refuses by name; one that builds and quietly
computes something else is unrecoverable). Nothing in the text said this
would be frozen, so nothing in the text can warn you.

## The model

> **A transform is a function `(F, T) -> R` over relations.**

`__FIT__` and `__THIS__` are its two parameters. At the top level they are
bound by `fit` and `transform`; at a call site they are passed explicitly.

```sql
SELECT (t.price - s.a) / s.s AS z
FROM __THIS__ t, (SELECT avg(price) a, stddev_pop(price) s FROM __FIT__) s
```

Which half is learned and which is live is read off the text. There is no
annotation to remember and none to forget.

In Python, `fit` is partial application:

```python
z : F -> Fitted
Fitted : (T -> R) with .params

af = a(D)          # Fitted — callable, and carries its captured environment as data
af(X)              # serving
af.params          # the artifact
```

`.params` is the reified closure. A plain Python closure would be
type-correct and unshippable — `lambda F: lambda T: ref(F, T)` captures the
whole training set and nothing outside can tell. Reifying it makes
`len(af.params)` against `len(D)` a number you can print, so well-behavedness
is a measurement rather than a rule.

SQL has no juxtaposition, so the curried spelling is Python-only. Measured:
`z(f)(t)` is a Parser Error in FROM, LATERAL, scalar and nested positions;
`z(f, t)` parses. The SQL surface is the uncurried two-argument call, which is
an ordinary table function and needs no new syntax.

## Freezing

> **Every maximal subquery whose leaves are all `__FIT__` and constants is
> evaluated once at fit and replaced by a table.**

Those tables are `params`. "Maximal" does the work: a relation mixing both
parameters is not frozen wholesale — its `__FIT__`-only subtrees are, and the
rest stays live.

```python
def fit(self, data):
    params = {name: run(sub, data) for name, sub in fit_only_subtrees(self)}
    return Fitted(substitute(self, params), params)
```

A transform never mentioning `__FIT__` is stateless: `fit` is a no-op and
`params` is empty.

Mechanism, measured: the repo parses with DuckDB's own `json_serialize_sql` /
`json_deserialize_sql`, and replacing a relation node with a `BASE_TABLE` node
naming a frozen table works end to end. A new batch demonstrably used frozen
parameters (`[0.5, 6.66]`) rather than recomputing (`[1.0, 1.0]`).

### The one refusal

A `__FIT__` subtree may not correlate to **`__THIS__`**.

```sql
SELECT (SELECT avg(price) FROM __FIT__ f WHERE f.cat = t.cat) AS m FROM __THIS__ t
-- CorrelatedFit: __FIT__ subquery references t.cat from the outer query
```

That subquery is per-serving-row, so it cannot be evaluated once into a table.
Supporting it means lifting the correlation to a `GROUP BY` and rewriting it
as a join — which is marginalization. **Future work, not a permanent
boundary**: it is the natural first surface marginalization should buy, and it
defers with it.

Correlation *inside* a closed `__FIT__` subtree is fine — it evaluates once as
a whole. That distinction is what lets per-group work with no new machinery.

Refused at construction (P7 — "refusals are construction-time and named":
refused at construction with an error naming the construct, never at fit,
never at serve, never silently).

Nothing else refuses.

### Ordered frames are legal

The failure that started this redesign is admitted here, written out:

```sql
SELECT t.price, ma.m FROM __THIS__ t
LEFT JOIN (SELECT ts, avg(price) OVER (ORDER BY ts ROWS 2 PRECEDING) AS m FROM __FIT__) ma
     USING (ts)
```

It says *join the training data on `ts`*. The hazard has not gone away — a
serving row with a new `ts` still gets NULL forever — but it is visible in the
text instead of hidden behind an inferred key, and the NULL arrives through
P14 ("the one NULL story": unseen group ⇒ LEFT JOIN miss ⇒ NULL output;
NULL-ness always flows through join data, never a lookup convention).

**Deliberately not refused.** The refusal existed because freezing was
implicit; once the author writes the join key, refusing overrules something
they can see.

## Calling a transform

### Nesting

```python
z     = SQLTransform("... FROM __THIS__ t, (SELECT avg(price) m FROM __FIT__) s")
outer = SQLTransform("SELECT * FROM z(__FIT__, __THIS__) WHERE z > 0")
```

Splice the member's SQL at the call site with its two parameters bound to the
arguments. One `fit` freezes everything.

**Rename the member's free references, not its own definitions.** Measured:
splicing a member as a parenthesised derived table already scopes its CTEs, so
its own definitions cannot collide with the outer's. The hazard runs the other
way — a *free* name the member resolved from the caller's frame gets captured
by an outer CTE that happens to share it:

```
member alone                        [20.0, 40.0, 60.0]           scale.factor = 2.0
spliced under `WITH scale AS (...)`  [10000.0, 20000.0, 30000.0]  captured: 1000.0
free references renamed              [20.0, 40.0, 60.0]           correct
```

Silent, no error, different numbers. So each free reference is rewritten to a
unique name (`z__scale`) bound to the object resolution already captured at
construction — ordinary capture-avoiding substitution. Frozen tables are named
the same way, so `params["z__cs"]` still says where each came from.

**Splice, never emit a DuckDB macro.** Measured: a table macro invoked under
`LATERAL` does not see the correlation and silently returns the whole-table
answer for every group.

```
LATERAL <spliced body>   S1/book=15,    S1/toy=30,    S2/book=100,  S2/toy=250
LATERAL m('t')           S1/book=43.3,  S1/toy=176.7, S2/book=43.3, S2/toy=176.7
```

It runs, returns the right shape, and is wrong. Splicing avoids it entirely.

Nesting depth is capped at 8, refusing by name beyond that.

### Chaining

Each stage appears twice, once per parameter:

```sql
FROM b( a(__FIT__, __FIT__),        -- b's training input: a fit-transformed on training data
        a(__FIT__, __THIS__) )      -- b's serving input
```

`a` runs twice at fit and once at serving. That is forced — b's fit data *is*
a's transformed training data — and writing it out is what makes it visible.

Measured, the two readings are distinguishable, so a gate on this bites:

```
b fit on a's transformed training data   S1 [0.0, 0.0667, 0.1333]
b fit on raw training data               S1 [0.0, 0.0625, 0.1250]
```

### Per group

`LATERAL` and a `WHERE`. No new syntax, no engine reaching inside the member.

```sql
SELECT x.* FROM (SELECT DISTINCT store FROM __FIT__) g,
  LATERAL z((FROM __FIT__  WHERE store = g.store),
            (FROM __THIS__ WHERE store = g.store)) x
```

**Both parameters must be sliced.** Filtering only `__FIT__` leaves `__THIS__`
whole, so every group crosses every row — measured, 4 rows in and 8 out, with
S1's statistics applied to S2's rows.

The member needs no notion of the group. A normalizer that z-scores globally
and never mentions `store` becomes per-store purely by what it is passed.

Rejected, with the reason each failed:

| approach | why not |
| --- | --- |
| `z(__THIS__, partition_by := [...])` | the engine rewrites the member's internals; nothing is readable at the call site |
| folding the key into the member's `GROUP BY` | agrees with `LATERAL` only for keyed reductions — on a top-N member it returned one group instead of all: `[('S2',300),('S2',200)]` |
| `z(t) PARTITION BY store`, `PER`, `USING`, `FOR EACH` | Parser Errors, all of them |
| `z(t) GROUP BY store` | parses, but already means something in SQL, so our reading would silently differ from the oracle's on identical text |

## Name resolution

Unresolved identifiers resolve against the caller's frame, exactly as
`duckdb.sql("SELECT * FROM my_df")` does. Locals first, then globals.

```python
store_dim = pa.table(...)                                  # a static table
z         = SQLTransform("... FROM __FIT__ ...")           # another transform

outer = SQLTransform("SELECT * FROM z(__FIT__, __THIS__)")
```

**Resolution happens once, at construction, and captures by value.** The frame
is not retained. Rebinding `z` afterwards does not change `outer`. A name
resolving to nothing refuses at construction, naming the identifier.

Skipped: an explicit-binding escape hatch for transforms built where the name
is not a local. DuckDB keeps `con.register` for this; add it when it bites.

## Foreign transforms

A transform written in Python supplies the pair directly:

```python
z = Transform(
    fit       = lambda F:    pa.table({"m": [pc.mean(F["price"]).as_py()]}),
    transform = lambda p, T: pa.table({"z": pc.divide(T["price"], p["m"][0])}),
)
```

An sklearn transformer is already this pair — `x_fit` is the UDAF half,
`x_transform` the UDF half:

```sql
SELECT sc_transform(f.theta, struct_pack(v := t.price)).v AS z
FROM __THIS__ t
LEFT JOIN (SELECT cat, sc_fit(struct_pack(v := price)) AS theta FROM __FIT__ GROUP BY cat) f
     USING (cat)
```

`sc` resolves from the caller's frame; `sc_fit`/`sc_transform` are found by
stripping the suffix. **θ is an opaque handle**, `Struct<type, id>` into a
registry of fitted estimators, so `Fitted` carries `instances` alongside
`params`. An SQL leaf gives an inspectable, shippable params table; a fitted
`RandomForest` gives a pointer. An id present but absent from `instances` is a
broken artifact and raises.

`sc_fit` over `__THIS__` is legal and means what it says — refit on the batch
you were handed. Under this design you cannot write it by accident, because
you had to type `__THIS__`.

Skipped: bare `sc(x)` sugar for global fit-transform.

### Why the pair, and not one function

An opaque `(F, T) -> R` cannot be split, so it must retain the training data:

```
z_sql.fit(D)        evaluate __FIT__ subtrees      params = {stats: 1 row}
z_sql.transform(X)  run residual against X         D never touched again

z_opaque.fit(D)     retain D                       params = {__fit__: |D| rows}
z_opaque.transform(X)  z(D, X)                     z sees all of D on every call
```

Same numbers, different artifact: serving one row costs `|D|`, per-group
multiplies it, and the training data ships with the model. So the one-argument
composite is the **oracle**, not the implementation — see the gates.

## Properties and gates

Named descriptively so citing one carries its meaning. Where a property
restates a law from `docs/properties.md` or `docs/kpis.md`, that is noted.

### The oracle

DuckDB's Python API cannot register a Python table function — `table_function`
only *calls* a named one. That is a benefit: the reference gets computed in
pyarrow, sharing no engine, no control flow, and no code with the SQL path, so
it cannot agree by sharing a bug.

| Property | Statement | Gate |
| --- | --- | --- |
| **Pair equals composite** | `z.transform(z.fit(F), T) == z_ref(F, T)` | `z_ref` is the opaque pyarrow version. The composite specifies the numbers; the pair must hit them while shipping params |
| **Per-group matches a Python loop** | `LATERAL` over sliced parameters equals per-group calls | Build the expected table with `pa.concat_tables([z_ref(F_g, T_g) for g in groups])`. Measured: S1 `[0.5, 1.0, 1.5]`, S2 `[0.333, 1.0, 1.667]` |
| **Chaining fits on transformed data** | `b`'s fit input is `a(F, F)`, not `F` | The two readings differ — `[0.0, 0.0667, 0.1333]` versus `[0.0, 0.0625, 0.1250]` — so the gate fails when the implementation picks wrong |

### Freezing

| Property | Statement | Gate |
| --- | --- | --- |
| **Freezing is complete** | Every maximal `__FIT__`-only subtree appears in `params`; nothing reading `__FIT__` survives into the residual | Two `__FIT__` CTEs plus one mixed relation; assert `params` has exactly the two and `"__FIT__" not in residual.sql` |
| **Freezing is faithful** | `t.fit(D).transform(D) == run(t, D)`, where `run` binds both parameters to `D`. **Column names too, not only values** (decided 2026-08-08, TASK-72): DuckDB names an unaliased select item after its printed text, so a rewrite that does not pin the name changes the schema and leaks `__param_N` into `get_feature_names_out()` | Property test over the corpus. The reference side is a binding, not a rewrite |
| **Freezing is observable** | For `D' != D`, `fit(D).transform(D')` uses `D`'s parameters | Measured: frozen `[0.5, 6.66]` versus recomputed `[1.0, 1.0]`. **Load-bearing — see below** |
| **Freezing is deterministic** | `fit(D)` twice yields equal params | Equality on two fits of the same data |
| **Fit ignores `__THIS__`** | `fit` reads only `__FIT__`; no serving batch need exist | `fit(D)` succeeds with nothing bound to `__THIS__` |
| **Statelessness is real** | A transform never naming `__FIT__` has empty params and a no-op `fit` | `fit(D1).transform(X) == fit(D2).transform(X)` for arbitrary `D1 != D2` |
| **Substitution is surgical** | The residual differs from the original only at the frozen subtrees | Text comparison against a hand-written expected residual |
| **Params are measurable** | `len(params)` is inspectable, so a training-set-retaining transform is visible | Assert a well-behaved transform's params are `O(1)` in `len(D)`; assert the degenerate `fit=identity` form reports `|D|` |

**"Freezing is observable" is what keeps "freezing is faithful" from being
vacuous.** An implementation that freezes nothing — `fit` a no-op, `transform`
rebinding `__FIT__` to the serving batch — passes faithfulness on every input.
The law only bites when paired with a second dataset. A suite pinning the
first without the second is testing nothing.

### Refusals

All at construction (P7).

| Property | Statement | Gate |
| --- | --- | --- |
| **`__THIS__`-correlated `__FIT__` refuses** | A `__FIT__` subquery referencing a `__THIS__` column raises `CorrelatedFit` | Construction raises, naming the correlated column. Assert it raises at construction, **not** at fit or transform |
| **Internal correlation is allowed** | A `LATERAL` inside a closed `__FIT__` subtree is admitted | The per-group example constructs, fits, and matches the pyarrow reference |
| **Unknown name refuses** | A name resolving to nothing raises, naming the identifier | Construction with an unbound name |
| **Depth cap holds** | Nesting deeper than 8 refuses by name | 9-deep raises; 8-deep succeeds |
| **No third mode** | Every transform serves or refuses by name (C5) | Corpus FAILED bucket pinned empty |

### Deliberate permissiveness

Cases this design allows that a previous draft refused. Each needs a gate
pinning current behaviour, so a future change cannot quietly alter it.

| Property | Statement | Gate |
| --- | --- | --- |
| **Ordered frames over `__FIT__` are legal** | An order-keyed frame over `__FIT__` freezes and joins by the order key | Pin it: a seen key returns the training value *ignoring the row's own*, an unseen key returns NULL. Both asserted, with a comment saying it is intended |
| **`__THIS__`-side aggregation is live** | An aggregate over `__THIS__` means what DuckDB means | Batch-dependent answers across two batches; nothing in params |
| **Transductive `sc_fit` is legal** | `sc_fit` over `__THIS__` refits per call | Two batches yield different parameters |

The ordered-frame gate pins a result that *looks like a bug*, which is why the
test must say it is intended — otherwise it gets "fixed" in a year and we are
back where this started.

### Calling

| Property | Statement | Gate |
| --- | --- | --- |
| **Both parameters slice** | Per-group passes sliced `__FIT__` *and* `__THIS__` | Filtering only `__FIT__` yields 8 rows from 4 with cross-group statistics; the correct form yields 4 |
| **Members splice, never macro** | No DuckDB macro is created for a member | A member applied per group returns per-group values, not the whole-table value repeated |
| **Splicing equals materializing** | A nested call behaves exactly as if the member were a registered table function | A table function call *is* its materialized output, so: compute the member with the pyarrow reference, `con.register` the result, run the outer query over that table, and require the spliced result to match. This is the nesting oracle |
| **Splicing is capture-free** | An outer CTE cannot rebind a name the member resolved from the caller's frame | Hostile outer: `WITH scale AS (SELECT 1000.0 AS factor)` wrapping a member that freely references a registered `scale`. Must give `[20, 40, 60]`, not `[10000, 20000, 30000]`. **This is the gate a naive splice fails** |
| **Splicing is text-checkable** | Spliced SQL equals a hand-written equivalent | Text comparison — reports *where* it broke, not just *that* it did |

### Name resolution

| Property | Statement | Gate |
| --- | --- | --- |
| **Capture is by value** | Rebinding a name after construction does not change the built transform | Build `outer` from `z`, rebind `z`, assert unchanged |
| **No frame is retained** | Construction holds no reference to the caller's frame | Construct inside a function, drop it, assert collection via `weakref` |

### Foreign transforms

| Property | Statement | Gate |
| --- | --- | --- |
| **Leaves need no special case** | An sklearn leaf freezes by the same rule as SQL | Faithfulness with a `StandardScaler` leaf |
| **The one NULL story** | Unseen key ⇒ join miss ⇒ NULL θ ⇒ NULL output, never a dropped row (P14) | Unseen key: the row is present, output NULL |
| **Broken artifacts raise** | A θ id absent from `instances` raises, never NULL | Delete an instance, transform, expect a raise naming the id |

## Slices

1. **The two parameters, freezing, `fit`/`transform`.** Single level, no
   nesting. Find maximal `__FIT__`-only subtrees, refuse `__THIS__`-correlated
   ones, freeze at fit, substitute at transform.
   Gates: **The oracle** (pair equals composite), all of **Freezing**,
   `CorrelatedFit` and **No third mode**, and both `__THIS__`-side entries
   under **Deliberate permissiveness**.
2. **Calling: nesting, chaining, per group.** Frame lookup, splice,
   name-prefix, depth cap.
   Gates: all of **Calling** and **Name resolution**, plus per-group and
   chaining from **The oracle**, **Unknown name refuses**, **Depth cap holds**.
3. **Foreign transforms.** The Python pair, `x_fit`/`x_transform`, θ handles,
   the instance registry.
   Gates: all of **Foreign transforms**, plus **Transductive `sc_fit`**.

The ordered-frame gate belongs to slice 1 and should be written first in that
slice. It is the property most likely to be "corrected" by someone who does
not know it is deliberate.

TDD, red before green. Lands alongside the existing modules in
`sql_transform/model/`; nothing is deleted until the old implementation and
the new one agree on the corpus.

## Deferred

- **Marginalization** — automatic lifting of aggregates over `__THIS__` into
  frozen relations. Everything is hand-written here.
- **Correlated `__FIT__` subqueries** — refused by `CorrelatedFit` in slice 1.
  Enabling it *is* marginalization, so it is the first thing that work should
  buy.
- **`SQLProjection`** as a second type, defined by row-locality of the
  residual, and **`compile()` / `Inference`** as the serving artifact and row
  path.
- **Freezing a constant.** A relation over a static table reads neither
  parameter, so nothing marks it for freezing. Pure performance, no semantics.
