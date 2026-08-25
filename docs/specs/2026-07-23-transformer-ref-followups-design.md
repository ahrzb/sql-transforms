# TASK-3 — Transformer-ref follow-ups (Part-2 authoring surface)

Design for the six acceptance criteria on TASK-3. Specs it builds on: doc-8
(composition), doc-7 (execution model). DataFusion remains the parity oracle
(decision-1).

Everything below was measured against the code at `2c9be9b`, not inferred. Where
the ticket text and the measured behaviour disagree, the measurement wins and the
disagreement is called out.

## Findings that changed the ticket

1. **The ticket's documented AC#5 workaround is broken.** `obj.feature_names_in_ =
   ["age", "income"]` fails — the native engine calls `.tolist()` on it
   (`src/lib.rs:114`), so a plain list raises `ValueError: could not read
   feature_names_in_: AttributeError: 'list' object has no attribute 'tolist'`. It
   needs `np.array([...])`. The design removes the need for the workaround entirely,
   so it is not documented.

2. **AC#5 cannot be solved by positional binding in Python alone.** The native
   engine reads `feature_names_in_` off the object at `InferFn` build
   (`src/lib.rs:103`) and reorders struct fields to that order (`src/expr.rs:414`),
   hard-erroring when the attribute is absent. Genuine positional binding would
   require Rust changes, a maturin rebuild, and a new parity surface. The design
   avoids all of that by synthesising the names onto a copy (§1).

3. **AC#4 is not busywork, but not for the reason the ticket gives.** The ticket
   calls a 3-level nesting test "low value". It becomes load-bearing *because of
   AC#1*: the new `consumed` set decides materialisation by nesting position, and
   3-level is the only shape where a ref is simultaneously consumed and a consumer.
   Existing coverage stops at 2 levels.

4. **AC#6 is not busywork.** The README is 139 lines and contains no
   transformer-ref documentation at all, while the output shape (a single Arrow
   struct column) is a real handoff trap.

5. **Out of scope, reported separately.** SQL named arguments are unreachable for
   our UDF: DataFusion 54's parser accepts `f(x => v)` and `f(x := v)`, but it is a
   per-function Rust-side `ScalarUDF` capability with no Python binding. Recorded by
   the PM as DRAFT-11. A scalar subquery in the projection also passes
   `parse_and_validate` unchecked; noted, not actioned — it maps to no AC.

## §1 Transformer detection and column binding (AC#5, AC#2b)

`is_transformer` currently gates on `feature_names_in_`, so an ndarray-fit
transformer is not recognised as a transformer at all. Replace the predicate with
`n_features_in_`, which `fit()` sets for both ndarray and DataFrame input and which
is absent until fitted:

| case | `transform` | `n_features_in_` | `feature_names_in_` |
|---|---|---|---|
| unfitted StandardScaler | yes | — | — |
| fitted on DataFrame | yes | 2 | `['age' 'income']` |
| fitted on ndarray | yes | 2 | — |
| OneHotEncoder, ndarray-fit | yes | 1 | — |

```python
def is_transformer(obj) -> bool:
    return hasattr(obj, "transform") and hasattr(obj, "n_features_in_")
```

Names are **metadata** — they travel as Arrow struct field names on the synthesised
`named_struct`, and both engines align on them. When sklearn did not record them,
synthesise them from the call site onto a copy, preserving doc-8's clone contract:

```python
feat = getattr(obj, "feature_names_in_", None)
if feat is None:
    if len(cols) != obj.n_features_in_:
        raise ValueError(
            f"{name} takes {obj.n_features_in_} columns (fitted without names, so "
            f"arguments bind positionally in call order), got {len(cols)}: {cols}")
    obj = copy.copy(obj)                    # never mutate the user's object
    obj.feature_names_in_ = np.array(cols)
elif set(cols) != {str(n) for n in feat}:
    raise ValueError(f"{name} columns {cols} must match feature_names_in_ {feat}")
```

`copy.copy` is shallow, so fitted state is shared rather than duplicated. Verified
end to end: original untouched, `transform == infer`, output matches sklearn, no
`src/*.rs` change.

**Accepted risk.** On the positional path only arity is checkable, not order, so
`{sc}(income, age)` against a transformer fitted as `[age, income]` silently swaps
features. This is identical to calling sklearn directly. Documented in §5; no
warning is emitted.

The same predicate yields AC#2b's error, replacing a misleading `TypeError` that
blamed the interpolation type rather than fittedness:

```python
if hasattr(v, "transform") and not hasattr(v, "n_features_in_"):
    raise ValueError(f"interpolation {{{item.expression}}}: {type(v).__name__} is "
                     f"not fitted -- call .fit(...) before referencing it")
```

**Behaviour change:** an unfitted transformer now raises `ValueError`, not
`TypeError`. Acceptable under v0/no-backward-compat.

Inference is unaffected. A transformer that becomes unfitted after `fit()` already
raises cleanly on the native path (`ValueError: transformer.transform failed:
NotFittedError: ...`); that path is not touched.

## §2 Probe once, materialise only when consumed (AC#1)

Every leaf ref calls `.transform()` twice at fit: once in `_derive_schemas` to
derive the output schema, once in `_materialize` to build a table that only an
*outer* ref's probe ever reads. For a leaf with no outer, that table is discarded.

This is the fit-time half of doc-7's "fuse at inference, stage at fit". Fit is a
staged cascade — the outer's schema depends on the inner's real output, so the
inner must be transformed forward. A leaf with no consumer has no next stage.

```python
def _probe(obj, cols, table) -> tuple[pa.Schema, pa.Schema, np.ndarray]:
    ...                                   # as _derive_schemas, also returns y

def _table_from_probe(y, out_schema) -> pa.Table:
    return pa.table([pa.array(y[:, i], type=f.type) for i, f in enumerate(out_schema)],
                    schema=out_schema)

# MUST be computed before any resolution: resolve() rewrites call args into a
# named_struct, destroying the nested-arg signal call_arg_ref() reads.
consumed = {inner for n in tfm_refs if (inner := call_arg_ref(_find_call(select, n)))}

# src is the training table for a leaf ref, or the inner ref's materialised
# output when this call's argument is another ref (resolved innermost-first).
src = table if inner is None else materialized[inner]
in_schema, out_schema, y = _probe(obj, cols, src)
if name in consumed:
    materialized[name] = _table_from_probe(y, out_schema)
```

`materialized` currently doubles as the already-resolved guard (`if name in
materialized: return`). Since leaves no longer populate it, that guard moves to a
separate `resolved: set[str]`.

`_materialize` is deleted. Call counts: `{sc}(a, b)` 2 → 1; `{pca}({sc}(a, b))` 4 → 2.

## §3 Aggregate-over-output pre-check (AC#2a)

Today `AVG({sc}(age, income)) OVER ()` fails with `Error during planning: Invalid
function '__compose_0__'. Did you mean 'power'?` — it names an internal placeholder
the user never wrote and suggests an unrelated function.

Cause: at fit, `find_window_aggregates` freezes the aggregate by evaluating it in
DataFusion before the transformer UDF is registered. The transformer's output does
not exist at that moment. This is a real limitation of the opaque mechanism
(decision-3), not a bug.

Expressing it properly is inherently two-stage — materialise the output, then
aggregate it — which needs a subquery. `parse_and_validate` rejects a subquery in
`FROM` ("FROM clause is required and must be a plain table"), so the surface cannot
express it at all.

```python
def _in_window_agg(node: exp.Expression) -> bool:
    p = node.parent
    while p is not None:
        if isinstance(p, exp.Window):
            return True
        p = p.parent
    return False

if _in_window_agg(n):
    raise ValueError(
        f"{name} output cannot feed a window aggregate: aggregating over transformer "
        f"output is inherently two-stage (materialise the output, then aggregate it), "
        f"which needs a subquery -- SQLTransform's single-SELECT surface has none. "
        f"Aggregate over an input column instead, or use a SQLTransform reference, "
        f"which inlines to a scalar.")
```

**Scope is critical.** The guard lives in `resolve_transformer_refs`, which only
ever sees `tfm_refs`, so the compose path is structurally untouched. A
`SQLTransform` ref inlines to a plain scalar, making an aggregate over it ordinary
flat SQL — measured working (`[12.5, 12.5]`) and pinned by a test so a later
refactor cannot silently break it.

### Iterative staging already works for SQLTransform refs

Verified, and worth recording because §3's error message points users at it:

```
fit-cascade {a}(col)          [0.4, 0.8, 1.2, 1.6]
aggregate OVER ref output     [0.4, 0.8, 1.2, 1.6]
{b}({a}(x)) + agg over it     [-0.6, -0.2, 0.2, 0.6]   agg 2.8e-17
transform == infer (fused)    True
```

Transform → transform → aggregate-over-that composes to arbitrary depth and still
fuses to one per-row expression at inference. Only the *opaque* transformer is
excluded.

### Unfit refs are fitted once globally, never per partition

An unfit ref under an outer `PARTITION BY` is fitted **once over all rows**; the
partitioning applies to the outer aggregate over its output. Measured with city a =
{10, 20} and city b = {30, 50}:

```
__STATE_R0__ = {'avg_age': [27.5]}     global mean, NOT per-city 15 / 40
outer output  = [0.545, 0.545, 1.455, 1.455]
```

This matches sklearn, where a Pipeline step is fitted once on all training data.
Per-group fitting is a different feature, served by `PARTITION BY` state directly.
A ref that partitions in its own definition is rejected ("referenced transform must
read exactly one input column"), consistent with doc-8's multi-input limit. The
semantic is undocumented and plausibly "fixable" in the wrong direction, so it gets
a lock-in test.

## §4 Tests (AC#3, AC#4)

Eleven tests. Six pass on arrival and pin existing behaviour; five require the new
code in §1–§3.

| # | Test | Asserts | Passes on arrival |
|---|---|---|---|
| 1 | mixed leaf + nested args | existing error | yes |
| 2 | columns vs `feature_names_in_` mismatch | existing error | yes |
| 3 | aggregate over opaque output | new guard (§3) | no |
| 4 | compose ref inside an aggregate still works | guard does not overreach | yes |
| 5 | unfitted ref | new error (§1) | no |
| 6 | transformer + `PARTITION BY` input col | lock-in; currently works | yes |
| 7 | unfit ref fitted once globally | lock-in (§3) | yes |
| 8 | ndarray-fit binds positionally, original untouched | new capability (§1) | no |
| 9 | arity mismatch on the positional path | new error (§1) | no |
| 10 | single ref probes `.transform()` once | AC#1, spy counts calls | no |
| 11 | 3-level nesting parity | AC#4 | yes |

Every test is mutation-checked: disable the guard or break the mechanism and
confirm the test fails. The six that pass on arrival (1, 2, 4, 6, 7, 11) receive
the same treatment TASK-2's AC#2 did — a passing test proves nothing until it has
been shown to fail for the right reason. For test 11 specifically, the mutation is
to break the §2 `consumed` logic, since that is what it exists to cover.

## §5 Documentation (AC#6, AC#5 doc half)

One new README section covering the two facts a user hits immediately: output is a
single Arrow struct column needing `.flatten()` for the sklearn handoff, and the
name-vs-position binding table with the order-safety warning from §1. Plus a
docstring on `is_transformer` recording the `n_features_in_` = "has been fitted"
contract.

The ticket's `obj.feature_names_in_ = names` workaround is deliberately not
documented: §1 removes the need for it, and as written it is broken.

## Out of scope

- SQL named arguments — DRAFT-11.
- Scalar subquery accepted by `parse_and_validate` — noted, maps to no AC.
- The DataFusion path leaking a raw `PyErr { type: <class ...` debug repr when a
  transformer raises at batch time. Accepted: decision-2 makes cross-engine error
  *type* matching an explicit non-goal, and AmirHossein confirmed exceptions need
  not match.
