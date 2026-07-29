---
id: DRAFT-23
title: "Native UDF families: typed confit.udfs.* entries with extracted state"
status: Draft
type: spike
created: 2026-07-29
---

## Where this came from

Design dialogue with AmirHossein, 2026-07-29, right after DRAFT-22 step 1
landed (PR #62) and while the Confit extern substrate was being built. The
shape is his: for known estimator families the *udf entry itself* goes
native — the SQL never changes. This is also the design half of the parked
"optimized sklearn transforms" milestone (milestone creation still gated on
his go).

## The design

The `udfs=` list is a **typed registry**. Same SQL text under every entry:

```python
DuckDBInferFn(
    'SELECT pca(__cf_p0.__cf_est, __cf_t.age, __cf_t.fare) AS e, ...'
    ' FROM __THIS__ AS __cf_t LEFT JOIN __CF_PARAMS_0__ AS __cf_p0 ON ((...))',
    static_tables={"__CF_PARAMS_0__": "<country, __cf_est>"},
    udfs=[confit.udfs.PCA(
        "pca", n_components=2,
        instances={0: {"mean": [37.2, 8.1], "components": [[...], [...]]},
                   1: {...}},          # plain arrays, extracted at fit — no pickle
    )],
)

# trees: fit step is an importer; keyless => literal id, no join at all
DuckDBInferFn(
    'SELECT score(0, __cf_t.age, __cf_t.fare) AS s, ...',
    udfs=[confit.udfs.TreeEnsemble("score", instances={0: "<trees as data>"})],
)
```

| entry | implementation | state | parity tier |
|---|---|---|---|
| `PythonTransform` | GIL trampoline -> `est.transform` | pickle | the oracle, always available |
| `confit.udfs.StandardScaler` | native affine kernel | arrays | bit-exact target |
| `confit.udfs.PCA` | native matvec kernel | arrays | toleranced (BLAS order unpinnable) |
| `confit.udfs.TreeEnsemble` | native tree walk | tree tables | bit-exact reachable (compares + fixed-order sums) |

Rules:

- **Hyperparams live on the entry** (`n_components=2`); fitted state per
  instance id. The entry is self-describing: `takes`/`returns` are
  *derived* by the class (PCA with n_components=2 knows returns is 2 wide),
  not hand-declared.
- **One SQL text across tiers.** Fallback -> optimized is swapping the list
  entry. The gate is "same query, same statics, swap the entry, compare"
  — no query rewrite to account for, so a disagreement is always the
  kernel's fault. Tolerance declared per family (table above), never
  discovered per bug.
- **Family recognition in `SQLProjection._prepare`, conservative:** if the
  fitted estimator (or every step of a fitted Pipeline) maps to a native
  family, emit the typed entry with extracted state; anything unrecognized
  falls through to `PythonTransform`. Recognition is per-estimator-class;
  the fallback existing is what lets recognition stay conservative.
- **No pickle for known families** — instances are arrays/tables,
  JSON/Arrow-serializable. This is what unlocks non-Python hosts (the
  wazero/chicory WASM path): a Go/Java service can serve per-country PCA
  because the entry is data plus a kernel that ships inside the engine.
  `PythonTransform` can never cross that boundary; native entries can.
- **Engine side:** each family is a new `ExternImpl` variant behind the
  same extern slot the DRAFT-22 substrate provides (Cx slot table:
  PyTrampoline | NativeKernel). Frontend/type-checking unchanged.
- **DuckDB batch side:** each `confit.udfs.*` class also registers a plain
  Python implementation of the same math (`create_function`), so the batch
  path and the differential gate run without the Rust engine in the loop.

## Sequencing

1. DRAFT-22 substrate lands (extern slots + PythonTransform trampoline +
   params-join wiring — in flight).
2. `confit.udfs.StandardScaler` first (affine, bit-exact target — proves
   the swap-the-entry gate), then `PCA` (tolerance story), then
   `TreeEnsemble` (importer fit step, `reads = ()`).
3. Interpretability follow-up, later: named per-component output columns.

## Rejected along the way

- **Desugar-into-SQL-expressions** for known families (DRAFT-22's original
  "column-width decision"): a query rewrite between tiers means the gate
  compares two different texts — weaker isolation, and wide params columns
  can't carry matrices/trees anyway. Superseded by typed entries.
- **Pickled sklearn as the serving artifact for known families**: keeps
  Python in the serve path and locks out non-Python hosts; it remains the
  catch-all tier only.
