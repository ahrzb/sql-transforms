---
id: DRAFT-7
title: 'Native-swap: native per-transformer (Tier 0)'
status: Draft
assignee: []
created_date: '2026-07-18 13:44'
updated_date: '2026-07-23 14:31'
labels:
  - sklearn
  - native-swap
dependencies: []
documentation:
  - doc-2 (sklearn transformer implementation plan)
  - doc-7 (Transformer execution model)
priority: high
ordinal: 5000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
WHAT A USER HITS
You build a normal preprocessing pipeline:

    scaler = StandardScaler().fit(train[["age", "income"]])
    t = SQLTransform("SELECT {scaler}(age, income) AS scaled FROM __THIS__").fit(train)
    t.infer({"age": 41, "income": 82000})   # single-row serving

Today this works, but every single call crosses into Python and runs sklearn's own .transform() â€” the opaque path. So the per-row serving latency is dominated by the Python/FFI boundary, which is exactly the cost the project exists to remove (see the boundary-bound finding). At n=1 you pay a full sklearn callout to compute (x - mean) / scale.

The user-visible symptom is not wrongness, it's speed: serving is far slower than it should be for arithmetic this trivial, and the transformer can't be served from the non-Python runtimes at all (Go/Java/WASM) because it needs a Python interpreter in the loop.

WHAT THIS TICKET DOES
Swap each Tier 0 transformer from the opaque sklearn callout to a NATIVE implementation expressed in the engine's own SQL/expression surface, so it fuses into the single per-row expression instead of calling out.

The insight (doc-7): these transformers are already expressible as window-aggregate SQL we compile today.
    StandardScaler  ==  (x - AVG(x) OVER ()) / STDDEV(x) OVER ()
    SimpleImputer   ==  COALESCE(x, <frozen fit-state>)
    OrdinalEncoder  ==  a code-map lookup
    OneHotEncoder   ==  one CAST(x = 'cat_i' AS INT) per learned category
So most of this is macro definitions over shipped machinery, not new engine code.

Each swap is diffed against the sklearn fallback as the oracle â€” the point is that switching a user from opaque to native must not change a single output value.

WHY IT IS A DRAFT
Needs design work before it is a dispatchable task (AmirHossein, 2026-07-23). Open questions: the state-shape contract per transformer (scalar vs list vs code-map vs per-group table), where the macro definitions live, how unknown-category handling is expressed, and whether OneHotEncoder's multi-output fan-out belongs here or in the feature-output work. Re-scope and re-promote to a TASK before dispatching.

Full plan context: doc-2 (sklearn transformer implementation plan), doc-7 (transformer execution model).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 StandardScaler native, differential parity vs sklearn fallback
- [ ] #2 SimpleImputer native + parity
- [ ] #3 OrdinalEncoder native + parity (unknown-category handling)
- [ ] #4 OneHotEncoder native + parity (multi-output)
<!-- AC:END -->

## Comments

<!-- COMMENTS:BEGIN -->
author: Iris (PM)
created: 2026-07-23 00:36
---
Dispatched to Wren (2026-07-23), next on the m-1 spine after TASK-13. Tier order per doc-2; sklearn fallback is the parity oracle per AC wording, DataFusion for SQL semantics (decision-1). TASK-6 (ColumnTransformer assembly) unblocks once this + TASK-13 are done Ã¢â‚¬â€ TASK-13 already landed.
---

author: Iris (PM)
created: 2026-07-23 00:37
---
Dispatch retracted Ã¢â‚¬â€ AmirHossein stopped it. New convention: PM proposes assignments, AmirHossein green-lights each dispatch explicitly before any dev is messaged.
---

author: Iris (PM)
created: 2026-07-23 01:01
---
Moved to Draft (2026-07-23): native per-transformer swap needs design work â€” not a well-scoped task yet. Re-scope + re-promote to a TASK before dispatch. AmirHossein's call.
---
<!-- COMMENTS:END -->

