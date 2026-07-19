---
id: doc-10
title: 'Feature-output model — records, dense, sparse (design)'
type: other
created_date: '2026-07-19 01:08'
---
The output side of the serving use case (from Fermi/Investigator, 2026-07-18; folded into the transformer-foundation phase, tasks TASK-13…17). Today the infer path returns pydantic records; sklearn interop needs numeric matrices, and text/categorical needs **sparse** output. Three output contracts: (1) pydantic records (current), (2) dense float64 `(n,k)`, (3) sparse CSR.

## Connective design decisions (the tasks carry the rest)
- **Sparse feature = a per-row COO struct column** `struct<indices: list<int32>, values: list<float64>>`. It's 1:1 with rows, so sparseness lives *inside the cell*, not in row cardinality — that's what lets **one SELECT = a mixed dense+sparse feature set**. Materializes to scipy CSR for free (concat arrays → data/indices, per-row lengths → indptr).
- **Dimension N + unseen-key handling come from the FITTED transform**, not the cell or a type-level policy. N pins `shape=(n,N)` so batch width can't drift — that drift is a **silent model-misalignment bug** (a batch missing the last vocab term materializes narrower). Hence the **width-invariant assert** (the sparse-column task, TASK-14).
- **One SELECT → dense⊕sparse via type-directed decompose+assemble** (the assembler task, TASK-16). sklearn `ColumnTransformer` is the **internal** assembly target (hstack + densify), *not* a user API — users write SQL, we compile the ColumnTransformer.
- Fitted domains (vocab/idf, categories) ride the existing static_tables/lookup mechanism — no new artifact store.
- **The tfidf / array multi-hot task (TASK-17) is the opaque one** — needs explode, so it maps onto the shipped opaque-transform mechanism (decision-3). The sparse-column / scalar-one-hot tasks are fixed-fanout and composable. This is the same fixed-fanout-composes / variable-expansion-is-opaque boundary the multi-language runtimes ([[doc-4]]) are built on.

## Usability signal (2026-07-18, House Prices)
The column numeric/categorical roles still live in Python for the sklearn handoff, so with the literal-SQL form the column list is **duplicated (SQL + Python)** — a real papercut on wide (80-col) datasets. Motivates the assembler task (TASK-16) owning column routing (compile the `ColumnTransformer` from the SQL, so roles aren't re-declared in Python). Not a separate task — an assembler design constraint.
