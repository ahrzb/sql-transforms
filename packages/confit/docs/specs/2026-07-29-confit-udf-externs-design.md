# Confit: params-join wiring + UDF externs (DRAFT-22, steps 2+3)

Full design: `backlog/drafts/draft-22 - UDF-protocol-and-Confit-externs-serving-transformers-without-special-support.md`
(+ its addendum and DRAFT-23). Scope per AmirHossein: **PythonTransform tier
only** — the opaque-callable fallback/oracle; native `confit.udfs.*` families
are the next loop (DRAFT-23).

## What Confit gains

**Step 3 — the marginalizer's join shape builds.**

- `IS NOT DISTINCT FROM` join keys against static build sides: a NULL key is
  an ordinary key value (one bucket) — encoded as a (validity, payload) pair
  in the probe map, on both the build and probe sides.
- The keyless always-true LEFT JOIN (`ON ((1 = 1))`) against a one-row
  static.
- Both under "map"/"filter" shapes: these joins are 0-or-1 by the params
  contract, and the LEFT-join rule of the exactly-one proof covers them.

**Step 2 — declared UDF externs.**

- New program header `extern @N: name(tys) -> (tys)` + `ecall` instruction.
  Operand layout: one (validity, payload) pair per declared param in; a
  whole-call validity (the NULL list, distinct from a list of NULLs) plus a
  (validity, payload) pair per declared return out.
- Both backends route through one shared `call_extern` that enforces the
  declared return shape — wrong length/type or a raised exception is a
  named trap. The cranelift path is an `extern "C"` helper over the same
  function (the no-drift rule).
- `DuckDBInferFn(..., udfs=[...])`: duck-typed protocol objects (`name`,
  `takes`, `returns`, scalar `__call__`; an `instances` attribute marks the
  implicit leading nullable-i64 instance id). The trampoline attaches to
  Python per call, converts NULL-masked scalars both ways under the
  declared types.
- Width-1 calls are ordinary scalar expressions (mid-expression works).
  Width-k calls are bare SELECT items emitting ONE `list | None` field —
  assembled at every boundary (row marshaller, generic path, `infer_arrow`
  as `pa.list_`), NULL list when the id/whole-call is NULL. Mid-expression
  width-k refuses by name.
- The contract is parameterized, not weakened: bit-exact vs DuckDB *with
  the same udfs registered* (`create_function`), or refuse. Unknown
  functions without a matching declaration refuse exactly as before; the
  static-only constant fallback is off when udfs are declared.

## Gate

- Rust: 189 tests including the DRAFT-22 serving-shape end-to-end, IR
  round-trip/verify for `extern`/`ecall`, and the cranelift-vs-interpreter
  random-IR differential.
- Python: `tests/test_params_joins.py` (INDF/keyless differentials vs
  DuckDB) and `tests/test_udfs.py` (marginalizer shape, wide list fields,
  arrow boundary, both backends agree, trap/refusal surface) — all
  differential against DuckDB with the same objects registered.

## Deliberately out (next loops)

- Native typed entries (`confit.udfs.PCA/StandardScaler/TreeEnsemble`) —
  DRAFT-23, behind the same extern slots.
- `SQLProjection.infer/infer_batch` wiring through this surface (the
  integration loop; needs this PR + sql-transform's step 1, both merged).
- Vectorized `apply_batch` binding for `infer_arrow` (today: per-row calls
  through the scalar protocol — correct, boundary-amortization later).
