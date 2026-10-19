# Compile bounded scans with Cranelift

**Done when:** ProbeRange and ProbeRead execute bounded scans with parity between interpreter and JIT.
**State:** Open. **Next:** Measure the smallest refused or divergent case with the optimizer-off oracle.

## Now

Any program with a scan uses interpreter-only MultiMap statics. ProbeRange and ProbeRead lack Cranelift code.

## Wanted

Implement both probe opcodes without changing row order, validity or forced trapping folds.

## Involved

- `packages/confit/src/specializer/exec/cranelift.rs`: `ProbeRange`, `ProbeRead`.
- `packages/confit/src/specializer/lower.rs`: `emit_fold`, `many_fold`.
- `packages/confit/src/specializer/exec/tests.rs`: backend differential tests.

## Out of scope

Increasing the 100-row scan bound.
