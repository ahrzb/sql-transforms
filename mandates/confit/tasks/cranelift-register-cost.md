# Reduce compilation near the register cap

**Done when:** A retained feature-width grid identifies and reduces the dominant Cranelift cost without invalid compiled programs.
**State:** Open. **Next:** Measure the smallest refused or divergent case with the optimizer-off oracle.

## Now

Normalizer l1 at 30 features builds in 22–24 s. At 24 features allocation costs 4.3 s, verification 0.8 s (1.7 s at 32), egraph 0.8 s (2026-10-05). A 100×100 checked-integer matvec took 443 s, not investigated.

## Wanted

Measure the checked matvec and Normalizer grid. Compare release verifier cost and grouping output columns into functions before choosing a change.

## Involved

- `packages/confit/src/specializer/exec/cranelift.rs`: compilation, verification and function grouping.

## Out of scope

Disabling correctness checks without equivalent retained evidence; changing numeric reduction order.
