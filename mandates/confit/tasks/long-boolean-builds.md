# Make long AND OR builds scale linearly

**Done when:** Retained AND/OR grids show linear build scaling and preserve short-circuit trap semantics.
**State:** Open. **Next:** Measure the smallest refused or divergent case with the optimizer-off oracle.

## Now

The old next-work list measured about 23 s for a 20,000-term chain. Two 6,000-term test_expression_depth.py chains overflow an 8 MB stack in an unoptimized debug build; the full suite used ulimit -s 262144.

## Wanted

Remove superlinear compilation and deep traversal failure, without reassociating observable evaluation.

## Involved

- `packages/confit/tests/test_expression_depth.py`: deep AND/OR chains.
- `packages/confit/src/specializer/lower.rs`: boolean lowering.
- `packages/confit/src/specializer/exec/cranelift.rs`: chain compilation.

## Out of scope

Reordering trapping terms or native compact guard rewrites already delivered.
