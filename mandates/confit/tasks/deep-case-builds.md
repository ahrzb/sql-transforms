# Make deep CASE builds scale linearly

**Done when:** Retained threshold and leaf grids show linear build scaling with unchanged answers and traps.
**State:** Open. **Next:** Retain a single-tree reproduction and measure scaling across threshold counts.

## Now

One IsotonicRegression CASE tree with 20,000 thresholds takes about 22 s. The sent native reproduction is not in the repository: `/mnt/project-files/transforms-loop/` (finding 49). Isotonic and quantile keep estimated build caps.

## Wanted

Reduce deep CASE compilation cost and report the grid needed to refit both `_build_estimate` formulas.

## Involved

- `packages/confit/src/specializer/lower.rs`: `select_case`.
- `packages/confit/src/specializer/exec/cranelift.rs`: CASE compilation.

## Out of scope

Speculative evaluation of trapping branches.
