# Bisect serving against the Python twin

**Done when:** The n=64 regression has an identified cause and a measured fix or explained twin-identity change.
**State:** Open. **Next:** Locate the original serving/twin benchmark and reproduce its n=64 comparison.

## Now

A fresh release wheel serves 1.20–1.80× slower than the Python twin at n=64. a6fa318 is the comparison point. The n=1 twin cell varies by 2×.

## Wanted

Bisect engine regression versus the twin’s identity change. Record n=64 before/after ratios in a warm process.

## Involved

- `packages/confit/`: serving benchmark and release extension.
- `packages/confit/src/specializer/exec/cranelift.rs`: generated execution.

## Out of scope

Using the unstable n=1 cell as the regression criterion.
