# Add remaining precedence hooks

**Done when:** Prefix ~ and NOT GLOB match DuckDB parse trees, values and output names.
**State:** Open. **Next:** Measure the smallest refused or divergent case with the optimizer-off oracle.

## Now

Binary precedence is pinned. DuckDB binds ~ looser than + but tighter than <<, so ~1+1 means ~(1+1). NOT GLOB needs another hook. DuckDB does not parse # either.

## Wanted

Extend the parser’s measured hooks without weakening parse-divergence guards.

## Involved

- `packages/confit/src/specializer/precedence.rs`: prefix and negated operators.
- `packages/confit/docs/specs/pins-wave5/operator-precedence.json`: parse pins.

## Out of scope

Adding # or replacing the regex engine.
