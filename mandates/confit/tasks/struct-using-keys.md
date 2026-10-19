# Serve merged struct USING keys

**Done when:** The static side of a merged struct USING key serves under supported map joins.
**State:** Open. **Next:** Measure the smallest refused or divergent case with the optimizer-off oracle.

## Now

The static side of a merged struct USING key still refuses. Different field-name sets can produce a constant-empty DuckDB join, while confit refuses.

## Wanted

Measure map-join key merging and field-name mismatch behaviour before widening.

## Involved

- `packages/confit/src/specializer/frontend/from.rs`: USING key binding.
- `packages/confit/tests/test_struct_outputs.py`: merged outputs.

## Out of scope

Any expansion of filter or many shapes.
