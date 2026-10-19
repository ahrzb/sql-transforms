# Serve remaining named and unnamed struct construction

**Done when:** row(...) and unnamed struct_pack arguments have parity tests and exact output names.
**State:** Open. **Next:** Measure the smallest refused or divergent case with the optimizer-off oracle.

## Now

Whole struct outputs serve, but `row(...)` and unnamed struct_pack arguments refuse. Opaque/list leaves and static merged struct USING keys remain separate work.

## Wanted

Implement DuckDB’s unnamed field naming and nested output shape.

## Involved

- `packages/confit/tests/test_struct_outputs.py`: constructor output names.
- `packages/confit/src/specializer/frontend/functions.rs`: struct constructors.

## Out of scope

Opaque/list leaves and widening many joins.
