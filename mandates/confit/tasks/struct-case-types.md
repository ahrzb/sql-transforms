# Unify struct CASE types

**Done when:** CASE arms with different struct types match DuckDB’s resulting schema and values.
**State:** Open. **Next:** Measure the smallest refused or divergent case with the optimizer-off oracle.

## Now

Same-type struct CASE serves. DuckDB unifies differing arm types; confit refuses. Struct row columns, nested fields, UDF/SQL calls, struct_pack and literals already serve (generator 2 mod 17, tag `struct-value`).

## Wanted

Implement measured field/type unification without losing nested validity.

## Involved

- `packages/confit/tests/test_struct_outputs.py`: CASE schemas.
- `packages/confit/src/specializer/frontend/typing.rs`: struct type unification.

## Out of scope

Unrelated query classes and changes to the oracle contract.
