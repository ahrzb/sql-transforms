# Serve list columns and nested list leaves

**Done when:** List columns and list leaves inside structs round-trip with parity and wrapped-query tests.
**State:** Open. **Next:** Measure the smallest refused or divergent case with the optimizer-off oracle.

## Now

Homogeneous list literals serve whole or through a constant index (`test_list_literals.py`). List columns, lists inside structs, opaque struct leaves and list-valued regex forms remain unserved. DuckDB refuses decimal256 static columns at Arrow registration.

## Wanted

Add list-column and nested-list representation first; retain a named boundary for opaque leaves and unsupported regex engines.

## Involved

- `packages/confit/tests/test_list_literals.py`: list output parity.
- `packages/confit/tests/test_struct_outputs.py`: nested leaves.
- `packages/confit/fuzz/exclusions.py`: wrapped list tolerance.

## Out of scope

decimal256 statics rejected upstream; regex engine replacement.
