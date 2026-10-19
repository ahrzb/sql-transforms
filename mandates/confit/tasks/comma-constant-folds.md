# Fold constant string terms before comma admission

**Done when:** Nontrapping constant string calls before comma keys serve with exact traps.
**State:** Open. **Next:** Measure the smallest refused or divergent case with the optimizer-off oracle.

## Now

check_comma_order rejects trapping WHERE terms before keys. Of 2,000 selected seeds in 9600008..9684714 (22 mod 41, not 29 mod 31), 297 formerly agreed cases refused: 206 BIGINT-max additions, 25 other arithmetic/UDF/math, 66 LIKE/string calls. Measured can_trap improvements already served 25 more comma cases on 30,050,000..30,099,999.

## Wanted

Measure and fold constant calls such as lpad('a',3,'x') before admission. Preserve actual trapping constants. Existing nontrapping string classification stays.

## Involved

- `packages/confit/src/specializer/frontend/from.rs`: `check_comma_order`.
- `packages/confit/src/specializer/plan.rs`: `can_trap`.
- `packages/confit/tests/test_string_ops_cannot_trap.py`: measured string classification.

## Out of scope

The already completed LIKE/GLOB/trim/search/replace classification; new many support.
