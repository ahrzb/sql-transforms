# Serve a row-local set-operation slice

**Done when:** One measured row-local set operation serves with exact multiplicity, NULL and type semantics.
**State:** Open. **Next:** Measure the smallest refused or divergent case with the optimizer-off oracle.

## Now

Set operations remain refused. Only batch-dependent forms are outside the target model.

## Wanted

Measure a static or row-local UNION slice first. Keep INTERSECT/EXCEPT and other set forms named until independently measured.

## Involved

- `packages/confit/src/specializer/frontend/`: set-query binding.
- `packages/confit/tests/test_known_limitations.py`: set-operation refusals.

## Out of scope

Set operations over the request batch.
