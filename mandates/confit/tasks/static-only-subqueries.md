# Find a static-only subquery witness

**Done when:** PR 4 has a generated static-only construction-time witness or a maintainer answer removing the held work.
**State:** Open; implementation waits for the maintainer’s answer. **Next:** Find a real candidate and present its optimizer-off behaviour to the maintainer.

## Now

Subquery design PR 4 waits on the maintainer. Its 48 measured candidates were unread CTEs, which now serve; the class has no generated case yet. See `packages/confit/docs/specs/2026-09-26-row-local-subqueries-design.md`, Measured recovery.

## Wanted

Establish a real still-refused static-only case before implementing construction-time computation.

## Involved

- `packages/confit/docs/specs/2026-09-26-row-local-subqueries-design.md`: PR 4 and Measured recovery.
- `packages/confit/fuzz/gen.py`: subquery generation.

## Out of scope

Implementing an unmeasured class or new filter/many support.
