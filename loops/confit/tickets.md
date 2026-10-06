# Confit ticket board

This board lists what is in progress in the confit loop, including inline work. It uses the shared form of
[`../workers.md`](../workers.md) §3. A worker's prompt is [`../worker-brief.md`](../worker-brief.md), then
[`worker-brief.md`](worker-brief.md), then the section of the worker's ticket below.

The first wave (the set of tickets that the supervisor started together) is merged. It had three tickets:

- T1 shares subexpressions within a call (`claude/cse-call-body`, #363).
- T2 refuses a build that would be too large before it compiles: in about 2.5 s, not 6 to 67 s
  (`claude/early-size-refusal`, #358).
- T3 serves UBIGINT and HUGEINT on a 128-bit lane (`claude/i128-hugeint`, #349).

T4 moved the loop files to `loops/confit/`. It merged inline as #370. Nothing is in progress.

| id | ticket | branch | depends on | overlaps | worker | PR | state |
|---|---|---|---|---|---|---|---|

Next up, when a worker slot frees and the account's usage warning clears:

- Struct-valued outputs (ruled class 3).
- A value that a SQL function body binds once. The native catalog's SplineTransformer needs it (see PLANS).
- Linear-time Cranelift builds of deep CASE trees. IsotonicRegression with 20 000 thresholds takes 22 s now.
- Build time that grows faster than linearly for long AND/OR chains. A chain of 20 000 terms takes about
  23 s.
