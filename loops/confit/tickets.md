# Confit ticket board

What is in flight in the confit loop, inline work included, in the shared
form of [`../workers.md`](../workers.md) §3. A worker's prompt is
[`../worker-brief.md`](../worker-brief.md), then
[`worker-brief.md`](worker-brief.md), then its ticket's section below.

Wave 1 is merged: T1 shared subexpressions within a call (`claude/cse-call-body`,
#363), T2 the early size refusal (`claude/early-size-refusal`, #358), T3
UBIGINT/HUGEINT on a 128-bit lane (`claude/i128-hugeint`, #349). T4, the move
to `loops/confit/`, merged inline as #370. Nothing is in flight.

| id | ticket | branch | depends on | overlaps | worker | PR | state |
|---|---|---|---|---|---|---|---|

Next up, once a slot frees and the account's usage warning clears:
struct-valued outputs (ruled class 3); emitting a shared
value at its first reading step instead of before the first item (PLANS);
superlinear build time of long AND/OR chains (20,000 terms: about 23 s).
