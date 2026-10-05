# Where a native entry may differ from its twin

**Question.** Three edges where exact parity needs either sklearn's own
behaviour or a confit capability that does not exist yet. What may the
entry do meanwhile?

1. **The twin raises on input sklearn's validation rejects** (most
   estimators refuse an infinite feature). Provisional: the entry may answer
   there; never the reverse.
   *Ground:* the serving query already trapped for the twin, so no served
   answer changes; making the entry trap needs `error()` and a per-estimator
   copy of sklearn's validation.
2. **An unknown instance id.** The twin raises ("params table and instances
   are from different fits"); the entry answers NULL. Provisional: tolerated
   until confit serves `error()` (PLANS, "Needs from confit").
   *Ground:* an unknown id is a broken artifact (params and instances from
   different fits), not a serving input.
3. **A NULL id with a struct return.** The twin answers a NULL struct, the
   entry a struct of NULL fields. Provisional: tolerated until confit serves
   a field read over a CASE-valued struct.
   *Ground:* every field read agrees, and serving SQL reads fields.

**What would close it.** An owner ruling that these three are acceptable as
stated, or that any of them must block the entry instead.
