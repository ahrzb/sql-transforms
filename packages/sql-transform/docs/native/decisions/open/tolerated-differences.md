# Where a native entry may differ from its twin

**Question.** Where exact parity needs sklearn's own behaviour, what may
the entry do meanwhile?

1. **The twin raises on input sklearn's validation rejects** (most
   estimators refuse an infinite feature). Provisional: the entry may answer
   there; never the reverse.
   *Ground:* the serving query already trapped for the twin, so no served
   answer changes; making the entry trap needs a per-estimator copy of
   sklearn's validation, raised with `error()`.

Two further edges were tolerated until confit could serve them, and are
exact now: an unknown instance id raises (confit's `error()`), and a NULL
id with a struct return is a NULL struct (`SqlFunction(null_when=...)`).

**What would close it.** An owner ruling that this is acceptable as stated,
or that it must block the entry instead.
