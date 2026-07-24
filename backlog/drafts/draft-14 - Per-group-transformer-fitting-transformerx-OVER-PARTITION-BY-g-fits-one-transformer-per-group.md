---
id: DRAFT-14
title: >-
  Per-group transformer fitting: transformer(x) OVER (PARTITION BY g) fits one
  transformer per group
status: Draft
assignee: []
created_date: '2026-07-23 15:38'
labels:
  - authoring-surface
  - transformer-refs
  - sklearn
  - feature
dependencies: []
documentation:
  - 'doc-7 (Transformer execution model — UDF/UDAF, macros, composition)'
  - doc-2 (sklearn transformer implementation plan)
  - 'doc-8 (Composition — {transform}(col) references)'
priority: medium
type: feature
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
ORIGIN: AmirHossein's proposal (2026-07-23), routed via Wren out of TASK-3 brainstorming. Measurements below are Wren's, taken on the current engine — not inferred.

WHAT A USER HITS
You are modeling something that spans groups with genuinely different scales — houses across countries, sales across regions, sensor readings across devices:

    SELECT {scaler}(price) AS price_z FROM __THIS__

This fits ONE scaler over all rows pooled. So every house in a cheap country reads as "below average" against the global mean, and the within-country signal — the thing you actually wanted — is flattened away. The user wants one scaler PER COUNTRY.

Measured on today's engine: with country a = {10, 20} and b = {30, 50}, an unfit ref's state holds avg_age = [27.5] — the global mean — not the per-country 15 and 40.

IMPORTANT: that is the CORRECT current semantic, not a bug. It matches sklearn, where a Pipeline step is fitted once on all training data. So this ticket is a genuinely new feature.

Today the only workaround is to split the frame by group yourself, fit N scalers, and stitch the results — which happens outside the library and throws away the single-artifact serving story that is the whole point.

PROPOSED SURFACE

    standardscaler(x) OVER (country)          -- AmirHossein's shorthand
    {scaler}(x) OVER (PARTITION BY country)   -- standard SQL spelling

THE SPLIT — two very different sizes of work, and the ticket should stay honest about that:

(a) SQL-EXPRESSIBLE transformers — StandardScaler, MinMaxScaler, MaxAbsScaler, mean-SimpleImputer.
    ALREADY WORKS END-TO-END TODAY. This is pure syntax sugar; no engine work. Measured:

        SELECT (x - AVG(x) OVER (PARTITION BY country)) /
               STDDEV(x) OVER (PARTITION BY country) AS z FROM __THIS__

        state __STATE_BY_country__ = {'country': ['b','a'],
                                      'avg_x':    [40.0, 15.0],
                                      'stddev_x': [14.142, 7.071]}   <- ONE ROW PER GROUP
        batch == infer == [-0.7071, 0.7071, -0.7071, 0.7071]
        unseen country at infer -> None (NULL), consistent with the existing unseen-partition semantic

    So for this class the work is a MACRO / DESUGAR layer — exactly doc-7's "transformers are macros over the
    window-agg/scalar SQL surface." Per-group state is the already-shipped PARTITION BY mechanism, and the
    cold-start story is inherited for free.

(b) OPAQUE sklearn objects — an arbitrary fitted transformer via {sc}(...).
    GENUINELY NEW WORK. Needs N fitted clones keyed by group, per-group fitted state threaded through the
    artifact and lookup path, and an explicit unseen-group policy at inference.

OPEN DESIGN QUESTIONS (why this is a draft, not a task)
1. UNSEEN-GROUP POLICY for (b): NULL, fall back to a globally-fitted transformer, or hard error? doc-2 calls
   unknown-category handling "a designed-in requirement, not a flag," so this wants a ruled decision rather
   than a default. Note (a) already answers it implicitly as NULL via the existing unseen-partition semantic —
   worth deciding whether (b) must match.
2. SYNTAX: accept the OVER (country) shorthand, or require the standard OVER (PARTITION BY country)?
   Wren leans standard, on the grounds that (a) then desugars literally into already-valid SQL that the engine
   machinery already speaks. A shorthand means new parsing for no new capability.
3. SCOPE: is (a) alone worth shipping first? It is nearly free and delivers the common case (scalers/imputers),
   while (b) is the real engine lift. Sequencing them as separate tickets is probably right.
4. Artifact size for (b): N fitted clones means the serialized artifact grows with group cardinality. Is there
   a ceiling, and what happens at high cardinality?

NOT TO BE CONFUSED WITH DRAFT-11 (SQL named arguments). That one is about how ARGUMENTS BIND to a transformer's
inputs. This one is about WHAT DATA a transformer is fitted on. Flagged explicitly so they do not get merged.

Context: doc-7 (transformer execution model), doc-2 (sklearn transformer plan), doc-8 (composition).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 DECISION: unseen-group policy at inference ruled explicitly (NULL vs global-fallback vs error), and whether the opaque path must match the SQL-expressible path's existing NULL semantic
- [ ] #2 DECISION: shorthand OVER (g) accepted, or standard OVER (PARTITION BY g) required
- [ ] #3 SQL-expressible transformers (StandardScaler/MinMaxScaler/MaxAbsScaler/mean-SimpleImputer) desugar to the existing PARTITION BY window-agg form, with transform == infer parity and one state row per group
- [ ] #4 Opaque sklearn refs fit one clone per group, with per-group state carried in the artifact and resolved at inference per the ruled unseen-group policy
- [ ] #5 Parity bar: per-group results match fitting the equivalent sklearn transformer separately on each group's rows
<!-- AC:END -->
