---
id: m-1
title: "Bulletproof opaque transforms, usable batch, acceptable inference"
---

## Description

**Goal: opaque transforms that are bulletproof and actually work.** Correctness and robustness
first — an arbitrary fitted sklearn transformer, run through the opaque path, should behave
correctly, fail loudly when misused, and produce bit-identical output on both engines.

Then a **good batch experience** (usable output shapes, a clean sklearn handoff, no manual
stitching) and **acceptable** inference performance. Acceptable, not optimal.

**Explicitly OUT of scope: optimizing transformers.** Compiling transformers down from the
opaque callout into native SQL/expression form — the per-transformer native swap, one-hot as
join-to-domain — is FOLLOW-UP work and gets its own milestone, not yet created (AmirHossein,
2026-07-23). Those tickets are parked as drafts. Do not pull them in here to "finish" this
milestone; a transform being slow is not a defect against this goal, a transform being wrong
or fragile is.

Rule of thumb for what belongs here: does it make the opaque path more correct, harder to
misuse, or nicer to consume in batch? Then yes. Does it make an already-working transform
faster? Then no — that is the next milestone.
