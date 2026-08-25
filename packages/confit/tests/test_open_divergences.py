"""Divergences we intend to CLOSE — one xfail-strict pin each, ticket named.

It has emptied and refilled inside a single day before -- five pins closed at
once on 2026-08-17 and the campaign that followed the oracle change refilled
it by evening. That is the intended rhythm, not churn: adding a pin here is
how a new divergence gets recorded, and emptying it again is what closing one
looks like.

The split from `known_divergences/` is by INTENT, not by severity:

    known_divergences/   behaviour we have decided to KEEP. Every entry
                         states the ground for keeping it, and its tests
                         PASS — they are regression pins on a settled
                         answer.

    this file            behaviour we have decided to CHANGE. Every entry is
                         xfail(strict=True) and names the task that closes
                         it. When the fix lands the pin flips loudly, and
                         the entry is deleted rather than edited.

Why the separation is worth a second file: mixing the two makes "is this on
purpose?" unanswerable at a glance, and a reader who assumes the wrong one
either implements something we chose not to have, or leaves a real bug
sitting under a paragraph explaining why it is fine. The census on
2026-08-16 found both mistakes already present.

strict=True is the load-bearing part. A pin that silently starts passing is
worse than no pin: it certifies work nobody did.

Empty as of 2026-08-25, when the last two pins closed. The struct leg FLIPPED
-- NATURAL and USING now key on a shared struct exactly like DuckDB -- and the
TIMESTAMP leg became a named REFUSAL rather than a wrong answer (the decision
of 2026-08-25 splits opaque scalar keys off into TASK-134, still open). The
live-oracle pins for both live in test_join_keys.py.
"""
