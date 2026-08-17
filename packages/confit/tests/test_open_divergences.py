"""Divergences we intend to CLOSE — one xfail-strict pin each, ticket named.

**This file is currently EMPTY, and that is the finding, not an oversight.**
Every pin it held (TASK-115, 116, 117, 118, 119) was closed on 2026-08-17;
each fix moved its pin to the suite that owns the subject, where it now
PASSES against the live oracle. Adding one here is how a new divergence gets
recorded, and emptying it again is what closing one looks like.

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
"""

from __future__ import annotations
