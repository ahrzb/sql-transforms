"""The fuzzer's own gate: machinery, not zero findings.

The fuzzer exists to find live bugs, so "no findings over N seeds" cannot be
the CI invariant — it would go red the moment the fuzzer works. What CI pins
instead: generation is deterministic, the oracle produces verdicts across the
seed range (with both AGREE and REFUSED present, else the grammar or the
oracle is broken), verdicts are reproducible, the planted known-live case
diverges-or-refuses, and the shrinker preserves a verdict while shrinking.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from fuzz import gen, oracle, shrink  # noqa: E402

N = 120  # seeds per smoke run; a campaign is 100x this


def test_generation_is_deterministic():
    for seed in range(0, N, 17):
        a, b = gen.gen(seed), gen.gen(seed)
        assert gen.render(a.query) == gen.render(b.query)
        assert a.rows == b.rows and a.row_schema == b.row_schema


def test_verdicts_cover_the_contract_and_reproduce():
    kinds = {}
    for seed in range(N):
        v = oracle.run_case(gen.gen(seed))
        assert v.kind in oracle.KINDS, v
        kinds.setdefault(v.kind, seed)
    # A grammar that never agrees tests nothing; one that never refuses
    # never touches the refusal boundary.
    assert "AGREE" in kinds, kinds
    assert "REFUSED" in kinds, kinds
    for seed in list(kinds.values())[:3]:
        v1 = oracle.run_case(gen.gen(seed))
        v2 = oracle.run_case(gen.gen(seed))
        assert (v1.kind, v1.klass) == (v2.kind, v2.klass)


def test_planted_over_modifier_diverges_or_refuses():
    """`abs(k) OVER ()` is live on master: DuckDB refuses it, confit builds
    it (the TASK-69 silent-drop class, one function over). The fuzzer must
    see that case as not-AGREE today, and as REFUSED once fixed — this
    assertion survives the fix."""
    case = gen.planted_over_case()
    v = oracle.run_case(case)
    assert v.kind in ("DIVERGE_BUILD", "REFUSED"), v


def test_shrinker_preserves_the_verdict_and_shrinks():
    case = gen.planted_over_case()
    before = oracle.run_case(case)
    small = shrink.shrink(case)
    after = oracle.run_case(small)
    assert (after.kind, after.klass) == (before.kind, before.klass)
    assert len(gen.render(small.query)) <= len(gen.render(case.query))
