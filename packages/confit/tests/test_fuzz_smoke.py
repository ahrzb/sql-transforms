"""The fuzzer's own gate: machinery, not zero findings.

The fuzzer exists to find live bugs, so "no findings over N seeds" cannot be
the CI invariant — it would go red the moment the fuzzer works. What CI pins
instead: generation is deterministic, the oracle produces verdicts across the
seed range (with both AGREE and REFUSED present, else the grammar or the
oracle is broken), verdicts are reproducible, the planted known-live case
diverges-or-refuses, and the shrinker preserves a verdict while shrinking.

And — added after a struct column in a static table turned out to be
unreachable by any seed — that the generator's table-column vocabulary is
not narrower than the boundary's. A campaign can only find bugs in the
inputs it can express, so a generator narrower than the API is a silent
coverage hole that no campaign size closes. Those are the parity tests
below; they are the reason this file exists as much as the machinery ones.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from fuzz import gen, oracle, shrink  # noqa: E402

N = 120  # seeds per smoke run; a campaign is 100x this

# The row-table boundary's accepted scalar vocabulary, verbatim from
# schema.rs::arrow_field_to_row_field. Anything here that the generator
# cannot put in a table column is a blind spot.
BOUNDARY_SCALARS = {
    "bool",
    "int8",
    "int16",
    "int32",
    "int64",
    "double",
    "string",
}

PARITY_SEEDS = N * 8  # parity is about reachability, so it needs seeds


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


def _walk_schema(schema: dict) -> tuple[set[str], bool, bool]:
    """(scalar storage types, saw a struct, saw an opaque) in one schema."""
    scalars: set[str] = set()
    struct = opaque = False
    for spec in schema.values():
        if isinstance(spec, gen.Struct):
            struct = True
            s, _, o = _walk_schema(dict(spec.fields))
            scalars |= s
            opaque |= o
        else:
            name = spec.rstrip("?")
            if name in BOUNDARY_SCALARS:
                scalars.add(name)
            else:
                opaque = True
    return scalars, struct, opaque


def _vocabulary(seeds: int):
    scalars: set[str] = set()
    row_struct = static_struct = opaque = False
    saw_static = False
    for seed in range(seeds):
        c = gen.gen(seed)
        s, st, o = _walk_schema(c.row_schema)
        scalars |= s
        row_struct |= st
        opaque |= o
        for sch, _ in c.statics.values():
            saw_static = True
            s, st, o = _walk_schema(sch)
            scalars |= s
            static_struct |= st
            opaque |= o
    assert saw_static, "no static tables generated at all — parity is unmeasurable"
    return scalars, row_struct, static_struct, opaque


def test_generator_reaches_every_boundary_scalar_type():
    """int8/int16/int32 were unreachable: every generated column was int64,
    so the narrow-width families (TASK-79, TASK-84, TASK-96) were invisible
    to the fuzzer no matter how many seeds it burned."""
    scalars, _, _, _ = _vocabulary(PARITY_SEEDS)
    missing = BOUNDARY_SCALARS - scalars
    assert not missing, f"generator cannot put {sorted(missing)} in a table column"


def test_generator_reaches_struct_columns_in_row_and_static_tables():
    """The hole that hid it: a struct column is lanes in a row table and is
    dropped from the catalogue in a static one, and no seed could express
    either."""
    _, row_struct, static_struct, _ = _vocabulary(PARITY_SEEDS)
    assert row_struct, "no struct column ever generated in a row table"
    assert static_struct, "no struct column ever generated in a static table"


def test_generator_reaches_an_out_of_vocabulary_column():
    """An unreferenced foreign column must not block a build. That rule had
    hand-written coverage and no generated coverage."""
    _, _, _, opaque = _vocabulary(PARITY_SEEDS)
    assert opaque, "no out-of-vocabulary column ever generated"


def test_shrinker_preserves_the_verdict_and_shrinks():
    case = gen.planted_over_case()
    before = oracle.run_case(case)
    small = shrink.shrink(case)
    after = oracle.run_case(small)
    assert (after.kind, after.klass) == (before.kind, before.klass)
    assert len(gen.render(small.query)) <= len(gen.render(case.query))
