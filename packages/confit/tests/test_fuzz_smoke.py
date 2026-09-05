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

import pyarrow as pa

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
    it -- the silently-dropped-modifier class, one function over. The fuzzer
    must see that case as not-AGREE today, and as REFUSED once fixed — this
    assertion survives the fix."""
    case = gen.planted_over_case()
    v = oracle.run_case(case)
    assert v.kind in ("DIVERGE_BUILD", "REFUSED"), v


def _planted_order_case(tag: str, limit: int = 600) -> gen.Case:
    """The first seed the static-order stream hands the named twin. Scanning
    beats hard-coding a seed: the twins ride an auxiliary stream (gen
    .static_order_case), and which seeds it claims is its own business."""
    for seed in range(limit):
        case = gen.gen(seed)
        if tag in case.tags:
            return case
    raise AssertionError(f"no {tag} case in seeds 0-{limit - 1}")


def test_a_planted_static_only_tie_refuses_under_its_own_name():
    """A refusal the campaign reports has to arrive as REFUSED under a class
    of its own, or it hides in another refusal's bucket. Planted because the
    grammar cannot reach it (gen.static_order_case says why), so the campaign
    would otherwise never see the shape at all."""
    v = oracle.run_case(_planted_order_case("static_tie_order"))
    assert v.kind == "REFUSED", v
    assert v.klass == oracle.TIE_KLASS, v


def test_a_planted_unique_sort_key_agrees_and_is_not_over_refused():
    """The twin that guards the price of the rule: the same query over a key
    that separates every row still serves. Were it refused, the oracle would
    say so as a finding rather than as one more REFUSED nobody reads."""
    v = oracle.run_case(_planted_order_case("static_tie_unique"))
    assert v.kind == "AGREE", v


def test_an_over_refused_unique_key_is_a_finding_not_a_refusal():
    """The detector itself, driven by a case whose tag lies about it: a
    unique-key twin that comes back with the tie refusal must be graded
    DIVERGE_BUILD, not filed under REFUSED."""
    case = _planted_order_case("static_tie_order")
    case.tags = ["static_tie_unique"]
    v = oracle.run_case(case)
    assert v.kind == "DIVERGE_BUILD", v
    assert v.klass == "tie-over-refusal", v


def _decimal_lit_case(pack: bool) -> gen.Case:
    """A bare decimal literal: DuckDB types `1.5` as DECIMAL(2,1) and we map
    it to f64, decimal arithmetic being unshipped. `pack` puts the literal in
    a struct lane, so the nested delta is exercised too."""
    e: gen.Node = gen.Lit(1.5, "float", bare_decimal=True)
    if pack:
        e = gen.StructPack([("f", e)])
    q = gen.Q([], gen.Sel([(e, "o0")], "__THIS__"))
    return gen.Case(-2, {"k": "int"}, [{"k": 1}], {}, [], None, q, None, None, [])


def test_an_unshipped_lane_is_classified_and_never_value_compared():
    """The oracle used to cast DuckDB's DECIMAL answer down to our f64 so the
    values could still be compared. That manufactured 1-ulp artifacts and
    graded a feature nobody has shipped as agreement. An unshipped width now
    gets its own verdict, and no comparison at all."""
    for pack in (False, True):
        v = oracle.run_case(_decimal_lit_case(pack))
        assert (v.kind, v.klass) == ("UNSHIPPED", "decimals"), v
        assert "decimal" in v.detail, v


def _static_decimal_lit_case() -> gen.Case:
    """The same bare decimal literal, reached from the STATIC-ONLY leg: FROM
    a static table, so no request row is involved in the answer."""
    e: gen.Node = gen.Lit(1.5, "float", bare_decimal=True)
    q = gen.Q([], gen.Sel([(e, "o0")], "s0"))
    statics = {"s0": ({"a": "int64"}, [{"a": 1}, {"a": 2}])}
    return gen.Case(-3, {"k": "int"}, [{"k": 1}], statics, [], None, q, None, None, [])


def test_the_static_only_leg_has_no_unshipped_width_to_classify():
    """The static-only leg value-compares with no schema check in front of
    it, which would break the rule above if our answer there carried a width
    of its own. It cannot today, and the reason is worth pinning: a
    static-tables-only query never prepares (its driving relation is not the
    row table), so DuckDB evaluates it and both the rows and the schema come
    back verbatim — decimals included. The AGREE below is a real agreement,
    not a gap absorbed by a missing check.

    This goes red the day our own evaluator answers that path, which is
    exactly when the leg owes the same classification the row path does.
    """
    case = _static_decimal_lit_case()
    fn = oracle._build(
        gen.render(case.query),
        oracle._arrow_schema(case.row_schema),
        {n: oracle._arrow_table(s, r) for n, (s, r) in case.statics.items()},
        [],
        None,
        False,
    )
    assert fn.backend == "constant", fn.backend
    assert fn.output_schema.field(0).type == pa.decimal128(2, 1), fn.output_schema
    assert oracle.run_case(case).kind == "AGREE"


def test_a_real_schema_difference_is_still_a_divergence():
    """Only the named unshipped classes take the UNSHIPPED exit; every other
    type or name mismatch stays the DIVERGE_VALUE "schema" it always was."""
    duck = pa.schema([("o0", pa.int64())])
    assert oracle._schema_delta(duck, pa.schema([("o0", pa.string())]))[0] == "diff"
    assert oracle._schema_delta(duck, pa.schema([("o1", pa.int64())]))[0] == "diff"
    assert oracle._schema_delta(duck, duck) is None


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
    """int8/int16/int32 were unreachable while every generated column was
    int64, so the narrow-width families were invisible to the fuzzer no
    matter how many seeds it burned."""
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
