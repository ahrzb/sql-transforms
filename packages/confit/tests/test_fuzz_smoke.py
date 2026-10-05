"""The fuzzer's own gate: machinery, not zero findings.

The fuzzer exists to find live bugs, so "no findings over N seeds" cannot be
the CI invariant — it would go red the moment the fuzzer works. What CI pins
instead: generation is deterministic, the oracle produces verdicts across the
seed range (with both AGREE and REFUSED present, else the grammar or the
oracle is broken), verdicts are reproducible, the planted known-live case
diverges-or-refuses, and the shrinker preserves a verdict while shrinking.

And that the generator's table-column vocabulary is not narrower than the
boundary's. A campaign can only find bugs in the
inputs it can express, so a generator narrower than the API is a silent
coverage hole that no campaign size closes. Those are the parity tests
below; they are the reason this file exists as much as the machinery ones.
"""

from __future__ import annotations

import functools
import sys
from pathlib import Path

import pyarrow as pa
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

from fuzz import gen, oracle, runner, shrink  # noqa: E402

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
    """`abs(k) OVER ()`: DuckDB refuses it -- the silently-dropped-modifier
    class. The fuzzer must see that case as not-AGREE: a build divergence if
    the engine serves it, REFUSED if it refuses."""
    case = gen.planted_over_case()
    v = oracle.run_case(case)
    assert v.kind in ("DIVERGE_BUILD", "REFUSED"), v


def _decimal_lit_case(pack: bool) -> gen.Case:
    """A bare decimal literal: DuckDB types `1.5` as DECIMAL(2,1), and so do
    we. `pack` puts the literal in a struct lane."""
    e: gen.Node = gen.Lit(1.5, "float", bare_decimal=True)
    if pack:
        e = gen.StructPack([("f", e)])
    q = gen.Q([], gen.Sel([(e, "o0")], "__THIS__"))
    return gen.Case(-2, {"k": "int"}, [{"k": 1}], {}, [], None, q, None, None, [])


def test_a_decimal_literal_agrees_in_its_own_width():
    """Decimals shipped, so the UNSHIPPED exit is gone for them: a bare
    decimal literal agrees as DECIMAL(2,1), top level and in a struct lane,
    and a decimal-vs-double delta would be a schema divergence."""
    for pack in (False, True):
        v = oracle.run_case(_decimal_lit_case(pack))
        assert v.kind == "AGREE", v
    duck = pa.schema([("o0", pa.decimal128(2, 1))])
    delta = oracle._schema_delta(duck, pa.schema([("o0", pa.float64())]))
    assert delta is not None and delta[0] == "diff", delta


def _static_decimal_lit_case() -> gen.Case:
    """A bare decimal literal FROM a static table: no request row is
    involved in the answer."""
    e: gen.Node = gen.Lit(1.5, "float", bare_decimal=True)
    q = gen.Q([], gen.Sel([(e, "o0")], "s0"))
    statics = {"s0": ({"a": "int64"}, [{"a": 1}, {"a": 2}])}
    return gen.Case(-3, {"k": "int"}, [{"k": 1}], statics, [], None, q, None, None, [])


def test_a_static_tables_only_case_refuses_at_build():
    """A query that reads no request table is outside the model
    (docs/decisions/closed/static-only-queries.md): it refuses at build,
    naming the driving relation, and the refusal keeps DuckDB's outcome."""
    v = oracle.run_case(_static_decimal_lit_case())
    assert v.kind == "REFUSED", v
    assert "driving relation" in v.detail, v.detail
    assert v.oracle == "serves", v


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
    """A struct column is lanes in a row table and is dropped from the
    catalogue in a static one; the generator must express both."""
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


def _row_path_agree_seed(monkeypatch, nonempty: bool = False) -> int:
    """The first smoke seed that AGREEs on the row path, i.e. one whose
    verdict went on through the confit-only boundary legs -- with at least
    one output row when `nonempty` (a planted disagreement that empties the
    answer needs rows to remove)."""
    calls = []
    real = oracle._extra_legs

    def spy(fn, case, table, got, tags):
        calls.append(len(got))
        return real(fn, case, table, got, tags)

    monkeypatch.setattr(oracle, "_extra_legs", spy)
    for seed in range(N):
        calls.clear()
        if (
            oracle.run_case(gen.gen(seed)).kind == "AGREE"
            and calls
            and (calls[0] > 0 or not nonempty)
        ):
            monkeypatch.setattr(oracle, "_extra_legs", real)
            return seed
    raise AssertionError("no row-path AGREE seed in the smoke range")


def test_opt_emulated_is_final_and_no_self_leg_replaces_it(monkeypatch):
    """OPT_EMULATED stops the case like any other mismatch: the boundary
    self-legs never run, so none of them can overwrite the primary finding.

    The class is empty in practice, so it is planted: the baseline reading is
    emptied (so it disagrees with us) while the optimizer-on reading is left
    alone (so it agrees), and a self-leg that would report DIVERGE_VALUE is
    standing by."""
    seed = _row_path_agree_seed(monkeypatch, nonempty=True)
    real_run = oracle._duck_run

    def off_disagrees(*a, **k):
        (out, phase, err), on = real_run(*a, **k)
        return (out.slice(0, 0) if len(out) else out, phase, err), on

    def poisoned_leg(fn, case, table, got, tags):
        return oracle.Verdict("DIVERGE_VALUE", "self-leg", "must not run", tags)

    monkeypatch.setattr(oracle, "_duck_run", off_disagrees)
    monkeypatch.setattr(oracle, "_extra_legs", poisoned_leg)
    v = oracle.run_case(gen.gen(seed))
    assert v.kind == "OPT_EMULATED", v


def test_a_refusal_keeps_the_oracle_outcome_it_already_computed():
    """REFUSED carries what the baseline reading did with the same query —
    served rows, rejected it at bind/build, or trapped at run time — so the
    campaign can report the cost of each refusal instead of discarding it."""
    outcomes = set()
    for seed in range(N):
        v = oracle.run_case(gen.gen(seed))
        if v.kind == "REFUSED":
            assert v.oracle in oracle.ORACLE_OUTCOMES, v
            assert v.to_json()["oracle"] == v.oracle
            outcomes.add(v.oracle)
        else:
            assert v.oracle == "", v
    assert "serves" in outcomes, outcomes


@functools.cache
def _refused_seed() -> int:
    return next(
        s
        for s in range(N)
        if oracle.run_case(gen.gen(s), report=False).kind == "REFUSED"
    )


def test_a_refusal_is_decided_before_duckdb_runs(monkeypatch):
    """The campaign worker sends a REFUSED verdict before the reading that
    only feeds the report, then `refusal_json` to complete it: the two
    together are exactly the in-process line."""
    seed = _refused_seed()
    whole = oracle.run_case_json(seed)

    def no_duckdb(*a, **k):
        raise AssertionError("DuckDB ran before the refusal was sent")

    with monkeypatch.context() as m:
        m.setattr(oracle, "_duck_run", no_duckdb)
        first = oracle.run_case_json(seed, report=False)
    assert first["kind"] == "REFUSED" and "oracle" not in first
    assert {**first, **oracle.refusal_json(seed)} == whole


def test_a_refusal_reading_that_raises_is_the_same_skip_either_way(monkeypatch):
    """An exception in the reading is the oracle's own bug: SKIP in-process,
    and the same SKIP when `refusal_json` completes the worker's line."""
    seed = _refused_seed()

    def boom(*a, **k):
        raise KeyError("planted")

    monkeypatch.setattr(oracle, "_duck_run", boom)
    whole = oracle.run_case_json(seed)
    assert (whole["kind"], whole["klass"]) == ("SKIP", "oracle:KeyError"), whole
    first = oracle.run_case_json(seed, report=False)
    assert {**first, **oracle.refusal_json(seed)} == whole


# --- one oracle database per process (fuzz.oracle.reuse_oracle) --------------


@pytest.fixture
def reused(monkeypatch):
    """Reuse switched on for one test, and its database closed after it."""
    monkeypatch.setattr(oracle._Reuse, "on", True)
    monkeypatch.setattr(oracle._Reuse, "con", None)
    monkeypatch.setattr(oracle._Reuse, "served", 0)
    yield oracle._Reuse
    oracle._discard()


def _count(con, sql: str) -> int:
    return con.execute(sql).fetchone()[0]


def test_a_reused_oracle_answers_like_a_fresh_one(reused):
    """Kind, class, detail and a refusal's outcome, case for case."""
    seeds = range(0, N, 2)
    reused.on = False
    fresh = [oracle.run_case(gen.gen(s)).to_json() for s in seeds]
    reused.on = True
    again = [oracle.run_case(gen.gen(s)).to_json() for s in seeds]
    assert again == fresh
    assert reused.served > 1  # the cases did share a database


def test_a_reused_oracle_is_emptied_and_a_leftover_discards_it(reused):
    case = next(c for c in map(gen.gen, range(PARITY_SEEDS)) if c.statics and c.udfs)
    sql = gen.render(case.query)
    oracle._duck_run(sql, case, oracle._udf_objs(case))
    con = reused.con
    assert _count(con, "SELECT count(*) FROM duckdb_tables()") == 0
    names = ", ".join(f"'{u.name}'" for u in case.udfs)
    in_use = f"SELECT count(*) FROM duckdb_functions() WHERE function_name IN ({names})"
    assert _count(con, in_use) == 0

    # Something the case did not make is still there after it: not reused.
    con.execute("CREATE TABLE stray AS SELECT 1 AS x")
    oracle._duck_run(sql, case, oracle._udf_objs(case))
    assert reused.con is None
    oracle._duck_run(sql, case, oracle._udf_objs(case))
    assert reused.con is not None and reused.con is not con


def test_a_failed_load_leaves_nothing_for_the_next_case(reused, monkeypatch):
    """A load that raises is the oracle's own bug (SKIP), and what the case
    had loaded before it does not reach the next case."""
    case = next(c for c in map(gen.gen, range(PARITY_SEEDS)) if c.statics)
    sql = gen.render(case.query)
    real = oracle.Oracle.load

    def load(self, name, table):
        if name == "__THIS__":
            raise RuntimeError("planted")
        return real(self, name, table)

    monkeypatch.setattr(oracle.Oracle, "load", load)
    with pytest.raises(RuntimeError, match="planted"):
        oracle._duck_run(sql, case, oracle._udf_objs(case))
    monkeypatch.setattr(oracle.Oracle, "load", real)
    oracle._duck_run(sql, case, oracle._udf_objs(case))  # same names load again
    assert _count(reused.con, "SELECT count(*) FROM duckdb_tables()") == 0


def test_a_reused_oracle_reads_the_next_baseline_with_the_optimizer_off(reused):
    """The bracket's second reading leaves the optimizer on; the next case's
    baseline reading must not inherit it. `WHERE 1 = 0` plans as
    EMPTY_RESULT only when the optimizer runs."""

    def optimized(con) -> bool:
        plan = con.execute("EXPLAIN SELECT 1 AS x WHERE 1 = 0").fetchall()
        return "EMPTY_RESULT" in plan[0][1]

    con = oracle._take()
    assert not optimized(con)
    con.optimizer_on()
    assert optimized(con)
    oracle._give_back(con, [], [])
    again = oracle._take()
    assert again is con and not optimized(again)
    oracle._give_back(again, [], [])


def test_a_reused_oracle_retires_after_its_cases(reused, monkeypatch):
    monkeypatch.setattr(oracle._Reuse, "CASES", 2)
    case = gen.gen(0)
    sql = gen.render(case.query)
    seen = []
    for _ in range(5):
        oracle._duck_run(sql, case, oracle._udf_objs(case))
        seen.append(reused.con)
    assert seen[0] is seen[1] and seen[1] is not seen[2] and seen[2] is seen[3]


def test_a_broken_non_null_promise_is_a_divergence(monkeypatch):
    """Our output schema's non-null promises are checked against our own
    rows, not against DuckDB's flags. Planted: the check reports a NULL."""
    seed = _row_path_agree_seed(monkeypatch)
    monkeypatch.setattr(oracle, "non_null_violation", lambda *a: "row 0 'o0'")
    v = oracle.run_case(gen.gen(seed))
    assert (v.kind, v.klass) == ("DIVERGE_VALUE", "unsound-non-null"), v


def test_a_verdict_line_carries_its_inputs_not_just_a_seed():
    """Seeds are not durable identities once the generator changes, so the
    line itself carries the SQL and every input needed to replay it, and
    survives a JSON round trip (the worker pipe and findings.jsonl)."""
    import json

    for seed in range(0, N, 7):
        line = json.loads(json.dumps(oracle.run_case_json(seed)))
        case = gen.gen(seed)
        assert line["sql"] == gen.render(case.query)
        inputs = line["inputs"]
        assert set(inputs) == {"row_schema", "rows", "statics", "udfs", "tree", "shape"}
        assert inputs["rows"] == json.loads(json.dumps(oracle.plain(case.rows)))
        assert set(inputs["statics"]) == set(case.statics)
        assert inputs["shape"] == case.shape


def test_no_feature_is_unshipped():
    """The UNSHIPPED machinery stays for the next unshipped width, but
    nothing takes it today: decimals, the last one, shipped."""
    assert oracle.UNSHIPPED_FEATURES == ()
    assert oracle.unshipped_reach("SELECT 2.5 AS o0 FROM __THIS__") == set()


def test_a_case_marks_each_phase_on_stderr_before_it_runs(capsys):
    """A worker killed mid-case cannot say where it was, so the case says it
    first: the last phase marker on stderr names the side that hung."""
    seed = next(s for s in range(N) if oracle.run_case(gen.gen(s)).kind == "AGREE")
    capsys.readouterr()
    oracle.run_case(gen.gen(seed))
    phases = [
        ln.split()[1]
        for ln in capsys.readouterr().err.splitlines()
        if ln.startswith(oracle.PHASE_MARK)
    ]
    assert phases[:3] == ["confit:build", "confit:run", "oracle"], phases


def test_a_case_revived_from_its_stored_inputs_answers_like_the_original():
    """`case_from_inputs` is what makes a stored case independent of the
    generator revision: the revived case must carry the same inputs, value
    types included (a Decimal stays a Decimal), and earn the same verdict."""
    import json

    for seed in range(0, N, 3):
        case = gen.gen(seed)
        stored = json.loads(
            json.dumps(
                {"sql": gen.render(case.query), "inputs": oracle.case_inputs(case)}
            )
        )
        revived = oracle.case_from_inputs(seed, stored["sql"], stored["inputs"])
        assert json.dumps(oracle.case_inputs(revived)) == json.dumps(
            oracle.case_inputs(case)
        ), seed
        assert repr(revived.rows) == repr(case.rows), seed
        assert repr(revived.statics) == repr(case.statics), seed
        a, b = oracle.run_case(case), oracle.run_case(revived)
        assert (a.kind, a.klass) == (b.kind, b.klass), seed


def test_the_subquery_candidate_snapshot_is_whole_and_revivable():
    """The frozen denominator of the subquery design: 150 phase-1 derived
    tables and 48 phase-2 CTEs, each replayable without the generator."""
    import collections
    import json

    path = (
        Path(__file__).parents[1] / "fuzz/corpora/subquery-candidates-2026-09-27.jsonl"
    )
    header, *lines = [json.loads(ln) for ln in path.read_text("utf-8").splitlines()]
    assert set(header) == {"provenance"}
    by = collections.Counter((c["phase"], c["form"]) for c in lines)
    assert by == {(1, "derived"): 150, (2, "cte"): 48}
    for c in lines:
        case = oracle.case_from_inputs(c["seed"], c["sql"], c["inputs"])
        assert case.sql and case.query is None


def test_a_resource_ceiling_is_excluded_by_name_and_duckdb_never_runs(monkeypatch):
    """Nightly seeds 1000308 and 1003321, shrunk: `repeat` or `lpad` to
    2**31 - 1 characters traps on the 1 GiB string budget (serving contract,
    exclusion: resource-ceilings). DuckDB would build the multi-gigabyte
    value instead, for minutes and uninterruptibly, so the case is EXCLUDED
    before DuckDB runs. Just under the budget is an ordinary comparison."""
    import pyarrow as pa
    from fuzz import parity

    def no_duckdb(*a, **k):
        raise AssertionError("DuckDB ran for an excluded case")

    big = pa.table({"c1": pa.array([2**31 - 1], pa.int32())})
    with monkeypatch.context() as m:
        m.setattr(oracle, "_duck_run", no_duckdb)
        for sql in (
            "SELECT repeat('a', c1) AS o0 FROM __THIS__",
            "SELECT lpad('x', c1, 'é☃') AS o0 FROM __THIS__",
        ):
            v = parity.verdict(sql, big)
            assert (v.kind, v.klass) == ("EXCLUDED", "resource-ceiling"), v
    assert "EXCLUDED" not in runner.GATED
    small = pa.table({"c1": pa.array([3], pa.int32())})
    sql = "SELECT repeat('a', c1) AS o0 FROM __THIS__"
    assert parity.verdict(sql, small).kind == "AGREE"


def test_the_generator_quotes_exactly_duckdbs_reserved_keywords():
    """`_ident` leaves a keyword bare only where DuckDB accepts it as a
    name; the engine's own list (frontend/expr.rs DUCKDB_RESERVED) is the
    same measurement."""
    import duckdb

    reserved = {
        k
        for (k,) in duckdb.sql(
            "SELECT keyword_name FROM duckdb_keywords() "
            "WHERE keyword_category = 'reserved'"
        ).fetchall()
    }
    assert gen.RESERVED == reserved
    assert gen._ident("from") == '"from"' and gen._ident("From") == '"From"'
    assert gen._ident("c0") == "c0"


# The seeds of the first nightly run's findings (issue ahrzb/sql-transforms#303),
# replayed through the same checks that flagged them.
NIGHTLY_303 = {1004531: "AGREE", 1003321: "EXCLUDED"}


@pytest.mark.parametrize("seed, kind", NIGHTLY_303.items())
def test_the_first_nightly_findings_stay_fixed(seed, kind):
    assert oracle.run_case(gen.gen(seed)).kind == kind


# Issue ahrzb/sql-transforms#305: a pure UDF over constant args folds at
# bind, so `udf0(NULL) * (overflow)` is NULL and `udf0(NULL, round(1.0e0),
# ..).f1` is DuckDB's SQLNULL.
NIGHTLY_305 = {1118442: "AGREE", 1120562: "AGREE"}


@pytest.mark.parametrize("seed, kind", NIGHTLY_305.items())
def test_the_second_nightly_findings_stay_fixed(seed, kind):
    assert oracle.run_case(gen.gen(seed)).kind == kind


# Issue ahrzb/sql-transforms#305, the six sharded nights from 2026-09-29: one
# representative per fixed class (tests/test_evaluation_order.py pins each
# rule on its own).
NIGHTLY_305_SHARDED = {
    # a UDF field over a closed constant argument confit's fold left unfinished
    **dict.fromkeys([1801793, 2254035, 2618315, 3015628, 3407126, 3811339], "AGREE"),
    # a narrow-width constant overflow the i64 fold hid
    **dict.fromkeys([2729519, 3829756, 2171165], "AGREE_TRAP"),
    # a static-only trapping ON conjunct, read through the key column
    **dict.fromkeys([2269747, 3078802, 4074032, 2070000], "REFUSED"),
    # a bare name both sides have, inside a JOIN ON key expression
    **dict.fromkeys([2017487, 2286850, 2926979, 3156364, 3589723], "REFUSED"),
    # a DOUBLE join key projects the static side's bits (-0.0)
    **dict.fromkeys([1913398, 2542589, 2678684, 3238498], "AGREE"),
    # a closed NULL operand spares its trapping sibling
    **dict.fromkeys(
        [2300537, 2323487, 2682080, 2692152, 3195590, 3279357, 3471867, 3725798]
        + [4025307, 3231941],
        "AGREE",
    ),
    # WHERE BETWEEN's upper bound is checked last
    3510602: "AGREE_TRAP",
    # 2026-10-05 nightly (seeds 4200000..4599999): a pure udf nested in a
    # closed call folds first; a tree call with a NULL id over a column
    # feature runs, so a trapping sibling traps.
    4234049: "AGREE",
    4316677: "AGREE_TRAP",
}


@pytest.mark.parametrize("seed, kind", NIGHTLY_305_SHARDED.items())
def test_the_sharded_nightly_findings_stay_fixed(seed, kind):
    assert oracle.run_case(gen.gen(seed)).kind == kind


# 4211849: a bare name shared with a static struct column is ambiguous
# (DuckDB's refusal), so qualifying it is a different query.
@pytest.mark.parametrize("seed", [1002698, 1002746, 1002993, 1004879, 4211849])
def test_the_first_nightly_spellings_stay_consistent(seed):
    from fuzz import metamorphic

    _, findings = metamorphic.check(seed)
    assert not findings, "\n".join(f.line() for f in findings)
