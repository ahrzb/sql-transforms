"""The exclusion registry (`fuzz/exclusions.py`) keeps its promises.

Each record must be current (measured on this oracle, with a ledger entry and
a ruling), alive (its canaries are still excused), and narrow (a fault
planted inside its scope is not excused). The ledger table is the registry's
own output.
"""

import dataclasses
import json
import re
import sys
from pathlib import Path

import pyarrow as pa
import pytest
from confit import DuckDBInferFn
from confit.oracle import Oracle

sys.path.insert(0, str(Path(__file__).parents[1]))

from fuzz import exclusions as X  # noqa: E402
from fuzz import gen, oracle  # noqa: E402
from fuzz.parity import case, table, verdict  # noqa: E402

DOCS = Path(__file__).parents[1] / "docs"
REPO = Path(__file__).parents[3]

# ------------------------------------------------------------------ current


@pytest.mark.parametrize("rec", X.every(), ids=lambda r: r.id)
def test_a_record_is_measured_on_this_oracle(rec):
    assert rec.oracle == Oracle.VERSION, (
        f"{rec.id} was measured on DuckDB {rec.oracle}; remeasure it on "
        f"{Oracle.VERSION} (its canaries) before it applies again"
    )


@pytest.mark.parametrize("rec", X.every(), ids=lambda r: r.id)
def test_a_record_cites_its_ledger_entry_and_ruling(rec):
    text = "\n".join(p.read_text() for p in (DOCS / "oracle").glob("*.md"))
    text += (DOCS / "specs/serving-contract.md").read_text()
    assert f"**{rec.ledger}**" in text, f"no ledger entry '{rec.ledger}'"
    assert rec.status in ("ruled", "kept")
    if rec.status == "ruled":
        # A ruling is a package doc, or a closed proposal (proposals/),
        # and so is each later ruling that widens it.
        for ruling in (rec.ruling, *getattr(rec, "extended_by", ())):
            base = REPO if ruling.startswith("proposals/") else DOCS
            assert (base / ruling).is_file(), f"no ruling at {ruling}"
    else:
        row = next(line for line in text.splitlines() if f"**{rec.ledger}**" in line)
        assert "kept" in row, f"{rec.id} is kept but the ledger says otherwise"


def test_record_ids_are_unique():
    ids = [r.id for r in X.every()]
    assert len(ids) == len(set(ids))


def test_the_ledger_table_is_the_registry():
    assert X.ledger_block(X.LEDGER.read_text()) == X.ledger_table(), (
        "regenerate it: uv run python -m fuzz.exclusions --ledger"
    )


# ------------------------------------------------------------------ alive


@pytest.mark.parametrize(
    "rule, seed",
    [
        pytest.param(r, s, id=f"{r.id}-{s}")
        for r in X.RULES
        if r.timeout is None
        for s in r.canaries
    ],
)
def test_a_canary_is_still_excused(rule, seed):
    v = oracle.run_case(gen.gen(seed), report=False)
    assert (v.kind, v.klass) == ("EXCLUDED", rule.id), f"{v.kind} {v.klass}: {v.detail}"
    if rule.post is not None:
        assert v.raw, "an excused case keeps its raw verdict"


@pytest.mark.parametrize(
    "rule, seed",
    [
        pytest.param(r, s, id=f"{r.id}-{s}")
        for r in X.RULES
        if r.timeout is not None
        for s in r.canaries
    ],
)
def test_a_timeout_canary_is_still_excused(rule, seed):
    """The oracle's own reading of these seeds runs past the time limit of
    the campaign, which the runner (`fuzz.runner`) owns; only the witness
    that follows it runs here."""
    hit = X.after_timeout(gen.gen(seed))
    assert hit is not None and hit[0] == rule.id, hit


@pytest.mark.parametrize(
    "rule, seed",
    [
        pytest.param(r, s, id=f"{r.id}-{s}")
        for r in X.RULES
        if r.post is not None
        for s in r.canaries
    ],
)
def test_a_canary_is_excused_before_the_bracket_too(rule, seed, monkeypatch):
    """The campaign worker settles the verdict without the bracket while the
    case holds the reused database, so a witness that reads DuckDB then
    reads a database of its own (`oracle._take`)."""
    monkeypatch.setattr(oracle._Reuse, "on", True)
    monkeypatch.setattr(oracle._Reuse, "con", None)
    monkeypatch.setattr(oracle._Reuse, "served", 0)
    monkeypatch.setattr(oracle._Reuse, "held", False)
    sent = []
    try:
        line = oracle.run_case_json(seed, report=False, baseline=sent.append)
    finally:
        oracle._discard()
    got = [(x["kind"], x["klass"]) for x in [*sent, line]]
    assert got == [("EXCLUDED", rule.id)] * 2, got


class _Pair:
    """A UDF with a two-field struct output, the derived-table refusal's
    trigger."""

    name = "pair"
    takes = pa.schema([("a", pa.int64())])
    returns = pa.struct([("f", pa.int64()), ("g", pa.int64())])

    def __call__(self, a):
        return (a, a)


@pytest.mark.parametrize(
    "tol, sql, shape",
    [
        (
            "struct- or list-valued column in a derived table",
            "SELECT o FROM (SELECT pair(k) AS o FROM __THIS__) AS d",
            None,
        ),
        (
            "a subquery under a shape='many' join",
            "SELECT v FROM (SELECT k FROM __THIS__) AS q JOIN s ON q.k = s.id",
            "many",
        ),
    ],
)
def test_a_rewrite_tolerance_still_names_a_refusal_confit_makes(tol, sql, shape):
    assert any(t.refusal == tol for t in X.REWRITE_TOLERANCES)
    schema = pa.schema([pa.field("k", pa.int64())])
    s = pa.table({"id": pa.array([1], pa.int64()), "v": pa.array([2], pa.int64())})
    with pytest.raises(ValueError, match=re.escape(tol)):
        DuckDBInferFn(sql, {"__THIS__": schema}, {"s": s}, udfs=[_Pair()], shape=shape)


@pytest.mark.parametrize("rec", X.SOURCE_EXCLUSIONS, ids=lambda r: r.id)
def test_a_source_exclusion_names_a_source_in_the_corpus(rec):
    corpus = Path(__file__).parent / "corpus" / "duckdb_mined.jsonl"
    sources = {json.loads(line).get("source") for line in corpus.open(encoding="utf-8")}
    assert rec.source in sources


# ------------------------------------------------------------------ scope

ROW = table({"c0": "int8"}, [{"c0": 100}])
EMPTY = table({"k": "int"}, [])
FULL = table({"k": "int"}, [{"k": 5}])
# Rows whose `=` key is NULL can match no request, so DuckDB's hash table
# stays empty (proposals/closed/all-null-key-static.md).
NULLS = table({"k": "int?"}, [{"k": None}, {"k": None}])
NULL1 = table({"k": "int?"}, [{"k": None}])


def _facts(sql, statics, shape=None):
    sch = pa.schema([pa.field("c0", pa.int8(), nullable=False)])
    return DuckDBInferFn(sql, {"__THIS__": sch}, statics, shape=shape).plan_facts


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT 1 AS o FROM __THIS__ JOIN s ON c0 = s.k",
        "SELECT 1 AS o FROM __THIS__ CROSS JOIN s",
        "SELECT 1 AS o FROM __THIS__ JOIN s ON 7 = s.k",
        "SELECT 1 AS o FROM (SELECT c0 + 1 AS x FROM __THIS__) AS d JOIN s ON x = s.k",
        "SELECT 1 AS o FROM __THIS__ LEFT JOIN f ON c0 = f.k JOIN s ON c0 = s.k",
    ],
)
def test_every_spelling_of_an_inner_join_on_an_empty_static_is_in_scope(sql):
    scope = X.joins_static_where(rows=0, kinds=("INNER", "CROSS"))
    assert scope(_facts(sql, {"s": EMPTY, "f": FULL})) == ["s"]


@pytest.mark.parametrize(
    "sql, statics",
    [
        ("SELECT 1 AS o FROM __THIS__ LEFT JOIN s ON c0 = s.k", {"s": EMPTY}),
        ("SELECT 1 AS o FROM __THIS__ JOIN s ON c0 = s.k", {"s": FULL}),
        ("SELECT 1 AS o FROM __THIS__", {"s": EMPTY}),
    ],
)
def test_near_misses_are_out_of_scope(sql, statics):
    scope = X.joins_static_where(rows=0, kinds=("INNER", "CROSS"))
    assert scope(_facts(sql, statics)) == []


@pytest.mark.parametrize(
    "sql, statics",
    [
        ("SELECT 1 AS o FROM __THIS__ JOIN s ON c0 = s.k", {"s": NULLS}),
        ("SELECT 1 AS o FROM __THIS__ JOIN s ON c0 = s.k", {"s": NULL1}),
        (
            "SELECT 1 AS o FROM (SELECT c0 + 1 AS x FROM __THIS__) AS d "
            "JOIN s ON x = s.k",
            {"s": NULLS},
        ),
        # Each row has a NULL in one of its two keys.
        (
            "SELECT 1 AS o FROM __THIS__ JOIN s ON c0 = s.k AND c0 = s.j",
            {
                "s": table(
                    {"k": "int?", "j": "int?"},
                    [{"k": None, "j": 1}, {"k": 1, "j": None}],
                )
            },
        ),
    ],
)
def test_an_inner_join_on_a_key_that_is_null_in_every_static_row_is_in_scope(
    sql, statics
):
    assert X._NULL_KEYED(_facts(sql, statics)) == ["s"]


@pytest.mark.parametrize(
    "sql, statics",
    [
        # A CROSS join pairs each request with every static row, NULL or not.
        ("SELECT 1 AS o FROM __THIS__ CROSS JOIN s", {"s": NULL1}),
        # IS NOT DISTINCT FROM matches a NULL key with a NULL.
        (
            "SELECT 1 AS o FROM __THIS__ JOIN s ON c0 IS NOT DISTINCT FROM s.k",
            {"s": NULL1},
        ),
        ("SELECT 1 AS o FROM __THIS__ LEFT JOIN s ON c0 = s.k", {"s": NULLS}),
        (
            "SELECT 1 AS o FROM __THIS__ JOIN s ON c0 = s.k",
            {"s": table({"k": "int?"}, [{"k": None}, {"k": 5}])},
        ),
    ],
)
def test_near_misses_of_the_null_key_scope_are_out_of_it(sql, statics):
    assert X._NULL_KEYED(_facts(sql, statics)) == []


# ------------------------------------------------------------------ narrow

# Without the rule, these cases are a trap mismatch: DIVERGE_TRAP, or
# OPT_EMULATED where optimizer-on DuckDB traps as confit does.
MISMATCH = ("DIVERGE_TRAP", "OPT_EMULATED")

# The empty-static rule's own shape, measured: confit traps evaluating the
# row side, DuckDB skips it under a FILTER and answers no rows.
TRAPS = (
    "SELECT 1 AS o FROM (SELECT c0 + c0 AS x FROM __THIS__ WHERE c0 > 0) AS d "
    "JOIN s ON x = s.k"
)


@pytest.mark.parametrize(
    "static", [EMPTY, NULLS, NULL1], ids=["empty", "nulls", "null"]
)
def test_a_hand_written_empty_static_trap_is_excused(static):
    v = verdict(TRAPS, ROW, statics={"s": static})
    assert (v.kind, v.klass) == ("EXCLUDED", "empty-static-trap-timing")
    # Optimizer-on DuckDB traps here too, so the raw verdict is OPT_EMULATED.
    assert v.raw.split()[0] in MISMATCH


def test_the_same_trap_on_a_left_join_is_compared():
    # Out of scope: DuckDB traps on a LEFT join, and so does confit.
    v = verdict(TRAPS.replace(" JOIN s", " LEFT JOIN s"), ROW, statics={"s": EMPTY})
    assert v.kind == "AGREE_TRAP"


@pytest.mark.parametrize(
    "sql",
    [
        TRAPS,
        # nightly seed 8522528: a CROSS JOIN whose WHERE key traps
        "SELECT 1 AS o FROM __THIS__ CROSS JOIN s "
        "WHERE (c0 + 9223372036854775807) = s.k",
    ],
)
@pytest.mark.parametrize("spec", ["timestamp", "timestamp?"])
def test_an_empty_static_with_a_timestamp_column_is_padded(sql, spec):
    # The witness needs a plain value for each column of the static. It had
    # none for a timestamp, so such a case stayed a finding.
    empty = table({"k": "int", "t": spec}, [])
    v = verdict(sql, ROW, statics={"s": empty})
    assert (v.kind, v.klass) == ("EXCLUDED", "empty-static-trap-timing")


class _Inject:
    """A confit build whose every run raises `message`: a planted fault."""

    def __init__(self, fn, message):
        self._fn, self._message = fn, message

    def __getattr__(self, name):
        return getattr(self._fn, name)

    def infer_arrow(self, _table):
        raise ValueError(self._message)


@pytest.fixture
def plant(monkeypatch):
    def install(message):
        real = oracle._build

        def build(*a, **kw):
            return _Inject(real(*a, **kw), message)

        monkeypatch.setattr(oracle, "_build", build)

    return install


@pytest.mark.parametrize("static", [EMPTY, NULLS], ids=["empty", "nulls"])
def test_a_planted_trap_of_another_category_is_excused(plant, static):
    # DuckDB's padded run traps with an Out of Range Error; confit's planted
    # trap is a Conversion Error. The witness accepts any trap, as AGREE_TRAP
    # does (proposals/closed/empty-static-witness-trap-order.md).
    plant("Conversion Error: planted")
    v = verdict(TRAPS, ROW, statics={"s": static})
    assert (v.kind, v.klass) == ("EXCLUDED", "empty-static-trap-timing")


@pytest.mark.parametrize("static", [EMPTY, NULLS], ids=["empty", "nulls"])
def test_a_planted_trap_where_duckdb_returns_rows_is_not_excused(plant, static):
    # Nothing on the row side traps, so DuckDB's padded run answers rows: the
    # planted trap is not the mechanism.
    plant("Out of Range Error: planted")
    sql = "SELECT 1 AS o FROM __THIS__ JOIN s ON c0 = s.k"
    v = verdict(sql, ROW, statics={"s": static})
    assert v.kind in MISMATCH, f"{v.kind} {v.klass}"


@pytest.mark.parametrize("static", [EMPTY, NULLS], ids=["empty", "nulls"])
def test_a_planted_trap_on_one_backend_is_not_excused(monkeypatch, static):
    # Nothing on the row side traps. The Cranelift build traps and the
    # interpreter answers no rows: a backend split, which the rule never
    # excuses.
    real = oracle._build

    def build(sql, schema, statics, udf_objs, shape, force_interp):
        fn = real(sql, schema, statics, udf_objs, shape, force_interp)
        return fn if force_interp else _Inject(fn, "Out of Range Error: planted")

    monkeypatch.setattr(oracle, "_build", build)
    sql = "SELECT 1 AS o FROM __THIS__ JOIN s ON c0 = s.k"
    v = verdict(sql, ROW, statics={"s": static})
    assert (v.kind, v.klass) == ("DIVERGE_VALUE", "backend-trap-split")


def test_a_ceiling_on_one_backend_is_not_excused():
    ceiling = f"ValueError: {X.RESOURCE_CEILINGS[0]}"
    one = X.Evidence(sql="", facts={}, trap_cl=ceiling, trap_in=None)
    both = dataclasses.replace(one, trap_in=ceiling)
    assert X.pre_oracle(one) is None
    assert X.pre_oracle(both) == ("resource-ceiling", ceiling)


def test_a_record_off_its_oracle_version_excuses_nothing(monkeypatch):
    rule = dataclasses.replace(X.RULES[1], oracle="0.0.0")
    monkeypatch.setattr(X, "RULES", (X.RULES[0], rule))
    v = verdict(TRAPS, ROW, statics={"s": EMPTY})
    assert v.kind in MISMATCH


@pytest.mark.parametrize("static", [EMPTY, NULLS], ids=["empty", "nulls"])
def test_the_padding_case_keeps_the_original_inputs(static):
    # The witness re-runs DuckDB on a copy; the case the verdict reports on
    # is unchanged.
    c = case(TRAPS, ROW, statics={"s": static})
    before = list(c.statics["s"][1])
    oracle.run_case(c)
    assert c.statics["s"][1] == before


# ------------------------------------------------------------ timeout witness

# The shape of the oracle-timeout rule: confit traps on the first request
# (a TINYINT overflow) and never builds the string of the second, whose
# length is 2^31 - 1. The oracle builds the string for every request first.
LONG = 2**31 - 1
LENGTHS = table(
    {"c0": "int8", "n": "int32"}, [{"c0": 100, "n": 5}, {"c0": 1, "n": LONG}]
)
TIMEOUT_RULE = "oracle-timeout-before-a-trap"


def _long(builder="lpad('x', n, 'y')", other="c0 + c0", rows=LENGTHS):
    return case(f"SELECT {builder} AS a, {other} AS b FROM __THIS__", rows)


@pytest.mark.parametrize(
    "builder", ["lpad('x', n, 'y')", "rpad('x', n, 'y')", "repeat('x', n)"]
)
def test_a_trap_before_a_long_string_is_excused(builder):
    hit = X.after_timeout(_long(builder))
    assert hit is not None and hit[0] == TIMEOUT_RULE, hit
    assert "Overflow in addition of INT8" in hit[1]


def test_a_witness_that_caps_no_length_excuses_nothing():
    # Each length is below the cap, so the long string does not cause the
    # time limit.
    rows = table({"c0": "int8", "n": "int32"}, [{"c0": 100, "n": 5}, {"c0": 1, "n": 7}])
    assert X.after_timeout(_long(rows=rows)) is None


def test_a_planted_trap_where_the_capped_oracle_returns_rows_is_not_excused(plant):
    # Nothing traps in the query, so the capped oracle answers rows.
    plant("Out of Range Error: planted")
    assert X.after_timeout(_long(other="c0")) is None


def test_a_planted_trap_of_another_kind_is_not_excused(plant):
    # The capped oracle raises an Out of Range Error. A cap can change what
    # the query computes, so an error of another kind can be the cap's own.
    plant("Conversion Error: planted")
    assert X.after_timeout(_long()) is None


def test_a_planted_trap_that_names_no_kind_is_excused(plant):
    # Some of confit's trap texts name no kind of error, such as "cannot take
    # square root of a negative number". Then any error of the oracle counts.
    plant("planted, with no kind")
    hit = X.after_timeout(_long())
    assert hit is not None and hit[0] == TIMEOUT_RULE, hit


def test_a_planted_trap_on_one_backend_is_not_excused_after_a_timeout(monkeypatch):
    # Nothing traps in the query. The Cranelift build traps and the
    # interpreter answers rows: a backend split, never excused.
    real = oracle._build

    def build(sql, schema, statics, udf_objs, shape, force_interp):
        fn = real(sql, schema, statics, udf_objs, shape, force_interp)
        return fn if force_interp else _Inject(fn, "Out of Range Error: planted")

    monkeypatch.setattr(oracle, "_build", build)
    assert X.after_timeout(_long(other="c0")) is None


def test_a_timeout_rule_off_its_oracle_version_excuses_nothing(monkeypatch):
    rules = tuple(
        dataclasses.replace(r, oracle="0.0.0") if r.timeout is not None else r
        for r in X.RULES
    )
    monkeypatch.setattr(X, "RULES", rules)
    assert X.after_timeout(_long()) is None
