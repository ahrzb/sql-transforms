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
        # A ruling is a package doc, or a loop's decision record (loops/).
        base = REPO if rec.ruling.startswith("loops/") else DOCS
        assert (base / rec.ruling).is_file(), f"no ruling at {rec.ruling}"
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
    [pytest.param(r, s, id=f"{r.id}-{s}") for r in X.RULES for s in r.canaries],
)
def test_a_canary_is_still_excused(rule, seed):
    v = oracle.run_case(gen.gen(seed), report=False)
    assert (v.kind, v.klass) == ("EXCLUDED", rule.id), f"{v.kind} {v.klass}: {v.detail}"
    if rule.post is not None:
        assert v.raw, "an excused case keeps its raw verdict"


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


def test_a_hand_written_empty_static_trap_is_excused():
    v = verdict(TRAPS, ROW, statics={"s": EMPTY})
    assert (v.kind, v.klass) == ("EXCLUDED", "empty-static-trap-timing")
    # Optimizer-on DuckDB traps here too, so the raw verdict is OPT_EMULATED.
    assert v.raw.split()[0] in MISMATCH


def test_the_same_trap_on_a_left_join_is_compared():
    # Out of scope: DuckDB traps on a LEFT join, and so does confit.
    v = verdict(TRAPS.replace(" JOIN s", " LEFT JOIN s"), ROW, statics={"s": EMPTY})
    assert v.kind == "AGREE_TRAP"


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


def test_a_planted_trap_of_another_category_is_not_excused(plant):
    # DuckDB's padded run traps with an Out of Range Error; confit's planted
    # trap is a Conversion Error.
    plant("Conversion Error: planted")
    v = verdict(TRAPS, ROW, statics={"s": EMPTY})
    assert v.kind in MISMATCH, f"{v.kind} {v.klass}"


def test_a_planted_trap_where_duckdb_returns_rows_is_not_excused(plant):
    # Nothing on the row side traps, so DuckDB's padded run answers rows: the
    # planted trap is not the mechanism.
    plant("Out of Range Error: planted")
    sql = "SELECT 1 AS o FROM __THIS__ JOIN s ON c0 = s.k"
    v = verdict(sql, ROW, statics={"s": EMPTY})
    assert v.kind in MISMATCH, f"{v.kind} {v.klass}"


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


def test_the_padding_case_keeps_the_original_inputs():
    # The witness re-runs DuckDB on a copy; the case the verdict reports on
    # is unchanged.
    c = case(TRAPS, ROW, statics={"s": EMPTY})
    oracle.run_case(c)
    assert c.statics["s"][1] == []
