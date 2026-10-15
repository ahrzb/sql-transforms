"""Equivalent spellings agree (fuzz/metamorphic.py), plus the fixes it found.

The oracle campaign compares confit with DuckDB one spelling at a time.
These checks compare confit with itself across spellings that DuckDB treats
as the same query, so an over-refusal that only one spelling hits shows up
even where DuckDB has nothing to diverge from. The deep run is
`python -m fuzz.metamorphic --n 2000`; this is the gate-sized slice.
"""

import sys
from pathlib import Path

import pyarrow as pa
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

from fuzz import metamorphic  # noqa: E402
from fuzz.parity import assert_parity  # noqa: E402

N = 300


def test_spellings_agree():
    counts, findings = metamorphic.run(range(N))
    assert not findings, "\n".join(f.line() for f in findings)
    # A rewrite that never applies checks nothing.
    for name in metamorphic.REWRITES:
        assert counts[(name, "checked")] > 0, name


# ------------------------------------------------ struct-leaf join keys
# Found by lane-sub/lane-paren/lane-fn: a static struct leaf spelled any
# way but the dot path was not recognized as an equality key, so a
# multi-row static table refused as keyless.

ROW = pa.schema([pa.field("a", pa.int64())])
D = pa.table(
    {
        "k": pa.array(
            [{"f": 1, "g": 10}, {"f": 2, "g": 20}, {"f": None, "g": 30}],
            pa.struct([("f", pa.int64()), ("g", pa.int64())]),
        ),
        "v": ["one", "two", "none"],
    }
)


@pytest.mark.parametrize(
    "key",
    ["d.k.f", "d.k['f']", "(d.k).f", "struct_extract(d.k, 'f')", "k['f']", "(k).f"],
)
def test_every_spelling_of_a_struct_leaf_is_a_join_key(key):
    sql = f"SELECT a, v FROM __THIS__ LEFT JOIN d ON a = {key}"
    rows = pa.table({"a": pa.array([1, 2, 3, None], pa.int64())})
    assert_parity(sql, rows, statics={"d": D}, expect="AGREE")


@pytest.mark.parametrize("key", ["d['k']['f']", "(d).k.f", "struct_extract(d, 'k').f"])
def test_a_relation_read_as_a_struct_is_its_column(key):
    # DuckDB reads a relation name no column shares as its row struct, so
    # `d['k']['f']` is `d.k.f` there, a join key like every spelling above.
    sql = f"SELECT a, v FROM __THIS__ LEFT JOIN d ON a = {key}"
    rows = pa.table({"a": pa.array([1, 2, 3, None], pa.int64())})
    assert_parity(sql, rows, statics={"d": D}, expect="AGREE")


# ------------------------------------------------- found by the nightly run
# (issue ahrzb/sql-transforms#303, window 1000000..1004999)

T = pa.table(
    {
        "a": pa.array([1, 2], pa.int64()),
        "from": pa.array([{"f0": 1}, {"f0": 2}], pa.struct([("f0", pa.int64())])),
    }
)
DIM = pa.table({"id": pa.array([1], pa.int64()), "v": ["x"]})


@pytest.mark.parametrize("qual", ['"d"', '"D"', "d", "D"])
def test_a_quoted_wildcard_qualifier_names_its_relation(qual):
    # `"d".*` kept its quotes and matched no relation (quote-all rewrite).
    sql = f"SELECT {qual}.* FROM __THIS__ LEFT JOIN d ON a = d.id"
    assert_parity(sql, T.select(["a"]), statics={"d": DIM}, expect="AGREE")


@pytest.mark.parametrize(
    "sql, kind",
    [
        # DuckDB's grammar will not start a column reference with a reserved
        # keyword; confit built `(from).f0` (lane-paren rewrite).
        ("SELECT (from).f0 AS x FROM __THIS__", "REFUSED"),
        ("SELECT from.f0 AS x FROM __THIS__", "REFUSED"),
        ('SELECT "from".f0 AS x FROM __THIS__', "AGREE"),
        ("SELECT __THIS__.from.f0 AS x FROM __THIS__", "AGREE"),
    ],
)
def test_a_reserved_keyword_starts_no_column_reference(sql, kind):
    v = assert_parity(sql, T, expect=kind)
    if kind == "REFUSED":
        assert v.oracle == "rejects", v


# ------------------------------------------------ a panic is a finding
# The nightly of 2026-10-15 died in this suite: a Rust panic reaches Python
# as pyo3's `PanicException`, a BaseException, which `outcome` did not
# catch, and a pool worker cannot send it back, since it does not pickle.


class PanicException(BaseException):
    """pyo3's exception for a Rust panic, by name: it has no module that
    imports."""


def _panics(*_args):
    raise PanicException("assertion `left == right` failed")


class _PanicsOnRun:
    def infer_arrow(self, _table):
        raise PanicException("index out of bounds")


@pytest.mark.parametrize(
    "build, msg",
    [
        (_panics, "PanicException: assertion `left == right` failed"),
        (lambda *_args: _PanicsOnRun(), "PanicException: index out of bounds"),
    ],
)
def test_a_panic_in_the_build_or_the_run_is_an_outcome(monkeypatch, build, msg):
    monkeypatch.setattr(metamorphic.oracle, "_build", build)
    case = metamorphic.gen.gen(0)
    assert metamorphic.outcome(case, "SELECT 1") == ("panic", msg)


def test_an_interrupt_is_not_a_panic(monkeypatch):
    def interrupted(*_args):
        raise KeyboardInterrupt

    monkeypatch.setattr(metamorphic.oracle, "_build", interrupted)
    with pytest.raises(KeyboardInterrupt):
        metamorphic.outcome(metamorphic.gen.gen(0), "SELECT 1")


def test_a_panic_is_a_finding_even_when_both_spellings_panic(monkeypatch):
    monkeypatch.setattr(metamorphic.oracle, "_build", _panics)
    counts, findings = metamorphic.check(0)
    assert findings
    assert sum(counts[(n, "finding")] for n in metamorphic.REWRITES) == len(findings)
    assert all(f.base[0] == f.other[0] == "panic" for f in findings)
    assert "panic: PanicException" in findings[0].line()


def test_no_tolerance_excuses_a_panic(monkeypatch):
    # The base spelling refuses as ambiguous, which excuses a qualified
    # spelling's other outcome as a different query, but not its panic.
    base = metamorphic.gen.render(metamorphic.gen.gen(0).query)
    monkeypatch.setattr(
        metamorphic, "REWRITES", {"qualify-row-cols": lambda case: "SELECT 2"}
    )
    monkeypatch.setattr(
        metamorphic,
        "outcome",
        lambda case, sql: (
            ("refuse", "BinderException: ambiguous column c0")
            if sql == base
            else ("panic", "PanicException: boom")
        ),
    )
    counts, findings = metamorphic.check(0)
    assert counts[("qualify-row-cols", "finding")] == 1
    assert counts[("qualify-row-cols", "not-equivalent")] == 0
    assert [f.other[0] for f in findings] == ["panic"]
