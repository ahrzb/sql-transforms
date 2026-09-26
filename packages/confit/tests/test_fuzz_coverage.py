"""(operator, argument-type, edge-class) coverage, read off generated SQL.

Raw query counts say how many cases ran; the triples say which operator saw
which argument type at which edge. These pin the extraction on hand-built
queries, then the report's reached-versus-agreed summary.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

from fuzz import coverage, gen, runner  # noqa: E402
from fuzz.gen import Bin, Call, Cast, Col, IsNull, Lit, Q, Sel  # noqa: E402


def _q(*exprs, where=None):
    return Q(
        [], Sel([(e, f"o{i}") for i, e in enumerate(exprs)], "__THIS__", where=where)
    )


@pytest.mark.parametrize(
    ("lit", "edge"),
    [
        (Lit(None, "int"), "null"),
        (Lit(0, "int"), "zero"),
        (Lit(-7, "int"), "negative"),
        (Lit(2**31 - 1, "int"), "extreme"),
        (Lit(float("nan"), "float"), "nonfinite"),
        (Lit(1e300, "float"), "extreme"),
        (Lit("", "str"), "empty"),
        (Lit("é☃", "str"), "non-ascii"),
        (Lit(5, "int"), "ordinary"),
    ],
)
def test_a_literal_argument_is_classed_by_its_edge(lit, edge):
    got = coverage.triples(_q(Call("abs", [lit])))
    assert ("abs", lit.ty, edge) in got


def test_operators_nest_and_read_through_where():
    q = _q(
        Bin("+", Col("c0", ty="int"), Call("abs", [Lit(0, "int")])),
        where=IsNull(Cast(Col("c1", ty="str"), "BIGINT")),
    )
    assert coverage.triples(q) == {
        ("+", "int", "column"),
        ("+", "expr", "expr"),
        ("abs", "int", "zero"),
        ("IS NULL", "expr", "expr"),
        ("CAST AS BIGINT", "str", "column"),
    }


def test_every_generated_case_yields_triples_it_can_serialize():
    for seed in range(0, 200, 13):
        t = coverage.triples(gen.gen(seed).query)
        assert all(isinstance(x, tuple) and len(x) == 3 for x in t)
        assert all(coverage.parse(coverage.key(x)) == x for x in t)


def test_the_report_counts_distinct_triples_reached_and_agreed(capsys, tmp_path):
    def r(kind, triples):
        return {
            "seed": 0,
            "sql": "",
            "kind": kind,
            "klass": "",
            "detail": "",
            "tags": [],
            "triples": [coverage.key(t) for t in triples],
        }

    results = [
        r("AGREE", [("abs", "int", "zero"), ("+", "int", "column")]),
        r("AGREE", [("abs", "int", "zero")]),
        r("REFUSED", [("abs", "int", "null"), ("lpad", "str", "empty")]),
    ]
    runner.report(results, tmp_path / "f.jsonl")
    out = capsys.readouterr().out
    lines = out[out.index("== coverage triples") :].split("\n\n")[0].splitlines()
    assert lines[1] == f"  {'distinct triples':18} reached {4:6}  agreed {2:6}"
    assert f"    {'abs':24} reached {2:4}  agreed {1:4}" in lines
    assert f"    {'lpad':24} reached {1:4}  agreed {0:4}" in lines
