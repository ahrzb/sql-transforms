"""The campaign report, driven with synthetic verdict dicts.

`fuzz.runner.report` is the only place a campaign's verdicts become the
sections a reader acts on, so what it groups, separates and writes is pinned
here directly: nothing below needs a worker, a seed or DuckDB.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

import pytest  # noqa: E402
from fuzz import runner  # noqa: E402


def _r(seed, kind, klass="", oracle="", detail="", tags=()):
    out = {
        "seed": seed,
        "sql": f"SELECT {seed}",
        "kind": kind,
        "klass": klass,
        "detail": detail,
        "tags": list(tags),
    }
    if oracle:
        out["oracle"] = oracle
    return out


def _section(text: str, title: str) -> str:
    """The lines under `== title ... ==` up to the next section header."""
    lines = text.splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.startswith(f"== {title}"))
    body = []
    for ln in lines[start + 1 :]:
        if ln.startswith("=="):
            break
        body.append(ln)
    return "\n".join(body).strip("\n")


def test_refusals_are_summarized_by_oracle_outcome_and_class(tmp_path, capsys):
    results = [
        _r(1, "REFUSED", "unsupported: lpad", "serves"),
        _r(2, "REFUSED", "unsupported: lpad", "serves"),
        _r(3, "REFUSED", "unsupported: lpad", "traps"),
        _r(4, "REFUSED", "bind error: no column", "rejects"),
    ]
    runner.report(results, tmp_path / "f.jsonl")
    sec = _section(capsys.readouterr().out, "refusals by oracle outcome")
    assert sec.splitlines() == [
        f"  {'serves':14} 2",
        f"    {2:6}  unsupported: lpad",
        f"  {'traps':14} 1",
        f"    {1:6}  unsupported: lpad",
        f"  {'rejects':14} 1",
        f"    {1:6}  bind error: no column",
    ]
    # A refusal is reporting, not a finding: nothing is written for it.
    assert (tmp_path / "f.jsonl").read_text() == ""


def test_findings_open_with_a_dated_provenance_header(tmp_path, capsys):
    prov = runner.provenance(start=10, n=3)
    for key in (
        "date",
        "engine_revision",
        "generator_revision",
        "reference",
        "seeds",
        "platform",
    ):
        assert key in prov, key
    assert prov["seeds"] == [10, 12]
    assert prov["reference"]["duckdb"] == prov["reference"]["oracle_version"]
    assert "optimizer-off" in prov["reference"]["baseline"]

    out = tmp_path / "f.jsonl"
    runner.report([_r(1, "DIVERGE_VALUE", "values")], out, provenance=prov)
    lines = [json.loads(x) for x in out.read_text().splitlines()]
    assert lines[0] == {"provenance": prov}
    assert [x["kind"] for x in lines[1:]] == ["DIVERGE_VALUE"]


def test_a_dead_worker_is_blamed_with_its_sql_and_inputs():
    """The worker never returned, so the parent regenerates the case: the
    finding carries SQL and inputs, not a bare seed and an empty string."""
    r = runner.blame(3, "TIMEOUT", "stderr tail")
    assert r["kind"] == "TIMEOUT" and r["klass"] == "timeout"
    assert r["sql"].startswith("SELECT")
    assert "rows" in r["inputs"]
    assert r["detail"] == "stderr tail"


def test_every_kind_lands_in_exactly_one_outcome_category():
    from fuzz import oracle

    kinds = set(oracle.KINDS) | {"TIMEOUT", "PANIC"}
    assert set(runner.CATEGORY) == kinds
    assert set(runner.CATEGORY.values()) == {
        "agreement",
        "mismatch",
        "unresolved",
        "refused",
        "unshipped",
    }


def test_unresolved_is_counted_apart_from_agreement_and_mismatch(tmp_path, capsys):
    results = [
        _r(1, "AGREE"),
        _r(2, "AGREE", tags=["order-by-unevaluated"]),
        _r(3, "AGREE_TRAP"),
        _r(4, "DIVERGE_VALUE", "values"),
        _r(5, "OPT_EMULATED", "x"),
        _r(6, "SKIP", "oracle:KeyError"),
        _r(7, "TIMEOUT", "timeout"),
        _r(8, "REFUSED", "unsupported: x", "serves"),
        _r(9, "UNSHIPPED", "decimals"),
    ]
    out = tmp_path / "f.jsonl"
    runner.report(results, out)
    sec = _section(capsys.readouterr().out, "outcomes")
    assert sec.splitlines() == [
        f"  {'agreement':11} {3:6}",
        f"  {'':11} {1:6}  of them order-by-unevaluated: sortedness not established",
        f"  {'mismatch':11} {2:6}",
        f"  {'unresolved':11} {2:6}  "
        "no verdict: neither agreement nor a confirmed defect",
        f"  {'refused':11} {1:6}",
        f"  {'unshipped':11} {1:6}",
    ]
    lines = [json.loads(x) for x in out.read_text().splitlines()]
    assert {x["kind"]: x["category"] for x in lines} == {
        "DIVERGE_VALUE": "mismatch",
        "OPT_EMULATED": "mismatch",
        "SKIP": "unresolved",
        "TIMEOUT": "unresolved",
    }


@pytest.mark.parametrize(
    ("msg", "prefixed", "named", "actionable"),
    [
        ("unsupported: DISTINCT", True, True, False),
        ("bind error: ambiguous column 'c0' in JOIN ON (qualify it)", True, True, True),
        (
            "udf 'udf0': a width-1 list return is a scalar — declare the element "
            "type rather than pa.list_(t, 1)",
            False,
            True,
            True,
        ),
        (
            "unsupported: FROM (SELECT 2.5e0 AS o0 FROM __THIS__) AS sub",
            True,
            False,
            False,
        ),
        ("unsupported: expression: a IN (SELECT b FROM s)", True, False, False),
        (
            "unsupported: join type FullOuter(On(Nested(BinaryOp { left: "
            'Identifier(Ident { value: "c0", quote_style: None',
            True,
            False,
            False,
        ),
        (
            "unsupported: static table 's0' column 'c0' has type timestamp[us], which "
            "this engine does not serve — project a served column instead",
            True,
            True,
            True,
        ),
    ],
)
def test_refusal_quality_reads_naming_and_actionability(
    msg, prefixed, named, actionable
):
    q = runner.refusal_quality(msg)
    assert (q["prefixed"], q["named"], q["actionable"]) == (prefixed, named, actionable)


def test_refusal_quality_is_reported_as_shares_of_all_refusals(tmp_path, capsys):
    results = [
        _r(1, "REFUSED", "unsupported: DISTINCT", "serves", "unsupported: DISTINCT"),
        _r(2, "REFUSED", "x", "serves", "unsupported: FROM (SELECT 1 AS o0) AS sub"),
        _r(
            3,
            "REFUSED",
            "y",
            "rejects",
            "bind error: ambiguous column 'c0' (qualify it)",
        ),
        _r(4, "AGREE"),
    ]
    runner.report(results, tmp_path / "f.jsonl")
    sec = _section(capsys.readouterr().out, "refusal quality")
    assert sec.splitlines()[:4] == [
        f"  {'refusals':20} {3:6}",
        f"  {'documented prefix':20} {3:6}",
        f"  {'names the construct':20} {2:6}",
        f"  {'actionable':20} {1:6}",
    ]
    assert sec.splitlines()[4:] == [
        "  echoes source text instead of naming:",
        f"    {1:6}  unsupported: FROM (SELECT …",
    ]


def test_unshipped_widths_are_reported_with_their_reach(tmp_path, capsys):
    results = [
        _r(1, "UNSHIPPED", "decimals", tags=["reaches:decimals"]),
        _r(2, "REFUSED", "unsupported: x", "serves", tags=["reaches:decimals"]),
        _r(3, "AGREE"),
    ]
    runner.report(results, tmp_path / "f.jsonl")
    sec = _section(capsys.readouterr().out, "unshipped features")
    assert sec.splitlines()[0] == (
        f"  {'decimals':10} reached {2:6}  UNSHIPPED {1:6}  REFUSED {1:6}"
    )


def test_an_unreached_width_is_not_reported_as_support(tmp_path, capsys):
    runner.report([_r(1, "AGREE")], tmp_path / "f.jsonl")
    sec = _section(capsys.readouterr().out, "unshipped features")
    assert "decimals" in sec and "not reached" in sec


def test_findings_and_coverage_are_what_the_contract_says(tmp_path, capsys):
    """Observable, not tuple membership: one verdict of every kind through
    the report. Every mismatch and every unanswered case reaches the findings
    file; agreement, refusal and an unshipped width never do; and only AGREE
    feeds the construct-coverage histogram (OPT_EMULATED is a finding, never
    coverage)."""
    from fuzz import oracle

    kinds = list(oracle.KINDS) + ["TIMEOUT", "PANIC"]
    results = [_r(i, k, "k", tags=[f"tag-{k}"]) for i, k in enumerate(kinds)]
    out = tmp_path / "f.jsonl"
    runner.report(results, out)
    written = {json.loads(x)["kind"] for x in out.read_text().splitlines()}
    assert written == {
        "DIVERGE_VALUE",
        "DIVERGE_BUILD",
        "DIVERGE_TRAP",
        "DIVERGE_OPT",
        "OPT_EMULATED",
        "BUILD_EXC",
        "SKIP",
        "TIMEOUT",
        "PANIC",
    }
    cover = _section(capsys.readouterr().out, "AGREE coverage by construct")
    assert cover.split() == ["1", "tag-AGREE"]
