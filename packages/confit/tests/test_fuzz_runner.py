"""The campaign runner's failure boundaries, with stand-in workers.

A campaign is evidence, so the ways it can quietly stop being evidence are
what this file pins: a worker that dies still owes us the inputs it died on,
prior artifacts are never clobbered, a worker that breaks the protocol fails
the run instead of shrinking it, and the summary distinguishes agreement,
refusal-with-oracle-outcome and unshipped from each other.

The workers here are two-line scripts, not fuzz.worker: the behaviours under
test are dying, hanging and lying, which a real worker only does by accident.
The real worker's own contract -- generate once, announce, then evaluate --
is exercised by the campaign smoke in test_fuzz_smoke.py and by the CLI.
"""

from __future__ import annotations

import decimal
import json
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

from fuzz import gen, runner, worker  # noqa: E402

CASE = (
    '{{"event": "case", "seed": {seed}, "sql": "SELECT {seed}", '
    '"tags": ["tag"], "case": {{"seed": {seed}, "rows": [{{"a": 1}}]}}}}'
)

AGREES = f"""
import json, sys
for line in sys.stdin:
    s = line.strip()
    if not s:
        continue
    seed = int(s)
    print('{CASE}'.format(seed=seed), flush=True)
    print(json.dumps({{"event": "result", "seed": seed, "sql": "SELECT %d" % seed,
                      "kind": "AGREE", "klass": "", "detail": "", "tags": ["tag"],
                      "oracle_outcome": None}}), flush=True)
"""

DIES_AFTER_ANNOUNCING = f"""
import os, sys
for line in sys.stdin:
    s = line.strip()
    if not s:
        continue
    seed = int(s)
    sys.stdout.write('{CASE}'.format(seed=seed) + chr(10))
    sys.stdout.flush()
    sys.stderr.write("worker exploded on " + s + chr(10))
    sys.stderr.flush()
    os._exit(3)
"""

SPEAKS_GARBAGE = """
import sys
for line in sys.stdin:
    if line.strip():
        sys.stdout.write("not json at all" + chr(10))
        sys.stdout.flush()
"""

MARKS_THAT_IT_RAN = """
import os, sys
open(os.environ["MARKER"], "w").close()
for line in sys.stdin:
    pass
"""


def _worker(tmp_path: Path, source: str, name: str = "w.py") -> list[str]:
    script = tmp_path / name
    script.write_text(source, encoding="utf-8")
    return [sys.executable, str(script)]


def _lines(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x]


@pytest.mark.parametrize("hang", [False, True])
def test_a_failed_worker_keeps_the_case_it_announced(tmp_path, hang):
    """Both crashes and deadlines preserve inputs and account for every seed."""
    out = tmp_path / "findings.jsonl"
    source = DIES_AFTER_ANNOUNCING
    if hang:
        source = source.replace("os._exit(3)", "import time; time.sleep(60)")
    # one death per seed, and the worker is replaced each time, so every seed
    # still answers: dying is a finding, not a lost case.
    results = runner.campaign(
        0, 2, 1, 3.0 if hang else 30.0, out, argv=_worker(tmp_path, source)
    )
    assert [r["kind"] for r in results] == ["TIMEOUT" if hang else "PANIC"] * 2
    assert [r["sql"] for r in results] == ["SELECT 0", "SELECT 1"]
    assert all(r["case_announced"] for r in results)
    assert all("worker exploded" in r["detail"] for r in results)

    cases = _lines(runner.artifacts_for(out).cases)
    assert [c["seed"] for c in cases] == [0, 1]
    assert cases[0]["case"] == {"seed": 0, "rows": [{"a": 1}]}


def test_every_verdict_is_kept_not_only_the_findings(tmp_path):
    """AGREE writes no finding, and used to leave no trace at all. The
    results and cases sidecars are what make "N cases, all agreed" checkable
    instead of a claim."""
    out = tmp_path / "findings.jsonl"
    art = runner.artifacts_for(out)
    results = runner.campaign(0, 3, 2, 30.0, out, argv=_worker(tmp_path, AGREES))

    assert [r["seed"] for r in results] == [0, 1, 2]
    assert _lines(art.findings) == []
    assert [r["kind"] for r in _lines(art.results)] == ["AGREE"] * 3
    assert sorted(c["seed"] for c in _lines(art.cases)) == [0, 1, 2]

    prov = json.loads(art.provenance.read_text(encoding="utf-8"))
    assert prov["campaign"]["status"] == "ok"
    assert prov["campaign"]["n"] == 3
    assert prov["campaign"]["started"] < prov["campaign"]["finished"]


def test_existing_artifacts_are_never_overwritten(tmp_path):
    """A campaign is hours of machine time and its findings file is the only
    record of it. Landing on one must cost a message, not the file."""
    out = tmp_path / "findings.jsonl"
    art = runner.artifacts_for(out)
    art.cases.write_text("earlier campaign\n", encoding="utf-8")

    with pytest.raises(runner.CampaignError) as e:
        runner.campaign(0, 1, 1, 30.0, out, argv=_worker(tmp_path, AGREES))

    assert art.cases.name in str(e.value)
    assert art.cases.read_text(encoding="utf-8") == "earlier campaign\n"
    assert not art.findings.exists()
    assert not art.results.exists()
    assert not art.provenance.exists()


def test_a_worker_that_breaks_the_protocol_fails_the_campaign(tmp_path):
    """Losing cases and printing a clean summary is the failure this exists
    to prevent: the run raises, says which seeds never answered, and keeps
    the partial evidence."""
    out = tmp_path / "findings.jsonl"
    art = runner.artifacts_for(out)

    with pytest.raises(runner.CampaignError) as e:
        runner.campaign(0, 3, 1, 30.0, out, argv=_worker(tmp_path, SPEAKS_GARBAGE))

    assert "non-JSON" in str(e.value)
    assert "never produced a verdict" in str(e.value)
    assert not art.findings.exists()  # no summary, no findings file
    prov = json.loads(art.provenance.read_text(encoding="utf-8"))
    assert prov["campaign"]["status"] == "failed"
    assert prov["campaign"]["verdicts_recorded"] < 3
    assert any("non-JSON" in x for x in prov["campaign"]["errors"])


def test_a_reference_that_will_not_open_starts_no_workers(tmp_path, monkeypatch):
    """A broken oracle turns every case into SKIP, and 20000 SKIPs read like
    a campaign that ran. Preflight is what keeps that from being reportable."""
    import confit.oracle

    def boom():
        raise RuntimeError("duckdb 9.9.9, expected 1.5.5")

    monkeypatch.setattr(confit.oracle, "Oracle", boom)
    marker = tmp_path / "ran.txt"
    monkeypatch.setenv("MARKER", str(marker))
    out = tmp_path / "findings.jsonl"

    with pytest.raises(runner.CampaignError) as e:
        runner.campaign(0, 5, 1, 30.0, out, argv=_worker(tmp_path, MARKS_THAT_IT_RAN))

    assert "preflight" in str(e.value) and "9.9.9" in str(e.value)
    assert not marker.exists()
    assert not runner.artifacts_for(out).results.exists()


def _r(kind, klass="", outcome=None, tags=(), seed=0):
    return {
        "seed": seed,
        "kind": kind,
        "klass": klass,
        "detail": "d",
        "sql": "SELECT 1",
        "tags": list(tags),
        "oracle_outcome": outcome,
    }


def test_the_report_keeps_agreement_refusal_cost_and_unshipped_apart(tmp_path, capsys):
    """Three populations that must never merge: AGREE is the only coverage,
    a refusal is summarized by what the ORACLE did with the same query, and
    an unmeasured outcome stays visibly unknown rather than counting as
    harmless."""
    results = [
        _r("AGREE", tags=["join"], seed=1),
        _r("REFUSED", "window function", "served", seed=2),
        _r("REFUSED", "window function", "build-error", seed=3),
        _r("REFUSED", "window function", None, seed=4),
        _r("REFUSED", "decimal arithmetic", "run-error", seed=5),
        _r("UNSHIPPED", "decimal-literal", tags=["lit"], seed=6),
        _r("OPT_EMULATED", "emulation", tags=["join"], seed=7),
        _r("DIVERGE_OPT", "optimizer diagnostic", seed=8),
    ]
    out = tmp_path / "f.jsonl"
    runner.report(results, out)
    printed = capsys.readouterr().out

    assert "served=1" in printed and "build-error=1" in printed
    assert "unknown=1" in printed and "run-error=1" in printed
    rows = [line.split() for line in printed.splitlines()]
    assert ["1", "join"] in rows
    assert ["1", "lit"] not in rows
    assert "decimal-literal" in printed
    assert {row["kind"] for row in _lines(out)} == {"OPT_EMULATED", "DIVERGE_OPT"}


def test_case_inputs_survive_json_with_their_types():
    """Encoding is the load-bearing half of keeping a case: an exact decimal
    that reached past 2^53, a NaN column and the query AST all have to come
    back as themselves, under STRICT json -- `allow_nan` would write a token
    no other reader accepts."""
    case = gen.gen(7)
    blob = json.dumps(worker.encode(case), allow_nan=False)
    back = json.loads(blob)

    assert back["$dataclass"] == "Case"
    assert back["fields"]["seed"] == 7
    assert back["fields"]["query"]["$dataclass"] == "Q"

    payload = worker.encode(
        {
            "exact": decimal.Decimal("123456789012345678.000001"),
            "nan": math.nan,
            "ninf": -math.inf,
            "struct": gen.Struct(fields=(("f0", "int64"),), nullable=True),
        }
    )
    round_tripped = json.loads(json.dumps(payload, allow_nan=False))
    assert round_tripped["exact"] == {"$decimal": "123456789012345678.000001"}
    assert round_tripped["nan"] == {"$float": "nan"}
    assert round_tripped["ninf"] == {"$float": "-inf"}
    assert round_tripped["struct"]["$dataclass"] == "Struct"
    assert round_tripped["struct"]["fields"]["fields"] == {
        "$tuple": [{"$tuple": ["f0", "int64"]}]
    }


def test_report_failure_keeps_partial_evidence_marked_failed(tmp_path, monkeypatch):
    def fail_report(*args):
        raise OSError("output unavailable")

    monkeypatch.setattr(runner, "report", fail_report)
    out = tmp_path / "failed.jsonl"
    with pytest.raises(runner.CampaignError):
        runner.campaign(0, 1, 1, 30.0, out, argv=_worker(tmp_path, AGREES))
    artifacts = runner.artifacts_for(out)
    assert _lines(artifacts.results)[0]["kind"] == "AGREE"
    provenance = json.loads(artifacts.provenance.read_text(encoding="utf-8"))
    assert provenance["campaign"]["status"] == "failed"


def test_unrecordable_inputs_are_not_evaluated(monkeypatch, capsys):
    def cannot_encode(value):
        raise TypeError("unsupported case value")

    def must_not_run(case):
        raise AssertionError("evaluated without recoverable inputs")

    monkeypatch.setattr(worker, "encode", cannot_encode)
    monkeypatch.setattr(worker, "run_case_json", must_not_run)
    worker.run_seed(7)
    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert events[0]["case"]["$error"].startswith("encode:")
    assert events[1]["kind"] == "SKIP"
