"""The runner's side of the worker protocol, driven by stand-in workers.

A REFUSED verdict line is followed by one more line (`fuzz.oracle.refusal_json`):
the report-only DuckDB reading, which the runner reads under its own budget.
Stand-ins answer every even seed AGREE and every odd seed REFUSED, then
complete, hang in or die in that reading, so each branch of the runner is
driven directly: no engine, no DuckDB, no real seed.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

from fuzz import runner  # noqa: E402

STAND_IN = r"""
import json, sys, time
mode = sys.argv[1]
for line in sys.stdin:
    seed = int(line)
    kind = "REFUSED" if seed % 2 else "AGREE"
    verdict = {"seed": seed, "sql": "", "kind": kind, "klass": "k", "detail": "",
               "tags": [], "triples": []}
    print(json.dumps(verdict), flush=True)
    if kind == "REFUSED":
        if mode == "hang":
            time.sleep(60)
        if mode == "die":
            sys.exit(3)
        print(json.dumps({"oracle": "serves"}), flush=True)
"""


def _stand_in(monkeypatch, mode: str) -> list:
    """Every worker the runner spawns is a stand-in in `mode`; returns the
    list the spawned processes are appended to."""
    spawned = []

    def spawn():
        err = tempfile.TemporaryFile()
        proc = subprocess.Popen(  # noqa: S603 — fixed argv
            [sys.executable, "-c", STAND_IN, mode],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=err,
            text=True,
        )
        spawned.append(proc)
        return proc, err

    monkeypatch.setattr(runner, "_spawn", spawn)
    return spawned


def _campaign(tmp_path, n=4):
    # 1 s is far more than a stand-in needs between its two lines, and all a
    # hanging one costs.
    results = runner.campaign(
        0, n, 1, timeout=10.0, out=tmp_path / "f.jsonl", report_timeout=1.0
    )
    return {r["seed"]: r for r in results}


@pytest.mark.parametrize(
    ("mode", "outcome", "workers"),
    [("complete", "serves", 1), ("hang", "over-budget", 3), ("die", "died", 3)],
)
def test_a_refusal_reading_costs_at_most_its_outcome(
    tmp_path, monkeypatch, mode, outcome, workers
):
    """Completed, the reading's line fills in the outcome. Outrun or died in,
    the worker is replaced and the verdict stays REFUSED: never a gated
    TIMEOUT or PANIC. Either way the next seed is still answered."""
    spawned = _stand_in(monkeypatch, mode)
    got = _campaign(tmp_path)
    assert sorted(got) == [0, 1, 2, 3]
    assert [got[s]["kind"] for s in range(4)] == ["AGREE", "REFUSED"] * 2
    assert got[1]["oracle"] == got[3]["oracle"] == outcome
    assert "oracle" not in got[0]
    assert runner.gate(list(got.values()), None) == []
    assert len(spawned) == workers


def test_an_unfinished_reading_is_reported_with_its_reason(tmp_path, capsys):
    results = [
        {"seed": 1, "kind": "REFUSED", "klass": "x", "detail": "", "tags": []},
        {"seed": 2, "kind": "REFUSED", "klass": "x", "detail": "", "tags": []},
    ]
    results[0]["oracle"], results[1]["oracle"] = runner.UNFINISHED
    runner.report(results, tmp_path / "f.jsonl")
    out = capsys.readouterr().out
    assert "over-budget    1  the reading ran past --report-timeout" in out
    assert "died           1  the worker died in the reading" in out
