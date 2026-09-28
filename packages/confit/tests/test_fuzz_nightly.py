"""The nightly's shards and its merged report, from synthetic summaries.

A night is split into shards that run on separate machines, and one report
is filed for all of them. What that report may not do is lose a shard's
seeds or findings without saying so; that is pinned here without running a
campaign.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

from fuzz import nightly  # noqa: E402


@pytest.mark.parametrize(("n", "k"), [(400_000, 4), (10, 3), (7, 7), (5, 1), (3, 5)])
def test_the_shards_cover_the_window_exactly_once(n, k):
    start = 1_234_567
    seeds = [s for i in range(k) for s in nightly.shard(start, n, i, k)]
    assert seeds == list(range(start, start + n))


def _summary(i, k, window, *, gated=(), found=0, examples=None, kinds=None):
    r = nightly.shard(*window, i, k)
    return {
        "seeds": [r.start, r.stop - 1],
        "shard": [i, k],
        "window": list(window),
        "workers": 4,
        "kinds": kinds or {"AGREE": len(r)},
        "gated": list(gated),
        "metamorphic": {
            "checked": 10,
            "findings": found,
            "examples": examples or {},
        },
        "t_campaign": 60.0 * (i + 1),
        "t_run": 70.0 * (i + 1),
    }


def _gated(seed, cases=1, kind="DIVERGE_VALUE", klass="values"):
    return {
        "kind": kind,
        "klass": klass,
        "cases": cases,
        "seed": seed,
        "shrunk": f"shrunk {seed}",
    }


def _merge(tmp_path, summaries, k) -> tuple[int, str]:
    dirs = []
    for s in summaries:
        d = tmp_path / f"shard-{s['shard'][0]}"
        d.mkdir()
        (d / "summary.json").write_text(json.dumps(s), encoding="utf-8")
        dirs.append(str(d))
    out = tmp_path / "merged"
    status = nightly.main(["--merge", *dirs, "--shards", str(k), "--out", str(out)])
    return status, (out / "report.md").read_text(encoding="utf-8")


def test_a_clean_night_passes_with_every_shard_counted(tmp_path, capsys):
    window = (1_000, 40)
    status, report = _merge(tmp_path, [_summary(i, 4, window) for i in range(4)], 4)
    assert status == 0
    assert report.startswith("# Nightly campaign, seeds 1000..1039 (4 of 4 shards)")
    assert "**pass**: 0 gated class(es), 0 metamorphic finding(s)." in report
    assert "Slowest shard: campaign 4m00s, whole run 4m40s, on 4 workers." in report
    assert "| AGREE | 40 |" in report
    assert "Metamorphic pairs checked: 40." in report


def test_a_class_found_in_two_shards_is_one_class_with_the_smaller_seed(tmp_path):
    window = (0, 200)
    shards = [
        _summary(1, 2, window, gated=[_gated(150, cases=2)], kinds={"AGREE": 98}),
        _summary(0, 2, window, gated=[_gated(50, cases=3)], kinds={"AGREE": 97}),
    ]
    shards[0]["kinds"]["DIVERGE_VALUE"] = 2
    shards[1]["kinds"]["DIVERGE_VALUE"] = 3
    status, report = _merge(tmp_path, shards, 2)
    assert status == 1
    assert "**FAIL**: 1 gated class(es)" in report
    assert "### DIVERGE_VALUE: values (5 case(s))" in report
    assert "shrunk 50" in report and "shrunk 150" not in report
    assert "| AGREE | 195 |" in report and "| DIVERGE_VALUE | 5 |" in report


def test_metamorphic_findings_fail_the_night_one_example_per_rewrite(tmp_path):
    window = (0, 20)
    shards = [
        _summary(0, 2, window, found=2, examples={"quote-all": "a", "wrap-cte": "b"}),
        _summary(1, 2, window, found=1, examples={"quote-all": "c"}),
    ]
    status, report = _merge(tmp_path, shards, 2)
    assert status == 1
    assert "0 gated class(es), 3 metamorphic finding(s)" in report
    body = report.split("## Metamorphic findings")[1]
    assert body.count("\na\n") == 1 and "\nb\n" in body and "\nc\n" not in body


def test_a_missing_shard_is_named_and_fails_the_night(tmp_path):
    window = (5_000, 30)
    status, report = _merge(
        tmp_path, [_summary(0, 3, window), _summary(2, 3, window)], 3
    )
    assert status == 1
    assert "(2 of 3 shards)" in report
    assert "**Missing:** shard 1/3 (seeds 5010..5019) left no summary" in report
    assert "**FAIL**" in report


def test_a_shard_outside_the_count_is_refused(tmp_path):
    out = tmp_path / "out"
    with pytest.raises(SystemExit):
        nightly.main(["--shard", "4/4", "--out", str(out)])
    assert not out.exists()
