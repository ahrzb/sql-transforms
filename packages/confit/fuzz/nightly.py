"""The nightly deep campaign: fresh seeds each night, findings shrunk, one report.

    uv run python -m fuzz.nightly --out nightly/
    uv run python -m fuzz.nightly --shard 2/4 --out shard2/
    uv run python -m fuzz.nightly --merge shard0/ shard1/ shard2/ shard3/ \\
        --shards 4 --out nightly/

Each night owns the next window of `N` seeds, so seeds that earlier nights
have not reached get covered. `--n` only chooses how many of them a run
takes, from the start of the window, so a shorter manual run is a prefix of
the scheduled one and never reaches another night's seeds; `--seed` names a
window of the run's own instead. `--shard I/K` runs the I-th of K contiguous
shards of the run's seeds, so one night can spread over K machines (the
workflow runs four). Each shard gets two checks:
- the strict oracle campaign (`fuzz.runner`, the acceptance gate);
- the metamorphic spelling suite (`fuzz.metamorphic`).

For each finding class, one representative seed is shrunk by `fuzz.shrink`.
That runs in a subprocess with a time cap, because a finding can be a panic
or a hang. A shard writes what it found to `<out>/summary.json` and renders
it as `<out>/report.md`. `--merge` renders the shards' summaries as the
night's ONE report, written to be filed as an issue. It names any shard that
left no summary, and leaves out, by name, any summary that is not a shard of
the same run. The exit status is 1 when anything gated, when the metamorphic
suite found anything, or when a shard is missing or left out.
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# Night 0. The window base sits far above the seeds hand campaigns use
# (0..~60k), so the nightly never re-covers them.
EPOCH = dt.date(2026, 9, 27)
BASE = 1_000_000
# A night's seeds, all shards together, and the stride between nights: night d
# owns BASE + d*N .. BASE + (d+1)*N - 1, whatever `--n` a run takes of it.
# Changing N moves every later window; moving forward skips seeds for good,
# which costs nothing, since every seed is a fresh case.
N = 400_000


def window(today: dt.date | None = None) -> int:
    """The first seed of tonight's window."""
    return BASE + ((today or dt.date.today()) - EPOCH).days * N


def shard(start: int, n: int, i: int, k: int) -> range:
    """The `i`-th of `k` shards of the window `start .. start + n - 1`: the
    `k` shards cover it exactly once, in order, their sizes within one seed
    of each other."""
    return range(start + i * n // k, start + (i + 1) * n // k)


def shrunk(seed: int, cap: float) -> str:
    """`fuzz.shrink`'s report for one seed, or why there is none."""
    try:
        p = subprocess.run(  # noqa: S603 — our own module, fixed argv
            [sys.executable, "-m", "fuzz.shrink", str(seed)],
            capture_output=True,
            text=True,
            timeout=cap,
            cwd=Path(__file__).parents[1],
        )
    except subprocess.TimeoutExpired:
        return f"(shrinking seed {seed} exceeded {cap:.0f}s)"
    if p.returncode != 0:
        return f"(shrink exited {p.returncode})\n{p.stderr[-1500:]}"
    return p.stdout.strip()


def _mmss(seconds: float) -> str:
    return f"{int(seconds // 60)}m{int(seconds % 60):02d}s"


def run(seeds: range, workers: int, timeout: float, shrink_cap: float, out: Path):
    """One shard's two checks, and one shrunk representative per gated
    class: the summary `render` reads."""
    # Here and not at the top: `--merge` needs neither, so it runs without
    # the engine built.
    from fuzz import metamorphic, runner

    t0 = time.monotonic()
    results = runner.campaign(
        seeds.start, len(seeds), workers, timeout, out / "findings.jsonl"
    )
    runner.write_cases(results, out / "cases.jsonl")
    t_campaign = time.monotonic() - t0

    m_counts, m_findings = metamorphic.run(seeds, workers)

    # One representative per gated (kind, class), smallest seed first.
    reps: dict[tuple, dict] = {}
    cases: collections.Counter = collections.Counter()
    for r in sorted(results, key=lambda r: r["seed"]):
        if r["kind"] in runner.GATED:
            key = (r["kind"], r.get("klass", ""))
            reps.setdefault(key, r)
            cases[key] += 1
    examples: dict[str, str] = {}
    for f in m_findings:
        examples.setdefault(f.rewrite, f.line())  # one example per rewrite
    return {
        "seeds": [seeds.start, seeds.stop - 1],
        "workers": workers,
        "kinds": dict(collections.Counter(r["kind"] for r in results)),
        "gated": [
            {
                "kind": kind,
                "klass": klass,
                "cases": cases[(kind, klass)],
                "seed": r["seed"],
                "shrunk": shrunk(r["seed"], shrink_cap),
            }
            for (kind, klass), r in reps.items()
        ],
        "metamorphic": {
            "checked": sum(v for (_, k), v in m_counts.items() if k == "checked"),
            "findings": len(m_findings),
            "examples": examples,
        },
        "t_campaign": t_campaign,
        "t_run": time.monotonic() - t0,
    }


def night_of(summaries: list[dict], k: int) -> tuple[list[dict], list[str]]:
    """The summaries that are shards of ONE run, and why each other one was
    left out. The run is `k` shards of the window most of them share. A
    summary of another window or shard count (a stale artifact, a shard run
    by hand) or a second summary for a shard already counted is not summed:
    its seeds are not the run's, and counting them would report seeds the
    run never checked."""
    windows = collections.Counter(
        tuple(s["window"]) for s in summaries if s["shard"][1] == k
    )
    night = windows.most_common(1)[0][0] if windows else None
    want = f"window {night[0]}+{night[1]} in {k} shards" if night else f"{k} shards"
    kept: list[dict] = []
    left_out: list[str] = []
    for s in sorted(summaries, key=lambda s: s["shard"][0]):
        (i, sk), (lo, hi) = s["shard"], s["seeds"]
        name = f"shard {i}/{sk} (seeds {lo}..{hi})"
        if sk != k or tuple(s["window"]) != night:
            got = f"window {s['window'][0]}+{s['window'][1]} in {sk} shards"
            left_out.append(f"{name} is from {got}, not {want}")
        elif any(x["shard"][0] == i for x in kept):
            left_out.append(f"{name} is a second summary for shard {i}")
        else:
            kept.append(s)
    return kept, left_out


def failed(summaries: list[dict], missing, left_out=()) -> bool:
    return bool(
        missing
        or left_out
        or any(s["gated"] or s["metamorphic"]["findings"] for s in summaries)
    )


def render(summaries: list[dict], k: int = 1, missing=(), left_out=()) -> str:
    """The report for a night's shard summaries, however many: verdicts
    summed, one shrunk representative per gated class (from the shard with
    the smallest seeds, so the smallest seed of the class), one metamorphic
    example per rewrite, each shard in `missing` by name, and each summary
    `night_of` left out, with why."""
    summaries = sorted(summaries, key=lambda s: s["seeds"][0])
    kinds: collections.Counter = collections.Counter()
    gated: dict[tuple, dict] = {}
    examples: dict[str, str] = {}
    checked = found = 0
    for s in summaries:
        kinds.update(s["kinds"])
        for g in s["gated"]:
            key = (g["kind"], g["klass"])
            if key in gated:
                gated[key]["cases"] += g["cases"]
            else:
                gated[key] = dict(g)
        checked += s["metamorphic"]["checked"]
        found += s["metamorphic"]["findings"]
        for rewrite, line in s["metamorphic"]["examples"].items():
            examples.setdefault(rewrite, line)

    title = "# Nightly campaign"
    if summaries:
        title += f", seeds {summaries[0]['seeds'][0]}..{summaries[-1]['seeds'][1]}"
    if k > 1:
        title += f" ({len(summaries)} of {k} shards)"
    status = "FAIL" if failed(summaries, missing, left_out) else "pass"
    lines = [
        title,
        "",
        f"**{status}**: {len(gated)} gated class(es), {found} metamorphic finding(s).",
    ]
    if summaries:
        took = (
            f"campaign {_mmss(max(s['t_campaign'] for s in summaries))}, "
            f"whole run {_mmss(max(s['t_run'] for s in summaries))}, "
            f"on {summaries[0]['workers']} workers"
        )
        many = len(summaries) > 1
        lines[-1] += f" Slowest shard: {took}." if many else f" {took.capitalize()}."
    lines.append("")
    window_of = next((s["window"] for s in summaries if "window" in s), None)
    for i in missing:
        seeds = ""
        if window_of is not None:
            r = shard(*window_of, i, k)
            seeds = f" (seeds {r.start}..{r.stop - 1})"
        lines += [
            f"**Missing:** shard {i}/{k}{seeds} left no summary, so its seeds "
            "went unchecked; see the run.",
            "",
        ]
    for why in left_out:
        lines += [f"**Left out:** {why}; none of its counts are in this report.", ""]
    lines += [
        "| verdict | cases |",
        "|---|---|",
        *(f"| {v} | {c} |" for v, c in sorted(kinds.items(), key=lambda kv: -kv[1])),
        "",
    ]
    if gated:
        lines += ["## Gated findings, one shrunk representative per class", ""]
        for g in gated.values():
            lines += [
                f"### {g['kind']}: {g['klass'] or '(no class)'} ({g['cases']} case(s))",
                "",
                "```",
                g["shrunk"],
                "```",
                "",
            ]
    if examples:
        lines += ["## Metamorphic findings", "", "```", *examples.values(), "```", ""]
    lines.append(f"Metamorphic pairs checked: {checked}.")
    lines.append(
        "Replay a seed: `uv run python -m fuzz.shrink <seed>` in packages/confit."
    )
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--n", type=int, default=N, help="seeds to run, all shards together"
    )
    ap.add_argument("--seed", type=int, help="window start (default: tonight's)")
    ap.add_argument(
        "--shard", default="0/1", metavar="I/K", help="run shard I (0-based) of K"
    )
    ap.add_argument(
        "--merge",
        nargs="+",
        type=Path,
        metavar="DIR",
        help="render these shards' summaries as one report instead of running",
    )
    ap.add_argument("--shards", type=int, help="with --merge: the night's shard count")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--timeout", type=float, default=20.0)
    ap.add_argument("--shrink-cap", type=float, default=300.0)
    ap.add_argument("--out", type=Path, default=Path("nightly"))
    a = ap.parse_args(argv)
    i, k = (int(x) for x in a.shard.split("/"))
    if not 0 <= i < k:
        ap.error(f"--shard {a.shard}: need 0 <= I < K")
    if a.seed is None and a.n > N:
        ap.error(
            f"--n {a.n} is more than a night's {N} seeds and would reach into "
            "the next night's window; pass --seed for a window of its own"
        )
    a.out.mkdir(parents=True, exist_ok=True)

    if a.merge:
        paths = [d / "summary.json" for d in a.merge]
        summaries = [
            json.loads(p.read_text(encoding="utf-8")) for p in paths if p.is_file()
        ]
        k = a.shards or max((s["shard"][1] for s in summaries), default=1)
        summaries, left_out = night_of(summaries, k)
        missing = sorted(set(range(k)) - {s["shard"][0] for s in summaries})
    else:
        start = a.seed if a.seed is not None else window()
        summary = run(
            shard(start, a.n, i, k), a.workers, a.timeout, a.shrink_cap, a.out
        )
        summary.update(shard=[i, k], window=[start, a.n])
        (a.out / "summary.json").write_text(
            json.dumps(summary, indent=1), encoding="utf-8"
        )
        summaries, missing, left_out = [summary], [], []

    report = render(summaries, k, missing, left_out)
    (a.out / "report.md").write_text(report, encoding="utf-8")
    print(report)
    return 1 if failed(summaries, missing, left_out) else 0


if __name__ == "__main__":
    sys.exit(main())
