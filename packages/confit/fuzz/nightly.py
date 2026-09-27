"""The nightly deep campaign: fresh seeds each night, findings shrunk, one report.

    uv run python -m fuzz.nightly --n 100000 --out nightly/

Each night takes the next window of seeds (`--seed` overrides), so seeds that
earlier nights have not reached get covered. Each window gets two checks:
- the strict oracle campaign (`fuzz.runner`, the acceptance gate);
- the metamorphic spelling suite (`fuzz.metamorphic`).

For each finding class, one representative seed is shrunk by `fuzz.shrink`.
That runs in a subprocess with a time cap, because a finding can be a panic
or a hang. Everything lands in `<out>/report.md`, which is written to be
filed as an issue. The exit status is 1 when anything gated.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import subprocess
import sys
import time
from pathlib import Path

from fuzz import metamorphic, runner

# Night 0. The window base sits far above the seeds hand campaigns use
# (0..~60k), so the nightly never re-covers them.
EPOCH = dt.date(2026, 9, 27)
BASE = 1_000_000


def window(n: int, today: dt.date | None = None) -> int:
    """The first seed of tonight's window of `n` seeds."""
    return BASE + ((today or dt.date.today()) - EPOCH).days * n


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


def main(argv=None) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=100_000)
    ap.add_argument("--seed", type=int, help="window start (default: tonight's)")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--timeout", type=float, default=20.0)
    ap.add_argument("--shrink-cap", type=float, default=300.0)
    ap.add_argument("--out", type=Path, default=Path("nightly"))
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    seed = a.seed if a.seed is not None else window(a.n)
    t0 = time.monotonic()

    results = runner.campaign(seed, a.n, a.workers, a.timeout, a.out / "findings.jsonl")
    runner.write_cases(results, a.out / "cases.jsonl")
    fails = runner.gate(results, None)
    t_campaign = time.monotonic() - t0

    m_counts, m_findings = metamorphic.run(range(seed, seed + a.n), a.workers)

    # One representative per gated (kind, class), smallest seed first.
    reps: dict[tuple, dict] = {}
    for r in sorted(results, key=lambda r: r["seed"]):
        if r["kind"] in runner.GATED:
            reps.setdefault((r["kind"], r.get("klass", "")), r)

    kinds: dict[str, int] = {}
    for r in results:
        kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
    lines = [
        f"# Nightly campaign, seeds {seed}..{seed + a.n - 1}",
        "",
        f"**{'FAIL' if fails or m_findings else 'pass'}**: "
        f"{len(reps)} gated class(es), {len(m_findings)} metamorphic finding(s). "
        f"Campaign {_mmss(t_campaign)}, whole run {_mmss(time.monotonic() - t0)}, "
        f"on {a.workers} workers.",
        "",
        "| verdict | cases |",
        "|---|---|",
        *(f"| {k} | {v} |" for k, v in sorted(kinds.items(), key=lambda kv: -kv[1])),
        "",
    ]
    if reps:
        lines += ["## Gated findings, one shrunk representative per class", ""]
        for (kind, klass), r in reps.items():
            n = sum(
                1 for x in results if (x["kind"], x.get("klass", "")) == (kind, klass)
            )
            lines += [
                f"### {kind}: {klass or '(no class)'} ({n} case(s))",
                "",
                "```",
                shrunk(r["seed"], a.shrink_cap),
                "```",
                "",
            ]
    if m_findings:
        lines += ["## Metamorphic findings", "", "```"]
        seen: set[str] = set()
        for f in m_findings:
            if f.rewrite not in seen:  # one example per rewrite
                seen.add(f.rewrite)
                lines.append(f.line())
        lines += ["```", ""]
    checked = sum(v for (_, k), v in m_counts.items() if k == "checked")
    lines.append(f"Metamorphic pairs checked: {checked}.")
    lines.append(
        "Replay a seed: `uv run python -m fuzz.shrink <seed>` in packages/confit."
    )
    report = "\n".join(lines) + "\n"
    (a.out / "report.md").write_text(report, encoding="utf-8")
    print(report)
    return 1 if fails or m_findings else 0


if __name__ == "__main__":
    sys.exit(main())
