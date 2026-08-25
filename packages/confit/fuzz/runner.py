"""Campaign runner: crash-isolated workers, timeouts, findings.jsonl, stats.

    uv run --directory packages/confit python -m fuzz.runner \
        --seed 0 --n 20000 --workers 8 --timeout 20 --out findings.jsonl

Each worker is a subprocess reading seeds line-by-line; a dead or hung worker
is killed, blamed for its in-flight seed (PANIC/TIMEOUT finding, stderr tail
attached), and replaced. Verdict counts, refusal classes, and a construct-
coverage histogram over AGREE cases print at the end — a grammar hole should
be visible, not silent.
"""

from __future__ import annotations

import argparse
import collections
import json
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

INTERESTING = (
    "DIVERGE_VALUE",
    "DIVERGE_BUILD",
    "DIVERGE_TRAP",
    # we match eager DuckDB, its optimizer makes the user's answer differ
    "DIVERGE_OPT",
    # we match the OPTIMIZER against the oracle: an emulation, and a bug
    "OPT_EMULATED",
    "BUILD_EXC",
    "PANIC",
    "TIMEOUT",
    "SKIP",
)

# Agreement, for the coverage histogram. OPT_EMULATED is NOT agreement: since
# the oracle became optimizer-off DuckDB it means we answer unlike the oracle,
# so it is a finding, and counting it as coverage would hide it twice.
COVERED = ("AGREE",)


def _spawn():
    """A worker subprocess and the temp file holding its stderr, as
    `(proc, err)`. The caller owns `err` and must close it."""
    err = tempfile.TemporaryFile()
    proc = subprocess.Popen(  # noqa: S603 — our own module, fixed argv
        [sys.executable, "-m", "fuzz.worker"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=err,
        cwd=Path(__file__).parents[1],
        text=True,
    )
    return proc, err


def _stderr_tail(err_file) -> str:
    try:
        err_file.seek(0)
        return err_file.read().decode(errors="replace")[-800:]
    except Exception:  # noqa: BLE001
        return ""


def _drive(seeds, results, timeout, lock):
    """One worker thread: seeds off the shared iterator (`lock` guards it)
    into a subprocess, verdict dicts onto `results`.

    `timeout` is per seed, in seconds. A worker that dies or outruns it is
    killed, blamed for the seed it was holding, and replaced.
    """
    proc, err = _spawn()
    while True:
        with lock:
            try:
                seed = next(seeds)
            except StopIteration:
                break
        try:
            proc.stdin.write(f"{seed}\n")
            proc.stdin.flush()
        except OSError:
            proc, err = _spawn()
            proc.stdin.write(f"{seed}\n")
            proc.stdin.flush()
        fired = threading.Event()

        def _kill(p=proc, f=fired):
            f.set()
            p.kill()

        timer = threading.Timer(timeout, _kill)
        timer.start()
        line = proc.stdout.readline()
        timer.cancel()
        if line:
            results.append(json.loads(line))
            continue
        kind = "TIMEOUT" if fired.is_set() else "PANIC"
        results.append(
            {
                "seed": seed,
                "kind": kind,
                "klass": kind.lower(),
                "detail": _stderr_tail(err),
                "sql": "",
                "tags": [],
            }
        )
        proc.kill()
        err.close()
        proc, err = _spawn()
    proc.stdin.close()
    proc.wait(timeout=10)
    err.close()


def campaign(start: int, n: int, workers: int, timeout: float, out: Path):
    """Seeds `start .. start + n - 1` across `workers` subprocesses: reports,
    writes the findings to `out`, and returns every verdict dict."""
    seeds = iter(range(start, start + n))
    results: list[dict] = []
    lock = threading.Lock()
    threads = [
        threading.Thread(target=_drive, args=(seeds, results, timeout, lock))
        for _ in range(workers)
    ]
    for t in threads:
        t.start()
    done = 0
    while any(t.is_alive() for t in threads):
        for t in threads:
            t.join(timeout=5)
        if len(results) - done >= 500:
            done = len(results)
            print(f"... {done}/{n}", file=sys.stderr)
    report(results, out)
    return results


def report(results: list[dict], out: Path):
    """Print the campaign summary and write every INTERESTING verdict to
    `out`, one JSON object per line. The file keeps the raw findings; only
    the printout collapses them to one example per (kind, klass)."""
    kinds = collections.Counter(r["kind"] for r in results)
    print("\n== verdicts ==")
    for k, c in kinds.most_common():
        print(f"  {k:14} {c}")

    refusals = collections.Counter(
        r["klass"] for r in results if r["kind"] == "REFUSED"
    )
    print("\n== top refusal classes ==")
    for k, c in refusals.most_common(15):
        print(f"  {c:6}  {k}")

    cover = collections.Counter(
        t for r in results if r["kind"] in COVERED for t in r["tags"]
    )
    print("\n== AGREE coverage by construct ==")
    for k, c in cover.most_common():
        print(f"  {c:6}  {k}")

    # The passes we reproduce on purpose, by the eager-baseline disagreement
    # they resolve. Empty here means no emulation was exercised at all, which
    # is a coverage hole rather than good news.
    emul = [r for r in results if r["kind"] == "OPT_EMULATED"]
    if emul:
        # Since the oracle became optimizer-off DuckDB these are BUGS, not
        # notes: an emulation means we answer like the optimizer and unlike the
        # oracle. Print a seed with each so it is reproducible, the way the
        # findings section does.
        print("\n== optimizer passes we still reproduce (each one is a bug) ==")
        seen: dict[str, dict] = {}
        for r in emul:
            seen.setdefault(r["klass"], r)
        for k, r in seen.items():
            n = sum(1 for x in emul if x["klass"] == k)
            print(f"  {n:6}  {k}   e.g. seed {r['seed']}: {r['sql'][:80]}")

    findings = [r for r in results if r["kind"] in INTERESTING]
    dedup: dict[tuple, dict] = {}
    for r in findings:
        dedup.setdefault((r["kind"], r["klass"]), r)
    with out.open("w", encoding="utf-8") as f:
        for r in findings:
            f.write(json.dumps(r) + "\n")
    print(f"\n== findings: {len(findings)} raw, {len(dedup)} classes -> {out} ==")
    for (kind, klass), r in sorted(dedup.items()):
        n = sum(1 for x in findings if (x["kind"], x["klass"]) == (kind, klass))
        print(f"  {n:6}  {kind:14} {klass}   e.g. seed {r['seed']}: {r['sql'][:90]}")


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--timeout", type=float, default=30.0)
    ap.add_argument("--out", type=Path, default=Path("findings.jsonl"))
    a = ap.parse_args()
    campaign(a.seed, a.n, a.workers, a.timeout, a.out)


if __name__ == "__main__":
    main()
