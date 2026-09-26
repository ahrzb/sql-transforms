"""Campaign runner: crash-isolated workers, timeouts, findings.jsonl, stats.

    uv run --directory packages/confit python -m fuzz.runner \
        --seed 0 --n 20000 --workers 8 --timeout 20 --out findings.jsonl

Each worker is a subprocess reading seeds line-by-line; a dead or hung worker
is killed, blamed for its in-flight seed (PANIC/TIMEOUT finding, stderr tail
attached), and replaced. Verdict counts, refusal classes, the unshipped-
feature bucket, and a construct-coverage histogram over AGREE cases print at
the end — a grammar hole should be visible, not silent.
"""

from __future__ import annotations

import argparse
import collections
import datetime
import hashlib
import json
import platform
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

import duckdb
from confit.oracle import Oracle

from . import gen as G
from .oracle import case_inputs

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
# UNSHIPPED is not agreement either — nothing was compared — and it is not a
# finding, so it gets its own section below rather than a place in either.
COVERED = ("AGREE",)

# Every verdict kind in exactly one reporting category. `unresolved` is the
# visible, separately counted bucket for cases with no verdict at all: the
# harness or a worker failed, so nothing was compared. It is not agreement,
# and not a confirmed defect either; a mismatch is never moved into it.
CATEGORY = {
    "AGREE": "agreement",
    "AGREE_TRAP": "agreement",
    "DIVERGE_VALUE": "mismatch",
    "DIVERGE_BUILD": "mismatch",
    "DIVERGE_TRAP": "mismatch",
    "DIVERGE_OPT": "mismatch",
    "OPT_EMULATED": "mismatch",
    "BUILD_EXC": "mismatch",
    "SKIP": "unresolved",
    "TIMEOUT": "unresolved",
    "PANIC": "unresolved",
    "REFUSED": "refused",
    "UNSHIPPED": "unshipped",
}
_CATEGORY_NOTE = {
    "unresolved": "no verdict: neither agreement nor a confirmed defect",
}


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
        results.append(blame(seed, kind, _stderr_tail(err)))
        proc.kill()
        err.close()
        proc, err = _spawn()
    proc.stdin.close()
    proc.wait(timeout=10)
    err.close()


def blame(seed: int, kind: str, detail: str) -> dict:
    """The finding for a worker that died or hung on `seed`. It returned
    nothing, so the case is regenerated here: generation is deterministic and
    cheap, and a finding with a bare seed is lost at the next generator change.
    """
    try:
        case = G.gen(seed)
        sql, inputs, tags = G.render(case.query), case_inputs(case), case.tags
    except Exception as e:  # noqa: BLE001 — the blame must still be recorded
        sql, inputs, tags = "", {"error": f"{type(e).__name__}: {e}"}, []
    return {
        "seed": seed,
        "kind": kind,
        "klass": kind.lower(),
        "detail": detail,
        "sql": sql,
        "tags": list(tags),
        "inputs": inputs,
    }


def _git(*args: str) -> str:
    try:
        return subprocess.run(  # noqa: S603 — fixed argv
            ["git", *args],  # noqa: S607
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def provenance(start: int, n: int) -> dict:
    """What a dated campaign result must say about itself: when it ran, which
    engine and generator produced it, and against which reference. Old runs
    stay history; this is what makes a new one comparable to the next."""
    gen_src = Path(G.__file__).read_bytes()
    return {
        "date": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
        "engine_revision": _git("rev-parse", "HEAD") or "unknown",
        "engine_dirty": bool(_git("status", "--porcelain", "--", "..")),
        "generator_revision": "sha256:" + hashlib.sha256(gen_src).hexdigest()[:16],
        "reference": {
            "duckdb": duckdb.__version__,
            "oracle_version": Oracle.VERSION,
            "baseline": "optimizer-off (PRAGMA disable_optimizer), native tables",
            "bracket": "optimizer-on, same connection",
        },
        "seeds": [start, start + n - 1],
        "platform": f"{platform.system()} {platform.machine()}, "
        f"python {platform.python_version()}",
    }


def campaign(start: int, n: int, workers: int, timeout: float, out: Path):
    """Seeds `start .. start + n - 1` across `workers` subprocesses: reports,
    writes the findings to `out`, and returns every verdict dict."""
    prov = provenance(start, n)  # stamped at the start: the run's date
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
    report(results, out, provenance=prov)
    return results


def report(results: list[dict], out: Path, provenance: dict | None = None):
    """Print the campaign summary and write every INTERESTING verdict to
    `out`, one JSON object per line, after a `{"provenance": ...}` header
    line when one is given. The file keeps the raw findings; only the
    printout collapses them to one example per (kind, klass)."""
    kinds = collections.Counter(r["kind"] for r in results)
    print("\n== verdicts ==")
    for k, c in kinds.most_common():
        print(f"  {k:14} {c}")

    # The same verdicts by category, over an explicit population. An AGREE
    # whose ORDER BY could not be evaluated is agreement on the multiset only,
    # so it is counted inside agreement and named, never silently.
    cats = collections.Counter(CATEGORY.get(r["kind"], "unresolved") for r in results)
    print(f"\n== outcomes (population: {len(results)} cases) ==")
    for cat in ("agreement", "mismatch", "unresolved", "refused", "unshipped"):
        note = _CATEGORY_NOTE.get(cat, "")
        print(f"  {cat:11} {cats[cat]:6}" + (f"  {note}" if note else ""))
        if cat == "agreement":
            weak = sum(
                1
                for r in results
                if r["kind"] == "AGREE" and "order-by-unevaluated" in r["tags"]
            )
            if weak:
                print(
                    f"  {'':11} {weak:6}  of them order-by-unevaluated: "
                    "sortedness not established"
                )

    # Refusals keep the baseline's outcome for the same query. Grouped by it,
    # "DuckDB serves, we refuse" is the cost side of each refusal class; it
    # is reporting, not a finding, so nothing here reaches `out`.
    refused = [r for r in results if r["kind"] == "REFUSED"]
    by_outcome = collections.Counter(r.get("oracle", "unknown") for r in refused)
    print("\n== refusals by oracle outcome ==")
    for outcome, n in by_outcome.most_common():
        print(f"  {outcome:14} {n}")
        classes = collections.Counter(
            r["klass"] for r in refused if r.get("oracle", "unknown") == outcome
        )
        for k, c in classes.most_common(15):
            print(f"    {c:6}  {k}")

    cover = collections.Counter(
        t for r in results if r["kind"] in COVERED for t in r["tags"]
    )
    print("\n== AGREE coverage by construct ==")
    for k, c in cover.most_common():
        print(f"  {c:6}  {k}")

    # Cases whose answer has a width we have not shipped: classified, never
    # value-compared, never counted as agreement. An EMPTY section means
    # either the feature shipped — delete its arm in fuzz/oracle.py — or the
    # grammar stopped reaching it, and both are worth seeing.
    uns = [r for r in results if r["kind"] == "UNSHIPPED"]
    print("\n== unshipped features (classified, not compared) ==")
    for k, c in collections.Counter(r["klass"] for r in uns).most_common():
        ex = next(x for x in uns if x["klass"] == k)
        print(f"  {c:6}  {k}   e.g. seed {ex['seed']}: {ex['detail'][:60]}")

    # Optimizer passes we reproduce, by the eager-baseline disagreement they
    # resolve. Each one is a BUG, not coverage: an emulation means we answer
    # like the optimizer and unlike the oracle, and the class should be
    # empty. Print a seed with each so it is reproducible, the way the
    # findings section does.
    emul = [r for r in results if r["kind"] == "OPT_EMULATED"]
    if emul:
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
        if provenance is not None:
            f.write(json.dumps({"provenance": provenance}) + "\n")
        for r in findings:
            cat = CATEGORY.get(r["kind"], "unresolved")
            f.write(json.dumps({**r, "category": cat}) + "\n")
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
