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
import re
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

import duckdb
from confit.oracle import Oracle

from . import coverage
from . import gen as G
from .oracle import PHASE_MARK, UNSHIPPED_FEATURES, case_inputs, unshipped_reach

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


def _stderr_all(err_file) -> str:
    try:
        err_file.seek(0)
        return err_file.read().decode(errors="replace")
    except Exception:  # noqa: BLE001
        return ""


def side_of(stderr: str) -> str:
    """Which side a dead worker was in: the last phase marker it wrote
    (`fuzz.oracle.PHASE_MARK`) names DuckDB (`oracle`), us (`confit:*`), or
    the harness itself (`harness:*`: startup, generation). `unknown` when it
    wrote none."""
    last = None
    for ln in stderr.splitlines():
        if ln.startswith(PHASE_MARK):
            last = ln.split()[1] if len(ln.split()) > 1 else None
    if last is None:
        return "unknown"
    if last.startswith("harness"):
        return "harness"
    return "oracle" if last == "oracle" else "confit"


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
        stderr = _stderr_all(err)
        results.append(blame(seed, kind, stderr[-800:], stderr))
        proc.kill()
        err.close()
        proc, err = _spawn()
    proc.stdin.close()
    proc.wait(timeout=10)
    err.close()


def blame(seed: int, kind: str, detail: str, stderr: str | None = None) -> dict:
    """The finding for a worker that died or hung on `seed`. It returned
    nothing, so the case is regenerated here: generation is deterministic and
    cheap, and a finding with a bare seed is lost at the next generator change.
    The side it died in comes from its last phase marker in `stderr` (the
    whole stream; `detail` is only its tail) and is part of the class:
    oracle-side and confit-side timeouts mean opposite things.
    """
    side = side_of(detail if stderr is None else stderr)
    try:
        case = G.gen(seed)
        sql, inputs = G.render(case.query), case_inputs(case)
        trip = sorted(coverage.key(t) for t in coverage.triples(case.query))
        tags = case.tags + [f"reaches:{f}" for f in sorted(unshipped_reach(sql))]
    except Exception as e:  # noqa: BLE001 — the blame must still be recorded
        sql, inputs, tags = "", {"error": f"{type(e).__name__}: {e}"}, []
        trip = []
    return {
        "seed": seed,
        "kind": kind,
        "klass": f"{kind.lower()}:{side}",
        "side": side,
        "detail": detail,
        "sql": sql,
        "tags": list(tags),
        "inputs": inputs,
        "triples": trip,
    }


# Refusal QUALITY, not prefix presence: the documented prefix is the weakest
# property. A message that echoes source or AST text instead of saying which
# construct is refused does not name it, and one with no remedy is not
# actionable. Heuristics over message text, for reporting only; they back no
# gate and no KPI.
_PREFIXES = ("unsupported:", "parse error:", "bind error:")
_ECHO = re.compile(
    r"^(?:\w[\w ]*: )?(?:FROM \(SELECT |expression: )"  # echoed SQL
    r"|\w\(\w+\(|\{ \w+: "  # a Rust Debug dump of the AST
)
_REMEDY = re.compile(
    r"—|--|\((?:qualify|use|call|cast)\b|\b(?:instead|declare|spell|"
    r"must be|make an?|use|call|project|qualify)\b"
)


def refusal_quality(msg: str) -> dict:
    """`prefixed` (a documented prefix), `named` (says which construct rather
    than echoing text) and `actionable` (says what to do) for one refusal,
    plus `echo`: the message up to where its echo starts, or None."""
    body = msg
    for p in _PREFIXES:
        if msg.startswith(p):
            body = msg[len(p) :].lstrip()
            break
    m = _ECHO.search(body)
    return {
        "prefixed": msg.startswith(_PREFIXES),
        "named": m is None,
        "actionable": bool(_REMEDY.search(body)),
        "echo": None if m is None else msg[: len(msg) - len(body) + m.end()] + "…",
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
    engine and generator produced it, and against which reference. This is
    what makes one run comparable to the next."""
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

    # Abstentions: cases with no verdict, as rates over the population, the
    # worker failures split by the side they died in. An AGREE whose ORDER BY
    # went unevaluated is a partial abstention (sortedness not checked), so it
    # is rated here too. UNSHIPPED is not an abstention; it has its own section.
    n = max(len(results), 1)
    print(f"\n== abstentions (rate over {len(results)} cases) ==")
    for kind in ("SKIP", "TIMEOUT", "PANIC"):
        hit = [r for r in results if r["kind"] == kind]
        sides = collections.Counter(r.get("side", "unknown") for r in hit)
        by_side = "  ".join(f"{k} {c}" for k, c in sorted(sides.items()))
        line = f"  {kind:22} {len(hit):6}  {100 * len(hit) / n:5.1f}%"
        print(line + (f"  {by_side}" if kind != "SKIP" and hit else ""))
    weak = sum(
        1
        for r in results
        if r["kind"] == "AGREE" and "order-by-unevaluated" in r["tags"]
    )
    print(f"  {'order-by-unevaluated':22} {weak:6}  {100 * weak / n:5.1f}%")

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

    # Quality over every refusal, then the echoing classes by name: those are
    # the refusal sites to give a construct name.
    quality = [refusal_quality(r["detail"]) for r in refused]
    print("\n== refusal quality ==")
    print(f"  {'refusals':20} {len(refused):6}")
    for key, label in (
        ("prefixed", "documented prefix"),
        ("named", "names the construct"),
        ("actionable", "actionable"),
    ):
        print(f"  {label:20} {sum(q[key] for q in quality):6}")
    echoes = collections.Counter(q["echo"] for q in quality if q["echo"])
    if echoes:
        print("  echoes source text instead of naming:")
        for k, c in echoes.most_common(10):
            print(f"    {c:6}  {k}")

    cover = collections.Counter(
        t for r in results if r["kind"] in COVERED for t in r["tags"]
    )
    print("\n== AGREE coverage by construct ==")
    for k, c in cover.most_common():
        print(f"  {c:6}  {k}")

    # (operator, argument-type, edge-class) triples: reached by any case,
    # agreed by an AGREE. Distinct triples, not query counts, so a thousand
    # cases of `abs(ordinary int)` count once. Operators sorted by the gap.
    reached: dict[str, set] = collections.defaultdict(set)
    agreed: dict[str, set] = collections.defaultdict(set)
    for r in results:
        for k in r.get("triples", ()):
            op = coverage.parse(k)[0]
            reached[op].add(k)
            if r["kind"] == "AGREE":
                agreed[op].add(k)
    n_reached = sum(len(v) for v in reached.values())
    n_agreed = sum(len(v) for v in agreed.values())
    print("\n== coverage triples (reached: any verdict, agreed: AGREE) ==")
    print(f"  {'distinct triples':18} reached {n_reached:6}  agreed {n_agreed:6}")
    for op in sorted(reached, key=lambda o: (len(agreed[o]) - len(reached[o]), o)):
        print(f"    {op:24} reached {len(reached[op]):4}  agreed {len(agreed[op]):4}")

    # Cases whose answer has a width we have not shipped: classified, never
    # value-compared, never counted as agreement. An empty bucket proves
    # nothing on its own, so each feature first says how many cases REACHED
    # its construct and what they got: reached and never UNSHIPPED means the
    # feature shipped (delete its arm in fuzz/oracle.py); not reached means
    # the grammar stopped emitting it.
    uns = [r for r in results if r["kind"] == "UNSHIPPED"]
    print("\n== unshipped features (classified, not compared) ==")
    for feat in UNSHIPPED_FEATURES:
        hit = [r for r in results if f"reaches:{feat}" in r["tags"]]
        if not hit:
            print(
                f"  {feat:10} not reached by the generator: an empty bucket "
                "here is not evidence of support"
            )
            continue
        got = collections.Counter(r["kind"] for r in hit).most_common()
        print(
            f"  {feat:10} reached {len(hit):6}  "
            + "  ".join(f"{k} {c:6}" for k, c in got)
        )
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
