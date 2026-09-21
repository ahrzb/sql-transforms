"""Campaign runner: crash-isolated workers, timeouts, artifacts, stats.

    uv run --directory packages/confit python -m fuzz.runner \
        --seed 0 --n 20000 --workers 8 --timeout 20

Each worker is a subprocess reading seeds line-by-line. It announces the case
it is about to run (fuzz/worker.py's two-line protocol) and the parent writes
that announcement to disk BEFORE waiting for the verdict, so a dead or hung
worker is killed, blamed for its in-flight seed (PANIC/TIMEOUT finding,
stderr tail attached), replaced — and the inputs it died on are still on
disk. Verdict counts, refusal classes with the oracle's own outcome, the
unshipped-feature bucket, and a construct-coverage histogram over AGREE cases
print at the end — a grammar hole should be visible, not silent.

# FOUR artifacts, because a findings file is not evidence of a campaign

`--out findings.jsonl` (default: a fresh timestamped name) names a set:

    findings.jsonl              the INTERESTING verdicts, unchanged
    findings.results.jsonl      EVERY verdict — passes, refusals, unshipped
    findings.cases.jsonl        the prepared case per seed: SQL, schemas,
                                rows, statics, UDF/tree specs
    findings.provenance.json    who ran what, when, against which reference

The three sidecars exist because the old single file discarded everything
that was not a finding, which left a campaign's own claim — "N cases, this
many agreed" — unreproducible and unauditable the moment the generator moved.
Seeds are not durable identities across generator changes (oracle policy,
2026-09-21: "Baselines"), so the case itself is recorded. None of this is a
replay framework: it is the inputs, written down, in a form a reader can read.

Existing artifacts are never overwritten. A campaign whose workers, protocol
or result count broke fails loudly and prints no summary: losing cases
silently and then reporting a clean run is the failure mode these artifacts
exist to rule out.
"""

from __future__ import annotations

import argparse
import collections
import datetime
import hashlib
import json
import math
import platform
import subprocess
import sys
import tempfile
import threading
from dataclasses import dataclass
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
# UNSHIPPED is not agreement either — nothing was compared — and it is not a
# finding, so it gets its own section below rather than a place in either.
COVERED = ("AGREE",)

PKG = Path(__file__).parents[1]  # packages/confit


class CampaignError(RuntimeError):
    """The campaign lost cases, or never had a reference to compare against.
    Raised instead of returning results a summary would misrepresent."""


@dataclass(frozen=True)
class Artifacts:
    """The four files one campaign writes. `findings` is the path the caller
    asked for; the rest are derived from its stem so a run's evidence stays
    together under one name."""

    findings: Path
    results: Path
    cases: Path
    provenance: Path

    def all(self) -> tuple[Path, ...]:
        return (self.findings, self.results, self.cases, self.provenance)


def artifacts_for(out: Path) -> Artifacts:
    stem = out.parent / out.name.removesuffix(".jsonl")
    return Artifacts(
        out,
        Path(f"{stem}.results.jsonl"),
        Path(f"{stem}.cases.jsonl"),
        Path(f"{stem}.provenance.json"),
    )


def _stamp() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")


def default_out() -> Path:
    """A fresh dated name, so back-to-back campaigns cannot land on each
    other and the default is dated the way stored evidence has to be."""
    return Path(f"findings-{_stamp()}.jsonl")


def _reserve(a: Artifacts) -> None:
    """Refuse the whole set if any member exists. Prior findings are the only
    record of a run that may have taken hours; clobbering them silently is
    not a thing a default should be able to do."""
    clash = [str(p) for p in a.all() if p.exists()]
    if clash:
        raise CampaignError(
            "refusing to overwrite existing campaign artifacts: "
            + ", ".join(clash)
            + " — pass --out with a path that does not exist yet"
        )
    a.findings.parent.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------- provenance


def _sha256(path: Path) -> str | None:
    try:
        with path.open("rb") as source:
            return hashlib.file_digest(source, "sha256").hexdigest()
    except OSError:
        return None


def _git(*args: str) -> str | None:
    """A git reading, or None when git cannot answer. None is recorded as
    "unknown" rather than smoothed into a clean-tree claim."""
    try:
        out = subprocess.run(  # noqa: S603 — fixed argv, our own repo
            ["git", *args],  # noqa: S607
            cwd=PKG,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout if out.returncode == 0 else None


def _engine_provenance() -> dict:
    """The revision of the tree, the dirty state of the tree, and the hash of
    the compiled extension actually imported — which is NOT implied by the
    revision: the binary is whatever the last `maturin develop` built."""
    head = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain")
    dirty = (
        sorted(line[3:] for line in status.splitlines()) if status is not None else []
    )
    out: dict = {
        "git_revision": head.strip() if head else None,
        "git_dirty": None if status is None else bool(dirty),
        "git_dirty_paths": dirty[:20],
        "git_dirty_count": len(dirty),
        "note": "revision and dirty state describe the working tree; the "
        "compiled extension below is what was actually imported",
    }
    if head is None or status is None:
        out["note"] = "git could not be read here: revision and dirty state are unknown"
    source_paths = (
        ":(top)packages/confit/confit",
        ":(top)packages/confit/fuzz",
        ":(top)packages/confit/src",
        ":(top)packages/sql-transform/sql_transform",
        ":(top)Cargo.toml",
        ":(top)Cargo.lock",
        ":(top)pyproject.toml",
        ":(top)uv.lock",
        ":(top)packages/confit/Cargo.toml",
        ":(top)packages/confit/pyproject.toml",
        ":(top)packages/sql-transform/pyproject.toml",
    )
    out["working_patch"] = _git(
        "diff", "--binary", "--no-ext-diff", "HEAD", "--", *source_paths
    )
    untracked = _git(
        "ls-files", "--others", "--exclude-standard", "-z", "--", *source_paths
    )
    out["untracked_sources"] = (
        None
        if untracked is None
        else {
            name: (PKG / name).read_text(encoding="utf-8")
            for name in untracked.split("\0")
            if name
        }
    )
    try:
        import confit
        from confit import _engine

        out["build_profile"] = confit.BUILD_PROFILE
        ext = Path(_engine.__file__)
        out["extension"] = {
            "path": str(ext),
            "sha256": _sha256(ext),
            "mtime": datetime.datetime.fromtimestamp(
                ext.stat().st_mtime, datetime.timezone.utc
            ).isoformat(),
        }
    except Exception as e:  # noqa: BLE001 — provenance never fails a campaign
        out["build_profile"] = None
        out["extension"] = {"error": f"{type(e).__name__}: {e}"}
    return out


def _package_provenance() -> dict:
    from importlib.metadata import PackageNotFoundError, version

    import duckdb
    import pyarrow

    versions = {
        "python": sys.version.split()[0],
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "duckdb": duckdb.__version__,
        "pyarrow": pyarrow.__version__,
    }
    for name in ("numpy", "scikit-learn", "confit", "sql-transform"):
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    return versions


def _generator_provenance() -> dict:
    """Hashes of the sources that DEFINE a case and its verdict. A campaign
    compared against a different gen.py is a different population, and the
    seed range alone cannot say so."""
    return {
        "gen": {"path": "fuzz/gen.py", "sha256": _sha256(PKG / "fuzz" / "gen.py")},
        "oracle": {
            "path": "fuzz/oracle.py",
            "sha256": _sha256(PKG / "fuzz" / "oracle.py"),
        },
        "worker": {
            "path": "fuzz/worker.py",
            "sha256": _sha256(PKG / "fuzz" / "worker.py"),
        },
    }


def preflight_reference() -> dict:
    """Open the oracle BEFORE a single worker starts.

    A reference that cannot be constructed — wrong DuckDB, a failed version
    assertion — makes every worker answer SKIP, and 20000 SKIPs look like a
    campaign that ran. It is not one. Raising here costs one connection and
    turns that into a message.
    """
    from confit.oracle import Oracle

    try:
        with Oracle() as o:
            version = o.answer("SELECT version()").column(0)[0].as_py()
            threads, timezone = o.execute(
                "SELECT current_setting('threads'), current_setting('TimeZone')"
            ).fetchone()
            settings = {
                "optimizer": "disabled by Oracle constructor",
                "threads": threads,
                "TimeZone": timezone,
            }
    except Exception as e:  # noqa: BLE001 — reported, not swallowed
        raise CampaignError(
            f"oracle preflight failed ({type(e).__name__}: {e}); no worker was started"
        ) from e
    return {
        "engine": "duckdb",
        "version": version,
        "expected_version": Oracle.VERSION,
        "constructor": "confit.oracle.Oracle (PRAGMA disable_optimizer)",
        "settings": settings,
        "readings": [
            "optimizer-off (the baseline every verdict is graded against)",
            "optimizer-on (what a user sees; the gap is DIVERGE_OPT)",
        ],
    }


def _safe(fn, what: str):
    """`fn()`, or the reason it could not be read. Provenance records what it
    could not learn; it never takes a campaign down and never leaves a gap
    that reads as a fact."""
    try:
        return fn()
    except Exception as e:  # noqa: BLE001
        return {"error": f"{what} unavailable: {type(e).__name__}: {e}"}


def _legend():
    from .worker import ENCODING_LEGEND

    return ENCODING_LEGEND


def _provenance(
    *,
    params: dict,
    reference: dict,
    artifacts: Artifacts,
    status: str,
    errors: list[str],
) -> dict:
    return {
        "campaign": {**params, "status": status, "errors": errors},
        "artifacts": {
            "findings": artifacts.findings.name,
            "results": artifacts.results.name,
            "cases": artifacts.cases.name,
            "provenance": artifacts.provenance.name,
        },
        "generator": _safe(_generator_provenance, "generator sources"),
        "engine": _safe(_engine_provenance, "engine build"),
        "packages": _safe(_package_provenance, "package versions"),
        "reference": reference,
        "case_encoding": _safe(_legend, "case encoding legend"),
    }


# ------------------------------------------------------------------ workers


def _spawn(argv: list[str]):
    """A worker subprocess and the temp file holding its stderr, as
    `(proc, err)`. The caller owns `err` and must close it."""
    err = tempfile.TemporaryFile()
    proc = subprocess.Popen(  # noqa: S603 — our own module, fixed argv
        argv,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=err,
        cwd=PKG,
        text=True,
        encoding="utf-8",
    )
    return proc, err


def worker_argv() -> list[str]:
    return [sys.executable, "-m", "fuzz.worker"]


def _stderr_tail(err_file) -> str:
    try:
        err_file.seek(0)
        return err_file.read().decode(errors="replace")[-800:]
    except Exception:  # noqa: BLE001
        return ""


def _parse(line: str, seed: int, want: str) -> dict:
    """One protocol line. A worker that speaks anything else is a broken
    campaign, not a finding: the seeds it was holding are unaccounted for."""
    try:
        obj = json.loads(line)
    except json.JSONDecodeError as e:
        raise CampaignError(
            f"worker spoke non-JSON on seed {seed} ({e}): {line[:200]!r}"
        ) from e
    if not isinstance(obj, dict) or obj.get("event") != want:
        event = obj.get("event") if isinstance(obj, dict) else type(obj).__name__
        raise CampaignError(
            f"worker sent {event!r} "
            f"where a {want!r} event was due for seed {seed}: {line[:200]!r}"
        )
    if obj.get("seed") != seed:
        raise CampaignError(
            f"worker answered seed {obj.get('seed')!r} while seed {seed} was in flight"
        )
    if want == "result":
        from .oracle import KINDS

        if obj.get("kind") not in KINDS or not isinstance(obj.get("tags"), list):
            raise CampaignError(f"invalid verdict payload for seed {seed}")
    return obj


def _blame(seed: int, kind: str, detail: str, announced: dict | None) -> dict:
    """The finding for a worker that died or hung. The announced case is what
    makes it actionable: the SQL and tags come from the inputs the worker
    committed to before it went silent."""
    return {
        "seed": seed,
        "kind": kind,
        "klass": kind.lower(),
        "detail": detail,
        "sql": (announced or {}).get("sql", ""),
        "tags": (announced or {}).get("tags", []),
        "oracle_outcome": None,
        "case_announced": announced is not None,
    }


def _close_worker(proc, err):
    if proc.poll() is None:
        proc.kill()
    proc.wait(timeout=10)
    for pipe in (proc.stdin, proc.stdout):
        if pipe is not None:
            try:
                pipe.close()
            except OSError:
                pass  # the worker may already have closed its pipe
    err.close()


def _drive(seeds, results, timeout, lock, argv, save_case):
    """One worker thread: seeds off the shared iterator (`lock` guards it)
    into a subprocess, verdict dicts onto `results`.

    `timeout` is per seed and covers generation AND evaluation, since the
    worker does both. A worker that dies or outruns it is killed, blamed for
    the seed it was holding, and replaced.
    """
    proc, err = _spawn(argv)
    try:
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
                _close_worker(proc, err)
                proc, err = _spawn(argv)
                proc.stdin.write(f"{seed}\n")
                proc.stdin.flush()
            fired = threading.Event()

            def _kill(p=proc, f=fired):
                f.set()
                p.kill()

            timer = threading.Timer(timeout, _kill)
            timer.start()
            announced = None
            result_line = ""
            try:
                case_line = proc.stdout.readline()
                if case_line:
                    announced = _parse(case_line, seed, "case")
                    # ON DISK before the verdict is awaited: everything after
                    # this point can crash without taking the inputs with it.
                    save_case(case_line)
                    result_line = proc.stdout.readline()
            finally:
                # A protocol error leaves this thread; the timer must not
                # outlive it and kill a subprocess someone else respawned.
                timer.cancel()
                timer.join()
            if result_line:
                verdict = _parse(result_line, seed, "result")
                verdict.pop("event")
                results.append(verdict)
                continue
            kind = "TIMEOUT" if fired.is_set() else "PANIC"
            results.append(_blame(seed, kind, _stderr_tail(err), announced))
            _close_worker(proc, err)
            proc, err = _spawn(argv)
        proc.stdin.close()
        proc.wait(timeout=10)
    finally:
        # Including the protocol-error exit: a worker left running would hold
        # its pipes open and outlive the campaign that gave up on it.
        _close_worker(proc, err)


def _case_sink(path: Path):
    """`(write, close)` over the cases file: raw worker lines, appended under
    a lock and flushed, so the file survives the parent as well."""
    fh = path.open("x", encoding="utf-8")
    lock = threading.Lock()

    def write(line: str) -> None:
        with lock:
            fh.write(line if line.endswith("\n") else line + "\n")
            fh.flush()

    return write, fh.close


def _audit(results: list[dict], start: int, n: int) -> list[str]:
    """Every way the result set can fail to be the campaign that was asked
    for. Empty means each seed answered exactly once."""
    seen = collections.Counter(r["seed"] for r in results)
    want = set(range(start, start + n))
    problems = []
    missing = sorted(want - set(seen))
    if missing:
        problems.append(
            f"{len(missing)} seed(s) never produced a verdict, e.g. {missing[:5]}"
        )
    dupes = sorted(s for s, c in seen.items() if c > 1)
    if dupes:
        problems.append(
            f"{len(dupes)} seed(s) answered more than once, e.g. {dupes[:5]}"
        )
    stray = sorted(set(seen) - want)
    if stray:
        problems.append(f"verdicts for unrequested seed(s): {stray[:5]}")
    return problems


def campaign(
    start: int,
    n: int,
    workers: int,
    timeout: float,
    out: Path,
    argv: list[str] | None = None,
):
    """Seeds `start .. start + n - 1` across `workers` subprocesses: writes
    the four artifacts, reports, and returns every verdict dict.

    Raises `CampaignError` — after the evidence is on disk — when the
    reference will not open, a worker breaks the protocol, or the verdicts do
    not account for every seed. No summary prints in that case.
    """
    if n <= 0 or workers <= 0 or not math.isfinite(timeout) or timeout <= 0:
        raise CampaignError("n, workers and timeout must be positive and finite")
    art = artifacts_for(out)
    _reserve(art)
    reference = preflight_reference()
    argv = argv or worker_argv()
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    provenance = _provenance(
        params={}, reference=reference, artifacts=art, status="running", errors=[]
    )

    seeds = iter(range(start, start + n))
    results: list[dict] = []
    errors: list[str] = []
    lock = threading.Lock()
    save_case, close_cases = _case_sink(art.cases)

    def run():
        try:
            _drive(seeds, results, timeout, lock, argv, save_case)
        except Exception as e:  # noqa: BLE001 — a dead thread must not be silent
            with lock:
                errors.append(f"{type(e).__name__}: {e}")

    threads = [threading.Thread(target=run) for _ in range(workers)]
    try:
        for t in threads:
            t.start()
        done = 0
        while any(t.is_alive() for t in threads):
            for t in threads:
                t.join(timeout=5)
            if len(results) - done >= 500:
                done = len(results)
                print(f"... {done}/{n}", file=sys.stderr)
    finally:
        close_cases()

    errors += _audit(results, start, n)
    results.sort(key=lambda r: r["seed"])
    with art.results.open("x", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    params = {
        "start_seed": start,
        "n": n,
        "workers": workers,
        "timeout": timeout,
        "started": started,
        "finished": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "verdicts_recorded": len(results),
    }
    if not errors:
        try:
            report(results, art.findings)
        except Exception as e:  # noqa: BLE001 — retain a failed status in provenance
            errors.append(f"report failed ({type(e).__name__}: {e})")
    provenance["campaign"] = {
        **params,
        "status": "failed" if errors else "ok",
        "errors": errors,
    }
    with art.provenance.open("x", encoding="utf-8") as f:
        json.dump(provenance, f, indent=2)
    if errors:
        print(
            f"\n== CAMPAIGN FAILED: {len(results)}/{n} verdicts; "
            f"cases and partial results kept in {art.cases} / {art.results} ==",
            file=sys.stderr,
        )
        raise CampaignError("; ".join(errors))
    print("\n== artifacts ==")
    for p in art.all():
        print(f"  {p}")
    return results


# ------------------------------------------------------------------- report


def _refusal_rows(results: list[dict]):
    """Refusals grouped by reason and the already-computed reference outcome.
    These are reporting populations, not correctness or cost adjudications.
    """
    by_class = collections.Counter(
        r["klass"] for r in results if r["kind"] == "REFUSED"
    )
    outcomes: dict[str, collections.Counter] = collections.defaultdict(
        collections.Counter
    )
    for r in results:
        if r["kind"] == "REFUSED":
            outcomes[r["klass"]][r.get("oracle_outcome") or "unknown"] += 1
    return [(c, k, outcomes[k]) for k, c in by_class.most_common()]


def report(results: list[dict], out: Path):
    """Print the campaign summary and write every INTERESTING verdict to
    `out`, one JSON object per line. The file keeps the raw findings; only
    the printout collapses them to one example per (kind, klass)."""
    kinds = collections.Counter(r["kind"] for r in results)
    print("\n== verdicts ==")
    for k, c in kinds.most_common():
        print(f"  {k:14} {c}")

    rows = _refusal_rows(results)
    if rows:
        total = collections.Counter()
        for _, _, oc in rows:
            total.update(oc)
        served = total.get("served", 0)
        print(
            "\n== refusals by class, with what the oracle did with the same "
            f"query ({sum(total.values())} refusals: {served} the oracle "
            f"served, {total.get('build-error', 0)} it refused at build, "
            f"{total.get('run-error', 0)} it trapped at run, "
            f"{total.get('unknown', 0)} unknown) =="
        )
        for c, klass, oc in rows[:15]:
            spread = " ".join(f"{name}={m}" for name, m in oc.most_common())
            print(f"  {c:6}  {klass:46} {spread}")

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

    # Optimizer emulation is a primary mismatch, not desired coverage.
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
    with out.open("x", encoding="utf-8") as f:
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
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="findings path; three sidecars are derived from its stem. "
        "Defaults to a fresh timestamped name, and an existing artifact is "
        "never overwritten.",
    )
    a = ap.parse_args()
    try:
        campaign(a.seed, a.n, a.workers, a.timeout, a.out or default_out())
    except CampaignError as e:
        print(f"campaign failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
