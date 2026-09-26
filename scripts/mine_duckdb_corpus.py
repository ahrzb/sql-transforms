"""Mine the duckdb/ source clone's sqllogictest corpus into replayable cases.

Walks a fixed set of test/sql subtrees, replays each file's setup statements in
an in-memory DuckDB, and keeps every `query` block whose SQL falls inside the
specializer's v0 subset. Expected outputs are recomputed by DuckDB at mining
time (the file's own expected blocks are ignored — sort modes and hashing make
them fiddly, and DuckDB itself is our oracle anyway).

Output: packages/confit/tests/corpus/duckdb_mined.jsonl, one case per line:
    {"source": ..., "setup": [...], "sql": ..., "cols": [...], "rows": [[...]]}

Replay contract: run `setup` in a fresh DuckDB to reconstruct the input tables,
feed them to the engine under test, and classify into THREE outcomes:
  match             — engine output equals `rows`
  clean-unsupported — engine rejects at build time with an "unsupported" error
  FAIL              — mismatch, wrong error, or crash
The shape filter below only drops what can never be v0 (aggregation, windows,
set ops, subqueries). It deliberately keeps SQL beyond the v0 builtin list
(exotic string functions, star-expansion sugar, `::` casts): those must fail
*cleanly* today, and each one the engine learns flips from clean-unsupported to
must-match — the corpus is the growth ladder, not a fixed pass bar.

Provenance: each run also writes `duckdb_mined.provenance.json` beside the
corpus — date, DuckDB version, the settings profile the expected rows were
computed under (a fresh default connection per file: OPTIMIZER ON, so these
rows are not observations of the optimizer-off oracle), the clone's and this
script's revisions, the mined directories, and the counts. A corpus file with
no stamp predates this and must not be given one after the fact.

ponytail: line-oriented parse, no real sqllogictest grammar. Files using
directives we don't model (require, loop, mode, ...) are skipped whole; a
mis-parsed edge case at worst drops a case, never fabricates one, because every
kept SQL is re-executed by DuckDB before it is recorded.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import duckdb

REPO = Path(__file__).resolve().parent.parent
CORPUS_DIRS = [
    "projection",
    "filter",
    "conjunction",
    "cast",
    "select",
    "join/inner",
    "join/left_outer",
    "function/string",
    "function/numeric",
    "function/generic",
    "function/operator",
]
OUT = REPO / "packages" / "confit" / "tests" / "corpus" / "duckdb_mined.jsonl"
STAMP = OUT.with_name("duckdb_mined.provenance.json")

# A whole file is skipped when it uses machinery the replayer doesn't model.
FILE_SKIP = re.compile(
    r"^(require|load|loop|foreach|endloop|mode|hash-threshold|restart|sleep|concurrentloop)\b",
    re.MULTILINE,
)

# v0 subset filter: single plain SELECT, no set ops / aggregation / windows /
# subqueries / CTEs. Conservative — a false reject only shrinks the corpus.
BANNED = re.compile(
    r"\b(WITH|UNION|EXCEPT|INTERSECT|GROUP\s+BY|HAVING|ORDER\s+BY|LIMIT|OFFSET|"
    r"OVER|DISTINCT|EXISTS|VALUES|SAMPLE|QUALIFY|WINDOW|LATERAL|PIVOT|UNNEST|"
    r"RECURSIVE|IN\s*\()\b",
    re.IGNORECASE,
)
SUBQUERY = re.compile(r"\(\s*SELECT\b", re.IGNORECASE)
SETUP_KEEP = re.compile(r"^(CREATE|INSERT|DROP|UPDATE|DELETE)\b", re.IGNORECASE)
SCALARS = (int, float, str, bool, type(None))

MAX_ROWS = 64  # bigger results are batch tests, not unit cases


def v0_ok(sql: str) -> bool:
    flat = " ".join(sql.split())
    return (
        flat.upper().startswith("SELECT")
        and ";" not in flat.rstrip(";")
        and not BANNED.search(flat)
        and not SUBQUERY.search(flat)
        and " FROM " in flat.upper()  # pure-literal SELECTs have no dynamic input
    )


def blocks(text: str):
    """Yield ('statement'|'query', sql) for the blocks we understand."""
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("statement ok") or line.startswith("query "):
            kind = "statement" if line.startswith("statement") else "query"
            i += 1
            sql_lines = []
            while i < len(lines) and lines[i].strip() not in ("", "----"):
                sql_lines.append(lines[i])
                i += 1
            # skip the file's expected block; we recompute via duckdb
            while i < len(lines) and lines[i].strip() != "":
                i += 1
            yield kind, "\n".join(sql_lines).strip()
        else:
            i += 1


def mine_file(path: Path, rel: str, out) -> tuple[int, int]:
    text = path.read_text(encoding="utf-8", errors="replace")
    if FILE_SKIP.search(text):
        return 0, 0
    con = duckdb.connect()
    setup: list[str] = []
    kept = seen = 0
    for kind, sql in blocks(text):
        if not sql:
            continue
        if kind == "statement":
            try:
                con.execute(sql)
            except Exception:  # noqa: BLE001 -- broken state; stop replaying
                break
            if SETUP_KEEP.match(sql):
                setup.append(sql)
            continue
        seen += 1
        if not v0_ok(sql):
            continue
        try:
            cur = con.execute(sql)
            cols = [d[0] for d in cur.description]
            rows = cur.fetchall()
        except Exception:  # noqa: BLE001, S112 -- needs state we skipped; drop case
            continue
        if len(rows) > MAX_ROWS:
            continue
        if any(not isinstance(v, SCALARS) for row in rows for v in row):
            continue
        out.write(
            json.dumps(
                {
                    "source": rel,
                    "setup": list(setup),
                    "sql": sql,
                    "cols": cols,
                    "rows": [list(r) for r in rows],
                }
            )
            + "\n"
        )
        kept += 1
    con.close()
    return kept, seen


def _git(cwd: Path, *args: str) -> str:
    try:
        return subprocess.run(  # noqa: S603 — fixed argv
            ["git", *args],  # noqa: S607
            capture_output=True,
            text=True,
            cwd=cwd,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def provenance(kept: int, seen: int, files: int) -> dict:
    """The stamp a mining run writes: what produced the expected rows."""
    con = duckdb.connect()
    try:
        threads = con.execute("SELECT current_setting('threads')").fetchone()[0]
    finally:
        con.close()
    return {
        "date": datetime.date.today().isoformat(),
        "duckdb": duckdb.__version__,
        "settings": {
            "connection": "fresh duckdb.connect() per source file, defaults",
            "optimizer": "on",
            "threads": threads,
        },
        "expected_rows": "recomputed by DuckDB at mining time; the files' own "
        "expected blocks are ignored",
        "duckdb_clone_revision": _git(REPO / "duckdb", "rev-parse", "HEAD"),
        "miner_revision": _git(REPO, "rev-parse", "HEAD"),
        "miner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16],
        "dirs": CORPUS_DIRS,
        "kept": kept,
        "seen": seen,
        "files": files,
    }


def main() -> int:
    root = REPO / "duckdb" / "test" / "sql"
    if not root.is_dir():
        print(f"duckdb clone not found at {root}", file=sys.stderr)
        return 1
    OUT.parent.mkdir(parents=True, exist_ok=True)
    total_kept = total_seen = files = 0
    with OUT.open("w", encoding="utf-8") as out:
        for d in CORPUS_DIRS:
            dir_kept = 0
            for path in sorted((root / d).rglob("*.test")):
                rel = path.relative_to(REPO / "duckdb").as_posix()
                kept, seen = mine_file(path, rel, out)
                dir_kept += kept
                total_kept += kept
                total_seen += seen
                files += 1
            print(f"{d:24s} +{dir_kept}")
    stamp = provenance(total_kept, total_seen, files)
    STAMP.write_text(json.dumps(stamp, indent=2) + "\n", encoding="utf-8")
    print(f"\n{total_kept} cases kept / {total_seen} queries seen / {files} files")
    print(f"-> {OUT.relative_to(REPO)}")
    print(f"-> {STAMP.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
