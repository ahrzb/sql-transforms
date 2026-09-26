"""The legacy pin corpus (`packages/confit/docs/specs/pins-*/*.json`): one
uniform header per file, derived only from what the file and git already say.

    uv run python scripts/pin_corpus.py header     # add or refresh `_pin`
    git diff                                       # the review surface

Each file gains a `_pin` object as its first key. It is inserted as text, so
the rest of the file stays byte-identical. Nothing is invented: a field the
file does not state is `"unknown"`, and `committed` is the git date the file
was added, never presented as a capture date. The header does not convert
the legacy fields or make an old capture a fresh observation of the
optimizer-off oracle (claim: pin-provenance).
"""

from __future__ import annotations

import collections
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PINS = ROOT / "packages/confit/docs/specs"
SCHEMA = 1

# Files whose recorded observations are optimizer plan choices (which side a
# hash join builds): visible only with the optimizer ON, so the capture was.
_OPTIMIZER_ON = {
    "pins-stageB/dup-key-equi.json": "records the optimizer's build/probe choice",
    "pins-stageB/order-contract.json": "records the optimizer's stream/build choice",
    "pins-stageB/self-join-stars.json": "records the optimizer's build/probe choice",
}


# The under-determined-field token (claim: under-determined-token): a pin
# field whose recorded value is not THE contracted answer. `by:<discriminator>`
# means it varies by a named discriminator (the recorded value holds for the
# capture's); `unspecified` means the contract leaves it open (the recorded
# value is one valid answer). `at` is a JSON pointer, `*` matching every list
# index. Reviewed data, like _OPTIMIZER_ON: added by hand, checked by test.
_ORDER = "row order is a hash-join artifact; the contract is the multiset"
_VARIES = {
    "pins-wave3/math_tail.json": [
        {
            "at": "/corrections/0/probes/*/bits",
            "mark": "by:platform",
            "note": "%/mod by zero NaN sign follows platform libm "
            "(divergence: nan-sign-per-platform)",
        },
    ],
    "pins-stageB/order-contract.json": [
        {"at": "/pins/*/result", "mark": "unspecified", "note": _ORDER},
    ],
    "pins-stageB/dup-key-equi.json": [
        {"at": "/pins/*/result", "mark": "unspecified", "note": _ORDER},
    ],
}
TOKENS = ("unspecified", "by:")


def resolve(d, pointer: str) -> list:
    """Every value a `varies` pointer names (`*` fans out over a list)."""
    found = [d]
    for part in pointer.strip("/").split("/"):
        nxt = []
        for node in found:
            if part == "*" and isinstance(node, list):
                nxt.extend(node)
            elif isinstance(node, list) and part.isdigit() and int(part) < len(node):
                nxt.append(node[int(part)])
            elif isinstance(node, dict) and part in node:
                nxt.append(node[part])
        found = nxt
    return found


def pin_files() -> list[Path]:
    return sorted(PINS.glob("pins-*/*.json"))


def _rel(p: Path) -> str:
    return p.relative_to(PINS).as_posix()


def _git_added(p: Path) -> str:
    run = subprocess.run(  # noqa: S603 — fixed argv
        ["git", "log", "--diff-filter=A", "--follow", "--format=%as", "--", str(p)],  # noqa: S607
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    dates = run.stdout.split()
    return dates[-1] if dates else "unknown"


def _engine(d: dict) -> tuple[str, str]:
    for key, engine in (
        ("duckdb_version", "duckdb"),
        ("spark_version", "spark"),
        ("sqlparser_version", "sqlparser"),
    ):
        if key in d:
            m = re.search(r"\d+\.\d+\.\d+", str(d[key]))
            return engine, m.group(0) if m else str(d[key])
    stated = {m.group(1) for t in _strings(d) for m in _DUCK_IN_PROSE.finditer(t)}
    if len(stated) == 1:  # the prose names one version; two would be a guess
        return "duckdb", stated.pop()
    meta = d.get("_meta", {})
    if "duckdb" in str(meta.get("engine", "")) or stated:
        return "duckdb", "unknown"
    return "unknown", "unknown"


def _subject(d: dict, p: Path) -> str:
    for key in ("area", "family", "topic"):
        if isinstance(d.get(key), str):
            return d[key]
    meta = d.get("_meta", {})
    if meta.get("task"):
        return f"{meta['task']}: {meta.get('spec', p.stem)}"
    return p.stem


_DUCK_IN_PROSE = re.compile(r"\bDuckDB (?:v)?(\d+\.\d+\.\d+)", re.I)
_CAPTURE = re.compile(
    r"\b(?:measured|validated|captured)\b[^\n]{0,60}?(20\d\d-\d\d-\d\d)", re.I
)


def _strings(o):
    if isinstance(o, dict):
        for k, v in o.items():
            if k != "_pin":
                yield from _strings(v)
    elif isinstance(o, list):
        for v in o:
            yield from _strings(v)
    elif isinstance(o, str):
        yield o


def _captured(d: dict) -> str:
    """The capture date the file states: an explicit `date`/`measured` in its
    metadata, else a date the prose ties to measuring. Never a date that
    merely appears (a DATE literal, a correction's date)."""
    for meta_key in ("_meta", "meta"):
        meta = d.get(meta_key)
        if isinstance(meta, dict):
            for k in ("date", "measured", "captured"):
                m = re.search(r"20\d\d-\d\d-\d\d", str(meta.get(k, "")))
                if m:
                    return m.group(0)
    for s in _strings(d):
        m = _CAPTURE.search(s)
        if m:
            return m.group(1)
    return "unknown"


# Where a pin may be cited from. The oracle spec's claims are the decisions a
# pin evidences; any other doc or test that cites a pin is recorded by path.
_CITERS = (
    "packages/confit/docs",
    "packages/confit/tests",
    "packages/confit/src",
    "docs",
)
_DIR_CITE = re.compile(r"(pins-[A-Za-z0-9]+)/(?:\*\.json|(?![\w.-]))")
_SLUG = re.compile(r"\*\*(claim|divergence|decision|exclusion|gap): ([a-z0-9-]+)")


def _citations() -> dict[str, set[str]]:
    """pin path (relative to specs/) -> the back-references that cite it."""
    rels = [_rel(p) for p in pin_files()]
    out: dict[str, set[str]] = {r: set() for r in rels}
    # A citation spells the pin as its specs/-relative path, or by its bare
    # file name when that name is unique in the corpus ("pins: struct-star.json").
    base = collections.Counter(r.split("/")[1] for r in rels)
    names = {r: r for r in rels}
    names.update({r.split("/")[1]: r for r in rels if base[r.split("/")[1]] == 1})
    for base in _CITERS:
        for f in sorted((ROOT / base).rglob("*")):
            if f.suffix not in (".md", ".py", ".rs") or not f.is_file():
                continue
            if f.is_relative_to(PINS) and f.parent.name.startswith("pins-"):
                continue
            text = f.read_text(encoding="utf-8", errors="replace")
            oracle = "docs/oracle/" in f.as_posix()
            # A whole-directory citation ("the JSON files in `pins-wave3/`",
            # "`pins-stageB/*.json`") cites every pin in that directory.
            for m in _DIR_CITE.finditer(text):
                for r in rels:
                    if r.startswith(m.group(1) + "/"):
                        out[r].add(f.relative_to(ROOT).as_posix())
            for n in names:
                i = text.find(n)
                while i != -1:
                    slug = None
                    if oracle:
                        last = None
                        for m in _SLUG.finditer(text, 0, i):
                            last = m
                        if last is not None:
                            slug = f"{last.group(1)}: {last.group(2)}"
                    out[names[n]].add(slug or f.relative_to(ROOT).as_posix())
                    i = text.find(n, i + 1)
    return out


_CITES: dict[str, set[str]] | None = None


def evidences(p: Path) -> list[str]:
    global _CITES  # noqa: PLW0603 — one scan per process
    if _CITES is None:
        _CITES = _citations()
    return sorted(_CITES.get(_rel(p), ()))


def header(p: Path, d: dict) -> dict:
    meta = d.get("_meta", {}) if isinstance(d.get("_meta"), dict) else {}
    engine, version = _engine(d)
    rel = _rel(p)
    optimizer = "on" if rel in _OPTIMIZER_ON else "unknown"
    h = {
        "schema": SCHEMA,
        "subject": _subject(d, p),
        "engine": engine,
        "engine_version": version,
        "optimizer": optimizer,
        "captured": _captured(d),
        "harness": meta.get("how", "unknown"),
        "committed": _git_added(p),
    }
    if rel in _OPTIMIZER_ON:
        h["optimizer_evidence"] = _OPTIMIZER_ON[rel]
    # The decisions this pin is evidence for, derived from who cites it; an
    # empty list is a finding (an orphan pin), not a gap to fill by hand.
    h["evidences"] = evidences(p)
    if rel in _VARIES:
        h["varies"] = _VARIES[rel]
    return h


def _without_header(text: str) -> str:
    """The file text with any existing `_pin` line removed."""
    return re.sub(r'\n[ \t]*"_pin": \{.*?\},(?=\n)', "", text, count=1, flags=re.S)


def write_header(p: Path, h: dict) -> None:
    text = _without_header(p.read_text(encoding="utf-8"))
    brace = text.index("{")
    rest = text[brace + 1 :]
    indent = re.match(r"\s*", rest).group(0).split("\n")[-1] or "  "
    line = f'\n{indent}"_pin": {json.dumps(h, ensure_ascii=False)},'
    new = text[: brace + 1] + line + rest
    json.loads(new)  # never write a file that stopped parsing
    p.write_text(new, encoding="utf-8")


def cmd_header() -> None:
    for p in pin_files():
        d = json.loads(p.read_text(encoding="utf-8"))
        write_header(p, header(p, d))
    print(f"headers written: {len(pin_files())} files")


def main(argv: list[str]) -> int:
    cmds = {"header": cmd_header}
    if len(argv) != 1 or argv[0] not in cmds:
        print(f"usage: pin_corpus.py {{{','.join(cmds)}}}", file=sys.stderr)
        return 2
    cmds[argv[0]]()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
