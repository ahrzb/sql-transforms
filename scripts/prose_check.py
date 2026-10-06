"""Check prose against the writing rules in AGENTS.md §1.

The checker reads its word list from AGENTS.md: the fenced block after the
line "<!-- prose-check: words -->". It does not read the _Avoid_ lists in
GLOSSARY.md. Those words are wrong only for one concept, so a reader checks
them, not a regular expression (the fresh-reader skill).

It checks these rules:

- avoid:      a word in the AGENTS.md list;
- length:     a sentence of more than 25 words;
- semicolon:  a semicolon that joins two sentences.

It ignores code blocks, inline code, tables, headings, quotes, link targets,
HTML comments and YAML front matter. The avoid rule also ignores a word in
double quotes, because that is an example of the word, not a use.

    python scripts/prose_check.py            # every file in scope
    python scripts/prose_check.py FILE...    # these files
    python scripts/prose_check.py -          # text on stdin (a PR body, a draft)

The scope is SCOPE below: the docs that the owner reviews or maintains. A
file joins the scope when someone rewrites it to the rules. Working files for
agents (PLANS.md, tickets.md, worker briefs) are not in the scope. A report is fixed when it is written, so a report dated before
REPORTS_FROM stays out of the scope.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MAX_WORDS = 25
REPORTS_FROM = "2026-10-07"
SCOPE = [
    "AGENTS.md",
    "GLOSSARY.md",
    "README.md",
    ".claude/skills/*/SKILL.md",
    "loops/*.md",
    "loops/confit/README.md",
    "loops/confit/goal.md",
    "loops/confit/decisions/README.md",
    "loops/confit/decisions/open/*.md",
    "loops/confit/reports/*.md",
]


def _avoid_words() -> list[str]:
    words = []
    agents = (ROOT / "AGENTS.md").read_text()
    block = re.search(r"<!-- prose-check: words -->\s*```\w*\n(.*?)```", agents, re.S)
    if block:
        words += [w.strip() for w in block.group(1).splitlines() if w.strip()]
    return words


def _in_scope(path: Path) -> bool:
    rel = path.resolve().relative_to(ROOT).as_posix()
    if not any(Path(rel).full_match(g) for g in SCOPE):
        return False
    m = re.match(r"loops/[^/]+/reports/(\d{4}-\d{2}-\d{2})-", rel)
    return not (m and m.group(1) < REPORTS_FROM)


def _prose_lines(text: str) -> list[tuple[int, str]]:
    """The lines of `text` that are prose, with code and markup removed."""
    out, fence, front = [], False, False
    lines = text.splitlines()
    for i, line in enumerate(lines, 1):
        s = line.strip()
        if i == 1 and s == "---":
            front = True
            continue
        if front:
            front = s != "---"
            continue
        if s.startswith("```"):
            fence = not fence
            continue
        if fence or s.startswith(("|", "#", "<!--", ">", "_Avoid_:")) or not s:
            out.append((i, ""))
            continue
        s = re.sub(r"`[^`]*`", "CODE", s)
        s = re.sub(r"\]\([^)]*\)", "]", s)
        s = re.sub(r"https?://\S+", "URL", s)
        out.append((i, s))
    return out


def _sentences(prose: list[tuple[int, str]]):
    """Yield (first line, sentence) for each sentence in the prose."""
    buf, start = [], 0
    for i, s in prose:
        # A blank line, or a new list item, ends the sentence before it.
        if not s or re.match(r"([-*]|\d+\.)\s", s):
            if buf:
                yield start, " ".join(buf)
            buf, start = [], i
            if not s:
                continue
            s = re.sub(r"^([-*]|\d+\.)\s+", "", s)
        if not buf:
            start = i
        buf.append(s)
        joined = " ".join(buf)
        parts = re.split(r'(?:(?<=[.!?:])|(?<=[.!?:]\*\*))\s+(?=[A-Z"(*_`[])', joined)
        for p in parts[:-1]:
            yield start, p
        buf = [parts[-1]]
    if buf:
        yield start, " ".join(buf)


def check(text: str, name: str, words: list[str]) -> list[str]:
    problems = []
    prose = _prose_lines(text)
    for i, s in prose:
        used = re.sub(r'"[^"]*"', "QUOTE", s)  # a quoted word is an example, not a use
        for w in words:
            if re.search(rf"(?<![\w-]){re.escape(w)}(?![\w-])", used, re.I):
                problems.append(f'{name}:{i}: avoid: "{w}" (AGENTS.md §1)')
        if re.search(r";\s+\w", s):
            problems.append(f"{name}:{i}: semicolon: write two sentences")
    for i, sent in _sentences(prose):
        n = len(re.findall(r"[\w'’-]+", re.sub(r"\*|_", "", sent)))
        if n > MAX_WORDS:
            problems.append(f"{name}:{i}: length: {n} words (maximum {MAX_WORDS})")
    return problems


def main(argv: list[str]) -> int:
    words = _avoid_words()
    if argv == ["-"]:
        problems = check(sys.stdin.read(), "<stdin>", words)
    else:
        if argv:
            files = [Path(a) for a in argv]
        else:
            files = sorted({p for g in SCOPE for p in ROOT.glob(g)})
        problems = []
        for f in files:
            if f.suffix == ".md" and _in_scope(f):
                rel = f.resolve().relative_to(ROOT).as_posix()
                problems += check(f.read_text(), rel, words)
    for p in problems:
        print(p)
    if problems:
        print(f"prose_check: {len(problems)} problems. The rules are in AGENTS.md §1.")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
