"""Reject unrecorded native bound relaxations: python scripts/check_bounds.py BASE."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

MANIFEST = "packages/sql-transform/spec/native/bounds.md"
ROOT = Path(__file__).resolve().parents[1]


def parse(text: str) -> dict[str, dict]:
    entries = {}
    for line in text.splitlines():
        if not line.startswith("| ") or line.startswith("| class |"):
            continue
        name, parity, ulps, functions, ks, hashes, _decisions = (
            cell.strip() for cell in line.strip("| ").split("|")
        )
        entries[name] = {
            "parity": parity,
            "ulps": json.loads(ulps),
            "functions": json.loads(functions),
            "ks": json.loads(ks),
            "hashes": json.loads(hashes),
        }
    if not entries:
        raise ValueError("empty or invalid native bounds manifest")
    return entries


def relaxations(base: dict, head: dict) -> list[str]:
    """New classes and lower ceilings are fine; implementations need a record."""
    findings = []
    for name, old in base.items():
        if name not in head:
            continue
        new = head[name]
        if new["ulps"] > old["ulps"]:
            findings.append(f"{name}: ulp ceiling {old['ulps']} -> {new['ulps']}")
        for key, label in (("functions", "function ulps"), ("ks", "K")):
            for witness, before in old[key].items():
                after = new[key].get(witness)
                if after is not None and (
                    max(after) if isinstance(after, list) else after
                ) > (max(before) if isinstance(before, list) else before):
                    findings.append(f"{name}: {label} {witness}: {before} -> {after}")
        if new["hashes"] != old["hashes"]:
            findings.append(f"{name}: K/S/tau/map source hashes changed")
    return findings


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True)  # noqa: S603,S607


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base_ref")
    args = parser.parse_args()
    # Older bases predate the manifest. Currency tests still require HEAD's.
    files = git("ls-tree", "--name-only", args.base_ref, "--", MANIFEST)
    if not files.strip():
        print(
            "Base predates the native bounds manifest; no approved bounds to compare."
        )
        return 0
    base = parse(git("show", f"{args.base_ref}:{MANIFEST}"))
    head = parse(git("show", f"HEAD:{MANIFEST}"))
    findings = relaxations(base, head)
    changed = git(
        "diff",
        "--name-only",
        args.base_ref,
        "HEAD",
        "--",
        "proposals/closed/",
    )
    for finding in findings:
        print(finding)
    if findings and not changed.strip():
        print(
            "Bound changes require a changed closed proposal under proposals/closed/."
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
