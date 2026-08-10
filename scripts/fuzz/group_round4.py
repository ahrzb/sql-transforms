"""Analyze round4 candidates: dedupe across backends, family-tag, group.

Read-only analysis of scripts/fuzz/output/round4-*.jsonl. Does not run the
engine or DuckDB — it groups the recorded candidates so the verifier can
reproduce only the interesting families.

Usage:
  uv run python scripts/fuzz/group_round4.py
"""

from __future__ import annotations

import collections
import json
import os
import re

OUT = os.path.join(os.path.dirname(__file__), "output")


def norm_case(case: dict) -> str:
    return json.dumps(case, sort_keys=True, ensure_ascii=False)


def main() -> None:
    recs: dict[str, list[dict]] = collections.OrderedDict()
    for fn in sorted(os.listdir(OUT)):
        if not (fn.startswith("round4-") and fn.endswith(".jsonl")):
            continue
        path = os.path.join(OUT, fn)
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:  # noqa: S112
                    continue
                key = norm_case(rec["case"])
                rec["_file"] = fn
                recs.setdefault(key, []).append(rec)

    groups: dict[str, list] = collections.defaultdict(list)
    for key, versions in recs.items():
        c = versions[0]["case"]
        classes = sorted({v["result"]["classifier"] for v in versions})
        family = tag(c, versions)
        groups[family].append((key, c, classes, versions))

    print(f"unique cases: {len(recs)}  groups: {len(groups)}")
    for fam, items in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        print(f"\n=== {fam}  ({len(items)}) ===")
        for _key, c, classes, versions in items[:4]:
            res = versions[0]["result"]
            sql = c["sql"]
            print(f"  [{','.join(classes)}] {sql[:110]}")
            print(
                f"      rows_n={len(c['rows'])} statics={list(c['statics'])} shape={c['shape']}"
            )
            print(
                f"      duck_rows={json.dumps(res['duck_rows'])[:120]} confit_rows={json.dumps(res['confit_rows'])[:120]}"
            )


def tag(c: dict, versions: list[dict]) -> str:
    sql = c["sql"]
    cls = versions[0]["result"]["classifier"]
    if cls.startswith("arrow_error"):
        return "arrow_error_struct"
    if cls == "S_arrow_schema_diff":
        if re.search(r"\b\d+(\.\d+)?\b", sql) or "CASE" in sql or "COALESCE" in sql:
            return "S_int_literal_width(TASK-79)"
        return "S_arrow_schema_diff"
    if cls == "E_engine_serves_duck_errors":
        if "negation" in str(versions[0]["result"].get("duck_err", "")):
            return "E_int_neg_overflow"
        return "E_engine_serves_duck_errors"
    if "1.5" in sql or re.search(r"SELECT -?\d+(\.\d+)? AS", sql):
        return "dec_literal_f64(documented)"
    if cls == "A_rows_differ":
        if "TRY_CAST" in sql or "::BIGINT" in sql or "CAST(" in sql and "BIGINT" in sql:
            return "A_trycast_str_to_int"
        return "A_rows_differ"
    return cls


if __name__ == "__main__":
    main()
