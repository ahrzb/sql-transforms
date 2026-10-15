"""Re-pin DuckDB's operator precedence against the installed DuckDB.

Writes `packages/confit/docs/specs/pins-wave5/operator-precedence.json`: for
each form, the tree DuckDB parses it to, or the error it refuses it with.
confit's parser settings (`src/specializer/precedence.rs`) are tested against
this file, so a version bump that moves a level surfaces as a reviewable diff.

    uv run python scripts/pin_precedence.py
    uv run python scripts/pin_corpus.py header
    git diff        # this diff IS the drift report

DuckDB parses `SELECT <form>`; `json_serialize_sql` gives its parse and
`json_deserialize_sql` prints the parse back with every operator in
parentheses, so the printed text is the tree.
"""

import datetime
import json
from pathlib import Path

import duckdb

PIN = (
    Path(__file__).resolve().parent.parent
    / "packages/confit/docs/specs/pins-wave5/operator-precedence.json"
)

# Every binary operator confit's parser meets, one or two of each level.
BINARY = [
    "OR",
    "AND",
    "=",
    "<>",
    "<",
    "IS DISTINCT FROM",
    "IS NOT DISTINCT FROM",
    "LIKE",
    "NOT LIKE",
    "ILIKE",
    "SIMILAR TO",
    "~",
    "!~",
    "^@",
    "||",
    "+",
    "-",
    "*",
    "/",
    "//",
    "%",
    "&",
    "|",
    "<<",
    ">>",
    "^",
]

# Each binary operator `x` beside a ternary, postfix or prefix form.
AROUND = [
    "a {x} b BETWEEN c AND d",
    "a BETWEEN b AND c {x} d",
    "a {x} b NOT BETWEEN c AND d",
    "a {x} b IN (c)",
    "a IN (b) {x} c",
    "a {x} b NOT IN (c)",
    "a {x} b IS NULL",
    "a IS NULL {x} b",
    "a {x} b IS TRUE",
    "NOT a {x} b",
    "a {x} NOT b",
    "- a {x} b",
    "a {x} - b",
    "a {x} b::INT",
]

# Forms of their own: chains of one level, and the operand of BETWEEN.
OTHER = [
    "a IS NULL IS NULL",
    "a BETWEEN NOT b AND c",
    "a NOT BETWEEN NOT b AND c",
    "a BETWEEN b AND NOT c",
    "a IN (b) IN (c)",
    "a BETWEEN b AND c BETWEEN d AND e",
    "a BETWEEN b AND c IN (d)",
    "a IN (b) BETWEEN c AND d",
    "NOT a IS DISTINCT FROM b AND c",
    "a OR b IS DISTINCT FROM c AND d",
    "a AND b IS DISTINCT FROM c OR d",
    "a IS DISTINCT FROM b = c AND d",
    "a = b IS DISTINCT FROM c OR d",
    "- a::INT",
    # A prefix NOT over EXISTS takes the operators above its level, as any
    # other NOT does.
    "NOT EXISTS (SELECT 1) IS NULL",
    "NOT EXISTS (SELECT 1) = b",
    "NOT EXISTS (SELECT 1) AND b",
    "NOT EXISTS (SELECT 1) OR b",
    "NOT EXISTS (SELECT 1) LIKE b",
    "NOT EXISTS (SELECT 1) IN (b)",
    "NOT EXISTS (SELECT 1) BETWEEN b AND c",
    "NOT EXISTS (SELECT 1) IS DISTINCT FROM b",
    "NOT EXISTS (SELECT 1) || b",
    "NOT EXISTS (SELECT 1) + b",
    "NOT EXISTS (SELECT 1)::INT",
    "NOT NOT EXISTS (SELECT 1)",
    "a AND NOT EXISTS (SELECT 1)",
    "a = NOT EXISTS (SELECT 1)",
    "a || NOT EXISTS (SELECT 1)",
    "a BETWEEN NOT EXISTS (SELECT 1) AND c",
    "EXISTS (SELECT 1) IS NULL",
    "EXISTS (SELECT 1) = b",
    "- EXISTS (SELECT 1)",
    # IN over a subquery takes the level of IN over a list.
    "a IN (SELECT 1) IS NULL",
    "a NOT IN (SELECT 1) IS NULL",
    "a IN (SELECT 1) IS TRUE",
    "NOT a IN (SELECT 1)",
    "NOT a NOT IN (SELECT 1)",
    "NOT a IN (SELECT 1) IS NULL",
    "a = b IN (SELECT 1)",
    "a IN (SELECT 1) = b",
    "a || b IN (SELECT 1)",
    "a IN (SELECT 1) || b",
    "a + b IN (SELECT 1)",
    "a OR b IN (SELECT 1)",
    "a IN (SELECT 1) AND b",
    "a IN (SELECT 1) IN (SELECT 1)",
    "a IN (SELECT 1) IN (b)",
    "a IN (b) IN (SELECT 1)",
    "a IN (SELECT 1) BETWEEN b AND c",
    "a BETWEEN b AND c IN (SELECT 1)",
    "a LIKE b IN (SELECT 1)",
    "a IN (SELECT 1) LIKE b",
    "a IS NULL IN (SELECT 1)",
    "a IS DISTINCT FROM b IN (SELECT 1)",
    "a IN (SELECT 1) IS DISTINCT FROM b",
    "a IN (SELECT 1)::INT",
    "- a IN (SELECT 1)",
    "a = NOT b IN (SELECT 1)",
    "a || NOT b IN (SELECT 1)",
]


def forms() -> list[str]:
    out = [f"a {x} b {y} c" for x in BINARY for y in BINARY]
    out += [f.format(x=x) for f in AROUND for x in BINARY]
    out += OTHER
    assert len(set(out)) == len(out), "a form is listed twice"
    return out


def main() -> None:
    con = duckdb.connect()
    trees, rejected = [], []
    for form in forms():
        sql = f"SELECT {form}"
        parsed = json.loads(
            con.execute("SELECT json_serialize_sql(?)", [sql]).fetchone()[0]
        )
        if parsed.get("error"):
            rejected.append(
                [form, f"{parsed['error_type']}: {parsed['error_message']}"]
            )
            continue
        text = con.execute(
            "SELECT json_deserialize_sql(?::JSON)", [json.dumps(parsed)]
        ).fetchone()[0]
        assert text.startswith("SELECT "), text
        trees.append([form, text.removeprefix("SELECT ")])
    head = {
        "area": "operator-precedence",
        "duckdb_version": duckdb.__version__,
        "_meta": {
            "date": datetime.date.today().isoformat(),
            "how": (
                "scripts/pin_precedence.py: json_serialize_sql, "
                "then json_deserialize_sql"
            ),
        },
        "method": (
            "DuckDB parses SELECT <form>. `trees` holds each form it parses, "
            "with its parse printed back by json_deserialize_sql, every operator "
            "in parentheses. `rejected` holds each form it refuses, with the "
            "error. One row per line."
        ),
    }
    lines = ["{"] + [f"  {json.dumps(k)}: {json.dumps(v)}," for k, v in head.items()]
    for key, rows in (("trees", trees), ("rejected", rejected)):
        lines.append(f'  "{key}": [')
        lines += [f"    {json.dumps(r)}," for r in rows]
        lines[-1] = lines[-1].rstrip(",")
        lines.append("  ],")
    lines[-1] = "  ]"
    lines.append("}")
    text = "\n".join(lines) + "\n"
    json.loads(text)
    PIN.write_text(text, encoding="utf-8")
    print(
        f"pinned {len(trees)} trees and {len(rejected)} refusals"
        f" on duckdb {duckdb.__version__} -> {PIN}"
    )


if __name__ == "__main__":
    main()
