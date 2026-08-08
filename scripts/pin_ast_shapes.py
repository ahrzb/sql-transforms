"""Re-pin `model/_shapes.json` against the installed DuckDB.

Not a code generator — the classes in `model/_nodes.py` are hand-written. This
only writes down what `json_serialize_sql` emits, for every tag the corpus
reaches, so that a version bump surfaces as a reviewable diff instead of as a
wrong answer three layers down.

    uv run python scripts/pin_ast_shapes.py
    git diff        # this diff IS the drift report

The corpus is imported from the test rather than restated, so the pin and the
gate cannot drift apart.
"""

import json
from pathlib import Path

import duckdb
from sql_transform.model._nodes_test import FIELDS_IN_CORPUS, PARSEABLE

MANIFEST = (
    Path(__file__).resolve().parent.parent
    / "packages/sql-transform/sql_transform/model/_shapes.json"
)


def main() -> None:
    manifest = {
        "duckdb": duckdb.__version__,
        "statements": len(PARSEABLE),
        "shapes": {k: sorted(v) for k, v in sorted(FIELDS_IN_CORPUS.items())},
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        f"pinned {len(manifest['shapes'])} shapes from {len(PARSEABLE)} statements"
        f" on duckdb {duckdb.__version__} -> {MANIFEST}"
    )


if __name__ == "__main__":
    main()
