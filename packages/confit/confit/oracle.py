"""THE ORACLE: DuckDB with the optimizer off, behind one constructor.

Anything that compares this engine against DuckDB -- tests, corpus gates, the
differential fuzzer -- gets its connection from here, so the oracle is a
property of the REPO rather than a per-call-site choice that can be forgotten.

Why optimizer-off is the oracle at all, measured 2026-08-17:

  * `PRAGMA disable_optimizer` == disabling all 33 named optimizers.
  * The BINDER is untouched, so output TYPES are identical (checked across
    narrow ints and decimals), constant folding still happens (`1 + 2` is
    int32 3), and bind-time constant errors still fire.
  * Execution-level LAZINESS is untouched: an untaken CASE arm, AND/OR
    short-circuit in both operand orders, and coalesce's later arguments all
    behave exactly as with the optimizer on.
  * What is removed is the plan rewriting -- statistics_propagation,
    expression_rewriter, filter pushdown/pullup, CSE, join reordering.

So the oracle is "the query as written, run by DuckDB's execution model, with
no plan rewriting" -- the same shape as this engine, and that is what makes it
matchable. The optimizer-on reading is NOT matchable in principle:
`statistics_propagation` reads a column's null statistic, so it answers the
same query over the same rows differently depending on the table's insert
history (proved in tests/known_divergences/test_trap_elision.py).

The oracle's tables are NATIVE tables, never registered arrow relations --
see `load`, where the difference is a semantic one, not a convenience.

`_raw_connect` is captured at import, before any test patches `duckdb`, and
every connection here goes through it. The connection is per-instance for the
same reason the module never assigns to `duckdb.connect`: `duckdb` is a shared
module, so an import-time assignment leaks into every other package's tests
for the rest of the session. It did once -- sql_transform's
single-evaluation tests count sklearn calls made through DuckDB, and losing
CSE doubled them.

A caller that WANTS the optimizer -- because it is documenting what the
optimizer does -- says so in its own body with `optimizer_on()`, which reads
as the deliberate exception it is.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import duckdb
import pyarrow as pa

_raw_connect = duckdb.connect

# Identical to the copies in the corpus and dialect gates: a mined setup can
# record two CREATEs for one table when the file's drop directive was skipped.
_CREATE_TABLE = re.compile(r'\s*CREATE\s+TABLE\s+"?([A-Za-z_]\w*)"?', re.IGNORECASE)


@dataclass(frozen=True)
class Trap:
    """A run the oracle refused: the exception's class name and message."""

    kind: str
    message: str

    def __str__(self) -> str:
        return f"{self.kind}: {self.message}"


class Oracle:
    """One optimizer-off DuckDB connection, with the setup verbs the
    comparisons need. Anything else is reached through the connection itself:
    unknown attributes forward to it, so the escape hatch is always open and
    no wrapper method has to be invented for it."""

    # Recorded, not asserted. Whether the gate pins ==VERSION or a floor is
    # the owner's ruling (oracle spec ASK-1); it lands as one assert here.
    VERSION = "1.5.5"

    Error = duckdb.Error

    def __init__(self) -> None:
        self.con = _raw_connect()
        self.con.execute("PRAGMA disable_optimizer")

    def __getattr__(self, name: str):
        # `con` arriving here means __init__ never finished; forwarding would
        # recurse forever, so let it surface as the AttributeError it is.
        if name == "con":
            raise AttributeError(name)
        return getattr(self.con, name)

    def table(self, name: str, coldefs: str, rows=()) -> Oracle:
        """Create a table from its SQL declaration and fill it in one
        executemany.

        `coldefs` stays SQL text on purpose: the declaration is often the
        measurement. A CTAS from the same values would silently drop NOT NULL
        (measured), so a caller that needs the constraint -- or an exact width,
        or a DEFAULT -- writes it and gets it.
        """
        self.con.execute(f'CREATE TABLE "{name}" ({coldefs})')
        if rows:
            marks = ", ".join(["?"] * len(rows[0]))
            # S608: the table name is the caller's own fixture name, and the
            # values go through placeholders.
            self.con.executemany(f'INSERT INTO "{name}" VALUES ({marks})', rows)  # noqa: S608
        return self

    def load(self, name: str, arrow_table: pa.Table) -> Oracle:
        """Materialize an arrow table as a NATIVE table.

        The CTAS is load-bearing, not a convenience: DuckDB pushes constant
        filters into registered-arrow scans with IEEE NaN semantics, which
        disagrees with its own native-table comparison order. A bare
        `register` is therefore a DIFFERENT oracle, and the engine follows the
        native-table one. Column widths survive the copy; NOT NULL does not.
        """
        alias = f"__arrow_{name}"
        self.con.register(alias, arrow_table)
        try:
            ddl = f'CREATE TABLE "{name}" AS SELECT * FROM "{alias}"'  # noqa: S608
            self.con.execute(ddl)
        finally:
            self.con.unregister(alias)
        return self

    def replay_setup(self, stmts) -> Oracle:
        """Execute a mined case's setup statements in order.

        A CREATE TABLE that hits 'already exists' is dropped and retried once,
        which is exactly what the source file did: its drop directive was
        skipped by the miner's line parser, so both CREATEs were recorded. Any
        other failure raises -- a case whose tables cannot be built has nothing
        to compare.
        """
        for stmt in stmts:
            try:
                self.con.execute(stmt)
            except duckdb.CatalogException as e:
                m = _CREATE_TABLE.match(stmt)
                if m and "already exists" in str(e):
                    self.con.execute(f'DROP TABLE "{m.group(1)}"')
                    self.con.execute(stmt)
                else:
                    raise
        return self

    def answer(self, sql: str) -> pa.Table:
        """The oracle's answer, raw.

        Nothing is normalized here -- not column names, not row order, not
        types. Duplicate column names survive because DuckDB emits them, and a
        caller comparing against this decides its own equality. A change to
        this method IS the oracle moving.
        """
        return self.con.execute(sql).to_arrow_table()

    def try_answer(self, sql: str) -> pa.Table | Trap:
        """`answer`, with a refusal returned instead of raised.

        UnicodeDecodeError counts as a refusal alongside duckdb.Error: RE2's
        `\\C` can serve raw bytes from inside a multibyte character, so DuckDB
        emits invalid UTF-8 that the client cannot decode. That is a property
        of the oracle, not the quirk of one test.
        """
        try:
            return self.answer(sql)
        except (duckdb.Error, UnicodeDecodeError) as e:
            return Trap(type(e).__name__, str(e))

    def catalog(self) -> list[tuple[str, list[tuple[str, str, bool]]]]:
        """Every table in main as (name, [(column, dtype, nullable)]), in the
        order DuckDB lists them."""
        tables = [
            r[0]
            for r in self.con.execute(
                "SELECT table_name FROM duckdb_tables() WHERE schema_name = 'main'"
            ).fetchall()
        ]
        return [
            (
                t,
                [
                    (name, dtype, nullable == "YES")
                    for name, dtype, nullable, *_ in self.con.execute(
                        f'DESCRIBE "{t}"'
                    ).fetchall()
                ],
            )
            for t in tables
        ]

    def optimizer_on(self) -> Oracle:
        """Turn the optimizer back on IN PLACE, for a caller documenting what
        the optimizer does.

        In place, on this same connection, on purpose: the two readings of a
        differential comparison must share one connection, or a table's insert
        history gives statistics_propagation different statistics to read and
        the difference masquerades as an optimizer effect.
        """
        self.con.execute("PRAGMA enable_optimizer")
        return self

    def __enter__(self) -> Oracle:
        return self

    def __exit__(self, *exc) -> None:
        self.con.close()
