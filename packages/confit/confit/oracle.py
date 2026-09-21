"""The fixed DuckDB reference, behind one optimizer-off constructor.

Tests, corpus gates, and the differential fuzzer obtain reference connections
here. Opening against another DuckDB version fails before connecting.

Disabling the optimizer removes its plan-rewrite passes, including
statistics-driven rewrites that can elide traps. It does not disable binding
or guarantee a unique answer for unordered or order-sensitive computations.
The oracle contract defines those comparison boundaries.

Tables are materialized as native tables: registered Arrow scans can have
different comparison behavior (see `load`). A diagnostic optimizer-on reading
uses `optimizer_on()` on the same connection, preserving loaded table state.

Capture `_raw_connect` without replacing `duckdb.connect` globally: other
packages need ordinary DuckDB behavior, and a global patch once changed
sql_transform's sklearn call counts by removing common-subexpression elimination.
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

    # Keep this aligned with the reviewed dev-environment pin. Use a runtime
    # check, not assert: optimized Python must not silently change the oracle.
    VERSION = "1.5.5"

    Error = duckdb.Error

    def __init__(self) -> None:
        if duckdb.__version__ != self.VERSION:
            raise RuntimeError(
                f"the oracle is DuckDB {self.VERSION}, but this interpreter "
                f"imports duckdb {duckdb.__version__}. Run inside the pinned "
                "environment (`uv sync --locked --group spark` at the repo root). "
                "Changing the reference version requires a separate reviewed decision."
            )
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
