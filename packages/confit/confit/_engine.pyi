from typing import Any

import pyarrow as pa
from pydantic import BaseModel

BUILD_PROFILE: str
""""debug" or "release" — benchmarks refuse a debug build."""

class DuckDBInferFn:
    """SQL specialized against frozen static tables, served bit-exact with DuckDB.

    Construction either succeeds with a function that matches DuckDB
    bit-for-bit — *with the declared `udfs` registered*, when there are any —
    or raises `ValueError` naming the unsupported construct. There is no
    third mode.
    """

    output_model: type[BaseModel] | None

    def __init__(
        self,
        sql: str,
        row_tables: dict[str, type[BaseModel]],
        static_tables: dict[str, pa.Table],
        udfs: list[Any] | None = None,
        output_model: type[BaseModel] | None = None,
        output: str | None = None,
        shape: str | None = None,
    ) -> None:
        """`udfs`: declared opaque scalar functions the SQL may call
        (DRAFT-22). Each object carries `name: str`, a `takes: pa.Schema`
        (one field per argument, names and types together, in call order) and
        a `returns: pa.DataType`, plus a scalar `__call__(*args)` receiving
        one plain Python value per argument (None for NULL) and returning a
        tuple of the output lanes, or None for an all-NULL result.

        The type vocabulary is exactly `bool` / `int64` / `double` / `string`
        — the engine computes in those four, so a narrower arrow type refuses
        rather than widening silently.

        `returns` is the SQL return TYPE, which also says how wide the call is
        and whether its lanes are addressable:

            pa.float64()          an ordinary scalar expression
            pa.struct([...])      width-k with addressable field names
                                  (TASK-63) — struct-valued at EVERY width,
                                  so `f(x).a` reads a lane off ONE call
            pa.list_(t, k)        width-k unnamed (the DRAFT-22 list
                                  boundary); FIXED size, because the width is
                                  part of the declaration, and k >= 2 — a
                                  width-1 list is a scalar and refuses

        Arguments bind against the DECLARED type, not their own: an argument
        widens to it exactly as DuckDB's implicit cast would (BIGINT into a
        declared DOUBLE), and anything else refuses by name. For a tree
        transform that is load-bearing rather than cosmetic — a declared
        BIGINT feature narrows to float32 in ONE step, while a declared DOUBLE
        is cast first and narrows from the double, which is a different leaf
        above 2**53.

        An object with an `instances` attribute is a fitted transformer: its
        implicit leading argument is a nullable BIGINT instance id (never
        written in `takes`).

        A udf name may not be a builtin (`least`, `round`, `upper`, …) and may
        not collide case-insensitively with another declared udf. The builtin
        catalogue is matched before the declared udfs, while DuckDB lets a
        registered function shadow its own builtin — so a collision is refused
        rather than resolved, in either direction.

        An object that also exposes `tree_tables() -> (nodes, models,
        compare_grid)` is a fitted tree ensemble, scored by the native kernel
        instead of a callback — no Python on the row path, and no `__call__`
        is ever made by the engine (it stays the DuckDB-side binding and the
        semantic contract the kernel is gated against). `nodes` columns:
        `model_id` | `tree_id` | `node_id` | `feature` (-1 on a leaf) |
        `threshold` | `left` | `right` | `missing_left` | `value`, grouped by
        model then tree, node ids dense from 0 per tree, children after their
        parent. `models` columns: `model_id` (dense from 0) | `base` | `agg`
        ("sum" | "mean") | `link` ("identity" | "sigmoid"). `compare_grid` is
        "float32" or "float64": which grid the thresholds were fitted on, and
        therefore how an INTEGER feature reaches the comparison. Features bind
        by POSITION, in `takes` order, after the instance id.

        Width-k calls are bare SELECT items emitting one `list | None` field
        — or, when `returns` is a struct, field access over the call binds
        each addressed name to one lane of a single evaluation (`(f(...)).a`
        or `struct_extract(f(...), 'a')`, usable mid-expression; textually
        identical calls share the one evaluation, mirroring DuckDB's CSE).
        The oracle statement is parameterized, not weakened: the engine
        matches DuckDB running the same SQL with these same objects
        registered via `create_function`. UDFs must be deterministic.

        `output`: "model" (typed, default) or "dict".

        `shape`: the output-multiplicity contract, proven at build time --
        "map" (exactly one row out per row in; rejects WHERE and inner joins),
        "filter" (0 or 1, the default), or "many" (0..N; the only shape under
        which join multiplicity will build).
        """

    @property
    def shape(self) -> str:
        """The declared row-shape contract: "map", "filter", or "many"."""

    @property
    def output(self) -> str:
        """The output mode: "model" (typed, default) or "dict"."""

    @property
    def backend(self) -> str:
        """Which engine executes: "cranelift", "interpreter", or "constant"."""

    @property
    def boundary(self) -> str:
        """How rows cross the Python boundary: "marshaller" (generated at
        prepare), "generic" (env-pinned baseline), or "constant"."""

    def infer(
        self,
        tables: dict[str, list[Any]] | None = None,
        **kwargs: list[Any],
    ) -> list[BaseModel]: ...
    def infer_rows(self, rows: list[Any]) -> list[Any]:
        """Row objects in, row objects out -- the low-latency serving path."""

    def infer_arrow(self, batch: pa.Table) -> pa.Table:
        """`pa.Table` in, `pa.Table` out, with no per-value Python objects.

        Faster than `infer_rows` from roughly 1k rows per call; below that the
        fixed pyarrow API cost dominates and the row path wins.
        """
