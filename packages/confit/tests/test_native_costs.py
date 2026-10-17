"""Fixed downstream shapes: ratios share a process and run without xdist."""

import statistics
import time

import pyarrow as pa
import pytest
from confit import DuckDBInferFn, SqlFunction
from confit import sql as S

F64 = pa.float64()


def _build(fn, takes, fields):
    call = f"{fn.name}({', '.join(takes.names)})"
    sql = "SELECT " + ", ".join(f"{call}.f{j} AS f{j}" for j in range(fields))
    return DuckDBInferFn(
        sql + " FROM __THIS__",
        row_tables={"__THIS__": takes},
        static_tables={},
        udfs=[fn],
    )


def _median_call(fn, rows):
    fn.infer_arrow(rows)
    samples = []
    for _ in range(9):
        start = time.perf_counter()
        for _ in range(10):
            fn.infer_arrow(rows)
        samples.append((time.perf_counter() - start) / 10)
    return statistics.median(samples)


@pytest.mark.cost
def test_a_shared_value_read_many_times_serves_like_plain_reads():
    # #459: branch-based coalesce copied the live stack at every read.
    takes = pa.schema([(f"x{i}", F64) for i in range(256)])
    rows = pa.table({name: [float(i + 1)] * 64 for i, name in enumerate(takes.names)})

    def build(coalesced):
        def body(*xs):
            values = (
                [S.fn("coalesce", x, S.lit(float("nan"))) for x in xs]
                if coalesced
                else xs
            )
            return {
                f"f{j}": sum((x * S.lit(1.0 + j / 16) for x in values[1:]), values[0])
                for j in range(16)
            }

        fn = SqlFunction(
            "reads", takes, pa.struct([(f"f{j}", F64) for j in range(16)]), body
        )
        return _build(fn, takes, 16)

    plain, shared = build(False), build(True)
    assert shared.infer_arrow(rows).equals(plain.infer_arrow(rows))
    baseline = _median_call(plain, rows)
    took = _median_call(shared, rows)
    print(
        f"shared reads: plain={baseline:.6f}s coalesce={took:.6f}s "
        f"ratio={took / baseline:.3f}"
    )
    assert took < 2 * baseline, (baseline, took)


@pytest.mark.cost
def test_a_guard_read_per_field_builds_like_an_unguarded_call():
    # #430: the same rejection check was compiled again at every field read.
    takes = pa.schema([(f"x{i}", F64) for i in range(64)])

    def build(guarded):
        def body(*xs):
            fields = {f"f{j}": xs[j % 64] * S.lit(1.0 + j / 512) for j in range(512)}
            if guarded:
                bad = S.fn("greatest", *[S.fn("abs", x) for x in xs]) > S.lit(1e300)
                fields["f0"] = S.case(bad, S.fn("error", S.lit("rejected"))).otherwise(
                    fields["f0"]
                )
            return fields

        fn = SqlFunction(
            "guarded",
            takes,
            pa.struct([(f"f{j}", F64) for j in range(512)]),
            body,
        )
        start = time.perf_counter()
        infer = _build(fn, takes, 512)
        return time.perf_counter() - start, infer

    plain, guarded = [], []
    for _ in range(3):
        baseline, _ = build(False)
        took, infer = build(True)
        plain.append(baseline)
        guarded.append(took)
    rows = pa.table({name: [1.0, 1e301] for name in takes.names})
    assert infer.infer_arrow(rows.slice(0, 1)).to_pylist()[0]["f0"] == 1.0
    with pytest.raises(Exception, match="rejected"):
        infer.infer_arrow(rows.slice(1, 1))
    baseline, took = statistics.median(plain), statistics.median(guarded)
    print(
        f"guard build: plain={baseline:.6f}s guarded={took:.6f}s "
        f"ratio={took / baseline:.3f}"
    )
    assert took < 2.5 * baseline, (baseline, took)
