"""Tests for the Rust SQL interpreter (sql_transform._interpreter).

Row tables are declared as Pydantic v2 model classes at InferFn construction
time; row inputs to .infer() are instances of those models. Every
behavioral test that involves computed values compares InferFn output
against real DataFusion batch output for the same SQL + data.
"""

import datafusion
import pyarrow as pa
import pytest
from pydantic import BaseModel, ValidationError

from sql_transform._interpreter import InferFn


def _expected(sql: str, data: dict) -> list[dict]:
    ctx = datafusion.SessionContext()
    ctx.from_pydict(data, name="data")
    return ctx.sql(sql).collect()[0].to_pylist()


def _as_dicts(rows: list) -> list[dict]:
    """`.infer()` now returns synthesized-model instances, not dicts —
    convert back to dicts to compare against real DataFusion output."""
    return [r.model_dump() for r in rows]


class Data(BaseModel):
    age: int
    name: str | None = None


def test_module_imports_and_constructs():
    fn = InferFn("SELECT age FROM data", row_tables={"data": Data}, static_tables={})
    assert fn is not None


def test_column_pass_through():
    sql = "SELECT age FROM data"
    fn = InferFn(sql, row_tables={"data": Data}, static_tables={})
    actual = fn.infer({"data": [Data(age=30)]})
    assert _as_dicts(actual) == _expected(sql, {"age": [30]})


def test_arithmetic_and_where():
    sql = "SELECT age, age * 2 AS doubled FROM data WHERE age > 18"
    fn = InferFn(sql, row_tables={"data": Data}, static_tables={})
    actual = fn.infer({"data": [Data(age=15), Data(age=25), Data(age=40)]})
    assert _as_dicts(actual) == _expected(sql, {"age": [15, 25, 40]})


def test_builtin_function_and_cast():
    sql = "SELECT UPPER(name) AS n, CAST(age AS VARCHAR) AS s FROM data"
    fn = InferFn(sql, row_tables={"data": Data}, static_tables={})
    actual = fn.infer({"data": [Data(age=30, name="alice")]})
    assert _as_dicts(actual) == _expected(sql, {"age": [30], "name": ["alice"]})


class A(BaseModel):
    id: int
    x: int


class B(BaseModel):
    id: int
    y: int


def test_cross_join():
    sql = "SELECT a.x, b.y FROM a, b"
    ctx = datafusion.SessionContext()
    ctx.from_pydict({"id": [1], "x": [10]}, name="a")
    ctx.from_pydict({"id": [1], "y": [20]}, name="b")
    expected = ctx.sql(sql).collect()[0].to_pylist()

    fn = InferFn(sql, row_tables={"a": A, "b": B}, static_tables={})
    actual = fn.infer({"a": [A(id=1, x=10)], "b": [B(id=1, y=20)]})
    assert _as_dicts(actual) == expected


def test_inner_join():
    sql = "SELECT a.x, b.y FROM a JOIN b ON a.id = b.id"
    ctx = datafusion.SessionContext()
    ctx.from_pydict({"id": [1, 2], "x": [10, 20]}, name="a")
    ctx.from_pydict({"id": [1, 2], "y": [100, 200]}, name="b")
    expected = ctx.sql(sql).collect()[0].to_pylist()

    fn = InferFn(sql, row_tables={"a": A, "b": B}, static_tables={})
    actual = fn.infer(
        {
            "a": [A(id=1, x=10), A(id=2, x=20)],
            "b": [B(id=1, y=100), B(id=2, y=200)],
        }
    )
    assert _as_dicts(actual) == expected


def test_aliased_row_table():
    sql = "SELECT d.age FROM data AS d WHERE d.age > 18"
    fn = InferFn(sql, row_tables={"data": Data}, static_tables={})
    actual = fn.infer({"data": [Data(age=15), Data(age=25)]})
    assert _as_dicts(actual) == _expected(sql, {"age": [15, 25]})


def test_join_row_and_static_table():
    ref_table = pa.table({"id": [1, 2], "y": [10, 20]})
    sql = "SELECT data.x, ref.y FROM data JOIN ref ON data.id = ref.id"

    class RowWithId(BaseModel):
        id: int
        x: int

    ctx = datafusion.SessionContext()
    ctx.from_pydict({"id": [1, 2], "x": [5, 6]}, name="data")
    ctx.from_arrow(ref_table, name="ref")
    expected = ctx.sql(sql).collect()[0].to_pylist()

    fn = InferFn(sql, row_tables={"data": RowWithId}, static_tables={"ref": ref_table})
    actual = fn.infer({"data": [RowWithId(id=1, x=5), RowWithId(id=2, x=6)]})
    assert _as_dicts(actual) == expected


def test_error_unknown_row_column():
    sql = "SELECT nonexistent FROM data"
    with pytest.raises(ValueError):
        InferFn(sql, row_tables={"data": Data}, static_tables={})


def test_error_unknown_static_column():
    ref_table = pa.table({"id": [1], "y": [10]})
    sql = "SELECT data.age, ref.nonexistent FROM data JOIN ref ON data.age = ref.id"
    with pytest.raises(ValueError):
        InferFn(sql, row_tables={"data": Data}, static_tables={"ref": ref_table})


def test_error_self_join_still_rejected():
    sql = "SELECT a.x FROM a JOIN a ON a.id = a.id"
    with pytest.raises(ValueError):
        InferFn(sql, row_tables={"a": A}, static_tables={})


def test_output_model_is_synthesized_and_typed():
    sql = "SELECT age, age * 2 AS doubled FROM data"
    fn = InferFn(sql, row_tables={"data": Data}, static_tables={})

    assert set(fn.output_model.model_fields) == {"age", "doubled"}
    assert fn.output_model.model_fields["age"].annotation is int
    assert fn.output_model.model_fields["doubled"].annotation is int

    results = fn.infer({"data": [Data(age=30)]})
    assert len(results) == 1
    assert isinstance(results[0], fn.output_model)
    assert results[0].age == 30
    assert results[0].doubled == 60


def test_output_model_nullable_column():
    sql = "SELECT name FROM data"
    fn = InferFn(sql, row_tables={"data": Data}, static_tables={})
    assert fn.output_model.model_fields["name"].annotation == (str | None)

    results = fn.infer({"data": [Data(age=1, name=None)]})
    assert results[0].name is None


def test_output_model_cast_type_is_exact():
    sql = "SELECT CAST(age AS VARCHAR) AS s FROM data"
    fn = InferFn(sql, row_tables={"data": Data}, static_tables={})
    assert fn.output_model.model_fields["s"].annotation is str


def test_output_model_division_promotes_to_float():
    sql = "SELECT age / 2 AS half FROM data"
    fn = InferFn(sql, row_tables={"data": Data}, static_tables={})
    # age (int) / 2 (int literal) -> Int per the truncating-int-division rule
    assert fn.output_model.model_fields["half"].annotation is int


def test_output_model_comparison_is_bool():
    sql = "SELECT age > 18 AS is_adult FROM data"
    fn = InferFn(sql, row_tables={"data": Data}, static_tables={})
    assert fn.output_model.model_fields["is_adult"].annotation is bool


class Result(BaseModel):
    age: int
    doubled: int


def test_user_supplied_output_model_compatible():
    sql = "SELECT age, age * 2 AS doubled FROM data"
    fn = InferFn(sql, row_tables={"data": Data}, static_tables={}, output_model=Result)
    assert fn.output_model is Result
    results = fn.infer({"data": [Data(age=30)]})
    assert isinstance(results[0], Result)
    assert results[0] == Result(age=30, doubled=60)


def test_user_supplied_output_model_widening_int_to_float_ok():
    class WideResult(BaseModel):
        age: float
        doubled: int

    sql = "SELECT age, age * 2 AS doubled FROM data"
    fn = InferFn(
        sql, row_tables={"data": Data}, static_tables={}, output_model=WideResult
    )
    results = fn.infer({"data": [Data(age=30)]})
    assert results[0].age == 30.0


def test_user_supplied_output_model_missing_field():
    class Incomplete(BaseModel):
        age: int
        # missing "doubled"

    sql = "SELECT age, age * 2 AS doubled FROM data"
    with pytest.raises(ValueError):
        InferFn(
            sql, row_tables={"data": Data}, static_tables={}, output_model=Incomplete
        )


def test_user_supplied_output_model_extra_field():
    class TooMany(BaseModel):
        age: int
        doubled: int
        unrelated: str

    sql = "SELECT age, age * 2 AS doubled FROM data"
    with pytest.raises(ValueError):
        InferFn(sql, row_tables={"data": Data}, static_tables={}, output_model=TooMany)


def test_user_supplied_output_model_provably_wrong_base_type():
    class WrongType(BaseModel):
        age: str  # query produces Int, str is not provably compatible

    sql = "SELECT age FROM data"
    with pytest.raises(ValueError):
        InferFn(
            sql, row_tables={"data": Data}, static_tables={}, output_model=WrongType
        )


def test_user_supplied_output_model_non_nullable_but_inferred_nullable_defers_to_runtime():  # noqa: E501
    class NonNullResult(BaseModel):
        name: str  # declared non-nullable; `name` is Optional[str] on Data

    sql = "SELECT name FROM data"
    # Build succeeds — we can't PROVE `name` will be null, only that we
    # can't prove it won't.
    fn = InferFn(
        sql, row_tables={"data": Data}, static_tables={}, output_model=NonNullResult
    )
    # A row that actually produces None fails at infer() time, not build time.
    with pytest.raises(ValidationError):
        fn.infer({"data": [Data(age=1, name=None)]})
    # A row with a real value works fine.
    result = fn.infer({"data": [Data(age=1, name="alice")]})
    assert result[0].name == "alice"


def test_duck_typed_row_instance_works():
    """A structurally-compatible instance of a DIFFERENT class than the one
    declared in row_tables still works — no isinstance check, just getattr."""

    class NotData:
        def __init__(self, age):
            self.age = age

    sql = "SELECT age FROM data"
    fn = InferFn(sql, row_tables={"data": Data}, static_tables={})
    actual = fn.infer({"data": [NotData(age=42)]})
    assert actual[0].age == 42


def test_row_instance_missing_referenced_attribute_raises_clear_error():
    class Incomplete:
        pass

    sql = "SELECT age FROM data"
    fn = InferFn(sql, row_tables={"data": Data}, static_tables={})
    with pytest.raises(ValueError, match="age"):
        fn.infer({"data": [Incomplete()]})


def test_unused_model_fields_are_never_touched():
    """A row model can have fields the query doesn't reference — getattr
    only pulls the columns actually used, so an unrelated/broken field on
    the instance is never even accessed."""
    from pydantic import ConfigDict

    class Poison:
        @property
        def unused(self):
            raise RuntimeError("should never be accessed")

    class WithExtra(BaseModel):
        model_config = ConfigDict(arbitrary_types_allowed=True)
        age: int
        unused: object = None

    sql = "SELECT age FROM data"
    fn = InferFn(sql, row_tables={"data": WithExtra}, static_tables={})
    row = WithExtra(age=5, unused=Poison())
    actual = fn.infer({"data": [row]})
    assert actual[0].age == 5


def test_literal_only_projection_preserves_row_count():
    """A projection that references NO columns of the row table still
    preserves the row count — exercises the row-conversion fallback path
    where `row_table_columns.get(table)` yields no columns to read."""
    sql = "SELECT 1 AS marker FROM data"
    fn = InferFn(sql, row_tables={"data": Data}, static_tables={})
    actual = fn.infer({"data": [Data(age=1), Data(age=2), Data(age=3)]})
    assert len(actual) == 3
    assert all(r.marker == 1 for r in actual)


def test_nested_object_passthrough_through_output_model():
    """A real (non-scalar) value flows through unchanged via the
    model_validate-based output conversion — Base::Other -> typing.Any,
    end-to-end with actual data."""
    from pydantic import ConfigDict

    class WithPayload(BaseModel):
        model_config = ConfigDict(arbitrary_types_allowed=True)
        age: int
        payload: object

    sql = "SELECT payload FROM data"
    fn = InferFn(sql, row_tables={"data": WithPayload}, static_tables={})
    payload = {"nested": [1, 2, 3]}
    result = fn.infer({"data": [WithPayload(age=1, payload=payload)]})
    assert result[0].payload == payload
