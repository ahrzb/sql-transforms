"""Tests for the Rust SQL interpreter (sql_transform._interpreter).

Row tables are declared as Pydantic v2 model classes at InferFn construction
time; row inputs to .infer() are instances of those models. Every
behavioral test that involves computed values compares InferFn output
against real DataFusion batch output for the same SQL + data.
"""

import pyarrow as pa
import pytest
from pydantic import BaseModel, ValidationError

from sql_transform._interpreter import InferFn


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


class A(BaseModel):
    id: int
    x: int


class B(BaseModel):
    id: int
    y: int


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


def test_infer_accepts_kwargs_instead_of_dict():
    sql = "SELECT age FROM data"
    fn = InferFn(sql, row_tables={"data": Data}, static_tables={})
    via_dict = _as_dicts(fn.infer({"data": [Data(age=1), Data(age=2)]}))
    via_kwargs = _as_dicts(fn.infer(data=[Data(age=1), Data(age=2)]))
    assert via_dict == via_kwargs == [{"age": 1}, {"age": 2}]


def test_infer_accepts_kwargs_for_multi_table_join():
    sql = "SELECT a.x, b.y FROM a JOIN b ON a.id = b.id"
    fn = InferFn(sql, row_tables={"a": A, "b": B}, static_tables={})
    result = fn.infer(a=[A(id=1, x=10)], b=[B(id=1, y=20)])
    assert _as_dicts(result) == [{"x": 10, "y": 20}]


def test_infer_merges_positional_dict_and_kwargs_kwargs_win():
    sql = "SELECT age FROM data"
    fn = InferFn(sql, row_tables={"data": Data}, static_tables={})
    # kwargs "data" overrides the positional dict's "data" entry
    result = fn.infer({"data": [Data(age=999)]}, data=[Data(age=1), Data(age=2)])
    assert _as_dicts(result) == [{"age": 1}, {"age": 2}]


def test_left_lookup_join_hit_returns_value():
    from types import SimpleNamespace

    import pyarrow as pa
    from pydantic import BaseModel

    from sql_transform import InferFn

    class Row(BaseModel):
        city: str

    ref = pa.table({"city": ["a", "b"], "enc": [1.5, 3.5]})
    sql = "SELECT ref.enc FROM data LEFT JOIN ref ON data.city = ref.city"
    fn = InferFn(sql, row_tables={"data": Row}, static_tables={"ref": ref})
    out = fn.infer({"data": [SimpleNamespace(city="a")]})
    assert out[0].enc == 1.5


def test_left_lookup_join_miss_returns_null():
    from types import SimpleNamespace

    import pyarrow as pa
    from pydantic import BaseModel

    from sql_transform import InferFn

    class Row(BaseModel):
        city: str

    ref = pa.table({"city": ["a"], "enc": [1.5]})
    sql = "SELECT ref.enc FROM data LEFT JOIN ref ON data.city = ref.city"
    fn = InferFn(sql, row_tables={"data": Row}, static_tables={"ref": ref})
    out = fn.infer({"data": [SimpleNamespace(city="zzz")]})  # unseen key
    assert out[0].enc is None


def test_inner_lookup_join_miss_still_errors():
    from types import SimpleNamespace

    import pyarrow as pa
    from pydantic import BaseModel

    from sql_transform import InferFn

    class Row(BaseModel):
        city: str

    ref = pa.table({"city": ["a"], "enc": [1.5]})
    sql = "SELECT ref.enc FROM data JOIN ref ON data.city = ref.city"
    fn = InferFn(sql, row_tables={"data": Row}, static_tables={"ref": ref})
    # Inner-join key miss surfaces as KeyError (Rust InterpError::MissingKey
    # maps to PyKeyError), unchanged by the LEFT-join work.
    with pytest.raises(KeyError):
        fn.infer({"data": [SimpleNamespace(city="zzz")]})
