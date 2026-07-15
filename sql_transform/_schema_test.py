"""Tests for pydantic model synthesis (__THIS__ schema)."""

import pyarrow as pa

from sql_transform._schema import synthesize_this_model


def test_synthesize_this_model_basic_types():
    schema = pa.schema(
        [
            pa.field("age", pa.int64(), nullable=False),
            pa.field("score", pa.float64(), nullable=False),
            pa.field("name", pa.string(), nullable=True),
            pa.field("active", pa.bool_(), nullable=False),
        ]
    )
    model = synthesize_this_model(schema)

    assert model.model_fields["age"].annotation is int
    assert model.model_fields["score"].annotation is float
    assert model.model_fields["name"].annotation == (str | None)
    assert model.model_fields["active"].annotation is bool


def test_synthesize_this_model_instantiates_from_values():
    schema = pa.schema([pa.field("age", pa.int64(), nullable=False)])
    model = synthesize_this_model(schema)
    instance = model(age=30)
    assert instance.age == 30


def test_synthesize_this_model_nullable_field_accepts_none():
    schema = pa.schema([pa.field("name", pa.string(), nullable=True)])
    model = synthesize_this_model(schema)
    instance = model(name=None)
    assert instance.name is None
