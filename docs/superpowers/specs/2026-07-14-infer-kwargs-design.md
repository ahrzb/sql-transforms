# `infer()` Kwargs Support — Design Spec

## Goal

Let `InferFn.infer()` accept table data as keyword arguments, in addition to the existing positional dict, so `fn.infer(data=[Data(...)])` works alongside `fn.infer({"data": [Data(...)]})`.

## Motivation

The current `infer(tables: dict[str, list[BaseModel]])` requires wrapping every call in an explicit dict literal even for the common case of a single or a few tables. Kwargs are more idiomatic Python for this shape.

## Public API

```python
fn.infer({"data": [Data(age=1), Data(age=2)]})   # unchanged
fn.infer(data=[Data(age=1), Data(age=2)])          # new
fn.infer(a=[A(id=1)], b=[B(id=1)])                  # new, multi-table
```

- `infer(self, tables: dict[str, list[BaseModel]] | None = None, **kwargs: list[BaseModel]) -> list[BaseModel]`
- Each kwarg's value is a `list[BaseModel]`, same shape as the dict form's values — this is not single-row sugar, it's an alternate way to write the same structure.
- **Merge rule:** if both `tables` and kwargs are given, they're merged as `{**tables, **kwargs}` — kwargs win on a key collision. No error path for "both given"; this matches ordinary Python dict-merge semantics and needs no extra validation.
- No other part of `InferFn` changes — `__init__`, `output_model`, error taxonomy, and row/output conversion are all untouched. This is a single-method, additive change to `infer()`'s calling convention only.

## Non-Goals

- No `infer_one()` method (considered during design, dropped — the single-row convenience didn't pull its weight against a plain kwargs shorthand).
- No single-instance-as-sugar-for-one-row-list behavior on `infer()`'s kwargs — a kwarg value must be a list, exactly like the dict form.

## Testing Strategy

- `infer(data=[...])` produces identical output to `infer({"data": [...]})` for the same data.
- Multi-table kwargs (`infer(a=[...], b=[...])`) work for a JOIN query.
- Positional `tables` dict + kwargs together merge correctly, with a kwarg overriding a same-named key in `tables`.
- Existing dict-only calls (all current tests) continue to pass unmodified.
