# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Essential Commands
- `mise run test` or `pytest` - Run all tests
- `mise run fmt` - Format and lint code (ruff check + format)
- `mise run check` - Run all checks (format, lint, test)
- `uv sync` - Install dependencies
- `mise tasks` - List all available tasks

### Running Specific Tests
- `pytest sql_transform/test_core.py::TestParser::test_parse_basic` - Run single test
- `pytest -k "test_parse"` - Run tests matching pattern
- `pytest -x -s -vv` - Stop on first failure, verbose output

### Type Checking & Linting
The project uses ruff for linting/formatting and mypy for type checking (via pytest-mypy).
Always run `mise run fmt` before committing to ensure code quality.

## Architecture Overview

This is a declarative ML preprocessing library that uses SQL syntax to define feature transformations. The core architecture follows this pattern:

```
SQL Query → Parser → AST → Fit Phase → Transform Phase
```

### Key Components

1. **SqlTransformContext** (`sql_transform/context.py`)
   - Central registry for transforms, UDFs, and DataFusion context
   - Factory for creating SQLTransformer instances
   - Manages transform resolution and custom function registration

2. **Parser** (`sql_transform/parser.py`)
   - Converts SQL into internal AST representation
   - Distinguishes between aggregations (built-in), transforms (custom), and window functions
   - Key classes: `AggregateFunction`, `TransformFunction`, `WindowSpecification`

3. **SQLTransformer** (`sql_transform/transformer.py`)
   - Implements scikit-learn-style fit/transform pattern
   - Fit phase: computes aggregations and statistics
   - Transform phase: applies transformations using precomputed values
   - Supports multi-format I/O (pandas, polars, dict, arrow, datafusion)

4. **Data Formats** (`sql_transform/data_formats.py`)
   - Handles conversion between different data formats
   - All processing happens in DataFusion for performance
   - Auto-detects input format and converts output back to same format

### Core Concepts

- **Aggregations**: Built-in SQL functions like `avg()`, `stddev()` that compute statistics during fit
- **Transforms**: Custom functions like `sklearn.standardize()` that need context resolution
- **Window Functions**: Support `OVER (PARTITION BY ...)` syntax for grouped operations
- **Fit/Transform Separation**: Statistics computed once during fit, applied during transform

### SQL Syntax Extensions

The parser supports standard SQL plus ML-specific extensions:
- `sklearn.function_name(args)` - sklearn-style transforms
- `avg(col) OVER (PARTITION BY group_col)` - window functions
- Anonymous functions automatically detected as aggregations or transforms

## Testing Strategy

- `sql_transform/test_core.py` - Consolidated test suite
- Tests cover: basic parsing, transformations, window functions, multi-format I/O
- Optional dependency tests (pandas, polars) skip gracefully if not installed
- Use parametrized tests for SQL parsing validation
- Test files always have the _test suffix, tests for file x.py, will be called x_test.py

## Current Status & Roadmap

**Current (Phase 1)**: Foundation with basic aggregations and window functions
**Next (Phase 2)**: Statistical transforms (standardization, scaling)
**Future**: Categorical encoding, text processing, type system

See VISION.md and TODO.md for detailed roadmap and immediate next steps.

## Development Notes

- Uses `uv` for dependency management
- DataFusion as the core execution engine for performance
- Optional dependencies: sklearn, pandas, polars
- All transforms must support both batch and potential single-record inference
- Maintain backward compatibility with scikit-learn transformer interface