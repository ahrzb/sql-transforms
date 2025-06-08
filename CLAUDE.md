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
- `pytest sql_transform/parser_test.py::TestParser::test_parse_basic` - Run single test
- `pytest -k "test_parse"` - Run tests matching pattern
- `pytest -x -s -vv` - Stop on first failure, verbose output

### Type Checking & Linting
The project uses ruff for linting/formatting and mypy for type checking (via pytest-mypy).
Always run `mise run fmt` before committing to ensure code quality.

## Development Workflow
- Always run `mise run fmt` after making changes to ensure consistent formatting
- All tests must pass before committing changes
- Follow the test naming convention: tests for `x.py` go in `x_test.py`

## Architecture Overview

This is a declarative ML preprocessing library that uses SQL syntax to define feature transformations. The core architecture follows this pattern:

```
SQL Query → Parser → AST → Fit Phase → Transform Phase
```

### Key Components

1. **SqlTransformContext** (`sql_transform/context.py`)
   - Central registry for aggregations, transforms, and DataFusion context
   - Factory for creating SQLTransformer instances
   - Manages function resolution through extensible registry system
   - Resolves SQL function names to either AggregateFunction or TransformFunction

2. **Function Registry** (`sql_transform/function_registry.py`)
   - **AggregationRegistry**: Built-in DataFusion aggregations (avg, sum, count, etc.)
   - **TransformRegistry**: Custom transforms with fit/transform pattern
   - **FunctionResolver**: Routes function names to appropriate registry
   - Supports sklearn integration via `SklearnTransformSpec`

3. **Parser** (`sql_transform/parser.py`)
   - Converts SQL into internal AST representation using context-based function resolution
   - Key classes: `AggregateFunction`, `TransformFunction`, `WindowSpecification`
   - No hardcoded function mappings - all functions resolved via context
   - Supports window functions with DataFusion integration

4. **SQLTransformer** (`sql_transform/transformer.py`)
   - Implements scikit-learn-style fit/transform pattern
   - Fit phase: computes aggregations and fits transforms using BFS dependency resolution
   - Transform phase: applies precomputed values and fitted transforms
   - Supports multi-format I/O (pandas, polars, dict, arrow, datafusion)

5. **Data Formats** (`sql_transform/data_formats.py`)
   - Handles conversion between different data formats
   - All processing happens in DataFusion for performance
   - Auto-detects input format and converts output back to same format

6. **Sklearn Integration** (`sql_transform/sklearn_integration.py`)
   - Optional sklearn transformer integration when sklearn is available
   - Maps SQL function names to sklearn classes (e.g., `sklearn.standardize` → `StandardScaler`)
   - Graceful degradation when sklearn is not installed

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

### Test File Organization
- **Component-specific tests**: `parser_test.py`, `transformer_test.py`, `context_test.py`, `data_formats_test.py`
- **Integration tests**: `sklearn_integration_test.py` for end-to-end sklearn functionality
- **Naming convention**: Tests for `x.py` are in `x_test.py` (not `test_x.py`)

### Test Coverage
- **Parser tests**: SQL parsing, function resolution, window functions
- **Transformer tests**: Fit/transform patterns, multi-format I/O, nested aggregations
- **Context tests**: Function registration, resolution, transformer creation
- **Data format tests**: Conversion between pandas, polars, arrow, dict formats
- **Optional dependency tests**: Skip gracefully when pandas/polars/sklearn unavailable

### Testing Best Practices
- Use parametrized tests for SQL parsing validation
- Test both positive and negative cases
- Verify type checking passes with mypy integration

## Current Status & Roadmap

### **Completed (Phase 1)**
- ✅ Foundation with extensible function registry
- ✅ Context-based aggregation and transform resolution
- ✅ Window functions with DataFusion integration
- ✅ Multi-format I/O (pandas, polars, dict, arrow)
- ✅ BFS dependency resolution for nested aggregations
- ✅ Sklearn transformer integration framework
- ✅ Comprehensive test suite with proper organization

### **Current Focus (Phase 2)**
- Basic statistical transforms (standardization, scaling)
- Enhanced sklearn integration with parameter mapping
- Performance optimization for large datasets

### **Future Phases**
- Categorical encoding (one-hot, ordinal, target encoding)
- Text processing (TF-IDF, tokenization)
- Advanced type system with validation
- Custom UDF registration API

See VISION.md and TODO.md for detailed roadmap and immediate next steps.

## Development Notes

- Uses `uv` for dependency management
- DataFusion as the core execution engine for performance
- Optional dependencies: sklearn, pandas, polars
- All transforms must support both batch and potential single-record inference
- Maintain backward compatibility with scikit-learn transformer interface