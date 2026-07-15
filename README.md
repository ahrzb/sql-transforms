# SQL Transforms

Declarative ML preprocessing pipelines using SQL syntax.

## Installation

### Core Installation
```bash
pip install sql-transform
```

### With sklearn Integration
```bash
pip install sql-transform[sklearn]
# or
pip install sql-transform[all]
```

### Development Installation
```bash
git clone https://github.com/ahrzb/sql-transforms.git
cd sql-transforms
uv sync
```

## Quick Start

### Basic Usage (No sklearn required)

```python
from sql_transform import SQLTransform
import pyarrow as pa

# Create sample data
data = pa.table({
    "feature1": [1.0, 2.0, 3.0, 4.0, 5.0],
    "feature2": [10, 20, 30, 40, 50],
})

# Define SQL transformation -- input table is always referenced as __THIS__
sql = """
SELECT
    feature1 / MEAN(feature1) OVER () AS feature1_norm,
    feature2 / SUM(feature2) OVER () AS feature2_share
FROM __THIS__
"""

# Fit, then batch-transform through DataFusion (pyarrow in / pyarrow out)
transformer = SQLTransform(sql)
transformer.fit(data)
result = transformer.transform(data)
print(result)

# Low-latency inference through the Rust engine (dict or Pydantic model in,
# typed model out). infer() for one row, infer_batch() for many.
one = transformer.infer({"feature1": 2.0, "feature2": 20})
print(one.feature1_norm)
many = transformer.infer_batch([{"feature1": 2.0, "feature2": 20}])
```

### With sklearn Integration

```python
from sql_transform import SQLTransform

# sklearn transforms are available when sklearn is installed
sql = """
SELECT 
    sklearn.standardize(feature1) as std_feature1,
    sklearn.minmax_scale(feature2, 0, 1) as scaled_feature2,
    sklearn.kbins_discretize(feature3, 5, 'uniform') as binned_feature3
FROM __THIS__
"""

transformer = SQLTransform(sql)
transformer.fit(data)
result = transformer.transform(data)
```

## Available Transforms

### Built-in Aggregations (Always Available)
- `avg()`, `mean()` - Average/mean
- `stddev()` - Standard deviation  
- Window functions: `over (partition by ...)`

### sklearn Transforms (Optional)

#### Scaling
- `sklearn.standardize(column)` - Z-score normalization
- `sklearn.minmax_scale(column, min, max)` - Min-max scaling
- `sklearn.robust_scale(column)` - Robust scaling (median/IQR)
- `sklearn.quantile_transform(column, n_quantiles, distribution)` - Quantile transformation

#### Binning
- `sklearn.kbins_discretize(column, n_bins, strategy)` - K-bins discretizer

#### Categorical Encoding  
- `sklearn.onehot_encode(column)` - One-hot encoding
- `sklearn.ordinal_encode(column)` - Ordinal encoding

#### Text Processing
- `sklearn.tfidf_vectorize(column, max_features)` - TF-IDF vectorization

## Features

###  Current
- Rust-backed inference via a rewritten-SQL pipeline (window aggregates against __THIS__/__STATE__)
- DataFusion-based execution
- Optional sklearn integration
- Fit/transform ML pattern

### Roadmap
- More statistical transforms
- Advanced text processing
- Time series features  
- Type system with schema inference

## Development

### Running Tests
```bash
mise run test
# or
pytest
```

### Code Formatting
```bash
mise run fmt
# or
ruff check . && ruff format .
```

### Check Available Tasks
```bash
mise tasks
```

## Architecture

```
SQL Query → Parser → AST → Analyzer → Optimizer → Code Generator
                              ↓
                         Fit Phase: Compute aggregations  
                              ↓
                      Transform Phase: Apply transformations
```

The system separates **fit** (compute statistics) and **transform** (apply transformations) phases, making it suitable for ML pipelines where you fit on training data and transform both training and test data.

## Examples

See the [examples](examples/) directory for more comprehensive examples and tutorials.

## Contributing

1. Pick an item from [docs/BACKLOG.md](docs/BACKLOG.md)
2. Create an issue
3. Implement with tests
4. Submit PR

See [docs/VISION.md](docs/VISION.md) for the project vision and
[docs/BACKLOG.md](docs/BACKLOG.md) for the roadmap.

## License

MIT