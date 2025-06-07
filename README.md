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
from sql_transform import SQLTransformer
import pyarrow as pa

# Create sample data
data = pa.table({
    "feature1": [1.0, 2.0, 3.0, 4.0, 5.0],
    "feature2": [10, 20, 30, 40, 50], 
    "class": ["A", "A", "B", "B", "A"]
})

# Define SQL transformation
sql = """
SELECT 
    feature1 - avg(feature1) as centered_feature1,
    avg(feature2) over (partition by class) as class_avg_feature2
FROM data
"""

# Fit and transform
transformer = SQLTransformer(sql)
transformer.fit(data)
result = transformer.transform(data)
print(result)
```

### With sklearn Integration

```python
from sql_transform import SQLTransformer

# sklearn transforms are available when sklearn is installed
sql = """
SELECT 
    sklearn.standardize(feature1) as std_feature1,
    sklearn.minmax_scale(feature2, 0, 1) as scaled_feature2,
    sklearn.kbins_discretize(feature3, 5, 'uniform') as binned_feature3
FROM data
"""

transformer = SQLTransformer(sql)
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
- Basic SQL parsing (SELECT with aggregations)
- Window functions with partitioning
- DataFusion-based execution
- Optional sklearn integration
- Fit/transform ML pattern

### Roadmap
- More statistical transforms
- Advanced text processing
- Time series features  
- Code generation for inference
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

1. Pick an item from [TODO.md](TODO.md)
2. Create an issue
3. Implement with tests
4. Submit PR

See [VISION.md](VISION.md) for the long-term roadmap.

## License

MIT