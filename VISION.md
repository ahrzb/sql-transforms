# SQL Transforms: Declarative ML Preprocessing Pipelines

## Vision

Transform machine learning preprocessing from imperative code to declarative SQL, enabling versioned, optimized, and statically typed feature engineering pipelines.

## The Problem

Current ML preprocessing has several pain points:

1. **Imperative Code Complexity**: Feature engineering pipelines are written as imperative Python code, making them hard to understand, version, and maintain
2. **Versioning Challenges**: Preprocessing logic is embedded in code, making it difficult to version alongside models
3. **Optimization Gaps**: Batch-oriented preprocessing doesn't translate well to single-record inference
4. **Type Safety**: DataFrames are untyped, leading to runtime errors and unclear schemas
5. **Reproducibility**: Complex preprocessing steps are hard to reproduce and audit

## The Solution

**SQL Transforms** provides a declarative SQL-based approach to ML preprocessing:

```sql
SELECT
  feature1 - mean(feature1) as normalized_feature1,
  mean(target) over (partition by feature2) as target_encoding_feature2,
  sklearn.kbins_discretize(feature3, nbins=3) as binned_feature3,
  sklearn.tfidf(user_udf_tokenize(some_text_field)) as tfidf_features
FROM 
  DATA(
    feature1 FLOAT, 
    feature2 ENUM(x, y, z), 
    feature3 FLOAT, 
    some_text_field STRING, 
    target TARGET(FLOAT)
  )
```

### Key Benefits

1. **Declarative**: Express *what* you want, not *how* to compute it
2. **Versioned**: SQL files can be versioned alongside models
3. **Optimized**: Generate efficient single-record inference code
4. **Typed**: Statically determine output schemas from SQL
5. **Familiar**: Leverage existing SQL knowledge
6. **Auditable**: Clear, readable preprocessing logic

## Architecture Overview

```
SQL Query → Parser → AST → Analyzer → Optimizer → Code Generator
                              ↓
                         Fit Phase: Compute aggregations
                              ↓
                      Transform Phase: Apply transformations
```

### Core Components

1. **SQL Parser**: Parse extended SQL with ML-specific functions
2. **Type System**: Schema inference and validation
3. **Aggregation Engine**: Compute statistics during fit phase
4. **Transform Engine**: Apply transformations using pre-computed values
5. **Code Generator**: Generate optimized inference code
6. **UDF Registry**: Custom user-defined functions

## Current Status

✅ **Phase 1: Foundation** (Current)
- Basic SQL parsing with sqlglot
- Simple aggregations (mean, stddev)
- Window functions with partitioning
- DataFusion-based execution
- Basic test coverage

## Roadmap

### Phase 2: Core ML Functions (Next 2-4 weeks)

**Milestone 2.1: Statistical Transforms**
- [ ] Standardization (z-score normalization)
- [ ] Min-max scaling
- [ ] Robust scaling (median, IQR)
- [ ] Quantile transforms

**Milestone 2.2: Categorical Encoding**
- [ ] Target encoding (mean encoding)
- [ ] Frequency encoding
- [ ] One-hot encoding
- [ ] Ordinal encoding

**Milestone 2.3: Binning & Discretization**
- [ ] Equal-width binning
- [ ] Equal-frequency binning
- [ ] K-means binning
- [ ] Custom threshold binning

### Phase 3: Advanced Features (1-2 months)

**Milestone 3.1: Text Processing**
- [ ] Tokenization UDFs
- [ ] TF-IDF vectorization
- [ ] Count vectorization
- [ ] N-gram features

**Milestone 3.2: Time Series**
- [ ] Lag features
- [ ] Rolling window aggregations
- [ ] Time-based grouping
- [ ] Seasonal decomposition

**Milestone 3.3: Feature Interactions**
- [ ] Polynomial features
- [ ] Feature crosses
- [ ] Arithmetic combinations
- [ ] Conditional features

### Phase 4: Production Ready (2-3 months)

**Milestone 4.1: Type System**
- [ ] Schema inference from SQL
- [ ] Type validation
- [ ] Enum support
- [ ] Complex types (arrays, structs)

**Milestone 4.2: Code Generation**
- [ ] Single-record inference functions
- [ ] Rust/Python code generation
- [ ] Performance benchmarking
- [ ] Memory optimization

**Milestone 4.3: Integration & Polish**
- [ ] Scikit-learn compatibility
- [ ] Better error messages
- [ ] Performance optimizations
- [ ] Comprehensive documentation

## Technical Deep Dives

### 1. SQL Extensions for ML

```sql
-- Statistical functions
SELECT 
  standardize(feature1) as std_feature1,
  minmax_scale(feature2, min=0, max=1) as scaled_feature2,
  quantile_transform(feature3, n_quantiles=100) as uniform_feature3

-- Categorical encoding
SELECT
  target_encode(category, target) as target_encoded_category,
  frequency_encode(category) as freq_encoded_category,
  onehot_encode(category) as category_onehot

-- Binning
SELECT
  equal_width_bins(feature, n_bins=5) as binned_feature,
  equal_freq_bins(feature, n_bins=5) as freq_binned_feature,
  kmeans_bins(feature, n_bins=3) as kmeans_binned_feature
```

### 2. Type System

```sql
-- Schema definitions
DATA(
  user_id INT64 PRIMARY KEY,
  age FLOAT CHECK(age > 0 AND age < 150),
  category ENUM('A', 'B', 'C'),
  text_field STRING,
  target TARGET(FLOAT)  -- Special target designation
)

-- Inferred output schema
OUTPUT(
  std_age FLOAT,
  category_encoded FLOAT,
  text_features ARRAY<FLOAT>
)
```

### 3. Fit/Transform Separation

```python
# Fit phase: compute aggregations
transformer = SQLTransformer(sql_query)
transformer.fit(training_data)

# Transform phase: apply transformations
transformed_data = transformer.transform(new_data)

# Single-record inference
inference_fn = transformer.compile()
result = inference_fn(single_record)
```

## Getting Started

### Immediate Next Steps (This Week)

1. **Expand Test Coverage**
   - Add more complex window function tests
   - Test edge cases (empty data, null values)
   - Performance benchmarks

2. **Improve Error Handling**
   - Better error messages
   - Validation of SQL queries
   - Type checking

3. **Documentation**
   - API documentation
   - Tutorial notebooks
   - Example use cases

### Contributing

1. Pick a milestone from Phase 2
2. Create an issue for the specific feature
3. Implement with tests
4. Add documentation
5. Submit PR

## Success Metrics

- **Developer Experience**: Time to implement a preprocessing pipeline
- **Performance**: Single-record inference latency vs batch processing
- **Adoption**: Usage in real ML projects
- **Correctness**: Parity with scikit-learn transforms

## Long-term Vision

Imagine a world where:
- ML preprocessing is as simple as writing SQL
- Feature engineering pipelines are version-controlled like code
- Single-record inference is optimized by default
- Data scientists can collaborate using familiar SQL syntax
- Preprocessing logic is auditable and explainable

This project aims to make that vision a reality.