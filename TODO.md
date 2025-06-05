# TODO: Immediate Next Steps

## This Week (Foundation Cleanup)

### High Priority
- [ ] **Fix mypy type issues** in transformer_test.py (sorted() function)
- [ ] **Add comprehensive error handling** in parser.py for unsupported SQL constructs
- [ ] **Improve test coverage** 
  - [ ] Edge cases: empty tables, null values
  - [ ] Invalid SQL queries
  - [ ] Mismatched schemas
- [ ] **Better error messages** when SQL parsing fails

### Medium Priority
- [ ] **Add support for more aggregation functions**
  - [ ] `min()`, `max()`
  - [ ] `count()`, `sum()`
  - [ ] `median()`, `percentile()`
- [ ] **Improve documentation**
  - [ ] Add docstrings to all classes/functions
  - [ ] Create simple tutorial notebook
  - [ ] Document current SQL syntax support

### Low Priority
- [ ] **Performance benchmarking** basic operations
- [ ] **Add logging** for debugging
- [ ] **CLI interface** for running SQL transforms

## Next 2 Weeks (Phase 2.1: Statistical Transforms)

### Core Features
- [ ] **Standardization (z-score)**
  ```sql
  SELECT standardize(feature1) as std_feature1 FROM data
  ```
- [ ] **Min-max scaling**
  ```sql
  SELECT minmax_scale(feature1, min=0, max=1) as scaled_feature1 FROM data
  ```
- [ ] **Robust scaling** (median/IQR based)
  ```sql
  SELECT robust_scale(feature1) as robust_feature1 FROM data
  ```

### Implementation Plan
1. **Extend parser.py** to recognize new function names
2. **Add new expression types** for scaling operations
3. **Implement fit logic** to compute required statistics
4. **Implement transform logic** to apply scaling
5. **Add comprehensive tests** for each scaling method

## Next Month (Phase 2.2: Categorical Encoding)

### Target Features
- [ ] **Target encoding**
  ```sql
  SELECT target_encode(category, target) as encoded_category FROM data
  ```
- [ ] **Frequency encoding**
  ```sql
  SELECT frequency_encode(category) as freq_category FROM data
  ```
- [ ] **One-hot encoding**
  ```sql
  SELECT onehot_encode(category) as category_* FROM data
  ```

## Technical Debt to Address

### Architecture
- [ ] **Separate concerns** - split parser.py into multiple modules
- [ ] **Add proper type hints** throughout codebase
- [ ] **Create base classes** for different transform types
- [ ] **Standardize test patterns** across all tests

### Code Quality
- [ ] **Add pre-commit hooks** for formatting/linting
- [ ] **Set up GitHub Actions** for CI/CD
- [ ] **Add code coverage** reporting
- [ ] **Improve variable naming** and code organization

## Ideas for Future Exploration

### Quick Wins
- [ ] **Support for multiple tables/joins**
- [ ] **SQL comments** and better formatting
- [ ] **Validation of column names** against input schema
- [ ] **Support for aliases** in more places

### Bigger Features
- [ ] **Custom UDF registration**
- [ ] **SQL file loading** (not just strings)
- [ ] **Caching of fitted transformers**
- [ ] **Serialization/deserialization** of fitted models

## Questions to Investigate

1. **Performance**: How does DataFusion compare to pandas for small datasets?
2. **Memory**: What's the memory overhead of our approach vs scikit-learn?
3. **Compatibility**: Can we make this a drop-in replacement for sklearn transformers?
4. **Syntax**: What SQL extensions would be most valuable for ML practitioners?

## Success Criteria for Each Phase

### Foundation (Current)
- [ ] All tests pass consistently
- [ ] No mypy/type errors
- [ ] Basic documentation exists
- [ ] Code is well-organized

### Phase 2.1 (Statistical Transforms)
- [ ] 3+ scaling methods implemented
- [ ] Performance comparable to sklearn
- [ ] Comprehensive test coverage
- [ ] Clear API documentation

### Phase 2.2 (Categorical Encoding)
- [ ] Target encoding with proper CV handling
- [ ] One-hot encoding with sparse output
- [ ] Frequency encoding
- [ ] Integration tests with real datasets