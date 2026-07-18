---
id: doc-1
title: DataFusion function catalogue
type: other
created_date: '2026-07-18 14:01'
---
# DataFusion function & aggregate catalogue

Auto-generated from the installed **DataFusion 54.0.0** (the engine behind the
`transform` / `fit` path) via its `information_schema.routines`. This is the
**parity target**: every function here runs in the DataFusion (batch) path, so the
native `InferFn` interpreter must match any of these it claims to support — and it is
also the menu the authoring SQL surface can draw from. See
[SQL_SUPPORT.md](../../docs/SQL_SUPPORT.md) for what the *interpreter* implements today.

Regenerate after a DataFusion upgrade: `uv run python
scripts/gen_datafusion_catalogue.py`. Signatures/descriptions are DataFusion's own
metadata verbatim, including a few upstream quirks where an alias documents under
another name's signature (e.g. `mean`→`avg`).

**Totals (54.0.0):** 45 aggregate · 11 window · 247 scalar (a few names appear in two sections — both an aggregate and a window function).

## Aggregate functions (45)

| Function | Signature | Description |
|---|---|---|
| `approx_distinct` | approx_distinct(expression) | Returns the approximate number of distinct input values calculated using the HyperLogLog algorithm. |
| `approx_median` | approx_median(expression) | Returns the approximate median (50th percentile) of input values. It is an alias of `approx_percentile_cont(0.5) WITHIN GROUP (ORDER BY x)`. |
| `approx_percentile_cont` | approx_percentile_cont(percentile [, centroids]) WITHIN GROUP (ORDER BY expression) | Returns the approximate percentile of input values using the t-digest algorithm. |
| `approx_percentile_cont_with_weight` | approx_percentile_cont_with_weight(weight, percentile [, centroids]) WITHIN GROUP (ORDER BY expression) | Returns the weighted approximate percentile of input values using the t-digest algorithm. |
| `array_agg` | array_agg(expression [ORDER BY expression]) | Returns an array created from the expression elements. If ordering is required, elements are inserted in the specified order. This aggregation function can only mix DISTINCT and ORDER BY if the ordering expression is exactly the same as the argument expression. |
| `avg` | avg(expression) | Returns the average of numeric values in the specified column. |
| `bit_and` | bit_and(expression) | Computes the bitwise AND of all non-null input values. |
| `bit_or` | bit_or(expression) | Computes the bitwise OR of all non-null input values. |
| `bit_xor` | bit_xor(expression) | Computes the bitwise exclusive OR of all non-null input values. |
| `bool_and` | bool_and(expression) | Returns true if all non-null input values are true, otherwise false. |
| `bool_or` | bool_and(expression) | Returns true if all non-null input values are true, otherwise false. |
| `corr` | corr(expression1, expression2) | Returns the coefficient of correlation between two numeric values. |
| `count` | count(expression) | Returns the number of non-null values in the specified column. To include null values in the total count, use `count(*)`. |
| `covar` | covar_samp(expression1, expression2) | Returns the sample covariance of a set of number pairs. |
| `covar_pop` | covar_samp(expression1, expression2) | Returns the sample covariance of a set of number pairs. |
| `covar_samp` | covar_samp(expression1, expression2) | Returns the sample covariance of a set of number pairs. |
| `first_value` | first_value(expression [ORDER BY expression]) | Returns the first element in an aggregation group according to the requested ordering. If no ordering is given, returns an arbitrary element from the group. |
| `grouping` | grouping(expression) | Returns 1 if the data is aggregated across the specified column, or 0 if it is not aggregated in the result set. |
| `last_value` | last_value(expression [ORDER BY expression]) | Returns the last element in an aggregation group according to the requested ordering. If no ordering is given, returns an arbitrary element from the group. |
| `max` | max(expression) | Returns the maximum value in the specified column. |
| `mean` | avg(expression) | Returns the average of numeric values in the specified column. |
| `median` | median(expression) | Returns the median value in the specified column. |
| `min` | min(expression) | Returns the minimum value in the specified column. |
| `nth_value` | nth_value(expression, n ORDER BY expression) | Returns the nth value in a group of values. |
| `percentile_cont` | percentile_cont(percentile) WITHIN GROUP (ORDER BY expression) | Returns the exact percentile of input values, interpolating between values if needed. |
| `quantile_cont` | percentile_cont(percentile) WITHIN GROUP (ORDER BY expression) | Returns the exact percentile of input values, interpolating between values if needed. |
| `regr_avgx` | regr_avgx(expression_y, expression_x) | Computes the average of the independent variable (input) expression_x for the non-null paired data points. |
| `regr_avgy` | regr_avgy(expression_y, expression_x) | Computes the average of the dependent variable (output) expression_y for the non-null paired data points. |
| `regr_count` | regr_count(expression_y, expression_x) | Counts the number of non-null paired data points. |
| `regr_intercept` | regr_intercept(expression_y, expression_x) | Computes the y-intercept of the linear regression line. For the equation (y = kx + b), this function returns b. |
| `regr_r2` | regr_r2(expression_y, expression_x) | Computes the square of the correlation coefficient between the independent and dependent variables. |
| `regr_slope` | regr_slope(expression_y, expression_x) | Returns the slope of the linear regression line for non-null pairs in aggregate columns. Given input column Y and X: regr_slope(Y, X) returns the slope (k in Y = k*X + b) using minimal RSS fitting. |
| `regr_sxx` | regr_sxx(expression_y, expression_x) | Computes the sum of squares of the independent variable. |
| `regr_sxy` | regr_sxy(expression_y, expression_x) | Computes the sum of products of paired data points. |
| `regr_syy` | regr_syy(expression_y, expression_x) | Computes the sum of squares of the dependent variable. |
| `stddev` | stddev(expression) | Returns the standard deviation of a set of numbers. |
| `stddev_pop` | stddev_pop(expression) | Returns the population standard deviation of a set of numbers. |
| `stddev_samp` | stddev(expression) | Returns the standard deviation of a set of numbers. |
| `string_agg` | string_agg([DISTINCT] expression, delimiter [ORDER BY expression]) | Concatenates the values of string expressions and places separator values between them. If ordering is required, strings are concatenated in the specified order. This aggregation function can only mix DISTINCT and ORDER BY if the ordering expression is exactly the same as the first argument expression. |
| `sum` | sum(expression) | Returns the sum of all values in the specified column. |
| `var` | var(expression) | Returns the statistical sample variance of a set of numbers. |
| `var_pop` | var_pop(expression) | Returns the statistical population variance of a set of numbers. |
| `var_population` | var_pop(expression) | Returns the statistical population variance of a set of numbers. |
| `var_samp` | var(expression) | Returns the statistical sample variance of a set of numbers. |
| `var_sample` | var(expression) | Returns the statistical sample variance of a set of numbers. |

## Window functions (11)

| Function | Signature | Description |
|---|---|---|
| `cume_dist` | cume_dist() | Relative rank of the current row: (number of rows preceding or peer with the current row) / (total rows). |
| `dense_rank` | dense_rank() | Returns the rank of the current row without gaps. This function ranks rows in a dense manner, meaning consecutive ranks are assigned even for identical values. |
| `first_value` | first_value(expression) | Returns value evaluated at the row that is the first row of the window frame. |
| `lag` | lag(expression, offset, default) | Returns value evaluated at the row that is offset rows before the current row within the partition; if there is no such row, instead return default (which must be of the same type as value). |
| `last_value` | last_value(expression) | Returns value evaluated at the row that is the last row of the window frame. |
| `lead` | lead(expression, offset, default) | Returns value evaluated at the row that is offset rows after the current row within the partition; if there is no such row, instead return default (which must be of the same type as value). |
| `nth_value` | nth_value(expression, n) | Returns the value evaluated at the nth row of the window frame (counting from 1). Returns NULL if no such row exists. |
| `ntile` | ntile(expression) | Integer ranging from 1 to the argument value, dividing the partition as equally as possible |
| `percent_rank` | percent_rank() | Returns the percentage rank of the current row within its partition. The value ranges from 0 to 1 and is computed as `(rank - 1) / (total_rows - 1)`. |
| `rank` | rank() | Returns the rank of the current row within its partition, allowing gaps between ranks. This function provides a ranking similar to `row_number`, but skips ranks for identical values. |
| `row_number` | row_number() | Number of the current row within its partition, counting from 1. |

## Scalar functions (247)

| Function | Signature | Description |
|---|---|---|
| `abs` | abs(numeric_expression) | Returns the absolute value of a number. |
| `acos` | acos(numeric_expression) | Returns the arc cosine or inverse cosine of a number. |
| `acosh` | acosh(numeric_expression) | Returns the area hyperbolic cosine or inverse hyperbolic cosine of a number. |
| `array_any_value` | array_any_value(array) | Returns the first non-null element in the array. |
| `array_append` | array_append(array, element) | Appends an element to the end of an array. |
| `array_cat` | array_concat(array[, ..., array_n]) | Concatenates arrays. |
| `array_compact` | array_compact(array) | Removes null values from the array. |
| `array_concat` | array_concat(array[, ..., array_n]) | Concatenates arrays. |
| `array_contains` | array_has(array, element) | Returns true if the array contains the element. |
| `array_dims` | array_dims(array) | Returns an array of the array's dimensions. |
| `array_distance` | array_distance(array1, array2) | Returns the Euclidean distance between two input arrays of equal length. |
| `array_distinct` | array_distinct(array) | Returns distinct values from the array after removing duplicates. |
| `array_element` | array_element(array, index) | Extracts the element with the index n from the array. |
| `array_empty` | empty(array) | Returns 1 for an empty array or 0 for a non-empty array. |
| `array_except` | array_except(array1, array2) | Returns an array of the elements that appear in the first array but not in the second. |
| `array_extract` | array_element(array, index) | Extracts the element with the index n from the array. |
| `array_has` | array_has(array, element) | Returns true if the array contains the element. |
| `array_has_all` | array_has_all(array, sub-array) | Returns true if all elements of sub-array exist in array. |
| `array_has_any` | array_has_any(array1, array2) | Returns true if the arrays have any elements in common. |
| `array_indexof` | array_position(array, element) array_position(array, element, index) | Returns the position of the first occurrence of the specified element in the array, or NULL if not found. Comparisons are done using `IS DISTINCT FROM` semantics, so NULL is considered to match NULL. |
| `array_intersect` | array_intersect(array1, array2) | Returns an array of elements in the intersection of array1 and array2. |
| `array_join` | array_to_string(array, delimiter[, null_string]) | Converts each element to its text representation. |
| `array_length` | array_length(array, dimension) | Returns the length of the array dimension. |
| `array_max` | array_max(array) | Returns the maximum value in the array. |
| `array_min` | array_min(array) | Returns the minimum value in the array. |
| `array_ndims` | array_ndims(array, element) | Returns the number of dimensions of the array. |
| `array_normalize` | array_normalize(array) | Returns the L2-normalized vector for the input numeric array, computed as `array[i] / sqrt(sum(array[i]^2))` per element. Returns NULL if the input is NULL, contains NULL elements, or has zero magnitude (all elements are zero). Returns an empty array for an empty input array. |
| `array_pop_back` | array_pop_back(array) | Returns the array without the last element. |
| `array_pop_front` | array_pop_front(array) | Returns the array without the first element. |
| `array_position` | array_position(array, element) array_position(array, element, index) | Returns the position of the first occurrence of the specified element in the array, or NULL if not found. Comparisons are done using `IS DISTINCT FROM` semantics, so NULL is considered to match NULL. |
| `array_positions` | array_positions(array, element) | Searches for an element in the array, returns all occurrences. |
| `array_prepend` | array_prepend(element, array) | Prepends an element to the beginning of an array. |
| `array_push_back` | array_append(array, element) | Appends an element to the end of an array. |
| `array_push_front` | array_prepend(element, array) | Prepends an element to the beginning of an array. |
| `array_remove` | array_remove(array, element) | Removes the first element from the array equal to the given value. NULL elements already in the array are preserved when removing a non-NULL value. If `element` evaluates to NULL, the result is NULL rather than removing NULL entries. |
| `array_remove_all` | array_remove_all(array, element) | Removes all elements from the array equal to the given value. NULL elements already in the array are preserved when removing a non-NULL value. If `element` evaluates to NULL, the result is NULL rather than removing NULL entries. |
| `array_remove_n` | array_remove_n(array, element, max) | Removes the first `max` elements from the array equal to the given value. NULL elements already in the array are preserved when removing a non-NULL value. If `element` evaluates to NULL, the result is NULL rather than removing NULL entries. |
| `array_repeat` | array_repeat(element, count) | Returns an array containing element `count` times. |
| `array_replace` | array_replace(array, from, to) | Replaces the first occurrence of the specified element with another specified element. |
| `array_replace_all` | array_replace_all(array, from, to) | Replaces all occurrences of the specified element with another specified element. |
| `array_replace_n` | array_replace_n(array, from, to, max) | Replaces the first `max` occurrences of the specified element with another specified element. |
| `array_resize` | array_resize(array, size, value) | Resizes the list to contain size elements. Initializes new elements with value or empty if value is not set. |
| `array_reverse` | array_reverse(array) | Returns the array with the order of the elements reversed. |
| `array_slice` | array_slice(array, begin, end) | Returns a slice of the array based on 1-indexed start and end positions. |
| `array_sort` | array_sort(array, desc, nulls_first) | Sort array. |
| `array_to_string` | array_to_string(array, delimiter[, null_string]) | Converts each element to its text representation. |
| `array_union` | array_union(array1, array2) | Returns an array of elements that are present in both arrays (all elements from both arrays) without duplicates. |
| `arrays_overlap` | array_has_any(array1, array2) | Returns true if the arrays have any elements in common. |
| `arrays_zip` | arrays_zip(array1[, ..., array_n]) | Returns an array of structs created by combining the elements of each input array at the same index. If the arrays have different lengths, shorter arrays are padded with NULLs. |
| `arrow_cast` | arrow_cast(expression, datatype) | Casts a value to a specific Arrow data type. |
| `arrow_field` | arrow_field(expression) | Returns a struct containing the Arrow field information of the expression, including name, data type, nullability, and metadata. |
| `arrow_metadata` | arrow_metadata(expression[, key]) | Returns the metadata of the input expression. If a key is provided, returns the value for that key. If no key is provided, returns a Map of all metadata. |
| `arrow_try_cast` | arrow_try_cast(expression, datatype) | Casts a value to a specific Arrow data type, returning NULL if the cast fails. |
| `arrow_typeof` | arrow_typeof(expression) | Returns the name of the underlying [Arrow data type](https://docs.rs/arrow/latest/arrow/datatypes/enum.DataType.html) of the expression. |
| `ascii` | ascii(str) | Returns the first Unicode scalar value of a string. |
| `asin` | asin(numeric_expression) | Returns the arc sine or inverse sine of a number. |
| `asinh` | asinh(numeric_expression) | Returns the area hyperbolic sine or inverse hyperbolic sine of a number. |
| `atan` | atan(numeric_expression) | Returns the arc tangent or inverse tangent of a number. |
| `atan2` | atan2(expression_y, expression_x) | Returns the arc tangent or inverse tangent of `expression_y / expression_x`. |
| `atanh` | atanh(numeric_expression) | Returns the area hyperbolic tangent or inverse hyperbolic tangent of a number. |
| `bit_length` | bit_length(str) | Returns the bit length of a string. |
| `btrim` | btrim(str[, trim_str]) | Trims the specified trim string from the start and end of a string. If no trim string is provided, all spaces are removed from the start and end of the input string. |
| `cardinality` | cardinality(array) | Returns the total number of elements in the array. |
| `cast_to_type` | cast_to_type(expression, reference) | Casts the first argument to the data type of the second argument. Only the type of the second argument is used; its value is ignored. |
| `cbrt` | cbrt(numeric_expression) | Returns the cube root of a number. |
| `ceil` | ceil(numeric_expression) | Returns the nearest integer greater than or equal to a number. |
| `char_length` | character_length(str) | Returns the number of characters in a string. |
| `character_length` | character_length(str) | Returns the number of characters in a string. |
| `chr` | chr(expression) | Returns a string containing the character with the specified Unicode scalar value. |
| `coalesce` | coalesce(expression1[, ..., expression_n]) | Returns the first of its arguments that is not _null_. Returns _null_ if all arguments are _null_. This function is often used to substitute a default value for _null_ values. |
| `concat` | concat(str[, ..., str_n]) | Concatenates multiple strings together. |
| `concat_ws` | concat_ws(separator, str[, ..., str_n]) | Concatenates multiple strings together with a specified separator. |
| `contains` | contains(str, search_str) | Return true if search_str is found within string (case-sensitive). |
| `cos` | cos(numeric_expression) | Returns the cosine of a number. |
| `cosh` | cosh(numeric_expression) | Returns the hyperbolic cosine of a number. |
| `cosine_distance` | cosine_distance(array1, array2) | Returns the cosine distance between two input arrays of equal length. The cosine distance is defined as 1 - cosine_similarity, i.e. `1 - dot(a,b) / (\|\|a\|\| * \|\|b\|\|)`. Returns NULL if either array is NULL or contains only zeros. |
| `cot` | cot(numeric_expression) | Returns the cotangent of a number. |
| `current_date` | current_date() (optional) SET datafusion.execution.time_zone = '+00:00'; SELECT current_date(); | Returns the current date in the session time zone. The `current_date()` return value is determined at query time and will return the same date, no matter when in the query plan the function executes. |
| `current_time` | current_time() (optional) SET datafusion.execution.time_zone = '+00:00'; SELECT current_time(); | Returns the current time in the session time zone. The `current_time()` return value is determined at query time and will return the same time, no matter when in the query plan the function executes. The session time zone can be set using the statement 'SET datafusion.execution.time_zone = desired time zone'. The time zone can be a value like +00:00, 'Europe/London' etc. |
| `current_timestamp` | now() | Returns the current timestamp in the system configured timezone (None by default). The `now()` return value is determined at query time and will return the same timestamp, no matter when in the query plan the function executes. |
| `date_bin` | date_bin(interval, expression, origin-timestamp) | Calculates time intervals and returns the start of the interval nearest to the specified timestamp. Use `date_bin` to downsample time series data by grouping rows into time-based "bins" or "windows" and applying an aggregate or selector function to each window. For example, if you "bin" or "window" data into 15 minute intervals, an input timestamp of `2023-01-01T18:18:18Z` will be updated to the start time of the 15 minute bin it is in: `2023-01-01T18:15:00Z`. |
| `date_format` | to_char(expression, format) | Returns a string representation of a date, time, timestamp or duration based on a [Chrono format](https://docs.rs/chrono/latest/chrono/format/strftime/index.html). Unlike the PostgreSQL equivalent of this function numerical formatting is not supported. |
| `date_part` | date_part(part, expression) | Returns the specified part of the date as an integer. |
| `date_trunc` | date_trunc(precision, expression) | Truncates a timestamp or time value to a specified precision. |
| `datepart` | date_part(part, expression) | Returns the specified part of the date as an integer. |
| `datetrunc` | date_trunc(precision, expression) | Truncates a timestamp or time value to a specified precision. |
| `decode` | decode(expression, format) | Decode binary data from textual representation in string. |
| `degrees` | degrees(numeric_expression) | Converts radians to degrees. |
| `digest` | digest(expression, algorithm) | Computes the binary hash of an expression using the specified algorithm. |
| `dot_product` | inner_product(array1, array2) | Returns the inner product (dot product) of two input arrays of equal length, computed as `sum(array1[i] * array2[i])`. Returns NULL if either array is NULL or contains NULL elements. Returns 0.0 for two empty arrays. |
| `element_at` | map_extract(map, key) | Returns a list containing the value for the given key or an empty list if the key is not present in the map. |
| `empty` | empty(array) | Returns 1 for an empty array or 0 for a non-empty array. |
| `encode` | encode(expression, format) | Encode binary data into a textual representation. |
| `ends_with` | ends_with(str, substr) | Tests if a string ends with a substring. |
| `exp` | exp(numeric_expression) | Returns the base-e exponential of a number. |
| `factorial` | factorial(numeric_expression) | Factorial of a non-negative integer. Errors if the argument is negative or the result overflows. |
| `find_in_set` | find_in_set(str, strlist) | Returns a value in the range of 1 to N if the string str is in the string list strlist consisting of N substrings. |
| `flatten` | flatten(array) | Converts an array of arrays to a flat array. - Applies to any depth of nested arrays - Does not change arrays that are already flat The flattened array contains all the elements from all source arrays. |
| `floor` | floor(numeric_expression) | Returns the nearest integer less than or equal to a number. |
| `from_unixtime` | from_unixtime(expression[, timezone]) | Converts an integer to RFC3339 timestamp format (`YYYY-MM-DDT00:00:00.000000000Z`). Integers and unsigned integers are interpreted as seconds since the unix epoch (`1970-01-01T00:00:00Z`) return the corresponding timestamp. |
| `gcd` | gcd(expression_x, expression_y) | Returns the greatest common divisor of `expression_x` and `expression_y`. Returns 0 if both inputs are zero. |
| `generate_series` | generate_series(stop) generate_series(start, stop[, step]) | Similar to the range function, but it includes the upper bound. |
| `get_field` | get_field(expression, field_name[, field_name2, ...]) | Returns a field within a map or a struct with the given key. Supports nested field access by providing multiple field names. Note: most users invoke `get_field` indirectly via field access syntax such as `my_struct_col['field_name']` which results in a call to `get_field(my_struct_col, 'field_name')`. Nested access like `my_struct['a']['b']` is optimized to a single call: `get_field(my_struct, 'a', 'b')`. |
| `greatest` | greatest(expression1[, ..., expression_n]) | Returns the greatest value in a list of expressions. Returns _null_ if all expressions are _null_. |
| `ifnull` | nvl(expression1, expression2) | Returns _expression2_ if _expression1_ is NULL otherwise it returns _expression1_ and _expression2_ is not evaluated. This function can be used to substitute a default value for NULL values. |
| `initcap` | initcap(str) | Capitalizes the first character in each word in the input string. Words are delimited by non-alphanumeric characters. |
| `inner_product` | inner_product(array1, array2) | Returns the inner product (dot product) of two input arrays of equal length, computed as `sum(array1[i] * array2[i])`. Returns NULL if either array is NULL or contains NULL elements. Returns 0.0 for two empty arrays. |
| `instr` | strpos(str, substr) | Returns the starting position of a specified substring in a string. Positions begin at 1. If the substring does not exist in the string, the function returns 0. |
| `isnan` | isnan(numeric_expression) | Returns true if a given number is +NaN or -NaN otherwise returns false. |
| `iszero` | iszero(numeric_expression) | Returns true if a given number is +0.0 or -0.0 otherwise returns false. |
| `lcm` | lcm(expression_x, expression_y) | Returns the least common multiple of `expression_x` and `expression_y`. Returns 0 if either input is zero. |
| `least` | least(expression1[, ..., expression_n]) | Returns the smallest value in a list of expressions. Returns _null_ if all expressions are _null_. |
| `left` | left(str, n) | Returns a specified number of characters from the left side of a string. |
| `length` | character_length(str) | Returns the number of characters in a string. |
| `levenshtein` | levenshtein(str1, str2) | Returns the [`Levenshtein distance`](https://en.wikipedia.org/wiki/Levenshtein_distance) between the two given strings. |
| `list_any_value` | array_any_value(array) | Returns the first non-null element in the array. |
| `list_append` | array_append(array, element) | Appends an element to the end of an array. |
| `list_cat` | array_concat(array[, ..., array_n]) | Concatenates arrays. |
| `list_compact` | array_compact(array) | Removes null values from the array. |
| `list_concat` | array_concat(array[, ..., array_n]) | Concatenates arrays. |
| `list_contains` | array_has(array, element) | Returns true if the array contains the element. |
| `list_dims` | array_dims(array) | Returns an array of the array's dimensions. |
| `list_distance` | array_distance(array1, array2) | Returns the Euclidean distance between two input arrays of equal length. |
| `list_distinct` | array_distinct(array) | Returns distinct values from the array after removing duplicates. |
| `list_element` | array_element(array, index) | Extracts the element with the index n from the array. |
| `list_empty` | empty(array) | Returns 1 for an empty array or 0 for a non-empty array. |
| `list_except` | array_except(array1, array2) | Returns an array of the elements that appear in the first array but not in the second. |
| `list_extract` | array_element(array, index) | Extracts the element with the index n from the array. |
| `list_has` | array_has(array, element) | Returns true if the array contains the element. |
| `list_has_all` | array_has_all(array, sub-array) | Returns true if all elements of sub-array exist in array. |
| `list_has_any` | array_has_any(array1, array2) | Returns true if the arrays have any elements in common. |
| `list_indexof` | array_position(array, element) array_position(array, element, index) | Returns the position of the first occurrence of the specified element in the array, or NULL if not found. Comparisons are done using `IS DISTINCT FROM` semantics, so NULL is considered to match NULL. |
| `list_intersect` | array_intersect(array1, array2) | Returns an array of elements in the intersection of array1 and array2. |
| `list_join` | array_to_string(array, delimiter[, null_string]) | Converts each element to its text representation. |
| `list_length` | array_length(array, dimension) | Returns the length of the array dimension. |
| `list_max` | array_max(array) | Returns the maximum value in the array. |
| `list_ndims` | array_ndims(array, element) | Returns the number of dimensions of the array. |
| `list_normalize` | array_normalize(array) | Returns the L2-normalized vector for the input numeric array, computed as `array[i] / sqrt(sum(array[i]^2))` per element. Returns NULL if the input is NULL, contains NULL elements, or has zero magnitude (all elements are zero). Returns an empty array for an empty input array. |
| `list_pop_back` | array_pop_back(array) | Returns the array without the last element. |
| `list_pop_front` | array_pop_front(array) | Returns the array without the first element. |
| `list_position` | array_position(array, element) array_position(array, element, index) | Returns the position of the first occurrence of the specified element in the array, or NULL if not found. Comparisons are done using `IS DISTINCT FROM` semantics, so NULL is considered to match NULL. |
| `list_positions` | array_positions(array, element) | Searches for an element in the array, returns all occurrences. |
| `list_prepend` | array_prepend(element, array) | Prepends an element to the beginning of an array. |
| `list_push_back` | array_append(array, element) | Appends an element to the end of an array. |
| `list_push_front` | array_prepend(element, array) | Prepends an element to the beginning of an array. |
| `list_remove` | array_remove(array, element) | Removes the first element from the array equal to the given value. NULL elements already in the array are preserved when removing a non-NULL value. If `element` evaluates to NULL, the result is NULL rather than removing NULL entries. |
| `list_remove_all` | array_remove_all(array, element) | Removes all elements from the array equal to the given value. NULL elements already in the array are preserved when removing a non-NULL value. If `element` evaluates to NULL, the result is NULL rather than removing NULL entries. |
| `list_remove_n` | array_remove_n(array, element, max) | Removes the first `max` elements from the array equal to the given value. NULL elements already in the array are preserved when removing a non-NULL value. If `element` evaluates to NULL, the result is NULL rather than removing NULL entries. |
| `list_repeat` | array_repeat(element, count) | Returns an array containing element `count` times. |
| `list_replace` | array_replace(array, from, to) | Replaces the first occurrence of the specified element with another specified element. |
| `list_replace_all` | array_replace_all(array, from, to) | Replaces all occurrences of the specified element with another specified element. |
| `list_replace_n` | array_replace_n(array, from, to, max) | Replaces the first `max` occurrences of the specified element with another specified element. |
| `list_resize` | array_resize(array, size, value) | Resizes the list to contain size elements. Initializes new elements with value or empty if value is not set. |
| `list_reverse` | array_reverse(array) | Returns the array with the order of the elements reversed. |
| `list_slice` | array_slice(array, begin, end) | Returns a slice of the array based on 1-indexed start and end positions. |
| `list_sort` | array_sort(array, desc, nulls_first) | Sort array. |
| `list_to_string` | array_to_string(array, delimiter[, null_string]) | Converts each element to its text representation. |
| `list_union` | array_union(array1, array2) | Returns an array of elements that are present in both arrays (all elements from both arrays) without duplicates. |
| `list_zip` | arrays_zip(array1[, ..., array_n]) | Returns an array of structs created by combining the elements of each input array at the same index. If the arrays have different lengths, shorter arrays are padded with NULLs. |
| `ln` | ln(numeric_expression) | Returns the natural logarithm of a number. |
| `log` | log(base, numeric_expression) log(numeric_expression) | Returns the base-x logarithm of a number. Can either provide a specified base, or if omitted then takes the base-10 of a number. |
| `log10` | log10(numeric_expression) | Returns the base-10 logarithm of a number. |
| `log2` | log2(numeric_expression) | Returns the base-2 logarithm of a number. |
| `lower` | lower(str) | Converts a string to lower-case. |
| `lpad` | lpad(str, n[, padding_str]) | Pads the left side of a string with another string to a specified string length. |
| `ltrim` | ltrim(str[, trim_str]) | Trims the specified trim string from the beginning of a string. If no trim string is provided, spaces are removed from the start of the input string. |
| `make_array` | make_array(expression1[, ..., expression_n]) | Returns an array using the specified input expressions. |
| `make_date` | make_date(year, month, day) | Make a date from year/month/day component parts. |
| `make_list` | make_array(expression1[, ..., expression_n]) | Returns an array using the specified input expressions. |
| `make_time` | make_time(hour, minute, second) | Make a time from hour/minute/second component parts. |
| `map` | map(key, value) map(key: value) make_map(['key1', 'key2'], ['value1', 'value2']) | Returns an Arrow map with the specified key-value pairs. The `make_map` function creates a map from two lists: one for keys and one for values. Each key must be unique and non-null. |
| `map_entries` | map_entries(map) | Returns a list of all entries in the map. |
| `map_extract` | map_extract(map, key) | Returns a list containing the value for the given key or an empty list if the key is not present in the map. |
| `map_keys` | map_keys(map) | Returns a list of all keys in the map. |
| `map_values` | map_values(map) | Returns a list of all values in the map. |
| `md5` | md5(expression) | Computes an MD5 128-bit checksum for a string expression. |
| `named_struct` | named_struct(expression1_name, expression1_input[, ..., expression_n_name, expression_n_input]) | Returns an Arrow struct using the specified name and input expressions pairs. For information on comparing and ordering struct values (including `NULL` handling), see [Comparison and Ordering](struct_coercion.md#comparison-and-ordering). |
| `nanvl` | nanvl(expression_x, expression_y) | Returns the first argument if it's not _NaN_. Returns the second argument otherwise. |
| `now` | now() | Returns the current timestamp in the system configured timezone (None by default). The `now()` return value is determined at query time and will return the same timestamp, no matter when in the query plan the function executes. |
| `nullif` | nullif(expression1, expression2) | Returns _null_ if _expression1_ equals _expression2_; otherwise it returns _expression1_. This can be used to perform the inverse operation of [`coalesce`](#coalesce). |
| `nvl` | nvl(expression1, expression2) | Returns _expression2_ if _expression1_ is NULL otherwise it returns _expression1_ and _expression2_ is not evaluated. This function can be used to substitute a default value for NULL values. |
| `nvl2` | nvl2(expression1, expression2, expression3) | Returns _expression2_ if _expression1_ is not NULL; otherwise it returns _expression3_. |
| `octet_length` | octet_length(str) | Returns the length of a string in bytes. |
| `overlay` | overlay(str PLACING substr FROM pos [FOR count]) | Returns the string which is replaced by another string from the specified position and specified count length. |
| `pi` | pi() | Returns an approximate value of π. |
| `position` | strpos(str, substr) | Returns the starting position of a specified substring in a string. Positions begin at 1. If the substring does not exist in the string, the function returns 0. |
| `pow` | power(base, exponent) | Returns a base expression raised to the power of an exponent. |
| `power` | power(base, exponent) | Returns a base expression raised to the power of an exponent. |
| `radians` | radians(numeric_expression) | Converts degrees to radians. |
| `rand` | random() | Returns a random float value in the range [0, 1). The random seed is unique to each row. |
| `random` | random() | Returns a random float value in the range [0, 1). The random seed is unique to each row. |
| `range` | range(stop) range(start, stop[, step]) | Returns an Arrow array between start and stop with step. The range start..end contains all values with start <= x < end. It is empty if start >= end. Step cannot be 0. |
| `regexp_count` | regexp_count(str, regexp[, start, flags]) | Returns the number of matches that a [regular expression](https://docs.rs/regex/latest/regex/#syntax) has in a string. |
| `regexp_instr` | regexp_instr(str, regexp[, start[, N[, flags[, subexpr]]]]) | Returns the position in a string where the specified occurrence of a POSIX regular expression is located. |
| `regexp_like` | regexp_like(str, regexp[, flags]) | Returns true if a [regular expression](https://docs.rs/regex/latest/regex/#syntax) has at least one match in a string, false otherwise. |
| `regexp_match` | regexp_match(str, regexp[, flags]) | Returns the first [regular expression](https://docs.rs/regex/latest/regex/#syntax) matches in a string. |
| `regexp_replace` | regexp_replace(str, regexp, replacement[, flags]) | Replaces substrings in a string that match a [regular expression](https://docs.rs/regex/latest/regex/#syntax). |
| `repeat` | repeat(str, n) | Returns a string with an input string repeated a specified number. |
| `replace` | replace(str, substr, replacement) | Replaces all occurrences of a specified substring in a string with a new substring. |
| `reverse` | reverse(str) | Reverses the character order of a string. |
| `right` | right(str, n) | Returns a specified number of characters from the right side of a string. |
| `round` | round(numeric_expression[, decimal_places]) | Rounds a number to the nearest integer. |
| `row` | struct(expression1[, ..., expression_n]) | Returns an Arrow struct using the specified input expressions optionally named. Fields in the returned struct use the optional name or the `cN` naming convention. For example: `c0`, `c1`, `c2`, etc. For information on comparing and ordering struct values (including `NULL` handling), see [Comparison and Ordering](struct_coercion.md#comparison-and-ordering). |
| `rpad` | rpad(str, n[, padding_str]) | Pads the right side of a string with another string to a specified string length. |
| `rtrim` | rtrim(str[, trim_str]) | Trims the specified trim string from the end of a string. If no trim string is provided, all spaces are removed from the end of the input string. |
| `sha224` | sha224(expression) | Computes the SHA-224 hash of a binary string. |
| `sha256` | sha256(expression) | Computes the SHA-256 hash of a binary string. |
| `sha384` | sha384(expression) | Computes the SHA-384 hash of a binary string. |
| `sha512` | sha512(expression) | Computes the SHA-512 hash of a binary string. |
| `signum` | signum(numeric_expression) | Returns the sign of a number. Negative numbers return `-1`. Zero and positive numbers return `1`. |
| `sin` | sin(numeric_expression) | Returns the sine of a number. |
| `sinh` | sinh(numeric_expression) | Returns the hyperbolic sine of a number. |
| `split_part` | split_part(str, delimiter, pos) | Splits a string based on a specified delimiter and returns the substring in the specified position. |
| `sqrt` | sqrt(numeric_expression) | Returns the square root of a number. |
| `starts_with` | starts_with(str, substr) | Tests if a string starts with a substring. |
| `string_to_array` | string_to_array(str, delimiter[, null_str]) | Splits a string into an array of substrings based on a delimiter. Any substrings matching the optional `null_str` argument are replaced with NULL. |
| `string_to_list` | string_to_array(str, delimiter[, null_str]) | Splits a string into an array of substrings based on a delimiter. Any substrings matching the optional `null_str` argument are replaced with NULL. |
| `strpos` | strpos(str, substr) | Returns the starting position of a specified substring in a string. Positions begin at 1. If the substring does not exist in the string, the function returns 0. |
| `struct` | struct(expression1[, ..., expression_n]) | Returns an Arrow struct using the specified input expressions optionally named. Fields in the returned struct use the optional name or the `cN` naming convention. For example: `c0`, `c1`, `c2`, etc. For information on comparing and ordering struct values (including `NULL` handling), see [Comparison and Ordering](struct_coercion.md#comparison-and-ordering). |
| `substr` | substr(str, start_pos[, length]) | Extracts a substring of a specified number of characters from a specific starting position in a string. |
| `substr_index` | substr_index(str, delim, count) | Returns the substring from str before count occurrences of the delimiter delim. If count is positive, everything to the left of the final delimiter (counting from the left) is returned. If count is negative, everything to the right of the final delimiter (counting from the right) is returned. |
| `substring` | substr(str, start_pos[, length]) | Extracts a substring of a specified number of characters from a specific starting position in a string. |
| `substring_index` | substr_index(str, delim, count) | Returns the substring from str before count occurrences of the delimiter delim. If count is positive, everything to the left of the final delimiter (counting from the left) is returned. If count is negative, everything to the right of the final delimiter (counting from the right) is returned. |
| `tan` | tan(numeric_expression) | Returns the tangent of a number. |
| `tanh` | tanh(numeric_expression) | Returns the hyperbolic tangent of a number. |
| `to_char` | to_char(expression, format) | Returns a string representation of a date, time, timestamp or duration based on a [Chrono format](https://docs.rs/chrono/latest/chrono/format/strftime/index.html). Unlike the PostgreSQL equivalent of this function numerical formatting is not supported. |
| `to_date` | to_date('2017-05-31', '%Y-%m-%d') | Converts a value to a date (`YYYY-MM-DD`). Supports strings, numeric and timestamp types as input. Strings are parsed as YYYY-MM-DD (e.g. '2023-07-20') if no [Chrono format](https://docs.rs/chrono/latest/chrono/format/strftime/index.html)s are provided. Integers and doubles are interpreted as days since the unix epoch (`1970-01-01T00:00:00Z`). Returns the corresponding date. Note: `to_date` returns Date32, which represents its values as the number of days since unix epoch(`1970-01-01`) stored as signed 32 bit value. The largest supported date value is `9999-12-31`. |
| `to_hex` | to_hex(int) | Converts an integer to a hexadecimal string. |
| `to_local_time` | to_local_time(expression) | Converts a timestamp with a timezone to a timestamp without a timezone (with no offset or timezone information). This function handles daylight saving time changes. |
| `to_time` | to_time('12:30:45', '%H:%M:%S') | Converts a value to a time (`HH:MM:SS.nnnnnnnnn`). Supports strings and timestamps as input. Strings are parsed as `HH:MM:SS`, `HH:MM:SS.nnnnnnnnn`, or `HH:MM` if no [Chrono format](https://docs.rs/chrono/latest/chrono/format/strftime/index.html)s are provided. Timestamps will have the time portion extracted. Returns the corresponding time. Note: `to_time` returns Time64(Nanosecond), which represents the time of day in nanoseconds since midnight. |
| `to_timestamp` | to_timestamp(expression[, ..., format_n]) | Converts a value to a timestamp (`YYYY-MM-DDT00:00:00.000000<TZ>`) in the session time zone. Supports strings, integer, unsigned integer, and double types as input. Strings are parsed as RFC3339 (e.g. '2023-07-20T05:44:00') if no [Chrono formats](https://docs.rs/chrono/latest/chrono/format/strftime/index.html) are provided. Strings that parse without a time zone are treated as if they are in the session time zone, or UTC if no session time zone is set. Integers, unsigned integers, and doubles are interpreted as seconds since the unix epoch (`1970-01-01T00:00:00Z`). Note: `to_timestamp` returns `Timestamp(ns, TimeZone)` where the time zone is the session time zone. The supported range for integer input is between`-9223372037` and `9223372036`. Supported range for string input is between `1677-09-21T00:12:44.0` and `2262-04-11T23:47:16.0`. Please use `to_timestamp_seconds` for the input outside of supported bounds. The session time zone can be set using the statement `SET TIMEZONE = 'desired time zone'`. The time zone can be a value like +00:00, 'Europe/London' etc. |
| `to_timestamp_micros` | to_timestamp_micros(expression[, ..., format_n]) | Converts a value to a timestamp (`YYYY-MM-DDT00:00:00.000000<TZ>`) in the session time zone. Supports strings, integer, unsigned integer, and double types as input. Strings are parsed as RFC3339 (e.g. '2023-07-20T05:44:00') if no [Chrono formats](https://docs.rs/chrono/latest/chrono/format/strftime/index.html) are provided. Strings that parse without a time zone are treated as if they are in the session time zone, or UTC if no session time zone is set. Integers, unsigned integers, and doubles are interpreted as microseconds since the unix epoch (`1970-01-01T00:00:00Z`). The session time zone can be set using the statement `SET TIMEZONE = 'desired time zone'`. The time zone can be a value like +00:00, 'Europe/London' etc. |
| `to_timestamp_millis` | to_timestamp_millis(expression[, ..., format_n]) | Converts a value to a timestamp (`YYYY-MM-DDT00:00:00.000<TZ>`) in the session time zone. Supports strings, integer, unsigned integer, and double types as input. Strings are parsed as RFC3339 (e.g. '2023-07-20T05:44:00') if no [Chrono formats](https://docs.rs/chrono/latest/chrono/format/strftime/index.html) are provided. Strings that parse without a time zone are treated as if they are in the session time zone, or UTC if no session time zone is set. Integers, unsigned integers, and doubles are interpreted as milliseconds since the unix epoch (`1970-01-01T00:00:00Z`). The session time zone can be set using the statement `SET TIMEZONE = 'desired time zone'`. The time zone can be a value like +00:00, 'Europe/London' etc. |
| `to_timestamp_nanos` | to_timestamp_nanos(expression[, ..., format_n]) | Converts a value to a timestamp (`YYYY-MM-DDT00:00:00.000000000<TZ>`) in the session time zone. Supports strings, integer, unsigned integer, and double types as input. Strings are parsed as RFC3339 (e.g. '2023-07-20T05:44:00') if no [Chrono formats](https://docs.rs/chrono/latest/chrono/format/strftime/index.html) are provided. Strings that parse without a time zone are treated as if they are in the session time zone. Integers, unsigned integers, and doubles are interpreted as nanoseconds since the unix epoch (`1970-01-01T00:00:00Z`). The session time zone can be set using the statement `SET TIMEZONE = 'desired time zone'`. The time zone can be a value like +00:00, 'Europe/London' etc. |
| `to_timestamp_seconds` | to_timestamp_seconds(expression[, ..., format_n]) | Converts a value to a timestamp (`YYYY-MM-DDT00:00:00<TZ>`) in the session time zone. Supports strings, integer, unsigned integer, and double types as input. Strings are parsed as RFC3339 (e.g. '2023-07-20T05:44:00') if no [Chrono formats](https://docs.rs/chrono/latest/chrono/format/strftime/index.html) are provided. Strings that parse without a time zone are treated as if they are in the session time zone, or UTC if no session time zone is set. Integers, unsigned integers, and doubles are interpreted as seconds since the unix epoch (`1970-01-01T00:00:00Z`). The session time zone can be set using the statement `SET TIMEZONE = 'desired time zone'`. The time zone can be a value like +00:00, 'Europe/London' etc. |
| `to_unixtime` | to_unixtime(expression[, ..., format_n]) | Converts a value to seconds since the unix epoch (`1970-01-01T00:00:00`). Supports strings, dates, timestamps, integer, unsigned integer, and float types as input. Strings are parsed as RFC3339 (e.g. '2023-07-20T05:44:00') if no [Chrono formats](https://docs.rs/chrono/latest/chrono/format/strftime/index.html) are provided. Integers, unsigned integers, and floats are interpreted as seconds since the unix epoch (`1970-01-01T00:00:00`). |
| `today` | current_date() (optional) SET datafusion.execution.time_zone = '+00:00'; SELECT current_date(); | Returns the current date in the session time zone. The `current_date()` return value is determined at query time and will return the same date, no matter when in the query plan the function executes. |
| `translate` | translate(str, from, to) | Performs character-wise substitution based on a mapping. |
| `trim` | btrim(str[, trim_str]) | Trims the specified trim string from the start and end of a string. If no trim string is provided, all spaces are removed from the start and end of the input string. |
| `trunc` | trunc(numeric_expression[, decimal_places]) | Truncates a number to a whole number or truncated to the specified decimal places. |
| `try_cast_to_type` | try_cast_to_type(expression, reference) | Casts the first argument to the data type of the second argument, returning NULL if the cast fails. Only the type of the second argument is used; its value is ignored. |
| `union_extract` | union_extract(union, field_name) | Returns the value of the given field in the union when selected, or NULL otherwise. |
| `union_tag` | union_tag(union_expression) | Returns the name of the currently selected field in the union |
| `upper` | upper(str) | Converts a string to upper-case. |
| `uuid` | uuid() | Returns [`UUID v4`](https://en.wikipedia.org/wiki/Universally_unique_identifier#Version_4_%28random%29) string value which is unique per row. |
| `version` | version() | Returns the version of DataFusion. |
| `with_metadata` | with_metadata(expression, key1, value1[, key2, value2, ...]) | Attaches Arrow field metadata (key/value pairs) to the input expression. Keys must be non-empty constant strings and values must be constant strings (empty values are allowed). Existing metadata on the input field is preserved; new keys overwrite on collision. This is the inverse of `arrow_metadata`. |
