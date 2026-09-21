# Confit's goal

Confit specializes SQL feature transforms into low-latency functions for
row-at-a-time inference, without running a SQL engine for each request.

Each request is evaluated independently, using its input row, frozen tables, and
declared UDFs. Parsing, binding, compilation, and static-data preparation happen
at construction.

## Required behavior

An accepted transform must preserve the behavior defined by the
[oracle contract](oracle/README.md). Unsupported constructs must be rejected at
construction with a diagnostic identifying the construct. Accepted transforms
may still encounter data-dependent runtime errors.

Correctness constrains optimization: expand the supported SQL surface and reduce
serving latency without weakening the contract. The performance objective is
single-digit-microsecond, in-process serving—not a latency guarantee for every
query or UDF.

## Scope

The answer for one request must not depend on other requests in the batch.
Aggregation over frozen rows matched by that request is inside the model;
aggregation, sorting, or windows across request rows are outside it. Queries
reading no request table are outside the model.

Confit owns serving. `sql_transform` owns authoring, fitting, window
marginalization, and estimator packing. Missing implementation is not, by itself,
a reason to exclude a transform from the target.

## Supporting specifications

- [Serving contract](specs/serving-contract.md): API, UDFs, and detailed restrictions.
- [Oracle](oracle/README.md): the SQL reference and comparison rules.
- [Success measures](specs/success-measures.md): correctness controls, coverage, and latency.
- [Dated reports](reports/): measurements and implementation gaps, not requirements.
- [Earlier citations](oracle/13-old-ids.md): locations of merged or moved definitions.
