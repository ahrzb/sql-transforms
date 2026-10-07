# sql-transform specification

This folder states what `sql_transform` does now. It holds no plans, no
proposals and no history. The [package README](../README.md) is the guide
for users. The [glossary](../../../GLOSSARY.md) defines the terms.

## Files

Read [the transform model](transform-model.md) first. Then read the file
for the part that you use.

| File | What it pins |
|---|---|
| [The transform model](transform-model.md) | Fit data and request data, fit as partial application, general transforms and projections, and the three separate checks. |
| [Composition and replay](composition.md) | Members and their two relation arguments, chaining, `run`, name capture, `source` and replay. |
| [General transforms](general-transforms.md) | The `SQLTransform` constructor, `Fitted`, the scikit-learn interface, output modes and lazy relations. |
| [Projections](projections.md) | The row-local rule and its named reasons, params join cardinality, the public fields of a fitted projection, the request schema, captured relations and caller catalogs. |
| [Connections and leases](connections.md) | Caller connections, leased names, cleanup and error order, and the thread setting. |
| [Freezing](fit/freezing.md) | Freezing order, the static DISTINCT pick, which params ship, artifact size and the training-set refusal. |
| [Decorrelation](fit/decorrelation.md) | The correlated fit subqueries that fit rewrites: predicate partition, key operators, misses and guards. |
| [Decorrelation refusals](fit/decorrelation-refusals.md) | Each named `CorrelatedFit` reason, and the cross-type key error. |
| [Relation callbacks](python/relation-callbacks.md) | `Transform` callbacks, their declarations, and the handle representations. |
| [Estimators and scalar UDFs](python/estimators.md) | What a projection may capture, the canonical fit and application forms, bundles, fit scopes, learned UDFs and IDs. |
| [Estimator data](python/estimator-data.md) | Feature types, values at the estimator boundary, learned outputs, fit order, empty groups and reserved columns. |
| [Tree models](python/tree-models.md) | `TreeBasedTransform`: the call, the estimators that it packs, the threshold and integer rules, summation order and missing values. |
| [Window marginalization](marginalization.md) | `SQLProjection.marginalize`: the accepted query, lookup keys, admitted windows, numerical fidelity and lookups. |
| [Unsupported forms](unsupported-forms.md) | Forms that refuse, each with its explicit alternative. |
| [Native catalog contract](native/catalog-contract.md) | What a catalog entry must equal, `to_native`, where the twin raises, and the scope. |
| [Native catalog coverage](native/coverage.md) | Each scikit-learn transformer: native, composition, out of scope or not yet. A script generates it. |
| [Verification](verification.md) | How the claims are checked: Evidence lines, the executable examples, the original-query differential and the corpus. |

## How this spec is written

- **Current state only.** A file states what the package does now, with
  only the reasons that a reader needs to understand a rule. Proposals go to
  [plan/](../plan/README.md). History goes to [records/](../records/README.md).
- **One home for each rule.** A rule is written once. Other files link to
  its claim.
- **Names, not numbers.** A file name is a topic in lowercase words with
  hyphens. A heading names its subject in words. File names and headings
  have no numbers, no section signs and no tags. The table above gives the
  reading order.
- **Claims.** Each rule starts with `**claim: <name>.**`. The name is
  lowercase words with hyphens, and it is unique in this folder. Cite a
  claim as `[claim: <name>](<file>.md)` in documents, and as
  `claim: <name>` in code and test comments.
- **Evidence.** Each claim ends with an `*Evidence:*` line that names the
  tests that fail if the rule breaks, as `<file>::<function>`. The file is
  relative to `sql_transform/`; a test of another package uses its path
  from the repository root. If no test pins the claim, the line says
  `not pinned by a test`, and the [open work](../plan/README.md#open-work)
  lists the claim.
- **Examples.** An example is a `pycon` block with its own imports. An
  empty line comes before the closing fence. `sql_transform/_docs_test.py`
  runs each `pycon` block in this folder. Do not add a Python example that
  does not run.
- **Changes.** A PR that changes behavior changes its claim in the same PR.
  A PR that removes a public interface removes its claims and adds an entry
  to the [changelog](../records/changelog.md).
- **Generated pages.** [native/coverage.md](native/coverage.md) is
  generated. Change its table only with the command that its first lines
  give.
- **Text and diagrams.** Text follows the `simple-english` skill. A diagram
  is a mermaid block.

## Examples

The examples assume Python and SQL knowledge. DuckDB executes batch SQL.
PyArrow (`pyarrow`) supplies columnar Arrow tables and their field schemas.
The examples import it as `pa`. The estimator examples use scikit-learn,
imported as `sklearn`. NumPy supplies arrays, and pandas supplies DataFrames
for the matching output modes.

## Other documents

- [Research](../research/README.md): the questions that the package
  investigated, with sourced facts.
- [Plan](../plan/README.md): draft RFCs, open decisions and open work.
- [Records](../records/README.md): closed decisions, applied RFCs, research
  lessons and the changelog.
- Confit's [serving contract](../../confit/docs/specs/serving-contract.md)
  defines the engine behavior that this spec does not own.
- The [system specification](../../../docs/specs/README.md) indexes the
  specs of the whole repository.
