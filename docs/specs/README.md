# System specification

This is the entry point for the current system specification.
Read it by topic, not by implementation date.
The linked chapters define current behavior, supported interfaces, and refusals.

The [glossary](../../GLOSSARY.md) defines the project terms.
The repository contains the `sql-transform` authoring package and the Confit serving engine.
The native catalog supplies explicit native implementations of supported fitted transforms.

## Chapters

| Topic | Current specification |
| --- | --- |
| Transform authoring | [sql-transform spec](../../packages/sql-transform/spec/README.md): fit/request SQL, composition, freezing, decorrelation, callbacks, estimators, tree models, artifacts, connections and window marginalization |
| Confit serving | [Serving contract](../../packages/confit/docs/specs/serving-contract.md): public engine interfaces, row shapes, types, UDFs, and admission |
| SQL compatibility | [Oracle](../../packages/confit/docs/oracle/README.md): fixed reference, comparison rules, ordering, errors, and permitted numerical bounds |
| Native catalog | [Native catalog contract](../../packages/sql-transform/spec/native/catalog-contract.md) and [coverage](../../packages/sql-transform/spec/native/coverage.md): explicit conversion and parity requirements |
| Verification | [Success measures](../../packages/confit/docs/specs/success-measures.md) and [properties](../../packages/confit/docs/properties.md): controls and their test evidence |

`sql-transform` keeps its [research](../../packages/sql-transform/research/README.md), [plan](../../packages/sql-transform/plan/README.md) and [records](../../packages/sql-transform/records/README.md) next to its spec.

Authoring admission, row-local projection checks, and Confit admission are separate.
A successful batch transform does not imply that Confit can serve it.
Native selection is explicit; the authoring model does not expand the native catalog.

## Maintenance

Update the relevant chapter when behavior changes.
Keep its interfaces, examples, refusals, and verification references consistent with the implementation.
Do not create another dated specification for the same behavior.
Use stable topic names and section anchors for references.

Dated designs and implementation plans record earlier work; they do not override these chapters.
Reports record observations, and decision records explain choices.
Backlogs and plans describe future work, not implemented behavior.
Historical evidence and reference data remain separate from the current specification.
