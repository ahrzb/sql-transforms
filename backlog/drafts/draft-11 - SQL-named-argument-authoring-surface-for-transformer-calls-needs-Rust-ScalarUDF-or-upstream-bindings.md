---
id: DRAFT-11
title: >-
  SQL named-argument authoring surface for transformer calls (needs Rust
  ScalarUDF or upstream bindings)
status: Draft
assignee: []
created_date: '2026-07-23 14:11'
updated_date: '2026-07-23 14:33'
labels:
  - transformer-refs
  - authoring-surface
  - native
  - ceiling
dependencies: []
references:
  - 'src/lib.rs:103'
  - 'src/expr.rs:414'
documentation:
  - 'doc-8 (Composition — {transform}(col) references)'
  - doc-7 (Transformer execution model)
priority: low
type: feature
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
WHAT A USER HITS
You have a fitted transformer and you want to be explicit about which column feeds which input — the way you would in any function call:

    SELECT {scaler}(age => applicant_age, income => gross_income) AS scaled FROM __THIS__

You cannot. Today binding is POSITIONAL and name-matched through metadata, so the user has to know the order the transformer was fit in and pass columns in that order:

    SELECT {scaler}(applicant_age, gross_income) AS scaled FROM __THIS__

Get the order wrong and you have silently swapped two features — the scaler happily standardizes income using age's mean. Nothing errors. For a transformer fit on 10 columns, positional binding is a genuine footgun, and named arguments are the obvious ergonomic fix (project goal 1 is ergonomic authoring).

WHY IT IS NOT A SMALL FIX (measured on datafusion 54.0.0, by Wren, 2026-07-23)
The blocker is a Python-BINDING gap, not a SQL-syntax gap:

    SELECT abs(x => age) FROM t    -> "Function 'abs' does not support named arguments"
    SELECT abs(x := age) FROM t    -> same

So the PARSER accepts both named-arg spellings and gets all the way to planning; the function then declines. Named-argument support is a per-function, Rust-side ScalarUDF capability. The Python bindings expose no way to opt in — datafusion.udf() has no named-arg parameter, and ScalarUDF exposes only [from_pycapsule, name, udf].

CONSEQUENCE for our model: what we call "input names" today are Arrow STRUCT FIELD NAMES carried by the named_struct we synthesize — metadata on the type, not SQL argument names. Both engines align on that metadata (native reads feature_names_in_ at src/lib.rs:103 and reorders at src/expr.rs:414, hard-erroring when it is absent).

Building a real named-argument surface therefore needs EITHER a Rust-side ScalarUDF registered via the PyCapsule path, OR upstream python-datafusion bindings that expose the capability. It is not a Python-side change.

WHY IT IS A DRAFT
No demand today — nobody has asked to author named args — and it needs design plus Rust/upstream work. Recorded primarily so nobody re-derives the finding. Becomes relevant if the ergonomic authoring goal ever wants named binding.

NOT TO BE CONFUSED WITH the ndarray-fit positional-binding problem, which was solved Python-only inside TASK-3: synthesize the names from the call site onto a copy.copy() of the transformer (doc-8 clone contract preserved, fitted state shared, no deep copy, no src/*.rs edit).

Context: doc-8 (composition — {transform}(col) references), doc-7 (transformer execution model).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Design decision recorded: Rust-side ScalarUDF via PyCapsule vs waiting on upstream python-datafusion bindings
- [ ] #2 {t}(name => col) authoring form works end-to-end with transform == infer parity (DataFusion oracle, decision-1)
- [ ] #3 Interaction with the existing feature_names_in_ / named_struct field-name mechanism is defined — named args must not silently diverge from struct field-name ordering
<!-- AC:END -->
