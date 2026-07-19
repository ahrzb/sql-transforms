---
id: DRAFT-2
title: Error attribution — failure to authored-SQL span to composite part
status: Draft
assignee: []
created_date: '2026-07-19 01:09'
labels:
  - composition
  - dx
milestone: m-1
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
A runtime interpreter failure (div/mod by zero, bad cast, unknown-category miss, type error) points at a node in the FUSED per-row expression, which has lost track of where it came from. Composition fuses N transformers' rewrites into one flat expression over __THIS__ + name-scoped __STATE_R{i}__ states, and nesting {a}({b}(x)) inlines b->a->outer, so a failing node can originate several layers deep. Goal: attribute a failure back -- failing op -> the span of authored SQL that produced it -> the specific transformer (and, for a composite, which referenced part at which nesting depth). This is the debuggability half of VISION hook 3 (provenance). Readiness done (5ac613e): the fit-cascade slice kept all inlining centralized in inline_references -- one choke point where origin tags can thread through. The REMAINING WORK NEEDS RUST: the composite's rewritten SQL reaches InferFn as a string, so a build-time tag on the sqlglot AST doesn't survive to the interpreter -- the tag must be propagated through the native engine and surfaced on error. Distinct from the error-type-parity non-goal (decision-2): that's which exception TYPE; this is LOCATING a failure in the source, and applies to both engines.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 origin tag threaded from inline_references through the interpreter and carried on the executing node
- [ ] #2 on a runtime failure, the raised error names the failing op, the authored-SQL span, and the specific transformer + nesting depth
<!-- AC:END -->
