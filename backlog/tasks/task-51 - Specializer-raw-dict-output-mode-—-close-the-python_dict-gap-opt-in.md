---
id: TASK-51
title: Specializer raw-dict output mode — close the python_dict gap (opt-in)
status: Done
assignee: []
created_date: '2026-07-26 20:17'
labels: []
milestone: m-7
dependencies:
  - TASK-50
type: feature
ordinal: 45000
---

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 output='dict' opt-in on DuckDBInferFn; typed default untouched; mutually exclusive with output_model; unknown values rejected
- [x] #2 All four emit paths honor it (marshaller, generic fallback, reentrant, constant engine) with fresh dicts per call
- [x] #3 Parity: dict mode == typed mode field-for-field, enforced in the serving parity gate + oracle tests
- [x] #4 Bench: spec_dict engine row lands the python_dict head-to-head
- [x] #5 gate green
<!-- AC:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Raw-dict output shipped as a strict opt-in: DuckDBInferFn(..., output="dict") returns per-row plain dicts, skipping model construction — the marshaller already built the dict the model would consume, so the modes agree field-for-field by construction (enforced in the serving parity gate and tests). All four emit paths honor the mode (generated marshaller, generic baseline, reentrant fallback, constant engine — the latter copying per call so caller mutation cannot leak). Guards: mutually exclusive with output_model, unknown values rejected, output getter exposes the mode. Measured at n=1024: 25-35% off the typed path; the handcrafted python_dict floor shrinks from the standing 1.3-2x caveat to 0.74-1.11x (spec_dict WINS house_prices) — the remaining gap is input marshalling, not output. Gate green (759 + 13 xfail); 6 new tests + parity-gate extension + spec_dict bench engine.
<!-- SECTION:FINAL_SUMMARY:END -->
