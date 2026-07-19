---
id: DRAFT-1
title: Estimator-interface compliance (compose-in / hook 1)
status: Draft
assignee: []
created_date: '2026-07-19 01:09'
labels:
  - sklearn
  - compose-in
milestone: m-1
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make OUR transformer objects pass sklearn's check_estimator conformance (fit/transform/get_feature_names_out/get_params/set_params/clone/n_features_in_/tags, etc.) so they drop into a stock sklearn Pipeline/ColumnTransformer and coexist with sklearn's own transformers -- the (a) compose-in direction, VISION hook 1. Deferred (2026-07-16): only matters once we surface our own estimator objects into someone else's sklearn pipeline; the near-term target is the other direction -- the fallback execution node that runs fitted sklearn estimators inside OUR engine (see decision-4).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 sklearn.utils.estimator_checks.check_estimator passes against our transformer base; gaps closed
- [ ] #2 get_feature_names_out provenance contract (hook 3) pinned for external consumers
<!-- AC:END -->
