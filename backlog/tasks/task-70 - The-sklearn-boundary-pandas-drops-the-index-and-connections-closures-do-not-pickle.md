---
id: TASK-70
title: >-
  The sklearn boundary: pandas drops the index, and connections/closures do not
  pickle
status: Done
assignee: []
created_date: '2026-08-07 21:42'
updated_date: '2026-08-07 22:01'
labels: []
dependencies: []
ordinal: 61000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Three independent defects at the sklearn edge. One is silent.
<!-- SECTION:DESCRIPTION:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
F4 (silent): _as_output now carries the caller's index through to pandas and
<!-- SECTION:NOTES:END -->
