---
id: TASK-72
title: Freezing loses the derived output column name
status: Done
assignee: []
created_date: '2026-08-07 21:42'
updated_date: '2026-08-07 22:17'
labels: []
dependencies: []
ordinal: 63000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`_plan.freeze` swaps the frozen query node for `_select_star(param)`
<!-- SECTION:DESCRIPTION:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Decision recorded in the spec's property table: 'freezing is faithful' covers
<!-- SECTION:NOTES:END -->
