# Records

This folder holds documents that are kept as written. Add to them; do not
rewrite them. When a link target moves, repair the link. A record that relies
on code or on a measurement names the commit that it read.

| Folder or file | Holds |
|---|---|
| [changelog.md](changelog.md) | Changes to the public interface, newest first. |
| `decisions/` | Closed decisions, in the format of the [decision guide](../plan/decisions/README.md). None yet. |
| `rfcs/` | RFCs that the owner approved or rejected, with the options that they weighed. None yet. |
| `research/` | What each research question meant for `sql_transform`. |

Research lessons:

- [decorrelation/type-ja-survey.md](research/decorrelation/type-ja-survey.md),
  written on 2026-08-08: whether a correlated `__FIT__` subquery can become a
  pre-aggregated params table, and how many refusals that costs.

A loop keeps its own decision records. The native catalog's records are in
[loops/native/decisions/](../../../loops/native/decisions/README.md).
