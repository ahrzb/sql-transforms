# Research

Each folder answers one research question about `sql_transform`. Its README
states the question and lists its notes. The notes answer it with sourced
facts. What the facts meant for `sql_transform` is in
[records/research/](../records/README.md).

| Question | Folder |
|---|---|
| Which correlated `__FIT__` subqueries can fit turn into params tables without changing an answer, when fit and serving run at different times? | [decorrelation](decorrelation/README.md) |

## How a note is written

- The line after the title is a status line:
  `**Status:** <fact base or research note>. <Measured or read> <date>.`
  It names the versions that it measured, for example DuckDB 1.5.5.
- Each fact has a source: a measurement with its query, inputs and date; a
  paper with its section; or code at a named commit.
- `[INFERENCE]` marks the note's own reasoning. `[UNVERIFIED]` marks a claim
  that the note could not confirm at its source.
- A script that a note uses is in the same folder. The note gives the
  command that runs it.
- A note changes in place when a fact changes. It holds no recommendation.
  A recommendation goes to an RFC or a decision record in
  [plan/](../plan/README.md).
