# Decisions

Questions about `sql_transform` that need the owner to choose. Each file
asks one question. The spec holds only the current rules. These files keep
the options and the reasons for each choice.

A loop asks its questions in its own `decisions/` folder, for example
[loops/native/decisions/](../../../../loops/native/decisions/README.md).
This folder is for questions from work outside a loop.

| Folder | Holds |
|---|---|
| `open/` | Questions that wait for the owner. [plan/README.md](../README.md#open-decisions) lists them. |
| `postponed/` | Questions left for later on purpose. The file says what holds meanwhile, and what brings the question back. |
| [records/decisions/](../../records/README.md) | Closed questions. The file records the choice, the date and the reason. |

## Writing a decision record

1. Read `records/decisions/` and `postponed/` first. If a record answers the
   question, do not ask it again.
2. Name the file after the question, in lowercase words with hyphens, for
   example `open/raw-fit-order-ties.md`. Do not number it.
3. Use the format below. Give at least two real options.
4. Add a line for it under "Open decisions" in
   [plan/README.md](../README.md#open-decisions).

## Closing a decision

1. Add a `## Decision` section. For a choice, give the option, the date and
   any change to the reasons. For a postponement, give the date, the rule
   that applies until the owner rules (as a link to its claim in the spec),
   and the condition that brings the question back.
2. Move the file to `records/decisions/` or to `postponed/`, with the same
   name.
3. Apply the ruling to the spec. Write the rule once, as a claim in its home
   file. Replace each other mention of the open decision with a link to that
   claim.
4. Remove its line from "Open decisions" in plan/README.md. A postponed
   decision moves to "Postponed decisions".

## Format

```markdown
# <The question, in plain words>

## Context

What this is about and why it matters: the goal that it serves, and what
goes wrong if the answer is wrong. Only what a reader needs to choose.

- **Read first:** the spec claims that a reader needs, as links.
- **Links:** the decisions that this one depends on, and those that it
  affects, by file name.

## Options

### <Option name>

What it is, in two or three sentences.

- **Gains:** what this gives.
- **Costs:** what this costs, and who pays.

### <Option name>

...

## Evidence

**Weight:** strong, moderate or thin. The measurements and research that the
options rely on, as links. Then the points that are not settled.

## Recommendation

The option, by name, and why, in a few sentences: which trade-off decides
it, and what would change the answer.
```

A record may hold several questions when they share context and their
answers constrain each other. Then each question is a section
`## <question>` with its own options and recommendation.
