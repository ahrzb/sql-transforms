# Running a loop with workers

Both loops run the same scheme. This page is the scheme; each loop's README
adds, under "Tickets and review", only what is specific to it: where its
tickets come from, the files that collide, its review checklist and its
after-merge steps.

The agent reading this is assumed to be as capable as the one that wrote it.
These are working rules and their reasons. When a rule and the situation
disagree, follow the reason and record the decision in the PR.

## 1. Two modes

The owner picks the mode, usually by budget. Both share one definition of
done, which the loop's README states (gate green, lint clean, PLANS and docs
updated, PR squash-merged).

- **Inline** (the default, the cheapest). The loop session does each ticket
  itself, one at a time, on its own branch. It keeps everything in one
  context, so nobody writes a hand-off. Use it for small items, items that
  touch everything, and the moments between waves.
- **Worker-driven** (when the owner asks for parallelism). The loop session
  becomes a **supervisor**: it splits work into tickets (§2), launches one
  worker session per ticket (§3), reviews each worker's PR (§4) and merges
  them one at a time (§5). It still does small tickets and its loop's
  standing duties itself, and never lets its own work block a review.

Owner rulings in force (2026-10-05), for both loops:

- one branch per ticket;
- as much parallelism as the tickets allow: more when they are independent,
  fewer when they collide;
- the supervisor reviews, gives feedback and merges; workers never merge;
- the cost is acceptable for now.

**Switching** is safe at any point, because the state lives in GitHub (open
PRs) and in the loop's files (`tickets.md`, `PLANS.md`), not in anyone's head.
Going inline: stop launching workers, let running ones finish or interrupt
them, take over their branches. Going worker-driven: write the next tickets on
the board and launch workers for them.

When the account warns that a usage limit is close, tell the owner and drop to
fewer workers or to inline mode.

## 2. Tickets

A ticket is the unit one agent finishes alone: one concern, one branch, one
PR, reviewable in one sitting, with acceptance criteria a reviewer can check.
If the PR description would need two headlines, it is two tickets.

**Size.** Roughly 1–4 hours of agent work, measuring included. Bigger: split
it, so each slice ships something served and tested. Smaller than about 20
minutes: do it inline, but still put it on the board.

**Every ticket states:**

- the goal (with measured numbers when the motive is performance);
- pointers to the code it touches;
- the acceptance criteria;
- its branch, `depends on` and `overlaps` (as on the board);
- plus the fields its loop's README requires.

Owner decisions (`decisions/open/`) are never tickets: the loop prepares the
evidence and the owner rules.

**How many at once.** Start with 2–3 workers and add more when merges flow
without rework. The ceiling is the number of tickets that touch disjoint code
and can merge in any order, and never more than 4 open PRs waiting on review:
above that, review is the bottleneck and its quality drops. A ticket that
depends on an unmerged PR waits; branch it from master only after that PR
merges.

**Where workers run.** Workers are separate cloud sessions, so they don't
share this machine's CPU. Local subagents (the Agent tool) suit research and
read-only investigation, not tickets.

## 3. The board

`<loop>/tickets.md` lists everything in flight, inline work included, so the
owner and the tracker see one picture. It starts with one table:

```
| id | ticket | branch | depends on | overlaps | worker | PR | state |
|---|---|---|---|---|---|---|---|
| T4 | one-line title | `claude/<loop>-<short-name>` | #341 (merged) | `plan.rs` | `session_…` or `inline` | #350 | in review |
```

- **id**: `T<n>`, numbered per loop and never reused. Said as "confit T4",
  "native T1".
- **branch**: `claude/<loop>-<short-name>`. Branches that existed before this
  rule keep their names.
- **overlaps**: the hot files it shares with another open ticket; merge the
  smaller one first.
- **state**: one of `ready`, `in progress`, `in review`,
  `changes requested`, `blocked: <what>`.

Below the table, a "Next up" line names what starts when a slot frees, then
one `## T<n>: <title>` section per ticket holding the exact text the worker
receives. Delete the row and its section when the PR merges, and move what it
learned into `PLANS.md`.

## 4. Launching a worker

Call `create_session` (Claude Code Remote) with:

- `source_url`: `https://github.com/ahrzb/sql-transforms` (inherits this
  session's environment);
- `title`: `<loop> T<n>: <title>`;
- `prompt`: [`worker-brief.md`](worker-brief.md), then
  `<loop>/worker-brief.md`, then the ticket's section from `tickets.md`. The
  worker starts with nothing else, so the prompt has to stand alone.

Record the session id on the board, and the PR number once it exists.

Workers cannot message back. Follow them through their PRs (subscribe with
`subscribe_pr_activity`), `get_session` (`status_bucket` reads `failed` when a
turn errored), and the session's events. Correct a worker with `send_message`,
stop it with `interrupt_session`, or take its ticket back inline.

## 5. Review

Review every worker PR as if you had written it, against the diff and its
evidence, not the description alone. Check scope first (the ticket and
nothing else; surprise refactors go back), then the loop's own checklist.
Re-run anything that looks off.

**Delivering a review.** Give feedback as GitHub review comments, marking each
one **blocking** or **optional** and saying the file, what is wrong, and what
would satisfy you. The supervisor and the workers share one GitHub account,
so a review can only be COMMENT (never "changes requested"), and a worker's
harness may skip it as an echo of its own post. So always also deliver the
review to the worker directly: `send_message` when the session has it,
otherwise a one-shot routine fired into the worker's session
(`create_trigger` with `persistent_session_id` = the worker and `run_once_at`
a minute out) whose prompt says the review is the supervisor's, names it, and
lists the blocking items.

The worker answers each comment with a push or a reasoned reply. When a
blocking comment can't be settled, finish the fix yourself on their branch
with an ordinary commit. Never force-push a worker's branch.

## 6. Merge

- Merge once the review is satisfied and CI is green on the current head:
  squash, with `expectedHeadSha`.
- One at a time. After each merge the other open PRs merge master in (never a
  rebase of someone else's branch); the worker does it, or the supervisor when
  it is trivial. Generated files are regenerated, never hand-merged.
- Then: update the board, remove the item from `PLANS.md` if the PR didn't,
  run the loop's after-merge steps, and tell the other loop when the merge
  serves one of its needs.
- A merged PR is final. Follow-ups are new tickets.
