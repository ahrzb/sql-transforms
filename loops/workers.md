# Running a loop with workers

Both loops run the same scheme, and this page describes it. The README of
each loop has a section called "Tickets and review". That section adds only
what is specific to the loop:

- where its tickets come from;
- the files that collide, which are files that two tickets both change;
- its review checklist;
- its steps after a merge.

Assume that the agent who reads this page is as capable as the agent who
wrote it. The page gives working rules and the reasons for them. If a rule
and the situation disagree, follow the reason. Then record the decision in
the PR.

## 1. Two modes

The owner picks the mode, usually by budget. Both modes share one definition
of done. The README of the loop states it: the gate passes, lint is clean,
PLANS and the docs are updated, and the PR is squash-merged.

- **Inline** is the default and the cheapest mode. The loop session does each
  ticket itself, one at a time, on its own branch. It keeps everything in one
  context, so nobody writes a hand-off. Use inline mode for small items, for
  items that touch everything, and for the time between waves.
- **Worker-driven** is the mode for when the owner asks for parallel work. The
  loop session becomes a **supervisor**. The supervisor does these things:
  1. It splits work into tickets (§2).
  2. It starts one worker session for each ticket (§3).
  3. It reviews the PR of each worker (§4).
  4. It merges the PRs one at a time (§5).

  The supervisor still does small tickets and the standing duties of its loop
  itself. It never lets its own work block a review.

The owner made these rulings on 2026-10-05. They are in force for both
loops:

- Use one branch for each ticket.
- Use as much parallel work as the tickets allow. Use more workers when the
  tickets are independent. Use fewer workers when the tickets collide.
- The supervisor reviews, gives feedback and merges. Workers never merge.
- The owner accepts the cost for now.

**Switching** between the modes is safe at any point. The state lives in
GitHub (the open PRs) and in the files of the loop (`tickets.md` and
`PLANS.md`). It does not live in the memory of any agent.

- To go to inline mode, stop the start of new workers. Let the running
  workers finish, or interrupt them. Then take over their branches.
- To go to worker-driven mode, write the next tickets on the board. Then
  start workers for them.

If the account warns that a usage limit is close, tell the owner. Then use
fewer workers, or use inline mode.

## 2. Tickets

A ticket is the unit of work that one agent finishes alone. It has one
concern, one branch and one PR. A reviewer can review it in one sitting. It
has acceptance criteria that a reviewer can check. If the PR description
needs two headlines, make two tickets.

**Size.** A ticket takes about 1 to 4 hours of agent work, and this includes
measuring.

- If a ticket is bigger, split it. Each slice must ship something that
  confit serves and that tests cover.
- If a ticket takes less than about 20 minutes, do it inline. Still put it
  on the board.

**Every ticket states:**

- the goal, with measured numbers when the motive is performance;
- pointers to the code that it touches;
- the acceptance criteria;
- its branch, `depends on` and `overlaps` (as on the board);
- the fields that the README of its loop requires.

A decision for the owner (`decisions/open/`) is never a ticket. The loop
prepares the evidence, and the owner rules.

**How many workers at once.** Start with 2 to 3 workers. Add more when merges
come without rework. The limit is the number of tickets that touch separate
code and can merge in any order. Never have more than 4 open PRs that wait
for review. Above 4, review limits the pace and its quality drops.

If a ticket depends on an unmerged PR, the ticket waits. Create its branch
from master only after that PR merges.

**Where workers run.** Workers are separate cloud sessions, so they do not
share the CPU of this machine. Local subagents (the Agent tool) suit
research and read-only investigation. They do not suit tickets.

## 3. The board

The file `<loop>/tickets.md` is the board. It lists everything in flight,
inline work included. The owner and the tracker then see one picture. The
file starts with one table:

```
| id | ticket | branch | depends on | overlaps | worker | PR | state |
|---|---|---|---|---|---|---|---|
| T4 | one-line title | `claude/<loop>-<short-name>` | #341 (merged) | `plan.rs` | `session_…` or `inline` | #350 | in review |
```

- **id**: `T<n>`. Number the tickets for each loop, and never reuse a number.
  Say "confit T4" or "native T1".
- **branch**: `claude/<loop>-<short-name>`. A branch that existed before this
  rule keeps its name.
- **overlaps**: the files that this ticket and another open ticket both
  change. Merge the smaller ticket first.
- **state**: one of `ready`, `in progress`, `in review`,
  `changes requested` or `blocked: <what>`.

Below the table, the board has these parts:

- A "Next up" line. It names the ticket that starts when the number of
  workers falls below the limit.
- One `## T<n>: <title>` section for each ticket. The section holds the
  exact text that the worker receives.

When the PR of a ticket merges, delete its row and its section. Then move
what the ticket taught into `PLANS.md`.

## 4. Launching a worker

Call `create_session` (Claude Code Remote) with these parameters:

- `source_url`: `https://github.com/ahrzb/sql-transforms`. The worker
  inherits the environment of this session.
- `title`: `<loop> T<n>: <title>`.
- `prompt`: the text of [`worker-brief.md`](worker-brief.md), then the text
  of `<loop>/worker-brief.md`, then the section of the ticket from
  `tickets.md`. The worker starts with nothing else, so the prompt must stand
  alone.

Record the session id on the board. Record the PR number when the PR exists.

Workers cannot send messages back. Follow a worker in these ways:

- Follow its PR. Subscribe with `subscribe_pr_activity`.
- Call `get_session`. The `status_bucket` reads `failed` when a turn errored.
- Read the events of the session.

To correct a worker, use `send_message`. To stop a worker, use
`interrupt_session`. You can also take its ticket back inline.

## 5. Review

Review every worker PR as if you wrote it. Check the diff and its evidence,
not the description alone. Check the scope first. The PR must hold the
ticket and nothing else, and a surprise refactor goes back to the worker.
Then check the checklist of the loop. Run again anything that looks wrong.

**Delivering a review.** Give feedback as GitHub review comments. Mark each
comment **blocking** or **optional**. In each comment, say the file, what is
wrong, and what would satisfy you.

The supervisor and the workers share one GitHub account. For this reason, a
review can only be COMMENT, and never "changes requested". Also, the harness
of a worker may skip the review, because it comes from the account that
posts the worker's own comments. So always also deliver the review to the worker directly:

- If the session has `send_message`, use it.
- If not, fire a one-shot routine into the session of the worker. Call
  `create_trigger` with `persistent_session_id` set to the worker and
  `run_once_at` set to one minute later. The prompt of the routine says that
  the review is the supervisor's, names the supervisor, and lists the
  blocking items.

The worker answers each comment with a push or with a reply that gives its
reason. If a blocking comment cannot be settled, finish the fix yourself on
the branch of the worker, with an ordinary commit. Never force-push the
branch of a worker.

## 6. Merge

- Merge when the review is satisfied and CI is green on the current head.
  Use squash, with `expectedHeadSha`.
- Merge one PR at a time. After each merge, the other open PRs merge master
  in. Never rebase the branch of another agent. The worker merges master in.
  If the merge is trivial, the supervisor may do it. Regenerate generated
  files. Never merge them by hand.
- Then do these steps:
  1. Update the board.
  2. If the PR did not remove the item from `PLANS.md`, remove it.
  3. Run the after-merge steps of the loop.
  4. If the merge serves a need of the other loop, tell that loop.
- A merged PR is final. Make each follow-up a new ticket.
