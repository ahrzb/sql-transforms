# How the confit loop runs

The confit loop turns `PLANS.md` into merged PRs, one verified change at a
time: it implements open work, checks it against the DuckDB oracle, merges it,
and moves on. This page describes how one agent drives that loop. It covers
both modes it can run in, how work is split into tickets, and how workers are
briefed, reviewed and merged.

The agent reading this is assumed to be as capable as the one that wrote it.
These are working rules and the reasons behind them, not a script. When a rule
and the situation disagree, follow the reason and record the decision in the
PR.

Related files:

- [`worker-brief.md`](worker-brief.md): the brief every worker gets, ahead of
  its ticket.
- [`tickets.md`](tickets.md): the live ticket board.
- [`../../PLANS.md`](../../PLANS.md): the open work, in priority order.
- [`../goal.md`](../goal.md): the goal and the contract that judges the work.

## 1. Two modes

The owner chooses the mode, usually by budget. In either mode the
**definition of done** is the same:

- the gate is green;
- lint passes;
- a campaign shows no gated class;
- `PLANS.md` and the docs are updated;
- the PR is squash-merged.

### Inline mode (the default when credits are tight)

The supervising agent does every ticket itself, one at a time, on its own
branch, using the cycle below.

1. Pick the top `PLANS.md` item.
2. Measure DuckDB first.
3. Implement the change.
4. Run the gate, lint, and a 10–20k-seed campaign.
5. Open the PR and wait for CI.
6. Squash-merge with the expected head SHA.
7. Reset the branch onto master.

Inline mode costs the least. It also keeps everything in one context, so no
one has to write a hand-off. Use it for:

- small items;
- items that touch everything;
- the moments between waves.

### Subagent-driven mode (when the owner asks for parallelism)

The agent becomes a **supervisor**:

1. It splits work into tickets (§2).
2. It launches one **worker** per ticket (§3).
3. It reviews each worker's PR and gives feedback (§4).
4. It merges the PRs one at a time (§5).

The supervisor still does some work itself:

- small tickets;
- nightly triage;
- requests from the native catalog loop.

It never lets its own work block a review.

Owner rulings in force (2026-10-05):

- one branch per ticket;
- the parallelism suits the tickets: more when they are independent, fewer
  when they collide;
- the supervisor reviews, gives feedback and merges;
- cost is acceptable for now.

### Switching modes

A switch is safe at any point, because the state lives in GitHub and in the
repo files, not in anyone's head:

- open PRs;
- `tickets.md`;
- `PLANS.md`.

**Going to inline mode:**

1. Stop launching workers.
2. Let running workers finish their PRs, or interrupt them.
3. Take over the open PRs yourself. Their branches are ordinary branches.

**Going to subagent-driven mode:**

1. Write the next tickets into `tickets.md`.
2. Launch workers for them.

When the account warns that a usage limit is close, say so to the owner and
drop to fewer workers or to inline mode.

## 2. Splitting work into tickets

A ticket is the unit a worker can finish alone: one PR, reviewable in one
sitting, with an acceptance criterion a reviewer can check.

**Where tickets come from.** Sources in priority order:

1. **Correctness first.** Bugs and live campaign or nightly divergences come
   before anything else (gated classes in issue "Nightly campaign findings").
   These are usually small, and inline work suits them.
2. **Needs from the native catalog loop.** `packages/sql-transform` keeps a
   "Needs from confit" list and sends messages. These unblock another loop,
   so they come ahead of the query classes. Confirm each one by reproducing
   it before ticketing it.
3. **The ruled query classes**, in the ruled order
   (`docs/decisions/closed/next-query-classes.md`). A class is usually several
   tickets, in slices.
4. **Performance and engineering items** from `PLANS.md`.

Owner decisions (`docs/decisions/open/`, issues titled "Decide: …") are
**never** tickets: the agent prepares the evidence and the owner rules.

**How to split.**

- **One ticket, one concern, one PR.** If the PR description would need two
  headlines, it is two tickets.
- **Slice big classes so each slice ships something served and tested.** For
  example, the i128 class was sliced like this:
  1. the unsigned widths that fit the i64 lane (inline, PR #341);
  2. UBIGINT/HUGEINT on a 128-bit lane (worker T3);
  3. later, exact wide aggregates.
- **Prefer tickets that touch disjoint files.** The hot files collide most:
  - `frontend/expr.rs`, `frontend/typing.rs` and `frontend/functions.rs`;
  - `plan.rs`;
  - `lower.rs`;
  - `exec/cranelift.rs`.

  Two tickets in the same hot file can still run in parallel when they touch
  different functions. Note the overlap in `tickets.md` and merge the smaller
  one first.
- **Size.** A ticket should be done in roughly 1–4 hours of agent work,
  including measuring and a campaign. Larger than that, split it. Smaller
  than about 20 minutes, do it inline.
- **Every ticket states:**
  - the goal, with measured numbers when the motive is performance;
  - pointers to the code it touches;
  - what to measure in DuckDB first;
  - what must refuse by name;
  - the acceptance criteria;
  - its branch name, `claude/<short-name>`.

**How much parallelism.**

- **Count the tickets that touch disjoint code and can merge in any order.**
  That number is the ceiling.
- **The supervisor must review everything.** Above about 3–4 open PRs,
  reviews become the bottleneck and the quality of review drops.
- **A ticket that depends on an unmerged PR waits.** Branch it from master
  only after that PR merges. Otherwise it rebases onto moving code.
- **Worker containers are separate**, so CPU is not shared. This container
  has 4 cores, and a gate run takes about 3 minutes. Run workers as separate
  cloud sessions (§3), not as local subagents that share this machine. Local
  subagents (the Agent tool, optionally in a worktree) suit research and
  read-only investigation.

## 3. Launching a worker

Use `create_session` (Claude Code Remote) with:

- `source_url` = `https://github.com/ahrzb/sql-transforms`, which inherits this
  session's environment;
- `title` = `confit T<n>: <ticket title>`;
- `prompt` = the full text of `worker-brief.md`, then the ticket text from
  `tickets.md`. The worker starts with nothing else, so the prompt has to
  stand alone.

Record the session id, the branch and, later, the PR number in `tickets.md`.

Workers cannot message back. Follow them through:

- their PRs (subscribe to each with `subscribe_pr_activity` once it exists);
- `get_session`, whose `status_bucket` reads `failed` when a turn errored;
- the session's events, when you need more detail.

A worker that stalls or goes off track can be interrupted
(`interrupt_session`), given a correction (`send_message`), or have its
ticket taken back inline.

## 4. Review

Review every worker PR as if you had written it, against the diff and its
evidence, not the description alone.

1. **Scope.** The PR does the ticket and nothing else. Surprise refactors go
   back.
2. **Oracle discipline.** DuckDB's behaviour was measured, not assumed. The
   tests are parity tests (`fuzz.parity.assert_parity`). Anything not
   reproduced exactly refuses by name.
3. **Trap semantics.** Where a change moves, shares, drops or reorders a
   computation, check that every row that trapped before still traps with
   the same class of error, and that no new trap appears.
4. **Evidence.** The PR shows:
   - before/after numbers for performance work;
   - gate output;
   - a campaign of at least 10k seeds with no gated class;
   - a seed-gated run counted by verdict, for generator changes.

   Re-run anything that looks off. CI only runs the gate, not a campaign.
5. **Generator hygiene.** Changes are seed-gated with their own RNG, so no
   other seed moves.
6. **Docs.** `PLANS.md` and `known-limitations.md` are updated, and the tests
   pin the new boundary.

Give feedback as GitHub review comments on the PR, which the worker is
subscribed to. Mark each comment either blocking or optional, and be
specific: the file, what is wrong, and what would satisfy you. The worker
answers each comment with a push or a reasoned reply. When a comment is
blocking and the worker cannot settle it, finish the fix yourself on their
branch with an ordinary commit; never force-push a worker's branch.

## 5. Merge

- **Merge once the review is satisfied and CI is green on the current head.**
  Squash-merge with `expectedHeadSha`.
- **Merge one at a time.** After each merge, the other open PRs merge master
  in, never by rebasing someone else's branch. The worker on that branch does
  it; or the supervisor does when it is trivial.
- **After a merge, give the other loop what it needs.**
  - When the merge unblocks the catalog loop, tell it with `send_message` to
    that loop's session (see PLANS "For the native catalog").
  - Remove the item from `PLANS.md`, if the PR did not already.
  - Update `tickets.md`.
- **A merged PR is final.** Follow-ups are new tickets.

## 6. Standing duties (both modes)

- **Nightly triage.** Read the latest "Nightly campaign findings" comment,
  then re-run each gated seed on master. Fix live classes, or pin them
  xfail-strict in `tests/test_open_divergences.py`. Re-run the metamorphic
  seeds too.
- **The native catalog loop.** Reproduce each request it sends, then fix it
  inline or ticket it. Tell that session when a fix lands.
- **Reports.** Write one when a trigger in
  [`docs/loop-reporting.md`](../../../../docs/loop-reporting.md) fires, in
  `docs/reports/`, opening with that page's KPI block and ending with what
  needs the owner. Title the PR `report: ...`. Never one per PR.
- **Owner questions.** OPT_EMULATED classes, exclusions and open decisions go
  to the owner with evidence, never into code.
- **Hygiene.**
  - Commit and push before ending a turn; the stop hook checks this.
  - Never put model names in repo artifacts.
  - Use the commit trailers in `worker-brief.md`.
  - Never run `pkill -f python`, which kills your own shell.
