# What the sql-transforms native loop teaches about running a loop

**Question:** how did the autonomous native-transform loop in `sql-transforms` actually get work done between 2026-10-05 and 2026-10-17? What caused its bugs and rework, and what should change so that a loop does better?

**Feeds:** the [native catalog](../../mandates/native/README.md) and [confit engine](../../mandates/confit/README.md) mandates and the open [questions](../../questions/README.md). A copy also sits in the maintainer's skills repository, as input to his mandate design.

**Read:** 2026-10-17, on `sql-transforms` master `40f6ab17` (#513). Sources: git history and PR bodies for `loops/` and `packages/sql-transform/`, the GitHub PR conversations, the 8 milestone readings, and the live code.

**Method:** a cheap read-only scout wrote the factual history. Then 12 reviewers critiqued it in four categories, three models per category: Astra (gpt-6-astra), Opus (claude-opus-5-5) and Muse (muse-wonder-full-1.0). The reviewers checked the history against the sources and corrected it. Some ran their own reproductions on the maintainer's Windows host. The supervisor spot-checked the claims that most depend on: #513 has 0 comments, 0 reviews and branch `claude/project-thread-2pg6vk`; #367 has 3 review comments; `check_test.py:206` treats −0.0 and 0.0 as one value; `mandates/native/brief.md` has had only a move and a path fix since #366. Other reviewer measurements are the reviewers' own and are marked as such below.

## Files

| File | What it holds |
|---|---|
| [history.md](history.md) | The scout's timeline, process description and bug table, plus metrics the supervisor computed with git and gh. It has known errors; the corrections are below. |
| [reviews.md](reviews.md) | The 12 reviews, verbatim, in four categories: correctness (A), process (B), cross-loop with confit (C), the loop's instructions (D) |
| [scripts/](scripts/) | Opus's row-census scripts behind the "NULL-id rows" finding |

## The loop in one paragraph

A supervisor session grows a catalog that translates fitted sklearn transformers into SQL. Each entry must equal its "twin" (the sklearn transformer itself) bit-exactly, or within a parity bound. Work is split into tickets T1–T58: one family per PR, with a gate and a squash merge. Milestone "readings" are reports that measure every success measure on one master commit. Owner decision records hold the questions that need the owner. The engine (confit) has its own loop, and the native loop sends it "needs". Over 12 days the loop took coverage from 1 to 51 of 68 classes in 94 PRs. It paused 8 days for the owner (10-06 → 10-14).

## Learnings for any loop

These generalise beyond sql-transforms. Each cites the evidence in reviews.md.

1. **Review must follow the risk of a change, not the execution mode.** Worker mode had real review. #367 caught a probe that raised on DuckDB newer than 1.5.5, and #384 blocked 21–46 s builds that had no cap. When the loop switched to one inline session (about #405), roughly 50 PRs merged with 0 comments. The "12 min median open→merge" was just CI time. The milestone readings became the review, one wave late. Bugs that got through this way: T37→T38 and T41→T43 (signed zero), T44→T46 (builds about 4× slower) and T49/T50→T53 (an uncapped 28.5 s build). (B: all three; A: Opus)
2. **A mode switch silently drops the rules attached to the other mode.** Ticket text stopped after T22. The board turned into a diary. ~45 PRs came from one branch, against the one-branch-per-ticket rule. Nobody recorded what was lost. (B, D: Opus)
3. **Lessons must have a home that the next worker's prompt reads.** 92 numbered findings stayed in readings and PLANS prose; PLANS grew from 1,163 to 5,776 words. The worker brief kept the 10-05 bound rule ("measure the max ulp … register that") after the owner's 10-06 ruling changed it. Proposed fix: a traps file distilled from the findings. A PR that fixes a repeatable finding adds a line to it. A PR that changes the contract updates the brief in the same PR. (D: all three)
4. **State one rule in one place.** Rules copied 2–4 times drifted into contradictions. The cycle says "squash-merge" while the shared brief says "you never merge". There were two different rules for shared machinery. The brief banned model names in commits and then required a model-name trailer. The #410 report rework came from one rule split across two files. (D: all three)
5. **"Follow the reason" needs reasons on the page and a ratchet.** The simple-English rewrite (#401/#408) grew the prompts by 15–33% and deleted the parentheticals that gave the reasons (Opus measured this). Departures from a rule were silent, or recorded only in PR bodies that no later cycle reads. Proposed: a departure that happens twice amends the rule, and a rule that the environment makes impossible is changed. (D: Opus, Muse)
6. **Owner prose and agent prompts are different audiences.** Applying owner-reading style to the loop's cycle steps cost tokens and removed reasons. The owner's experience did not improve, because the owner reads reports, not cycle steps. (D: Opus, Muse; this is an owner decision)
7. **A finding should end as a tool or test, not as another ticket.** Each of 7+ build-estimate refits rebuilt its measurement scripts outside the repo (`/mnt/project-files/…`). Fix tickets of the same kind kept coming back. Promote each recurring check into the repo: a script for each number, a test for each class of bug. (B: Opus; C: all three)
8. **Evidence must be rerunnable from the repo.** PR bodies and readings cite scripts outside git, so a reviewer or the owner cannot reproduce the numbers. Proposed rule: every number cites a script in the repo at that head. (B: Opus; C: Muse, Astra)
9. **Watch what a check actually compares, not how many seeds it runs.** The gate passed 37 steps that it had compared only on NULL-id rows (Opus measured this with [scripts/](scripts/)). It treated −0.0 and 0.0 as equal under any nonzero bound. And it accepted overflowed bounds where a long double is a double, which includes Windows (Astra reproduced `near(1.4e154, 1.5e154, 1, square) → True`). More seeds could not find any of these. (A: all three)
10. **Generators grow after the bug.** ±inf, booleans, all-missing strings and a fixture that depended on global RNG were each added only after a bug bit. Proposed: generic rows derived from the fitted state (knots, edges, ±1 ulp), a ladder of magnitudes and widths, and a property test that a translation either serves or refuses by name. (A: all three)
11. **Make "never widen the acceptance predicate" mechanical.** Bounds were plain code literals, so a raised bound looked like any other edit. T54 replaced an owner-ruled S with no record. Proposed: a generated bounds manifest whose growth blocks a merge unless a closed record covers it. (A: all three)
12. **A loop that depends on another loop needs a cost contract, not only a request channel.** Confit's #363 slowdown was flagged in its review as "optional"; #456 carried no timing numbers at all. The native loop found both later, by hand, in its readings. Needs were prose kept in two PLANS files, and two needs were never sent. Proposed: ratio cost tests in the upstream gate, a "cost notice" in the other direction, and one needs registry with ids and statuses. (C: all three)
13. **A workaround of an upstream heuristic needs a lifecycle.** T44's `x - 0.0` spelling exploits confit's internal `MIN_SHARED_SIZE = 6`. It caused the T45, T46 and T48 rework, and the need behind it was then marked "no longer needed". Keep such spellings in one place, pin the upstream behaviour with a test, and keep the need open. (C: Opus, Muse, Astra)
14. **Caps hide the cost they create.** Confit ranked native needs "low priority (the catalog builds within its caps)". That reasoning is circular, because the caps exist to refuse the configurations the need would fix. Rank needs by measured cost. (C: Opus, Astra)
15. **Reports can become the most expensive output.** The readings total about 71k words, and reading 8 alone is 18k. Opus infers about 13 h of 50 active inline hours sat between the last ticket merge and the reading merge. Once review returns, readings can shrink to the KPI block, findings, needs and next steps. (B: all three)
16. **Easiest-first picking leaves the hard tail unexamined.** The 17 classes left are labelled "needs a design" with no triage. Opus says NCA and BernoulliRBM are matvec forms that are cheap. Also, the coverage KPI counts the 19 classes that need `allow_bound` the same as bit-exact ones. (B: Muse, Opus)
17. **Orphans appear at pauses.** PR #432 has been open since 10-06, from another session, and is not on the board. Resuming should reconcile open PRs with the board. (B: all three)

## Where the reviewers disagree

| Topic | Positions |
|---|---|
| Signed zero under a bound | Opus/Muse: make it a hard failure. Muse limits it to where every product is exactly zero, because the sign of an underflow depends on BLAS. Astra: the contract allows it today, so track it separately. |
| Automating refits | Muse/Opus: script the refits. Astra: compare and explain a cost change before calibrating, or automation accepts upstream regressions faster. |
| Share of tickets that are rework | The history's "28%" mixed causes. True estimate refits were about 15%. But only 17 of 46 tickets from T13 to T58 added a class (Opus). |
| The 8-day pause | Muse: a decision bottleneck. Astra: the owner asked for it. |

## Corrections to history.md

- T38 and T43 were not "detected by 200-seed parity". Parity passed both, by construction, and scripts outside git found them.
- T26 followed native's own T25 guard, not confit #430. It merged before #430, and T28 undid it. T46 came from native's T44, and T53 from a missing cap. Only T13, T28, T39 and T40 followed confit changes.
- #456→#461 took 4 h 7 min to confit's fix, and about 7.4 h to native's last refit. Not "~8 h".
- T58's cross-host evidence is 8 BLAS setups on one machine, not a second host.
- #415 (±inf in the generator) merged on 10-06, not 10-14.
- Its §2 describes the written review process, not the practised one.
- The git dump given to reviewers filtered on a wrong path (`…/src/sql_transform/native`), so code-file stats were missing and #406 was dropped. The rule-diff dump left out `report-format.md`. Corrected counts: 166 commits, 58 ticket commits.

## Questions for the maintainer that the review raised

These are now in [questions/open/](../../questions/README.md):
- signed zero under a bound;
- the 7 s build budget;
- the writing scope for agent prompts;
- counting bounded classes in the KPI;
- ulp bounds chosen as the measured maximum;
- ratifying T54's Nystroem S.
