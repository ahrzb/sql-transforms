# May a delegate run several tasks at once through helpers?

**Asked by:** the session that moved the loops into mandates, 2026-10-17. **Holds:** starting runbook helpers on more than one task at a time, under either mandate.
**Against:** `40f6ab17`, the master commit before the move.

## Context

Before the move, each mandate was an agent loop with two modes:

- In inline mode, the loop session did one task at a time itself.
- In worker-driven mode, the loop session started one worker session for each task. It allowed up to 4 open pull requests. It reviewed and merged them one at a time.

On 2026-10-05, the maintainer ruled to use as much parallel work as the tasks allow.

The `repo-docs` skill now governs the mandates. Its mandate rules say "Work on one task at a time". They also say "Supervising helpers is part of your current task, never a reason to start another." Thus a delegate can use helpers only inside its current task, and worker-driven mode does not fit.

Until the maintainer answers, both mandates work on one task at a time. A runbook helper may still do the current task.

The maintainer is designing the review structure and the two modes himself (2026-10-17). This question asks only what happens to worker-driven mode at the move.

## Read first

- [Native catalog](../../mandates/native/README.md): its **Limits** allow 4 claimed tasks.
- [The native loop review](../../research/native-loop-review/README.md), learnings 1 and 2: review in worker-driven mode, and what changed when the loop went inline.

## Options

### An exception for parallel helpers

Add this to `AGENTS.md`: "Exception to repo-docs **Working on tasks**, 'Work on one task at a time': a delegate may supervise one runbook helper for each claimed task, up to the **Limits** in its mandate's README. While the helpers write, the delegate does no task's work itself."

- **Gains:** the mandates keep the parallel work of worker-driven mode. On 2026-10-05, that mode took native coverage from 1 to 24 of the 68 classes in scope.
- **Costs:** the supervisor reviewed the workers' changes, and `repo-docs` does not count that review as independent. Merges of parallel branches also conflict in shared files, such as the catalog fixtures.

### One task at a time

Keep the `repo-docs` rule. Worker-driven mode ends.

- **Gains:** no exception, and less state to track.
- **Costs:** less parallel work. After 2026-10-06, the native loop already worked one task at a time.

### One task for each group of tasks

Make one task that lists several families, and give each family to a helper.

- **Gains:** no exception.
- **Costs:** such a task does not fit one fresh session, which **Working on tasks** requires.

## Recommendation

Add the exception for parallel helpers, and decide it together with the review structure, because the two depend on each other. This changes if that structure needs an independent reviewer for each task, and the delegate cannot start them in parallel.

## Answer

