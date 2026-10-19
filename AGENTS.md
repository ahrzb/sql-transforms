# Instructions for agents

These rules apply to every agent that works in this repository: a delegate of a mandate in [`mandates/`](mandates/README.md), its helpers, and a single session.

1. **Text for the maintainer.** Load the `simple-english` skill
   (`.claude/skills/simple-english/SKILL.md`) before you write such text.
   This is a doc that the maintainer reviews or maintains, a report, a question,
   a proposal, or the summary of a large technical PR. Other text, such as code
   comments, a tracker, a task description and an ordinary PR body, needs only to be clear.
2. **Terms.** [`GLOSSARY.md`](GLOSSARY.md) holds the project's domain terms. Use
   them when you name a concept of the project.
3. **Questions for the maintainer.** Ask the maintainer only through a question in
   [`questions/open/`](questions/README.md), as the `repo-docs` skill says.
   For `sql-transform` work outside a mandate, its
   [decision guide](packages/sql-transform/plan/decisions/README.md) still applies.
4. **Package documents.** `packages/sql-transform/` keeps its documents in
   four folders: `spec/` (what the package does now), `research/` (sourced
   facts), `plan/` (draft RFCs, open decisions and open work) and `records/`
   (closed decisions, applied RFCs, research lessons and the changelog, kept
   as written). The README of each folder gives its rules. Do not commit an
   implementation plan.

## Commands (repo root)

- Install: `uv sync --locked --group spark`. Never drop `--group spark`, because that uninstalls pyspark. Cloud sessions enter the pinned Nix dev shell first, through `.claude/hooks/session-start.sh`.
- Gate: `uv run --no-sync python scripts/gate.py`. It must print `gate: green`.
- Lint: `git add -A` first, because pre-commit skips untracked files; then `uv run --no-sync pre-commit run --all-files`. Read the full output. After an autofix, run it again.
- Never run a command such as `pkill -f python`; it kills your own shell.

## Commits and pull requests

- Never put a model name or identifier in commits, PR text or code, except in the attribution trailer below.
- In a cloud session, end each commit message with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>` and `Claude-Session: <your session URL>`.
- End a PR body with `🤖 Generated with [Claude Code](https://claude.com/claude-code)` and your session URL.
- Commit and push before you end a turn; the stop hook checks it.

## Helpers in cloud sessions

- Start a runbook helper with `create_session`: `source_url` `https://github.com/ahrzb/sql-transforms`, title `<mandate>: <task>`, and a prompt that names the mandate and its task, calls the session a helper, and links the mandate's README, brief and task description.
- Follow a helper through its PR (`subscribe_pr_activity`) and `get_session`; correct it with `send_message`; stop it with `interrupt_session`.
- The delegate and its helpers share one GitHub account, so a review can only be a comment, and the helper's harness may skip it. Deliver each review to the helper directly as well: with `send_message`, or with a one-shot `create_trigger` into its session.
- Never force-push a helper's branch. Finish an unsettled fix on its branch with an ordinary commit.

## Docs

- Before writing, moving or citing meta material, load the `repo-docs` skill.
- Before starting or resuming work under a mandate, also load its `mandates.md`.

Domain words: `GLOSSARY.md`. Meta words: `repo-docs`.
Link check: none. Verification: `uv run --no-sync python scripts/gate.py`, and `uv run --no-sync pre-commit run --all-files`. Review: each merged PR carries a **Review** section from `repo-docs`; independent review only where a mandate's README requires it.
Exception to the repo-docs **Layout**: `packages/sql-transform/` keeps its `spec/`, `research/`, `plan/` and `records/` folders (rule 4), and `packages/confit/docs/` keeps its specs, oracle chapters and reports.
