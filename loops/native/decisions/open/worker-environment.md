# The worker environment: a setup script for uv and Python 3.14.8

**Question.** Will the owner add a setup script to the cloud environment?
The script installs a current uv, the Python package manager that the
repository uses, and CPython 3.14.8 in each new session.

**Why it matters.** A new session in this environment starts with uv
0.8.17. That version of uv offers only CPython 3.14.0rc2, a release
candidate of Python 3.14. The repository asks for Python 3.14
(`.python-version`), and the lock file `uv.lock` pins pydantic 2.13.4.
`sql_transform` imports pydantic, and that pydantic does not import on
3.14.0rc2. So `sql_transform` does not import, and the gate cannot run.

**Evidence.**

- The four workers of the wave of 2026-10-05 each installed a newer uv and
  CPython 3.14.8 before they could run the gate. The milestone report of
  2026-10-06 records this as finding 50.
- The research session of PR #404 could not import `sql_transform`. The
  import fails inside pydantic, at `_eval_type(prefer_fwd_module=)`.
  `native.check`, the parity test of the catalog, needs that import. So
  that session could not run the parity test on a family with an ulp bound
  above 0.
- The fix works. An agent installed uv 0.12.23 and CPython 3.14.8 by hand
  in this supervisor session, and the gate passes here.

**Options.**

1. **Add a setup script.** The owner adds it in the environment's settings:
   the cloud environment menu in the session's title bar, then Edit, then
   Setup script. New sessions then run it.

   ```sh
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ~/.local/bin/uv python install 3.14.8
   ```

2. **Keep the environment.** Each ticket keeps a note that says how to
   install both. Each worker then spends time on the install before its
   real work.
3. **Pin a pydantic that imports on 3.14.0rc2.** This changes the lock file
   for a release candidate. The loops do not own the dependencies.

**Recommendation.** Option 1. It costs one change in the settings, and it
removes the same failure from every new session of both loops.
