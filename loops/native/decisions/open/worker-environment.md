# The worker environment: a setup script for uv and Python 3.14.8

**Question.** Will the owner add a setup script to the cloud environment?
The script installs a current uv, the Python package manager that the
repository uses, and CPython 3.14.8 in each new session.

**Why it matters.** Some new sessions in this environment start with uv
0.8.17. That version of uv offers only CPython 3.14.0rc2, a release
candidate of Python 3.14. The repository asks for Python 3.14
(`.python-version`), and the lock file `uv.lock` pins pydantic 2.13.4.
`sql_transform` imports pydantic, and that pydantic does not import on
3.14.0rc2. So `sql_transform` does not import, and the gate cannot run.

**Evidence.**

- The four workers of the wave of 2026-10-05 each installed a newer uv and
  CPython 3.14.8 before they could run the gate. The milestone report of
  2026-10-06 records this as finding 50.
- The research session of PR #404 could not import `sql_transform`,
  because pydantic fails on 3.14.0rc2. The parity test of the catalog
  (`native.check`) needs that import. So that session could not run the
  parity test on a family that has an ulp bound above 0.
- The fix works. An agent installed uv 0.12.23 and CPython 3.14.8 by hand
  in this supervisor session, and the gate passes here.
- One new session does not have the problem. This is the session of the
  project thread "Transforms Loop". It started on 2026-10-06 at 03:12 UTC,
  in the same environment, and its container image has uv 0.11.32. In
  that session, `uv sync --locked` installed CPython 3.14.6. Then
  `sql_transform` imported, and the parity tests passed. So the version of
  uv may depend on how a session starts. A setup script makes all sessions
  the same.

**Options.**

1. **Add a setup script.** The owner adds it in the project's settings.
   Open Project settings, then the Cloud environment menu, then the gear
   next to the selected environment, then Setup script. New sessions then
   run the script.

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
