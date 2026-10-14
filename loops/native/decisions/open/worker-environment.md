# The worker environment: a setup script for uv and Python 3.14.8

**Question.** Will the owner add a setup script to the cloud environment?
The script installs a current uv and CPython 3.14.8 in each new session.
uv is the Python package manager of the repository. CPython is the
standard Python interpreter.

**Why it matters.** Some new sessions in this environment start with uv
0.8.17. That version of uv offers only CPython 3.14.0rc2, a release
candidate of Python 3.14. The repository asks for Python 3.14
(`.python-version`), and the lock file `uv.lock` pins pydantic 2.13.4.
`sql_transform` imports pydantic, and that pydantic does not import on
3.14.0rc2. So `sql_transform` does not import, and the gate cannot run.

**Evidence.**

- The four workers of the wave of 2026-10-05 each installed a newer uv and
  CPython 3.14.8 before they could run the gate. The native catalog's
  [reading 3](../../reports/2026-10-06-reading-3.md) records this as
  finding 50.
- The session that researched the owner's rulings of 2026-10-06 (PR #404)
  could not import `sql_transform`, because pydantic fails on 3.14.0rc2.
  So that session could not run the catalog's parity test
  (`native.check`) on a family with an ulp bound above 0.
- The fix works. On 2026-10-06, an agent installed uv 0.12.23 and CPython
  3.14.8 by hand in the native loop's supervisor session of that day, and
  the gate passed there.
- One new session does not have the problem. This is the session of the
  project thread "Transforms Loop", from which the native loop runs. It
  started on 2026-10-06 at 03:12 UTC, in the same environment, and its
  container image has uv 0.11.32. In that session, `uv sync --locked`
  installed CPython 3.14.6. Then `sql_transform` imported, and the parity
  tests passed. So the version of uv may depend on how a session starts.
  A setup script makes all sessions the same.

**Options.**

1. **Add a setup script.** The owner adds it in the project's settings.
   Open Project settings, then the Cloud environment menu, then the gear
   next to the selected environment, then Setup script. New sessions then
   run the script.

   ```sh
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ~/.local/bin/uv python install 3.14.8
   ```

2. **Keep the environment.** Each ticket keeps a note on how to install uv
   and CPython 3.14.8. Each worker then spends time on the install before
   its real work.
3. **Pin a pydantic that imports on 3.14.0rc2.** This changes the lock file
   only for a release candidate of Python. The loops do not own the
   project's dependencies.

**Recommendation.** Option 1. It costs one change in the settings, and it
removes the same failure from every new session of both loops (the native
loop and the confit loop).

**Facts since #428 (2026-10-14).** The owner's PR #428 runs each cloud
session in a Nix dev shell. Nix is a package manager, and the dev shell is
a shell whose tools, such as uv and CPython, come from Nix, at the
versions that `flake.nix` and `flake.lock` pin. At the start of each session, a hook
(`.claude/hooks/session-start.sh`) enters the dev shell. Then it installs
the project's Python packages with `uv sync`.
[docs/cloud-environment.md](../../../../docs/cloud-environment.md) gives a
setup script for the cloud environment.

The session of the "Transforms Loop" project thread found these facts
after the owner resumed the native loop on 2026-10-14:

- In that session, the dev shell provided uv 0.12.22 and CPython 3.14.7.
  Both come from Nix. The project's Python environment (`.venv`) uses that
  CPython. `sql_transform` imports, and the gate passed on the loop's two
  PRs of that day (#437 and #439).
- The failure of this record did not occur in that session. Its CPython
  was 3.14.7, not 3.14.0rc2.
- Nix builds its CPython 3.14.7 without profile-guided optimization. This
  build option makes CPython faster: the build runs a test workload, and
  then compiles CPython again for the code paths that the workload used.
  uv's CPython 3.14.6 has the option. Python's build flags
  (`sysconfig.get_config_var("CONFIG_ARGS")`) show both facts.
- The native catalog measured the test step with the most output fields of
  each of its 34 classes on both CPythons. On the CPython of Nix, a 64-row
  call of a twin took a median of 1.23 times as long (0.98 to 1.67 times).
  A native call took 1.05 times as long. These are three runs on each
  CPython, on one machine, on 2026-10-14
  ([reading 5](../../reports/2026-10-14-reading-5.md), section 3).
- In the dev shell, the command that builds confit's Rust extension
  (`maturin develop --release`) stops. It installs the extension with uv,
  and uv does not install into the CPython of Nix. Nix marks that CPython
  as "externally managed", which means that no package manager may install
  into it. With `--skip-install`, maturin builds the extension in confit's
  source folder, and `.venv` loads it from there.

#428 may answer the question of this record. If the owner rules that #428
answers it, this record moves to `closed/`, with #428 as the ruling.
