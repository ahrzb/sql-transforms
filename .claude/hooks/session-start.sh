#!/bin/bash
# Cloud sessions: enter the flake dev shell (flake.nix) for every Bash
# command, then sync the Python env the way CI does.
#
# The shell's variables are written to $CLAUDE_ENV_FILE, which Claude Code
# sources before each command, so `uv`, `cargo`, `python3` and `java` all
# resolve to the pinned Nix versions without a `nix develop` wrapper.
set -euo pipefail

if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

cd "$CLAUDE_PROJECT_DIR"

if ! command -v nix >/dev/null 2>&1; then
  # The cloud image ships a multi-user Nix install; its profile script puts
  # nix on PATH, but hooks do not run a login shell.
  if [ -e /nix/var/nix/profiles/default/etc/profile.d/nix-daemon.sh ]; then
    . /nix/var/nix/profiles/default/etc/profile.d/nix-daemon.sh
  fi
  export PATH="/nix/var/nix/profiles/default/bin:$PATH"
fi

# Keep only what the dev shell adds or changes. Build-sandbox variables
# (TMPDIR pointing at a throwaway build dir, phases, SOURCE_DATE_EPOCH, ...)
# would break ordinary commands, so they are left out.
nix develop --command env -0 | python3 -c '
import os, shlex, sys

base = dict(os.environ)
skip = {
    "TMPDIR", "TEMP", "TMP", "TEMPDIR", "NIX_BUILD_TOP", "NIX_BUILD_CORES",
    "NIX_LOG_FD", "NIX_STORE", "PWD", "OLDPWD", "SHLVL", "_", "HOME", "SHELL",
    "TERM", "TZ", "LANG", "LC_ALL", "SOURCE_DATE_EPOCH", "PYTHONPATH",
    "PYTHONHASHSEED", "PYTHONNOUSERSITE", "IN_NIX_SHELL", "CONFIG_SHELL",
    "NIX_GCROOT", "NIX_ENFORCE_NO_NATIVE", "DETERMINISTIC_BUILD",
}
for item in sys.stdin.buffer.read().split(b"\0"):
    if not item or b"=" not in item:
        continue
    key, _, val = item.decode().partition("=")
    # Lower-case keys are mkShell derivation attributes (buildPhase, out, ...).
    if key in skip or not key[0].isupper() or key.startswith("BASH_FUNC_"):
        continue
    if base.get(key) == val:
        continue
    print(f"export {key}={shlex.quote(val)}")
' > "$CLAUDE_PROJECT_DIR/.claude/nix-env.sh"

if [ -n "${CLAUDE_ENV_FILE:-}" ]; then
  echo ". \"$CLAUDE_PROJECT_DIR/.claude/nix-env.sh\"" >> "$CLAUDE_ENV_FILE"
fi

. "$CLAUDE_PROJECT_DIR/.claude/nix-env.sh"
# Same groups as CI: the cross-engine gate needs pyspark.
uv sync --locked --group spark
