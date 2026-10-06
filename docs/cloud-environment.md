# The cloud environment

This page tells how Claude Code cloud sessions get the toolchain of this
repository. The toolchain is Python 3.14, uv, Rust and JDK 21. Nix supplies
all of it through the dev shell in [`flake.nix`](../flake.nix).

## What each part does

| part | where it is | what it does |
|---|---|---|
| Nix | the cloud image | The image has Nix 2.34 with flakes on. No daemon runs. The root user writes to the Nix store directly. |
| Setup script | the environment settings on claude.ai | It installs Nix if the image does not have it. With the current image, it does nothing. |
| Session start hook | [`.claude/hooks/session-start.sh`](../.claude/hooks/session-start.sh) | It enters the dev shell, gives its variables to each command of the session, and runs `uv sync --locked --group spark`. |

The setup script and the session start hook are different things. The
owner sets the setup script in the environment settings, and it runs once
when the container starts. The hook is in the repository, and Claude Code
runs it at the start of each session.

## The suggested setup script

Put this script in the environment settings: Project settings, then the
Cloud environment menu, then the gear next to the environment, then Setup
script.

```bash
#!/bin/bash
set -euo pipefail

# The cloud image has Nix now. This step installs it only if a later image
# does not have it. `--init none` gives a root-only install with no daemon,
# because the container has no init system.
if [ ! -x /nix/var/nix/profiles/default/bin/nix ]; then
  curl -sSfL https://artifacts.nixos.org/nix-installer \
    | sh -s -- install linux --no-confirm --init none
fi
```

The script does not build the dev shell. The session start hook builds it,
because the hook runs in the repository and uses the pinned `flake.lock`.

## Network access

The "Trusted" network access level of the default environment lets Nix
fetch everything that it needs. If you change to "Custom", keep the default
package managers and add these hosts:

- `cache.nixos.org`: the binary cache. Nix downloads built packages from it.
- `channels.nixos.org` and `releases.nixos.org`: the nixpkgs source. The
  channel URL redirects to a fixed release URL, and `flake.lock` holds that
  release URL.
- `artifacts.nixos.org` and `github.com`: the Nix installer. The setup
  script needs these hosts only when it installs Nix.

## Why nixpkgs does not come from GitHub

All network traffic of a cloud session goes through a proxy. For GitHub, the
proxy allows only the repositories of the session, and it refuses other
repositories with HTTP 403. Thus Nix cannot fetch a `github:NixOS/nixpkgs`
flake input. For this reason, `flake.nix` takes nixpkgs from the channels.nixos.org
tarball. Do not add other `github:` inputs to the flake.

## Wheels from PyPI and the Nix Python

uv uses the Python from Nix and does not download its own Python. The
PyPI wheels of pyarrow and duckdb need `libstdc++.so.6`, and the Nix Python
does not search `/usr/lib` for it. On Linux, the dev shell puts the
libstdc++ and zlib libraries of Nix in `LD_LIBRARY_PATH`. Without this
variable, `import pyarrow` fails.

## How we checked it

On 2026-10-06, in a cloud session, we ran these commands in the dev shell:

1. `uv sync --locked --group spark`
2. `pre-commit run --all-files`
3. `scripts/gate.py`. 319 cargo tests and 7636 pytest tests passed. The
   pytest tests include the test that runs printed SQL on a local Spark
   session (`test_dialect_cross_engine_gate.py`). This test needs JDK 21.

We also ran the session start hook from an empty environment. After the
hook, the Spark gate test and `cargo build` passed.
