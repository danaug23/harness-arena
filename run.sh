#!/usr/bin/env bash
#
# Convenience wrapper for harness-arena on Linux and macOS.
#
# Forwards everything to the harness-arena CLI. This exists only so the repo
# works before `pip install -e .`; once installed, `harness-arena <command>` is
# the same thing.
#
#   ./run.sh init                    point it at your model server
#   ./run.sh doctor                  check Docker, Harbor and the endpoint
#   ./run.sh bench --subset stratified-25
#   ./run.sh dash                    live dashboard
#
# Python is located in this order:
#   1. $HARNESS_ARENA_PYTHON, if set
#   2. an active virtualenv / conda env
#   3. python3 / python on PATH

set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

find_python() {
  if [ -n "${HARNESS_ARENA_PYTHON:-}" ]; then
    if [ -x "$HARNESS_ARENA_PYTHON" ]; then
      printf '%s' "$HARNESS_ARENA_PYTHON"
      return 0
    fi
    echo "HARNESS_ARENA_PYTHON is set to '$HARNESS_ARENA_PYTHON' but is not executable." >&2
    return 1
  fi

  for prefix in "${VIRTUAL_ENV:-}" "${CONDA_PREFIX:-}"; do
    [ -n "$prefix" ] || continue
    if [ -x "$prefix/bin/python" ]; then
      printf '%s' "$prefix/bin/python"
      return 0
    fi
  done

  for name in python3 python; do
    if command -v "$name" >/dev/null 2>&1; then
      command -v "$name"
      return 0
    fi
  done

  cat >&2 <<'EOF'
No Python found.

Create the environment first, then re-run:

  conda env create -f environment.yml
  conda activate harness-arena

or with a plain virtualenv:

  python3 -m venv .venv
  source .venv/bin/activate
  pip install -e .

or point this script at an interpreter directly:

  export HARNESS_ARENA_PYTHON=/path/to/python
EOF
  return 1
}

python_bin="$(find_python)"

cd "$root"
# PYTHONPATH so `python -m bench` works from a bare checkout, with no install.
export PYTHONPATH="$root${PYTHONPATH:+:$PYTHONPATH}"
exec "$python_bin" -m bench "$@"
