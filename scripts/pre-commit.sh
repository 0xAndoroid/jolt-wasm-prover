#!/usr/bin/env bash
# Repo-local pre-commit entry point, invoked by the universal ~/.git-hooks/pre-commit.
# Runs every executable scripts/lint/*.sh with --staged; each script decides whether any
# staged file concerns it and exits 0 silently otherwise.
set -euo pipefail

cd "$(dirname "$0")/.."
status=0
for script in scripts/lint/*.sh; do
    [ -x "$script" ] || continue
    if ! "$script" --staged; then
        echo "pre-commit: $script failed" >&2
        status=1
    fi
done
exit "$status"
