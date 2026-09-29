#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
if [[ "${1:-}" == --staged ]]; then
    git diff --cached --name-only --diff-filter=ACMRD -- '*.py' 'ruff.toml' 'scripts/lint/python.sh' | grep -q . || exit 0
    command -v uvx > /dev/null 2>&1 || exit 0
fi
uvx ruff@0.16.9 check bench
uvx ruff@0.16.9 format --check bench
