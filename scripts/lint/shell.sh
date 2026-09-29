#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
if [[ "${1:-}" == --staged ]]; then
    git diff --cached --name-only --diff-filter=ACMRD -- '*.sh' '.editorconfig' | grep -q . || exit 0
    command -v shellcheck > /dev/null 2>&1 && command -v shfmt > /dev/null 2>&1 || exit 0
fi
files=(setup-wasm-deps.sh scripts/build-pages.sh scripts/pre-commit.sh scripts/lint/*.sh)
shellcheck -x "${files[@]}"
shfmt -d "${files[@]}"
