#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
if [[ "${1:-}" == --staged ]]; then
    git diff --cached --name-only --diff-filter=ACMRD -- 'src/*.rs' 'src/**/*.rs' 'preprocessing/*.rs' 'Cargo.toml' 'Cargo.lock' 'rust-toolchain.toml' 'scripts/lint/rust.sh' | grep -q . || exit 0
    command -v cargo > /dev/null 2>&1 && cargo clippy --version > /dev/null 2>&1 && cargo fmt --version > /dev/null 2>&1 || exit 0
fi
cargo fmt -q --message-format=short -- --check
cargo clippy -q --all-targets --features native --message-format=short -- -D warnings
