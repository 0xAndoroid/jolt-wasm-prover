#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
if [[ "${1:-}" == --staged ]]; then
    git diff --cached --name-only --diff-filter=ACMRD -- 'src/*.rs' 'src/**/*.rs' 'preprocessing/*.rs' 'Cargo.toml' 'Cargo.lock' 'rust-toolchain.toml' 'scripts/lint/rust.sh' | grep -q . || exit 0
    command -v cargo > /dev/null 2>&1 && cargo clippy --version > /dev/null 2>&1 && cargo fmt --version > /dev/null 2>&1 || exit 0
fi
cargo fmt -q --message-format=short -- --check
cargo clippy -q --all-targets --features native --message-format=short -- -D warnings
if [[ -d .wasm-deps ]]; then
    device_features="trace-commit-device,digit-range-device,relation-range-device"
    cargo clippy -q --all-targets --features "native,$device_features" --message-format=short -- -D warnings
    RUSTC_BOOTSTRAP=1 cargo clippy -q --lib --target wasm32-unknown-unknown -Z build-std=panic_abort,std --features "webgpu,$device_features" --message-format=short -- -D warnings
else
    echo 'skip: patched checkouts absent; WASM bindings, tracing, GPU modules, and reference devices are not linted'
fi
