#!/usr/bin/env bash
# Usage: scripts/lint/rust.sh [--staged | native | device | wasm]
# No argument runs every scope; `device` and `wasm` need the patched checkouts from ./setup-wasm-deps.sh.
set -euo pipefail
cd "$(dirname "$0")/../.."
scope=${1:-all}
if [[ "$scope" == --staged ]]; then
    git diff --cached --name-only --diff-filter=ACMRD -- 'src/*.rs' 'src/**/*.rs' 'preprocessing/*.rs' 'Cargo.toml' 'Cargo.lock' 'rust-toolchain.toml' 'scripts/lint/rust.sh' | grep -q . || exit 0
    command -v cargo > /dev/null 2>&1 && cargo clippy --version > /dev/null 2>&1 && cargo fmt --version > /dev/null 2>&1 || exit 0
    scope=all
fi
device_features="trace-commit-device,digit-range-device,relation-range-device"
case "$scope" in
    all | native)
        cargo fmt -q --message-format=short -- --check
        cargo clippy -q --all-targets --features native --message-format=short -- -D warnings
        [[ "$scope" == native ]] && exit 0
        ;;
    device | wasm) ;;
    *)
        echo "usage: $0 [--staged | native | device | wasm]" >&2
        exit 2
        ;;
esac
if [[ ! -d .wasm-deps ]]; then
    if [[ "$scope" == all ]]; then
        echo 'skip: patched checkouts absent; WASM bindings, tracing, GPU modules, and reference devices are not linted'
        exit 0
    fi
    echo "error: scope '$scope' needs ./setup-wasm-deps.sh" >&2
    exit 1
fi
if [[ "$scope" != wasm ]]; then
    cargo clippy -q --all-targets --features "native,$device_features" --message-format=short -- -D warnings
fi
if [[ "$scope" != device ]]; then
    RUSTC_BOOTSTRAP=1 cargo clippy -q --lib --target wasm32-unknown-unknown -Z build-std=panic_abort,std --features "webgpu,$device_features" --message-format=short -- -D warnings
fi
