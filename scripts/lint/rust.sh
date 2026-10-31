#!/usr/bin/env bash
# Usage: scripts/lint/rust.sh [--staged | native | device | wasm | fix [CARGO_ARGS...]]
# No argument runs every scope; `device` and `wasm` need the local overrides ./setup-wasm-deps.sh writes into Cargo.toml.
set -euo pipefail
cd "$(dirname "$0")/../.."

refuse() {
    echo "error: host gate: $1; not starting cargo clippy" >&2
    exit 1
}

# Called immediately before each Clippy build. The shared Mac Studio (ComputerName PikaStudio) starts
# one only while swap used is under 8192M and the 1-minute load average under 36; a failed lookup or
# an unparsable reading refuses. Other hosts are not checked. Comparing integer parts is exact:
# x.yz < N exactly when x < N. LC_ALL=C: sysctl prints its numbers with the locale's decimal separator.
host_gate() {
    local os name swap load swap_re='used = ([0-9]+)\.[0-9]+M' load_re='^[{] ([0-9]+)\.[0-9]+ '
    os=$(uname -s) && [[ -n $os ]] || refuse 'uname -s failed'
    [[ $os == Darwin ]] || return 0
    name=$(LC_ALL=C /usr/sbin/scutil --get ComputerName) && [[ -n $name ]] || refuse 'scutil --get ComputerName failed'
    [[ $name == PikaStudio ]] || return 0
    swap=$(LC_ALL=C /usr/sbin/sysctl -n vm.swapusage) || swap=
    load=$(LC_ALL=C /usr/sbin/sysctl -n vm.loadavg) || load=
    [[ $swap =~ $swap_re ]] && ((BASH_REMATCH[1] < 8192)) && [[ $load =~ $load_re ]] && ((BASH_REMATCH[1] < 36)) && return 0
    refuse "PikaStudio needs swap used under 8192M and load1 under 36, read vm.swapusage '$swap' and vm.loadavg '$load'"
}

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
        host_gate
        cargo clippy -q --all-targets --features native --message-format=short -- -D warnings
        [[ "$scope" == native ]] && exit 0
        ;;
    device | wasm) ;;
    fix)
        host_gate
        cargo clippy -q --all-targets --features native --message-format=short --fix --allow-dirty "${@:2}"
        exit 0
        ;;
    *)
        echo "usage: $0 [--staged | native | device | wasm | fix [CARGO_ARGS...]]" >&2
        exit 2
        ;;
esac
# ./setup-wasm-deps.sh writes this line into Cargo.toml; --revert removes it but leaves .wasm-deps/ behind.
if ! grep -q '^# LOCAL patched checkouts' Cargo.toml; then
    if [[ "$scope" == all ]]; then
        echo 'skip: Cargo.toml has no local overrides from ./setup-wasm-deps.sh; WASM bindings, tracing, GPU modules, and reference devices are not linted'
        exit 0
    fi
    echo "error: scope '$scope' needs the local overrides ./setup-wasm-deps.sh writes into Cargo.toml" >&2
    exit 1
fi
if [[ "$scope" != wasm ]]; then
    host_gate
    cargo clippy -q --all-targets --features "native,$device_features" --message-format=short -- -D warnings
fi
if [[ "$scope" != device ]]; then
    host_gate
    RUSTC_BOOTSTRAP=1 cargo clippy -q --lib --target wasm32-unknown-unknown -Z build-std=panic_abort,std --features "webgpu,$device_features" --message-format=short -- -D warnings
fi
