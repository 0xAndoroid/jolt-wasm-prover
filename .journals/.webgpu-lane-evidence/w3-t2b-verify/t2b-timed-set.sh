#!/bin/bash
# T2b timed set: holds the bench lock for the WHOLE set (mkdir loop), gates
# every run on ioreg GPU-util < 10%, watchdogs each run against the known
# ~1-in-8 GPU-arm 2^22 engine wedge (missed SAB wakeup — hangs, not
# corrupts), and releases the lock on ANY exit.
#
# Usage: PORT=8095 ./t2b-timed-set.sh <results-dir>
set -u
DIR="${1:?results dir}"
PORT="${PORT:-8095}"
BASE="http://localhost:${PORT}"
mkdir -p "$DIR"

LOCK=/tmp/jolt-wasm-bench.lock.d
until mkdir "$LOCK" 2>/dev/null; do sleep 15; done
echo "t2b pid $$" > "$LOCK/owner"
trap '/bin/rm -rf "$LOCK"' EXIT
echo "lock acquired $(date +%H:%M:%S)" | tee "$DIR/log.txt"

# Served-wasm identity gate (foreign-server protection).
A=$(curl -s "$BASE/pkg/jolt_wasm_prover_bg.wasm" | shasum -a 256 | cut -d' ' -f1)
B=$(shasum -a 256 pkg/jolt_wasm_prover_bg.wasm | cut -d' ' -f1)
if [ "$A" != "$B" ]; then
    echo "WASM IDENTITY MISMATCH served=$A local=$B — aborting" | tee -a "$DIR/log.txt"
    exit 2
fi
echo "wasm identity ok $A" >> "$DIR/log.txt"

# Tint vector tier on the final kernels (2 s of GPU, inside the window).
(cd "$HOME/dev/jolt/.worktrees/w3-t2b" \
    && node crates/jolt-kernels/wgsl-check/check.mjs crates/jolt-kernels/target/wgsl-vectors) \
    >> "$DIR/log.txt" 2>&1
grep -E 'Tint parity' "$DIR/log.txt" | tail -1

gpu_gate() {
    for _ in $(seq 1 120); do
        u=$(ioreg -r -d 1 -w 0 -c IOAccelerator 2>/dev/null \
            | grep -o '"Device Utilization %"=[0-9]*' | head -1 | grep -o '[0-9]*$')
        [ "${u:-100}" -lt 10 ] && return 0
        sleep 5
    done
    echo "gpu gate timed out (util=$u)" | tee -a "$DIR/log.txt"
    return 1
}

# one_run <label> <iters> <webgpu:0|1> <timeout-s> <traced:0|1>
one_run() {
    local label=$1 iters=$2 gpu=$3 cap=$4 traced=$5 rc
    gpu_gate || return 1
    echo "=== $label $(date +%H:%M:%S) loadavg $(sysctl -n vm.loadavg)" >> "$DIR/log.txt"
    if [ "$traced" = 1 ]; then
        BENCH_WEBGPU=$gpu BENCH_URL=$BASE PW_CHANNEL=chrome \
            perl -e 'alarm shift; exec @ARGV' "$cap" \
            node trace-chain.mjs "$iters" 1 "$DIR/$label" \
            >> "$DIR/$label.out" 2>>"$DIR/log.txt"
    else
        BENCH_TRACING=0 BENCH_WEBGPU=$gpu BENCH_URL=$BASE PW_CHANNEL=chrome \
            perl -e 'alarm shift; exec @ARGV' "$cap" \
            node bench-chain.mjs "$iters" 1 \
            >> "$DIR/$label.out" 2>>"$DIR/log.txt"
    fi
    rc=$?
    if [ $rc -ne 0 ]; then
        echo "RUN $label rc=$rc (timeout=wedge?)" | tee -a "$DIR/log.txt"
        # SIGALRM kills node but leaves its playwright chrome tree (and any
        # in-flight GPU work) alive — reap by profile pattern (only bench
        # chromes use playwright temp profiles), then let gpu_gate settle.
        pkill -f 'user-data-dir=.*playwright' 2>/dev/null
        sleep 5
    fi
    return $rc
}

WEDGES=0
# @2^20 pairs (278 iters): off x3, on x3 — ~30 s/run, cap 240 s.
for i in 1 2 3; do one_run "off20-$i" 278 0 240 0; done
for i in 1 2 3; do one_run "on20-$i" 278 1 240 0 || WEDGES=$((WEDGES+1)); done
# @2^22 pairs (1112 iters): off x3, on x3 (+1 retry per wedge, max 2) — cap 300 s.
for i in 1 2 3; do one_run "off22-$i" 1112 0 300 0; done
for i in 1 2 3; do
    if ! one_run "on22-$i" 1112 1 300 0; then
        WEDGES=$((WEDGES+1))
        [ $WEDGES -le 2 ] && one_run "on22-${i}r" 1112 1 300 0 || true
    fi
done
# Traced attribution pairs (st5 window): 1 run each, off/on, both scales.
one_run "tr-off20" 278 0 300 1
one_run "tr-on20" 278 1 300 1 || WEDGES=$((WEDGES+1))
one_run "tr-off22" 1112 0 420 1
one_run "tr-on22" 1112 1 420 1 || WEDGES=$((WEDGES+1))

echo "WEDGES=$WEDGES" | tee -a "$DIR/log.txt"
grep -h proveSeconds "$DIR"/*.out 2>/dev/null | tail -30
echo "done $(date +%H:%M:%S)" | tee -a "$DIR/log.txt"
