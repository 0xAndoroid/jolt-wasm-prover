#!/bin/bash
# @2^22-only redo under BOTH gates (GPU util < 10% AND 1-min loadavg < 4):
# the first set's walls were polluted by a co-running fat-LTO build.
set -u
DIR="${1:?results dir}"; PORT="${PORT:-8095}"; BASE="http://localhost:${PORT}"
mkdir -p "$DIR"
LOCK=/tmp/jolt-wasm-bench.lock.d
until mkdir "$LOCK" 2>/dev/null; do sleep 15; done
echo "t2b-22-redo pid $$" > "$LOCK/owner"
trap '/bin/rm -rf "$LOCK"' EXIT
echo "lock acquired $(date +%H:%M:%S)" | tee "$DIR/log.txt"
A=$(curl -s "$BASE/pkg/jolt_wasm_prover_bg.wasm" | shasum -a 256 | cut -d' ' -f1)
B=$(shasum -a 256 pkg/jolt_wasm_prover_bg.wasm | cut -d' ' -f1)
[ "$A" = "$B" ] || { echo "WASM IDENTITY MISMATCH" | tee -a "$DIR/log.txt"; exit 2; }

quiet_gate() {
    for _ in $(seq 1 240); do
        u=$(ioreg -r -d 1 -w 0 -c IOAccelerator 2>/dev/null \
            | grep -o '"Device Utilization %"=[0-9]*' | head -1 | grep -o '[0-9]*$')
        l=$(sysctl -n vm.loadavg | awk '{print int($2)}')
        [ "${u:-100}" -lt 10 ] && [ "${l:-99}" -lt 4 ] && return 0
        sleep 10
    done
    echo "quiet gate timed out (gpu=$u load=$l)" | tee -a "$DIR/log.txt"
    return 1
}
one_run() {
    local label=$1 gpu=$2 cap=$3 traced=$4 rc
    quiet_gate || return 1
    echo "=== $label $(date +%H:%M:%S) loadavg $(sysctl -n vm.loadavg)" >> "$DIR/log.txt"
    if [ "$traced" = 1 ]; then
        BENCH_WEBGPU=$gpu BENCH_URL=$BASE PW_CHANNEL=chrome \
            perl -e 'alarm shift; exec @ARGV' "$cap" \
            node trace-chain.mjs 1112 1 "$DIR/$label" >> "$DIR/$label.out" 2>>"$DIR/log.txt"
    else
        BENCH_TRACING=0 BENCH_WEBGPU=$gpu BENCH_URL=$BASE PW_CHANNEL=chrome \
            perl -e 'alarm shift; exec @ARGV' "$cap" \
            node bench-chain.mjs 1112 1 >> "$DIR/$label.out" 2>>"$DIR/log.txt"
    fi
    rc=$?
    if [ $rc -ne 0 ]; then
        echo "RUN $label rc=$rc" | tee -a "$DIR/log.txt"
        pkill -f 'user-data-dir=.*playwright' 2>/dev/null
        sleep 5
    fi
    return $rc
}
WEDGES=0
for i in 1 2 3; do one_run "off22-$i" 0 300 0; done
for i in 1 2 3; do
    if ! one_run "on22-$i" 1 300 0; then
        WEDGES=$((WEDGES+1)); [ $WEDGES -le 2 ] && one_run "on22-${i}r" 1 300 0 || true
    fi
done
one_run "tr-off22" 0 420 1
one_run "tr-on22" 1 420 1 || WEDGES=$((WEDGES+1))
echo "WEDGES=$WEDGES" | tee -a "$DIR/log.txt"
echo "done $(date +%H:%M:%S)" | tee -a "$DIR/log.txt"
