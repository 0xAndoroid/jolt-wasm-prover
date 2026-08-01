#!/bin/bash
# W5-U2a wall re-run with full logging (the timed-set's tail -2 ate the numbers).
set -u
cd "$(dirname "$0")"
PORT="${PORT:-8153}"
BASE="http://localhost:$PORT/bench.html"
gpu() { ioreg -r -c IOAccelerator 2>/dev/null | grep -o '"Device Utilization %"=[0-9]*' | head -1; }
run() {
    local iters=$1 label=$2 cfg=$3 runs=$4
    echo "-- $label (iters $iters, x$runs)"; gpu; uptime | sed 's/.*load/load/'
    if [ -z "$cfg" ]; then
        BENCH_BASE="$BASE" BENCH_TRACING=0 PW_CHANNEL=chrome \
            perl -e 'alarm 900; exec @ARGV' -- node bench-chain.mjs "$iters" "$runs" 2>&1 | grep -E "^\{|prove [0-9]"
    else
        BENCH_BASE="$BASE" BENCH_TRACING=0 BENCH_WEBGPU="$cfg" PW_CHANNEL=chrome \
            perl -e 'alarm 900; exec @ARGV' -- node bench-chain.mjs "$iters" "$runs" 2>&1 | grep -E "^\{|prove [0-9]"
    fi
    gpu
}
echo "== walls @2^22 =="
run 1112 "2^22 OFF" "" 3
run 1112 "2^22 ON" '{}' 3
echo "== walls @2^20 =="
run 278 "2^20 OFF" "" 3
run 278 "2^20 ON" '{}' 3
echo done
