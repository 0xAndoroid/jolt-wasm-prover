#!/bin/bash
# W5-M canonical block A: @2^22 anchors, OFF x5 + ON x5 (med5).
set -u
cd "$(dirname "$0")"
PORT="${PORT:-8181}"
BASE="http://localhost:$PORT/bench.html"
gpu() { ioreg -r -c IOAccelerator 2>/dev/null | grep -o '"Device Utilization %"=[0-9]*' | head -1; }
run() {
    local iters=$1 label=$2 cfg=$3 runs=$4
    echo "-- $label (iters $iters, x$runs)"; gpu; uptime | sed 's/.*load/load/'
    if [ -z "$cfg" ]; then
        BENCH_BASE="$BASE" BENCH_TRACING=0 PW_CHANNEL=chrome \
            perl -e 'alarm 1200; exec @ARGV' -- node bench-chain.mjs "$iters" "$runs" 2>/dev/null
    else
        BENCH_BASE="$BASE" BENCH_TRACING=0 BENCH_WEBGPU="$cfg" PW_CHANNEL=chrome \
            perl -e 'alarm 1200; exec @ARGV' -- node bench-chain.mjs "$iters" "$runs" 2>/dev/null
    fi
    gpu
}
echo "== W5-M block A: 2^22 anchors =="
run 1112 "2^22 OFF" "" 5
run 1112 "2^22 ON" '{}' 5
echo done
