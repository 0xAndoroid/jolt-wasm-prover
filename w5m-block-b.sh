#!/bin/bash
set -u
cd "$(dirname "$0")"
BASE="http://localhost:8181/bench.html"
gpu() { ioreg -r -c IOAccelerator 2>/dev/null | grep -o '"Device Utilization %"=[0-9]*' | head -1; }
run() {
    local iters=$1 label=$2 cfg=$3 runs=$4
    echo "-- $label"; gpu; uptime | sed 's/.*load/load/'
    if [ -z "$cfg" ]; then
        BENCH_BASE="$BASE" BENCH_TRACING=0 PW_CHANNEL=chrome perl -e 'alarm 1500; exec @ARGV' -- node bench-chain.mjs "$iters" "$runs" 2>/dev/null
    else
        BENCH_BASE="$BASE" BENCH_TRACING=0 BENCH_WEBGPU="$cfg" PW_CHANNEL=chrome perl -e 'alarm 1500; exec @ARGV' -- node bench-chain.mjs "$iters" "$runs" 2>/dev/null
    fi
}
echo "== block B: 2^20 anchors (med5) =="
run 278 "2^20 OFF x5" "" 5
run 278 "2^20 ON x5" '{}' 5
echo "== block B: ladder 2^16/2^18/2^21 (med3) =="
run 17,69,556 "ladder OFF x3" "" 3
run 17,69,556 "ladder ON x3" '{}' 3
echo done
