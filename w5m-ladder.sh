#!/bin/bash
set -u
cd "$(dirname "$0")"
BASE="http://localhost:8181/bench.html"
echo "-- ladder OFF x3"; ioreg -r -c IOAccelerator 2>/dev/null | grep -o '"Device Utilization %"=[0-9]*' | head -1
BENCH_BASE="$BASE" BENCH_TRACING=0 PW_CHANNEL=chrome perl -e 'alarm 1500; exec @ARGV' -- node bench-chain.mjs 17,69,556 3 > traces-w5m/ladder-off.json 2>/dev/null
echo "-- ladder ON x3"
BENCH_BASE="$BASE" BENCH_TRACING=0 BENCH_WEBGPU='{}' PW_CHANNEL=chrome perl -e 'alarm 1500; exec @ARGV' -- node bench-chain.mjs 17,69,556 3 > traces-w5m/ladder-on.json 2>/dev/null
echo done
