#!/bin/bash
set -u
cd "$(dirname "$0")"
BASE="http://localhost:8183/bench.html"
echo "-- CONTROL (U2a-tip pair) 2^22 ON x3"; ioreg -r -c IOAccelerator 2>/dev/null | grep -o '"Device Utilization %"=[0-9]*' | head -1; uptime | sed 's/.*load/load/'
BENCH_BASE="$BASE" BENCH_TRACING=0 BENCH_WEBGPU='{}' PW_CHANNEL=chrome perl -e 'alarm 700; exec @ARGV' -- node bench-chain.mjs 1112 3 2>/dev/null | tail -14
