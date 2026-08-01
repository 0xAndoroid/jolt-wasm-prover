#!/bin/bash
set -u
cd "$(dirname "$0")"
PORT=8153
BASE="http://localhost:$PORT/bench.html"
gpu() { ioreg -r -c IOAccelerator 2>/dev/null | grep -o '"Device Utilization %"=[0-9]*' | head -1; }
for i in 1 2 3 4 5 6; do u=$(gpu); echo "settle: $u"; case "$u" in *=[0-9]|*=1[0-5]) break;; esac; sleep 15; done
./u2a-walls.sh
echo "== traced ON @2^22 (final build) =="
gpu
BENCH_BASE="$BASE" BENCH_WEBGPU='{}' PW_CHANNEL=chrome \
    perl -e 'alarm 700; exec @ARGV' -- node trace-chain.mjs 1112 1 traces-u2a/on-final 2>/dev/null | tail -1
echo "== re-soak 5 @2^22 (final build) =="
gpu
PW_CHANNEL=chrome perl -e 'alarm 1500; exec @ARGV' -- \
    node soak-webgpu.mjs 1112 5 5 "$BASE" traces-u2a/u2a-soak-final.jsonl 2>&1 | tail -3
echo "== done ==": gpu; uptime
