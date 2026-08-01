#!/bin/bash
set -u
cd "$(dirname "$0")"
BASE="http://localhost:8181/bench.html"
echo "== soak 10 @2^22 (batch 5) =="; ioreg -r -c IOAccelerator 2>/dev/null | grep -o '"Device Utilization %"=[0-9]*' | head -1
PW_CHANNEL=chrome perl -e 'alarm 2400; exec @ARGV' -- node soak-webgpu.mjs 1112 10 5 "$BASE" traces-w5m/soak-2e22.jsonl 2>&1 | tail -4
echo "== soak 5 @2^20 (batch 5) =="
PW_CHANNEL=chrome perl -e 'alarm 900; exec @ARGV' -- node soak-webgpu.mjs 278 5 5 "$BASE" traces-w5m/soak-2e20.jsonl 2>&1 | tail -4
echo done
