#!/bin/bash
# W5-U2b timed set (run while holding /tmp/jolt-wasm-bench.lock.d).
# Server on $PORT must serve THIS worktree's pkg (hash-verified below).
# Usage: PORT=8143 ./u2b-timed-set.sh
set -u
cd "$(dirname "$0")"
PORT="${PORT:-8143}"
BASE="http://localhost:$PORT/bench.html"
gpu() { ioreg -r -c IOAccelerator 2>/dev/null | grep -o '"Device Utilization %"=[0-9]*' | head -1; }
run() { # iters label BENCH_WEBGPU-json-or-empty runs
    local iters=$1 label=$2 cfg=$3 runs=$4
    echo "-- $label (iters $iters, cfg '$cfg', x$runs)"; gpu; uptime | sed 's/.*load/load/'
    if [ -z "$cfg" ]; then
        BENCH_BASE="$BASE" BENCH_TRACING=0 PW_CHANNEL=chrome \
            perl -e 'alarm 700; exec @ARGV' -- node bench-chain.mjs "$iters" "$runs" 2>/dev/null | tail -2
    else
        BENCH_BASE="$BASE" BENCH_TRACING=0 BENCH_WEBGPU="$cfg" PW_CHANNEL=chrome \
            perl -e 'alarm 700; exec @ARGV' -- node bench-chain.mjs "$iters" "$runs" 2>/dev/null | tail -2
    fi
}

echo "== identity =="
served=$(curl -s "http://localhost:$PORT/pkg/jolt_wasm_prover_bg.wasm" | shasum -a 256 | awk '{print $1}')
mine=$(shasum -a 256 pkg/jolt_wasm_prover_bg.wasm | awk '{print $1}')
[ "$served" = "$mine" ] || { echo "WASM IDENTITY MISMATCH — ABORT"; exit 1; }
echo "identity OK (${mine:0:16})"; uptime; gpu

echo "== @2^22 paired (off then on), x3 each =="
run 1112 "2^22 OFF" "" 3
run 1112 "2^22 ON" '{}' 3

echo "== @2^20 sign check (off then on), x3 each =="
run 278 "2^20 OFF" "" 3
run 278 "2^20 ON" '{}' 3

echo "== traced pair @2^22 =="
gpu
BENCH_BASE="$BASE" BENCH_WEBGPU='{}' PW_CHANNEL=chrome \
    perl -e 'alarm 700; exec @ARGV' -- node trace-chain.mjs 1112 1 traces-u2b/on 2>/dev/null | tail -1
BENCH_BASE="$BASE" PW_CHANNEL=chrome \
    perl -e 'alarm 700; exec @ARGV' -- node trace-chain.mjs 1112 1 traces-u2b/off 2>/dev/null | tail -1

echo "== soak 8 @2^22 =="
gpu
PW_CHANNEL=chrome perl -e 'alarm 2400; exec @ARGV' -- \
    node soak-webgpu.mjs 1112 8 4 "$BASE" traces-u2b/u2b-soak.jsonl 2>&1 | tail -4

echo "== done =="; gpu; uptime
