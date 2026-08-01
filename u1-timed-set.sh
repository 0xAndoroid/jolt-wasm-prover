#!/bin/bash
# W5-U1 timed set (run while holding /tmp/jolt-wasm-bench.lock.d).
# Server on $PORT must serve THIS worktree's pkg (hash-verified below).
# Usage: PORT=8131 ./u1-timed-set.sh
set -u
cd "$(dirname "$0")"
PORT="${PORT:-8131}"
gpu() { ioreg -r -c IOAccelerator 2>/dev/null | grep -o '"Device Utilization %"=[0-9]*' | head -1; }
run() { # iters label BENCH_WEBGPU-json runs
    local iters=$1 label=$2 cfg=$3 runs=$4
    echo "-- $label (iters $iters, cfg $cfg, x$runs)"; gpu
    BENCH_BASE="http://localhost:$PORT" BENCH_WEBGPU="$cfg" PW_CHANNEL=chrome \
        perl -e 'alarm 400; exec @ARGV' -- node bench-chain.mjs "$iters" "$runs" 2>/dev/null | tail -1
}

echo "== identity =="
served=$(curl -s "http://localhost:$PORT/pkg/jolt_wasm_prover_bg.wasm" | shasum -a 256 | awk '{print $1}')
mine=$(shasum -a 256 pkg/jolt_wasm_prover_bg.wasm | awk '{print $1}')
[ "$served" = "$mine" ] || { echo "WASM IDENTITY MISMATCH — ABORT"; exit 1; }
echo "identity OK (${mine:0:16})"; uptime; gpu

echo "== fraction sweep @2^22 (coalesce default), med3 =="
for f in 0.1 0.2 0.3 0.4 0.5; do
    run 1112 "f=$f" "{\"millerCpuFraction\":$f}" 3
done

echo "== coalesce A/B @2^22 at best-f (EDIT after sweep; default f shown), med3 =="
run 1112 "coalesce=default" '{}' 3
run 1112 "coalesce=0" '{"millerCoalesce":0}' 3

echo "== @2^20 sign check (best-f + default), med3 =="
run 278 "2^20 default" '{}' 3
run 278 "2^20 f=0.3" '{"millerCpuFraction":0.3}' 3

echo "== traced pair @2^22 (st0 window, best config) =="
gpu; BENCH_BASE="http://localhost:$PORT" BENCH_WEBGPU='{}' PW_CHANNEL=chrome \
    perl -e 'alarm 400; exec @ARGV' -- node trace-chain.mjs 1112 1 traces-u1/timed-on 2>/dev/null | tail -1

echo "== done =="; gpu; uptime
