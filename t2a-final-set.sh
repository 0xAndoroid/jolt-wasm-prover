#!/bin/bash
# W3-T2a final timed window (run while holding /tmp/jolt-wasm-bench.lock.d).
# Server on 8129 must serve THIS worktree's pkg (hash-verified below).
set -u
cd "$(dirname "$0")"
PORT=8129
gpu() { ioreg -r -c IOAccelerator 2>/dev/null | grep -o '"Device Utilization %"=[0-9]*' | head -1; }

echo "== identity =="
served=$(curl -s http://localhost:$PORT/pkg/jolt_wasm_prover_bg.wasm | shasum -a 256 | awk '{print $1}')
mine=$(shasum -a 256 pkg/jolt_wasm_prover_bg.wasm | awk '{print $1}')
[ "$served" = "$mine" ] || { echo "WASM IDENTITY MISMATCH — ABORT"; exit 1; }
echo "wasm identity OK (${mine:0:16})"; uptime; gpu

echo "== browser byte gate @2^16 (build3) =="
PW_CHANNEL=chrome node gate-webgpu.mjs 17 http://localhost:$PORT 2>/dev/null | tail -1

echo "== @2^22: 3x on =="
for i in 1 2 3; do gpu; PW_CHANNEL=chrome timeout 200 node t2a-trace.mjs 1112 on 0 0 "" $PORT 1 2>/dev/null; done
echo "== @2^22: 2x off =="
for i in 1 2; do PW_CHANNEL=chrome timeout 200 node t2a-trace.mjs 1112 off 0 0 "" $PORT 1 2>/dev/null; done
echo "== @2^20: 3x on =="
for i in 1 2 3; do gpu; PW_CHANNEL=chrome timeout 120 node t2a-trace.mjs 278 on 0 0 "" $PORT 1 2>/dev/null; done
echo "== @2^20: 2x off =="
for i in 1 2; do PW_CHANNEL=chrome timeout 120 node t2a-trace.mjs 278 off 0 0 "" $PORT 1 2>/dev/null; done

echo "== traced pair @2^22 (st6b attribution) =="
gpu; PW_CHANNEL=chrome timeout 220 node t2a-trace.mjs 1112 off 0 0 traces-t2a/final-off 8129 1 2>/dev/null
gpu; PW_CHANNEL=chrome timeout 220 node t2a-trace.mjs 1112 on 0 0 traces-t2a/final-on 8129 1 2>/dev/null

echo "== repeat-prove probe @2^22 arm-on (2 proves, one session) =="
gpu; PW_CHANNEL=chrome timeout 400 node t2a-trace.mjs 1112 on 0 0 "" $PORT 2 2>/dev/null
echo "probe exit: $? (124 = second prove hung)"

echo "== done =="; gpu; uptime
