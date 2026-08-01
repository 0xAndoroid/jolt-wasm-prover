#!/bin/bash
set -u
cd "$(dirname "$0")"
BASE="http://localhost:8181/bench.html"
echo "== traced OFF @2^22 =="
BENCH_BASE="$BASE" PW_CHANNEL=chrome perl -e 'alarm 900; exec @ARGV' -- node trace-chain.mjs 1112 1 traces-w5m/off 2>/dev/null | tail -2
echo "== traced ON @2^22 =="
BENCH_BASE="$BASE" BENCH_WEBGPU='{}' PW_CHANNEL=chrome perl -e 'alarm 900; exec @ARGV' -- node trace-chain.mjs 1112 1 traces-w5m/on 2>/dev/null | tail -2
