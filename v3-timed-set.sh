#!/bin/bash
# W6-V3 timed set, one lock window: traced ON @2^22 (stage attribution +
# engagement spans), then the untraced same-window control pair (ON x3,
# OFF x3). All stdout tee'd to evidence files (W5-M ops note).
set -uo pipefail
EV=.webgpu-lane-evidence/v3
mkdir -p "$EV"
PORT=8163
ITERS_22=1112

echo "=== traced ON @2^22 ===" >&2
PW_CHANNEL=chrome node t2a-trace.mjs $ITERS_22 on 0 0 "$EV/trace-on-2e22.json" $PORT 1 \
  2>>"$EV/timed.stderr" | tee "$EV/traced-on-2e22.jsonl"

echo "=== untraced ON x3 @2^22 ===" >&2
BENCH_TRACING=0 BENCH_WEBGPU=1 BENCH_URL="http://localhost:$PORT" PW_CHANNEL=chrome \
  node bench-chain.mjs $ITERS_22 3 2>>"$EV/timed.stderr" | tee "$EV/on-2e22.jsonl"

echo "=== untraced OFF x3 @2^22 ===" >&2
BENCH_TRACING=0 BENCH_URL="http://localhost:$PORT" PW_CHANNEL=chrome \
  node bench-chain.mjs $ITERS_22 3 2>>"$EV/timed.stderr" | tee "$EV/off-2e22.jsonl"
