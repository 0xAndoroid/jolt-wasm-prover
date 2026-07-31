#!/bin/bash
# W3-T4 attribution block — run under t4-run.sh (lock + gate + server + hash).
# Traced triple @2^22 (barrier / pipelined / CPU) + untraced anchors.
# Wedge watchdog: known engine hang (~2/13 GPU-arm 2^22 runs, W4-R's ticket) —
# kill the runner by PID after WATCHDOG_S, count it, retry once.
set -uo pipefail
cd "$(dirname "$0")"
export PW_CHANNEL="${PW_CHANNEL:-chrome}"
ITERS="${ITERS:-1112}"
WATCHDOG_S="${WATCHDOG_S:-360}"
mkdir -p traces
WEDGES=0

run_watched() {
    local label="$1"; shift
    for attempt in 1 2 3; do
        echo "=== [$label] attempt $attempt: $*" >&2
        "$@" & local pid=$!
        local waited=0
        while kill -0 "$pid" 2>/dev/null; do
            sleep 5; waited=$((waited + 5))
            if (( waited >= WATCHDOG_S )); then
                echo "=== [$label] WEDGED after ${WATCHDOG_S}s — killing PID $pid" >&2
                kill "$pid" 2>/dev/null; sleep 2; kill -9 "$pid" 2>/dev/null
                WEDGES=$((WEDGES + 1))
                break
            fi
        done
        if (( waited < WATCHDOG_S )); then
            wait "$pid"; local status=$?
            if (( status == 0 )); then return 0; fi
            echo "=== [$label] exited $status" >&2
        fi
    done
    echo "=== [$label] FAILED after 3 attempts" >&2
    return 1
}

run_watched browser-gate-2^16 node gate-webgpu.mjs 17 "${BENCH_BASE:-http://127.0.0.1:8097}"

run_watched gpu-nopipe-traced env BENCH_WEBGPU='{"commitPipeline":false}' node trace-chain.mjs "$ITERS" 1 traces/t4-gpu-nopipe
run_watched gpu-pipe-traced   env BENCH_WEBGPU='{}' node trace-chain.mjs "$ITERS" 1 traces/t4-gpu-pipe
run_watched cpu-traced        node trace-chain.mjs "$ITERS" 1 traces/t4-cpu

run_watched gpu-nopipe-anchor env BENCH_TRACING=0 BENCH_WEBGPU='{"commitPipeline":false}' node bench-chain.mjs "$ITERS" 1
run_watched gpu-pipe-anchor   env BENCH_TRACING=0 BENCH_WEBGPU='{}' node bench-chain.mjs "$ITERS" 1
run_watched gpu-xyzz-anchor   env BENCH_TRACING=0 BENCH_WEBGPU='{"bucketXyzz":true}' node bench-chain.mjs "$ITERS" 1
run_watched cpu-anchor        env BENCH_TRACING=0 node bench-chain.mjs "$ITERS" 1

echo "=== attribution block done; wedges=$WEDGES" >&2
