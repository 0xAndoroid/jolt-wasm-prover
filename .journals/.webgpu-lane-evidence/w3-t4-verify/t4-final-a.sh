#!/bin/bash
# W3-T4 final timed set, window A (@2^22) — run under t4-run.sh.
# Headline medians (off x3, on x3) + A/B singles (xyzz, commit-gated) +
# traced pair for st0 confirmation.
set -uo pipefail
cd "$(dirname "$0")"
export PW_CHANNEL="${PW_CHANNEL:-chrome}"
ITERS=1112
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

run_watched gate-2^16 node gate-webgpu.mjs 17 "${BENCH_BASE:-http://127.0.0.1:8097}"

run_watched off-x3 env BENCH_TRACING=0 node bench-chain.mjs "$ITERS" 3
run_watched on-x3  env BENCH_TRACING=0 BENCH_WEBGPU='{}' node bench-chain.mjs "$ITERS" 3

run_watched on-xyzz-single   env BENCH_TRACING=0 BENCH_WEBGPU='{"bucketXyzz":true}' node bench-chain.mjs "$ITERS" 1
run_watched on-commit-gated  env BENCH_TRACING=0 BENCH_WEBGPU='{"minTermsCommit":1073741824}' node bench-chain.mjs "$ITERS" 1

run_watched traced-on  env BENCH_WEBGPU='{}' node trace-chain.mjs "$ITERS" 1 traces/t4f-on
run_watched traced-off node trace-chain.mjs "$ITERS" 1 traces/t4f-off

echo "=== window A done; wedges=$WEDGES" >&2
