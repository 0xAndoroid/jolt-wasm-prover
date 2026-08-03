#!/bin/bash
# W3-T4 final timed set, window B (@2^20) — run under t4-run.sh.
# off/on medians + slot-sign arms (T2b reproduction on the T1-tip tree).
set -uo pipefail
cd "$(dirname "$0")"
export PW_CHANNEL="${PW_CHANNEL:-chrome}"
ITERS=278
WATCHDOG_S="${WATCHDOG_S:-180}"
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

run_watched off20-x3 env BENCH_TRACING=0 node bench-chain.mjs "$ITERS" 3
run_watched on20-x3  env BENCH_TRACING=0 BENCH_WEBGPU='{}' node bench-chain.mjs "$ITERS" 3
run_watched on20-commit-gated env BENCH_TRACING=0 BENCH_WEBGPU='{"minTermsCommit":1073741824}' node bench-chain.mjs "$ITERS" 1
run_watched on20-xyzz env BENCH_TRACING=0 BENCH_WEBGPU='{"bucketXyzz":true}' node bench-chain.mjs "$ITERS" 1

echo "=== window B done; wedges=$WEDGES" >&2
