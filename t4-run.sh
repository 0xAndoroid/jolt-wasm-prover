#!/bin/bash
# W3-T4 bench protocol wrapper. Usage:
#   ./t4-run.sh [--no-lock] [--] <command...>
# Protocol (journal §lock etiquette, §server-hijack, incident #3):
#   1. bench lock taken IN-SCRIPT (mkdir loop) unless --no-lock
#   2. ioreg GPU-util gate (<=3%) before the command
#   3. fresh server on PORT (default 8097) from THIS worktree, PID-tracked
#   4. served-wasm-hash == built-pkg-hash asserted
#   5. command runs with BENCH_BASE pointing at the lane server
#   6. server killed by PID, lock released
set -euo pipefail
cd "$(dirname "$0")"

PORT="${PORT:-8097}"
LOCK=/tmp/jolt-wasm-bench.lock.d
TAKE_LOCK=1
if [[ "${1:-}" == "--no-lock" ]]; then TAKE_LOCK=0; shift; fi
[[ "${1:-}" == "--" ]] && shift

gpu_util() {
    ioreg -r -d 1 -w0 -c IOAccelerator 2>/dev/null \
        | /usr/bin/grep -o '"Device Utilization %"=[0-9]*' \
        | /usr/bin/grep -o '[0-9]*$' | head -1
}

cleanup() {
    if [[ -n "${SERVER_PID:-}" ]]; then kill "$SERVER_PID" 2>/dev/null || true; fi
    if [[ "$TAKE_LOCK" == 1 && "${LOCK_HELD:-0}" == 1 ]]; then rmdir "$LOCK" 2>/dev/null || true; fi
}
trap cleanup EXIT

if [[ "$TAKE_LOCK" == 1 ]]; then
    echo "[t4-run] acquiring bench lock..." >&2
    for _ in $(seq 1 720); do
        if mkdir "$LOCK" 2>/dev/null; then LOCK_HELD=1; break; fi
        sleep 10
    done
    if [[ "${LOCK_HELD:-0}" != 1 ]]; then echo "[t4-run] lock timeout (2h)" >&2; exit 75; fi
    echo "[t4-run] lock held" >&2
fi

# GPU-util gate: wait for idle device (zombie/co-load guard)
for _ in $(seq 1 60); do
    UTIL="$(gpu_util || echo 0)"
    [[ -z "$UTIL" ]] && UTIL=0
    if (( UTIL <= 3 )); then break; fi
    sleep 5
done
echo "[t4-run] gpu-util=$UTIL loadavg=$(sysctl -n vm.loadavg)" >&2

# Fresh lane server (compressCache staleness + hijack protocol)
PORT="$PORT" node server.mjs >/tmp/t4-server-$PORT.log 2>&1 &
SERVER_PID=$!
sleep 1
if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "[t4-run] server failed to start (port busy?)" >&2
    cat /tmp/t4-server-$PORT.log >&2
    exit 1
fi

BUILT_HASH=$(shasum -a 256 pkg/jolt_wasm_prover_bg.wasm | cut -d' ' -f1)
SERVED_HASH=$(curl -s "http://127.0.0.1:$PORT/pkg/jolt_wasm_prover_bg.wasm" | shasum -a 256 | cut -d' ' -f1)
if [[ "$BUILT_HASH" != "$SERVED_HASH" ]]; then
    echo "[t4-run] HASH MISMATCH built=$BUILT_HASH served=$SERVED_HASH" >&2
    exit 1
fi
echo "[t4-run] wasm hash verified: ${BUILT_HASH:0:16} (server pid $SERVER_PID port $PORT)" >&2

BENCH_BASE="http://127.0.0.1:$PORT" "$@"
STATUS=$?
echo "[t4-run] done (exit $STATUS) gpu-util-after=$(gpu_util)" >&2
exit $STATUS
