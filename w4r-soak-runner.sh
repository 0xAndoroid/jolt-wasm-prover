#!/bin/bash
# W4-R soak runner: bench-lock + GPU-quiet gate + served-wasm identity
# assert around soak-webgpu.mjs. Usage:
#   ./w4r-soak-runner.sh <iters> <runs> <label> [batch]
# Env: PORT (default 8096), EXPECT_HASH (optional pkg hash pin).
set -euo pipefail
ITERS=${1:?iters}
RUNS=${2:?runs}
LABEL=${3:?label}
BATCH=${4:-5}
PORT=${PORT:-8096}
LOCK=/tmp/jolt-wasm-bench.lock.d

cd "$(dirname "$0")"

# 1. Lock (mkdir-atomic), waiting politely.
for i in $(seq 1 720); do
    if mkdir "$LOCK" 2>/dev/null; then
        echo "w4-r soak $LABEL pid $$ $(date '+%H:%M:%S')" > "$LOCK/owner"
        trap '/bin/rm -rf "$LOCK"' EXIT
        break
    fi
    if [ "$i" = 720 ]; then echo "LOCK WAIT TIMEOUT (1h)"; exit 3; fi
    sleep 5
done
echo "lock taken: $(cat "$LOCK/owner")"

# 2. GPU quiet gate (<10% util; zombie-dispatch discipline).
for i in $(seq 1 60); do
    UTIL=$(ioreg -r -d 1 -w 0 -c IOAccelerator 2>/dev/null | grep -o '"Device Utilization %"=[0-9]*' | head -1 | grep -o '[0-9]*$' || echo 100)
    if [ "${UTIL:-100}" -lt 10 ]; then echo "GPU util ${UTIL}% — quiet"; break; fi
    if [ "$i" = 60 ]; then echo "GPU NEVER QUIET (${UTIL}%)"; exit 4; fi
    sleep 5
done

# 3. Server identity: served bytes == built pkg (== EXPECT_HASH if pinned).
SERVED=$(curl -s "http://localhost:$PORT/pkg/jolt_wasm_prover_bg.wasm" | shasum -a 256 | cut -d' ' -f1)
BUILT=$(shasum -a 256 pkg/jolt_wasm_prover_bg.wasm | cut -d' ' -f1)
if [ "$SERVED" != "$BUILT" ]; then echo "IDENTITY MISMATCH served=$SERVED built=$BUILT"; exit 5; fi
if [ -n "${EXPECT_HASH:-}" ] && [ "$SERVED" != "$EXPECT_HASH" ]; then echo "PIN MISMATCH served=$SERVED expected=$EXPECT_HASH"; exit 5; fi
echo "identity ok: $SERVED"
uptime

# 4. Soak.
RC=0
PW_CHANNEL=chrome node soak-webgpu.mjs "$ITERS" "$RUNS" "$BATCH" "http://localhost:$PORT" "soak-$LABEL.jsonl" || RC=$?
echo "soak rc=$RC"
exit $RC
