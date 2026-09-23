#!/usr/bin/env bash
# Assemble frontend/dist/ for Cloudflare Pages from a pkg/ built with the full
# browser feature set (GPU mailbox + trace-commit + digit-range devices).
#
#   ./scripts/build-pages.sh          use the existing pkg/ (feature set is checked)
#   ./scripts/build-pages.sh --wasm   run ./setup-wasm-deps.sh + the wasm build first
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
FEATURES="webgpu,trace-commit-device,digit-range-device"
WASM_CMD="RUSTC_BOOTSTRAP=1 CARGO_UNSTABLE_BUILD_STD=\"panic_abort,std\" wasm-pack build --release --target web -- --features $FEATURES"
MAX_FILE_BYTES=$((25 * 1024 * 1024)) # Cloudflare Pages per-file cap

die() { echo "Error: $*" >&2; exit 1; }

if [ "${1:-}" = "--wasm" ]; then
  "$ROOT/setup-wasm-deps.sh"
  (cd "$ROOT" && RUSTC_BOOTSTRAP=1 CARGO_UNSTABLE_BUILD_STD="panic_abort,std" \
    wasm-pack build --release --target web -- --features "$FEATURES")
fi

WASM="$ROOT/pkg/jolt_wasm_prover_bg.wasm"
JS="$ROOT/pkg/jolt_wasm_prover.js"
[ -f "$WASM" ] || die "pkg/jolt_wasm_prover_bg.wasm not found. Run: $WASM_CMD"
# The shipped build must carry every GPU feature. `gpu_mailbox_ptr` is exported
# unconditionally (returns 0 without `webgpu`), so probe the install messages of
# src/gpu/{trace_commit,digit_range}.rs instead: each compiles only with `webgpu`
# plus its device feature.
grep -qaF '[gpu] trace commit device' "$WASM" || die "pkg/ lacks webgpu,trace-commit-device. Run: $WASM_CMD"
grep -qaF '[gpu] digit range device' "$WASM" || die "pkg/ lacks webgpu,digit-range-device. Run: $WASM_CMD"

cd "$ROOT/frontend"
npm ci
npm run build

DIST="$ROOT/frontend/dist"
mkdir -p "$DIST/pkg"
cp "$JS" "$WASM" "$DIST/pkg/"
cp -r "$ROOT/pkg/snippets" "$DIST/pkg/"

for f in _headers _redirects worker.js gpu-proxy.js wgsl/fp128.wgsl wgsl/commit/commit_accumulate.wgsl \
         wgsl/digit_range/round0.wgsl akita_schedules.bin sha2_program.bin keccak_program.bin; do
  [ -f "$DIST/$f" ] || die "frontend/dist/$f missing after the Vite build"
done

too_big=$(find "$DIST" -type f -size +"$MAX_FILE_BYTES"c)
[ -z "$too_big" ] || die "over the 25 MiB Pages per-file cap:"$'\n'"$too_big"

echo "wasm: $(du -h "$WASM" | cut -f1)  dist total: $(du -sh "$DIST" | cut -f1)"
echo "Build complete: frontend/dist/ ready for deployment"
echo "Deploy with: npx wrangler pages deploy frontend/dist --project-name=jolt-wasm-prover"
