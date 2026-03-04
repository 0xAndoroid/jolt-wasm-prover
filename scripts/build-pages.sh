#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

if [ ! -f "$ROOT/pkg/jolt_wasm_prover_bg.wasm" ]; then
  echo "Error: pkg/jolt_wasm_prover_bg.wasm not found." >&2
  echo "Run: CARGO_UNSTABLE_BUILD_STD=\"panic_abort,std\" wasm-pack build --release --target web" >&2
  exit 1
fi

cd "$ROOT/frontend"
npm ci
npm run build

mkdir -p "$ROOT/frontend/dist/pkg"
cp "$ROOT/pkg/jolt_wasm_prover.js" "$ROOT/frontend/dist/pkg/"
cp "$ROOT/pkg/jolt_wasm_prover_bg.wasm" "$ROOT/frontend/dist/pkg/"
cp -r "$ROOT/pkg/snippets" "$ROOT/frontend/dist/pkg/"

echo "Build complete: frontend/dist/ ready for deployment"
echo "Deploy with: wrangler pages deploy frontend/dist --project-name=jolt-wasm-prover"
