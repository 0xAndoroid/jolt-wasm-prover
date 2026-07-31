# CLAUDE.md

## Project Overview

WASM prover/verifier demo for [Jolt](https://github.com/a16z/jolt) zkVM. Compiles Jolt's modular prover (`jolt-prover`, backend-optimized path) and verifier to WebAssembly, runs in browser with multithreading via `wasm-bindgen-rayon`.

## Commands

```bash
# One-time: patched WASM deps (browser proving needs two wasm32 fixes, see README)
./setup-wasm-deps.sh          # clone pinned upstreams, apply patches/, rewrite Cargo.toml block
./setup-wasm-deps.sh --revert # restore the pinned upstream-rev block

# Build WASM package (outputs to pkg/)
CARGO_UNSTABLE_BUILD_STD="panic_abort,std" wasm-pack build --release --target web

# Build native preprocessing generator
cargo build --release --features native

# Generate preprocessing artifacts into frontend/public/
cargo run --release --features native --bin generate-preprocessing

# Roundtrip test: full native prove+verify from the serialized artifacts
cargo run --release --features native --bin test-roundtrip

# Clippy / format
cargo clippy --all --message-format=short -q
cargo fmt -q

# Dev server (serves on http://localhost:8080)
node server.mjs
```

## Benchmarking

Requires: `npm install`, system Chrome (bench scripts use Playwright `channel: 'chrome'`), dev server running.

```bash
node server.mjs &
node bench.mjs            # sha2 demo via the React UI (default 3 runs)
node bench-chain.mjs 278  # sha2-chain at a given iteration count, via worker.js directly
```

`bench.mjs` outputs JSON to stdout; per-run timings go to stderr. sha2-chain iteration counts map to padded trace lengths: 17 → 2^16, 69 → 2^18, 278 → 2^20, 1112 → 2^22.

**server.mjs caches compressed responses in memory with no mtime check — restart it after every wasm rebuild or you test stale bytes.**

## Architecture

- `src/lib.rs` — `#[wasm_bindgen]` exports: `WasmProver`, `WasmVerifier`, tracing (`init_inlines` export kept as a no-op — inline registration is inventory-based link-time ctors and worker.js no longer calls it)
- `src/engine.rs` — the prove/verify pipeline shared by wasm and native: deserialize SRS (`ark-serialize` uncompressed) + verifier preprocessing (bincode2), assemble `JoltProverPreprocessing`, trace via `TracerBackend`, prove via `jolt_prover::prove` over `JoltBackend::optimized()`
- `src/wasm_tracing.rs` — Chrome Trace Format layer for `tracing`, outputs Perfetto-compatible JSON (per-thread tids)
- `preprocessing/generate.rs` — native binary: compiles guests, generates Dory SRS, serializes prover/verifier preprocessing to `frontend/public/`
- `preprocessing/test_roundtrip.rs` — native binary: full modular prove+verify from the shipped bytes (catches everything except 32-bit-isms)
- `guests/{sha2,secp256k1,sha3-chain,sha2-chain}/` — RISC-V guest programs using `jolt-sdk`; sha2-chain takes an iteration count for tunable trace length
- `frontend/` — Vite + React + TypeScript + Tailwind + shadcn frontend
- `frontend/public/` — preprocessing `.bin` files, guest `.elf` files, `worker.js`
- `server.mjs` — Node.js production server serving `frontend/dist/` with COOP/COEP headers

## Feature Flags

- **default** (no features) — WASM library build (`cdylib`)
- **`native`** — enables `jolt-sdk` host machinery and guest compilation; required for the preprocessing binaries

## Key Dependencies

- Jolt modular crates (`jolt-prover`, `jolt-verifier`, `jolt-dory`, `tracer`, `common`, `jolt-sdk`, `jolt-inlines-*`, …) from `https://github.com/a16z/jolt` rev `70a294ad58629af59ab89f646d6d0079b57174cb` (branch `perf/manycore-scaling`)
- Arkworks from `a16z/arkworks-algebra` branch `dev/twist-shout`, pinned by the committed `Cargo.lock` to `76bb3a4518928f1ff7f15875f940d614bb9845e6`; the `[patch.crates-io]` block redirects registry `ark-*` (dory-pcs's deps) onto the fork — one ark world, mirroring the jolt workspace's `[replace]`
- `dory-pcs` 0.4.0 from crates.io
- Browser proving needs two unpushed wasm32 fixes shipped as `patches/` + `setup-wasm-deps.sh` (see README "WASM runtime patches"); native builds work from the pinned revs directly

## WASM Build Requirements

- Nightly Rust (for `build-std` with atomics; tree needs ≥1.95 nightly)
- After switching nightly versions, run `cargo clean --target wasm32-unknown-unknown` — stale `std` artifacts cause `condvar wait not supported` panics
- `.cargo/config.toml` sets `+atomics,+bulk-memory,+mutable-globals`, 4 GB max memory, and `--export=__heap_base` (required by wasm-bindgen's threads transform)
- `wasm-pack` for building the WASM package
- `wasm-opt = false` — the wasm-opt pass was neutral-to-harmful here
- Profile `lto = true` (don't put `-C lto=fat` in rustflags — it breaks rlib targets under build-std)

## Serialization

- `{name}_prover.bin` — dory-pcs `ArkworksProverSetup` (SRS), **uncompressed** arkworks serialization for fast deserialization in WASM
- `{name}_verifier.bin` — `jolt_verifier::JoltVerifierPreprocessing`, bincode2/serde
- Proofs and `JoltDevice` (program IO) — bincode2

## Threads

- React UI (`use-prover.ts`): `min(hardwareConcurrency, 8)` threads
- `worker.js` consumers (bench-chain.mjs): caller-controlled, typically 10
- Brave underreports `hardwareConcurrency` (fingerprinting protection) → fewer threads → slower proving

## Deployment

### Cloudflare Pages

Output directory: `frontend/dist/` (with `pkg/` copied in).

```bash
CARGO_UNSTABLE_BUILD_STD="panic_abort,std" wasm-pack build --release --target web
cd frontend && npm run build && cd ..
cp -r pkg/ frontend/dist/pkg/
npx wrangler pages deploy frontend/dist/ --project-name <project-name>
```

Required files in `frontend/public/` (copied to `dist/` by Vite):
- `_headers` — COOP/COEP, CSP, security headers (includes `cloudflareinsights.com` for CF analytics beacon)
- `_redirects` — `/pkg/` → `/pkg/jolt_wasm_prover.js` redirect (needed by `wasm-bindgen-rayon`'s `workerHelpers.js` which does `import('../../..')`)

### Local (`server.mjs`)

`server.mjs` serves `frontend/dist/` and `pkg/` with security headers, path traversal protection, and compression. Binds to `127.0.0.1:8080`. For external access, use Cloudflare Tunnel — never expose directly.

## Comment Policy

- No comments restating what code does
- No commented-out code
- Keep WHY comments, SAFETY comments, WARNING comments
