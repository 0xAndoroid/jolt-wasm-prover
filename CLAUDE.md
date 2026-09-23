# CLAUDE.md

## Project Overview

WASM prover/verifier demo for [Jolt](https://github.com/a16z/jolt) zkVM. Compiles Jolt's modular prover (`jolt-prover`, backend-optimized path) and verifier to WebAssembly on the **Akita (lattice) protocol** — `jolt-prover/akita`, packed one-hot trace commitment over the Solinas field p = 2^128 − 2^32 + 22537 (`jolt-field/solinas`), no elliptic curves — and runs it in the browser with multithreading via `wasm-bindgen-rayon`. Transparent (non-zk): `akita` and `zk` are mutually exclusive in jolt-prover.

## Commands

```bash
# One-time: patched WASM deps (browser proving needs wasm32 fixes, see README)
./setup-wasm-deps.sh          # clone pinned upstreams, apply patches/, rewrite Cargo.toml block
./setup-wasm-deps.sh --revert # restore the pinned upstream-rev block

# Build WASM package (outputs to pkg/). Stable toolchain + RUSTC_BOOTSTRAP for build-std.
RUSTC_BOOTSTRAP=1 CARGO_UNSTABLE_BUILD_STD="panic_abort,std" wasm-pack build --release --target web

# Build native preprocessing generator (needs the `jolt` CLI from the pinned jolt rev on PATH)
cargo build --release --features native

# Generate artifacts into frontend/public/ (guest ELFs, program preprocessing, Akita schedule bundle)
cargo run --release --features native --bin generate-preprocessing

# Roundtrip test: full native prove+verify from the serialized artifacts
cargo run --release --features native --bin test-roundtrip

# Clippy / format (root package only — the guest crates are riscv-only and
# their jolt-sdk host glue does not compile against an akita jolt-prover)
cargo clippy --all --all-targets --message-format=short -q   # root workspace; guests/ is its own workspace
cargo fmt -q

# Dev server (serves on http://localhost:8080)
node server.mjs
```

## Benchmarking

Requires: `npm install`, dev server running. Scripts default to the Playwright-bundled Chromium; `PW_BROWSER=webkit` runs the bundled WebKit (Safari engine); `PW_CHANNEL=chrome` selects system Chrome.

```bash
node server.mjs &
node bench.mjs            # sha2 demo via the React UI (default 3 runs)
node bench-chain.mjs 278  # sha2-chain at a given iteration count, via worker.js directly
```

`bench.mjs` outputs JSON to stdout; per-run timings go to stderr. sha2-chain iteration counts map to padded trace lengths: 17 → 2^16, 69 → 2^18, 278 → 2^20, 1112 → 2^22. `bench-chain.mjs` reports the trace / Akita setup / prove split per run.

**server.mjs caches compressed responses in memory with no mtime check — restart it after every wasm rebuild or you test stale bytes.**

## Architecture

- `src/lib.rs` — `#[wasm_bindgen]` exports: `WasmProver`, `WasmVerifier`, tracing (`init_inlines` export kept as a no-op — inline registration is inventory-based link-time ctors and worker.js no longer calls it)
- `src/engine.rs` — the prove/verify pipeline shared by wasm and native: decode the Akita schedule bundle + `JoltProgramPreprocessing` (bincode2), trace via `TracerBackend::trace_compact`, derive `ProverConfig`, build the shape-exact Akita setup with `jolt_prover::akita::preprocessing::preprocess_full` (the "setup" phase, per proof), prove via `jolt_prover::prove` over `JoltAkitaBackend::optimized()`, return proof + program IO + the verifier preprocessing for that shape
- `src/wasm_tracing.rs` — Chrome Trace Format layer for `tracing`, outputs Perfetto-compatible JSON (per-thread tids)
- `preprocessing/generate.rs` — native binary: compiles guests through `jolt-host` (the `jolt` CLI), writes `{name}.elf`, `{name}_program.bin`, and `akita_schedules.bin` to `frontend/public/`. Guest memory/trace parameters live in its `GUESTS` table (heap sizes must match the guests' `#[jolt::provable(heap_size = ..)]`)
- `preprocessing/test_roundtrip.rs` — native binary: full prove+verify from the shipped bytes, same code path as the browser (catches everything except 32-bit-isms)
- `guests/{sha2,secp256k1,sha3-chain,sha2-chain}/` — RISC-V guest programs using `jolt-sdk`, in their own cargo workspace (`guests/Cargo.toml`): the `#[jolt::provable]` host glue needs `jolt-sdk/host` (Dory-only), which cannot share a graph with `jolt-prover/akita`; built only through `jolt build` by `generate-preprocessing`, never natively
- `frontend/` — Vite + React + TypeScript + Tailwind + shadcn frontend
- `frontend/public/` — artifacts (`akita_schedules.bin`, `{name}_program.bin`, `{name}.elf`), `worker.js`
- `server.mjs` — Node.js production server serving `frontend/dist/` with COOP/COEP headers

## Feature Flags

- **default** (no features) — WASM library build (`cdylib`)
- **`native`** — enables the preprocessing binaries (`jolt-host` guest compilation)

## Key Dependencies

- Jolt modular crates (`jolt-prover` + `jolt-verifier` with `akita`, `jolt-akita`, `jolt-witness`, `jolt-program`, `tracer`, `common`, `jolt-host`, `jolt-inlines-*`, guests' `jolt-sdk`) from `https://github.com/a16z/jolt` rev `d39bd518a65ea89401de63c3343e98fbad5f1b80` (main, after PR #1818 removed `jolt-prover-legacy`)
- Akita crates from `https://github.com/LayerZero-Labs/akita.git` rev `252abb895046cc1d5b9955a26a2ad2318148ac26` (jolt's pin). They depend on `jolt-field` at their own jolt rev; `[patch."https://github.com/a16z/jolt"]` redirects that onto the pinned rev via the `0xAndoroid/jolt` mirror (cargo refuses same-URL git patches)
- Arkworks from `a16z/arkworks-algebra` branch `dev/twist-shout`, pinned by the committed `Cargo.lock` to `76bb3a4518928f1ff7f15875f940d614bb9845e6`; the `[patch.crates-io]` block redirects registry `ark-*` (dory-pcs's deps — still in the graph through jolt-prover, though the Dory path is compiled out) onto the fork — one ark world
- jolt-sdk's host side is Dory-only and does not compile against an akita `jolt-prover`; the native binaries use `jolt-host` directly
- Browser proving needs wasm32 fixes shipped as `patches/` + `setup-wasm-deps.sh` (see README "WASM runtime patches"); native builds work from the pinned revs directly

## WASM Build Requirements

- Stable Rust `1.95` pinned in `rust-toolchain.toml` (matches jolt's workspace). Current nightlies alias `Infallible` to `!` and break `allocative` (a hard dependency via `common/std`) with conflicting impls, so `-Z build-std` runs on stable via `RUSTC_BOOTSTRAP=1`
- After switching toolchain versions, run `cargo clean --target wasm32-unknown-unknown` — stale `std` artifacts cause `condvar wait not supported` panics
- `.cargo/config.toml` sets `+atomics,+bulk-memory,+mutable-globals`, 4 GB max memory, a 32 MiB main stack (Akita kernels recurse deeply), and `--export=__heap_base` (required by wasm-bindgen's threads transform); `worker.js` sizes rayon worker stacks with `thread_stack_size` at init
- `wasm-pack` for building the WASM package
- `wasm-opt = false` — the wasm-opt pass was neutral-to-harmful here
- Profile `lto = true` (don't put `-C lto=fat` in rustflags — it breaks rlib targets under build-std)

## Serialization

- `akita_schedules.bin` — `jolt_akita::AkitaScheduleArtifacts` (the three `.aks` catalogs minus the `num_vars >= 32` rows wasm32 cannot deserialize — see README "Protocol"), bincode2/serde, shared by all programs
- `{name}_program.bin` — `jolt_program::preprocess::JoltProgramPreprocessing`, bincode2/serde
- Verifier preprocessing (`JoltVerifierPreprocessing<AkitaScheme, NoVectorCommitment>`) is emitted by the prover per proof (bincode2) — the Akita verifier setup is exact in the proof shape, so there is no static per-program verifier artifact; the prover setup is not serializable at all
- Proofs and `JoltDevice` (program IO) — bincode2

## Threads

- React UI (`use-prover.ts`): `min(hardwareConcurrency, 8)` threads
- `worker.js` consumers (bench-chain.mjs): caller-controlled, typically 10
- Brave underreports `hardwareConcurrency` (fingerprinting protection) → fewer threads → slower proving

## Deployment

### Cloudflare Pages

Output directory: `frontend/dist/` (with `pkg/` copied in).

```bash
RUSTC_BOOTSTRAP=1 CARGO_UNSTABLE_BUILD_STD="panic_abort,std" wasm-pack build --release --target web
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
