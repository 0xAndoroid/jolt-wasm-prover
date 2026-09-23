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
# ... with the experimental WebGPU harness compiled in (see README "WebGPU (experimental)")
RUSTC_BOOTSTRAP=1 CARGO_UNSTABLE_BUILD_STD="panic_abort,std" wasm-pack build --release --target web -- --features webgpu
# ... plus the GPU one-hot trace commit (needs ./setup-wasm-deps.sh first, patch 0006)
RUSTC_BOOTSTRAP=1 CARGO_UNSTABLE_BUILD_STD="panic_abort,std" wasm-pack build --release --target web -- --features webgpu,trace-commit-device
# ... plus the GPU digit-range sumcheck rounds (patch 0007; W2 increment 0.447 s of 2.78 s at 2^18 idle, kill rule 0.425 s → unparked)
RUSTC_BOOTSTRAP=1 CARGO_UNSTABLE_BUILD_STD="panic_abort,std" wasm-pack build --release --target web -- --features webgpu,trace-commit-device,digit-range-device

# Build native preprocessing generator (needs the `jolt` CLI from the pinned jolt rev on PATH)
cargo build --release --features native

# Generate artifacts into frontend/public/ (guest ELFs, program preprocessing, Akita schedule bundle)
cargo run --release --features native --bin generate-preprocessing

# Roundtrip test: full native prove+verify from the serialized artifacts
cargo run --release --features native --bin test-roundtrip

# Trace-commit device seam (patches/0006, needs ./setup-wasm-deps.sh first):
# `trace-commit-device` compiles the seam glue; JOLT_TRACE_COMMIT_DEVICE=cpu-ref
# routes stage 0 through the CPU reference device (docs/trace-commit-device.md)
JOLT_TRACE_COMMIT_DEVICE=cpu-ref cargo run --release --features native,trace-commit-device --bin test-roundtrip
RUST_LOG=jolt_akita=info cargo run --release --features native,trace-commit-device --bin test-roundtrip  # prints the stage-0 shape + extraction time

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

WebGPU harness (Python Playwright, `uv run --with playwright python -m playwright install webkit chromium` once; the node Playwright's WebKit has no WebGPU):

```bash
uv run --with playwright python bench/bench_webgpu.py --iters 17,69,278,556 --runs 4 --gpu both   # gpu on vs off: proofs byte-identical, verify=true; commit stage breakdown + off/on/ratio table
uv run --with playwright python bench/bench_webgpu.py --iters 69 --runs 5 --gpu all                # off / w1 (commit only) / on (W1+W2) / w2 (digit range only); W2 increment = w1 − on
uv run --with playwright python bench/bench_webgpu.py --iters 69 --runs 1 --gpu on --parity 6      # CPU shadow of the first 6 GPU digit-range rounds per instance; aborts on mismatch
uv run --with playwright python bench/test_fp128_wgsl.py                                          # fp128.wgsl vs Python ints mod p, 100k vectors
uv run --with playwright python bench/proto/commit_proto.py --shape full                          # shipped commit kernels vs Python reference (also --shape small --variant v9 --chunk 64, --shape stress --chunk 2048)
```

`bench_webgpu.py` needs `node server.mjs` restarted after every wasm/frontend rebuild (it caches). GPU-commit chunk rule: CHUNK = smallest of 64..2048 dividing P with PART scratch `(P/CHUNK)·64·blocks·512·32 B` ≤ 256 MiB; GPU memory at 2^21 ≈ 576 MiB (A 64 + A2 128 + codes 128 + PART 256).

`bench.mjs` outputs JSON to stdout; per-run timings go to stderr. sha2-chain iteration counts map to padded trace lengths: 17 → 2^16, 69 → 2^18, 278 → 2^20, 556 → 2^21 — the wasm32 ceiling (2^22 needs a one-hot polynomial with 2^32 coefficients; see README "Protocol"). `bench-chain.mjs` reports the trace / Akita setup / prove split per run.

**server.mjs caches compressed responses in memory with no mtime check — restart it after every wasm rebuild or you test stale bytes.** It serves `frontend/dist/`, so edits to `frontend/public/` (worker.js, gpu-proxy.js, wgsl/) need `cd frontend && npm run build` before the restart.

## Architecture

- `src/lib.rs` — `#[wasm_bindgen]` exports: `WasmProver`, `WasmVerifier`, tracing (`init_inlines` export kept as a no-op — inline registration is inventory-based link-time ctors and worker.js no longer calls it)
- `src/engine.rs` — the prove/verify pipeline shared by wasm and native: decode the Akita schedule bundle + `JoltProgramPreprocessing` (bincode2), trace via `TracerBackend::trace_compact`, derive `ProverConfig`, build the shape-exact Akita setup with `jolt_prover::akita::preprocessing::preprocess_full` (the "setup" phase, per proof), prove via `jolt_prover::prove` over `JoltAkitaBackend::optimized()`, return proof + program IO + the verifier preprocessing for that shape
- `src/gpu/` — WebGPU harness behind the `webgpu` cargo feature: `mailbox.rs` (`#[repr(C)]` static in shared wasm memory; Rust fills op/args/regions, `Atomics.notify`s the doorbell and `Atomics.wait`s on `status`), `selftest.rs` (2^20 `a·b+c` vs `AkitaField`, 200 NOP round trips). `engine::prove` runs `gpu::preflight()` before the phase clock and reports `gpu_status` etc.; `disabled`/`unavailable` never touch the mailbox
- `frontend/public/gpu-proxy.js` — dedicated Worker owning the `GPUDevice`; mirrors the mailbox word layout by hand (keep in sync with `mailbox.rs`); ops NOP/CREATE_BUFFER/UPLOAD/DESTROY/RUN/RUN_SEQ/DOWNLOAD; shaders from `frontend/public/wgsl/` (`fp128.wgsl` library; `commit/{common,prep,commit_accumulate,reduce}.wgsl` share `common.wgsl`), RUN arg 22 is the `CHUNK` pipeline override. `worker.js` spawns it when `init` carries `gpu: true`
- `src/gpu/digit_range.rs` — `WebGpuDigitRange`, the `akita_prover::DigitRangeDevice` for the browser (features `webgpu,trace-commit-device,digit-range-device`): installed with the trace-commit device; takes basis-8 direct-leaf instances with ≥ 2^16 digits, runs g = min(ring_bits, log2 n − 12) rounds (one RUN_SEQ op per round: params + eq tables inline, 80-byte message inline readback; kernels `frontend/public/wgsl/digit_range/`), downloads the folded table for the CPU tail. Breakdown → `gpu_digit_range` JSON + `[gpu] digit range …` console line. W2 verdict: 0.447 s increment over W1 at 2^18 on an idle host (kill rule 0.425 s) → unparked, marginal; see `.journals/webgpu-akita-w2.md`
- `src/gpu/trace_commit.rs` — `WebGpuTraceCommit`, the `jolt_akita::TraceCommitDevice` for the browser (features `webgpu,trace-commit-device`): installed once after the preflight self-test passes, declines to the CPU kernels when the GPU is off or the shape is not K16/D512/n_a1/digits1/colcap64; persistent A/A2/codes/PART buffers per (P, blocks), RES read back per call; stage breakdown → `ProveResult.gpu_commit` JSON + `[gpu] trace commit …` console line
- `src/trace_commit_reference.rs` — `CpuReferenceDevice`, the CPU oracle for the stage-0 trace-commit device ABI (`docs/trace-commit-device.md`); feature `trace-commit-device`
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
- **`trace-commit-device`** — `engine::install_trace_commit_device` + `CpuReferenceDevice`; with `webgpu` also `gpu::trace_commit::WebGpuTraceCommit`. Needs the patched `jolt-akita` from `./setup-wasm-deps.sh` (patch 0006), so it is off by default to keep pinned-rev native builds working
- **`digit-range-device`** — `engine::install_digit_range_device` + the `akita-prover` dependency for the seam types; needs patch 0007 from `./setup-wasm-deps.sh`, off by default for the same reason. `webgpu,trace-commit-device,digit-range-device` together give the W1+W2 browser build
- **`webgpu`** — compiles the GPU mailbox + self-test into the wasm build; without it `gpu_mailbox_ptr()` returns 0 and worker.js reports `unavailable`

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

Output directory: `frontend/dist/` (with `pkg/` copied in). The shipped wasm is the full browser feature set; `scripts/build-pages.sh` refuses a `pkg/` built without it. Checklist: DEPLOY.md.

```bash
./setup-wasm-deps.sh
RUSTC_BOOTSTRAP=1 CARGO_UNSTABLE_BUILD_STD="panic_abort,std" wasm-pack build --release --target web -- --features webgpu,trace-commit-device,digit-range-device
./scripts/build-pages.sh          # npm ci + vite build + pkg copy + 25 MiB per-file check (`--wasm` runs the two lines above first)
npx wrangler pages deploy frontend/dist --project-name=jolt-wasm-prover   # add --branch preview for a preview URL
```

Required files in `frontend/public/` (copied to `dist/` by Vite):
- `_headers` — COOP/COEP, CSP, security headers (includes `cloudflareinsights.com` for CF analytics beacon); same set as `server.mjs`, keep them in sync. `no-cache` on `/pkg/*`, `/worker.js`, `/gpu-proxy.js`, `/wgsl/*`
- `_redirects` — `/pkg/` → `/pkg/jolt_wasm_prover.js` redirect (needed by `wasm-bindgen-rayon`'s `workerHelpers.js` which does `import('../../..')`)

### Local (`server.mjs`)

`server.mjs` serves `frontend/dist/` and `pkg/` with security headers, path traversal protection, and compression. Binds to `127.0.0.1:8080`. For external access, use Cloudflare Tunnel — never expose directly.

## Comment Policy

- No comments restating what code does
- No commented-out code
- Keep WHY comments, SAFETY comments, WARNING comments
