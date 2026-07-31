# Jolt WASM Prover

In-browser zero-knowledge proving and verification using [Jolt](https://github.com/a16z/jolt). Compiles the Jolt zkVM prover and verifier to WebAssembly with multithreading support via `SharedArrayBuffer` and `wasm-bindgen-rayon`.

## Programs

Four guest programs are included:

| Program | Description | Guest crate |
|---------|-------------|-------------|
| **SHA-256** | Hash arbitrary input | `guests/sha2` |
| **ECDSA** | Secp256k1 signature verification | `guests/secp256k1` |
| **Keccak Chain** | Iterated Keccak-256 hashing | `guests/sha3-chain` |
| **SHA-256 Chain** | Iterated SHA-256 (tunable trace length, for scale benchmarks) | `guests/sha2-chain` |

## Prerequisites

- Rust nightly (managed via `rust-toolchain.toml`)
- `wasm-pack`: `curl https://drager.github.io/wasm-pack/installer/init.sh -sSf | bash`
- Node.js (for the dev server)

## Quick Start

### 1. Generate preprocessing data

Preprocessing generates the Dory SRS, compiles guest ELFs, and serializes prover/verifier preprocessing into `frontend/public/`.

```bash
cargo run --release --features native --bin generate-preprocessing
```

This produces per-program files in `frontend/public/`:
- `{name}_prover.bin` — prover preprocessing (Dory SRS + shared preprocessing)
- `{name}_verifier.bin` — verifier preprocessing (Dory verifier setup + shared preprocessing)
- `{name}.elf` — compiled guest RISC-V ELF

### 2. Set up patched WASM dependencies

```bash
./setup-wasm-deps.sh
```

Browser proving currently needs two one-line wasm32 fixes that are not
upstream yet — see [WASM runtime patches](#wasm-runtime-patches-pending-upstream).
Native builds (step 1, roundtrip test) work without this step.

### 3. Build WASM

```bash
CARGO_UNSTABLE_BUILD_STD="panic_abort,std" wasm-pack build --release --target web
```

Outputs the WASM package to `pkg/`.

### 4. Run (dev)

```bash
cd frontend && npm install && npm run dev
# Open http://localhost:8080
```

Or build and serve with the production server:

```bash
cd frontend && npm run build
node server.mjs
# Open http://localhost:8080
```

Both set `Cross-Origin-Opener-Policy` and `Cross-Origin-Embedder-Policy` headers required for `SharedArrayBuffer`.

## Architecture

```
src/lib.rs          WASM entry point — WasmProver, WasmVerifier, init_inlines
src/wasm_tracing.rs Chrome Trace Format profiling (Perfetto-compatible)
preprocessing/      Native binaries for generating preprocessing data
guests/             RISC-V guest programs (compiled to ELF by jolt-sdk)
frontend/           Vite + React + TypeScript + Tailwind frontend
frontend/public/    Preprocessing artifacts + worker.js
server.mjs          Production server with COOP/COEP headers
```

### WASM API

```javascript
// Initialize
await init();
await initThreadPool(navigator.hardwareConcurrency);
init_tracing();  // optional: enables Perfetto-compatible tracing
init_inlines();  // registers optimized inline implementations

// Prove
const prover = new WasmProver(proverPreprocessingBytes, elfBytes);
const result = prover.prove_sha2(inputBytes);
// result.proof, result.program_io, result.proof_size, result.num_cycles

// Verify
const verifier = new WasmVerifier(verifierPreprocessingBytes);
const valid = verifier.verify(result.proof, result.program_io);
```

### Build flags

The `.cargo/config.toml` configures the WASM build with:
- **Atomics + shared memory** — enables `wasm-bindgen-rayon` multithreading
- **4 GB max memory** — required for prover memory usage
- **`build-std`** — rebuilds `std` with atomics support (requires nightly)

## Roundtrip Testing

Validates that preprocessing serialization is deterministic and cross-platform:

```bash
cargo run --release --features native --bin test-roundtrip
```

## Dependency pins

Jolt crates come from [a16z/jolt](https://github.com/a16z/jolt) at rev
`be900fc55de099c4cb50ee79310d624ed9488af8` (branch `perf/kernels-optimized`,
PR #1714) — the optimized kernel backend rebased over main's BlindFold ZK
support (#1690), so one pin carries both. The extra `perf/manycore-scaling`
commits from the previous pin are not included.
Arkworks comes from [a16z/arkworks-algebra](https://github.com/a16z/arkworks-algebra)
branch `dev/twist-shout`; the committed `Cargo.lock` pins it to
`76bb3a4518928f1ff7f15875f940d614bb9845e6`. The `[patch.crates-io]` block in
`Cargo.toml` redirects the registry `ark-*` crates (pulled in by `dory-pcs`)
onto the same fork so the whole graph shares one set of arkworks types.
Build with the committed lockfile; `cargo update` can move the arkworks
branch resolution.

## WASM runtime patches (pending upstream)

The repo builds everywhere from the pinned upstream revs, and native binaries
are fully functional. Proving on `wasm32` additionally needs two one-line
fixes that are not upstream yet, shipped in `patches/`:

1. `0001-jolt-coefflut-u64.patch` — `CoeffLut::saturated()` in
   `jolt-kernels` computes `len * len` in `usize`; at the 65536-entry table
   this is 2^32, which wraps to 0 on 32-bit targets and later panics every
   rayon worker with an index-out-of-bounds. Native 64-bit is unaffected.
2. `0002-arkworks-wasm-nested-pool.patch` — `msm_bigint_wnaf` in `ark-ec`
   builds a nested `rayon::ThreadPoolBuilder` per chunk; `build()` panics on
   `wasm32`, where wasm-bindgen-rayon provides exactly one fixed global pool.
   The fix runs the inner parallel MSM on the global pool.

What breaks without them, empirically (HeadlessChrome 150, M4, at the pinned
revs): the default sha2 demo (2^13 trace) **hangs** — proving never completes
and no error surfaces; sha2-chain at 2^16 **panics** in every rayon worker
with `index out of bounds: the len is 0` at
`jolt-kernels/src/optimized/registers_read_write.rs:439` (the CoeffLut fix's
exact target). With both patches applied, sha2 proves and sha2-chain at 2^16
proves and verifies in-browser. Native 64-bit builds are unaffected either
way (the roundtrip test passes without the patches).

`./setup-wasm-deps.sh` clones both upstreams at the pinned revs into
`.wasm-deps/`, applies the patches, and rewrites the marked override block in
`Cargo.toml` onto the patched checkouts. `./setup-wasm-deps.sh --revert`
restores the pinned block.
