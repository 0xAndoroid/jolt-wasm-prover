# Jolt WASM Prover

In-browser proving and verification using [Jolt](https://github.com/a16z/jolt). Compiles the Jolt zkVM's modular prover and verifier to WebAssembly on the **Akita lattice protocol** (`jolt-prover/akita`) with multithreading support via `SharedArrayBuffer` and `wasm-bindgen-rayon`.

## Programs

Four guest programs are included:

| Program | Description | Guest crate |
|---------|-------------|-------------|
| **SHA-256** | Hash arbitrary input | `guests/sha2` |
| **ECDSA** | Secp256k1 signature verification | `guests/secp256k1` |
| **Keccak Chain** | Iterated Keccak-256 hashing | `guests/sha3-chain` |
| **SHA-256 Chain** | Iterated SHA-256 (tunable trace length, for scale benchmarks) | `guests/sha2-chain` |

## Prerequisites

- Rust `1.95` stable (pinned via `rust-toolchain.toml`; the listed components and targets install on first use)
- The `jolt` CLI from the pinned jolt rev, on `PATH` (`cargo install --path . --bin jolt` inside an a16z/jolt checkout at that rev) — `generate-preprocessing` compiles the guests through it
- `wasm-pack`: `curl https://drager.github.io/wasm-pack/installer/init.sh -sSf | bash`
- Node.js (for the dev server)

## Quick Start

### 1. Generate artifacts

Compiles the guest ELFs, serializes each program's preprocessing, and bundles the Akita schedule catalogs into `frontend/public/`.

```bash
cargo run --release --features native --bin generate-preprocessing
```

This produces:
- `akita_schedules.bin` — the three base Akita schedule catalogs (`jolt-akita/schedules/*.aks`), shared by every program, trimmed to the rows a 32-bit target can deserialize (see [Protocol](#protocol))
- `{name}_program.bin` — `JoltProgramPreprocessing` (bytecode tables, memory layout, max trace length)
- `{name}.elf` — compiled guest RISC-V ELF

There is no static prover artifact. The Akita commitment setup is exact in the proof shape (padded trace length, RAM size, bytecode size) and its prover half is not serializable, so the browser derives it per proof from the schedule catalogs and the program preprocessing (the "setup" phase, reported separately). The verifier preprocessing for that shape comes out of the same step; the prover returns it next to the proof and the demo's verifier consumes it. A real deployment pins one verifier preprocessing per program and shape.

### 2. Set up patched WASM dependencies

```bash
./setup-wasm-deps.sh
```

Browser proving currently needs a few small wasm32 fixes that are not
upstream yet — see [WASM runtime patches](#wasm-runtime-patches-pending-upstream).
Native builds (step 1, roundtrip test) work without this step.

### 3. Build WASM

```bash
RUSTC_BOOTSTRAP=1 CARGO_UNSTABLE_BUILD_STD="panic_abort,std" wasm-pack build --release --target web
```

Outputs the WASM package to `pkg/`. `build-std` (atomics-enabled `std`) is a
nightly cargo feature; `RUSTC_BOOTSTRAP=1` unlocks it on the pinned stable
toolchain. Current nightlies alias `core::convert::Infallible` to `!`, which
breaks `allocative` 0.3 (a hard dependency through jolt's `common/std`) with
conflicting trait impls — hence stable.

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
src/engine.rs       Shared prove/verify pipeline (trace → config → Akita setup → prove; verify)
src/wasm_tracing.rs Chrome Trace Format profiling (Perfetto-compatible)
preprocessing/      Native binaries: artifact generation, roundtrip test
guests/             RISC-V guest programs, own cargo workspace (compiled to ELF via the jolt CLI)
frontend/           Vite + React + TypeScript + Tailwind frontend
frontend/public/    Artifacts + worker.js
server.mjs          Production server with COOP/COEP headers
```

### WASM API

```javascript
// Initialize (worker stacks: Akita kernels recurse deeply)
await init({ module_or_path: wasmUrl, thread_stack_size: 32 * 1024 * 1024 });
await initThreadPool(navigator.hardwareConcurrency);
init_tracing();  // optional: enables Perfetto-compatible tracing

// Prove
const prover = new WasmProver(scheduleArtifactsBytes, programPreprocessingBytes, elfBytes);
const result = prover.prove_sha2(inputBytes);
// result.proof, result.program_io, result.verifier_preprocessing,
// result.proof_size, result.num_cycles, result.padded_cycles,
// result.trace_ms, result.setup_ms, result.prove_ms

// Verify
const verifier = new WasmVerifier(result.verifier_preprocessing);
const valid = verifier.verify(result.proof, result.program_io);
```

### Build flags

The `.cargo/config.toml` configures the WASM build with:
- **Atomics + shared memory** — enables `wasm-bindgen-rayon` multithreading
- **4 GB max memory** — required for prover memory usage
- **32 MiB main-thread stack** — the Akita backend kernels recurse deeply (natively they run on 64 MiB rayon worker stacks); `worker.js` sizes the rayon worker stacks the same way through wasm-bindgen's `thread_stack_size`
- **`build-std`** — rebuilds `std` with atomics support (nightly cargo feature, unlocked on stable with `RUSTC_BOOTSTRAP=1`)

## Roundtrip Testing

Runs the exact browser code path natively from the shipped artifacts (setup derivation, prove, verify):

```bash
cargo run --release --features native --bin test-roundtrip
```

## Benchmarks

```bash
node server.mjs &
node bench.mjs            # sha2 demo through the React UI
node bench-chain.mjs 278  # sha2-chain ladder through worker.js (per-phase split)
```

Both default to the Playwright-bundled Chromium; `PW_BROWSER=webkit` runs the bundled WebKit (Safari's engine), `PW_CHANNEL=chrome` selects system Chrome.

## Protocol

`jolt-prover/akita` selects the packed lattice pipeline: one native `OneHotTrace` commitment group over the Solinas field p = 2^128 − 2^32 + 22537 (`jolt-field/solinas`), the shared stage 1–7 sumchecks on `JoltAkitaBackend::optimized()`, and one native grouped opening (Akita PCS, [LayerZero-Labs/akita](https://github.com/LayerZero-Labs/akita)). Proofs are transparent: `akita` and `zk` (BlindFold) are mutually exclusive in jolt-prover, so this demo has no zero-knowledge mode.

**32-bit schedule subset.** Akita's schedule rows for committed groups of 2^32 or more coefficients (`num_vars >= 32`) carry `usize` fields above `u32::MAX`, which wasm32 cannot deserialize. `generate-preprocessing` drops those rows (dense 36/42, one-hot K=16 40/46, K=256 40/64 kept); nothing a browser can hold is lost and row lookup is exact-key, so the remaining shapes resolve unchanged. The catalog digest bound into the transcript differs from jolt's full catalogs, so browser proofs verify against the verifier preprocessing the prover emits (which carries the same trimmed catalogs), not against a verifier loaded with the packaged `.aks` files.

## Dependency pins

Jolt crates come from [a16z/jolt](https://github.com/a16z/jolt) at rev
`d39bd518a65ea89401de63c3343e98fbad5f1b80` (main, after PR #1818 removed
`jolt-prover-legacy`). Akita crates come from
[LayerZero-Labs/akita](https://github.com/LayerZero-Labs/akita) at rev
`252abb895046cc1d5b9955a26a2ad2318148ac26`, jolt's own pin.
Arkworks comes from [a16z/arkworks-algebra](https://github.com/a16z/arkworks-algebra)
branch `dev/twist-shout`; the committed `Cargo.lock` pins it to
`76bb3a4518928f1ff7f15875f940d614bb9845e6`. The `[patch.crates-io]` block in
`Cargo.toml` redirects the registry `ark-*` crates (pulled in by `dory-pcs`,
which stays in the graph through `jolt-prover` even though the Dory path is
compiled out) onto the same fork so the whole graph shares one set of
arkworks types.

The akita crates depend on `jolt-field` from a16z/jolt at their own rev.
The `[patch."https://github.com/a16z/jolt"]` block redirects that package
onto the pinned rev so the graph holds one field identity (the jolt
workspace does the same with a path patch). Cargo refuses a git patch onto
the same repository URL, so the redirect points at a mirror of a16z/jolt
(`0xAndoroid/jolt`) serving the identical commit.

jolt-sdk's host side is Dory-only and does not compile against a
jolt-prover built with `akita`; the native binaries drive `jolt-host`
directly and the guest crates are never compiled natively.

Build with the committed lockfile; `cargo update` can move the arkworks
branch resolution.

## WASM runtime patches (pending upstream)

The repo builds everywhere from the pinned upstream revs, and native binaries
are fully functional. Proving on `wasm32` additionally needs five small fixes
that are not upstream yet, shipped in `patches/`:

1. `0001-jolt-coefflut-u64.patch` — `CoeffLut::saturated()` in
   `jolt-kernels` computes `len * len` in `usize`; at the 65536-entry table
   this is 2^32, which wraps to 0 on 32-bit targets and later panics every
   rayon worker with an index-out-of-bounds. Native 64-bit is unaffected.
2. `0002-arkworks-wasm-nested-pool.patch` — `msm_bigint_wnaf` in `ark-ec`
   builds a nested `rayon::ThreadPoolBuilder` per chunk; `build()` panics on
   `wasm32`, where wasm-bindgen-rayon provides exactly one fixed global pool.
   Kept for the arkworks code still linked in; the Akita path does not
   call it.
3. `0003-jolt-akita-wasm-pool.patch` — `jolt-akita` runs every backend call
   on a dedicated rayon pool with 64 MiB worker stacks; `build()` panics on
   `wasm32`. The fix runs the closure on the global pool there and leaves
   worker stack sizing to the embedder (`thread_stack_size`).
4. `0004-akita-wasm-instant.patch` — `akita-prover` and `akita-pcs` read
   `std::time::Instant` for diagnostics timing; `Instant::now()` panics on
   `wasm32-unknown-unknown`. The fix routes those imports through a
   per-crate shim that is a zero-duration stand-in on wasm32.
5. `0005-akita-types-wasm32-shift.patch` — `akita-types` computes
   `1usize << 52` as the exact-f64 integer bound; that constant overflows
   `usize` on 32-bit targets (compile error E0080). The fix makes it `u64`.

`./setup-wasm-deps.sh` clones the three upstreams at the pinned revs into
`.wasm-deps/`, applies the patches, and rewrites the marked override block in
`Cargo.toml` onto the patched checkouts. `./setup-wasm-deps.sh --revert`
restores the pinned block.
