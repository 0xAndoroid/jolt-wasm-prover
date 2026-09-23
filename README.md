# Jolt WASM Prover

In-browser proving and verification using [Jolt](https://github.com/a16z/jolt). Compiles the Jolt zkVM's modular prover and verifier to WebAssembly on the **Akita lattice protocol** (`jolt-prover/akita`) with multithreading support via `SharedArrayBuffer` and `wasm-bindgen-rayon`.

## Programs

Four guest programs are included:

| Program | Description | Guest crate |
|---------|-------------|-------------|
| **SHA-256** | Hash arbitrary input | `guests/sha2` |
| **ECDSA** | secp256k1 ECDSA verification of one fixed signature (`jolt-inlines-secp256k1` inline) | `guests/secp256k1` |
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

There is no static prover artifact. The Akita commitment setup is exact in the proof shape (padded trace length, RAM size, bytecode size) and its prover half is not serializable, so the browser derives it per proof from the schedule catalogs and the program preprocessing (the "setup" phase, reported separately). The verifier preprocessing for that shape comes out of the same step; the prover returns it next to the proof and the demo's verifier consumes it. A real deployment pins one verifier preprocessing per program and shape; the demo's verification is not independent of the prover, since the program digest, memory layout, and schedule catalogs it checks against all arrive from the proving side.

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
src/gpu/            WebGPU harness (feature `webgpu`): mailbox transport, self-test
preprocessing/      Native binaries: artifact generation, roundtrip test
guests/             RISC-V guest programs, own cargo workspace (compiled to ELF via the jolt CLI)
frontend/           Vite + React + TypeScript + Tailwind frontend
frontend/public/    Artifacts, worker.js, gpu-proxy.js, wgsl/
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

Runs the exact browser code path natively from the shipped artifacts (setup derivation, prove, verify) for `sha2`, `ecdsa` and `sha2_chain`:

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

## WebGPU (experimental)

Feature-flagged offload of Akita field work to the GPU. W0 ships the
transport, an fp128 WGSL library, a correctness oracle and a bench; W1 runs
the stage-0 one-hot trace commit (`TracePackedOneHot::commit_inner`) on the
GPU through the `trace-commit-device` seam; W2 runs the first rounds of the
stage-1 basis-8 digit-range sumchecks through the `digit-range-device` seam.
Both are exact arithmetic on the prover's own tables, so proofs stay
byte-identical with the GPU on or off.

```bash
# Build with the harness compiled in (default builds leave it out entirely)
RUSTC_BOOTSTRAP=1 CARGO_UNSTABLE_BUILD_STD="panic_abort,std" wasm-pack build --release --target web -- --features webgpu
# ... plus the GPU trace commit (needs ./setup-wasm-deps.sh for patch 0006)
RUSTC_BOOTSTRAP=1 CARGO_UNSTABLE_BUILD_STD="panic_abort,std" wasm-pack build --release --target web -- --features webgpu,trace-commit-device
# ... plus the GPU digit-range rounds (patch 0007)
RUSTC_BOOTSTRAP=1 CARGO_UNSTABLE_BUILD_STD="panic_abort,std" wasm-pack build --release --target web -- --features webgpu,trace-commit-device,digit-range-device

# Oracle bench: gpu on vs off, proofs must match, verify must pass. Needs node server.mjs, which
# serves frontend/dist/: after editing worker.js, gpu-proxy.js or wgsl/ run `cd frontend && npm run build`
# and restart the server (it caches responses). Prints the GPU commit stage breakdown and a final
# off/on/ratio table per size.
uv run --with playwright python bench/bench_webgpu.py --iters 17,69,278,556 --runs 4 --gpu both --browser webkit
# --gpu all runs off / w1 (commit only) / on (W1+W2) / w2 (digit range only) and prints the W2 increment
# (w1 − on); --parity N shadows the first N GPU digit-range rounds with the CPU prover and aborts on a mismatch
uv run --with playwright python bench/bench_webgpu.py --iters 69 --runs 5 --gpu all
uv run --with playwright python bench/bench_webgpu.py --iters 69 --runs 1 --gpu on --parity 6

# Standalone kernel harness (WGSL vs Python reference, no server needed)
uv run --with playwright python bench/proto/commit_proto.py --shape full

# fp128 WGSL vs Python big ints, 100k random vectors + edge cases (no server needed)
uv run --with playwright python bench/test_fp128_wgsl.py
```

Browsers for the Python scripts: `uv run --with playwright python -m playwright install webkit chromium` (the node Playwright in `node_modules` bundles an older WebKit without WebGPU).

How it works:

- The prover blocks its Worker threads, and a blocked thread cannot service WebGPU promises. `frontend/public/gpu-proxy.js` is a separate dedicated Worker that owns the `GPUDevice`; it receives the shared `WebAssembly.Memory` and the address of a `#[repr(C)]` mailbox (`src/gpu/mailbox.rs`) via `postMessage`.
- Rust (`gpu::call`) writes op, args and memory regions into the mailbox, bumps the doorbell with `Atomics.notify` and blocks in `Atomics.wait` until the proxy flips `status`. The proxy `Atomics.waitAsync`s on the doorbell, copies `upload` regions with `queue.writeBuffer`, runs the op, writes `readback` regions back through a staging buffer and never blocks. Ops: NOP, CREATE_BUFFER (persistent handle), UPLOAD, DESTROY, RUN (shader id, workgroups, ≤16 u32 params in a uniform, ≤8 bindings as handles or inline regions).
- Shaders live in `frontend/public/wgsl/` (`fp128.wgsl` library + kernels); the proxy prepends the library to every kernel. WGSL has no 64-bit integers, so 32×32 products come from 16-bit halves and reduction uses 2^128 ≡ C (mod p) as wrapping `+C` folds.
- `worker.js` spawns the proxy when the `init` message carries `gpu: true` and reports `gpu: {status, adapter, features, limits}` in `init-done`. Every prove then runs a GPU self-test (2^20 random `a·b + c` mul-adds vs `AkitaField` on the CPU; `a` goes through a persistent 256 MiB buffer — CREATE_BUFFER / UPLOAD / handle binding / DESTROY — `b`, `c` as 16 MiB inline uploads, the result as a 16 MiB readback) plus 200 NOP round trips and reports `gpu_status` (`disabled` | `unavailable` | `ok` | `error: …`), `gpu_selftest_ms`, `gpu_selftest_mismatches`, `gpu_roundtrip_us` (mean over the 200 NOPs — WebKit coarsens `performance.now()` to 1 ms). That preflight is the entire gpu=on overhead in W0.
- Fallback: no `navigator.gpu`, no adapter, a proxy load failure or a wasm built without the feature all yield `gpu_status: unavailable` and the prove proceeds on the CPU unchanged. `gpu::call` is never entered unless the proxy reported ready.
- Trace commit on the GPU (`src/gpu/trace_commit.rs`, features `webgpu,trace-commit-device`): `WebGpuTraceCommit` is installed as the `jolt_akita::TraceCommitDevice` the first time the preflight self-test passes, and declines (`None` → CPU kernels, bit-identical) whenever the GPU is toggled off or the shape is not the shipped K=16 / D=512 / n_a=1 / one inner digit / 64-column geometry. Per call: pack the hot indices + masks into one byte per (row, column) (rayon), UPLOAD A and the codes into persistent buffers keyed by (P, blocks), RUN `prep` (negacyclic rotation table A2), `commit_accumulate` (workgroups (P/CHUNK, 32, blocks), 128 threads, `CHUNK` as a pipeline override), `reduce` with RES read back inline; the seam validates length and canonical limbs before converting to rings. CHUNK is the smallest of 64..2048 that divides P and keeps the PART scratch `(P/CHUNK)·64·blocks·512·32 B` under 256 MiB (2^16/2^18 → 64, 2^20 → 128, 2^21 → 256). GPU memory at 2^21: A 64 MiB + A2 128 MiB + codes 128 MiB + PART 256 MiB. The stage split (pack / upload / gpu = prep + accumulate / readback = reduce + RES readback, ms) is logged to the console as `[gpu] trace commit …` and returned as `gpu_commit` JSON on the prove result; the tracing span is `trace_onehot_commit_gpu`.
- Digit-range rounds on the GPU (`src/gpu/digit_range.rs`, features `webgpu,trace-commit-device,digit-range-device`): `WebGpuDigitRange` is installed as the `akita_prover::DigitRangeDevice` together with the trace-commit device and answers the basis-8 direct-leaf instances with ≥ 2^16 digits. Per instance: the packed digits go up once, then one RUN_SEQ mailbox op per round carries the challenge and the prover's own `GruenSplitEq` tables inline and reads the 5-coefficient round message back inline; rounds 0–2 histogram the packed digits (LUT0 host-side, LUT1 on the device), round 3 materialises the folded field table, rounds ≥ 4 fuse fold + eval; g = min(ring_bits, log2 n − 12) rounds run on the GPU (6 at every shipped size), then the folded table is downloaded and akita's `LowBasisRangeCheckProver` finishes the sumcheck on the CPU. Basis-16/32 instances and small b=8 instances never leave the CPU. Breakdown → `gpu_digit_range` JSON (`instances`, `gpu_rounds`, `ops`, `upload_ms`, `rounds_ms`, `download_ms`, `total_ms`) and a `[gpu] digit range …` console line per instance; `set_digit_range_parity_rounds(N)` (bench `--parity N`) shadows the first N rounds with the CPU prover.
- W2 result (this MacBook, idle host, WebKit, 8 threads, warm median of 5, 2^18): CPU only 2.78 s, W1 2.01 s, W1+W2 1.57 s — the W2 increment over W1 is 0.447 s (16 % of the CPU prove), just over the 0.425 s (15 %) kill rule, so W2 is unparked behind the `digit-range-device` feature. The GPU work itself is ~41 ms for the three instances; the saving is bounded by the CPU time of the six ring rounds, not by the kernels. Under host load the same pairing read 0.26 s (superseded). Details in `.journals/webgpu-akita-w2.md`.
- Secure-context trap: `navigator.gpu` is undefined on `about:blank` and plain-http non-localhost origins; `http://localhost` is fine. Playwright's Chromium only exposes WebGPU with `--enable-unsafe-webgpu --use-angle=metal` (`--browser chromium-unsafe`); plain `--browser chromium` exercises the fallback.

## Protocol

`jolt-prover/akita` selects the packed lattice pipeline: one native `OneHotTrace` commitment group over the Solinas field p = 2^128 − 2^32 + 22537 (`jolt-field/solinas`), the shared stage 1–7 sumchecks on `JoltAkitaBackend::optimized()`, and one native grouped opening (Akita PCS, [LayerZero-Labs/akita](https://github.com/LayerZero-Labs/akita)). Proofs are transparent: `akita` and `zk` (BlindFold) are mutually exclusive in jolt-prover, so this demo has no zero-knowledge mode.

**32-bit schedule subset and the 2^21 ceiling.** Akita's schedule rows for committed groups of 2^32 or more coefficients (`num_vars >= 32`) carry `usize` fields above `u32::MAX`, which wasm32 cannot deserialize. `generate-preprocessing` drops those rows (dense 36/42, one-hot K=16 40/46, K=256 40/64 kept). Those groups are polynomials wasm32 cannot address at all: the K=16 one-hot trace group has log₂(T) + 10 variables, so a padded trace of 2^22 already fails at setup with `polynomial with 32 variables exceeds the addressable domain`, before any catalog lookup. The browser ceiling is therefore **2^21 padded cycles** — the `max_trace_length` the shipped program preprocessing admits — and within it row lookup is exact-key, so the remaining shapes resolve unchanged. The catalog digest bound into the transcript differs from jolt's full catalogs, so browser proofs verify against the verifier preprocessing the prover emits (which carries the same trimmed catalogs), not against a verifier loaded with the packaged `.aks` files.

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
that are not upstream yet, plus one device seam, shipped in `patches/`:

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
6. `0006-jolt-akita-trace-commit-device.patch` — not a wasm32 fix: adds
   `jolt_akita::TraceCommitDevice`, a process-global seam that lets the
   stage-0 one-hot trace commit run on an external device (WebGPU) with the
   CPU kernels as fallback. No behaviour change unless a device is installed.
   The root crate uses it behind the `trace-commit-device` feature; contract
   in [docs/trace-commit-device.md](docs/trace-commit-device.md).
7. `0007-akita-digit-range-device.patch` — not a wasm32 fix: adds
   `akita_prover::DigitRangeDevice`, a process-global seam consulted on the
   basis-8 direct leaf of the stage-1 digit-range sumcheck. A device answers
   round messages from host-uploaded eq tables; `DeviceLeaf` wraps the CPU
   prover so the transcript, the eq-factored driver and the verifier are
   untouched, and resumes the CPU prover from the device's folded table
   (`LowBasisRangeCheckProver::from_materialized`). No behaviour change
   unless a device is installed. The root crate uses it behind the
   `digit-range-device` feature (which adds `akita-prover` as a direct dependency).

`./setup-wasm-deps.sh` clones the three upstreams at the pinned revs into
`.wasm-deps/`, applies the patches, and rewrites the marked override block in
`Cargo.toml` onto the patched checkouts. `./setup-wasm-deps.sh --revert`
restores the pinned block.
