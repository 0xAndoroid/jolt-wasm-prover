# Jolt WASM Prover

In-browser zero-knowledge proving and verification using [Jolt](https://github.com/a16z/jolt). Compiles the Jolt zkVM prover and verifier to WebAssembly with multithreading support via `SharedArrayBuffer` and `wasm-bindgen-rayon`.

## Programs

Three guest programs are included:

| Program | Description | Guest crate |
|---------|-------------|-------------|
| **SHA-256** | Hash arbitrary input | `guests/sha2` |
| **ECDSA** | Secp256k1 signature verification | `guests/secp256k1` |
| **Keccak Chain** | Iterated Keccak-256 hashing | `guests/sha3-chain` |

## Prerequisites

- Rust nightly (managed via `rust-toolchain.toml`)
- `wasm-pack`: `curl https://drager.github.io/wasm-pack/installer/init.sh -sSf | bash`
- Node.js (for the dev server)

## Quick Start

### 1. Generate preprocessing data

Preprocessing generates the Dory SRS, compiles guest ELFs, and serializes prover/verifier preprocessing into `www/`.

```bash
cargo run --release --features native --bin generate-preprocessing
```

This produces per-program files in `www/`:
- `{name}_prover.bin` — prover preprocessing (Dory SRS + shared preprocessing)
- `{name}_verifier.bin` — verifier preprocessing (Dory verifier setup + shared preprocessing)
- `{name}.elf` — compiled guest RISC-V ELF

### 2. Build WASM

```bash
wasm-pack build --release --target web
```

Outputs the WASM package to `pkg/`.

### 3. Run

```bash
node server.mjs
# Open http://localhost:8080
```

The server sets `Cross-Origin-Opener-Policy` and `Cross-Origin-Embedder-Policy` headers required for `SharedArrayBuffer`.

## Architecture

```
src/lib.rs          WASM entry point — WasmProver, WasmVerifier, init_inlines
src/wasm_tracing.rs Chrome Trace Format profiling (Perfetto-compatible)
preprocessing/      Native binaries for generating preprocessing data
guests/             RISC-V guest programs (compiled to ELF by jolt-sdk)
www/                Static frontend + preprocessing artifacts
server.mjs          Dev server with COOP/COEP headers
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
