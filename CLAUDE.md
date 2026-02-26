# CLAUDE.md

## Project Overview

WASM prover/verifier demo for [Jolt](https://github.com/a16z/jolt) zkVM. Compiles Jolt's prover and verifier to WebAssembly, runs in browser with multithreading.

## Commands

```bash
# Build (native preprocessing generator)
cargo build --release --features native

# Build WASM
wasm-pack build --release --target web

# Generate preprocessing artifacts into www/
cargo run --release --features native --bin generate-preprocessing

# Roundtrip test for serialization
cargo run --release --features native --bin test-roundtrip

# Clippy
cargo clippy --all --message-format=short -q

# Format
cargo fmt -q

# Dev server
node server.mjs
```

## Architecture

- `src/lib.rs` — `#[wasm_bindgen]` exports: `WasmProver`, `WasmVerifier`, `init_inlines`, tracing
- `src/wasm_tracing.rs` — Chrome Trace Format layer for `tracing`, outputs Perfetto-compatible JSON
- `preprocessing/generate.rs` — native binary: compiles guests, generates Dory SRS, serializes prover/verifier preprocessing to `www/`
- `preprocessing/test_roundtrip.rs` — native binary: validates serialize/deserialize roundtrip
- `guests/{sha2,secp256k1,sha3-chain}/` — RISC-V guest programs using `jolt-sdk`
- `www/` — static frontend, preprocessing `.bin` files, guest `.elf` files
- `server.mjs` — Node.js dev server with COOP/COEP headers for SharedArrayBuffer

## Feature Flags

- **default** (no features) — WASM library build (`cdylib`)
- **`native`** — enables `jolt-core/host` and guest compilation; required for preprocessing binaries

## Key Dependencies

All Jolt crates pulled from `https://github.com/a16z/jolt` (default branch). Arkworks from `a16z/arkworks-algebra` branch `dev/twist-shout`. Dory from `https://github.com/a16z/dory` (git HEAD, includes cross-platform usize serialization fix).

## WASM Build Requirements

- Nightly Rust (for `build-std` with atomics)
- `.cargo/config.toml` sets `+atomics,+bulk-memory,+mutable-globals` and 4 GB max memory
- `wasm-pack` for building the WASM package

## Serialization

Preprocessing uses **uncompressed** arkworks serialization (`Compress::No`) for fast deserialization in WASM. Proofs use **compressed** serialization (`Serializable::serialize_to_bytes`) for smaller transfer size.

## Comment Policy

- No comments restating what code does
- No commented-out code
- Keep WHY comments, SAFETY comments, WARNING comments
