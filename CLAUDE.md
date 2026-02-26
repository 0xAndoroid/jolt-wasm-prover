# CLAUDE.md

## Project Overview

WASM prover/verifier demo for [Jolt](https://github.com/a16z/jolt) zkVM. Compiles Jolt's prover and verifier to WebAssembly, runs in browser with multithreading via `wasm-bindgen-rayon`.

## Commands

```bash
# Build WASM package (outputs to pkg/)
CARGO_UNSTABLE_BUILD_STD="panic_abort,std" wasm-pack build --release --target web

# Build native preprocessing generator
cargo build --release --features native

# Generate preprocessing artifacts into www/
cargo run --release --features native --bin generate-preprocessing

# Roundtrip test for serialization
cargo run --release --features native --bin test-roundtrip

# Clippy / format
cargo clippy --all --message-format=short -q
cargo fmt -q

# Dev server (serves on http://localhost:8080)
node server.mjs
```

## Benchmarking

Requires: `npm install`, Playwright Chromium (`npx playwright install chromium`), dev server running.

```bash
# Start server in background
node server.mjs &

# Run SHA-2 proving benchmark (default 3 runs)
node bench.mjs

# Custom run count
node bench.mjs 5
```

Outputs JSON to stdout: `{"runs":[1.66,1.54,1.53],"avg":1.577,"min":1.53,"max":1.66}`

Per-run timings printed to stderr. Runs headless Chromium (12 threads max) against `http://localhost:8080`.

**Browser performance varies**: Safari ~1.63s, Chromium/Playwright ~1.55s, Brave ~2.12s (shields/fingerprinting protection throttles `hardwareConcurrency` and adds overhead).

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

## WASM Proving Performance Optimization

### Modifiable vs Non-Modifiable Code

Modifiable (local path dependency via `~/dev/arkworks-algebra`, branch `wasm`):
- `ark-ff`, `ark-ec`, `ark-serialize`, `ark-bn254`, `ark-poly`, `jolt-optimizations`

Non-modifiable (git dependencies):
- `jolt-core` (a16z/jolt, branch wasm-clean)
- `dory-pcs` (a16z/dory)

### Results: 2.677s → 1.57s avg (41% improvement)

### Changes Applied (arkworks-algebra, branch wasm)

1. **Parallel MSM for WASM** (`ec/src/scalar_mul/variable_base/mod.rs`)
   - Removed sequential WASM fallback, restored parallel inner MSM via `msm_bigint_wnaf_parallel`
   - WASM chunk sizing: `num_chunks = cur_num_threads` (no nested ThreadPoolBuilder)
   - On WASM, rayon's global thread pool is fixed at init; nested `ThreadPoolBuilder::new()` panics
   - The outer `cfg_chunks!` splits bases across threads, inner `msm_bigint_wnaf_parallel` parallelizes windows via `into_par_iter().map_with()` for bucket reuse
   - **Impact: 2.677s → 2.22s (biggest single win)**

2. **MIN_PAR_SIZE threshold** (`dory_g1.rs`, `dory_g2.rs`, `dory_utils.rs`, `glv_two.rs`, `glv_four.rs`)
   - `const MIN_PAR_SIZE: usize = 64` — par_iter only when collection ≥ 64 elements
   - Eliminates rayon overhead on small arrays (e.g. 2-element GLV bases, 16-element tables)
   - Rayon's work-stealing overhead is ~1-5μs per task; for <64 elements the overhead exceeds the parallelism benefit

3. **G2 affine mixed addition + thread-local cache** (`glv_four.rs`)
   - Thread-local cache (512 bytes) for affine Frobenius bases — avoids recomputing `normalize_batch` for repeated base points
   - `shamir_glv_mul_4d_affine` uses mixed addition (projective += affine) which costs 7M+3S vs 12M+2S for projective += projective
   - Raw byte comparison via `unsafe ptr::read` for cache key (avoids expensive projective PartialEq which does field divisions)
   - Cache hit pattern: in Dory's inner loop, the same base point is multiplied by different scalars repeatedly

4. **G1 affine mixed addition + batch normalization** (`glv_two.rs`, `dory_g1.rs`)
   - `shamir_glv_mul_2d_affine` takes pre-normalized affine bases, uses mixed addition loop
   - `glv_endomorphism_affine`: just `(x, y) → (β·x, y)` — single Fq multiplication (vs projective version that also multiplies Z)
   - Vector operations (`vector_add_scalar_mul_g1_online`, `vector_scalar_mul_add_gamma_g1_online`) batch-normalize ALL points in one `normalize_batch` call (1 batch inversion for N points) instead of per-element normalization

5. **Serial batch addition** (`batch_addition.rs`)
   - Removed all par_iter from batch addition — batch_inversion + affine formulas are memory-bound, not CPU-bound
   - The batch inversion is a sequential scan (Montgomery's trick), parallelizing it adds overhead without benefit
   - **Impact: compute_tier1_commitment_onehot -23.8% (219→167ms)**

6. **Zero-allocation scalar decomposition** (`decomp_4d.rs`, `decomp_2d.rs`)
   - Replaced `num_bigint::BigInt` heap-based decomposition with raw u64 limb arithmetic
   - 4D (G2): precomputed table of 256 power-of-2 decompositions (`POWER_OF_2_DECOMPOSITIONS`), bit-scanning via `trailing_zeros`. Each scalar processed by iterating set bits and accumulating i128 coefficients
   - 2D (G1): field multiplication `scalar * N22` gives remainder mod r, schoolbook multiplication gives full product, exact division via Montgomery's method (`r_inv mod 2^64`) recovers quotient. Rounding via `2*remainder > r` check
   - Removed `num-bigint`, `num-rational`, `num-integer`, `num-traits` dependencies entirely
   - `num_bigint::BigInt` was allocating on heap for every scalar decomposition — thousands of allocations per proof

7. **Eliminated wasteful 256-entry table builds** (`dory_utils.rs`, `dory_g2.rs`)
   - G2 online functions (`vector_scalar_mul_add_online`, `vector_scalar_mul_v_add_g_precomputed`, `vector_add_scalar_mul_g2_online`, `vector_scalar_mul_add_gamma_g2_online`) were building a 256-entry `PrecomputedShamir4Table` per element for SINGLE USE
   - Table construction: ~256 projective additions (~10K Fq2 muls) vs direct affine Shamir with ~66-bit sub-scalars: ~4K Fq2 muls
   - Replaced with: batch `normalize_batch` on all generators/v → affine Frobenius (`frobenius_psi_power_affine`) → `shamir_glv_mul_4d_affine`
   - **Impact: 1.84s → 1.57s (14.5% improvement)**

8. **Direct affine Frobenius** (`frobenius.rs`)
   - `frobenius_psi_power_affine` operates directly on Fq2 affine coordinates
   - For odd powers: conjugate x and y (negate c1 component of Fq2), then multiply by precomputed coefficients
   - Previously: converted affine→projective, applied Frobenius on projective (3 Fq2 muls + conjugation), then converted back via `normalize_batch` (inversion + 2 muls). Now: 2 Fq2 muls + conditional conjugation

9. **Bug fix: decompose_scalar_2d sign error** (`decomp_2d.rs`)
   - The Babai rounding formula had wrong sign for k1
   - Root cause: the 2D GLV lattice vectors have mixed sign conventions:
     - v1 = (N11, -N12): satisfies N11 - N12·λ ≡ 0 (mod r)
     - v2 = (N21, N22): satisfies N21 + N22·λ ≡ 0 (mod r), where N21 = N12
   - Correct: `k1 = scalar - (q1*N11 + q2*N21)` — the code had MINUS instead of PLUS
   - k2 formula `q1*N12 - q2*N22` was correct
   - Bug was masked in production because both prover and verifier used the same wrong decomposition, so proofs verified despite incorrect intermediate values
   - All 8 `glv_two_tests` now pass (were 0/8 before fix)

### Approaches That Failed

| Approach | Result | Why |
|----------|--------|-----|
| NAF (Non-Adjacent Form) | +526ms regression | WASM instruction overhead from NAF recoding loop (conditional ops + BigInt div-by-2 via shift+borrow) exceeds savings from ~33% fewer EC additions. NAF saves ~1/3 additions but adds ~2× instruction overhead per scalar bit |
| SIMD (+simd128) | +478ms regression | LLVM autovectorizes carry-heavy Montgomery multiplication into 128-bit SIMD ops. But WASM SIMD v128 has no native carry propagation — compiler synthesizes it with extra shuffles/extracts. Net effect: more instructions than scalar code |
| PrecomputedShamir4Table cache (48KB per point) | +510ms regression | Each table is 256 × G2Projective = 256 × 192 bytes = 48KB. Accessing random table entries thrashes L1 cache (typically 32-64KB). Sequential Shamir with mixed addition has better cache locality |
| MSM window size tuning (reduce c by 1-2 for WASM) | Neutral | Actual window sizes: N=512→c=8 (256 buckets), N=8192→c=10 (1024 buckets). Memory note claiming c=14-16 was wrong. Current formula `ln_without_floats(size) + 2` already near-optimal for these sizes |
| Combined affine Shamir table (G1, 3-entry: P, λP, P+λP) | Neutral | Reduces additions when both bits set (25% of iterations save 1 mixed add). But precomputation (1 projective addition + batch normalize for P+λP) costs ~212M Fq muls per element, while savings are ~310M over 128 bits. Net: ~100M savings ≈ 0.8μs per element — within noise |
| Single MSM chunk (no outer parallelism) | Regression | Loses inter-chunk parallelism. The outer chunking across threads is important for WASM where all threads share a single pool |

### Profiling (trace-capture.mjs)

Top bottlenecks by self-time after all optimizations:

| Span | Self-time | % | Modifiable? |
|------|-----------|---|-------------|
| InstructionReadRafSumcheckProver::compute_message | 412ms | 26% | No (jolt-core) |
| multi_pair_g2_setup_parallel | 338ms | 21% | No (dory-pcs) |
| bound_poly_var_top_zero_optimized | 188ms | 12% | No (jolt-core) |
| vector_scalar_mul_add_gamma_g1_online | 117ms | 7% | Yes (dory_g1.rs) — at algorithmic limit |
| msm_field_elements | 105ms | 7% | Yes (ark-ec MSM) — near-optimal window |
| multi_pair_parallel | 92ms | 6% | No (dory-pcs) |
| multi_pair_g1_setup_parallel | 67ms | 4% | No (dory-pcs) |
| BytecodeReadRafSumcheckProver::compute_message | 57ms | 4% | No (jolt-core) |
| DoryProverState::apply_first_challenge | 53ms | 3% | No (dory-pcs) |
| DoryProverState::apply_second_challenge | 40ms | 3% | No (dory-pcs) |

**~67% of proving time is now in non-modifiable code** (jolt-core sumcheck provers, dory-pcs pairing setup).

Remaining modifiable self-time: ~260ms total:
- `vector_scalar_mul_add_gamma_g1_online` (117ms): does v[i] = scalar*v[i] + gamma[i] using 2D GLV + affine Shamir. Core loop is 128 doublings + ~64 mixed additions per element. At algorithmic limit for bit-by-bit Shamir.
- `msm_field_elements` (105ms, 59 calls): standard Pippenger MSM via `msm_signed` → `msm_bigint_wnaf`. Called from jolt-core via `ArkVariableBaseMSM::msm_serial`. Window size and parallelism already optimal.
- `msm_i128` (24ms self, 130ms total): handles i128-scalar MSMs. Calls into `msm_field_elements` for the field-element portion.

### Technical Details

#### Sign Conventions (CRITICAL)
- **4D decomposition** (`decomp_4d.rs`): `signs[i] = true` means coefficient is NEGATIVE
- **4D Shamir** (`glv_four.rs`): `if signs[i] { -bases[i] } else { bases[i] }` — negates base for negative coefficients
- **2D decomposition** (`decomp_2d.rs`): `signs[i] = !k_neg` — `true` means POSITIVE
- **2D Shamir** (`glv_two.rs`): `if signs[i] { bases[i] } else { -bases[i] }` — keeps base for positive coefficients
- These opposite conventions are intentional and internally consistent

#### GLV Lattice Vectors (BN254 G1)
- N11 = 147946756881789319000765030803803410728
- N12 = N21 = 9931322734385697763
- N22 = 147946756881789319010696353538189108491
- det(B) = N11*N22 + N12² = r (the scalar field order)
- Verified: N11*P - N12*φ(P) = 0 and N21*P + N22*φ(P) = 0
- φ²(P) + φ(P) + P = 0 (λ² + λ + 1 ≡ 0 mod r)

#### MSM Architecture
- `msm_bigint_wnaf` (outer): chunks bases across `cur_num_threads` on WASM (no nested ThreadPoolBuilder)
- `msm_bigint_wnaf_parallel` (inner): WNAF digit extraction (parallel flat_map), then parallel window processing via `into_par_iter().map_with()` for bucket Vec reuse
- `msm_signed` (grouping): partitions scalars by bit-size into {±1, ±u8, ±u16, ±u32, ±u64, full}. For full-size field elements (the common case), all scalars go to the `msm_bigint_wnaf` group
- `msm_field_elements` (jolt-core): calls `msm_serial` which calls `msm_bigint_serial` → `msm_signed(serial=true)`. The `serial=true` flag only affects sub-MSMs for small scalars (empty for full-size inputs). The full-scalar `msm_bigint_wnaf` is always parallel.

#### WASM Field Arithmetic
- BN254 Fq: no-carry CIOS Montgomery mul (CAN_USE_NO_CARRY_MUL_OPT=true), N=4 limbs
- `widening_mul` on WASM: splits u64×u64 into 4× u32×u32 (no native 64-bit multiply)
- Fq2 mul uses `sum_of_products<M=2>` (48 mac ops), equivalent cost to Karatsuba (already optimal)
- EC doubling (a=0 for BN254): 3M+4S in projective
- Mixed addition (projective += affine): 7M+3S+1SOP — ~30% cheaper than projective += projective (12M+2S)
- Build settings already optimal: LTO=fat, codegen-units=1, opt-level=3, build-std
- `wasm-opt = false` in `[package.metadata.wasm-pack.profile.release]` — wasm-opt pass was neutral/harmful

#### Thread Configuration
- `www/worker.js` caps threads: `Math.min(navigator.hardwareConcurrency || 4, 12)`
- Brave browser may report lower `hardwareConcurrency` due to fingerprinting protection → fewer threads → slower proving
- Playwright headless Chromium reports true core count, capped at 12
