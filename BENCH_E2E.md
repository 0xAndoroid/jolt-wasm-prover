# End-to-end Jolt prover: CPU Dory vs full-GPU Dory

Full Jolt proofs (keccak-chain guest) with the polynomial commitment scheme
swapped between the stock CPU `DoryCommitmentScheme` and the full-GPU
`GpuDoryCommitmentScheme` (`gpu-pcs` crate), measured native (Metal) and
in-browser (Chrome WebGPU). Every proof in every configuration is verified
by the **stock verifier**; witness commitments are Transparent-mode and
**byte-identical between CPU and GPU runs** (`bench-e2e --check` asserts
this; the ZK opening is OsRng-blinded and therefore nondeterministic by
design — even two CPU runs differ).

## What runs on the GPU

The GPU PCS unfuses jolt-core's streamed witness commit (the `ChunkState`
associated type carries raw rows instead of MSM results — no jolt-core
patch) and materializes the polynomial data on the device, so the **entire
Dory PCS is GPU-resident**:

| Phase | Stock CPU path | GPU path |
|---|---|---|
| tier-1 dense rows (RdInc/RamInc) | streamed `msm_i128` per row | batched row MSMs (split Pippenger pipeline) |
| tier-1 one-hot rows (~40 Ra polys) | streamed batch additions per row | gather-add kernel (`onehot.wgsl`) |
| tier-2 commitments | `multi_pair_g2_setup` per poly | one batched multipairing per coalesced batch (prepared lines) |
| row-commitment RLC (`combine_hints`) | GLV vector fold | GPU fold passes over resident row buffers |
| joint RLC matrix | never materialized (streamed) | `rlc.wgsl` combine over resident poly data |
| opening `v = L^T M` (VMV) | streamed from the trace | GPU VMV kernel over the joint matrix |
| opening reduce rounds (multipairings, MSMs, folds) | `DoryProverState` | GPU round pipeline (`open.rs`) |
| evaluation `y` (ZK) | streamed `poly.evaluate` | `<v_vec, right>` dot from the GPU VMV output |
| transcript, ZK blinds/masks, final exponentiations | CPU | CPU (cheap, sequential by nature) |

CPU-side the witness is still *generated* by jolt-core's streaming code
(that is execution-trace processing, not Dory); the rows are buffered and
shipped to the GPU. Memory cost of unfusing at T=2^20: ~190 MB of one-hot
indices + ~64 MB of dense rows (CPU + GPU copies) plus the 512 MB joint
matrix on the GPU — the streamed CPU path needs none of that and remains
the default for non-benchmark builds (the PCS type parameter is the switch).

Advice polynomials are out of scope (CPU commit, opening asserts their
absence); the benchmark guests use none.

## Native (Apple Silicon, Metal)

Guest: `sha3-chain` (keccak), 300 iterations → 1,003,399 cycles, padded
T = 2^20. Main Dory context: K_chunk = 16, matrix 2^12 x 2^12 (square),
44 committed polynomials (2 dense + 42 one-hot). 3 runs after a discarded
warmup; medians; every proof verified by the stock verifier.

| | trace | **prove e2e** | runs |
|---|---|---|---|
| CPU Dory | 0.05 s | **9.011 s** | 8.99 / 9.01 / 9.04 |
| GPU Dory | 0.05 s | **44.815 s** | 44.81 / 44.81 / 44.84 |

**The optimized CPU baseline wins e2e by ~5x.** The earlier standalone
dory-gpu benchmark (GPU opening 7.4 s vs 37 s CPU at 2^20) compared against
*stock dory-pcs routines*; the e2e prover instead uses jolt-core's
GLV/prepared-line optimized routines across 12 cores — a ~10-30x faster CPU
opponent, and the GPU port does not overcome that.

### Phase breakdown (one run, wall time; busy-times for parallel CPU spans)

| Phase | CPU | GPU | notes |
|---|---|---|---|
| witness commit total (`generate_and_commit_witness_polynomials`) | 4.86 s | 29.6 s | |
| — tier-1 dense (2 polys) | 3.0 s busy (`msm_i128` x512) | 15.1 s | GPU: full 254-bit Pippenger per 2^20-entry matrix; CPU: small-scalar i128 MSM |
| — tier-1 one-hot (42 polys) | 11.3 s busy (x10240 rows) | 9.1 s | GPU gather kernel is occupancy-starved (4096 threads/dispatch) |
| — tier-2 multipairings | 37-41 s busy / x42 (~3.2 s wall at 12 threads) | 4.3 s | GPU: 5-6 coalesced batches, prepared-line Miller |
| stage 8 (batched opening) | 2.04 s | 13.0 s | |
| — `combine_hints` (row RLC) | 0.95 s | 1.5 s | |
| — opening proper | 1.10 s (`create_evaluation_proof`) | 11.4 s (joint matrix + VMV ~0.2 s; reduce rounds 11.3 s) | GPU rounds are dispatch-bound: ~450 sequential Miller dispatches per multipairing (Metal compiler limits, see CLAUDE.md) |

Attribution of the GPU loss, in order:

1. **Dense tier-1 MSMs (15.1 s for two polynomials).** RdInc/RamInc scalars
   are small (i128 increments); the CPU exploits that (`msm_i128`), the GPU
   pipeline runs full-width 254-bit windows. Small-scalar window clamping is
   the obvious lever.
2. **Opening reduce rounds (11.3 s).** The known dispatch-granularity
   bottleneck: Apple's Metal compiler forces small kernels, so a
   multipairing is ~450 sequential dispatches. Workgroup-cooperative Fq12
   kernels (fusing an ate step into 2-3 dispatches) are the next lever.
3. **One-hot gather (9.1 s, ~230 ms/poly).** One thread per output row =
   4096 threads — far below occupancy. Striped partial accumulators with a
   reduce pass would multiply parallelism by ~32.
4. Tier-2 batching works as designed (4.3 s for 184k pairings).

## Browser (headless Chromium, WebGPU/Tint + wasm, 10 worker threads)

Same guest and protocol: the prover runs in a worker (rayon pool of 10),
the GPU engine on a dedicated worker sharing the wasm heap (shared-memory
job queue, `Atomics.waitAsync` pump). Warmup run discarded (the first GPU
run pays Tint pipeline compilation: 746 s at 2^18, 1447 s at 2^20 — Tint
recompiles the unrolled BN254 modules per fresh device). Every proof
verified in-browser by the stock `WasmVerifier`.

| scale (cycles) | CPU-WASM Dory | full-GPU Dory | GPU/CPU | wasm heap peak |
|---|---|---|---|---|
| 2^18 (201,547) | **10.71 s** (10.66/10.71/10.76) | **221.1 s** (189.3/221.1/234.3) | 20.6x | 0.23 / 0.30 GB |
| 2^20 (1,003,399) | **26.80 s** (26.75/26.80/27.38) | **848.6 s** (single timed run) | 31.7x | 0.95 / 1.20 GB |

The browser amplifies every native bottleneck: Tint+AGX executes the same
WGSL 3-17x slower than naga+Metal and per-dispatch overhead is larger, so
the sequential-dispatch pairing structure dominates even harder. The 2^20
materialized path fits wasm's 4 GB cap with room (1.2 GB peak). GPU
run-to-run variance is high (189-234 s at 2^18) — thermals plus wasm heap
growth; medians over 3 runs except browser-GPU 2^20 (one timed run,
~14 min each).

## Summary

| | CPU Dory e2e | full-GPU Dory e2e |
|---|---|---|
| native Metal, 2^20 | **9.01 s** | 44.8 s (5.0x slower) |
| browser, 2^18 | **10.71 s** | 221.1 s (20.6x) |
| browser, 2^20 | **26.80 s** | 848.6 s (31.7x) |

The full-GPU port is correct everywhere (stock-verifier acceptance in all
configurations, byte-identical commitments) but loses end-to-end in both
environments. The standalone-bench conclusion ("native GPU wins") does not
transfer: it measured against stock dory-pcs CPU routines, while the e2e
prover's CPU baseline is the heavily optimized GLV/prepared-line code on
all cores. Closing the native 5x gap needs, in impact order: small-scalar
window clamping for the dense tier-1 MSMs, workgroup-cooperative Fq12
kernels to collapse the ~450-dispatch multipairings, and striped one-hot
gather accumulation.

## Reproducing

```bash
# Native (release build, requires the local arkworks fork checkout)
cargo build --release --features native --bin bench-e2e
./target/release/bench-e2e --guest keccak --iters 300 --check      # gates
./target/release/bench-e2e --guest keccak --iters 300 --pcs cpu --runs 3
./target/release/bench-e2e --guest keccak --iters 300 --pcs gpu --runs 3

# Browser (after wasm-pack + frontend build + server, see CLAUDE.md)
node e2e-bench-browser.mjs 300 3 cpu,gpu
```

Correctness gates, all green:

- `cargo nextest run -p dory-gpu` — 30 tests: kernel-level vs arkworks,
  Transparent proofs byte-identical to dory-pcs (square + rectangular,
  matrix-resident + virtual + unfused-commit paths), ZK proofs accepted by
  the stock dory-pcs verifier, batched tier-2 vs `multi_pair_g2_setup`
  (including identity rows).
- `bench-e2e --check` — full Jolt proofs from both PCS implementations
  verify with the stock verifier; witness commitments byte-identical
  (exercised at 2^15 rectangular and 2^20 square).
