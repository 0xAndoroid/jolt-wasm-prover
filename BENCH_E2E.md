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

## Algorithm parity with jolt-core's CPU tricks (optimization round)

The first measurement round ran textbook GPU pipelines against jolt-core's
production routines. The second round ports the CPU exploits one-for-one:

1. **Small-scalar signed MSM** (jolt-core `msm_signed`/`msm_i128`): dense
   tier-1 scalars are recoded CPU-side into sign + canonical magnitude
   (a small negative is `p - |s|` canonically, which would force full
   width), the per-scalar sign is folded into the signed-digit decoder, and
   the Pippenger window count is clamped to the actual magnitude width —
   RdInc/RamInc run 9 windows instead of 32.
2. **GLV decompositions** (jolt-core `JoltG1Routines`/`JoltG2Routines`
   folds): the per-round challenge folds and the row-commitment RLC
   CPU-decompose the shared scalar with the fork's exact
   `decomp_2d`/`decomp_4d` and run 2-point (G1, 128-bit) / 4-point (G2,
   ~66-bit, lazy ψ recompute) Shamir ladders — 2-4x fewer doublings than
   the 254-bit ladder.
3. **Prepared pairing lines**: already at parity for fixed-G2 pairings
   (tier-2 and D1 consume uploaded arkworks `G2Prepared` lines). The
   varying-Q pairings (C±, later-round D2) — where the CPU also computes
   lines per call — now run the G2 doubling/addition line steps on-device
   inside the same dispatch as the evaluation (below).
4. **Occupancy/dispatch shape**: the one-hot gather is striped 32x with a
   per-row reduce (all polys in one compute pass); Miller loops gained a
   workgroup-cooperative variant executing the ENTIRE ate loop in a single
   dispatch — a 32-lane workgroup per pair runs a CPU-built stream of Fq2
   micro-ops (one `fq2_mul` call site total, barriers in uniform control
   flow) over a shared-memory slot file, which is what Apple's Metal
   compiler constraints allow of the CUDA warp-cooperative design. It wins
   only in the dispatch-bound regime: above ~512 pairs the sequential
   per-pair-thread pipeline (full occupancy, register-resident Fq6 chains)
   is faster, so dispatch is hybrid (`COOP_MAX_PAIRS`).

### Native per-phase, before → after (one run, keccak 2^20)

| Phase | GPU before | GPU after | CPU (busy/12 threads) |
|---|---|---|---|
| tier-1 dense (2 polys) | 15.1 s | **5.6 s** | 3.0 s busy (`msm_i128`) |
| tier-1 one-hot (42 polys) | 9.1 s | **6.8 s** | 11.6 s busy |
| tier-2 multipairings | 4.3 s | **4.4 s** | 33.0 s busy (~2.8 s wall) |
| `combine_hints` (row RLC) | 1.5 s | **0.7 s** | 0.9 s |
| opening rounds | 11.3 s | **9.1 s** | ~1.0 s (in `create_evaluation_proof`) |
| witness commit total | 29.6 s | **17.9 s** | 4.9 s |
| stage 8 total | 13.0 s | **10.0 s** | 2.1 s |
| **prove e2e** | **44.8 s** | **30.2 s** | **9.1 s** |

Remaining gap analysis (native): the CPU's 12 performance cores execute
~3.7 GHz superscalar 64-bit Montgomery arithmetic with GLV + prepared
lines; the GPU runs 16-bit-limb unrolled WGSL of the same algorithms. The
residual loss is spread across the opening's wide rounds (9.1 s — early
rounds are occupancy-fine but each round is still ~450 dispatch barriers x
2 multipairings plus G2 MSMs), the one-hot gather's uncoalesced base loads
(6.8 s), and the dense MSM's bucket passes (5.6 s). Next levers, in
impact order: two-pairs-per-workgroup cooperative scheduling with f kept
in registers (lifts the <512-pair crossover), digit-sorted bucket
accumulation for the MSMs, and column-major index layout for the gather.

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

| scale (cycles) | CPU-WASM Dory | full-GPU before | full-GPU after parity round |
|---|---|---|---|
| 2^18 (201,547) | **10.69 s** (10.69/10.69/10.90) | 221.1 s (189.3/221.1/234.3) | **212.9 s** (190.8/212.9/238.5) |
| 2^20 (1,003,399) | **32.61 s** (32.54/32.61/32.63) | 848.6 s (single timed run) | **876.6 s** (single timed run) |

(CPU 2^20 measured 26.8 s in the first session and 32.6 s in this one —
long-session thermal drift; pairs above are same-session. GPU heap peaks
0.31 / 1.17 GB — fits wasm's 4 GB cap.)

**The browser numbers did not move with the parity round, and that is the
finding:** the same kernels got 33% faster on native Metal, so the
browser's binding constraint is not algorithm choice or dispatch count but
Tint's codegen of the unrolled BN254 field arithmetic (the known 3-17x
penalty vs naga+Metal). Routing every multipairing through the
single-dispatch cooperative path — eliminating the ~450-dispatch
structure entirely — measured the same within thermal variance (212.9 vs
223.7 medians, identical first runs), confirming per-dispatch overhead is
not the limiter either. A browser-side win needs Tint to emit better code
for the 16-bit-limb Montgomery kernels (or a different limb encoding
tuned for Tint), not protocol-level restructuring.

## Summary

| | CPU Dory e2e | GPU (first round) | GPU (parity round) |
|---|---|---|---|
| native Metal, 2^20 | **9.10 s** | 44.8 s (5.0x) | **30.2 s** (3.3x) |
| browser, 2^18 | **10.69 s** | 221.1 s | **212.9 s** (19.9x) |
| browser, 2^20 | **32.61 s** | 848.6 s | **876.6 s** (26.9x) |

The full-GPU port is correct everywhere (stock-verifier acceptance in all
configurations, byte-identical commitments) and now runs jolt-core's own
algorithms — small-scalar signed MSMs, GLV-decomposed folds, prepared
pairing lines — yet still loses end-to-end: natively 3.3x (down from
5.0x), in-browser ~20-27x. The remaining native gap is hardware-shaped
(12 superscalar 3.7 GHz cores on 64-bit Montgomery limbs vs 16-bit-limb
WGSL), concentrated in the opening's wide rounds, the gather's uncoalesced
loads, and the MSM bucket passes; the browser gap is dominated by Tint's
arithmetic codegen.

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

- `cargo nextest run -p dory-gpu` — 33 tests: kernel-level vs arkworks
  (incl. the cooperative Miller kernels and GLV folds bit-matching
  arkworks with masked/identity inputs), Transparent proofs byte-identical
  to dory-pcs (square + rectangular, matrix-resident + virtual +
  unfused-commit paths), ZK proofs accepted by the stock dory-pcs
  verifier, batched tier-2 vs `multi_pair_g2_setup` (including identity
  rows).
- `bench-e2e --check` — full Jolt proofs from both PCS implementations
  verify with the stock verifier; witness commitments byte-identical
  (exercised at 2^15 rectangular and 2^20 square).
