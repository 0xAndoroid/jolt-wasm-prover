# W1 prototype — one-hot trace commit accumulate on WebGPU (standalone)

Standalone WGSL kernels + a Playwright/WebKit harness for the Akita prover's #1 hot spot
(`trace_onehot::commit` accumulate: 26 % of prove time, 0.73 s CPU-WASM at 2^18 / 8 threads).
No Rust, no wasm. Numbers below: Apple M5 Max (40 GPU cores), Playwright headless WebKit, sha2-chain 2^18 shape
(T = 262144 rows, 57 columns / capacity 64, K = 16, D = 512, 4 blocks x 2048 positions, 256 output rings).

```
uv run --with playwright --with numpy python bench/proto/commit_proto.py --shape full            # chosen kernel
uv run --with playwright --with numpy python bench/proto/commit_proto.py --shape small --variant v9 --chunk 64   # exact, 16384 coefs
uv run --with playwright --with numpy python bench/proto/commit_proto.py --shape stress --chunk 2048           # digit-accumulator bound
uv run python bench/proto/gen_variants.py                                                          # rewrite commit_accumulate.wgsl
```

Files: `common.wgsl` (Params, constants) · `prep.wgsl` · `commit_accumulate.wgsl` (chosen, generated) · `reduce.wgsl` ·
`commit_accumulate_v1.wgsl` (hand-written baseline) · `harness.html` · `commit_proto.py` · `gen_variants.py`.
Variants v2b–v14 are not checked in: `--variant vN` builds the source in memory from `gen_variants.VARIANTS`.

Correctness shapes (Python big-int reference, independent of the kernel: `rot(A,s)[i] = A_ext[(i−s) mod 1024]`
with `A_ext = [A, −A]`, sum, `% p`): `small` checks every coefficient (8 columns × 4 blocks × 512) and covers A ∈ {p−1, 0},
rows with no committed column, hot = 0 with and without the mask bit, every shift 0..511 (asserted); `stress` (2048 positions × 32 rows,
all committed, A = p−1 everywhere, `--chunk 2048`) makes coefficient 511 sum exactly 65536 terms of digit 0xFFFF in one
accumulator and checks every coefficient; `full` spot-checks 36 coefficients.

## Results (full 2^18 shape, median GPU timestamp of 5–10 runs; wall = batch of N submits / N)

| variant | mapping (cols x coefs / thread, threads) | chunk | GPU ms | wall ms | correctness |
|---|---|---|---|---|---|
| v1 | 1 x 1, global gather, 64 | – | 50.6 | 49.0 | PASS 36/36 |
| v2 | 8 x 1, staged 16-bit digits (32 KB), 512 | 128 | 34.8 | 35.8 | PASS 36/36 |
| v2b | 8 x 1, staged limbs (16 KB), 512 | 64 | 30.7 | 31.4 | PASS 36/36 |
| v3 | 4 x 1, 512 | 64 | 28.6 | 29.2 | PASS 36/36 |
| v4 | 8 x 2, 256 | 256 | 27.8 | 27.8 | PASS 36/36 |
| v5 | 4 x 2, 256 | 32 | 18.6 | 19.2 | PASS 36/36 |
| v6 | 4 x 2, full-limb + high-half accumulators | 32 | 20.8 | 21.4 | PASS 36/36 |
| v7 | 4 x 2, wrapping sum + carry counters | 32 | 25.1 | 25.8 | PASS 36/36 |
| v8 | 4 x 4, 128 | 32 | 21.8 | 22.4 | PASS 36/36 |
| **v9 = commit_accumulate.wgsl** | **2 x 4, 128** | **64** | **17.05** | **17.4** | **PASS 68/68 + small exact 16384/16384 + stress exact 1024/1024** (`--spot 64`) |
| v10 | 2 x 2, 256 | 32 | 20.2 | 21.2 | PASS 36/36 |
| v11 | 2 x 4, full-limb + high-half | 32 | 20.1 | 22.0 | PASS 36/36 |
| v12 | 2 x 8, 64 (128 accumulator words) | 32 | 44.1 | 46.2 | PASS 36/36 (register spill) |
| v13 | 1 x 8, 64 | 32 | 23.2 | 24.6 | PASS 36/36 |
| v14 | 1 x 4, 128 | 32 | 20.9 | 21.8 | PASS 36/36 |

Chosen kernel, other costs at 2^18: prep (negate/extend A → A2, 32 MB) 0.05 ms · reduce 0.29 ms (chunk 64) ·
upload A 16 MB 3 ms · upload hot 16 MB 3 ms (`queue.writeBuffer` + `onSubmittedWorkDone`) · readback 2 MB 2 ms
(`mapAsync`) · pipeline compile 51–74 ms first time per source, ~1 ms when WebKit's shader cache hits.
CPU-WASM reference 730 ms → **~43x on the kernel, ~30x including transfers**.

Probe: the same kernel with the digit split removed (wrong results) runs in 12.7 ms → the threadgroup-memory
gather (7.4e9 x 16 B = 118 GB, ~7 TB/s) is the floor of this formulation; ALU adds ~4 ms on top.

## Algorithm

- Sign folded into the index: `A2[q][j] = A[q][j]` for j < 512, `p − A[q][j−512]` for j ≥ 512, so
  `rot(A[q], s)[i] = A2[q][(i − s) mod 1024]` with no per-term sign handling (prep.wgsl).
- No carries in the inner loop: each fp128 term is split into eight 16-bit digits and each digit is summed in its
  own u32 (`lo += x & 0xFFFF; hi += x >> 16` on vec4<u32>); exact for ≤ 65536 terms per accumulator, which a chunk of
  ≤ 2048 positions (≤ 65536 rows) guarantees. reduce.wgsl sums chunk partials with 64-bit digit totals, reassembles
  the 160-bit value, folds `2^128 ≡ C` twice, one conditional subtract → canonical.
- Workgroup = (chunk of CHUNK positions, group of COLS columns, block). Per position: barrier, stage `A2[q]`
  (1024 x vec4 = 16 KB) into `var<workgroup>`, barrier, then for r in 0..32 read the row's code bytes (uniform), and for
  each committed column add `S[(i − 16r − hot) & 1023]` into that column's accumulators for the thread's CPT
  coefficients `i + t·512/CPT`. Fewer columns and more coefficients per thread amortize the code decode and index math
  and keep accumulators ≤ 64 words (128 words spill: v12).

## Data layouts (what W1 integration must produce / consume)

- `A` (binding: storage): `positions x 512 x vec4<u32>` canonical fp128, little-endian limbs — `A[q*512 + i]`.
  The prep pass writes `A2` (2x size) from it; A2 never leaves the GPU.
- `hot` (storage, `array<u32>` viewed as bytes): row-major, `column_capacity` bytes per row (64 at this shape →
  16 MB), byte = `hot[row][c]` (0..15) if committed (`hot != 0 || mask bit c`), else `0xFF`; padding columns 0xFF.
  This replaces the separate `hot[row][c]` u8 + `mask[row]` u64 arrays: the CPU folds the mask in while packing.
- `Params` uniform (5 x u32, 32-byte buffer): positions_per_block, num_chunks, blocks_per_column, column_capacity,
  part_mode (0). The chunk size is the main kernel's `CHUNK` override constant; `num_chunks = positions / CHUNK`,
  `CHUNK ≤ 2048` (digit accumulators: 2048 positions × 32 rows = 65536 terms; `commit_proto.py` asserts this).
- `PART` scratch: `num_chunks x column_capacity x blocks x 512 x 32 B` (128 MB at chunk 64, 64 MB at chunk 128
  (+0.7 ms)). Index `((chunk*colcap + c)*blocks + b)*512 + i`, two vec4: digits {0,2,4,6} then {1,3,5,7}.
- `RES`: `(c*blocks + b)*512 + i` → vec4<u32> canonical fp128, i.e. rings ordered column-major then block; 2 MB.
- Dispatch: main `(num_chunks, column_capacity/2, blocks)` with 128 threads; prep `positions*1024/256`;
  reduce `column_capacity*blocks*512/64`.
- Device limits to request: `maxComputeWorkgroupStorageSize` (WebKit default 16 KB is exactly enough for 16 KB
  staging but 512-thread variants need `maxComputeInvocationsPerWorkgroup`/`maxComputeWorkgroupSizeX` raised, else a
  message-less `GPUPipelineError`), `maxStorageBufferBindingSize`/`maxBufferSize` for buffers > 128 MB.

## WebKit notes

- `about:blank` is not a secure context; the harness serves the page via `page.route` on `http://localhost:7777`.
- `timestamp-query` works (`timestampWrites` on compute passes, `resolveQuerySet`); `performance.now()` is 1 ms coarse.
- No uniformity or loop-limit diagnostics were hit; `override` constants work with `layout: 'auto'`.
- No shader compile warnings; 128-line unrolled kernels compile in ~55 ms.

## Extrapolation

Work scales with rows: 2^20 ≈ 4x → ~70 ms GPU (vs ~2.9 s CPU-WASM), 2^21 ≈ 8x → ~140 ms, plus 4x/8x hot upload
(64/128 MB → ~12/25 ms) and result readback if blocks_per_column grows (8/16 MB). PART scratch scales with
blocks x num_chunks: use chunk 256+ or run block ranges sequentially at those sizes. Exact catalog parameters
(positions_per_block, K, D) may differ at 2^20/2^21; if positions_per_block exceeds 2048 the chunk must stay ≤ 2048
positions (digit accumulators), which the reduce already handles for any term count < 2^32.

## Open doors

- Threadgroup gather is the floor (12.7 ms); the only big lever left is reuse of `S` loads across contributions
  (e.g. shift-bucketing per position needs ≥ 16 columns per thread → register pressure) — not attempted.
- Padded `S` (1408 entries) would drop the `& 1023` on 3 of 4 coefficient indices (~4 % ALU).
- Chunk 32 is 0.15 ms faster on the main pass but doubles PART (256 MB); reduce cost grows accordingly.
