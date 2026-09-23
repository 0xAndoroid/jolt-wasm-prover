# webgpu-akita · W1 kernel prototype (standalone WGSL, WebKit)

Branch `wgpu-akita/w1-kernel-proto`, files `bench/proto/`. Decision ledger: `.audit/webgpu-akita-w1-proto.tsv` (untracked).

## Outcome
- Chosen kernel `bench/proto/commit_accumulate.wgsl` (= v9: 2 columns x 4 coefficients per thread, 128 threads,
  chunk 64): **17.05 ms GPU / 17.4 ms wall** at the 2^18 shape on M5 Max headless WebKit, vs 730 ms CPU-WASM.
- Correct: small shape exact 16384/16384 coefficients; full shape 68/68 spot sums + padding columns zero; every
  variant PASS. Reference = Python big-int sums of rot(A[q], s) per the CPU `shift_accumulate` semantics.
- Transfers: A 16 MB 3 ms, hot 16 MB 3 ms, result 2 MB readback 2 ms; first pipeline compile ~55 ms.

## What was tried (numbers in bench/proto/README.md)
- V1 global gather 50.6 ms → staged workgroup A2 (V2/V2b) 31–35 ms → fewer columns per workgroup and more coefficients
  per thread (V3–V5, V9) 17 ms. 128 accumulator words (V12) spill (44 ms). Alternate accumulation schemes (full-limb +
  high-half, wrapping sum + carry counters) are slower than the plain 16-bit digit split.
- Probe without the digit split: 12.7 ms → threadgroup gather bandwidth is the floor of this formulation.
- Shift-bucketing (V3 in the brief) skipped: it shares only the load/index, not the adds, and the loads are the floor
  only when the per-thread register budget allows many columns.

## Pitfalls found
- WebKit `requestDevice` defaults: 256 invocations / 16 KB workgroup storage; exceeding them gives a message-less
  `GPUPipelineError` — request adapter limits explicitly.
- `about:blank` has no `navigator.gpu`; serve over http://localhost via `page.route`.
- `performance.now()` is 1 ms coarse; batch ≥ 5 dispatches for wall time; timestamp-query is available and used.

## Open doors
- Load reuse across contributions (needs ≥ 16 columns of accumulators per thread), padded `S` to drop the index masks,
  chunk/partial-buffer tradeoff at 2^20+ (128 MB PART at chunk 64; run block ranges sequentially or chunk 256).
- Integration must pack hot+mask into the byte-code layout (0xFF = uncommitted) and pass CHUNK as an override.

## Review 1 (PR #6)

Variant files v2b–v14 dropped (generator builds them in memory); Params trimmed to the fields kernels read;
`stress` shape added for the 65536-term digit bound; GPU errors now fail the run. Numbers re-measured within 1 %.
