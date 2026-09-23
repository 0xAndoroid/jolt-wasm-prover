# W2 prototype — digit-range sumcheck rounds in WGSL (standalone)

Stage-8 `digit_range_prove`, four b=8 direct-leaf instances (rounds 24/21/20/19, trace 2^18),
run as dependent WebGPU rounds under headless WebKit (Apple GPU, MacBook). CPU-WASM baseline
for the same four instances: **≈500 ms**. Kill rule for W2: save ≥420 ms.

## Verdict: GO (marginal) — saves ≈425–440 ms, margin +5…+20 ms over the 420 ms bar

| | ms |
|---|---|
| GPU busy, 4 instances (84 rounds) | 30.1 |
| dependent gaps (84 × ~0.25 median) | ~21 |
| **4-instance wall, measured (hash host)** | **57** (55–63 across 5 quiet runs; one 87 ms outlier) |
| uploads: packed 3-bit 6 MiB / i8 16 MiB / eq tables | 1 / 4 / <1 |
| mailbox extrapolation (+2 × 100 µs per round × 84) | +17 |
| **projected in-prover wall** | **≈75 (packed) … ≈80 (i8)** |
| saving vs 500 ms CPU | ≈420–425 |

Margin is 1–5 % of the bar: compute is not the risk (30 ms), the 84 dependent round trips are.
Lever if the transcript allows it: run the four instances in lockstep (one submit + one 320 B
readback per round index) → 24 trips instead of 84, ≈ −27 ms, saving ≈450 ms. Sequential
transcripts (instance i+1 after instance i) keep 84 trips.

## Per instance (hash host, packed source, quiet box: load1 7.4, no rustc)

| rounds | table bytes (i8 / packed) | upload ms (i8 / packed / eq) | compact r0–2 | materialize r3 | field r≥4 | GPU busy | gap median / max | wall |
|---|---|---|---|---|---|---|---|---|
| 24 | 16 MiB / 6 MiB | 4 / 1 / 0 | 12.77 | 1.80 | 3.88 | 18.45 | 0.245 / 0.893 | 29 |
| 21 | 2 MiB / 0.75 MiB | 0 / 0 / 0 | 2.37 | 0.35 | 1.92 | 4.65 | 0.226 / 0.358 | 10 |
| 20 | 1 MiB / 0.38 MiB | 0 / 0 / 0 | 1.61 | 0.24 | 2.42 | 4.27 | 0.234 / 0.491 | 10 |
| 19 | 0.5 MiB / 0.19 MiB | 0 / 0 / 0 | 1.01 | 0.16 | 1.60 | 2.78 | 0.241 / 0.700 | 8 |
| **Σ** | | 4 / 1 / <1 | **17.76** | **2.55** | **9.83** | **30.14** | | **57** |

R=24 per-pass GPU ms: round0 5.41 · lut+round1 3.48 · lut+field(r2) 3.65 · r3 1.76 · r4 0.96 ·
r5 0.61 · r6 0.34 · r7 0.22 · r8 0.15 · r≥9 ≈0.10 (reduce 0.04 each, launch-bound tail).
Host time per round (challenge hash + bookkeeping): ≤1 ms per instance total.

## Micro numbers

- fp128 mul: **14.2 Gmul/s** (2^22 threads × 16 dependent muls, median of 5, spot-checked in Python).
  Field rounds reach ~20 Gmul/s equivalent (independent muls per pair), so the microbench is a floor.
- Dependent round-trip floor: **0.226 ms wall / 0.217 ms GPU gap** per iteration
  (empty dispatch + 80 B `mapAsync`, 84 sequential). Real rounds add ~0.02–0.05 ms.
- Timestamp resolution 0.209 µs; per-pass timestamps in every dispatch.
- Adapter limits (WebKit/Apple): maxStorageBufferBindingSize 2 147 483 644, maxBufferSize
  2 147 483 644, workgroup storage 32 KiB, 1024 invocations/wg, 44 storage buffers/stage.

## Correctness gate

`--shape small` (rounds 12) compares against an exact Python reference: every round message,
every folded table (rounds 3–11), the final 2-element table, and the final value against the
MLE of the range image at r. PASS for `--host fixed --source packed` and `--host hash --source i8`.
Full set checks challenge derivation, sumcheck consistency P_k(0)+P_k(1)=P_{k−1}(r_{k−1}) with
claim 0, and the final claim eq(τ,r)·Q(final). PASS on all three full runs.

## Deviations from akita (integration must close these)

1. **Challenge derivation**: 4-lane murmur fmix over the 20 message words (or a fixed list) instead
   of the Blake2b transcript. Same dependent structure; the prover keeps its transcript on the host.
2. **Message format**: 5 Q-coefficients (80 B) readback; akita builds `EqFactoredUniPoly` from the
   same q_coeffs — no extra work, but the byte layout is ours.
3. **Compact rounds are a different algorithm** (class histograms + LUTs, GPU-built LUT1/LUT2f):
   identical messages, more muls in round 2 (≈2.1n vs akita's 1.4n).
4. **Gruen split is fixed** (`split = 1 + (R−1)//2`), eq tables uploaded once per instance rather
   than re-derived per round; identical values.
5. **Final fold on the host** (2-element table read with the last round); no `live_len` zero-tail
   handling, no sparse x/y phase, no Blake2b grind.
6. **Round-3 materialization is fused** with its evaluation (akita materializes then evaluates).

## Open risks

- Margin +5…+20 ms is inside run-to-run noise (one 87 ms outlier at load1 ≈ 7.4). Re-measure inside
  the real mailbox before committing W2 effort; the decision flips at ~0.45 ms per round trip.
- WebKit dependent gap 0.22–0.25 ms is the floor of this box; other GPUs/browsers may be 2–4× worse.
- Memory: 2^20 trace (rounds 26/23/22/21) needs i8 64 MiB + tables 192 MiB per instance; 2^21
  doubles that. Well inside the 2 GiB binding limit, but `maxWg = 8192` partials must grow to
  16384 at 2^26 (field rounds with ppt 8).
- Round-1 workgroup atomic histogram assumes `inner ≥ 2048` groups or falls back to smaller blocks;
  exact for ≤65536 terms per class (2^16 digits per workgroup).

## Run

```
uv run --with playwright --with numpy python bench/proto-w2/digit_range_proto.py --shape small --host fixed --source packed
uv run --with playwright --with numpy python bench/proto-w2/digit_range_proto.py --set full --host hash --source packed --floor-iters 84 --mulbench --out out.json
```
Files: `fp128.wgsl` (from w0-harness), `dr_common.wgsl`, `round0.wgsl`, `lut.wgsl`, `round1.wgsl`,
`field.wgsl`, `reduce.wgsl`, `nop.wgsl`, `mulbench.wgsl`, `harness.html`, `digit_range_proto.py`.
