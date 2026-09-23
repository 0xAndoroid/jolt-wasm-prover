# WebGPU Akita — W2: digit-range sumcheck rounds on the GPU

Decision ledger: `.audit/webgpu-akita-w2.tsv` (local, untracked).

## What landed (PR #10, branch `wgpu-akita/w2-digit-range`, stacked on #8)

- `patches/0007-akita-digit-range-device.patch` — `akita_prover::DigitRangeDevice`, a process-global seam consulted on the basis-8 direct leaf of the stage-1 digit-range sumcheck. The device gets the packed digits + eq point once per instance and answers one round message (5 q-coefficients, 80 bytes) per `round()` call from host-uploaded `GruenSplitEq` tables; `DeviceLeaf` wraps `LowBasisRangeCheckProver` so `prove_eq_factored_sumcheck`, the transcript and the verifier are untouched. The CPU tail resumes from the downloaded table via `LowBasisRangeCheckProver::from_materialized`. Parity shadow (`set_digit_range_parity_rounds`) runs the CPU prover on the same challenges for the first N rounds and aborts on the first mismatch.
- `src/gpu/digit_range.rs` — `WebGpuDigitRange`, installed with the trace-commit device after the preflight passes; declines below 2^16 digits or below 4 GPU rounds. Rounds 0–2 read the packed digits (histogram kernels, host-side LUT0 / device LUT1), round 3 materialises the folded field table, rounds ≥4 fuse fold+eval; g = min(ring_bits, log2 n − 12) rounds per instance. One RUN_SEQ mailbox op per round (params + eq tables inline, message inline readback), buffers grow-only across instances.
- `frontend/public/wgsl/digit_range/*` (from `bench/proto-w2`), `gpu-proxy.js` RUN_SEQ / DOWNLOAD ops, `worker.js` `gpu: 'w1' | 'w2'` knobs, `bench_webgpu.py --gpu all --parity N`.

## Numbers (this MacBook, idle host, headless WebKit, 8 threads, warm median of 5 in quiet windows)

| size | CPU only | W1 | W1+W2 | W1 − (W1+W2) | % of CPU | digit-range GPU total (ms) |
|---|---|---|---|---|---|---|
| 2^16 | 1.205 | 1.029 | 0.847 | 0.182 | 15.1 % | 15.8 (1 inst) |
| 2^18 | 2.778 | 2.013 | 1.566 | **0.447** | **16.1 %** | 40.9 (3 inst) |
| 2^20 | 7.735 | 4.463 | 3.551 | 0.912 | 11.8 % | 53.4 (3 inst) |
| 2^21 | 14.939 | 8.095 | 6.938 | 1.157 | 7.7 % | 35.6 (1 inst) |

- Kill rule (2^18, W1 on → W1+W2 on ≥ 0.425 s): **0.447 s → UNPARK (marginal)**; the worst 2^18 run pairing is 0.427 s, still above the bar. Sha equal across modes and verify=true on all 72 runs.
- Under host load (superseded; other agents' rustc builds and iOS simulators ran throughout, load 7–300) the same pairing read 0.079 / 0.256 / 0.405 / 1.677 s at 2^16 / 2^18 / 2^20 / 2^21 (2^18 CPU-only 2.820, W1 2.087, W1+W2 1.831) → PARK; the idle re-run reversed that verdict.
- Per instance at 2^18 (n digits / GPU rounds / upload + rounds + download ms, loaded run): 10 664 384 / 6 / 1.4 + 19 + 2.0; 1 933 824 / 6 / 0.3 + 12 + 0.5; 777 984 / 6 / 0.3 + 9.5 + 0.7. 8 mailbox ops per instance (6 rounds + LUT0 upload + table download). GPU busy is a minority of the round wall at the small instances — 6 dependent trips of ~1.5–2 ms each dominate.
- Proof sha256 identical across all four modes at every size, verify=true, `gpu_status ok`, 0 self-test mismatches. Parity shadow: 4 rounds at 2^16, 6 rounds × 3 instances at 2^18, no mismatch.

## Open doors

- Only the ring (y-phase) rounds run on the GPU: g = ring_bits = 6 at every shipped size, so the CPU tail still does the x-phase rounds on an n/64 table.
- Basis-16/32 instances and the b=8 instances below 2^16 digits stay on the CPU.
