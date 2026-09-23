# WebGPU Akita — W1: one-hot trace commit on the GPU

Decision ledger: `.audit/webgpu-akita-w1.tsv` (local, untracked).

## What landed (PR #8, branch `wgpu-akita/w1-commit-gpu`, stacked on #5 + #7)

- `frontend/public/wgsl/commit/{common,prep,commit_accumulate,reduce}.wgsl` — the measured kernels moved out of `bench/proto/`; proto scripts read them from there. `gpu-proxy.js`: `CHUNK` pipeline override from RUN arg 22, raised compute limits.
- `src/gpu/trace_commit.rs` — `WebGpuTraceCommit: jolt_akita::TraceCommitDevice`. Installed once after the preflight self-test passes; declines when the GPU is off or the shape is not K16/D512/n_a1/digits1/colcap64. Codes packed on the CPU (rayon, ≤12 ms at 2^21, so the kernel keeps its 0xFF byte layout); A/A2/codes/PART persistent per (P, blocks); RES read back per call; CHUNK = smallest of 64..2048 dividing P with PART ≤ 256 MiB (2^16/18 → 64, 2^20 → 128, 2^21 → 256).
- Stage breakdown → `ProveResult.gpu_commit` JSON + `[gpu] trace commit …` console line + span `trace_onehot_commit_gpu`; `bench_webgpu.py` takes `--iters` lists and prints the off/on/ratio table.

## Numbers (this Mac, headless WebKit, 8 threads, warm median of 3 after 1 warm-up; other agents' builds were running, load 5–10)

| size | prove off (s) | prove on (s) | on/off | GPU commit (ms): pack + upload + gpu + readback + convert = total (convert = a canonical-limb check the seam already does; removed in review) |
|---|---|---|---|---|
| 2^16 | 1.205 | 0.972 | 0.81 | 0.5 + 2.4 + 12.3 + 0.7 + 0.7 = 16.4 |
| 2^18 | 2.814 | 1.901 | 0.68 | 1.5 + 6.2 + 24.3 + 1.6 + 1.1 = 35.1 |
| 2^20 | 7.783 | 4.264 | 0.55 | 6.0 + 12.8 + 59.0 + 2.2 + 2.0 = 82.0 |
| 2^21 | 14.620 | 7.928 | 0.54 | 11.8 + 32.6 + 107.5 + 2.5 + 2.0 = 156.6 |

- Cold (first prove incl. wasm compile): off 7.9 / 2.8 / 8.1 / 14.8 s, on 7.6 / 1.9 / 5.6 / 7.7 s.
- CPU stage-0 commit in wasm at 2^18 (span `TracePackedOneHot::commit_inner`): 757 ms → device path 44.5 ms (17×). The 97 ms figure in `docs/trace-commit-device.md` is native.
- Proof sha256 identical on/off at all four sizes (`4359b6b8…` at 2^16 matches the W0 journal), verify=true, `gpu_status ok`, 0 self-test mismatches.
- Standalone harness: `commit_proto.py --shape small --variant v9 --chunk 64`, `--shape stress --chunk 2048`, `--shape full` all PASS from the new kernel path.

## Open doors

- `on` prove times are stable to ±1%; `off` varies with host load (2^18 measured 2.65–4.0 s across runs). Re-measure on an idle host for the record.
- wasm-target clippy (`--target wasm32-unknown-unknown --features webgpu,trace-commit-device -D warnings`) passes except the pre-existing `declare_interior_mutable_const` on `mailbox.rs:49` (from #5).
- Uploads are ~20% of the GPU commit at 2^21 (A 64 MiB + codes 128 MiB); reading hot+masks directly in the kernel would drop the codes upload to 1/64.
