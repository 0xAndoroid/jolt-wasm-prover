# webgpu-akita W0 — harness

Ledger: `.audit/webgpu-akita-w0.tsv` (untracked, one row per unit/verdict).

## What landed
- `webgpu` cargo feature (default off): `src/gpu/` — `#[repr(C)]` mailbox static in shared wasm memory, `gpu::call` blocks the calling prover thread with `js_sys::Atomics::wait` (stable API; the `memory_atomic_wait32` intrinsic is still nightly-gated on 1.95), single-owner `Mutex`, runtime toggle `set_gpu_enabled` / `set_gpu_unavailable`, `gpu::preflight()` = 200 NOP round trips + 2^20 `a·b+c` self-test vs `AkitaField`.
- `frontend/public/gpu-proxy.js` — dedicated Worker owning the `GPUDevice`; `Atomics.waitAsync` doorbell loop; ops NOP / CREATE_BUFFER / UPLOAD / DESTROY / RUN; error scopes → mailbox error string. The self-test is `RUN` of `fp128_ops` with op=3 rather than a dedicated op.
- `frontend/public/wgsl/fp128.wgsl` — 4×u32 canonical limbs, 16-bit-half `mul_wide`, schoolbook 4×4, reduction as wrapping `+C` folds (2^128 ≡ C); `fp128_ops.wgsl` elementwise mul/add/sub/muladd kernel.
- `worker.js`: `init.data.gpu === true` spawns the proxy, `init-done.gpu` report, `prove-done` carries `gpuStatus/gpuSelftestMs/gpuSelftestMismatches/gpuRoundtripUs`. React UI untouched.
- `bench/bench_webgpu.py` (gpu on/off oracle bench, proof sha256 equality), `bench/test_fp128_wgsl.py` (WGSL vs Python ints).

## Numbers (Mac mini, headless WebKit 26.6 / Playwright build 2359, 8 threads, iters 17 = 2^16 padded)
- V1 gpu on vs off: proof sha256 `4359b6b8…5312e70` identical, verify=true both; gpu_status ok / disabled; self-test 2^20 mul-adds 26–108 ms, 0 mismatches; NOP round trip 70–150 µs quiet machine (330–1320 µs with a build finishing).
- V2 fp128.wgsl: 101,000 vectors (1,000 edge incl. 0, 1, p−1, p, p+1, 2^128−1, C) × mul/add/sub/muladd = 0 mismatches vs Python ints. Fold-2 carry bug in `fp128_mul` caught by the edge cases (result landed one limb high).
- V3 no-feature build gpu=off warm prove 1.82 s; webgpu build gpu=off 1.73–1.94 s, gpu=on 1.74–1.94 s (same runs, noise ≈ 5–10 %). gpu=on overhead = preflight only.
- V4 Chromium headless without `--enable-unsafe-webgpu`: requestAdapter null → gpu_status unavailable, prove + verify ok, same sha. Chromium `--enable-unsafe-webgpu --use-angle=metal`: ok, self-test 88 ms, round trip 70 µs.
- W1 sizing: the self-test routes `a` through a 256 MiB persistent buffer (CREATE_BUFFER / UPLOAD / handle binding / DESTROY) and 16 MiB inline uploads + 16 MiB readback each prove.

## Open doors
- Mailbox latency vs. `memory_atomic_wait32`: only worth revisiting if W1 issues many small ops (design is one big op at a time).
- fp128 mul cost is unoptimized (schoolbook + 16-bit halves); W1 decides the internal representation.
- Chromium WebGPU only under `--enable-unsafe-webgpu --use-angle=metal` (headless); WebKit is the reference engine.
