# webgpu-akita W0 — harness

Ledger: `.audit/webgpu-akita-w0.tsv` (untracked, one row per unit/verdict).

## What landed
- `webgpu` cargo feature (default off): `src/gpu/` — `#[repr(C)]` mailbox static in shared wasm memory, `gpu::call` blocks the calling prover thread with `js_sys::Atomics::wait` (stable API; the `memory_atomic_wait32` intrinsic is still nightly-gated on 1.95), single-owner `Mutex`, runtime toggle `set_gpu_enabled` / `set_gpu_unavailable`, `gpu::preflight()` = 200 NOP round trips + 2^20 `a·b+c` self-test vs `AkitaField`.
- `frontend/public/gpu-proxy.js` — dedicated Worker owning the `GPUDevice`; `Atomics.waitAsync` doorbell loop; ops NOP / CREATE_BUFFER / UPLOAD / DESTROY / RUN; error scopes → mailbox error string. The self-test is `RUN` of `fp128_ops` with op=3 rather than a dedicated op.
- `frontend/public/wgsl/fp128.wgsl` — 4×u32 canonical limbs, 16-bit-half `mul_wide`, schoolbook 4×4, reduction as wrapping `+C` folds (2^128 ≡ C); `fp128_ops.wgsl` elementwise mul/add/sub/muladd kernel.
- `worker.js`: `init.data.gpu === true` spawns the proxy, `init-done.gpu` report, `prove-done` carries `gpuStatus/gpuSelftestMs/gpuSelftestMismatches/gpuRoundtripUs`. React UI untouched.
- `bench/bench_webgpu.py` (gpu on/off oracle bench, proof sha256 equality), `bench/test_fp128_wgsl.py` (WGSL vs Python ints).

## Numbers
(filled at the end of W0 — see PR description)

## Open doors
- Mailbox latency vs. `memory_atomic_wait32`: only worth revisiting if W1 issues many small ops (design is one big op at a time).
- fp128 mul cost is unoptimized (schoolbook + 16-bit halves); W1 decides the internal representation.
- Chromium WebGPU only under `--enable-unsafe-webgpu --use-angle=metal` (headless); WebKit is the reference engine.
