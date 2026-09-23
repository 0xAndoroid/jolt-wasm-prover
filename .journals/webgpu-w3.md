# webgpu-w3 — GPU in product + deployment readiness + W3 perf (orchestrator journal)

Ledger: `.audit/webgpu-w3.tsv` (one row per decision; this file = current wave only). Prior waves: `.journals/archive/`, `.journals/webgpu-akita.md`, `.audit/webgpu-akita.tsv`.
Orchestrator task c45196f3 · kanban #562 · playbook: vault `reference/feature-orchestrator-playbook.md`.

## Phases (user, Sep 23 2026)
1. GPU in the product: mode selector (GPU (GPU + CPU) / CPU only, default by navigator.gpu + self-test, localStorage), self-test once per session, mailbox wait timeout + batched W2 RUNs, per-proof mode + timing in UI.
2. Deployment readiness: full wasm + frontend build, Playwright headless WebKit pass (both modes, 2^18 + 2^20, byte-identical proofs), 375/1440 screenshots, DEPLOY.md redeploy checklist. No production deploy (report the command).
3. W3 perf: (1) stage-2 sumcheck on the W2 seam, (2) digit-range remainder (b=16/32 + CPU tail), (3) commit polish. Kill rule: idle E2E ≥5 % at 2^18 or ≥8 % at 2^20, byte-identical, verify=true.

## Playbook steps — Phase 1
1. plan 1/1 — fable-medium: spec touches Rust mailbox + worker.js/gpu-proxy.js + React UI → planner writes `## Plan (phase 1)` below. [done]
   - implement ci 1/1 (fable-low, parallel): repo has NO CI (`gh pr view 10` → 0 checks); add `.github/workflows/ci.yml` (fmt --check, frontend typecheck+build) so "green CI" is a real gate. [done — PR #12 → 63a7f41]
2. implement 1/1 — fable-high, worktree `webgpu-w3/gpu-in-product`. [running]
3. review i/3 — fresh fable-medium each round. [pending]
4. merge 1/1 — shell: CI green, squash, ff main, delete branch, wt remove. [pending]
5. deploy 1/1 — skip: phase 2 owns the build + preview; production deploy is the user's. [pending]

## Playbook steps — Phase 2
(filled when phase 1 merges)

## Playbook steps — Phase 3
(filled when phase 2 merges)

## Plan (phase 1)
Numbers measured from source (main @ 7a26b84):
- Preflight per prove today (`engine::prove:157` → `gpu::preflight` → `selftest::run`): 200 NOP + CREATE_BUFFER(256 MiB) + UPLOAD(16 MiB) + RUN(2^20 muladd: 32 MiB inline upload + 16 MiB readback) + DESTROY = **204 mailbox trips + 64 MiB of copies per prove**, plus a 2^20 CPU `AkitaField` check.
- W2 trips per instance (`digit_range.rs`; g = min(ring_bits, log2 n − 12) = 6 at 2^18, 3 instances/prove): warm (handles cached) = UPLOAD digits 1 + RUN_SEQ×g + DOWNLOAD 1 = **8**; cold first instance = CREATE_BUFFER×7 + UPLOAD lut0 + UPLOAD digits + g + DOWNLOAD = **16**; grow (bigger instance after a smaller one) = +DESTROY×7 = **23**. Warm prove = 3×8 = 24 trips at ~1.5–2 ms each (W2 journal).
- Mailbox: `Atomics::wait` with no timeout (`mailbox.rs:150`); words DOORBELL 0 / STATUS 1 / OP 2 / ARGS 4..36 / REGIONS 36..60 / RET 60..68 / ERROR_LEN 68 / ERROR 69.. mirrored in `gpu-proxy.js` `W`. MAX_ARGS 32 (RUN_SEQ round 2 uses 28), MAX_REGIONS 8 (all 8 used by RUN_SEQ).

### W2 round-trip table (per instance, g GPU rounds)
| path | before | after | how |
|---|---|---|---|
| warm | g+2 (8) | g+1 (7) | last round's RUN_SEQ carries the table readback (post-copy spec below) |
| cold | g+10 (16) | g+3 (9) | one `OP_ALLOC` (destroy list + create list, ≤7 new handles in RET) + one multi-target UPLOAD (lut0 + digits) |
| grow | g+17 (23) | g+3 (9) | the same `OP_ALLOC` destroys the old 7 handles |
Inherently sequential: the g RUN_SEQ rounds — round k's params carry challenge r_{k−1}, which the transcript derives from round k−1's 80 B message (readback). Floor = g trips; folding the digits UPLOAD into round 0 (pre-upload spec) reaches it, skip unless free (1 trip ≈ 1.5 ms).

### File-by-file
- `src/gpu/mailbox.rs`: `call()` loops `Atomics::wait_with_timeout(view, STATUS, BUSY, remaining)` against a deadline (`static TIMEOUT_MS: AtomicU32`, default 30 000 — W1 accumulate at 2^21 is the longest op). On timeout: set `static DEAD: AtomicBool`, return `Err(GpuError("gpu proxy unresponsive after N ms (op X)"))`; the callers `Box::leak` their upload/readback Vecs on that branch so a late proxy write lands in still-owned memory (the hazard the no-timeout comment guards). Every later `call` short-circuits `Err("gpu proxy dead")`; add `pub fn is_dead()`. RUN_SEQ post-copy spec in fixed `args[28..32] = [copy_src_region, copy_dst_region, dst_byte_offset, byte_len]` (0 len = none); `W` layout unchanged.
- `src/gpu/mod.rs`: real `set_disabled()` (ENABLED→DISABLED, proxy stays alive); `preflight()` split into `selftest()` (once per session, stores `static LAST: Mutex<GpuReport>`) and `status_report()` (STATE + cached selftest numbers, no mailbox) for `engine::prove`; device `install_once` stays on the selftest-ok path.
- `src/engine.rs:157`: `gpu::preflight()` → `gpu::status_report()`; `mailbox::is_dead()` → status `error: gpu proxy dead`, devices decline via `is_enabled()`.
- `src/lib.rs` exports added: `gpu_selftest() -> String` (JSON `{status, selftest_ms, mismatches, roundtrip_us}`), `set_gpu_disabled()`, `set_gpu_op_timeout_ms(u32)`, `gpu_is_dead() -> bool`. Unchanged: `set_gpu_enabled/unavailable/commit_enabled/digit_range_enabled`, `gpu_mailbox_ptr`, ProveResult getters (bench keeps `gpuSelftestMs`/`gpuRoundtripUs`, now from the cache).
- `src/gpu/digit_range.rs`: `handles_for` → one `OP_ALLOC`; `open_session` → one multi UPLOAD; `run_round(last)`: `out` = 80 B + table bytes, post-copy `last_dst → R_OUT @ 80`, table stashed in `Session.table`; `download_table` only if the stash is `None`.
- `src/gpu/trace_commit.rs`: untouched (optionally `buffers_for` → `OP_ALLOC`, cold only).
- `frontend/public/gpu-proxy.js`: `OP.ALLOC = 8` (`args=[n_destroy, handles…, n_create, sizes…]`, RET = new handles), UPLOAD multi-target (`args=[n,(handle,off,region)…]`, legacy `[handle,off]` when n absent → keep old shape by making `args[0]=0xFFFF_FFFF` the multi marker or bump to `OP.UPLOAD_MULTI = 9` — prefer the new op id), RUN_SEQ post-copy via `enc.copyBufferToBuffer` before submit.
- `frontend/public/worker.js`: `initGpu()` also runs `gpu_selftest()` and folds it into `init-done.gpu` `{status:'ok'|'unavailable'|'disabled'|'error: …', reason?, adapter?, selftestMs, roundtripUs}`; `set_gpu_op_timeout_ms(data.gpuTimeoutMs ?? 30000)`. New `{type:'set-gpu', data:{enabled}}` → enabled: proxy live ? `set_gpu_enabled()` : `await initGpu()`; disabled: `set_gpu_disabled()`; replies `{type:'gpu-status', gpu}`. After a prove/verify error with `gpu_is_dead()`: `gpuProxy.terminate(); gpuProxy = null; set_gpu_unavailable()`; the `error` message carries `gpu:{status:'unavailable', reason:'GPU proxy unresponsive — switched to CPU'}`. `init.data.gpu` (bool | 'w1' | 'w2') and `prove-done` fields unchanged.
- `frontend/src/lib/types.ts`: `GpuInfo`, `ProveMode = 'gpu' | 'cpu'`, `init.data.gpu?: boolean`, `set-gpu` request, `gpu-status` response, `init-done.gpu`, `error.gpu?`, prove-done `gpuStatus/gpuCommit/gpuDigitRange` typed; `ProgramState.lastProof: {mode, proveMs, totalMs, commitMs?, digitRangeMs?} | null`.
- `frontend/src/lib/constants.ts`: `MODE_STORAGE_KEY = 'jolt-prover-mode'` (`'gpu' | 'cpu'`, written only on user click, never on auto-fallback).
- `frontend/src/hooks/use-prover.ts`: state `gpu: GpuInfo | null`, `mode: ProveMode`, `modeReason: string | null`; init sends `gpu: stored !== 'cpu' && 'gpu' in navigator`; `init-done`/`gpu-status`: `status === 'ok'` → gpu, else cpu + reason; `setMode(m)` → localStorage + `set-gpu`; `prove-done` → `lastProof` (mode = `gpuStatus === 'ok' ? 'gpu' : 'cpu'`, ms from the JSON `total_ms`) + log line `mode GPU · prove 2.78s · commit 310 ms · digit range 447 ms`.
- `frontend/src/components/mode-selector.tsx` (new): `role="radiogroup"`, two `Button size="sm"` segments (`variant` outline when selected, ghost otherwise) "GPU (GPU + CPU)" / "CPU only", `rounded-md`, neutral — mode is a fact, not a verdict, so no accent (DESIGN: colour only for meaning); disabled while `status` is `proving`/`loading`; GPU segment disabled + `text-xs text-muted-foreground` reason line when unavailable. Mounted in `app.tsx` header before `StatusBadge`.
- `frontend/src/components/program-panel.tsx`: proof row gets `Badge variant="outline"` "GPU"/"CPU" + `text-xs text-muted-foreground` timings from `lastProof`.

### Switching model
- No worker re-init, no reload. GPU→CPU: `set-gpu {enabled:false}` → `set_gpu_disabled()`; proxy and device buffers stay. CPU→GPU with the proxy live: `set_gpu_enabled()` only (selftest cached). CPU→GPU after init-without-gpu: worker.js `initGpu()` = spawn `gpu-proxy.js`, `init` handshake, `gpu_selftest()`; on failure `set_gpu_unavailable()`, UI shows the reason, choice not persisted. Selector disabled mid-prove (the worker thread is blocked in the sync prove; a queued `set-gpu` applies afterwards).

### Risks
- Mailbox layout mirrored by hand in `gpu-proxy.js`: `W` stays unchanged, only new op ids + fixed arg slots; assert `args[28..32]` are zero in every existing RUN_SEQ builder (round 2 uses 0..27).
- Timeout leak: bounded per prove (eq tables + 80 B + table, or one RES ≤ 1 MiB); the GPU path is dead for the session afterwards, CPU continues.
- A hung prove cannot be interrupted by `set-gpu` (rayon threads blocked) — only the timeout ends it.
- wasm rebuild: `./setup-wasm-deps.sh` then `RUSTC_BOOTSTRAP=1 CARGO_UNSTABLE_BUILD_STD="panic_abort,std" wasm-pack build --release --target web -- --features webgpu,trace-commit-device,digit-range-device`; `cd frontend && npm run build`; restart `node server.mjs` (in-memory cache).
- bench compatibility: `init.gpu` shapes and `gpuStatus/gpuSelftestMs/gpuRoundtripUs` must keep `bench_webgpu.py --gpu all --parity 6` green.

### Verification
- `uv run --with playwright python bench/bench_webgpu.py --iters 69 --runs 2 --gpu both --browser webkit`: verify=true both modes, `proofBytesIdentical`, `gpuStatus ok` (on) / `disabled` (off); `--gpu all --parity 6 --runs 1` covers the batched RUN_SEQ (digit-range `ops` per instance = g+1 warm).
- New `bench/test_ui_modes.py` (Playwright headless WebKit, React UI at :8080): (1) default lands on GPU when `navigator.gpu` + selftest ok, badge/log say GPU; (2) click "CPU only" → prove → badge CPU, proof sha256 equals the GPU proof; (3) reload → mode restored from `localStorage['jolt-prover-mode']`; (4) back to GPU without reload → prove ok; (5) plain `chromium` (no adapter) → CPU with reason shown, GPU segment disabled.
- Dead-proxy test: init with `gpuTimeoutMs: 2000`; the page sends the worker a test-only `{type:'kill-gpu-proxy'}` (worker.js: `gpuProxy.terminate()`, ≤5 lines) right before `prove`; expect the prove error to contain `gpu proxy unresponsive` within ~3 s, then `gpu-status unavailable`, then a CPU prove succeeds in the same session.
- Screenshots 375/1440 of header + proof row (phase 2 reuses them); `cargo clippy --all --all-targets -q`, `cargo fmt`, `npm run build` typecheck.

## Implementation notes (phase 1)
- Branch `webgpu-w3/gpu-in-product`; ledger `.audit/webgpu-w3.tsv`.
- Mailbox: `Atomics.wait_with_timeout` against a per-op deadline (`set_gpu_op_timeout_ms`, default 30 s); timeout sets `DEAD`, every later `call` returns `gpu proxy dead`. Callers `mem::forget` their readback Vecs on the dead branch (selftest `out`, trace-commit RES, digit-range message+table buffer) — the proxy may still write them.
- W2 trips: last RUN_SEQ carries `[src_region, R_OUT, 80, table_len]` in `args[28..32]` → proxy `copyBufferToBuffer` before submit; `OP_ALLOC` (destroy list + create list, RET = handles) and `OP_UPLOAD_MULTI` (digits + lut0) landed within the cap. Measured ops/instance at 2^18 (g = 6): warm 7, cold 8 (alloc + upload + 6 rounds); grow would be 8 too. `download 0.0 ms` in every run.
- Selftest once per session: `gpu::selftest()` caches in `LAST`, `status_report()` never enters the mailbox; `engine::prove` reads the cache. worker.js `initGpu()` runs it after the proxy handshake and turns a failed selftest into `unavailable` + reason.
- Switching: `set-gpu {enabled}` → `set_gpu_disabled()` / `initGpu()` (re-uses the live proxy, cached selftest); reply `gpu-status`. UI: `ModeSelector` (radiogroup, two neutral segments, reason line when unavailable), `lastProof` badge + timings in the proof row, `mode …` and `Proof SHA-256` log lines.
- **Deviation:** the dead proxy is orphaned, not terminated. Headless WebKit crashes the whole page when a worker that owns a `GPUDevice` is terminated (reproduced with a kill-only variant; Chromium is fine). The test hook is therefore `hang-gpu-proxy` (proxy stops serving) instead of `kill-gpu-proxy`. The state machine is unchanged: `gpu_is_dead()` → `set_gpu_unavailable()`, error carries `gpu.status unavailable`, GPU segment disabled for the session.
- Deviation: `init.gpu` is `stored !== 'cpu'` (no `'gpu' in navigator` check in the page) — the worker reports the reason (`navigator.gpu missing…` / `requestAdapter returned null`) so the UI can show it.
- Test: `bench/test_ui_modes.py` uses a fresh page in the same context instead of `page.reload()` (second 4 GB wasm memory in one page crashes WebKit).
- Numbers (2^18, WebKit, 8 threads): prove on 2.31 s / off 3.71 s warm; both verify, proof bytes identical; parity 6 rounds ok; dead-proxy time-to-error 2.1 s with a 2 s deadline, then a CPU prove verified in the same session.
