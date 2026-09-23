# Cloudflare Pages Deployment

Domain: `jolt.rs` (already on Cloudflare). Project `jolt-wasm-prover`, output directory
`frontend/dist/` (`wrangler.toml`). No CI build: the wasm is compiled locally with
`build-std` against the patched deps from `./setup-wasm-deps.sh` (~5 min), then the
static output is uploaded.

## Redeploy checklist

1. Prerequisites: the toolchain from `rust-toolchain.toml` (stable 1.95 + `rust-src` +
   `wasm32-unknown-unknown`, installed by rustup on first `cargo` run), `wasm-pack`,
   Node.js ≥ 20 with npm, `wrangler` (`npm i -g wrangler`, `wrangler login`; the
   Playwright smoke needs `uv` and `uv run --with playwright python -m playwright install webkit`).
2. Sync: `git fetch origin && git merge --ff-only origin/main`; artifacts in
   `frontend/public/` (`*.bin`, `*.elf`) are committed — regenerate only when guests or
   the jolt pin changed: `cargo run --release --features native --bin generate-preprocessing`.
3. Patched wasm deps: `./setup-wasm-deps.sh` (rewrites the override block in `Cargo.toml`
   to point at `.wasm-deps/`; never commit that state, `./setup-wasm-deps.sh --revert` restores it).
4. Wasm, full browser feature set (GPU mailbox + trace-commit + digit-range devices):
   ```bash
   RUSTC_BOOTSTRAP=1 CARGO_UNSTABLE_BUILD_STD="panic_abort,std" wasm-pack build --release --target web -- --features webgpu,trace-commit-device,digit-range-device
   ```
   After a toolchain change run `cargo clean --target wasm32-unknown-unknown` first.
5. `./scripts/build-pages.sh` — checks that `pkg/` carries all three features, runs
   `npm ci && npm run build`, copies `pkg/` into `frontend/dist/pkg/`, checks the GPU files
   landed and that no file exceeds the 25 MiB Pages cap (`--wasm` runs steps 3–4 for you).
6. Local smoke (same headers as production, see below):
   ```bash
   node server.mjs &                                   # restart after every rebuild — it caches
   uv run --with playwright python bench/test_ui_modes.py --shots /tmp/shots
   uv run --with playwright python bench/bench_webgpu.py --iters 69 --runs 1 --gpu both
   ```
   Expect `test_ui_modes: all checks passed` (all three tabs prove + verify in GPU and CPU mode,
   proof SHA-256 equal across modes) and `proofBytesIdentical` / `valid=True` from the bench.
7. Optional preview: `npx wrangler pages deploy frontend/dist --project-name=jolt-wasm-prover --branch preview`
   → `https://preview.jolt-wasm-prover.pages.dev`. `_headers` applies there too, so the
   GPU path works on the preview; check it in a WebGPU browser before promoting.
8. Production: `npx wrangler pages deploy frontend/dist --project-name=jolt-wasm-prover`
9. Post-deploy: open https://jolt.rs in a WebGPU browser (Chrome, Safari 26+): the header
   selector shows **GPU** selected with no "GPU unavailable" line; Generate Proof on SHA-256
   → proof row badge `GPU`, Verify Proof → `Valid`; switch to **CPU only** and repeat.
   In a browser without WebGPU the selector lands on CPU and shows the reason.

## First-time setup

```bash
wrangler pages project create jolt-wasm-prover --production-branch main
wrangler pages project edit jolt-wasm-prover --domains jolt.rs
```

For a subdomain, add a CNAME in the Cloudflare DNS dashboard pointing to `jolt-wasm-prover.pages.dev`.

## What gets deployed

```
frontend/dist/
├── index.html
├── _headers, _redirects  ← COOP/COEP + CSP + cache rules; /pkg → /pkg/jolt_wasm_prover.js
├── assets/               ← Vite-bundled JS/CSS (hashed filenames)
├── pkg/                  ← jolt_wasm_prover.js, jolt_wasm_prover_bg.wasm (22.4 MiB, 25 MiB cap), snippets/
├── worker.js             ← prover Web Worker (rayon thread pool)
├── gpu-proxy.js          ← module Worker owning the GPUDevice
├── wgsl/                 ← fp128 library, commit/ and digit_range/ kernels
├── *.bin, *.elf          ← preprocessing artifacts + guest ELFs (~2.3 MB)
└── favicon.png, jolt_alpha.png, og-image.png
```

Total ~26 MB; the per-file cap is 25 MiB (`build-pages.sh` enforces it), 20k files.

## Headers

`_headers` (production) and `server.mjs` (local) set the same header set on every path,
so the local smoke exercises the production policy:

| Header | Value | Why |
|--------|-------|-----|
| Cross-Origin-Opener-Policy | same-origin | SharedArrayBuffer (wasm threads) |
| Cross-Origin-Embedder-Policy | require-corp | SharedArrayBuffer |
| Content-Security-Policy | `default-src 'self'`, `script-src 'self' 'wasm-unsafe-eval'` + CF insights, `worker-src 'self' blob:`, `connect-src 'self'` + CF insights, Google Fonts | wasm + both Workers (`worker.js`, `gpu-proxy.js`) + `wgsl/` fetches are same-origin |
| X-Content-Type-Options | nosniff | Prevent MIME sniffing |
| X-Frame-Options | DENY | Prevent iframe embedding |

`_headers` also marks `/pkg/*`, `/worker.js`, `/gpu-proxy.js` and `/wgsl/*` `Cache-Control: no-cache`:
the mailbox word layout is mirrored by hand between the wasm and `gpu-proxy.js`, so they must
never be served from different deploys.

## Notes

- Cloudflare handles brotli/gzip at the edge and caches static assets; Vite's hashed
  filenames bust the JS/CSS cache, the rules above cover the rest.
- The GPU path needs WebGPU in the browser; without it (or when the session self-test
  fails) the UI falls back to CPU-only and shows the reason. Proofs are byte-identical
  in both modes.
