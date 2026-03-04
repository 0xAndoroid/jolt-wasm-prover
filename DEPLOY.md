# Cloudflare Pages Deployment

Domain: `jolt.rs` (already on Cloudflare)

## Prerequisites

- WASM package built locally (`pkg/` directory — requires nightly Rust + wasm-pack + local arkworks-algebra)
- Preprocessing artifacts generated (`frontend/public/*.bin`, `*.elf` — already in git)
- Node.js, npm
- `wrangler` CLI: `npm i -g wrangler` and `wrangler login`

## First-time setup

Create the Pages project:

```bash
wrangler pages project create jolt-wasm-prover --production-branch main
```

Add custom domain (Cloudflare dashboard or CLI):

```bash
wrangler pages project edit jolt-wasm-prover --domains jolt.rs
```

If using a subdomain like `demo.jolt.rs`, add a CNAME record in the Cloudflare DNS dashboard pointing to `jolt-wasm-prover.pages.dev`.

## Build

```bash
./scripts/build-pages.sh
```

This:
1. Builds the Vite frontend into `frontend/dist/`
2. Copies `pkg/` (WASM + JS bindings + rayon worker helpers) into `frontend/dist/pkg/`
3. Includes `_headers` file (COOP/COEP for SharedArrayBuffer, CSP, security headers)

## Deploy

```bash
wrangler pages deploy frontend/dist --project-name=jolt-wasm-prover
```

## What gets deployed

```
frontend/dist/
├── index.html
├── _headers              ← COOP/COEP + security headers
├── assets/               ← Vite-bundled JS/CSS (hashed filenames)
├── pkg/                  ← WASM package (copied from pkg/)
│   ├── jolt_wasm_prover.js
│   ├── jolt_wasm_prover_bg.wasm  (21 MB)
│   └── snippets/                 (rayon worker helpers)
├── worker.js             ← Web Worker entry point
├── *.bin                 ← Preprocessing artifacts (~22 MB total)
├── *.elf                 ← Guest ELF binaries
├── favicon.png
└── jolt_alpha.png
```

Total deployment size: ~50 MB. Within Cloudflare Pages free tier limits (25 MB per file, 20k files).

## Headers

The `_headers` file sets on all paths (`/*`):

| Header | Value | Why |
|--------|-------|-----|
| Cross-Origin-Opener-Policy | same-origin | Required for SharedArrayBuffer (WASM multithreading) |
| Cross-Origin-Embedder-Policy | require-corp | Required for SharedArrayBuffer |
| Content-Security-Policy | (restricted) | Allows self + wasm-unsafe-eval + Google Fonts |
| X-Content-Type-Options | nosniff | Prevent MIME sniffing |
| X-Frame-Options | DENY | Prevent iframe embedding |

## Notes

- **No CI build**: WASM compilation requires nightly Rust with `build-std`, local `arkworks-algebra` path dependency, and ~5 min compile time. Build locally, deploy the output.
- **Compression**: Cloudflare Pages handles brotli/gzip automatically at the edge.
- **Caching**: Cloudflare caches static assets at the CDN. Vite's hashed filenames ensure cache busting for JS/CSS. The WASM and preprocessing files use content-based URLs from the worker.
