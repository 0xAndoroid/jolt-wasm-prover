# sha2-chain prove/verify oracle bench through frontend/public/worker.js with
# the GPU harness on and/or off. In `--gpu both` mode the proof bytes of every
# run must be identical across modes (the GPU path must not change the
# proof) and every verify must pass; the script exits non-zero otherwise.
#
# Usage:
#   node server.mjs &            # restart after every wasm rebuild (it caches)
#   uv run --with playwright python bench/bench_webgpu.py --iters 17 --runs 3 --gpu both \
#       [--browser webkit|chromium|chromium-unsafe] [--threads 8] [--url http://localhost:8080]
# Browsers once: uv run --with playwright python -m playwright install webkit chromium
#
# One JSON line per run on stdout, a summary object at the end; progress on
# stderr. `chromium` (no --enable-unsafe-webgpu) has no WebGPU adapter and
# exercises the CPU fallback (gpu_status "unavailable").

import argparse
import json
import os
import statistics
import sys

from playwright.sync_api import sync_playwright

SETUP_JS = """
async ({ threads, gpu }) => {
    window.__bench = { pending: new Map() };
    const worker = new Worker('/worker.js', { type: 'module' });
    window.__bench.worker = worker;
    worker.onmessage = (e) => {
        const { type } = e.data;
        const resolver = window.__bench.pending.get(type);
        if (resolver) {
            window.__bench.pending.delete(type);
            resolver(e.data);
        }
        if (type === 'error') {
            for (const [, r] of window.__bench.pending) r({ type: 'error', error: e.data.error });
            window.__bench.pending.clear();
        }
    };
    window.__bench.wait = (type) => new Promise((resolve) => window.__bench.pending.set(type, resolve));

    const initDone = window.__bench.wait('init-done');
    worker.postMessage({ type: 'init', data: { numThreads: threads, gpu } });
    const init = await initDone;
    if (init.type === 'error') throw new Error(init.error);

    const [program, elf] = await Promise.all(
        ['sha2_chain_program.bin', 'sha2_chain.elf'].map((f) => fetch(`/${f}?bench`).then((r) => {
            if (!r.ok) throw new Error(`fetch ${f}: ${r.status}`);
            return r.arrayBuffer();
        })),
    );
    const loaded = window.__bench.wait('program-loaded');
    worker.postMessage({ type: 'load-program', data: { program: 'sha2-chain', programPreprocessing: program, elfBytes: elf } }, [program, elf]);
    const l = await loaded;
    if (l.type === 'error') throw new Error(l.error);
    return init.gpu;
}
"""

RUN_JS = """
async ({ iters }) => {
    const done = window.__bench.wait('prove-done');
    window.__bench.worker.postMessage({ type: 'prove', data: { program: 'sha2-chain', input: Array.from(new Uint8Array(32).fill(5)), numIters: iters } });
    const p = await done;
    if (p.type === 'error') return { error: p.error };
    const digest = await crypto.subtle.digest('SHA-256', p.proof);
    const sha = Array.from(new Uint8Array(digest)).map((b) => b.toString(16).padStart(2, '0')).join('');
    const verified = window.__bench.wait('verify-done');
    window.__bench.worker.postMessage({ type: 'verify', data: { program: 'sha2-chain', proof: p.proof, programIo: p.programIo, verifierPreprocessing: p.verifierPreprocessing } });
    const v = await verified;
    if (v.type === 'error') return { error: v.error };
    return {
        totalSeconds: p.elapsed / 1000,
        traceSeconds: p.traceMs / 1000,
        setupSeconds: p.setupMs / 1000,
        proveSeconds: p.proveMs / 1000,
        verifySeconds: v.elapsed / 1000,
        valid: v.valid,
        numCycles: p.numCycles,
        paddedCycles: p.paddedCycles,
        proofSize: p.proofSize,
        proofSha256: sha,
        peakMB: p.peakMemory / 1024 / 1024,
        gpuStatus: p.gpuStatus,
        gpuSelftestMs: p.gpuSelftestMs,
        gpuSelftestMismatches: p.gpuSelftestMismatches,
        gpuRoundtripUs: p.gpuRoundtripUs,
    };
}
"""

TEARDOWN_JS = "() => { window.__bench.worker.terminate(); window.__bench = null; }"


def launch(p, browser):
    if browser == "webkit":
        return p.webkit.launch(headless=True)
    args = ["--enable-unsafe-webgpu", "--use-angle=metal"] if browser == "chromium-unsafe" else []
    return p.chromium.launch(headless=True, args=args)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=17, help="sha2-chain iterations (17 -> 2^16 padded)")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--gpu", default="both", choices=["both", "on", "off"])
    ap.add_argument("--browser", default="webkit", choices=["webkit", "chromium", "chromium-unsafe"])
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--url", default=os.environ.get("BENCH_URL", "http://localhost:8080"))
    args = ap.parse_args()
    modes = {"both": [True, False], "on": [True], "off": [False]}[args.gpu]

    results = []
    with sync_playwright() as p:
        browser = launch(p, args.browser)
        for gpu in modes:
            label = "on" if gpu else "off"
            # Fresh page per mode: WebKit crashes when a second 4 GB wasm
            # memory is instantiated in a page whose first worker just died.
            page = browser.new_page()
            page.set_default_timeout(1_800_000)
            page.on("console", lambda msg: sys.stderr.write(f"[page] {msg.text}\n"))
            page.goto(args.url, wait_until="domcontentloaded")
            init = page.evaluate(SETUP_JS, {"threads": args.threads, "gpu": gpu})
            sys.stderr.write(f"gpu={label}: worker ready, init gpu={json.dumps(init)}\n")
            for i in range(args.runs):
                r = page.evaluate(RUN_JS, {"iters": args.iters})
                if "error" in r:
                    print(json.dumps({"gpu": label, "iters": args.iters, "run": i + 1, "error": r["error"]}))
                    sys.stderr.write(f"gpu={label} run {i + 1}: ERROR {r['error']}\n")
                    break
                rec = {"gpu": label, "iters": args.iters, "run": i + 1, "log2Padded": r["paddedCycles"].bit_length() - 1, **r, "init": init}
                results.append(rec)
                print(json.dumps(rec), flush=True)
                sys.stderr.write(
                    f"gpu={label} run {i + 1}: total {r['totalSeconds']:.2f}s [trace {r['traceSeconds']:.2f} + setup {r['setupSeconds']:.2f} + prove {r['proveSeconds']:.2f}] "
                    f"verify {r['verifySeconds']:.2f}s valid={r['valid']} sha={r['proofSha256'][:16]} peak {r['peakMB']:.0f} MB "
                    f"gpu_status={r['gpuStatus']} selftest {r['gpuSelftestMs']:.0f} ms ({r['gpuSelftestMismatches']} mism) roundtrip {r['gpuRoundtripUs']:.0f} us\n"
                )
            page.evaluate(TEARDOWN_JS)
            page.close()
        browser.close()

    summary = {"iters": args.iters, "browser": args.browser, "threads": args.threads, "modes": {}}
    ok = True
    for gpu in modes:
        label = "on" if gpu else "off"
        rs = [r for r in results if r["gpu"] == label]
        if len(rs) < args.runs:
            ok = False
        if not rs:
            continue
        warm = rs[1:] if len(rs) >= 2 else rs
        summary["modes"][label] = {
            "runs": len(rs),
            "log2Padded": rs[0]["log2Padded"],
            "coldTotalSeconds": round(rs[0]["totalSeconds"], 3),
            "warmMedianTotalSeconds": round(statistics.median(r["totalSeconds"] for r in warm), 3),
            "warmMedianProveSeconds": round(statistics.median(r["proveSeconds"] for r in warm), 3),
            "verifySeconds": round(rs[0]["verifySeconds"], 3),
            "allValid": all(r["valid"] for r in rs),
            "proofSha256": sorted({r["proofSha256"] for r in rs}),
            "gpuStatus": rs[0]["gpuStatus"],
            "gpuSelftestMs": round(rs[0]["gpuSelftestMs"], 1),
            "gpuRoundtripUs": round(rs[0]["gpuRoundtripUs"], 1),
            "peakMB": round(max(r["peakMB"] for r in rs)),
        }
        ok = ok and summary["modes"][label]["allValid"] and len(summary["modes"][label]["proofSha256"]) == 1
    if args.gpu == "both" and len(summary["modes"]) == 2:
        same = summary["modes"]["on"]["proofSha256"] == summary["modes"]["off"]["proofSha256"]
        summary["proofBytesIdentical"] = same
        ok = ok and same
    summary["ok"] = ok
    print(json.dumps({"summary": summary}, indent=1))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
