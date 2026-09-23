# sha2-chain prove/verify oracle bench through frontend/public/worker.js with
# the GPU harness on and/or off. In `--gpu both` mode the proof bytes of every
# run must be identical across modes (the GPU path must not change the
# proof) and every verify must pass; the script exits non-zero otherwise.
#
# Usage:
#   node server.mjs &            # restart after every wasm rebuild (it caches)
#   uv run --with playwright python bench/bench_webgpu.py --iters 17,69 --runs 3 --gpu both \
#       [--browser webkit|chromium|chromium-unsafe] [--threads 8] [--url http://localhost:8080]
# Browsers once: uv run --with playwright python -m playwright install webkit chromium
#
# One JSON line per run on stdout, a summary object at the end (per size:
# warm-median prove seconds off/on, the on/off ratio and the GPU trace-commit
# stage breakdown reported by the prover); progress on stderr. `chromium` (no --enable-unsafe-webgpu) has no WebGPU adapter and
# exercises the CPU fallback (gpu_status "unavailable").

import argparse
import json
import os
import statistics
import sys

from playwright.sync_api import sync_playwright

SETUP_JS = """
async ({ threads, gpu, parityRounds }) => {
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
    worker.postMessage({ type: 'init', data: { numThreads: threads, gpu, parityRounds } });
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
        gpuCommit: p.gpuCommit ? JSON.parse(p.gpuCommit) : null,
        gpuDigitRange: p.gpuDigitRange ? JSON.parse(p.gpuDigitRange) : null,
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
    ap.add_argument("--iters", default="17", help="comma-separated sha2-chain iteration counts (17 -> 2^16 padded, 69 -> 2^18, 278 -> 2^20, 556 -> 2^21)")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--gpu", default="both", choices=["both", "on", "off", "w1", "w2", "all"],
                    help="on = W1+W2, w1 = commit only, w2 = digit-range only, all = off,w1,on,w2")
    ap.add_argument("--parity", type=int, default=0, help="check the first N GPU digit-range rounds per instance against the CPU prover")
    ap.add_argument("--browser", default="webkit", choices=["webkit", "chromium", "chromium-unsafe"])
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--url", default=os.environ.get("BENCH_URL", "http://localhost:8080"))
    args = ap.parse_args()
    modes = {"both": ["on", "off"], "on": ["on"], "off": ["off"], "w1": ["w1"], "w2": ["w2"], "all": ["off", "w1", "on", "w2"]}[args.gpu]
    gpu_arg = {"on": True, "off": False, "w1": "w1", "w2": "w2"}
    iters_list = [int(x) for x in args.iters.split(",")]

    results = []
    with sync_playwright() as p:
        browser = launch(p, args.browser)
        for label in modes:
            # Fresh page per mode: WebKit crashes when a second 4 GB wasm
            # memory is instantiated in a page whose first worker just died.
            page = browser.new_page()
            page.set_default_timeout(1_800_000)
            page.on("console", lambda msg: sys.stderr.write(f"[page] {msg.text}\n"))
            page.goto(args.url, wait_until="domcontentloaded")
            init = page.evaluate(SETUP_JS, {"threads": args.threads, "gpu": gpu_arg[label], "parityRounds": args.parity})
            sys.stderr.write(f"gpu={label}: worker ready, init gpu={json.dumps(init)}\n")
            for iters in iters_list:
                for i in range(args.runs):
                    r = page.evaluate(RUN_JS, {"iters": iters})
                    if "error" in r:
                        print(json.dumps({"gpu": label, "iters": iters, "run": i + 1, "error": r["error"]}))
                        sys.stderr.write(f"gpu={label} iters={iters} run {i + 1}: ERROR {r['error']}\n")
                        break
                    rec = {"gpu": label, "iters": iters, "run": i + 1, "log2Padded": r["paddedCycles"].bit_length() - 1, **r, "init": init}
                    results.append(rec)
                    print(json.dumps(rec), flush=True)
                    commit = r.get("gpuCommit")
                    commit_str = (
                        f" commit[pack {commit['pack_ms']:.1f} + upload {commit['upload_ms']:.1f} + gpu {commit['gpu_ms']:.1f}"
                        f" + readback {commit['readback_ms']:.1f} = {commit['total_ms']:.1f} ms x{commit['calls']}]"
                        if commit else ""
                    )
                    dr = r.get("gpuDigitRange")
                    commit_str += (
                        f" digit-range[{dr['instances']} inst {dr['gpu_rounds']} rounds {dr['ops']} ops: upload {dr['upload_ms']:.1f}"
                        f" + rounds {dr['rounds_ms']:.1f} + download {dr['download_ms']:.1f} = {dr['total_ms']:.1f} ms]"
                        if dr else ""
                    )
                    sys.stderr.write(
                        f"gpu={label} 2^{rec['log2Padded']} run {i + 1}: total {r['totalSeconds']:.2f}s [trace {r['traceSeconds']:.2f} + setup {r['setupSeconds']:.2f} + prove {r['proveSeconds']:.2f}] "
                        f"verify {r['verifySeconds']:.2f}s valid={r['valid']} sha={r['proofSha256'][:16]} peak {r['peakMB']:.0f} MB "
                        f"gpu_status={r['gpuStatus']} selftest {r['gpuSelftestMs']:.0f} ms ({r['gpuSelftestMismatches']} mism) roundtrip {r['gpuRoundtripUs']:.0f} us{commit_str}\n"
                    )
            page.evaluate(TEARDOWN_JS)
            page.close()
        browser.close()

    summary = {"iters": iters_list, "browser": args.browser, "threads": args.threads, "sizes": {}}
    ok = True
    table = []
    for iters in iters_list:
        size = {}
        for label in modes:
            rs = [r for r in results if r["gpu"] == label and r["iters"] == iters]
            if len(rs) < args.runs:
                ok = False
            if not rs:
                continue
            warm = rs[1:] if len(rs) >= 2 else rs
            mode = {
                "runs": len(rs),
                "log2Padded": rs[0]["log2Padded"],
                "coldProveSeconds": round(rs[0]["proveSeconds"], 3),
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
            commits = [r["gpuCommit"] for r in warm if r.get("gpuCommit")]
            if commits:
                mode["commitMs"] = {k: round(statistics.median(c[k] for c in commits), 2) for k in commits[0] if k != "calls"}
                mode["commitCalls"] = commits[0]["calls"]
            # gpu=on must really have run the GPU path (plain chromium has no adapter and
            # exercises the fallback); a quiet fall-through to the CPU is not a pass.
            expected = "disabled" if label == "off" else ("unavailable" if args.browser == "chromium" else "ok")
            mode["gpuStatusExpected"] = expected
            gpu_ok = all(r["gpuStatus"] == expected and not r["gpuSelftestMismatches"] for r in rs)
            drs = [r["gpuDigitRange"] for r in warm if r.get("gpuDigitRange")]
            if drs:
                mode["digitRangeMs"] = {k: round(statistics.median(d[k] for d in drs), 2) for k in drs[0]}
            size[label] = mode
            ok = ok and mode["allValid"] and len(mode["proofSha256"]) == 1 and gpu_ok
        shas = {tuple(m["proofSha256"]) for m in size.values()}
        if len(size) > 1:
            size["proofBytesIdentical"] = len(shas) == 1
            ok = ok and size["proofBytesIdentical"]
        if "on" in size and "off" in size:
            size["proveRatioOnOff"] = round(size["on"]["warmMedianProveSeconds"] / size["off"]["warmMedianProveSeconds"], 3)
        if "on" in size and "w1" in size:
            size["w2IncrementSeconds"] = round(size["w1"]["warmMedianProveSeconds"] - size["on"]["warmMedianProveSeconds"], 3)
        summary["sizes"][str(iters)] = size
        if size:
            log2 = next(iter(m["log2Padded"] for m in size.values() if isinstance(m, dict)))
            medians = {m: size.get(m, {}).get("warmMedianProveSeconds") for m in ("off", "w1", "on", "w2")}
            table.append((log2, medians, size.get("w2IncrementSeconds"), size.get("proofBytesIdentical"), {m: size.get(m, {}).get("digitRangeMs") for m in ("on", "w2")}))
    summary["ok"] = ok
    print(json.dumps({"summary": summary}, indent=1))
    fmt = lambda v: "-" if v is None else f"{v:.3f}"
    sys.stderr.write("\nsize   prove off   prove w1   prove on   prove w2   w2 incr (s)  sha equal  digit range (ms)\n")
    for log2, med, incr, same, dr in table:
        d = dr.get("on") or dr.get("w2")
        stages = (
            f"{d['instances']} inst {d['gpu_rounds']} rounds {d['ops']} ops: upload {d['upload_ms']:.1f} rounds {d['rounds_ms']:.1f} download {d['download_ms']:.1f} total {d['total_ms']:.1f}"
            if d else "-"
        )
        sys.stderr.write(f"2^{log2:<4} {fmt(med['off']):>9}  {fmt(med['w1']):>9}  {fmt(med['on']):>9}  {fmt(med['w2']):>9}  {fmt(incr):>11}  {str(same):>9}  {stages}\n")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
