# End-to-end check of the GPU / CPU mode selector in the React UI plus the
# dead-proxy fallback of worker.js. Exits non-zero on the first failure.
#
#   node server.mjs &            # restart after every wasm/frontend rebuild (it caches)
#   uv run --with playwright python bench/test_ui_modes.py [--url http://localhost:8080] [--shots DIR]
#
# WebKit (has a WebGPU adapter): default mode GPU, CPU proof bytes == GPU
# proof bytes for SHA-256 / Keccak Chain / ECDSA, mode persists across reload,
# GPU re-enabled without reload.
# Chromium (no adapter): lands on CPU with the reason shown, GPU segment
# disabled. Dead proxy: init with a 2 s op timeout, hang the proxy, the prove
# fails with `gpu proxy unresponsive`, the next prove runs on the CPU.

import argparse
import os
import re
import sys
import time

from playwright.sync_api import sync_playwright

RADIO = '[role=radiogroup] [role=radio]'
PANEL = '[role=tabpanel][data-state=active]'
CHECKED = f'{RADIO}[aria-checked="true"]'
SHA_RE = re.compile(r"Proof SHA-256: ([0-9a-f]{64})")

DEAD_PROXY_JS = """
async ({ iters }) => {
    const worker = new Worker('/worker.js', { type: 'module' });
    const pending = new Map();
    worker.onmessage = (e) => {
        const r = pending.get(e.data.type) || (e.data.type === 'error' && pending.get('*'));
        if (r) { pending.clear(); r(e.data); }
    };
    const wait = (type) => new Promise((resolve) => { pending.set(type, resolve); pending.set('*', resolve); });
    const send = (type, data, reply) => { const p = wait(reply); worker.postMessage({ type, data }); return p; };
    const init = await send('init', { numThreads: 4, gpu: true, gpuTimeoutMs: 2000 }, 'init-done');
    if (init.type === 'error') throw new Error(init.error);
    const [program, elf] = await Promise.all(['sha2_chain_program.bin', 'sha2_chain.elf'].map((f) => fetch(`/${f}?ui-test`).then((r) => r.arrayBuffer())));
    const loaded = wait('program-loaded');
    worker.postMessage({ type: 'load-program', data: { program: 'sha2-chain', programPreprocessing: program, elfBytes: elf } }, [program, elf]);
    await loaded;
    worker.postMessage({ type: 'hang-gpu-proxy' });
    const input = Array.from(new Uint8Array(32).fill(5));
    const t0 = performance.now();
    const dead = await send('prove', { program: 'sha2-chain', input, numIters: iters }, 'prove-done');
    const deadMs = performance.now() - t0;
    const again = await send('prove', { program: 'sha2-chain', input, numIters: iters }, 'prove-done');
    let valid = null;
    if (again.type === 'prove-done') {
        const v = await send('verify', { program: 'sha2-chain', proof: again.proof, programIo: again.programIo, verifierPreprocessing: again.verifierPreprocessing }, 'verify-done');
        valid = v.valid;
    }
    worker.terminate();
    return { init: init.gpu, dead: { type: dead.type, error: dead.error, gpu: dead.gpu, ms: deadMs }, again: { type: again.type, error: again.error, gpuStatus: again.gpuStatus, valid } };
}
"""


def check(cond, msg):
    if not cond:
        sys.stderr.write(f"FAIL: {msg}\n")
        sys.exit(1)
    sys.stderr.write(f"ok: {msg}\n")


def wait_ready(page):
    page.wait_for_function("() => document.querySelector('#status')?.classList.contains('ready')")


def checked_label(page):
    return page.locator(CHECKED).inner_text().strip()


def prove(page):
    seen = len(SHA_RE.findall(page.locator(f"{PANEL} .output").inner_text()))
    page.locator(f"{PANEL} .prove-btn").click()
    page.wait_for_function(
        f"(n) => (document.querySelector('{PANEL} .output')?.innerText.match(/Proof SHA-256: [0-9a-f]{{64}}/g) || []).length > n",
        arg=seen,
    )
    sha = SHA_RE.findall(page.locator(f"{PANEL} .output").inner_text())[-1]
    mode = page.locator(f"{PANEL} [data-proof-mode]").get_attribute("data-proof-mode")
    return sha, mode


def verify(page, label):
    page.locator(f"{PANEL} .verify-btn").click()
    # "Invalid" does not contain "Valid" (lowercase v); wait for either so a bad proof fails fast
    page.wait_for_function(f"() => /Valid|Invalid/.test(document.querySelector('{PANEL}')?.innerText || '')")
    check("Invalid" not in page.locator(PANEL).inner_text(), f"{label}: proof verified in the UI")


def select_tab(page, name):
    page.get_by_role("tab", name=name).click()
    page.wait_for_selector(f"{PANEL} .prove-btn")


def no_overflow(page):
    return page.evaluate("() => document.documentElement.scrollWidth <= document.documentElement.clientWidth")


def shoot(page, shots, name):
    if not shots:
        return
    os.makedirs(shots, exist_ok=True)
    for width in (375, 1440):
        page.set_viewport_size({"width": width, "height": 900})
        page.wait_for_timeout(300)
        check(no_overflow(page), f"no horizontal overflow at {width}px ({name})")
        page.screenshot(path=os.path.join(shots, f"{name}-{width}.png"), full_page=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=os.environ.get("BENCH_URL", "http://localhost:8080"))
    ap.add_argument("--shots", default=None, help="directory for 375/1440 px screenshots")
    ap.add_argument("--iters", type=int, default=17, help="sha2-chain iterations for the dead-proxy prove")
    args = ap.parse_args()

    with sync_playwright() as p:
        browser = p.webkit.launch(headless=True)
        context = browser.new_context(viewport={"width": 1440, "height": 900})
        context.set_default_timeout(600_000)

        def open_page():
            page = context.new_page()
            page.on("console", lambda m: sys.stderr.write(f"[webkit] {m.text}\n") if m.text.startswith("[gpu") or "error" in m.type else None)
            page.goto(args.url, wait_until="domcontentloaded")
            wait_ready(page)
            return page

        page = open_page()
        check(checked_label(page).startswith("GPU"), f"default mode is GPU ({checked_label(page)})")
        check(page.locator("text=GPU unavailable").count() == 0, "no unavailability reason shown")
        check(page.evaluate("() => localStorage.getItem('jolt-prover-mode')") is None, "default mode leaves localStorage untouched")
        sha_gpu, mode = prove(page)
        check(mode == "gpu", f"first proof ran in GPU mode (badge {mode})")
        verify(page, "SHA-256 GPU")
        select_tab(page, "Keccak Chain")
        keccak_gpu, mode = prove(page)
        check(mode == "gpu", f"Keccak Chain proof ran in GPU mode (badge {mode})")
        verify(page, "Keccak Chain GPU")
        select_tab(page, "ECDSA")
        ecdsa_gpu, mode = prove(page)
        check(mode == "gpu", f"ECDSA proof ran in GPU mode (badge {mode})")
        verify(page, "ECDSA GPU")

        page.locator(RADIO, has_text="CPU only").click()
        page.wait_for_function(f"() => document.querySelector('{CHECKED}')?.innerText.includes('CPU')")
        ecdsa_cpu, mode = prove(page)
        check(mode == "cpu", f"ECDSA proof ran in CPU mode (badge {mode})")
        check(ecdsa_cpu == ecdsa_gpu, f"ECDSA CPU proof bytes == GPU proof bytes ({ecdsa_cpu[:16]})")
        verify(page, "ECDSA CPU")
        sys.stderr.write(page.locator(f"{PANEL} .output").inner_text() + "\n")
        shoot(page, args.shots, "webkit-ecdsa")
        select_tab(page, "Keccak Chain")
        keccak_cpu, mode = prove(page)
        check(mode == "cpu", f"Keccak Chain proof ran in CPU mode (badge {mode})")
        check(keccak_cpu == keccak_gpu, f"Keccak Chain CPU proof bytes == GPU proof bytes ({keccak_cpu[:16]})")
        verify(page, "Keccak Chain CPU")
        select_tab(page, "SHA-256")
        sha_cpu, mode = prove(page)
        check(mode == "cpu", f"second SHA-256 proof ran in CPU mode (badge {mode})")
        check(sha_cpu == sha_gpu, f"SHA-256 CPU proof bytes == GPU proof bytes ({sha_cpu[:16]})")
        verify(page, "SHA-256 CPU")
        shoot(page, args.shots, "webkit-cpu")
        check(page.evaluate("() => localStorage.getItem('jolt-prover-mode')") == "cpu", "choice persisted as 'cpu'")

        # A fresh page in the same context stands in for a reload: WebKit crashes when a
        # second 4 GB wasm memory is instantiated in a page whose first worker just died.
        page.close()
        page = open_page()
        check(checked_label(page) == "CPU only", "mode restored from localStorage in a new page")

        page.locator(RADIO, has_text="GPU").click()
        page.wait_for_function(f"() => document.querySelector('{CHECKED}')?.innerText.includes('GPU')")
        wait_ready(page)
        sha_gpu2, mode = prove(page)
        check(mode == "gpu" and sha_gpu2 == sha_gpu, f"GPU re-enabled without reload, proof identical ({mode})")
        shoot(page, args.shots, "webkit-gpu")
        page.close()
        browser.close()

        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 900})
        page.set_default_timeout(600_000)
        page.goto(args.url, wait_until="domcontentloaded")
        wait_ready(page)
        check(checked_label(page) == "CPU only", "chromium without an adapter lands on CPU")
        check(page.locator(RADIO, has_text="GPU").is_disabled(), "GPU segment disabled")
        reason = page.locator("text=GPU unavailable").inner_text()
        check(bool(reason), f"reason shown: {reason}")
        shoot(page, args.shots, "chromium")
        page.close()
        browser.close()

        browser = p.webkit.launch(headless=True)
        page = browser.new_page()
        page.set_default_timeout(600_000)
        page.on("console", lambda m: sys.stderr.write(f"[dead] {m.text}\n") if m.type == "error" else None)
        page.goto(args.url, wait_until="domcontentloaded")
        t0 = time.time()
        r = page.evaluate(DEAD_PROXY_JS, {"iters": args.iters})
        sys.stderr.write(f"dead-proxy: {r}\n")
        check(r["init"]["status"] == "ok", "dead-proxy session started with the GPU on")
        check(r["dead"]["type"] == "error" and "gpu proxy unresponsive" in (r["dead"]["error"] or ""), f"prove failed with the timeout error in {r['dead']['ms'] / 1000:.1f}s")
        check(r["dead"]["ms"] < 10_000, "time to error under 10 s")
        check((r["dead"]["gpu"] or {}).get("status") == "unavailable", "error carries gpu.status unavailable")
        check(r["again"]["type"] == "prove-done" and r["again"]["gpuStatus"] == "unavailable" and r["again"]["valid"] is True,
              f"next prove in the same session ran on the CPU and verified ({time.time() - t0:.0f}s total)")
        page.close()
        browser.close()
    print("test_ui_modes: all checks passed")


if __name__ == "__main__":
    main()
