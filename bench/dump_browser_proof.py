# Prove a program in the browser through frontend/public/worker.js (GPU off)
# and dump the proof, program IO and verifier preprocessing bytes so the
# native roundtrip can compare and cross-verify them:
#
#   node server.mjs &            # restart after every wasm/frontend rebuild (it caches)
#   uv run --with playwright python bench/dump_browser_proof.py --out /tmp/browser
#   JOLT_ROUNDTRIP_VERIFY_DIR=/tmp/browser cargo run --release --features native --bin test-roundtrip
#
# The default input is the sha2 roundtrip input of preprocessing/test_roundtrip.rs.

import argparse
import hashlib
import os
import sys

from playwright.sync_api import sync_playwright

PROVE_JS = """
async ({ input }) => {
    const pending = new Map();
    const worker = new Worker('/worker.js', { type: 'module' });
    worker.onmessage = (e) => {
        const r = pending.get(e.data.type);
        if (r) { pending.delete(e.data.type); r(e.data); }
        if (e.data.type === 'error') { for (const [, r] of pending) r(e.data); pending.clear(); }
    };
    const wait = (type) => new Promise((resolve) => pending.set(type, resolve));
    const send = (type, data, reply, transfer) => { const p = wait(reply); worker.postMessage({ type, data }, transfer || []); return p; };
    const init = await send('init', { numThreads: 4, gpu: false }, 'init-done');
    if (init.type === 'error') throw new Error(init.error);
    const [program, elf] = await Promise.all(['sha2_program.bin', 'sha2.elf'].map((f) => fetch(`/${f}?dump`).then((r) => r.arrayBuffer())));
    await send('load-program', { program: 'sha2', programPreprocessing: program, elfBytes: elf }, 'program-loaded', [program, elf]);
    const p = await send('prove', { program: 'sha2', input }, 'prove-done');
    if (p.type === 'error') throw new Error(p.error);
    const v = await send('verify', { program: 'sha2', proof: p.proof, programIo: p.programIo, verifierPreprocessing: p.verifierPreprocessing }, 'verify-done');
    return { valid: v.valid, cycles: p.numCycles, padded: p.paddedCycles, proof: Array.from(p.proof), io: Array.from(p.programIo), vp: Array.from(p.verifierPreprocessing) };
}
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8080")
    ap.add_argument("--out", required=True, help="directory for sha2_{proof,io,verifier_preprocessing}.bin")
    ap.add_argument("--input", default="jolt wasm prover roundtrip test input", help="sha2 guest input (UTF-8)")
    ap.add_argument("--browser", default="chromium", choices=["chromium", "webkit"])
    args = ap.parse_args()

    with sync_playwright() as pw:
        browser = getattr(pw, args.browser).launch(headless=True)
        page = browser.new_page()
        page.on("console", lambda m: sys.stderr.write(f"[page] {m.text}\n"))
        page.goto(args.url, wait_until="domcontentloaded")
        r = page.evaluate(PROVE_JS, {"input": list(args.input.encode())})
        browser.close()

    os.makedirs(args.out, exist_ok=True)
    for key, suffix in (("proof", "proof"), ("io", "io"), ("vp", "verifier_preprocessing")):
        with open(os.path.join(args.out, f"sha2_{suffix}.bin"), "wb") as f:
            f.write(bytes(r[key]))
    print(
        f"sha2: {r['cycles']} cycles, padded {r['padded']}, proof {len(r['proof'])} bytes, "
        f"verify={r['valid']}, sha256 {hashlib.sha256(bytes(r['proof'])).hexdigest()}"
    )
    if not r["valid"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
