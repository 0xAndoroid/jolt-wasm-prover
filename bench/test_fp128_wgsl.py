# Oracle test for frontend/public/wgsl/fp128.wgsl: runs mul / add / sub /
# mul-add over random and edge-case u128 inputs (including non-canonical
# values >= p) on the GPU of headless WebKit and compares every result with
# Python big-int arithmetic mod p.
#
# Usage: uv run --with playwright python bench/test_fp128_wgsl.py [--n 100000] [--browser webkit|chromium-unsafe] [--seed 1]
# Needs the browsers once: uv run --with playwright python -m playwright install webkit chromium
# No server required: the page is route-fulfilled at http://localhost/
# (WebGPU needs a secure context; about:blank has no navigator.gpu).

import argparse
import base64
import random
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

C = 0xFFFF_A7F7
P = (1 << 128) - C
WGSL_DIR = Path(__file__).resolve().parent.parent / "frontend" / "public" / "wgsl"
EDGES = [0, 1, P - 1, P, P + 1, (1 << 128) - 1, C, 1 << 64, (1 << 64) - 1, 1 << 127]
OPS = {
    "mul": lambda a, b, c: (a * b) % P,
    "add": lambda a, b, c: (a + b) % P,
    "sub": lambda a, b, c: (a - b) % P,
    "muladd": lambda a, b, c: (a * b + c) % P,
}

PAGE_JS = """
async ({ src, a, b, c, n }) => {
    const dec = (s) => Uint8Array.from(atob(s), (ch) => ch.charCodeAt(0));
    const enc = (bytes) => {
        let s = '';
        for (let i = 0; i < bytes.length; i += 0x8000) s += String.fromCharCode.apply(null, bytes.subarray(i, i + 0x8000));
        return btoa(s);
    };
    if (!navigator.gpu) throw new Error('navigator.gpu missing');
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error('no adapter');
    const device = await adapter.requestDevice();
    const module = device.createShaderModule({ code: src });
    const info = await module.getCompilationInfo();
    const errors = info.messages.filter((m) => m.type === 'error').map((m) => `${m.lineNum}:${m.linePos} ${m.message}`);
    if (errors.length) throw new Error('WGSL: ' + errors.join('\\n'));
    const pipeline = device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: 'main' } });
    const storage = (bytes) => {
        const buf = device.createBuffer({ size: bytes.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
        device.queue.writeBuffer(buf, 0, bytes);
        return buf;
    };
    const [ab, bb, cb] = [dec(a), dec(b), dec(c)];
    const bufA = storage(ab), bufB = storage(bb), bufC = storage(cb);
    const out = device.createBuffer({ size: ab.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const uniform = device.createBuffer({ size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const results = {};
    for (const [name, op] of [['mul', 0], ['add', 1], ['sub', 2], ['muladd', 3]]) {
        const params = new Uint32Array(16);
        params[0] = n; params[1] = op;
        device.queue.writeBuffer(uniform, 0, params);
        const bindGroup = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [
            { binding: 0, resource: { buffer: uniform } },
            { binding: 1, resource: { buffer: bufA } },
            { binding: 2, resource: { buffer: bufB } },
            { binding: 3, resource: { buffer: bufC } },
            { binding: 4, resource: { buffer: out } },
        ] });
        const staging = device.createBuffer({ size: ab.byteLength, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
        const encoder = device.createCommandEncoder();
        const pass = encoder.beginComputePass();
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bindGroup);
        pass.dispatchWorkgroups(Math.ceil(n / 256));
        pass.end();
        encoder.copyBufferToBuffer(out, 0, staging, 0, ab.byteLength);
        const t0 = performance.now();
        device.queue.submit([encoder.finish()]);
        await staging.mapAsync(GPUMapMode.READ);
        results[name] = { bytes: enc(new Uint8Array(staging.getMappedRange())), ms: performance.now() - t0 };
        staging.unmap();
    }
    const ai = adapter.info || {};
    return { results, adapter: `${ai.vendor || '?'} ${ai.architecture || ''}`.trim() };
}
"""


def to_bytes(values):
    return b"".join(v.to_bytes(16, "little") for v in values)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100_000, help="random vectors (edge cases are added on top)")
    ap.add_argument("--browser", default="webkit", choices=["webkit", "chromium-unsafe"])
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    a, b, c = [], [], []
    for x in EDGES:
        for y in EDGES:
            for z in EDGES:
                a.append(x)
                b.append(y)
                c.append(z)
    for _ in range(args.n):
        a.append(rng.getrandbits(128))
        b.append(rng.getrandbits(128))
        c.append(rng.getrandbits(128))
    n = len(a)
    src = (WGSL_DIR / "fp128.wgsl").read_text() + "\n" + (WGSL_DIR / "fp128_ops.wgsl").read_text()

    with sync_playwright() as p:
        if args.browser == "webkit":
            browser = p.webkit.launch(headless=True)
        else:
            browser = p.chromium.launch(headless=True, args=["--enable-unsafe-webgpu", "--use-angle=metal"])
        page = browser.new_page()
        page.route("http://localhost/fp128-test", lambda route: route.fulfill(status=200, content_type="text/html", body="<!doctype html><title>fp128</title>"))
        page.goto("http://localhost/fp128-test")
        out = page.evaluate(
            PAGE_JS,
            {
                "src": src,
                "a": base64.b64encode(to_bytes(a)).decode(),
                "b": base64.b64encode(to_bytes(b)).decode(),
                "c": base64.b64encode(to_bytes(c)).decode(),
                "n": n,
            },
        )
        browser.close()

    failed = 0
    for name, ref in OPS.items():
        got = base64.b64decode(out["results"][name]["bytes"])
        mismatches = []
        for i in range(n):
            g = int.from_bytes(got[i * 16 : i * 16 + 16], "little")
            e = ref(a[i], b[i], c[i])
            if g != e:
                mismatches.append((i, a[i], b[i], c[i], g, e))
        status = "PASS" if not mismatches else "FAIL"
        print(f"{name:7s} {status} n={n} mismatches={len(mismatches)} gpu_ms={out['results'][name]['ms']:.1f}")
        for i, x, y, z, g, e in mismatches[:5]:
            print(f"  #{i}: a={x:#x} b={y:#x} c={z:#x}\n      got={g:#x}\n      exp={e:#x}")
        failed += len(mismatches)
    print(f"adapter={out['adapter']} browser={args.browser} vectors={n} ({len(EDGES) ** 3} edge, {args.n} random)")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
