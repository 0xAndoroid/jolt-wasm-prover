# Oracle test for frontend/public/wgsl/stage2/*: drives the fused stage-2 round kernels (compact round 0,
# compact materialize, field folds; factored alpha x lane_w + packing segments and dense weight tables; the
# sparse additional cubic; the reduce) over a synthetic instance in headless WebKit and compares every
# round's ten terms and the final tables with a Python big-int reference that folds dense tables.
#
# Usage: uv run --with playwright python bench/test_stage2_wgsl.py [--log-domain 16] [--seed 1] [--browser webkit|chromium-unsafe]

import argparse
import base64
import random
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

C = 0xFFFF_A7F7
P = (1 << 128) - C
WGSL = Path(__file__).resolve().parent.parent / "frontend" / "public" / "wgsl"
SHADERS = {"round": ["fp128", "stage2/common", "stage2/round"], "additional": ["fp128", "stage2/common", "stage2/additional"], "reduce": ["fp128", "stage2/common", "stage2/reduce"]}
WG = 256
NT = 6
HDR = 7  # vec4 slots of aux header

PAGE_JS = """
async ({ sources, buffers, jobs }) => {
    const dec = (s) => Uint8Array.from(atob(s), (ch) => ch.charCodeAt(0));
    const enc = (bytes) => { let s = ''; for (let i = 0; i < bytes.length; i += 0x8000) s += String.fromCharCode.apply(null, bytes.subarray(i, i + 0x8000)); return btoa(s); };
    const adapter = await navigator.gpu.requestAdapter();
    const device = await adapter.requestDevice({ requiredLimits: { maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize, maxBufferSize: adapter.limits.maxBufferSize, maxComputeWorkgroupStorageSize: adapter.limits.maxComputeWorkgroupStorageSize } });
    device.addEventListener('uncapturederror', (e) => { throw new Error('uncaptured: ' + e.error.message); });
    const pipelines = {};
    for (const [name, src] of Object.entries(sources)) {
        const module = device.createShaderModule({ code: src });
        const info = await module.getCompilationInfo();
        const errors = info.messages.filter((m) => m.type === 'error').map((m) => `${name} ${m.lineNum}:${m.linePos} ${m.message}`);
        if (errors.length) throw new Error('WGSL: ' + errors.join('\\n'));
        pipelines[name] = await device.createComputePipelineAsync({ layout: 'auto', compute: { module, entryPoint: 'main' } });
    }
    device.pushErrorScope('validation');
    const bufs = {};
    for (const [name, spec] of Object.entries(buffers)) {
        const size = Math.max(16, spec.size || dec(spec.data).byteLength);
        bufs[name] = device.createBuffer({ size, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
        if (spec.data) device.queue.writeBuffer(bufs[name], 0, dec(spec.data));
    }
    const uniforms = [];
    const results = [];
    for (const job of jobs) {
        for (const [name, data] of Object.entries(job.upload || {})) device.queue.writeBuffer(bufs[name], 0, dec(data));
        const enc2 = device.createCommandEncoder();
        job.passes.forEach((p, i) => {
            while (uniforms.length <= i) uniforms.push(device.createBuffer({ size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }));
            device.queue.writeBuffer(uniforms[i], 0, new Uint32Array(p.params));
            const entries = [{ binding: 0, resource: { buffer: uniforms[i] } }];
            for (const [b, name] of p.binds) entries.push({ binding: b, resource: { buffer: bufs[name] } });
            const pass = enc2.beginComputePass();
            pass.setPipeline(pipelines[p.shader]);
            pass.setBindGroup(0, device.createBindGroup({ layout: pipelines[p.shader].getBindGroupLayout(0), entries }));
            pass.dispatchWorkgroups(p.wgs);
            pass.end();
        });
        const stagings = [];
        for (const [name, len] of job.readback) {
            const st = device.createBuffer({ size: len, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
            enc2.copyBufferToBuffer(bufs[name], 0, st, 0, len);
            stagings.push([name, st]);
        }
        device.queue.submit([enc2.finish()]);
        const out = {};
        for (const [name, st] of stagings) { await st.mapAsync(GPUMapMode.READ); out[name] = enc(new Uint8Array(st.getMappedRange())); st.unmap(); st.destroy(); }
        results.push(out);
    }
    const verr = await device.popErrorScope();
    if (verr) throw new Error('validation: ' + verr.message);
    return results;
}
"""


def fe(v):
    return v % P


def to_bytes(values):
    return b"".join((v % P).to_bytes(16, "little") for v in values)


def from_bytes(b):
    return [int.from_bytes(b[i : i + 16], "little") for i in range(0, len(b), 16)]


def u32s(values):
    return b"".join(int(v).to_bytes(4, "little") for v in values)


def pack_digits(digits, bw):
    acc = 0
    for i, d in enumerate(digits):
        acc |= (d & ((1 << bw) - 1)) << (i * bw)
    n = (len(digits) * bw + 7) // 8
    return acc.to_bytes(n + 4, "little")


def fold(table, r):
    return [fe(table[2 * j] + r * ((table[2 * j + 1] if 2 * j + 1 < len(table) else 0) - table[2 * j])) for j in range((len(table) + 1) // 2)]


def reference_terms(w, p, e_first, e_second, n_units):
    """Ten terms: norm q0 q1 q2 (eq-weighted), rel c0 c1 c2 over pairs j < n_units."""
    mask = len(e_first) - 1
    bits = len(e_first).bit_length() - 1
    acc = [0] * 6
    for j in range(n_units):
        L = w[2 * j] if 2 * j < len(w) else 0
        R = w[2 * j + 1] if 2 * j + 1 < len(w) else 0
        dw = R - L
        P0, P1 = p[2 * j], p[2 * j + 1]
        dp = P1 - P0
        e = e_first[j & mask] * e_second[j >> bits]
        acc[0] += e * (L * (L + 1))
        acc[1] += e * (dw * (2 * L + 1))
        acc[2] += e * dw * dw
        acc[3] += L * P0
        acc[4] += L * dp + dw * P0
        acc[5] += dw * dp
    return [fe(a) for a in acc]


def reference_additional(w, sparse):
    """Cubic from sorted sparse (index, linear, batched_binary) over the current table w."""
    acc = [0] * 4
    i = 0
    while i < len(sparse):
        parent = sparse[i][0] >> 1
        lin = [0, 0]
        bin_ = [0, 0]
        while i < len(sparse) and sparse[i][0] >> 1 == parent:
            idx, l, b = sparse[i]
            lin[idx & 1] = l
            bin_[idx & 1] = b
            i += 1
        L = w[2 * parent] if 2 * parent < len(w) else 0
        R = w[2 * parent + 1] if 2 * parent + 1 < len(w) else 0
        dw = R - L
        dl = lin[1] - lin[0]
        db = bin_[1] - bin_[0]
        q0, q1, q2 = L * (L + 1), dw * (2 * L + 1), dw * dw
        acc[0] += L * lin[0] + bin_[0] * q0
        acc[1] += L * dl + dw * lin[0] + bin_[0] * q1 + db * q0
        acc[2] += dw * dl + bin_[0] * q2 + db * q1
        acc[3] += db * q2
    return [fe(a) for a in acc]


def bind_sparse(sparse, r):
    out = []
    i = 0
    while i < len(sparse):
        parent = sparse[i][0] >> 1
        lin = bin_ = 0
        while i < len(sparse) and sparse[i][0] >> 1 == parent:
            idx, l, b = sparse[i]
            scale = r if idx & 1 else (1 - r)
            lin += scale * l
            bin_ += scale * b
            i += 1
        lin, bin_ = fe(lin), fe(bin_)
        if lin or bin_:
            out.append((parent, lin, bin_))
    return out


def pairs_blob(sparse):
    """[m,0,0,0] l0 l1 b0 b1 per parent."""
    out = b""
    n = 0
    i = 0
    while i < len(sparse):
        parent = sparse[i][0] >> 1
        lin = [0, 0]
        bin_ = [0, 0]
        while i < len(sparse) and sparse[i][0] >> 1 == parent:
            idx, l, b = sparse[i]
            lin[idx & 1] = l
            bin_[idx & 1] = b
            i += 1
        out += u32s([parent, 0, 0, 0]) + to_bytes([lin[0], lin[1], bin_[0], bin_[1]])
        n += 1
    return out, n


def ppt_for(units):
    ppt = 8
    while ppt > 1 and units < WG * 1024 * ppt:
        ppt //= 2
    return ppt


def run_instance(rng, log_domain, factored, addends=True):
    """Build the GPU jobs and the Python expectations for one instance; returns (buffers, jobs, expected_terms, final_tables).

    `addends=False` runs without sparse additional terms (no additional pass, zero-count reduce)."""
    cc_bits = 6
    cc = 1 << cc_bits
    lane_bits = log_domain - cc_bits
    cap = 1 << lane_bits
    live_lanes = max(1, int(cap * 0.71))
    live_len = live_lanes * cc
    domain = 1 << log_domain
    bw = 3
    digits = [rng.randrange(-4, 4) for _ in range(live_len)]
    alpha = [rng.getrandbits(128) % P for _ in range(cc)]
    lane_w = [rng.getrandbits(128) % P for _ in range(cap)]
    sources = [[rng.getrandbits(128) % P for _ in range(4 * cc)], [rng.getrandbits(128) % P for _ in range(5 * cc)]]
    # (target_lane_start, lane_count, source_lane_start, source_index, factor)
    segments = [(3, 2, 1, 0, rng.getrandbits(128) % P), (10, 3, 0, 1, rng.getrandbits(128) % P), (live_lanes - 3, 2, 2, 1, rng.getrandbits(128) % P), (live_lanes - 1, 1, 3, 0, rng.getrandbits(128) % P)]
    # per-lane expansion: lane_map[lane] = start << 8 | count into per-lane segments (source_lane, source_index, factor)
    lane_segs = []
    lane_map = [0] * live_lanes
    for (t0, n, s0, si, f) in segments:
        for lane in range(t0, t0 + n):
            lane_map[lane] = (len(lane_segs) << 8) | 1
            lane_segs.append((s0 + lane - t0, si, f))
    # one sparse-kind lane with two segments (lane_map count 2)
    multi = [(6, 2, 0, rng.getrandbits(128) % P), (6, 1, 1, rng.getrandbits(128) % P)]
    lane_map[6] = (len(lane_segs) << 8) | len(multi)
    lane_segs.extend((sl, si, f) for (_, sl, si, f) in multi)
    p0 = [alpha[i & (cc - 1)] * lane_w[i >> cc_bits] % P for i in range(domain)]
    for lane, m in enumerate(lane_map):
        for (sl, si, f) in lane_segs[m >> 8 : (m >> 8) + (m & 0xFF)]:
            for c in range(cc):
                p0[lane * cc + c] = fe(p0[lane * cc + c] + f * sources[si][sl * cc + c])
    sparse = {}
    for _ in range(40):
        sparse[rng.randrange(live_len)] = [rng.getrandbits(128) % P, 0]
    start = rng.randrange(live_len - 200)
    for i in range(start, start + 150):
        sparse.setdefault(i, [0, 0])[1] = rng.getrandbits(128) % P
    sparse = sorted((i, l, b) for i, (l, b) in sparse.items()) if addends else []

    w = [d % P for d in digits]
    p = p0[:]
    rounds = log_domain - 4
    table_len = lambda k: (domain >> k) + -(-live_len // (1 << k))  # noqa: E731
    ta_cap = max(table_len(k) for k in range(1, rounds) if k % 2 == 1)
    tb_cap = max([table_len(k) for k in range(2, rounds) if k % 2 == 0] or [1])
    buffers = {
        "digits": {"data": base64.b64encode(pack_digits(digits, bw)).decode()},
        "lane_w": {"data": base64.b64encode(to_bytes(lane_w) + u32s(lane_map) + b"\0" * 16).decode()},
        "U": {"data": base64.b64encode(to_bytes(p0 if not factored else [0])).decode(), "size": domain * 16},
        "TA": {"size": ta_cap * 16},
        "TB": {"size": tb_cap * 16},
        "partials": {"size": 8192 * NT * 16},
        "out": {"size": 16 * 16},
        "aux": {"size": (HDR + 2 * (1 << (log_domain // 2 + 1)) + cc + sum(len(s) for s in sources) + 2 * len(lane_segs) + 5 * 4096) * 16},
    }
    jobs, expected = [], []
    alpha_k, src_k, sparse_k = alpha[:], [s[:] for s in sources], sparse
    r_prev = None
    w_prev = w
    for k in range(rounds):
        d_k = domain >> k
        n_units = d_k // 2
        nf = 1 << ((log_domain - 1 - k) // 2)
        ns = n_units // nf
        e_first = [rng.getrandbits(128) % P for _ in range(nf)]
        e_second = [rng.getrandbits(128) % P for _ in range(ns)]
        if k > 0:
            w_prev = w
            w = fold(w, r_prev)
            p = fold(p, r_prev)
            sparse_k = bind_sparse(sparse_k, r_prev)
            if k <= cc_bits:
                alpha_k = fold(alpha_k, r_prev)
                cck = cc >> k
                src_k = [[fe(s[lane * (2 * cck) + 2 * c] + r_prev * (s[lane * (2 * cck) + 2 * c + 1] - s[lane * (2 * cck) + 2 * c])) for lane in range(len(s) // (2 * cck)) for c in range(cck)] for s in src_k]
        expected.append(reference_terms(w, p, e_first, e_second, n_units) + reference_additional(w, sparse_k))
        cw = max(cc_bits - k, 0)
        off_first = HDR
        off_second = off_first + nf
        alpha_off = off_second + ns
        o = alpha_off + len(alpha_k)
        src_offs = []
        for s in src_k:
            src_offs.append(o)
            o += len(s)
        seg_off = o
        pairs_off = seg_off + 2 * len(lane_segs)
        pblob, n_pairs = pairs_blob(sparse_k)
        aux = u32s([cw, alpha_off, seg_off, len(lane_segs), 0, 0, pairs_off, n_pairs, cap, len(src_k), live_lanes, 0])
        aux += u32s(src_offs + [0] * (16 - len(src_offs)))
        aux += to_bytes(e_first) + to_bytes(e_second) + to_bytes(alpha_k) + b"".join(to_bytes(s) for s in src_k)
        for (sl, si, f) in lane_segs:
            aux += u32s([sl, si, 0, 0]) + to_bytes([f])
        aux += pblob
        src_mode = 0 if k == 0 else (1 if k == 1 else 2)
        src, dst = ("U", "TA") if k <= 1 else (("TA", "TB") if k % 2 == 0 else ("TB", "TA"))
        dst_w_off, src_w_off = d_k, 2 * d_k
        if factored and k <= cc_bits:
            wflags = 1 | (2 if k == cc_bits else 0)
        else:
            wflags = 0 if k == 0 else 6
        ppt = min(ppt_for(n_units), nf)
        wgs = -(-n_units // (WG * ppt))
        live_param = live_len if k <= 1 else len(w_prev)
        r_words = [int.from_bytes((r_prev or 0).to_bytes(16, "little")[i : i + 4], "little") for i in range(0, 16, 4)]
        params = [n_units, nf.bit_length() - 1, off_first, off_second, ppt, bw, src_mode, 0] + r_words + [live_param, src_w_off, dst_w_off, wflags]
        add_wgs = -(-n_pairs // WG)
        add_off = NT * wgs
        add_params = [n_pairs, 0, add_off, 0, 1, bw, src_mode, 0] + [0] * 4 + [live_len if k == 0 else len(w), 0, dst_w_off, 0]
        red_params = [wgs, add_wgs, add_off, 0, 1, 0, 0, 0] + [0] * 8
        passes = [{"shader": "round", "wgs": wgs, "params": params, "binds": [[1, "digits"], [2, "aux"], [3, "partials"], [4, src], [5, dst], [6, "lane_w"]]}]
        if n_pairs:
            passes.append({"shader": "additional", "wgs": add_wgs, "params": add_params, "binds": [[1, "digits"], [2, "aux"], [3, "partials"], [5, dst]]})
        passes.append({"shader": "reduce", "wgs": 1, "params": red_params, "binds": [[3, "partials"], [4, "out"]]})
        readback = [["out", 160]]
        if k == rounds - 1:
            readback.append([dst, (d_k + len(w)) * 16])
        jobs.append({"upload": {"aux": base64.b64encode(aux).decode()}, "passes": passes, "readback": readback})
        r_prev = rng.getrandbits(128) % P
    return buffers, jobs, expected, (w, p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-domain", type=int, default=16)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--browser", default="webkit", choices=["webkit", "chromium-unsafe"])
    args = ap.parse_args()
    sources = {name: "\n".join((WGSL / f"{part}.wgsl").read_text() for part in parts) for name, parts in SHADERS.items()}
    rng = random.Random(args.seed)
    failed = 0
    pairs_total = 0
    with sync_playwright() as pw:
        browser = pw.webkit.launch(headless=True) if args.browser == "webkit" else pw.chromium.launch(headless=True, args=["--enable-unsafe-webgpu", "--use-angle=metal"])
        page = browser.new_page()
        page.route("http://localhost/stage2-test", lambda route: route.fulfill(status=200, content_type="text/html", body="<!doctype html><title>stage2</title>"))
        page.goto("http://localhost/stage2-test")
        for factored, addends in ((True, True), (False, True), (True, False)):
            buffers, jobs, expected, (w, p) = run_instance(rng, args.log_domain, factored, addends)
            results = page.evaluate(PAGE_JS, {"sources": sources, "buffers": buffers, "jobs": jobs})
            mode = ("factored" if factored else "dense") + ("" if addends else "/noadd")
            for k, (res, exp) in enumerate(zip(results, expected)):
                got = from_bytes(base64.b64decode(res["out"]))[:10]
                pairs_total += (1 << args.log_domain) >> (k + 1)
                bad = [t for t in range(10) if got[t] != exp[t]]
                status = "PASS" if not bad else "FAIL"
                if bad or k < 3 or k == len(expected) - 1:
                    print(f"{mode:14s} round {k:2d} {status} n_units={(1 << args.log_domain) >> (k + 1)} bad_terms={bad}")
                for t in bad[:3]:
                    print(f"   term {t}: got={got[t]:#x} exp={exp[t]:#x}")
                failed += len(bad)
            final = results[-1]
            dst_name = [n for n in final if n != "out"][0]
            tab = from_bytes(base64.b64decode(final[dst_name]))
            d_k = len(p)
            p_ok = tab[:d_k] == p
            w_ok = tab[d_k : d_k + len(w)] == w
            print(f"{mode:14s} final tables: P {'PASS' if p_ok else 'FAIL'} ({d_k} entries) W {'PASS' if w_ok else 'FAIL'} ({len(w)} entries)")
            failed += (not p_ok) + (not w_ok)
        browser.close()
    print(f"pairs evaluated: {pairs_total}, failures: {failed}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
