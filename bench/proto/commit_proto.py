"""Standalone WebGPU prototype of the Akita one-hot trace commit accumulate.

    uv run --with playwright --with numpy python bench/proto/commit_proto.py \
        [--shape small|full] [--variant v1|v2|v2b] [--chunk N] [--repeat N] [--spot N] [--seed S]

Launches Playwright's headless WebKit, serves bench/proto/harness.html + shaders + data via
page.route on http://localhost:7777, runs prep -> main -> reduce, verifies against a Python
big-int reference, prints one JSON summary line.
"""
import argparse, base64, json, mimetypes, sys, time
from pathlib import Path

import numpy as np
from playwright.sync_api import sync_playwright

HERE = Path(__file__).resolve().parent
C = 0xFFFFA7F7
P = (1 << 128) - C
MASK32 = (1 << 32) - 1

SHAPES = {
    # T rows, real columns, column_capacity, blocks_per_column, positions_per_block
    "small": dict(T=8192, cols=8, colcap=8, blocks=4, positions=64),
    "full": dict(T=262144, cols=57, colcap=64, blocks=4, positions=2048),
}
VARIANTS = {
    "v1": dict(file="commit_accumulate_v1.wgsl", chunked=False),
    "v2": dict(file="commit_accumulate_v2.wgsl", chunked=True),
    "v2b": dict(file="commit_accumulate_v2b.wgsl", chunked=True),
}


def gen_data(shape, seed):
    rng = np.random.default_rng(seed)
    T, cols, colcap, blocks, pos = (shape[k] for k in ("T", "cols", "colcap", "blocks", "positions"))
    assert T == blocks * pos * 32
    # A: positions x 512 coefficients x 4 limbs, canonical (< p).
    A = rng.integers(0, 1 << 32, size=(pos, 512, 4), dtype=np.uint64).astype(np.uint32)
    top = (A[..., 1] == MASK32) & (A[..., 2] == MASK32) & (A[..., 3] == MASK32)
    A[top, 3] = 0
    pm1 = np.array([0x5808, MASK32, MASK32, MASK32], dtype=np.uint32)  # p - 1
    A[0, 0] = pm1
    A[0, 1] = 0
    A[1, 511] = pm1
    A[pos - 1, 0] = pm1
    A[pos - 1, 255] = 0
    # hot / mask -> code bytes: hot (0..15) if committed else 0xFF; padded to colcap.
    hot = rng.integers(0, 16, size=(T, cols), dtype=np.uint8)
    mask = rng.random((T, cols)) < 0.5
    committed = (hot != 0) | mask
    code = np.full((T, colcap), 0xFF, dtype=np.uint8)
    code[:, :cols] = np.where(committed, hot, 0xFF)
    code[0:4, :] = 0xFF                       # rows with every column uncommitted
    code[T - 1, :cols] = 0                    # last row: hot = 0 with mask bit on every column
    return A, code


def to_int(limbs):
    """(..., 4) uint32 limbs -> object array of Python ints."""
    o = limbs.astype(object)
    return o[..., 0] | (o[..., 1] << 32) | (o[..., 2] << 64) | (o[..., 3] << 96)


def ref_rows(A_ext, code, shape, c, b):
    """Exact R[c][b] (512 Python ints, mod p) via Σ_q Σ_r rot(A[q], s)."""
    pos = shape["positions"]
    acc = np.zeros(512, dtype=object)
    idx = np.arange(512)
    for q in range(pos):
        row0 = (b * pos + q) * 32
        codes = code[row0:row0 + 32, c].astype(np.int64)
        r = np.nonzero(codes < 16)[0]
        if r.size == 0:
            continue
        s = 16 * r + codes[r]
        acc += A_ext[q][(idx[None, :] - s[:, None]) % 1024].sum(axis=0)
    return [int(v) % P for v in acc]


def ref_coeff(A_ext, code, shape, c, b, i):
    pos = shape["positions"]
    row0 = b * pos * 32
    codes = code[row0:row0 + pos * 32, c].astype(np.int64)
    rows = np.nonzero(codes < 16)[0]
    q = rows // 32
    s = 16 * (rows % 32) + codes[rows]
    return int(A_ext[q, (i - s) % 1024].sum()) % P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", default="small", choices=SHAPES)
    ap.add_argument("--variant", default="v2", choices=VARIANTS)
    ap.add_argument("--chunk", type=int, default=128)
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument("--spot", type=int, default=32, help="random coefficients checked at the full shape")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--no-verify", action="store_true")
    args = ap.parse_args()
    shape = SHAPES[args.shape]
    var = VARIANTS[args.variant]
    pos, colcap, blocks = shape["positions"], shape["colcap"], shape["blocks"]
    chunk = min(args.chunk, pos) if var["chunked"] else pos
    num_chunks = pos // chunk
    assert pos % chunk == 0 and colcap % 8 == 0
    params = dict(positions=pos, colcap_vec2=colcap // 8, chunk=chunk, num_chunks=num_chunks, blocks=blocks, colcap=colcap)
    dispatch = [num_chunks, colcap // 8, blocks] if var["chunked"] else [512 // 64, colcap, blocks]
    constants = {"CHUNK": chunk} if var["chunked"] else {}

    t0 = time.time()
    A, code = gen_data(shape, args.seed)
    gen_s = time.time() - t0
    files = {
        "/config.json": ("application/json", json.dumps(dict(variant_file=var["file"], params=params, dispatch=dispatch,
                                                              constants=constants, repeat=args.repeat)).encode()),
        "/a.bin": ("application/octet-stream", A.tobytes()),
        "/hot.bin": ("application/octet-stream", code.tobytes()),
    }

    def route(r):
        path = r.request.url.split("localhost:7777", 1)[1].split("?")[0]
        if path in files:
            ct, body = files[path]
            return r.fulfill(status=200, content_type=ct, body=body)
        f = HERE / path.lstrip("/")
        if f.is_file():
            return r.fulfill(status=200, content_type=mimetypes.guess_type(f.name)[0] or "text/plain", body=f.read_bytes())
        r.fulfill(status=404, body="nope")

    with sync_playwright() as p:
        browser = p.webkit.launch(headless=True)
        page = browser.new_page()
        page.route("http://localhost:7777/**", route)
        page.goto("http://localhost:7777/harness.html")
        out = page.evaluate("window.result")
        browser.close()
    if "error" in out:
        print(json.dumps(out, indent=1))
        sys.exit(1)

    res = np.frombuffer(base64.b64decode(out.pop("result_b64")), dtype=np.uint32).reshape(colcap, blocks, 512, 4)
    summary = dict(shape=args.shape, variant=args.variant, chunk=chunk, dispatch=dispatch, data_gen_s=round(gen_s, 2), **out)

    if not args.no_verify:
        t0 = time.time()
        R = to_int(res)
        A_int = to_int(A)
        A_ext = np.concatenate([A_int, -A_int], axis=1)          # rot(A,s)[i] = A_ext[(i - s) mod 1024]
        canon = bool((R < P).all())
        pad_zero = bool((R[shape["cols"]:] == 0).all())
        mism = 0
        if args.shape == "small":
            checked = shape["cols"] * blocks * 512
            for c in range(shape["cols"]):
                for b in range(blocks):
                    exp = ref_rows(A_ext, code, shape, c, b)
                    mism += sum(1 for i in range(512) if exp[i] != R[c, b, i])
        else:
            rng = np.random.default_rng(args.seed + 1)
            picks = [(int(c), int(b), int(i)) for c, b, i in zip(rng.integers(0, shape["cols"], args.spot),
                                                                   rng.integers(0, blocks, args.spot), rng.integers(0, 512, args.spot))]
            picks += [(0, 0, 0), (0, 0, 511), (shape["cols"] - 1, blocks - 1, 511), (5, 3, 0)]
            checked = len(picks)
            for c, b, i in picks:
                if ref_coeff(A_ext, code, shape, c, b, i) != R[c, b, i]:
                    mism += 1
        summary["correctness"] = dict(checked=checked, mismatches=mism, canonical=canon, padding_zero=pad_zero,
                                      verdict="PASS" if (mism == 0 and canon and pad_zero) else "FAIL", verify_s=round(time.time() - t0, 1))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
