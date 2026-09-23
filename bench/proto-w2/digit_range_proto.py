"""Standalone WebGPU prototype of the Akita stage-8 digit-range sumcheck (b = 8 direct leaf).

    uv run --with playwright --with numpy python bench/proto-w2/digit_range_proto.py \
        [--rounds R ...] [--set full] [--shape small] [--source packed|i8] [--host hash|fixed] \
        [--floor-iters 84] [--mulbench] [--seed S] [--out FILE.json]

Serves harness.html + kernels + data via page.route on http://localhost:7778 in headless WebKit, runs the
R-round dependent chain per instance, checks every message against a Python big-int reference (small shape:
exact per-round messages + folded tables; every shape: sumcheck consistency + final Q check), prints JSON.
"""
import argparse, base64, json, mimetypes, sys, time
from pathlib import Path

import numpy as np
from playwright.sync_api import sync_playwright

HERE = Path(__file__).resolve().parent
PORT = 7778
C = 0xFFFFA7F7
P = (1 << 128) - C
V = [0, 2, 6, 12]
M32 = 0xFFFFFFFF


def fmix(x):
    x ^= x >> 16; x = (x * 0x85EBCA6B) & M32; x ^= x >> 13; x = (x * 0xC2B2AE35) & M32; x ^= x >> 16
    return x


def hash_challenge(words):
    h = [0x9E3779B9, 0x85EBCA6B, 0xC2B2AE35, 0x27D4EB2F]
    for i, w in enumerate(words):
        for k in range(4):
            h[k] = fmix((h[k] ^ w ^ (((i * 4 + k) * 0x9E3779B9) & M32)) & M32)
    h[3] &= 0x7FFFFFFF
    return h


def limbs_to_int(l):
    return l[0] | (l[1] << 32) | (l[2] << 64) | (l[3] << 96)


def int_to_limbs(x):
    return [(x >> (32 * i)) & M32 for i in range(4)]


def eq_table(taus):
    """eq(tau, z) for z in 0..2^len, little-endian (bit t of z <-> taus[t])."""
    t = [1]
    for tau in taus:
        t = [a * (1 - tau) % P for a in t] + [a * tau % P for a in t]
    return t


def poly_mul(a, b):
    out = [0] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            out[i + j] = (out[i + j] + x * y) % P
    return out


def q_coeffs(L, Rt):
    """Coefficients of Q(L + (Rt-L) X), Q(r) = (r^2 - 2r)(r^2 - 18r + 72)."""
    lin = [L % P, (Rt - L) % P]
    sq = poly_mul(lin, lin)
    A = [(sq[0] - 2 * lin[0]) % P, (sq[1] - 2 * lin[1]) % P, sq[2]]
    B = [(sq[0] - 18 * lin[0] + 72) % P, (sq[1] - 18 * lin[1]) % P, sq[2]]
    return poly_mul(A, B)


def lut0():
    out = [0] * 80
    for cp in range(16):
        co = q_coeffs(V[cp & 3], V[cp >> 2])
        for c in range(5):
            v = co[c] if co[c] < P // 2 else co[c] - P
            assert -2**31 <= v < 2**31
            out[c * 16 + cp] = v
    return out


def eq_rounds(tau):
    """Fixed-split Gruen tables per round; returns (concatenated limbs, per-round meta)."""
    R = len(tau)
    split = 1 + (R - 1) // 2
    data, meta, tables = [], [], []
    for k in range(R):
        first = tau[k + 1:split] if k + 1 < split else []
        second = tau[split:] if k + 1 <= split else tau[k + 1:]
        ef, es = eq_table(first), eq_table(second)
        assert len(ef) * len(es) * 2 == (1 << (R - k))
        meta.append(dict(inner_bits=len(first), off_first=len(data), off_second=len(data) + len(ef)))
        data += ef + es
        tables.append((ef, es))
    arr = np.array([int_to_limbs(x) for x in data], dtype=np.uint32)
    return arr, meta, tables


def pack3(w):
    bits = (w.astype(np.int32) + 4).astype(np.uint32).reshape(-1, 8)
    octets = np.zeros(len(bits), dtype=np.uint32)
    for t in range(8):
        octets |= bits[:, t] << (3 * t)
    o = octets.reshape(-1, 4)
    words = np.empty((len(o), 3), dtype=np.uint32)
    words[:, 0] = (o[:, 0] | (o[:, 1] << 24)) & M32
    words[:, 1] = ((o[:, 1] >> 8) | (o[:, 2] << 16)) & M32
    words[:, 2] = ((o[:, 2] >> 16) | (o[:, 3] << 8)) & M32
    return words


def reference_round(table, ef, es, inner_bits):
    inner = 1 << inner_bits
    acc = [0] * 5
    for j in range(len(table) // 2):
        w = ef[j & (inner - 1)] * es[j >> inner_bits] % P
        co = q_coeffs(table[2 * j], table[2 * j + 1])
        for c in range(5):
            acc[c] = (acc[c] + w * co[c]) % P
    return acc


def fold(table, r):
    return [(table[2 * j] + r * (table[2 * j + 1] - table[2 * j])) % P for j in range(len(table) // 2)]


def check_instance(inst, out, tau, tables, meta, v, exact, fixed):
    """Returns dict of verdicts. `v` = range-image ints (None for big shapes)."""
    R = inst["rounds"]
    msgs = [[limbs_to_int(m[4 * c:4 * c + 4]) for c in range(5)] for m in out["msgs"]]
    rs = [limbs_to_int(r) for r in out["challenges"]]
    res = dict(rounds=R, checks={})
    # challenges derived as specified
    ok_ch = all((rs[k] == limbs_to_int(fixed[k])) if fixed else (rs[k] == limbs_to_int(hash_challenge(out["msgs"][k]))) for k in range(R))
    res["checks"]["challenge_derivation"] = ok_ch
    # sumcheck consistency: P_k(X) = s_k * l_k(X) * q_k(X); P_k(0)+P_k(1) == P_{k-1}(r_{k-1}); claim 0.
    def evalp(co, x):
        return sum(c * pow(x, i, P) for i, c in enumerate(co)) % P
    s, prev, ok_sc = 1, 0, True
    for k in range(R):
        l = lambda x, t=tau[k]: ((1 - t) * (1 - x) + t * x) % P
        Pk = lambda x: s * l(x) * evalp(msgs[k], x) % P
        if (Pk(0) + Pk(1)) % P != prev:
            ok_sc = False
        prev = Pk(rs[k])
        s = s * l(rs[k]) % P
    res["checks"]["sumcheck_consistency"] = ok_sc
    final_tab = [limbs_to_int(w) for w in np.frombuffer(base64.b64decode(next(d["b64"] for d in out["dumps"] if d["tag"] == "final_table")), dtype=np.uint32).reshape(2, 4).tolist()]
    final = fold(final_tab, rs[-1])[0]
    Qf = final * (final - 2) % P * (final - 6) % P * (final - 12) % P
    res["checks"]["final_claim"] = (s * Qf % P) == prev   # eq(tau, r) * Q(final) == P_{R-1}(r_{R-1})
    if exact:
        table = list(v)
        mism_msgs, mism_tables = 0, 0
        dumps = {d["tag"]: d["b64"] for d in out["dumps"]}
        for k in range(R):
            ef, es = tables[k]
            ref = reference_round(table, ef, es, meta[k]["inner_bits"])
            if ref != msgs[k]:
                mism_msgs += 1
            tag = f"table_in_r{k}"   # table written by round k = digits folded by r_0..r_{k-1}
            if tag in dumps:
                got = [limbs_to_int(w) for w in np.frombuffer(base64.b64decode(dumps[tag]), dtype=np.uint32).reshape(-1, 4).tolist()]
                if got != table:
                    mism_tables += 1
            if k == R - 1:
                last_in = list(table)   # 2-element table the last round wrote; final fold happens on the host
            table = fold(table, rs[k])
        res["checks"]["exact_messages"] = dict(rounds=R, mismatched=mism_msgs)
        res["checks"]["exact_tables"] = dict(dumped=len([t for t in dumps if t.startswith("table_in")]), mismatched=mism_tables)
        res["checks"]["final_table"] = final_tab == last_in
        mle = sum(a * b for a, b in zip(eq_table(rs), v)) % P
        res["checks"]["final_equals_mle"] = mle == final
    res["verdict"] = "PASS" if all(c is True or (isinstance(c, dict) and c.get("mismatched", 1) == 0) for c in res["checks"].values()) else "FAIL"
    return res


def timing(out):
    """Per-round GPU busy, dependent gaps, phase sums from timestamp passes."""
    ts = out["ts"]
    if not ts:
        return {}
    rounds = {}
    for p in ts:
        k = int(p["tag"][1:].split(":")[0])
        rounds.setdefault(k, []).append(p)
    per = []
    ks = sorted(rounds)
    for k in ks:
        ps = rounds[k]
        busy = sum(p["end"] - p["begin"] for p in ps)
        gap = rounds[k + 1][0]["begin"] - ps[-1]["end"] if k + 1 in rounds else None
        per.append(dict(k=k, busy_ms=round(busy, 4), gap_ms=None if gap is None else round(gap, 4), passes={p["tag"].split(":")[1]: round(p["end"] - p["begin"], 4) for p in ps}))
    def phase(lo, hi):
        return round(sum(p["busy_ms"] for p in per if lo <= p["k"] < hi), 3)
    R = len(ks)
    gaps = [p["gap_ms"] for p in per if p["gap_ms"] is not None]
    span = ts[-1]["end"] - ts[0]["begin"]
    return dict(compact_ms=phase(0, 3), materialize_r3_ms=phase(3, 4), field_ms=phase(4, R), gpu_busy_ms=round(sum(p["busy_ms"] for p in per), 3),
                gap_median_ms=round(sorted(gaps)[len(gaps) // 2], 4), gap_mean_ms=round(sum(gaps) / len(gaps), 4), gap_max_ms=round(max(gaps), 4),
                gpu_span_ms=round(span, 3), inst_wall_ms=out["inst_wall_ms"], host_ms_total=round(sum(out["host_ms"]), 3), per_round=per)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, nargs="*", default=None)
    ap.add_argument("--set", choices=["full"], default=None)
    ap.add_argument("--shape", choices=["small"], default=None)
    ap.add_argument("--source", choices=["packed", "i8"], default="packed")
    ap.add_argument("--host", choices=["hash", "fixed"], default="hash")
    ap.add_argument("--floor-iters", type=int, default=0)
    ap.add_argument("--mulbench", action="store_true")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    rounds = args.rounds or ([24, 21, 20, 19] if args.set == "full" else [])
    if args.shape == "small":
        rounds = [12]
    if not rounds:
        ap.error("nothing to run: pass --rounds R ..., --set full or --shape small")
    exact = args.shape == "small"
    rng = np.random.default_rng(args.seed)

    files, instances, refs = {}, [], []
    t0 = time.time()
    for idx, R in enumerate(rounds):
        assert 12 <= R <= 27
        n = 1 << R
        w = rng.integers(-4, 4, size=n, dtype=np.int8)
        tau = [int(x) for x in rng.integers(0, 1 << 62, size=(R, 2), dtype=np.int64) @ np.array([1, 1 << 62], dtype=object) % P]
        eq_arr, meta, tables = eq_rounds(tau)
        fixed = [int_to_limbs(int(x)) for x in (rng.integers(0, 1 << 62, size=(R, 2), dtype=np.int64) @ np.array([1, 1 << 62], dtype=object) % P)] if args.host == "fixed" else None
        files[f"/d{idx}_i8.bin"] = ("application/octet-stream", w.tobytes())
        files[f"/d{idx}_p3.bin"] = ("application/octet-stream", pack3(w).tobytes())
        files[f"/d{idx}_eq.bin"] = ("application/octet-stream", eq_arr.tobytes())
        instances.append(dict(rounds=R, i8_url=f"/d{idx}_i8.bin", packed_url=f"/d{idx}_p3.bin", eq_url=f"/d{idx}_eq.bin", eq_rounds=meta, lut0=lut0(),
                              packed=args.source == "packed", fixed_challenges=fixed, dump=exact))
        v = [int(x) for x in (w.astype(np.int64) * (w.astype(np.int64) + 1))] if exact else None
        refs.append((tau, tables, meta, v, fixed))
    cfg = dict(instances=instances, floor_iters=args.floor_iters, mulbench=dict(n=1 << 22, k=16, reps=5) if args.mulbench else None)
    files["/config.json"] = ("application/json", json.dumps(cfg).encode())
    gen_s = time.time() - t0

    def route(r):
        path = r.request.url.split(f"localhost:{PORT}", 1)[1].split("?")[0]
        if path in files:
            ct, body = files[path]
            return r.fulfill(status=200, content_type=ct, body=body)
        f = HERE / path.lstrip("/")
        if f.is_file():
            return r.fulfill(status=200, content_type=mimetypes.guess_type(f.name)[0] or "text/plain", body=f.read_bytes())
        r.fulfill(status=404, body="nope")

    t0 = time.time()
    with sync_playwright() as p:
        browser = p.webkit.launch(headless=True)
        page = browser.new_page()
        page.route(f"http://localhost:{PORT}/**", route)
        page.goto(f"http://localhost:{PORT}/harness.html")
        out = page.evaluate("window.result")
        browser.close()
    browser_s = time.time() - t0
    if args.out:
        Path(args.out + ".raw.json").write_text(json.dumps(dict(out=out, refs=[dict(tau=r[0], meta=r[2], v=r[3], fixed=r[4]) for r in refs])))
    if "error" in out or out.get("errors"):
        print(json.dumps(out, indent=1)[:6000])
        sys.exit(1)

    summary = dict(adapter=out["adapter"], hasTs=out["hasTs"], limits=out["limits"], compile_ms={k: v["ms"] for k, v in out["compileInfo"].items()},
                   source=args.source, host=args.host, data_gen_s=round(gen_s, 1), browser_s=round(browser_s, 1), instances=[])
    all_ts = [x for inst in out["instances"] for p in (inst["ts"] or []) for x in (p["begin"], p["end"])]
    if all_ts:
        d = sorted(set(all_ts)); deltas = [b - a for a, b in zip(d, d[1:]) if b - a > 0]
        summary["ts_min_delta_us"] = round(min(deltas) * 1000, 3) if deltas else None
    fail = False
    for inst, (tau, tables, meta, v, fixed), spec in zip(out["instances"], refs, instances):
        chk = check_instance(spec, inst, tau, tables, meta, v, exact, fixed)
        fail |= chk["verdict"] != "PASS"
        summary["instances"].append(dict(rounds=inst["rounds"], upload=inst["upload"], fetch_ms=inst["fetch_ms"], timing=timing(inst), plan=inst["plan"], correctness=chk))
    if out.get("floor"):
        fl = out["floor"]; ts = fl["ts"] or []
        gaps = [b["begin"] - a["end"] for a, b in zip(ts, ts[1:])]
        summary["floor"] = dict(iters=fl["iters"], wall_per_iter_ms=fl["per_iter_ms"], wall_median_ms=round(sorted(fl["walls"])[len(fl["walls"]) // 2], 3),
                                gpu_gap_median_ms=round(sorted(gaps)[len(gaps) // 2], 4) if gaps else None, gpu_gap_max_ms=round(max(gaps), 4) if gaps else None)
    if out.get("mulbench"):
        mb = out["mulbench"]
        # spot-check first 3 outputs against Python
        ok = True
        for i in range(3):
            x = limbs_to_int([(i * 2654435761 + 1) & M32, i ^ 0x9E3779B9, (i * 40503 + 7) & M32, ((i >> 3) + 0x12345) & M32])
            b = limbs_to_int([(i + 3) & M32, (i * 7 + 1) & M32, 0xDEADBEEF ^ i, (i * 31) & 0x7FFFFFFF])
            for _ in range(mb["k"]):
                x = x * b % P
            ok &= limbs_to_int(mb["sample_first64"][4 * i:4 * i + 4]) == x
        summary["mulbench"] = dict(n=mb["n"], k=mb["k"], gpu_ms=[round(t, 3) for t in mb["gpu_ms"]], median_ms=round(mb["median_ms"], 3) if mb["median_ms"] else None, gmul_s=mb["gmul_s"], spot_check=ok)
        fail |= not ok
    summary["verdict"] = "FAIL" if fail else "PASS"
    js = json.dumps(summary, indent=1)
    if args.out:
        Path(args.out).write_text(js)
    # compact print: drop per-round detail unless small
    for i in summary["instances"]:
        if i["rounds"] > 12:
            i["timing"] = {k: v for k, v in i["timing"].items() if k != "per_round"} | {"per_round_gaps_ms": [p["gap_ms"] for p in i["timing"].get("per_round", [])]}
            i.pop("plan", None)
    print(json.dumps(summary, indent=1))
    sys.exit(1 if fail else 0)


if __name__ == "__main__":
    main()
