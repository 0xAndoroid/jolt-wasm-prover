# Aggregate a Chrome-Trace-Format dump from worker.js `get-trace` (see
# `bench_webgpu.py --dump-trace`): per-span inclusive totals, the per-level
# `stage2_sumcheck` round sequences and the `stage2_plan` / digit-range log
# lines. Spans are paired per (tid, name) LIFO, so per-name inclusive totals
# do not depend on nesting.
#
#   uv run python bench/trace_spans.py TRACE.json [--top 40] [--rounds]

import argparse
import json
from collections import defaultdict


def load(path):
    with open(path) as f:
        t = json.load(f)
    return t if isinstance(t, list) else t["traceEvents"]


def pair_spans(events):
    """Yield (name, tid, start_us, end_us, args) for every B/E pair."""
    open_stack = defaultdict(list)
    for e in events:
        key = (e.get("tid"), e.get("name"))
        if e.get("ph") == "B":
            open_stack[key].append(e)
        elif e.get("ph") == "E":
            stack = open_stack.get(key)
            if stack:
                b = stack.pop()
                yield e["name"], e.get("tid"), b["ts"], e["ts"], b.get("args") or {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--top", type=int, default=40)
    ap.add_argument("--rounds", action="store_true", help="print every sumcheck_round of every stage2_sumcheck")
    args = ap.parse_args()
    events = load(args.trace)
    spans = list(pair_spans(events))

    totals = defaultdict(lambda: [0.0, 0])
    for name, _tid, s, e, _a in spans:
        totals[name][0] += e - s
        totals[name][1] += 1
    print(f"{'span':70} {'total s':>9} {'count':>6}")
    for name, (us, n) in sorted(totals.items(), key=lambda kv: -kv[1][0])[: args.top]:
        print(f"{name[:70]:70} {us / 1e6:9.3f} {n:6}")

    # Log lines (instant events carry their fields in args).
    plans = [e for e in events if e.get("ph") == "i" and (e.get("args") or {}).get("message") in ("stage2_plan", "digit_range_direct_leaf_plan")]
    if plans:
        print("\nlog lines:")
        for e in plans:
            a = dict(e["args"])
            msg = a.pop("message")
            print(f"  {e['ts'] / 1e6:8.3f}s {msg} " + " ".join(f"{k}={v}" for k, v in a.items()))

    # Per-level stage2: rounds nested inside each stage2_sumcheck span (same tid, by time).
    stage2 = sorted([(s, e, tid) for name, tid, s, e, _ in spans if name == "stage2_sumcheck"])
    rounds = [(s, e, tid, a) for name, tid, s, e, a in spans if name == "sumcheck_round"]
    folds = [(s, e, tid) for name, tid, s, e, _ in spans if name in ("RelationRangeImageProver::fold_round", "sumcheck_round_fold")]
    new_spans = [(s, e, tid) for name, tid, s, e, _ in spans if name == "RelationRangeImageProver::new"]
    if stage2:
        print("\nstage2_sumcheck per level (inclusive ms; rounds = sumcheck_round spans inside; new = RelationRangeImageProver::new):")
        grand = 0.0
        eligible = 0.0
        for i, (s, e, tid) in enumerate(stage2):
            inner = sorted((rs, re, a) for rs, re, rtid, a in rounds if rtid == tid and rs >= s and re <= e)
            new_ms = sum(ne - ns for ns, ne, ntid in new_spans if ntid == tid and ns >= s and ne <= e) / 1e3
            nv = len(inner)
            per_round = [(re - rs) / 1e3 for rs, re, _ in inner]
            # eligible = rounds whose live table is >= 2^12 pairs*2 (table_len field) plus setup
            elig = new_ms + sum(ms for (rs, re, a), ms in zip(inner, per_round) if int(a.get("table_len", 0)) > (1 << 12))
            grand += (e - s) / 1e3
            eligible += elig
            print(f"  level {i}: total {(e - s) / 1e3:8.1f} ms  num_vars {nv:3}  new {new_ms:6.1f}  eligible(table>2^12 + new) {elig:7.1f}  rounds: " + " ".join(f"{ms:.1f}" for ms in per_round[:16]) + (" …" if nv > 16 else ""))
            if args.rounds:
                for (rs, re, a), ms in zip(inner, per_round):
                    print(f"      round {a.get('round')} table_len {a.get('table_len')} {ms:.2f} ms")
        print(f"  all levels: {grand:.1f} ms; eligible {eligible:.1f} ms")

    dr = [(s, e, tid) for name, tid, s, e, _ in spans if name == "digit_range_prove"]
    if dr:
        print(f"\ndigit_range_prove: {len(dr)} instances, {sum(e - s for s, e, _ in dr) / 1e3:.1f} ms inclusive (subtract the [gpu] digit range wall for the CPU remainder)")
        for name in ("digit_range_direct_leaf", "digit_range_product_substage", "physical_l2_norm", "digit_range_direct_leaf_fold"):
            if name in totals:
                print(f"  {name}: {totals[name][0] / 1e3:.1f} ms x{totals[name][1]}")


if __name__ == "__main__":
    main()
