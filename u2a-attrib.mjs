#!/usr/bin/env node
// W5-U2a attribution: reconstruct B/E span tree from a wasm_tracing Chrome-format trace,
// report total/self wall per span name, grouped by prove_stage, for the spartan/glue pool.
import fs from "node:fs";

const file = process.argv[2] ?? "traces-u2b/on-r1.json";
const filter = process.argv[3] ? new RegExp(process.argv[3], "i") : null;
const evs = JSON.parse(fs.readFileSync(file, "utf8"));

// Per-tid stacks; accumulate per (stage, name): {n, total, self}
const stacks = new Map(); // tid -> [{name, ts, child}]
const acc = new Map(); // key "stage\x00name" -> {n,total,self}
let stage = "pre";
const stageWall = new Map(); // stage -> wall

function bump(stage, name, total, self) {
  const k = stage + "\x00" + name;
  const a = acc.get(k) ?? { n: 0, total: 0, self: 0 };
  a.n++; a.total += total; a.self += self;
  acc.set(k, a);
}

for (const e of evs) {
  if (e.ph === "B") {
    if (/^prove_stage/.test(e.name)) stage = e.name;
    let st = stacks.get(e.tid);
    if (!st) stacks.set(e.tid, (st = []));
    st.push({ name: e.name, ts: e.ts, child: 0, stage });
  } else if (e.ph === "E") {
    const st = stacks.get(e.tid);
    if (!st || !st.length) continue;
    // unwind to matching name (tolerate dropped events)
    let idx = st.length - 1;
    while (idx >= 0 && st[idx].name !== e.name) idx--;
    if (idx < 0) continue;
    const frames = st.splice(idx);
    const f = frames[0];
    const total = e.ts - f.ts;
    bump(f.stage, f.name, total, total - f.child);
    if (st.length) st[st.length - 1].child += total;
    if (/^prove_stage/.test(f.name)) stageWall.set(f.name, (stageWall.get(f.name) ?? 0) + total);
  }
}

const rows = [...acc.entries()].map(([k, v]) => {
  const [stage, name] = k.split("\x00");
  return { stage, name, n: v.n, total: v.total / 1e6, self: v.self / 1e6 };
});

console.log("== stage walls ==");
for (const [s, w] of [...stageWall.entries()].sort())
  console.log(`${s}  ${(w / 1e6).toFixed(2)}s`);

console.log("\n== spans (total desc; total>=50ms) ==");
rows
  .filter((r) => r.total >= 0.05 && !/^prove_stage|^jolt_prover|^engine::prove$/.test(r.name))
  .filter((r) => !filter || filter.test(r.name))
  .sort((a, b) => b.total - a.total)
  .slice(0, 60)
  .forEach((r) =>
    console.log(
      `${r.total.toFixed(3).padStart(8)}s total ${r.self.toFixed(3).padStart(8)}s self  n=${String(r.n).padStart(4)}  [${r.stage}] ${r.name}`,
    ),
  );
