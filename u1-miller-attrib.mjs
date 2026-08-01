// W5-U1 attribution: decompose the miller absorb pool from a Chrome Trace
// Format prove trace (wasm_tracing output). Usage: node u1-miller-attrib.mjs <trace.json>
import { readFileSync } from 'node:fs';

const trace = JSON.parse(readFileSync(process.argv[2], 'utf8'));

// Reconstruct spans per tid with a B/E stack; record parent chain.
const stacks = new Map(); // tid -> stack of open spans
const spans = [];
for (const e of trace) {
  if (e.ph !== 'B' && e.ph !== 'E') continue;
  if (!stacks.has(e.tid)) stacks.set(e.tid, []);
  const st = stacks.get(e.tid);
  if (e.ph === 'B') {
    const span = { name: e.name, tid: e.tid, ts: e.ts, args: e.args || {}, parent: st.length ? st[st.length - 1] : null, children: [] };
    if (span.parent) span.parent.children.push(span);
    st.push(span);
    spans.push(span);
  } else {
    // close the innermost matching open span
    for (let i = st.length - 1; i >= 0; i--) {
      if (st[i].name === e.name) { st[i].end = e.ts; st.splice(i, 1); break; }
    }
  }
}
for (const s of spans) if (s.end === undefined) s.end = s.ts;
const dur = (s) => s.end - s.ts;
const ms = (us) => (us / 1000).toFixed(1);

const ancestors = (s) => { const out = []; for (let p = s.parent; p; p = p.parent) out.push(p.name); return out; };

// Stage classification: nearest ancestor that is a recognizable phase marker.
function classify(s) {
  const chain = ancestors(s);
  if (chain.some(n => /stream_finish_one_hot|commit_consume|commit_hot|stream_feed/.test(n))) return 'st0-commit-absorb';
  if (chain.some(n => /prove_batch|DoryProverState|reduce|JointOpening|opening/i.test(n))) return 'st8-opening';
  return chain[Math.min(2, chain.length - 1)] || chain[0] || 'root';
}

const miller = spans.filter(s => s.name === 'miller_webgpu');
const served = miller.filter(s => dur(s) > 2000); // >2ms = really ran (declines are µs)
const declined = miller.filter(s => dur(s) <= 2000);

console.log(`miller_webgpu spans: ${miller.length} (served ${served.length}, declined/gated ${declined.length})`);
console.log(`declined pair counts: ${[...new Set(declined.map(s => s.args.pairs))].sort((a, b) => a - b).join(',')}`);

// Entry-point classification (prepared vs computed line source):
// multi_pair_g1_setup = qs from SRS window -> prepared; g2_setup / multi_pair = computed.
function lineSource(s) {
  const chain = ancestors(s);
  if (chain.includes('BN254::multi_pair_g1_setup')) return 'prepared';
  if (chain.includes('BN254::multi_pair_g2_setup')) return 'computed-g2setup';
  if (chain.includes('BN254::multi_pair')) return 'computed-multipair';
  return 'unknown';
}

// Per-group accounting.
const groups = new Map();
for (const s of served) {
  const key = `${classify(s)} | ${lineSource(s)} | ${s.args.pairs}p`;
  if (!groups.has(key)) groups.set(key, { n: 0, total: 0, min: Infinity, max: 0, pairs: 0 });
  const g = groups.get(key);
  g.n++; g.total += dur(s); g.min = Math.min(g.min, dur(s)); g.max = Math.max(g.max, dur(s));
  g.pairs += s.args.pairs;
}
console.log('\n== served groups (worker-time = blocked-caller time incl. queue wait) ==');
for (const [k, g] of [...groups.entries()].sort((a, b) => b[1].total - a[1].total)) {
  console.log(`${ms(g.total).padStart(9)} ms  n=${String(g.n).padStart(3)}  med~${ms(g.total / g.n).padStart(7)}  min ${ms(g.min).padStart(6)}  max ${ms(g.max).padStart(7)}  ${k}`);
}

// Wall (union) vs worker-seconds of the served set + queue-depth histogram.
const events = [];
for (const s of served) { events.push([s.ts, 1]); events.push([s.end, -1]); }
events.sort((a, b) => a[0] - b[0] || b[1] - a[1]);
let depth = 0, last = null, union = 0; const depthTime = new Map();
for (const [t, d] of events) {
  if (last !== null && depth > 0) {
    union += t - last;
    depthTime.set(depth, (depthTime.get(depth) || 0) + (t - last));
  }
  depth += d; last = t;
}
const workerSec = served.reduce((a, s) => a + dur(s), 0);
console.log(`\nunion wall of served miller spans: ${ms(union)} ms; worker-time sum: ${ms(workerSec)} ms; avg depth ${(workerSec / union).toFixed(2)}`);
console.log('depth histogram (ms at concurrent-served-depth):');
for (const [d, t] of [...depthTime.entries()].sort((a, b) => a[0] - b[0])) console.log(`  depth ${d}: ${ms(t)} ms`);

// Service-time estimate: within each overlapping cluster, gaps between
// successive END times = device pass service (passes run back-to-back).
const byEnd = [...served].sort((a, b) => a.end - b.end);
const services = [];
for (let i = 1; i < byEnd.length; i++) {
  const gap = byEnd[i].end - byEnd[i - 1].end;
  if (byEnd[i].ts < byEnd[i - 1].end && gap > 0) services.push({ gap, pairs: byEnd[i].args.pairs });
}
const svcByPairs = new Map();
for (const s of services) {
  if (!svcByPairs.has(s.pairs)) svcByPairs.set(s.pairs, []);
  svcByPairs.get(s.pairs).push(s.gap);
}
console.log('\nservice-time estimate (end-to-end gaps while overlapped, by pairs):');
for (const [p, gaps] of [...svcByPairs.entries()].sort((a, b) => a[0] - b[0])) {
  gaps.sort((a, b) => a - b);
  const med = gaps[Math.floor(gaps.length / 2)];
  console.log(`  ${String(p).padStart(5)}p: n=${gaps.length} med ${ms(med)} ms  p25 ${ms(gaps[Math.floor(gaps.length * .25)])}  p75 ${ms(gaps[Math.floor(gaps.length * .75)])}`);
}

// Timeline of served calls (start-ordered) to see the burst structure.
console.log('\ntimeline (served, start-ordered): ts_s dur_ms pairs tid class');
for (const s of [...served].sort((a, b) => a.ts - b.ts)) {
  console.log(`  ${(s.ts / 1e6).toFixed(2).padStart(7)}  ${ms(dur(s)).padStart(8)}  ${String(s.args.pairs).padStart(5)}p  t${s.tid}  ${classify(s)} ${lineSource(s)}`);
}

// Where does the whole prove's time sit? top-level stage spans on tid 1/2.
console.log('\n== top-of-stack big spans (>1s) ==');
for (const s of spans.filter(s => dur(s) > 1e6 && ancestors(s).length <= 2).sort((a, b) => a.ts - b.ts)) {
  console.log(`  ${(s.ts / 1e6).toFixed(2).padStart(7)}s +${(dur(s) / 1e6).toFixed(2)}s  t${s.tid} ${'  '.repeat(ancestors(s).length)}${s.name}`);
}
