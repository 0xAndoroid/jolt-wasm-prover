// Aggregates results/*.json into the campaign report table and verdicts.

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ARMS = ['node-default', 'node-flagged', 'chrome-default', 'chrome-flagged'];
const KERNEL_ORDER = ['cios16_june', 'cios16_lit', 'lazy15u', 'lazy14u', 'lazy13u', 'lazy13r'];
// mul32 ops per mont-mul: n*(a*b) + n*(q*p) + n*(q computation)
const MUL32 = { cios16_june: 528, cios16_lit: 528, lazy15u: 595, lazy14u: 741, lazy13u: 820, lazy13r: 820 };

const reports = {};
for (const arm of ARMS) {
  const f = path.join(__dirname, 'results', `${arm}.json`);
  if (fs.existsSync(f)) reports[arm] = JSON.parse(fs.readFileSync(f, 'utf8'));
}

const fmt = (x, d = 3) => (x === undefined || x === null ? '—' : x.toFixed(d));

function bestChain(rep, kernel) {
  let best = null;
  for (const r of rep.rows) {
    if (r.type !== 'chain_timed' || !r.label.startsWith(`chain/${kernel}/`)) continue;
    if (r.ok && (!best || (r.gmulGpu ?? r.gmulWall) > (best.gmulGpu ?? best.gmulWall))) best = r;
  }
  return best;
}

console.log('## Chain (dependent-mul throughput), best workgroup size, Gmul/s device (GPU timestamps)\n');
let header = '| kernel | mul32/mul |';
for (const arm of ARMS) header += ` ${arm} |`;
console.log(header);
console.log('|---|---|' + ARMS.map(() => '---|').join(''));
for (const k of KERNEL_ORDER) {
  let line = `| ${k} | ${MUL32[k]} |`;
  for (const arm of ARMS) {
    const rep = reports[arm];
    const b = rep && bestChain(rep, k);
    line += b ? ` ${fmt(b.gmulGpu ?? b.gmulWall)} (wg${b.label.split('wg')[1]}) |` : ' — |';
  }
  console.log(line);
}

console.log('\n## Implied mul32 throughput (rate x mul32-count), chrome-default best\n');
for (const k of KERNEL_ORDER) {
  const b = reports['chrome-default'] && bestChain(reports['chrome-default'], k);
  if (b) console.log(`  ${k}: ${fmt((b.gmulGpu ?? b.gmulWall) * MUL32[k], 0)} G mul32/s`);
}

console.log('\n## Streaming (1 mul per element, N=2^20)\n');
console.log('| kind | arm | Gmul/s | GB/s |');
console.log('|---|---|---|---|');
for (const arm of ARMS) {
  const rep = reports[arm];
  if (!rep) continue;
  for (const r of rep.rows.filter((r) => r.type === 'stream_timed')) {
    console.log(`| ${r.label.split('/')[1]} | ${arm} | ${r.ok ? fmt(r.gmulGpu ?? r.gmulWall) : 'FAIL'} | ${fmt(r.gbps, 1)} |`);
  }
}

console.log('\n## Saturation (lazy13u / cios16_june @ wg128, chrome-default)\n');
if (reports['chrome-default']) {
  for (const r of reports['chrome-default'].rows.filter((r) => r.label.startsWith('sat/') || (r.label.startsWith('chain/') && r.label.includes('wg128')))) {
    if (r.label.includes('lazy13u') || r.label.includes('cios16_june')) {
      const n = r.label.startsWith('sat/') ? r.label.split('/n')[1] : '65536';
      console.log(`  ${r.label.includes('lazy13u') ? 'lazy13u' : 'cios16_june'} n=${n}: ${fmt(r.gmulGpu ?? r.gmulWall)} Gmul/s`);
    }
  }
}

console.log('\n## Dispatch overhead + pipeline compile\n');
for (const arm of ARMS) {
  const rep = reports[arm];
  if (!rep) continue;
  const o = rep.rows.find((r) => r.type === 'overhead');
  const compiles = rep.rows.filter((r) => r.type === 'chain_timed' && r.pipelineMs > 20).map((r) => r.pipelineMs);
  const maxCompile = Math.max(...rep.rows.filter((r) => r.pipelineMs !== undefined).map((r) => r.pipelineMs));
  console.log(`  ${arm}: in-pass dispatch ${fmt((o.perDispatchInPassGpuMs ?? o.perDispatchInPassMs) * 1000, 1)}us, submit->done median ${fmt(o.singleSubmitMedianMs, 2)}ms (min ${fmt(o.singleSubmitMinMs, 2)}), max pipeline compile ${fmt(maxCompile, 0)}ms, wall/arm ${(rep.armWallMs / 1000).toFixed(0)}s${rep.contendedLock ? ' [LOCK CONTENDED]' : ''}`);
}

// Cross-harness bitwise agreement: KAT out arrays (fixed k) and the sat
// n262144 checksums (k clamped to 64 on every arm).
console.log('\n## Cross-harness bitwise agreement\n');
const armList = Object.keys(reports);
const base = reports[armList[0]];
let agree = true;
for (let i = 0; i < base.rows.length; i++) {
  const label = base.rows[i].label;
  if (base.rows[i].type === 'kat') {
    const outs = armList.map((a) => JSON.stringify(reports[a].raw[i].out));
    if (!outs.every((o) => o === outs[0])) { agree = false; console.log(`  MISMATCH kat out: ${label}`); }
  }
  if (label.startsWith('sat/') && label.endsWith('n262144')) {
    const eq = armList.map((a) => `${reports[a].rows[i].checksum} (k=${reports[a].rows[i].k},d=${reports[a].rows[i].d})`);
    const same = new Set(eq).size === 1;
    if (!same) agree = false;
    console.log(`  ${label}: ${same ? 'checksums identical' : 'MISMATCH'} across ${armList.length} arms ${same ? '' : JSON.stringify(eq)}`);
  }
}
console.log(`  KAT outputs bitwise ${agree ? 'IDENTICAL' : 'DIFFER'} across arms: ${armList.join(', ')}`);

console.log('\n## Verdict arithmetic\n');
const bestOverall = Math.max(...armList.map((a) => {
  const r = reports[a].rows.find((r) => r.label === 'sat/cios16_june/n262144');
  return r ? (r.gmulGpu ?? r.gmulWall) : 0;
}));
const c = reports['chrome-default'];
const b16 = bestChain(c, 'cios16_june'), b13 = bestChain(c, 'lazy13u');
console.log(`  13-bit vs 16-bit (chrome-default best): ${fmt(b13.gmulGpu)} / ${fmt(b16.gmulGpu)} = ${fmt(b13.gmulGpu / b16.gmulGpu, 2)}x  (ZPrize M1/M3 saw ~4-5x the other way)`);
console.log(`  Best Tint-path device rate: ${fmt(bestOverall)} Gmul/s (cios16 @ n=262144)`);
console.log(`  vs Metal-native CIOS 1.65: ${fmt(bestOverall / 1.65, 2)}x  |  vs CPU-native all-core 0.59: ${fmt(bestOverall / 0.59, 2)}x`);
console.log(`  GO threshold ~1.0 Gmul/s: ${bestOverall >= 1.0 ? 'GO' : bestOverall >= 0.8 ? 'MARGINAL (within 10-20% of threshold)' : 'NO-GO'}`);
