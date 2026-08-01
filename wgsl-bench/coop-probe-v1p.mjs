// W6-V1p coop-Miller step-0 probe: run June's donor coop_pairing.wgsl AS-IS
// under Tint (node-webgpu/Dawn, default toggles = production-faithful) at
// production burst sizes. KAT-gated timing: every timed config first
// byte-compares the f buffer against the arkworks reference exported by
// dory-gpu/examples/coop_probe_export.rs (donor tree, webgpu-e2e-bench).
//
//   node coop-probe-v1p.mjs [--data /tmp/coop-probe-data] [--reps 5] [--kat-only]
//
// Timed passes take the campaign mkdir-lock (/tmp/jolt-wasm-bench.lock.d)
// with dead-owner reclaim + GPU-util gate; the whole GPU phase holds it
// (~2-4 min, under the 10-min etiquette cap). Watchdog per phase: a wedged
// dispatch (the June kernel is a new barrier-dense shape class on this die)
// exits 9 instead of leaving an hour-long zombie.

import fs from 'node:fs';
import path from 'node:path';
import { execSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const args = process.argv.slice(2);
const argOf = (k, d) => {
  const i = args.indexOf(k);
  return i >= 0 ? args[i + 1] : d;
};
const DATA = argOf('--data', '/tmp/coop-probe-data');
const REPS = Number(argOf('--reps', '5'));
const KAT_ONLY = args.includes('--kat-only');
const LOCK = '/tmp/jolt-wasm-bench.lock.d';
const OWNER = `w6-v1p pid=${process.pid}`;

const manifest = JSON.parse(fs.readFileSync(path.join(DATA, 'manifest.json'), 'utf8'));
const loadWords = (name) => {
  const b = fs.readFileSync(path.join(DATA, name));
  return new Uint32Array(b.buffer, b.byteOffset, b.byteLength / 4);
};

// --- campaign lock (u3-bench-lock.sh semantics: dead-owner reclaim, util gate,
// release only what we own, installed only after acquire succeeds) ---
function gpuUtil() {
  try {
    const out = execSync(
      `ioreg -r -c IOAccelerator -d 1 2>/dev/null | grep -o '"Device Utilization %"=[0-9]*' | grep -o '[0-9]*$' | head -1`,
      { encoding: 'utf8' },
    ).trim();
    return out ? Number(out) : 0;
  } catch {
    return 0;
  }
}
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
let lockHeld = false;
function isAlive(pid) {
  try {
    process.kill(pid, 0);
    return true;
  } catch {
    return false;
  }
}
async function acquireLock() {
  const start = Date.now();
  for (;;) {
    try {
      fs.mkdirSync(LOCK);
      fs.writeFileSync(path.join(LOCK, 'owner'), OWNER);
      lockHeld = true;
      break;
    } catch {
      try {
        const holder = fs.readFileSync(path.join(LOCK, 'owner'), 'utf8');
        const pid = Number((holder.match(/pid=(\d+)/) || [])[1]);
        if (pid && !isAlive(pid)) {
          console.error(`lock: dead owner (${holder.trim()}) — reclaiming`);
          fs.rmSync(LOCK, { recursive: true, force: true });
          continue;
        }
        console.error(`lock: held by ${holder.trim()} — waiting`);
      } catch {}
      if (Date.now() - start > 30 * 60_000) throw new Error('lock timeout after 30min');
      await sleep(15_000);
    }
  }
  for (let i = 0; i < 40; i++) {
    const u = gpuUtil();
    if (u < 10) return;
    console.error(`gpu busy (${u}%) — waiting (${i + 1})`);
    await sleep(15_000);
  }
  throw new Error('gpu still busy after util-gate wait');
}
function releaseLock() {
  if (!lockHeld) return;
  try {
    if (fs.readFileSync(path.join(LOCK, 'owner'), 'utf8').includes(`pid=${process.pid}`)) {
      fs.rmSync(LOCK, { recursive: true, force: true });
    }
  } catch {}
  lockHeld = false;
}
process.on('exit', releaseLock);
for (const sig of ['SIGINT', 'SIGTERM']) process.on(sig, () => process.exit(1));

// --- watchdog: one active timer, re-armed per phase ---
let wdTimer = null;
function watchdog(label, ms) {
  if (wdTimer) clearTimeout(wdTimer);
  wdTimer = setTimeout(() => {
    console.error(`WATCHDOG ${label} ${ms}ms — exiting (possible wedged dispatch)`);
    process.exit(9);
  }, ms);
}

const { create, globals } = await import('webgpu');
Object.assign(globalThis, globals);
const gpu = create([]);
const adapter = await gpu.requestAdapter({ powerPreference: 'high-performance' });
if (!adapter) throw new Error('no adapter');
const hasTs = adapter.features.has('timestamp-query');
const device = await adapter.requestDevice({
  requiredFeatures: hasTs ? ['timestamp-query'] : [],
  requiredLimits: {
    maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
    maxBufferSize: adapter.limits.maxBufferSize,
  },
});
let lost = null;
device.lost.then((l) => {
  lost = `${l.reason}: ${l.message}`;
  console.error(`DEVICE LOST: ${lost}`);
});
const tick = typeof device.tick === 'function' ? setInterval(() => device.tick(), 1) : null;

function makeStorage(u32, label, usage = GPUBufferUsage.STORAGE) {
  const byteLen = Math.max(4, u32.byteLength);
  const buf = device.createBuffer({ label, size: byteLen, usage, mappedAtCreation: true });
  new Uint8Array(buf.getMappedRange()).set(new Uint8Array(u32.buffer, u32.byteOffset, u32.byteLength));
  buf.unmap();
  return buf;
}

async function readBack(srcBuf, words) {
  const staging = device.createBuffer({
    size: words * 4,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });
  const enc = device.createCommandEncoder();
  enc.copyBufferToBuffer(srcBuf, 0, staging, 0, words * 4);
  device.queue.submit([enc.finish()]);
  await staging.mapAsync(GPUMapMode.READ);
  const data = new Uint32Array(staging.getMappedRange().slice(0));
  staging.unmap();
  staging.destroy();
  return data;
}

const now = () => performance.now();

// --- pipeline (compile outside the lock; report Tint diagnostics) ---
watchdog('pipeline-compile', 15 * 60_000);
const shader = fs.readFileSync(path.join(DATA, 'coop_module.wgsl'), 'utf8');
device.pushErrorScope('validation');
const t0 = now();
const module = device.createShaderModule({ code: shader });
const moduleMs = now() - t0;
const t1 = now();
const pipeline = await device.createComputePipelineAsync({
  layout: 'auto',
  compute: { module, entryPoint: 'miller_coop' },
});
const pipelineMs = now() - t1;
const valErr = await device.popErrorScope();
let tintMessages = [];
if (module.getCompilationInfo) {
  const ci = await module.getCompilationInfo();
  tintMessages = ci.messages.map((m) => `${m.type} ${m.lineNum}:${m.linePos} ${m.message}`);
}
if (valErr) {
  console.error(`PIPELINE VALIDATION: ${valErr.message}`);
  console.error(tintMessages.join('\n'));
  process.exit(2);
}
console.error(
  `pipeline: module ${moduleMs.toFixed(0)}ms, compile ${pipelineMs.toFixed(0)}ms, tint messages: ${tintMessages.length}`,
);
for (const m of tintMessages) console.error(`  tint: ${m}`);

// --- shared buffers ---
const shared = {
  pMain: makeStorage(loadWords('p_main.bin'), 'p-main'),
  qMain: makeStorage(loadWords('q_main.bin'), 'q-main'),
  prepared: makeStorage(loadWords('prepared_main.bin'), 'prepared-main'),
  opsPrep: makeStorage(loadWords('ops_prep.bin'), 'ops-prep'),
  groupsPrep: makeStorage(loadWords('groups_prep.bin'), 'groups-prep'),
  opsComp: makeStorage(loadWords('ops_comp.bin'), 'ops-comp'),
  groupsComp: makeStorage(loadWords('groups_comp.bin'), 'groups-comp'),
  consts: makeStorage(loadWords('consts.bin'), 'consts'),
  dummy: makeStorage(new Uint32Array([0]), 'dummy'),
};
const expPrep8192 = loadWords('expected_prep_8192.bin');

const tsCtx = hasTs
  ? {
      querySet: device.createQuerySet({ type: 'timestamp', count: 2 }),
      resolve: device.createBuffer({
        size: 16,
        usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
      }),
      staging: device.createBuffer({
        size: 16,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      }),
    }
  : null;

function encodeDispatch(bg, wgX, wgY, withTs) {
  const enc = device.createCommandEncoder();
  const desc = {};
  if (withTs && tsCtx) {
    desc.timestampWrites = {
      querySet: tsCtx.querySet,
      beginningOfPassWriteIndex: 0,
      endOfPassWriteIndex: 1,
    };
  }
  const pass = enc.beginComputePass(desc);
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bg);
  pass.dispatchWorkgroups(wgX, wgY, 1);
  pass.end();
  if (withTs && tsCtx) {
    enc.resolveQuerySet(tsCtx.querySet, 0, 2, tsCtx.resolve, 0);
    enc.copyBufferToBuffer(tsCtx.resolve, 0, tsCtx.staging, 0, 16);
  }
  return enc.finish();
}

async function submitWait(cmd) {
  const t = now();
  device.queue.submit([cmd]);
  await device.queue.onSubmittedWorkDone();
  return now() - t;
}

async function readTsMs() {
  await tsCtx.staging.mapAsync(GPUMapMode.READ);
  const v = new BigUint64Array(tsCtx.staging.getMappedRange().slice(0));
  tsCtx.staging.unmap();
  return Number(v[1] - v[0]) / 1e6;
}

const median = (xs) => {
  const s = [...xs].sort((a, b) => a - b);
  return s[Math.floor(s.length / 2)];
};

// Every prep config with prepMod === n and n <= 8192 maps pair i to
// (p[i], q[i]) — a prefix of the 8192-pair reference, so sweep points are
// KAT-gated for free.
const configs = [
  { name: 'tiny_prep_1', kind: 'prep', n: 1, prepMod: 1, timed: false, wdMs: 120_000 },
  { name: 'kat8_prep', kind: 'prep', n: 8, prepMod: 8, pFile: 'p_kat_prep.bin', expFile: 'expected_kat_prep.bin', timed: false },
  { name: 'kat8_comp', kind: 'comp', n: 8, pFile: 'p_kat_comp.bin', qFile: 'q_kat_comp.bin', expFile: 'expected_kat_comp.bin', timed: false },
  ...[64, 256, 512, 1024, 2048, 4096, 8192].map((n) => ({
    name: `prep_${n}`, kind: 'prep', n, prepMod: n, timed: true,
  })),
  { name: 'prep_32768', kind: 'prep', n: 32768, prepMod: 8192, expFile: 'expected_prep_32768.bin', timed: true },
  ...[512, 2048, 8192].map((n) => ({
    name: `comp_${n}`, kind: 'comp', n, prepMod: 1, expFile: `expected_comp_${n}.bin`, timed: true,
  })),
];

async function runConfig(cfg) {
  const { n, kind } = cfg;
  // Worst plausible rate ~40µs/pair; budget KAT + warmup + reps generously.
  watchdog(cfg.name, cfg.wdMs || Math.max(60_000, n * 0.04 * (REPS + 3) + 30_000));
  const p = cfg.pFile ? makeStorage(loadWords(cfg.pFile), `p-${cfg.name}`) : shared.pMain;
  const q = kind === 'comp' ? (cfg.qFile ? makeStorage(loadWords(cfg.qFile), `q-${cfg.name}`) : shared.qMain) : shared.dummy;
  const prepared = kind === 'prep' ? shared.prepared : shared.dummy;
  const ops = kind === 'prep' ? shared.opsPrep : shared.opsComp;
  const groups = kind === 'prep' ? shared.groupsPrep : shared.groupsComp;
  const nGroups = kind === 'prep' ? manifest.prep.groups : manifest.comp.groups;
  const wgX = Math.min(n, 32768);
  const wgY = Math.ceil(n / wgX);

  const params = device.createBuffer({
    size: 32,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(
    params, 0,
    new Uint32Array([n, manifest.stride, cfg.prepMod, nGroups, wgX, kind === 'comp' ? 1 : 0, 0, 0]),
  );
  const f = device.createBuffer({
    label: `f-${cfg.name}`,
    size: n * 96 * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
  const bg = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: params } },
      { binding: 1, resource: { buffer: f } },
      { binding: 3, resource: { buffer: q } },
      { binding: 5, resource: { buffer: p } },
      { binding: 6, resource: { buffer: shared.consts } },
      { binding: 7, resource: { buffer: prepared } },
      { binding: 8, resource: { buffer: ops } },
      { binding: 9, resource: { buffer: groups } },
    ],
  });

  const res = { name: cfg.name, n, kind };
  // KAT (gates timing)
  const katWall = await submitWait(encodeDispatch(bg, wgX, wgY, false));
  const got = await readBack(f, n * 96);
  const expected = cfg.expFile ? loadWords(cfg.expFile) : expPrep8192.subarray(0, n * 96);
  let mismatch = -1;
  for (let i = 0; i < n * 96; i++) {
    if (got[i] !== expected[i]) {
      mismatch = i;
      break;
    }
  }
  res.kat = mismatch < 0 ? 'pass' : `FAIL word ${mismatch} (pair ${Math.floor(mismatch / 96)}, coeff word ${mismatch % 96})`;
  res.katWallMs = +katWall.toFixed(1);
  console.error(`${cfg.name}: KAT ${res.kat} (${katWall.toFixed(0)}ms)`);
  if (mismatch >= 0) {
    const base = mismatch - (mismatch % 16);
    res.gotSample = Array.from(got.subarray(base, base + 16));
    res.expSample = Array.from(expected.subarray(base, base + 16));
  }

  if (mismatch < 0 && cfg.timed && !KAT_ONLY) {
    await submitWait(encodeDispatch(bg, wgX, wgY, false)); // warmup
    const wallMs = [], gpuMs = [];
    for (let r = 0; r < REPS; r++) {
      wallMs.push(await submitWait(encodeDispatch(bg, wgX, wgY, true)));
      if (tsCtx) gpuMs.push(await readTsMs());
    }
    res.wallMs = wallMs.map((x) => +x.toFixed(2));
    res.gpuMs = tsCtx ? gpuMs.map((x) => +x.toFixed(2)) : null;
    const med = median(tsCtx ? gpuMs : wallMs);
    res.medMs = +med.toFixed(2);
    res.usPerPair = +((med * 1000) / n).toFixed(2);
    console.error(
      `${cfg.name}: med ${res.medMs}ms = ${res.usPerPair}µs/pair (wall med ${median(wallMs).toFixed(1)}ms)`,
    );
  }

  params.destroy();
  f.destroy();
  if (cfg.pFile) p.destroy();
  if (cfg.qFile) q.destroy();
  return res;
}

console.error('acquiring campaign lock…');
await acquireLock();
console.error(`lock acquired (${OWNER}); gpu util ${gpuUtil()}%`);
const phaseDeadline = setTimeout(() => {
  console.error('LOCK BUDGET 9min exceeded — aborting to release the window');
  process.exit(10);
}, 9 * 60_000);

const results = [];
let aborted = null;
try {
  for (const cfg of configs) {
    const r = await runConfig(cfg);
    results.push(r);
    if (r.kat !== 'pass') {
      aborted = `KAT failure at ${cfg.name} — timing not KAT-gated, stopping`;
      console.error(aborted);
      break;
    }
    if (lost) {
      aborted = `device lost: ${lost}`;
      break;
    }
  }
} finally {
  clearTimeout(phaseDeadline);
  if (wdTimer) clearTimeout(wdTimer);
  releaseLock();
}

const out = {
  when: new Date().toISOString(),
  adapter: adapter.info
    ? { vendor: adapter.info.vendor, architecture: adapter.info.architecture, description: adapter.info.description }
    : null,
  hasTs,
  moduleMs: +moduleMs.toFixed(0),
  pipelineMs: +pipelineMs.toFixed(0),
  tintMessages,
  manifest: { stride: manifest.stride, prep: manifest.prep, comp: manifest.comp },
  reps: REPS,
  aborted,
  results,
};
fs.mkdirSync(path.join(__dirname, 'results'), { recursive: true });
fs.writeFileSync(path.join(__dirname, 'results', 'coop-probe-v1p.json'), JSON.stringify(out, null, 2));
console.log(JSON.stringify(out, null, 2));

if (tick) clearInterval(tick);
device.destroy();
process.exit(aborted ? 3 : 0);
