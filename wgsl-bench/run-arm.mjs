// Runs one benchmark arm end-to-end and writes results/<arm>[-<suite>].json.
//
//   node run-arm.mjs <node-default|node-flagged|chrome-default|chrome-flagged>
//                    [--suite bn254|fp128] [--quick] [--kat-only]
//
// Timed passes take the campaign mkdir-lock (/tmp/jolt-wasm-bench.lock.d);
// --kat-only skips the lock since nothing is timed.

import fs from 'node:fs';
import http from 'node:http';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const LOCK = '/tmp/jolt-wasm-bench.lock.d';
const DAWN_TOGGLES = 'disable_robustness,disable_workgroup_init';
const CHROME_BASE_ARGS = ['--enable-unsafe-webgpu', '--enable-gpu', '--use-angle=metal'];
const CHROME_FLAGGED_ARGS = [
  `--enable-dawn-features=${DAWN_TOGGLES}`,
  '--enable-webgpu-developer-features',
];

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function acquireLock() {
  const start = Date.now();
  let contended = false;
  for (;;) {
    try {
      fs.mkdirSync(LOCK);
      return contended;
    } catch {
      contended = true;
      if (Date.now() - start > 600_000) {
        console.error('WARN: lock timeout after 10min — proceeding, results may be co-run polluted');
        return contended;
      }
      await sleep(5000);
    }
  }
}

function releaseLock() {
  try { fs.rmdirSync(LOCK); } catch {}
}

async function runNode(flagged, jobs) {
  const { create, globals } = await import('webgpu');
  Object.assign(globalThis, globals);
  const gpu = create(flagged ? [`enable-dawn-features=${DAWN_TOGGLES}`] : []);
  const { runJobs } = await import('./executor.mjs');
  return runJobs(gpu, jobs);
}

async function runChrome(flagged, jobs) {
  const { chromium } = await import('playwright');
  const server = http.createServer((req, res) => {
    res.setHeader('content-type', 'text/html');
    res.end('<html><body>bench</body></html>');
  });
  await new Promise((r) => server.listen(0, '127.0.0.1', r));
  const port = server.address().port;
  const args = [...CHROME_BASE_ARGS, ...(flagged ? CHROME_FLAGGED_ARGS : [])];
  // Full Chrome (probe-verified Metal adapter in headless), never the
  // headless-shell. Prefer a complete playwright Chrome-for-Testing install,
  // else fall back to system Chrome — other campaigns prune/reinstall the
  // playwright cache under us mid-run.
  const pwCache = '/Users/andoroid/Library/Caches/ms-playwright';
  const candidates = fs.readdirSync(pwCache)
    .filter((d) => /^chromium-\d+$/.test(d)).sort().reverse()
    .map((d) => `${pwCache}/${d}/chrome-mac-arm64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing`)
    .filter((p) => fs.existsSync(p) && fs.existsSync(path.join(p, '../../Frameworks')));
  const executablePath = candidates[0] ?? '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
  const browser = await chromium.launch({ headless: true, executablePath, args });
  try {
    const page = await browser.newPage();
    page.on('console', (msg) => {
      if (msg.type() === 'error' || msg.type() === 'warning') console.error(`[chrome] ${msg.text()}`);
    });
    await page.goto(`http://127.0.0.1:${port}/`);
    const execSrc = fs.readFileSync(path.join(__dirname, 'executor.mjs'), 'utf8');
    page.setDefaultTimeout(1_200_000);
    return await page.evaluate(async ({ execSrc, jobs }) => {
      const mod = await import('data:text/javascript;charset=utf-8,' + encodeURIComponent(execSrc));
      return await mod.runJobs(navigator.gpu, jobs);
    }, { execSrc, jobs });
  } finally {
    await browser.close();
    server.close();
  }
}

const arm = process.argv[2];
const quick = process.argv.includes('--quick');
const katOnly = process.argv.includes('--kat-only');
const suiteIdx = process.argv.indexOf('--suite');
const suite = suiteIdx >= 0 ? process.argv[suiteIdx + 1] : 'bn254';
if (!['node-default', 'node-flagged', 'chrome-default', 'chrome-flagged'].includes(arm) ||
    !['bn254', 'fp128', 'ec'].includes(suite)) {
  console.error('usage: node run-arm.mjs <node-default|node-flagged|chrome-default|chrome-flagged> [--suite bn254|fp128|ec] [--quick] [--kat-only]');
  process.exit(2);
}
const suiteModules = { bn254: './jobs.mjs', fp128: './fp128-jobs.mjs', ec: './ec-jobs.mjs' };
const { buildJobs, verifyAndRate } = await import(suiteModules[suite]);

const profIdx = process.argv.indexOf('--profile');
const profile = profIdx >= 0 ? process.argv[profIdx + 1] : quick ? 'quick' : 'full';
let jobs = buildJobs(profile);
if (katOnly) jobs = jobs.filter((j) => j.type === 'kat');

const flagged = arm.endsWith('flagged');
const isChrome = arm.startsWith('chrome');

let contended = false;
if (!katOnly) contended = await acquireLock();
let outcome;
try {
  const t0 = Date.now();
  outcome = isChrome ? await runChrome(flagged, jobs) : await runNode(flagged, jobs);
  outcome.armWallMs = Date.now() - t0;
} finally {
  if (!katOnly) releaseLock();
}

if (outcome.error) {
  console.error(`ARM FAILED: ${outcome.error}`);
  process.exit(1);
}

const sw = `${outcome.adapterInfo.vendor} ${outcome.adapterInfo.architecture} ${outcome.adapterInfo.description}`.toLowerCase();
const software = sw.includes('swiftshader') || sw.includes('llvmpipe') || sw.includes('software');
const rows = verifyAndRate(jobs, outcome.results);

const report = {
  arm, suite, profile, quick, katOnly, contendedLock: contended, software,
  adapterInfo: outcome.adapterInfo, hasTs: outcome.hasTs,
  armWallMs: outcome.armWallMs,
  rows,
  raw: outcome.results,
  timestamp: new Date().toISOString(),
};
fs.mkdirSync(path.join(__dirname, 'results'), { recursive: true });
const suffix = (suite === 'bn254' ? '' : `-${suite}`) + (katOnly ? '-kat' : profile !== 'full' ? `-${profile}` : '');
const file = path.join(__dirname, 'results', `${arm}${suffix}.json`);
fs.writeFileSync(file, JSON.stringify(report, null, 1));

console.log(`arm=${arm} software=${software} hasTs=${outcome.hasTs} adapter=${JSON.stringify(outcome.adapterInfo)}`);
for (const r of rows) {
  if (r.error) { console.log(`  ${r.label}: ERROR ${r.error}`); continue; }
  const rate = r.gmulGpu ?? r.gmulWall;
  const bits = [
    r.ok === false ? `FAIL ${r.fails.slice(0, 2).join('; ')}` : 'ok',
    rate !== undefined ? `${rate.toFixed(3)} Gmul/s${r.gmulGpu ? ' (gpu)' : ' (wall)'}` : '',
    r.mops !== undefined && r.mops !== null ? `${r.mops.toFixed(2)} Mops/s` : '',
    r.gbps ? `${r.gbps.toFixed(1)} GB/s` : '',
    r.pipelineMs !== undefined ? `pipe ${r.pipelineMs?.toFixed(0)}ms` : '',
    r.k ? `k=${r.k} d=${r.d}` : '',
    r.singleSubmitMedianMs !== undefined
      ? `dispatch-in-pass ${(r.perDispatchInPassGpuMs ?? r.perDispatchInPassMs).toFixed(3)}ms, submit->done ${r.singleSubmitMedianMs.toFixed(2)}ms (min ${r.singleSubmitMinMs.toFixed(2)})`
      : '',
  ].filter(Boolean).join(' | ');
  console.log(`  ${r.label}: ${bits}`);
}
if (software) {
  console.error('WARN: software adapter — timings are not GPU numbers');
  process.exit(3);
}
console.log(`written: ${file}`);
