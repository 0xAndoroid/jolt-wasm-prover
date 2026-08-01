// Zero-dependency headless driver: serves the repro, launches a browser
// BINARY directly (no Playwright/puppeteer), and receives the results via
// the page's ?post=1 beacon.
//
//   node run-standalone.mjs /path/to/Chromium [extra-query]
//   node run-standalone.mjs "$HOME/Library/Caches/ms-playwright/chromium-*/chrome-mac/Chromium.app/Contents/MacOS/Chromium" 'brows=640&repeats=10'
//
// Prints results JSON to stdout. Exit codes: 0 = sweep completed (verdict in
// JSON), 2 = harness failure (browser died / timeout), 3 = page ran but got
// no hardware WebGPU adapter (vacuous — wrong browser build or flags).
import { createServer } from 'node:http';
import { spawn } from 'node:child_process';
import { readFileSync, mkdtempSync, rmSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { tmpdir } from 'node:os';

const bin = process.argv[2];
if (!bin) { console.error('usage: node run-standalone.mjs /path/to/chrome-binary [extra-query]'); process.exit(2); }
const extra = process.argv[3] ? '&' + process.argv[3] : '';
const here = dirname(fileURLToPath(import.meta.url));

let resolveResult;
const resultArrived = new Promise((r) => { resolveResult = r; });

const server = createServer((req, res) => {
  if (req.method === 'POST' && req.url.startsWith('/result')) {
    let body = '';
    req.on('data', (c) => { body += c; });
    req.on('end', () => { res.statusCode = 204; res.end(); resolveResult(JSON.parse(body)); });
    return;
  }
  try {
    const path = req.url.split('?')[0];
    const file = path === '/' ? 'index.html' : path.slice(1);
    if (file.includes('..')) throw new Error('nope');
    res.setHeader('content-type', file.endsWith('.html') ? 'text/html' : 'application/octet-stream');
    res.end(readFileSync(join(here, file)));
  } catch {
    res.statusCode = 404;
    res.end('not found');
  }
});
await new Promise((r) => server.listen(0, '127.0.0.1', r));
const port = server.address().port;

const profile = mkdtempSync(join(tmpdir(), 'bucket-repro-profile-'));
// PWFLAGS=1 adds Playwright's Chromium switch set (notably
// --disable-field-trial-config: no Finch trials — the flag environment the
// production corruption was observed under).
const pwFlags = process.env.PWFLAGS === '1' ? [
  '--disable-field-trial-config',
  '--disable-background-timer-throttling',
  '--disable-backgrounding-occluded-windows',
  '--disable-back-forward-cache',
  '--disable-breakpad',
  '--disable-client-side-phishing-detection',
  '--disable-component-extensions-with-background-pages',
  '--disable-component-update',
  '--disable-default-apps',
  '--disable-dev-shm-usage',
  '--disable-extensions',
  '--disable-features=AcceptCHFrame,AutoExpandDetailsElement,AvoidUnnecessaryBeforeUnloadCheckSync,CertificateTransparencyComponentUpdater,DeferRendererTasksAfterInput,DestroyProfileOnBrowserClose,DialMediaRouteProvider,ExtensionManifestV2Disabled,GlobalMediaControls,HttpsUpgrades,ImprovedCookieControls,LazyFrameLoading,LensOverlay,MediaRouter,PaintHolding,ThirdPartyStoragePartitioning,Translate',
  '--allow-pre-commit-input',
  '--disable-hang-monitor',
  '--disable-ipc-flooding-protection',
  '--disable-popup-blocking',
  '--disable-prompt-on-repost',
  '--disable-renderer-backgrounding',
  '--force-color-profile=srgb',
  '--metrics-recording-only',
  '--password-store=basic',
  '--use-mock-keychain',
  '--no-service-autorun',
  '--export-tagged-pdf',
] : [];
const child = spawn(bin, [
  '--headless=new',
  `--user-data-dir=${profile}`,
  '--no-first-run',
  '--no-default-browser-check',
  '--disable-background-networking',
  ...pwFlags,
  `http://127.0.0.1:${port}/?autorun=1&post=1${extra}`,
], { stdio: ['ignore', 'ignore', 'pipe'] });
child.stderr.on('data', (d) => process.stderr.write('[browser] ' + d));
const died = new Promise((r) => child.on('exit', (code) => r({ died: code })));

const outcome = await Promise.race([
  resultArrived,
  died,
  new Promise((r) => setTimeout(() => r({ timeout: true }), 900_000)),
]);

try { child.kill('SIGKILL'); } catch {}
await died;   // profile files stay busy until the process is really gone
server.close();
server.closeAllConnections?.();
try { rmSync(profile, { recursive: true, force: true }); } catch {}

if (outcome?.died !== undefined || outcome?.timeout) {
  console.error(`FAILED: ${outcome.timeout ? 'timeout' : `browser exited (${outcome.died}) before posting results`}`);
  process.exit(2);
}
console.log(JSON.stringify(outcome, null, 1));
console.error(`VERDICT: ${outcome.verdict}`);
if (outcome.verdict === 'ERROR' || outcome.adapter?.vendor !== 'apple') {
  console.error(`VACUOUS: adapter=${JSON.stringify(outcome.adapter ?? null)} error=${outcome.error ?? ''} — not a hardware Metal adapter`);
  process.exit(3);
}
process.exit(0);
