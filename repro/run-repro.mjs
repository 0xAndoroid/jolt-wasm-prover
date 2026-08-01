// Headless driver for repro/index.html: serves the directory on an
// ephemeral port, opens the page with ?autorun=1 in a REAL Chrome/Chromium
// (Playwright's default headless shell has NO WebGPU — always set
// PW_CHANNEL=chromium|chrome or PW_EXECUTABLE), waits for the sweep, prints
// the results JSON to stdout.
//
//   PW_CHANNEL=chromium node run-repro.mjs [extra-query]
//   PW_EXECUTABLE=/path/to/Chromium node run-repro.mjs 'brows=640&repeats=10'
//
// Exit codes: 0 = sweep completed (verdict inside JSON), 2 = harness failure.
import { chromium } from 'playwright';
import { createServer } from 'node:http';
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const extra = process.argv[2] ? '&' + process.argv[2] : '';

const server = createServer((req, res) => {
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

const browser = await chromium.launch({
  headless: true,
  channel: process.env.PW_CHANNEL || undefined,
  executablePath: process.env.PW_EXECUTABLE || undefined,
});
process.stderr.write(`browser: ${browser.version()}\n`);
const page = await browser.newPage();
page.on('console', (msg) => process.stderr.write('[page] ' + msg.text() + '\n'));
page.on('pageerror', (err) => process.stderr.write('[pageerror] ' + err.message + '\n'));

let result;
try {
  await page.goto(`http://127.0.0.1:${port}/?autorun=1${extra}`, { waitUntil: 'domcontentloaded' });
  await page.waitForFunction(() => window.__REPRO_DONE === true, null, { timeout: 900_000 });
  result = await page.evaluate(() => window.__REPRO_RESULT);
} catch (e) {
  process.stderr.write(`FAILED: ${e.message}\n`);
  await browser.close();
  server.close();
  process.exit(2);
}
await browser.close();
server.close();
console.log(JSON.stringify(result, null, 1));
process.stderr.write(`VERDICT: ${result.verdict}\n`);
