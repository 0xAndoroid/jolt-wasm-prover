import { chromium } from 'playwright';
import { readFileSync, writeFileSync } from 'fs';

const TIMEOUT = 120_000;

async function run() {
    const browser = await chromium.launch({ headless: true });
    const context = await browser.newContext({ acceptDownloads: true });
    const page = await context.newPage();

    page.on('console', msg => {
        const text = msg.text();
        if (text.startsWith('[sha2]')) process.stderr.write(text + '\n');
    });

    await page.goto('http://localhost:8080', { waitUntil: 'domcontentloaded' });

    await page.waitForFunction(
        () => document.getElementById('status')?.className === 'ready',
        { timeout: TIMEOUT },
    );

    const output = page.locator('#page-sha2 .output');
    await output.evaluate(el => el.textContent = '');
    await page.click('#page-sha2 .prove-btn');

    const result = await output.evaluateHandle(
        (el) => new Promise((resolve) => {
            const obs = new MutationObserver(() => {
                const m = el.textContent.match(/Proof generated in ([\d.]+)s/);
                if (m) { obs.disconnect(); resolve(m[1]); }
            });
            obs.observe(el, { childList: true, characterData: true, subtree: true });
            const m = el.textContent.match(/Proof generated in ([\d.]+)s/);
            if (m) { obs.disconnect(); resolve(m[1]); }
        }),
    );
    const seconds = parseFloat(await result.jsonValue());
    process.stderr.write(`Proof generated in ${seconds.toFixed(2)}s\n`);

    // Click trace download button and capture the download
    const [download] = await Promise.all([
        page.waitForEvent('download', { timeout: 30000 }),
        page.click('#page-sha2 .trace-btn'),
    ]);

    const path = await download.path();
    const trace = readFileSync(path, 'utf-8');
    writeFileSync('trace.json', trace);
    process.stderr.write(`Trace saved to trace.json (${(trace.length / 1024).toFixed(0)} KB)\n`);

    await browser.close();
}

run().catch(e => { console.error(e); process.exit(1); });
