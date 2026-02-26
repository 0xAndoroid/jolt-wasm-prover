import { chromium } from 'playwright';

const RUNS = parseInt(process.argv[2] || '3', 10);
const TIMEOUT = 120_000;

async function run() {
    const browser = await chromium.launch({ headless: true });
    const context = await browser.newContext();
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

    const timings = [];
    for (let i = 0; i < RUNS; i++) {
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
        timings.push(seconds);
        process.stderr.write(`  run ${i + 1}: ${seconds.toFixed(2)}s\n`);

        await page.waitForFunction(
            () => !document.querySelector('#page-sha2 .prove-btn').disabled,
            { timeout: TIMEOUT },
        );
    }

    await browser.close();

    const avg = timings.reduce((a, b) => a + b, 0) / timings.length;
    const min = Math.min(...timings);
    const max = Math.max(...timings);
    console.log(JSON.stringify({ runs: timings, avg: +avg.toFixed(3), min: +min.toFixed(3), max: +max.toFixed(3) }));
}

run().catch(e => { console.error(e); process.exit(1); });
