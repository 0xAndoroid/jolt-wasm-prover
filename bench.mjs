import { chromium } from 'playwright';

const RUNS = parseInt(process.argv[2] || '3', 10);
const TIMEOUT = 120_000;

async function run() {
    const browser = await chromium.launch({
        headless: true,
        channel: process.env.PW_CHANNEL || undefined,
    });
    const context = await browser.newContext();
    const page = await context.newPage();

    page.on('console', msg => {
        const text = msg.text();
        if (text.startsWith('[sha2]')) process.stderr.write(text + '\n');
    });

    await page.goto('http://localhost:8080', { waitUntil: 'domcontentloaded' });

    await page.waitForFunction(
        () => document.getElementById('status')?.classList.contains('ready'),
        { timeout: TIMEOUT },
    );

    const timings = [];
    for (let i = 0; i < RUNS; i++) {
        await page.click('#page-sha2 .prove-btn');

        // The output element only exists once the first log line lands, so
        // count completed proofs instead of clearing the log between runs.
        await page.waitForFunction(
            (n) => {
                const el = document.querySelector('#page-sha2 .output');
                return el && (el.textContent.match(/Proof generated in [\d.]+s/g) || []).length >= n;
            },
            i + 1,
            { timeout: TIMEOUT },
        );
        const seconds = await page.evaluate(() => {
            const m = document.querySelector('#page-sha2 .output')
                .textContent.match(/Proof generated in ([\d.]+)s/g);
            return parseFloat(m[m.length - 1].match(/([\d.]+)s/)[1]);
        });
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
