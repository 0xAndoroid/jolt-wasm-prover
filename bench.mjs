import { webkit } from 'playwright';

const RUNS = Math.trunc(Number(process.argv[2] || '3'));
const TIMEOUT = Math.trunc(Number(process.env['BENCH_TIMEOUT'] || '120000'));

async function run() {
    const browser = await webkit.launch({headless: true});
    const context = await browser.newContext();
    const page = await context.newPage();

    page.on('console', msg => {
        const text = msg.text();
        if (text.startsWith('[sha2]')) process.stderr.write(text + '\n');
    });

    await page.goto(process.env['BENCH_URL'] || 'http://localhost:8080', { waitUntil: 'domcontentloaded' });

    await page.waitForFunction(
        () => document.querySelector('#status')?.classList.contains('ready'),
        undefined,
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
            const text = document.querySelector('#page-sha2 .output')?.textContent ?? '';
            const match = [...text.matchAll(/Proof generated in ([\d.]+)s/g)].at(-1)?.[1];
            if (match === undefined) throw new Error('proof timing missing');
            return Number(match);
        });
        timings.push(seconds);
        process.stderr.write(`  run ${i + 1}: ${seconds.toFixed(2)}s\n`);

        await page.waitForFunction(
            () => !document.querySelector('button.prove-btn')?.hasAttribute('disabled'),
            { timeout: TIMEOUT },
        );
    }

    await browser.close();

    const avg = timings.reduce((a, b) => a + b, 0) / timings.length;
    const min = Math.min(...timings);
    const max = Math.max(...timings);
    console.log(JSON.stringify({ runs: timings, avg: +avg.toFixed(3), min: +min.toFixed(3), max: +max.toFixed(3) }));
}

try {
    await run();
} catch (e) {
    console.error(e);
    process.exitCode = 1;
}
