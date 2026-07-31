// W2c: CPU-WASM streaming-bandwidth microbench driver. For each thread
// count, spawns a FRESH worker (wasm-bindgen-rayon pools are fixed at init)
// on the bare bench.html page, runs the bench_stream kernels, terminates.
//
// Usage: node bench-stream.mjs [threadsCsv=1,4,8,10] [passes=7]
// Emits one JSON line per (threads, kind, size) to stdout.

import { chromium } from 'playwright';

const THREADS = (process.argv[2] || '1,4,8,10').split(',').map((s) => parseInt(s, 10));
const PASSES = parseInt(process.argv[3] || '7', 10);

// kind -> log2 element counts (u64 kernels: 2^25 = 256MiB/array;
// Fr kernels: 2^23 = 256MiB src). Sizes chosen to sit far outside LLC.
const MATRIX = [
    ['copy_u64', [25]],
    ['sum_u64', [25, 26]],
    ['scale_u64', [25]],
    ['triad_u64', [25]],
    ['fold_fr', [23, 24]],
    ['scale_fr', [23]],
    ['mulchain_fr', [20]],
];

async function run() {
    const browser = await chromium.launch({
        headless: true,
        channel: process.env.PW_CHANNEL || undefined,
    });
    const page = await browser.newContext().then((c) => c.newPage());
    page.on('console', (msg) => process.stderr.write('[page] ' + msg.text() + '\n'));
    await page.goto('http://localhost:8080/bench.html', { waitUntil: 'domcontentloaded' });

    for (const threads of THREADS) {
        await page.evaluate(async ({ threads }) => {
            window.__bench = { pending: new Map() };
            const worker = new Worker('/worker.js', { type: 'module' });
            window.__bench.worker = worker;
            worker.onmessage = (e) => {
                const { type } = e.data;
                const resolver = window.__bench.pending.get(type);
                if (resolver) {
                    window.__bench.pending.delete(type);
                    resolver(e.data);
                }
                if (type === 'error') {
                    for (const [k, r] of window.__bench.pending) r({ type: 'error', error: e.data.error });
                    window.__bench.pending.clear();
                }
            };
            window.__bench.wait = (type) =>
                new Promise((resolve) => window.__bench.pending.set(type, resolve));
            const initDone = window.__bench.wait('init-done');
            worker.postMessage({ type: 'init', data: { numThreads: threads, tracing: false } });
            await initDone;
        }, { threads });
        process.stderr.write(`--- worker up, threads=${threads}\n`);

        for (const [kind, sizes] of MATRIX) {
            for (const log2Len of sizes) {
                const r = await page.evaluate(
                    async ({ kind, log2Len, passes }) => {
                        const done = window.__bench.wait('bench-done');
                        window.__bench.worker.postMessage({
                            type: 'bench-stream',
                            data: { kind, log2Len, passes },
                        });
                        const msg = await done;
                        return msg.type === 'error' ? { error: msg.error } : msg.result;
                    },
                    { kind, log2Len, passes: PASSES },
                );
                if (r.error) {
                    process.stderr.write(`${kind}@2^${log2Len} t=${threads}: ERROR ${r.error}\n`);
                    continue;
                }
                const sorted = [...r.times_ms].sort((a, b) => a - b);
                const med = sorted[Math.floor(sorted.length / 2)];
                const gbs = r.bytes_per_pass / (med / 1000) / 1e9;
                const gmuls = r.fr_muls_per_pass ? r.fr_muls_per_pass / (med / 1000) / 1e9 : null;
                const rec = {
                    threads, kind, log2Len,
                    bytesPerPass: r.bytes_per_pass,
                    medianMs: +med.toFixed(2),
                    minMs: +sorted[0].toFixed(2),
                    gbPerSec: +gbs.toFixed(2),
                    ...(gmuls !== null ? { gmulPerSec: +gmuls.toFixed(4) } : {}),
                    timesMs: r.times_ms.map((t) => +t.toFixed(1)),
                };
                console.log(JSON.stringify(rec));
                process.stderr.write(
                    `${kind}@2^${log2Len} t=${threads}: ${med.toFixed(1)}ms  ${gbs.toFixed(1)} GB/s` +
                    (gmuls !== null ? `  ${gmuls.toFixed(3)} Gmul/s` : '') + '\n',
                );
            }
        }

        await page.evaluate(() => window.__bench.worker.terminate());
        process.stderr.write(`--- worker terminated, threads=${threads}\n`);
    }

    await browser.close();
}

run().catch((e) => { console.error(e); process.exit(1); });
