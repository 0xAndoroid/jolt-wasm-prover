// W2c: capture Perfetto traces of sha2-chain proves at a given iteration
// count. Drives frontend/public/worker.js directly with tracing enabled;
// per run: clear-trace -> prove -> get-trace -> write trace file.
//
// Usage: node trace-chain.mjs <iters> [runs=2] [outPrefix=trace]
//   writes {outPrefix}-r{N}.json + one JSON summary line to stdout.

import { chromium } from 'playwright';
import { writeFileSync } from 'fs';

const iters = parseInt(process.argv[2], 10);
const RUNS = parseInt(process.argv[3] || '2', 10);
const PREFIX = process.argv[4] || 'trace';
if (!iters) {
    console.error('usage: node trace-chain.mjs <iters> [runs] [outPrefix]');
    process.exit(1);
}

async function run() {
    const browser = await chromium.launch({
        headless: true,
        channel: process.env.PW_CHANNEL || undefined,
    });
    const page = await browser.newContext().then((c) => c.newPage());
    page.on('console', (msg) => process.stderr.write('[page] ' + msg.text() + '\n'));
    await page.goto((process.env.BENCH_BASE || process.env.BENCH_URL || 'http://localhost:8080') + '/bench.html', { waitUntil: 'domcontentloaded' });

    const webgpu = process.env.BENCH_WEBGPU ? JSON.parse(process.env.BENCH_WEBGPU) : null;
    await page.evaluate(async ({ webgpu }) => {
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
        worker.postMessage({
            type: 'init',
            data: { numThreads: Math.min(navigator.hardwareConcurrency || 4, 12), tracing: true, webgpu },
        });
        await initDone;

        const files = ['sha2_chain_prover.bin', 'sha2_chain_verifier.bin', 'sha2_chain.elf'];
        const [prover, verifier, elf] = await Promise.all(
            files.map((f) => fetch(`/${f}?bench`).then((r) => {
                if (!r.ok) throw new Error(`fetch ${f}: ${r.status}`);
                return r.arrayBuffer();
            })),
        );
        const loaded = window.__bench.wait('program-loaded');
        worker.postMessage(
            {
                type: 'load-program',
                data: {
                    program: 'sha2-chain',
                    proverPreprocessing: prover,
                    verifierPreprocessing: verifier,
                    elfBytes: elf,
                },
            },
            [prover, verifier, elf],
        );
        await loaded;
    }, { webgpu });
    process.stderr.write('worker ready (tracing on), sha2-chain loaded\n');

    const walls = [];
    for (let i = 0; i < RUNS; i++) {
        const r = await page.evaluate(
            async ({ iters }) => {
                const cleared = window.__bench.wait('trace-cleared');
                window.__bench.worker.postMessage({ type: 'clear-trace' });
                await cleared;

                const done = window.__bench.wait('prove-done');
                window.__bench.worker.postMessage({
                    type: 'prove',
                    data: {
                        program: 'sha2-chain',
                        input: Array.from(new Uint8Array(32).fill(5)),
                        numIters: iters,
                    },
                });
                const proveMsg = await done;
                if (proveMsg.type === 'error') return { error: proveMsg.error };

                const traced = window.__bench.wait('trace');
                window.__bench.worker.postMessage({ type: 'get-trace' });
                const traceMsg = await traced;

                return {
                    proveSeconds: proveMsg.elapsed / 1000,
                    paddedCycles: proveMsg.paddedCycles,
                    numCycles: proveMsg.numCycles,
                    peakMemory: proveMsg.peakMemory,
                    trace: traceMsg.trace,
                };
            },
            { iters },
        );
        if (r.error) {
            console.error(`run ${i + 1}: ERROR ${r.error}`);
            process.exit(1);
        }
        const file = `${PREFIX}-r${i + 1}.json`;
        writeFileSync(file, r.trace);
        walls.push(r.proveSeconds);
        process.stderr.write(
            `run ${i + 1}: prove ${r.proveSeconds.toFixed(2)}s (2^${Math.log2(r.paddedCycles)}), ` +
            `trace ${(r.trace.length / 1e6).toFixed(1)} MB -> ${file}\n`,
        );
        console.log(JSON.stringify({
            run: i + 1, iters, proveSeconds: r.proveSeconds,
            paddedCycles: r.paddedCycles, peakMemory: r.peakMemory, file,
        }));
    }

    await browser.close();
    console.log(JSON.stringify({ iters, walls, prefix: PREFIX }));
}

run().catch((e) => { console.error(e); process.exit(1); });
