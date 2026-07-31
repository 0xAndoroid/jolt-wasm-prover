// sha2-chain ladder benchmark: drives frontend/public/worker.js directly
// (no React UI) against http://localhost:8080.
//
// Usage: node bench-chain.mjs [itersCsv] [runsPerScale]
//   itersCsv: comma-separated sha2-chain iteration counts (default targets
//             padded 2^16,2^18,2^20,2^21,2^22)
//   runsPerScale: default 3
//
// Emits one JSON line per completed run to stdout and a summary at the end.

import { chromium } from 'playwright';

const CYCLES_PER_SHA256 = 3396;
const targetIters = (scale) =>
    Math.max(1, Math.round((2 ** scale * 0.9) / CYCLES_PER_SHA256));

const DEFAULT_SCALES = [16, 18, 20, 21, 22];
const itersList = process.argv[2]
    ? process.argv[2].split(',').map((s) => parseInt(s, 10))
    : DEFAULT_SCALES.map(targetIters);
const RUNS = parseInt(process.argv[3] || '3', 10);
const TIMEOUT = 1_800_000;

async function run() {
    const browser = await chromium.launch({
        headless: true,
        channel: process.env.PW_CHANNEL || undefined,
    });
    const page = await browser.newContext().then((c) => c.newPage());
    page.on('console', (msg) => process.stderr.write('[page] ' + msg.text() + '\n'));
    await page.goto('http://localhost:8080', { waitUntil: 'domcontentloaded' });

    const tracing = process.env.BENCH_TRACING !== '0';
    await page.evaluate(async ({ tracing }) => {
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
            data: { numThreads: Math.min(navigator.hardwareConcurrency || 4, 12), tracing },
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
    }, { tracing });
    process.stderr.write(`worker ready, sha2-chain loaded (tracing=${tracing})\n`);

    const results = [];
    for (const iters of itersList) {
        for (let i = 0; i < RUNS; i++) {
            const r = await page.evaluate(
                async ({ iters }) => {
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

                    const verified = window.__bench.wait('verify-done');
                    window.__bench.worker.postMessage({
                        type: 'verify',
                        data: {
                            program: 'sha2-chain',
                            proof: proveMsg.proof,
                            programIo: proveMsg.programIo,
                        },
                    });
                    const verifyMsg = await verified;
                    if (verifyMsg.type === 'error') return { error: verifyMsg.error };

                    return {
                        proveSeconds: proveMsg.elapsed / 1000,
                        verifySeconds: verifyMsg.elapsed / 1000,
                        valid: verifyMsg.valid,
                        numCycles: proveMsg.numCycles,
                        paddedCycles: proveMsg.paddedCycles,
                        proofSize: proveMsg.proofSize,
                        peakMemory: proveMsg.peakMemory,
                    };
                },
                { iters },
            );
            if (r.error) {
                console.log(JSON.stringify({ iters, run: i + 1, error: r.error }));
                process.stderr.write(`iters=${iters} run ${i + 1}: ERROR ${r.error}\n`);
                break;
            }
            const rec = {
                iters,
                run: i + 1,
                log2Padded: Math.log2(r.paddedCycles),
                ...r,
                mhz: r.paddedCycles / r.proveSeconds / 1e6,
            };
            results.push(rec);
            console.log(JSON.stringify(rec));
            process.stderr.write(
                `iters=${iters} run ${i + 1}: prove ${r.proveSeconds.toFixed(2)}s ` +
                `(2^${Math.log2(r.paddedCycles)} padded, ${rec.mhz.toFixed(3)} MHz), ` +
                `verify ${r.verifySeconds.toFixed(2)}s, valid=${r.valid}, ` +
                `peak ${(r.peakMemory / 1024 / 1024).toFixed(0)} MB\n`,
            );
        }
    }

    await browser.close();

    const byIters = {};
    for (const r of results) {
        (byIters[r.iters] ||= []).push(r.proveSeconds);
    }
    const summary = Object.entries(byIters).map(([iters, times]) => {
        const sorted = [...times].sort((a, b) => a - b);
        const rec = results.find((r) => r.iters === +iters);
        return {
            iters: +iters,
            log2Padded: rec.log2Padded,
            runs: times.map((t) => +t.toFixed(2)),
            median: +sorted[Math.floor(sorted.length / 2)].toFixed(2),
            medianMhz: +(rec.paddedCycles / sorted[Math.floor(sorted.length / 2)] / 1e6).toFixed(3),
            verifySeconds: rec.verifySeconds,
            proofSize: rec.proofSize,
        };
    });
    console.log(JSON.stringify({ summary }, null, 1));
}

run().catch((e) => {
    console.error(e);
    process.exit(1);
});
