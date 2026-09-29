// sha2-chain ladder benchmark: drives frontend/public/worker.js directly
// (no React UI) against http://localhost:8080.
//
// Usage: node bench-chain.mjs [itersCsv] [runsPerScale]
//   itersCsv: comma-separated sha2-chain iteration counts (default targets
//             padded 2^16,2^18,2^20,2^21)
//   runsPerScale: default 3
//
// Emits one JSON line per completed run to stdout and a summary at the end.

import { chromium, webkit } from 'playwright';

const CYCLES_PER_SHA256 = 3396;
/** @param {number} scale */
const targetIters = (scale) =>
    Math.max(1, Math.round((2 ** scale * 0.9) / CYCLES_PER_SHA256));

// 2^21 is the wasm32 ceiling: a 2^22 trace needs a one-hot polynomial with
// 2^32 coefficients (see README "Protocol").
const DEFAULT_SCALES = [16, 18, 20, 21];
const itersList = process.argv[2]
    ? process.argv[2].split(',').map((s) => Math.trunc(Number(s)))
    : DEFAULT_SCALES.map(targetIters);
const RUNS = Math.trunc(Number(process.argv[3] || '3'));

async function run() {
    const engine = process.env['PW_BROWSER'] === 'webkit' ? webkit : chromium;
    const channel = process.env['PW_CHANNEL'];
    const browser = await engine.launch({
        headless: true,
        ...(engine === chromium && channel ? { channel } : {}),
    });
    const page = await browser.newContext().then((c) => c.newPage());
    page.on('console', (msg) => process.stderr.write('[page] ' + msg.text() + '\n'));
    await page.goto(process.env['BENCH_URL'] || 'http://localhost:8080', { waitUntil: 'domcontentloaded' });

    await page.evaluate(async () => {
        /** @type {Window["__bench"]["pending"]} */
        const pending = new Map();
        const worker = new Worker('/worker.js', { type: 'module' });
        window.__bench = {
            pending, worker,
            wait: (type) => new Promise((resolve) => { pending.set(type, resolve); }),
        };
        worker.onmessage = (/** @type {MessageEvent<import("./frontend/types/runtime").BenchReply>} */ e) => {
            const { type } = e.data;
            const resolver = window.__bench.pending.get(type);
            if (resolver) {
                window.__bench.pending.delete(type);
                resolver(e.data);
            }
            if (type === 'error') {
                for (const r of window.__bench.pending.values()) r({ type: 'error', error: e.data.error });
                window.__bench.pending.clear();
            }
        };

        const initDone = window.__bench.wait('init-done');
        worker.postMessage({
            type: 'init',
            data: { numThreads: Math.min(navigator.hardwareConcurrency || 4, 12) },
        });
        await initDone;

        const files = /** @type {const} */ (['sha2_chain_program.bin', 'sha2_chain.elf']);
        const [program, elf] = await Promise.all(
            files.map((f) => fetch(`/${f}?bench`).then((r) => {
                if (!r.ok) throw new Error(`fetch ${f}: ${r.status}`);
                return r.arrayBuffer();
            })),
        );
        if (!program || !elf) throw new Error("program artifacts missing");
        const loaded = window.__bench.wait('program-loaded');
        worker.postMessage(
            {
                type: 'load-program',
                data: {
                    program: 'sha2-chain',
                    programPreprocessing: program,
                    elfBytes: elf,
                },
            },
            [program, elf],
        );
        await loaded;
    });
    process.stderr.write('worker ready, sha2-chain loaded\n');

    /** @type {{iters: number, run: number, log2Padded: number, paddedCycles: number | null, proveSeconds: number, verifySeconds: number, proofSize: number}[]} */
    const results = [];
    for (const iters of itersList) {
        for (let i = 0; i < RUNS; i++) {
            const r = await page.evaluate(
                async ({ iters: iterations }) => {
                    const done = window.__bench.wait('prove-done');
                    window.__bench.worker.postMessage({
                        type: 'prove',
                        data: {
                            program: 'sha2-chain',
                            input: Array.from(new Uint8Array(32).fill(5)),
                            numIters: iterations,
                        },
                    });
                    const proveMsg = await done;
                    if (proveMsg.type === 'error') return { error: proveMsg.error };
                    if (proveMsg.type !== 'prove-done') throw new Error(`unexpected reply ${proveMsg.type}`);

                    const verified = window.__bench.wait('verify-done');
                    window.__bench.worker.postMessage({
                        type: 'verify',
                        data: {
                            program: 'sha2-chain',
                            proof: proveMsg.proof,
                            programIo: proveMsg.programIo,
                            verifierPreprocessing: proveMsg.verifierPreprocessing,
                        },
                    });
                    const verifyMsg = await verified;
                    if (verifyMsg.type === 'error') return { error: verifyMsg.error };
                    if (verifyMsg.type !== 'verify-done') throw new Error(`unexpected reply ${verifyMsg.type}`);

                    return {
                        proveSeconds: proveMsg.elapsed / 1000,
                        traceSeconds: proveMsg.traceMs / 1000,
                        setupSeconds: proveMsg.setupMs / 1000,
                        proveOnlySeconds: proveMsg.proveMs / 1000,
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
            if ('error' in r) {
                console.log(JSON.stringify({ iters, run: i + 1, error: r.error }));
                process.stderr.write(`iters=${iters} run ${i + 1}: ERROR ${r.error}\n`);
                break;
            }
            const rec = {
                iters,
                run: i + 1,
                log2Padded: Math.log2(r.paddedCycles ?? 0),
                ...r,
                mhz: (r.paddedCycles ?? 0) / r.proveSeconds / 1e6,
            };
            results.push(rec);
            console.log(JSON.stringify(rec));
            process.stderr.write(
                `iters=${iters} run ${i + 1}: prove ${r.proveSeconds.toFixed(2)}s ` +
                `[trace ${r.traceSeconds.toFixed(2)} + setup ${r.setupSeconds.toFixed(2)} + prove ${r.proveOnlySeconds.toFixed(2)}] ` +
                `(2^${Math.log2(r.paddedCycles ?? 0)} padded, ${rec.mhz.toFixed(3)} MHz), ` +
                `verify ${r.verifySeconds.toFixed(2)}s, valid=${r.valid}, ` +
                `peak ${((r.peakMemory ?? 0) / 1024 / 1024).toFixed(0)} MB\n`,
            );
        }
    }

    await browser.close();

    /** @type {Record<number, number[]>} */
    const byIters = {};
    for (const r of results) {
        (byIters[r.iters] ||= []).push(r.proveSeconds);
    }
    const summary = Object.entries(byIters).map(([iters, times]) => {
        const sorted = times.toSorted((a, b) => a - b);
        const rec = results.find((r) => r.iters === +iters);
        const median = sorted[Math.floor(sorted.length / 2)];
        if (!rec || median === undefined) throw new Error("missing benchmark run");
        return {
            iters: +iters,
            log2Padded: rec.log2Padded,
            runs: times.map((t) => +t.toFixed(2)),
            median: +median.toFixed(2),
            medianMhz: +((rec.paddedCycles ?? 0) / median / 1e6).toFixed(3),
            verifySeconds: rec.verifySeconds,
            proofSize: rec.proofSize,
        };
    });
    console.log(JSON.stringify({ summary }, null, 1));
}

try {
    await run();
} catch (e) {
    console.error(e);
    process.exit(1);
}
