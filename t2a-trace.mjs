// W3-T2a: traced or untraced sha2-chain prove with the webgpu arm
// configurable — the trace-chain.mjs pattern plus worker webgpu config.
//
// Usage: node t2a-trace.mjs <iters> <arm: off|on> [minTerms=0(default)] \
//          [handoffLen=0(default)] [traceOut=""] [port=8080] [runs=1]
//   stdout: one JSON line per run {wall, paddedCycles, peakMemory, file?}.

import { chromium } from 'playwright';
import { writeFileSync } from 'fs';

const iters = parseInt(process.argv[2], 10);
const arm = process.argv[3] || 'off';
const minTerms = parseInt(process.argv[4] || '0', 10);
const handoffLen = parseInt(process.argv[5] || '0', 10);
const traceOut = process.argv[6] || '';
const port = parseInt(process.argv[7] || '8080', 10);
const runs = parseInt(process.argv[8] || '1', 10);
if (!iters) {
    console.error('usage: node t2a-trace.mjs <iters> <off|on> [minTerms] [handoff] [traceOut] [port] [runs]');
    process.exit(1);
}

const browser = await chromium.launch({
    headless: true,
    channel: process.env.PW_CHANNEL || 'chrome',
});
const page = await browser.newContext().then((c) => c.newPage());
page.on('console', (msg) => process.stderr.write('[page] ' + msg.text() + '\n'));
await page.goto(`http://localhost:${port}/bench.html`, { waitUntil: 'domcontentloaded' });

await page.evaluate(
    async ({ arm, minTerms, handoffLen, tracing }) => {
        window.__b = { pending: new Map() };
        const worker = new Worker('/worker.js', { type: 'module' });
        window.__b.worker = worker;
        worker.onmessage = (e) => {
            const r = window.__b.pending.get(e.data.type);
            if (r) { window.__b.pending.delete(e.data.type); r(e.data); }
            if (e.data.type === 'error') {
                for (const [, r] of window.__b.pending) r({ type: 'error', error: e.data.error });
                window.__b.pending.clear();
            }
        };
        window.__b.wait = (t) => new Promise((res) => window.__b.pending.set(t, res));

        const initDone = window.__b.wait('init-done');
        worker.postMessage({
            type: 'init',
            data: {
                numThreads: Math.min(navigator.hardwareConcurrency || 4, 12),
                tracing,
                webgpu: arm === 'on'
                    ? { minTerms: minTerms || 0, handoffLen: handoffLen || 0 }
                    : null,
            },
        });
        const init = await initDone;
        if (arm === 'on' && !init.webgpu) throw new Error('webgpu init failed');

        const files = ['sha2_chain_prover.bin', 'sha2_chain_verifier.bin', 'sha2_chain.elf'];
        const [prover, verifier, elf] = await Promise.all(
            files.map((f) => fetch(`/${f}?bench`).then((r) => {
                if (!r.ok) throw new Error(`fetch ${f}: ${r.status}`);
                return r.arrayBuffer();
            })),
        );
        const loaded = window.__b.wait('program-loaded');
        worker.postMessage({
            type: 'load-program',
            data: {
                program: 'sha2-chain',
                proverPreprocessing: prover,
                verifierPreprocessing: verifier,
                elfBytes: elf,
            },
        }, [prover, verifier, elf]);
        await loaded;
    },
    { arm, minTerms, handoffLen, tracing: !!traceOut },
);

for (let i = 0; i < runs; i++) {
    const r = await page.evaluate(
        async ({ iters, wantTrace }) => {
            if (wantTrace) {
                const cleared = window.__b.wait('trace-cleared');
                window.__b.worker.postMessage({ type: 'clear-trace' });
                await cleared;
            }
            const done = window.__b.wait('prove-done');
            window.__b.worker.postMessage({
                type: 'prove',
                data: {
                    program: 'sha2-chain',
                    input: Array.from(new Uint8Array(32).fill(5)),
                    numIters: iters,
                },
            });
            const proveMsg = await done;
            if (proveMsg.type === 'error') return { error: proveMsg.error };
            let trace = null;
            if (wantTrace) {
                const traced = window.__b.wait('trace');
                window.__b.worker.postMessage({ type: 'get-trace' });
                trace = (await traced).trace;
            }
            return {
                wall: proveMsg.elapsed / 1000,
                paddedCycles: proveMsg.paddedCycles,
                peakMemory: proveMsg.peakMemory,
                trace,
            };
        },
        { iters, wantTrace: !!traceOut },
    );
    if (r.error) { console.error('ERROR:', r.error); process.exit(1); }
    let file;
    if (traceOut) {
        file = runs > 1 ? `${traceOut}-r${i + 1}.json` : `${traceOut}.json`;
        writeFileSync(file, r.trace);
    }
    console.log(JSON.stringify({
        run: i + 1, arm, minTerms, handoffLen,
        wall: +r.wall.toFixed(2),
        log2: Math.log2(r.paddedCycles),
        peakMemory: r.peakMemory,
        ...(file ? { file } : {}),
    }));
}
await browser.close();
