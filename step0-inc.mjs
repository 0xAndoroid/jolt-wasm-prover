// W3-T2a step 0: measure the LIVE inc_claim_reduction slot's browser
// per-round latency — device round wall (submit + 64 B readback through the
// SAB/gpu-worker path) and the handoff readback cost — at a given
// handoffLen. Drives worker.js like trace-chain.mjs, with the webgpu arm on
// and tracing enabled, then extracts the webgpu_inc_device_round /
// webgpu_inc_handoff spans plus the IncClaimReduction::prove_round spans
// from the Perfetto trace.
//
// Usage: node step0-inc.mjs <iters> <handoffLen> [minTerms=1] [port=8080]
//   stdout: one JSON line with per-round stats.

import { chromium } from 'playwright';

const iters = parseInt(process.argv[2], 10);
const handoffLen = parseInt(process.argv[3], 10);
const minTerms = parseInt(process.argv[4] || '1', 10);
const port = parseInt(process.argv[5] || '8080', 10);
if (!iters || Number.isNaN(handoffLen)) {
    console.error('usage: node step0-inc.mjs <iters> <handoffLen> [minTerms] [port]');
    process.exit(1);
}

function stats(durs) {
    if (durs.length === 0) return null;
    const s = [...durs].sort((a, b) => a - b);
    return {
        n: s.length,
        min: +s[0].toFixed(3),
        med: +s[Math.floor(s.length / 2)].toFixed(3),
        max: +s[s.length - 1].toFixed(3),
        sum: +durs.reduce((a, b) => a + b, 0).toFixed(3),
    };
}

async function run() {
    const browser = await chromium.launch({
        headless: true,
        channel: process.env.PW_CHANNEL || 'chrome',
    });
    const page = await browser.newContext().then((c) => c.newPage());
    page.on('console', (msg) => process.stderr.write('[page] ' + msg.text() + '\n'));
    await page.goto(`http://localhost:${port}/bench.html`, { waitUntil: 'domcontentloaded' });

    const r = await page.evaluate(
        async ({ iters, minTerms, handoffLen }) => {
            const pending = new Map();
            const worker = new Worker('/worker.js', { type: 'module' });
            worker.onmessage = (e) => {
                const resolver = pending.get(e.data.type);
                if (resolver) { pending.delete(e.data.type); resolver(e.data); }
                if (e.data.type === 'error') {
                    for (const [, r] of pending) r({ type: 'error', error: e.data.error });
                    pending.clear();
                }
            };
            const wait = (type) => new Promise((resolve) => pending.set(type, resolve));

            const initDone = wait('init-done');
            worker.postMessage({
                type: 'init',
                data: {
                    numThreads: Math.min(navigator.hardwareConcurrency || 4, 12),
                    tracing: true,
                    webgpu: { minTerms, handoffLen },
                },
            });
            const init = await initDone;
            if (!init.webgpu) return { error: 'webgpu init failed (CPU-only)' };

            const files = ['sha2_chain_prover.bin', 'sha2_chain_verifier.bin', 'sha2_chain.elf'];
            const [prover, verifier, elf] = await Promise.all(
                files.map((f) => fetch(`/${f}?bench`).then((r) => {
                    if (!r.ok) throw new Error(`fetch ${f}: ${r.status}`);
                    return r.arrayBuffer();
                })),
            );
            const loaded = wait('program-loaded');
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

            const cleared = wait('trace-cleared');
            worker.postMessage({ type: 'clear-trace' });
            await cleared;

            const done = wait('prove-done');
            worker.postMessage({
                type: 'prove',
                data: {
                    program: 'sha2-chain',
                    input: Array.from(new Uint8Array(32).fill(5)),
                    numIters: iters,
                },
            });
            const proveMsg = await done;
            if (proveMsg.type === 'error') return { error: proveMsg.error };

            const traced = wait('trace');
            worker.postMessage({ type: 'get-trace' });
            const traceMsg = await traced;
            return {
                proveSeconds: proveMsg.elapsed / 1000,
                paddedCycles: proveMsg.paddedCycles,
                trace: traceMsg.trace,
            };
        },
        { iters, minTerms, handoffLen },
    );
    await browser.close();
    if (r.error) { console.error('ERROR:', r.error); process.exit(1); }

    // Chrome Trace Format: B/E events per tid. Pair spans by (tid, name) stack.
    const events = JSON.parse(r.trace).traceEvents ?? JSON.parse(r.trace);
    const stacks = new Map();
    const spans = [];
    for (const ev of events) {
        if (ev.ph !== 'B' && ev.ph !== 'E') continue;
        const key = ev.tid;
        if (!stacks.has(key)) stacks.set(key, []);
        const stack = stacks.get(key);
        if (ev.ph === 'B') stack.push(ev);
        else {
            const open = stack.pop();
            if (open) spans.push({ name: open.name, dur: (ev.ts - open.ts) / 1000, args: open.args });
        }
    }
    const byName = (name) => spans.filter((s) => s.name === name);
    const devRounds = byName('webgpu_inc_device_round');
    const handoff = byName('webgpu_inc_handoff');
    const proveRounds = byName('IncClaimReduction::prove_round');
    // Host-tail rounds = prove_round spans NOT containing a device span:
    // device rounds and prove_round are 1:1 while on device, so subtract.
    const out = {
        iters,
        log2: Math.log2(r.paddedCycles),
        minTerms,
        handoffLen,
        proveSeconds: +r.proveSeconds.toFixed(2),
        deviceRounds: stats(devRounds.map((s) => s.dur)),
        deviceRoundsByLen: Object.fromEntries(
            [...new Set(devRounds.map((s) => s.args?.len))].map((len) => [
                len,
                stats(devRounds.filter((s) => s.args?.len === len).map((s) => s.dur)),
            ]),
        ),
        handoff: stats(handoff.map((s) => s.dur)),
        handoffLenActual: handoff[0]?.args?.len,
        incProveRounds: stats(proveRounds.map((s) => s.dur)),
    };
    console.log(JSON.stringify(out));
}

run().catch((e) => { console.error(e); process.exit(1); });
