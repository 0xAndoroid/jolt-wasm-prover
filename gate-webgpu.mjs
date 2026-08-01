// WebGPU-arm proof gate: proves+verifies sha2-chain in the browser with the
// webgpu arm ENABLED (device slots forced via minTerms=1) and DISABLED, and
// byte-compares the proofs (SHA-256 in-page). The prover is deterministic
// (pinned natively and re-pinned here by the off/off pair), so byte
// equality is the pass criterion, not just verification.
//
// Usage: node gate-webgpu.mjs [iters] [baseUrl]
//   iters default 17 (padded 2^16); baseUrl default http://localhost:8091
// Requires the dev server on baseUrl (PORT=8091 node server.mjs).

import { chromium } from 'playwright';

const ITERS = parseInt(process.argv[2] || '17', 10);
const BASE = process.argv[3] || 'http://localhost:8091';

async function proveOnce(browser, { webgpu, label, millerCpuFraction = -1, minTermsBytecode = 0 }) {
    const page = await browser.newContext().then((c) => c.newPage());
    page.on('console', (msg) =>
        process.stderr.write(`[${label}] ${msg.text()}\n`),
    );
    await page.goto(BASE, { waitUntil: 'domcontentloaded' });

    const result = await page.evaluate(
        async ({ webgpu, iters, millerCpuFraction, minTermsBytecode }) => {
            const pending = new Map();
            const worker = new Worker('/worker.js', { type: 'module' });
            worker.onmessage = (e) => {
                const resolver = pending.get(e.data.type);
                if (resolver) {
                    pending.delete(e.data.type);
                    resolver(e.data);
                }
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
                    webgpu: webgpu ? { minTerms: 1, millerCpuFraction, minTermsBytecode } : null,
                },
            });
            const init = await initDone;
            if (init.type === 'error') return { error: init.error };

            const files = ['sha2_chain_prover.bin', 'sha2_chain_verifier.bin', 'sha2_chain.elf'];
            const [prover, verifier, elf] = await Promise.all(
                files.map((f) => fetch(`/${f}?gate`).then((r) => r.arrayBuffer())),
            );
            const loaded = wait('program-loaded');
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

            const verified = wait('verify-done');
            worker.postMessage({
                type: 'verify',
                data: {
                    program: 'sha2-chain',
                    proof: proveMsg.proof,
                    programIo: proveMsg.programIo,
                },
            });
            const verifyMsg = await verified;
            if (verifyMsg.type === 'error') return { error: verifyMsg.error };

            const digest = await crypto.subtle.digest('SHA-256', new Uint8Array(proveMsg.proof));
            const hex = [...new Uint8Array(digest)]
                .map((b) => b.toString(16).padStart(2, '0'))
                .join('');
            worker.terminate();
            return {
                webgpuInit: init.webgpu,
                proveSeconds: proveMsg.elapsed / 1000,
                paddedCycles: proveMsg.paddedCycles,
                proofSize: proveMsg.proofSize,
                millerServed: proveMsg.millerServed,
                valid: verifyMsg.valid,
                proofSha256: hex,
            };
        },
        { webgpu, iters: ITERS, millerCpuFraction, minTermsBytecode },
    );
    await page.context().close();
    return result;
}

const browser = await chromium.launch({
    headless: true,
    channel: process.env.PW_CHANNEL || 'chrome',
});

const off1 = await proveOnce(browser, { webgpu: false, label: 'off-1' });
const off2 = await proveOnce(browser, { webgpu: false, label: 'off-2' });
const on = await proveOnce(browser, { webgpu: true, label: 'on' });
// Miller hybrid-fraction sweep: any split serves the identical GT, so both
// extremes must byte-match too (0 = all-device shard, 1 = all-CPU shard).
const onF0 = await proveOnce(browser, { webgpu: true, label: 'on-f0', millerCpuFraction: 0 });
const onF1 = await proveOnce(browser, { webgpu: true, label: 'on-f1', millerCpuFraction: 1 });
// W5-U4: bytecode twin OFF (per-slot min_terms pushed above any real trace),
// everything else on — the twin must be byte-invariant.
const onBc0 = await proveOnce(browser, { webgpu: true, label: 'on-bc0', minTermsBytecode: 1 << 30 });
await browser.close();

const runs = [['off-1', off1], ['off-2', off2], ['on', on], ['on-f0', onF0], ['on-f1', onF1], ['on-bc0', onBc0]];
for (const [label, r] of runs) {
    if (r.error) {
        console.error(`${label}: ERROR ${r.error}`);
        process.exit(2);
    }
    console.error(
        `${label}: padded 2^${Math.log2(r.paddedCycles)}, prove ${r.proveSeconds.toFixed(2)}s, ` +
        `valid=${r.valid}, proof ${r.proofSize}B, sha256 ${r.proofSha256.slice(0, 16)}…` +
        (r.webgpuInit ? ` (gpu-worker ${r.webgpuInit.readyMs.toFixed(0)}ms, warmup ${r.webgpuInit.warmupMs.toFixed(0)}ms, miller ${r.millerServed})` : ''),
    );
}

const deterministic = off1.proofSha256 === off2.proofSha256;
const parity = [on, onF0, onF1, onBc0].every((r) => r.proofSha256 === off1.proofSha256);
const engaged = !!on.webgpuInit;
const millerEngaged = on.millerServed > 0 && onF0.millerServed > 0 && onF1.millerServed > 0;
console.log(JSON.stringify({
    iters: ITERS,
    paddedCycles: off1.paddedCycles,
    deterministic,
    webgpuEngaged: engaged,
    millerServed: { on: on.millerServed, f0: onF0.millerServed, f1: onF1.millerServed },
    byteParity: parity,
    proveSeconds: {
        off: [off1.proveSeconds, off2.proveSeconds],
        on: on.proveSeconds,
        onF0: onF0.proveSeconds,
        onF1: onF1.proveSeconds,
    },
    webgpuInit: on.webgpuInit,
    allValid: runs.every(([, r]) => r.valid),
}));
if (!deterministic) {
    console.error('NOTE: prover nondeterministic in-browser — byte gate not applicable');
    process.exit(engaged && on.valid ? 0 : 1);
}
process.exit(deterministic && parity && engaged && millerEngaged && runs.every(([, r]) => r.valid) ? 0 : 1);
