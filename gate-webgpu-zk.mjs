// WebGPU-arm ZK proof gate: proves+verifies sha2-chain in the browser on a
// zk-feature build (hiding commitments + BlindFold tail), CPU arm and
// webgpu arm, N of each.
//
// UNLIKE gate-webgpu.mjs THERE IS NO byteParity ASSERT: the zk prover is
// randomized by design (fresh OsRng blinds per prove — no seedable rng
// exists in the wire), so proof bytes MUST differ across runs and across
// arms. Asserting byte equality would false-fail; asserting it off is not
// enough either — this gate asserts the INVERSE (hiding-live: every proof
// SHA pairwise distinct), which catches a zk build silently routed to the
// transparent finishes. The cross-arm byte gate for the device commit
// surface lives in jolt-kernels' `webgpu_commit_matches_optimized*` battery
// under `--features webgpu,zk` (deterministic tier-1 rows lockstep).
//
// Pass criteria: all proofs valid, webgpu arm engaged (gpu-worker up,
// miller served), all proof SHAs distinct, proof sizes in a tight envelope.
//
// Usage: node gate-webgpu-zk.mjs [iters] [baseUrl] [runsPerArm]
//   iters default 17 (padded 2^16); baseUrl default http://localhost:8184

import { chromium } from 'playwright';

const ITERS = parseInt(process.argv[2] || '17', 10);
const BASE = process.argv[3] || 'http://localhost:8184';
const RUNS = parseInt(process.argv[4] || '2', 10);

async function proveOnce(browser, { webgpu, label }) {
    const page = await browser.newContext().then((c) => c.newPage());
    page.on('console', (msg) =>
        process.stderr.write(`[${label}] ${msg.text()}\n`),
    );
    await page.goto(BASE, { waitUntil: 'domcontentloaded' });

    const result = await page.evaluate(
        async ({ webgpu, iters }) => {
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
                    webgpu: webgpu ? { minTerms: 1 } : null,
                },
            });
            const init = await initDone;
            if (init.type === 'error') return { error: init.error };

            const files = ['sha2_chain_prover.bin', 'sha2_chain_verifier.bin', 'sha2_chain.elf'];
            const [prover, verifier, elf] = await Promise.all(
                files.map((f) => fetch(`/${f}?zkgate`).then((r) => r.arrayBuffer())),
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
        { webgpu, iters: ITERS },
    );
    await page.context().close();
    return result;
}

const browser = await chromium.launch({
    headless: true,
    channel: process.env.PW_CHANNEL || 'chrome',
});

const runs = [];
for (let i = 1; i <= RUNS; i++) {
    runs.push([`zk-off-${i}`, await proveOnce(browser, { webgpu: false, label: `zk-off-${i}` })]);
}
for (let i = 1; i <= RUNS; i++) {
    runs.push([`zk-on-${i}`, await proveOnce(browser, { webgpu: true, label: `zk-on-${i}` })]);
}
await browser.close();

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

const shas = runs.map(([, r]) => r.proofSha256);
const hidingLive = new Set(shas).size === shas.length;
const onRuns = runs.filter(([label]) => label.startsWith('zk-on'));
const engaged = onRuns.every(([, r]) => !!r.webgpuInit);
const millerEngaged = onRuns.every(([, r]) => r.millerServed > 0);
const sizes = runs.map(([, r]) => r.proofSize);
const sizeEnvelope = (Math.max(...sizes) - Math.min(...sizes)) / Math.max(...sizes);
const allValid = runs.every(([, r]) => r.valid);
console.log(JSON.stringify({
    iters: ITERS,
    paddedCycles: runs[0][1].paddedCycles,
    zk: true,
    allValid,
    hidingLive,
    webgpuEngaged: engaged,
    millerServed: Object.fromEntries(onRuns.map(([label, r]) => [label, r.millerServed])),
    proofSizes: sizes,
    sizeEnvelopePct: +(sizeEnvelope * 100).toFixed(3),
    proveSeconds: Object.fromEntries(runs.map(([label, r]) => [label, +r.proveSeconds.toFixed(2)])),
    webgpuInit: onRuns[0][1].webgpuInit,
}));
process.exit(allValid && hidingLive && engaged && millerEngaged && sizeEnvelope <= 0.01 ? 0 : 1);
