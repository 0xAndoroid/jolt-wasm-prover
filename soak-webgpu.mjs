// W4-R soak: N GPU-arm sha2-chain proves at a given scale, counting wedges.
// One worker per BATCH (heap high-water accumulates across runs, matching
// the bench-chain conditions the T1 wedges appeared under); per-run wall
// timeout; the page relays BroadcastChannel('jolt-webgpu-wedge') traffic
// (gpu-worker watchdog signatures + 30 s heartbeats) into the page console,
// which we capture — a wedged run therefore records WHY it wedged.
//
// Usage: node soak-webgpu.mjs <iters> <runs> [batchSize] [baseUrl] [outJsonl]
//   iters: 1112 = padded 2^22, 278 = 2^20
//   default batchSize 5, baseUrl http://localhost:8094, out soak-results.jsonl
// Exit code = number of failed/wedged runs.

import { chromium } from 'playwright';
import fs from 'node:fs';

const ITERS = parseInt(process.argv[2] || '1112', 10);
const RUNS = parseInt(process.argv[3] || '15', 10);
const BATCH = parseInt(process.argv[4] || '5', 10);
const BASE = process.argv[5] || 'http://localhost:8094';
const OUT = process.argv[6] || 'soak-results.jsonl';
const RUN_TIMEOUT_MS = ITERS >= 1000 ? 300_000 : 150_000;
// SOAK_RECYCLE=1: recycle the worker between runs WITHIN a batch — the
// same-page-sequential reliability mode the wave-3 worker-recycle fix
// ships (batch > 1 without it reproduces the run-4-6 device_lost).
const RECYCLE = process.env.SOAK_RECYCLE === '1';

const log = (s) => process.stderr.write(s + '\n');

async function runBatch(browser, batchIndex, runsInBatch, results) {
    const context = await browser.newContext();
    const page = await context.newPage();
    const consoleTail = [];
    const wedgeSignatures = [];
    page.on('console', (msg) => {
        const text = msg.text();
        consoleTail.push(text);
        if (consoleTail.length > 200) consoleTail.shift();
        if (text.includes('[wedge-channel]') || text.includes('[gpu-worker][wedge]')) {
            wedgeSignatures.push(text);
            log(`  !! ${text}`);
        }
    });
    page.on('pageerror', (err) => log(`  [pageerror] ${err.message}`));
    await page.goto(BASE, { waitUntil: 'domcontentloaded' });

    const initOk = await page.evaluate(async () => {
        // Wedge relay: gpu-worker watchdog → page console → harness.
        const chan = new BroadcastChannel('jolt-webgpu-wedge');
        window.__wedgeCount = 0;
        window.__lastHeartbeat = 0;
        chan.onmessage = (m) => {
            if (m.data.kind === 'heartbeat') {
                window.__lastHeartbeat = Date.now();
                return;
            }
            window.__wedgeCount++;
            console.log('[wedge-channel] ' + JSON.stringify(m.data));
        };

        const pending = new Map();
        window.__soak = { pending };
        window.__soak.boot = async () => {
            const s = window.__soak;
            s.pending.clear();
            const worker = new Worker('/worker.js', { type: 'module' });
            s.worker = worker;
            worker.onmessage = (e) => {
                const resolver = s.pending.get(e.data.type);
                if (resolver) {
                    s.pending.delete(e.data.type);
                    resolver(e.data);
                }
                if (e.data.type === 'error') {
                    for (const [, r] of s.pending) r({ type: 'error', error: e.data.error });
                    s.pending.clear();
                }
            };
            const wait = (type) => new Promise((resolve) => s.pending.set(type, resolve));

            const initDone = wait('init-done');
            worker.postMessage({
                type: 'init',
                data: {
                    numThreads: Math.min(navigator.hardwareConcurrency || 4, 12),
                    tracing: false,
                    webgpu: {},
                },
            });
            const init = await initDone;
            if (init.type === 'error') return { error: init.error };
            if (!init.webgpu) return { error: 'webgpu did not initialize (CPU-only)' };

            const files = ['sha2_chain_prover.bin', 'sha2_chain_verifier.bin', 'sha2_chain.elf'];
            const [prover, verifier, elf] = await Promise.all(
                files.map((f) => fetch(`/${f}?soak`).then((r) => r.arrayBuffer())),
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
            const l = await loaded;
            if (l.type === 'error') return { error: l.error };
            return { ok: true };
        };
        // Same-PAGE worker recycle (wave-3 fix): terminates the worker
        // (its gpu-worker dies with it) and boots a fresh one; the wedge
        // relay is page-scoped and survives.
        window.__soak.recycle = async () => {
            const t0 = performance.now();
            window.__soak.worker.terminate();
            const r = await window.__soak.boot();
            return { ...r, ms: Math.round(performance.now() - t0) };
        };
        return window.__soak.boot();
    });
    if (initOk.error) {
        results.push({ batch: batchIndex, run: -1, outcome: 'init-failed', error: initOk.error });
        await context.close();
        return false;
    }

    for (let r = 0; r < runsInBatch; r++) {
        const label = `b${batchIndex}r${r}`;
        const started = Date.now();
        let rec;
        try {
            rec = await page.evaluate(
                async ({ iters, timeoutMs }) => {
                    const { worker, pending } = window.__soak;
                    const wait = (type) =>
                        new Promise((resolve) => pending.set(type, resolve));
                    const input = new Uint8Array(32).fill(7);

                    // Padded-target hint: the tracer reserves its rows vec
                    // once (W5-U3b) — matters at 2^23.
                    const expectedRows = 2 ** Math.ceil(Math.log2(iters * 3396));
                    const proveDone = wait('prove-done');
                    const t0 = performance.now();
                    worker.postMessage({
                        type: 'prove',
                        data: { program: 'sha2-chain', input: Array.from(input), numIters: iters, expectedRows },
                    });
                    const timeout = new Promise((resolve) =>
                        setTimeout(() => resolve({ type: 'timeout' }), timeoutMs),
                    );
                    const proved = await Promise.race([proveDone, timeout]);
                    if (proved.type !== 'prove-done') {
                        return {
                            outcome: proved.type === 'timeout' ? 'WEDGE-timeout' : 'error',
                            error: proved.error || null,
                            wedgeCount: window.__wedgeCount,
                            lastHeartbeatAgoMs:
                                window.__lastHeartbeat ? Date.now() - window.__lastHeartbeat : -1,
                        };
                    }
                    const proveS = (performance.now() - t0) / 1000;

                    const verifyDone = wait('verify-done');
                    worker.postMessage({
                        type: 'verify',
                        data: {
                            program: 'sha2-chain',
                            proof: proved.proof,
                            programIo: proved.programIo,
                        },
                    });
                    const verified = await Promise.race([
                        verifyDone,
                        new Promise((resolve) =>
                            setTimeout(() => resolve({ type: 'timeout' }), 60_000),
                        ),
                    ]);
                    return {
                        outcome:
                            verified.type === 'verify-done' && verified.valid
                                ? 'ok'
                                : 'verify-failed',
                        prove_s: Math.round(proveS * 100) / 100,
                        verify_s:
                            verified.type === 'verify-done'
                                ? Math.round(verified.elapsed) / 1000
                                : null,
                        peakMemoryMB: Math.round(proved.peakMemory / 1048576),
                        millerServed: proved.millerServed,
                        wedgeCount: window.__wedgeCount,
                    };
                },
                { iters: ITERS, timeoutMs: RUN_TIMEOUT_MS },
            );
        } catch (err) {
            rec = { outcome: 'harness-error', error: String(err) };
        }
        rec.batch = batchIndex;
        rec.run = r;
        rec.iters = ITERS;
        rec.wall_s = Math.round((Date.now() - started) / 100) / 10;
        if (rec.outcome !== 'ok') rec.wedgeSignatures = wedgeSignatures.slice(-10);
        results.push(rec);
        fs.appendFileSync(OUT, JSON.stringify(rec) + '\n');
        log(
            `[${label}] ${rec.outcome} prove=${rec.prove_s ?? '-'}s peak=${rec.peakMemoryMB ?? '-'}MB miller=${rec.millerServed ?? '-'} wedgeMsgs=${rec.wedgeCount ?? 0}`,
        );
        if (rec.outcome !== 'ok') {
            // A wedged worker never recovers the run; abandon the batch so
            // remaining runs start on a fresh worker.
            await context.close();
            return false;
        }
        if (RECYCLE && r < runsInBatch - 1) {
            const rr = await page.evaluate(() => window.__soak.recycle());
            if (rr.error) {
                log(`  recycle failed: ${rr.error}`);
                await context.close();
                return false;
            }
            log(`  recycled worker in ${rr.ms} ms`);
        }
    }
    await context.close();
    return true;
}

const browser = await chromium.launch({
    channel: process.env.PW_CHANNEL || 'chrome',
    headless: true,
    args: ['--enable-features=SharedArrayBuffer'],
});
const results = [];
let batchIndex = 0;
for (;;) {
    const doneRuns = results.filter((r) => r.run >= 0).length;
    if (doneRuns >= RUNS || batchIndex > RUNS * 2) break;
    const inBatch = Math.min(BATCH, RUNS - doneRuns);
    log(`=== batch ${batchIndex}: ${inBatch} runs @ iters=${ITERS} ===`);
    await runBatch(browser, batchIndex, inBatch, results);
    batchIndex++;
}
await browser.close();

const ok = results.filter((r) => r.outcome === 'ok').length;
const wedged = results.filter((r) => String(r.outcome).startsWith('WEDGE')).length;
const other = results.length - ok - wedged;
const proveTimes = results.filter((r) => r.prove_s).map((r) => r.prove_s).sort((a, b) => a - b);
const median = proveTimes.length ? proveTimes[proveTimes.length >> 1] : null;
console.log(
    JSON.stringify({
        iters: ITERS,
        runs: results.filter((r) => r.run >= 0).length,
        ok,
        wedged,
        other,
        median_prove_s: median,
        out: OUT,
    }),
);
process.exit(wedged + other);
