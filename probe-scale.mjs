// W5-U3 heap probe: drives one sha2-chain prove at a target scale while
// sampling the wasm heap from the PAGE (the prove worker is blocked, the
// page event loop is not): watermark = memory.buffer.byteLength, live =
// the counting allocator's counters read straight out of shared linear
// memory. Samples drain to node continuously so a renderer crash loses at
// most one drain interval. [memprobe] console lines (stage spans +
// allocation sites, emitted by the wasm side) are parsed into events.
//
// Usage: node probe-scale.mjs [iters] [runs]
//   iters: sha2-chain iterations (default 2223 ≈ 2^23 padded)
//   runs:  proves per session (default 1)
// Env: BENCH_BASE (default http://localhost:8080)
//      BENCH_WEBGPU ('' = CPU arm | '1' | JSON config = GPU arm)
//      BENCH_TRACING ('0' disables the tracing layer — keep on: stage
//                     marks ride it)
//      PROBE_POLL_MS (default 100), PROBE_OUT (JSON evidence path)

import { chromium } from 'playwright';
import fs from 'fs';

const iters = parseInt(process.argv[2] || '2223', 10);
const RUNS = parseInt(process.argv[3] || '1', 10);
const POLL_MS = parseInt(process.env.PROBE_POLL_MS || '100', 10);
const OUT = process.env.PROBE_OUT ||
    `.webgpu-lane-evidence/u3-probe-${iters}it-${Date.now()}.json`;
const MB = 1024 * 1024;

const samples = [];   // [dateNowMs, watermarkBytes, liveBytes, peakLiveBytes]
const events = [];    // parsed [memprobe] lines: {t, kind, name, live_mb, wm_mb, raw}
const consoleLines = [];

function parseMemprobe(text) {
    const m = text.match(
        /^\[memprobe\] (\S+) (\S+) live_mb=([\d.]+) wm_mb=([\d.]+) t=(\d+)(.*)$/);
    if (m) {
        events.push({
            t: +m[5], kind: m[1], name: m[2],
            live_mb: +m[3], wm_mb: +m[4], extra: m[6].trim(),
        });
        return;
    }
    if (text.startsWith('[memprobe] OOM')) {
        events.push({ t: Date.now(), kind: 'OOM', name: 'OOM', raw: text });
    }
}

async function run() {
    const browser = await chromium.launch({
        headless: true,
        channel: process.env.PW_CHANNEL || 'chrome',
        args: ['--enable-features=SharedArrayBuffer'],
    });
    const page = await browser.newContext().then((c) => c.newPage());
    page.on('console', (msg) => {
        const text = msg.text();
        consoleLines.push([Date.now(), text]);
        if (text.startsWith('[memprobe]')) parseMemprobe(text);
        process.stderr.write('[page] ' + text + '\n');
    });
    await page.goto(process.env.BENCH_BASE || 'http://localhost:8080',
        { waitUntil: 'domcontentloaded' });

    const tracing = process.env.BENCH_TRACING !== '0';
    const webgpu = !process.env.BENCH_WEBGPU ? null
        : process.env.BENCH_WEBGPU === '1' ? {}
        : JSON.parse(process.env.BENCH_WEBGPU);

    await page.evaluate(async ({ tracing, webgpu, pollMs }) => {
        window.__bench = { pending: new Map() };
        window.__probe = { samples: [], memory: null, ptr: 0 };
        const worker = new Worker('/worker.js', { type: 'module' });
        window.__bench.worker = worker;
        worker.onmessage = (e) => {
            const { type } = e.data;
            if (type === 'init-done' && e.data.memory) {
                window.__probe.memory = e.data.memory;
                window.__probe.ptr = e.data.memCountersPtr >>> 0;
                setInterval(() => {
                    const p = window.__probe;
                    const buf = p.memory.buffer;
                    let live = 0, peak = 0;
                    try {
                        const v = new Uint32Array(buf, p.ptr, 2);
                        live = v[0]; peak = v[1];
                    } catch { /* ptr misaligned or buffer shrunk: keep zeros */ }
                    p.samples.push([Date.now(), buf.byteLength, live, peak]);
                }, pollMs);
            }
            const resolver = window.__bench.pending.get(type);
            if (resolver) {
                window.__bench.pending.delete(type);
                resolver(e.data);
            }
            if (type === 'error') {
                for (const [, r] of window.__bench.pending)
                    r({ type: 'error', error: e.data.error, peakMemory: e.data.peakMemory });
                window.__bench.pending.clear();
            }
        };
        window.__bench.wait = (type) =>
            new Promise((resolve) => window.__bench.pending.set(type, resolve));

        const initDone = window.__bench.wait('init-done');
        worker.postMessage({
            type: 'init',
            data: {
                numThreads: Math.min(navigator.hardwareConcurrency || 4, 12),
                tracing, webgpu,
            },
        });
        await initDone;

        const files = ['sha2_chain_prover.bin', 'sha2_chain_verifier.bin', 'sha2_chain.elf'];
        const [prover, verifier, elf] = await Promise.all(
            files.map((f) => fetch(`/${f}?probe`).then((r) => {
                if (!r.ok) throw new Error(`fetch ${f}: ${r.status}`);
                return r.arrayBuffer();
            })),
        );
        const loaded = window.__bench.wait('program-loaded');
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
    }, { tracing, webgpu, pollMs: POLL_MS });
    process.stderr.write(
        `worker ready (tracing=${tracing}, webgpu=${JSON.stringify(webgpu)}, ` +
        `poll=${POLL_MS}ms), target iters=${iters}\n`);

    // Crash-safe continuous drain of page-side samples into node.
    let draining = true;
    const drainLoop = (async () => {
        while (draining) {
            await new Promise((r) => setTimeout(r, 500));
            try {
                const batch = await page.evaluate(() => {
                    // Belt-and-braces sample here too: page timers can
                    // throttle in headless, CDP evaluates do not.
                    const p = window.__probe;
                    if (p.memory) {
                        const buf = p.memory.buffer;
                        let live = 0, peak = 0;
                        try {
                            const v = new Uint32Array(buf, p.ptr, 2);
                            live = v[0]; peak = v[1];
                        } catch { /* keep zeros */ }
                        p.samples.push([Date.now(), buf.byteLength, live, peak]);
                    }
                    const s = p.samples;
                    p.samples = [];
                    return s;
                });
                samples.push(...batch);
            } catch { draining = false; }
        }
    })();

    const results = [];
    for (let i = 0; i < RUNS; i++) {
        let r;
        try {
            r = await page.evaluate(async ({ iters }) => {
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
                if (proveMsg.type === 'error')
                    return { error: proveMsg.error, peakMemory: proveMsg.peakMemory };

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
                return {
                    proveSeconds: proveMsg.elapsed / 1000,
                    verifySeconds: verifyMsg.elapsed / 1000,
                    valid: verifyMsg.valid,
                    numCycles: proveMsg.numCycles,
                    paddedCycles: proveMsg.paddedCycles,
                    proofSize: proveMsg.proofSize,
                    peakMemory: proveMsg.peakMemory,
                    millerServed: proveMsg.millerServed,
                };
            }, { iters });
        } catch (e) {
            r = { error: `page/renderer: ${e.message}` };
        }
        results.push({ run: i + 1, ...r });
        if (r.error) {
            process.stderr.write(`run ${i + 1}: DEATH — ${r.error} ` +
                `(watermark at death ${(r.peakMemory / MB || 0).toFixed(0)} MB)\n`);
            break;
        }
        process.stderr.write(
            `run ${i + 1}: prove ${r.proveSeconds.toFixed(2)}s ` +
            `(2^${Math.log2(r.paddedCycles).toFixed(2)} padded), ` +
            `verify ${r.verifySeconds.toFixed(2)}s valid=${r.valid}, ` +
            `peak wm ${(r.peakMemory / MB).toFixed(0)} MB, miller ${r.millerServed}\n`);
    }

    // Final drain, then stop.
    await new Promise((r) => setTimeout(r, 700));
    draining = false;
    await drainLoop;
    await browser.close();

    const meta = {
        iters, runs: RUNS, webgpu, pollMs: POLL_MS,
        base: process.env.BENCH_BASE || 'http://localhost:8080',
        when: new Date().toISOString(),
    };
    fs.mkdirSync(OUT.substring(0, OUT.lastIndexOf('/')) || '.', { recursive: true });
    fs.writeFileSync(OUT, JSON.stringify({ meta, results, events, samples }, null, 0));

    // Human summary: stage ledger + peaks.
    const wmPeak = Math.max(0, ...samples.map((s) => s[1]));
    const livePeak = Math.max(0, ...samples.map((s) => s[3]));
    process.stderr.write(`\n=== probe summary (${OUT}) ===\n`);
    process.stderr.write(`samples=${samples.length} ` +
        `wm_peak=${(wmPeak / MB).toFixed(0)}MB live_peak=${(livePeak / MB).toFixed(0)}MB\n`);
    const opens = new Map();
    for (const ev of events) {
        if (ev.kind === 'B') opens.set(ev.name + '@' + ev.t, ev);
        if (ev.kind === 'i' || ev.kind === 'OOM') {
            process.stderr.write(`  [site] ${ev.name} live=${ev.live_mb}MB ` +
                `wm=${ev.wm_mb}MB ${ev.extra || ev.raw || ''}\n`);
        }
    }
    const stages = events.filter((e) => e.kind === 'B' || e.kind === 'E');
    for (let i = 0; i < stages.length; i++) {
        const b = stages[i];
        if (b.kind !== 'B') continue;
        const e = stages.slice(i + 1).find((x) => x.kind === 'E' && x.name === b.name);
        process.stderr.write(
            `  [stage] ${b.name} ${e ? ((e.t - b.t) / 1000).toFixed(1) + 's' : 'UNFINISHED'} ` +
            `live ${b.live_mb}→${e ? e.live_mb : '?'}MB wm ${b.wm_mb}→${e ? e.wm_mb : '?'}MB\n`);
    }
    console.log(JSON.stringify({ meta, results, wmPeakMB: +(wmPeak / MB).toFixed(0), livePeakMB: +(livePeak / MB).toFixed(0) }));
}

run().catch((e) => {
    console.error(e);
    process.exit(1);
});
