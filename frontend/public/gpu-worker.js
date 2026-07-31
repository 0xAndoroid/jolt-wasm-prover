// Dedicated GPU worker for the webgpu prover arm: owns the WebGPU device
// and pumps the shared-memory job queue in jolt-kernels' engine.
// Initialized with the prove worker's compiled module + shared memory so
// both sides see one wasm heap. The loop awaits jolt_webgpu_drain (jobs run
// on this worker's event loop, which drives the WebGPU callbacks) and parks
// on Atomics.waitAsync between signals; this worker never blocks.
//
// W4-R reliability layer. During a prove this is the ONLY live event loop
// in the process (the prove worker is blocked inside the wasm call, rayon
// workers are futex-parked), so it is also the process watchdog:
// - The park is BOUNDED (100 ms): a missed wakeup of any origin costs
//   ≤100 ms of staleness instead of a permanent wedge, and every rescue is
//   counted + reported, so a lost-wakeup cause self-diagnoses.
// - A drain that throws (a wasm trap = any engine-side Rust panic, e.g.
//   OOM at the 4 GiB heap cap or a device loss surfacing in read_buffer)
//   is reported with the engine signature and the pump CONTINUES — the old
//   loop exited here, silently wedging every later caller forever.
// - A 5 s watchdog samples jolt_webgpu_introspect() and reports stalls
//   (queued-but-unpumped work, a job stuck/died mid-run, device errors)
//   via console.error and BroadcastChannel('jolt-webgpu-wedge') — the
//   channel reaches the page's main thread, which stays responsive.

import init, {
    jolt_webgpu_signal_addr,
    jolt_webgpu_drain,
    jolt_webgpu_introspect,
} from '/pkg/jolt_wasm_prover.js';

const PARK_TIMEOUT_MS = 100;
const WATCHDOG_MS = 5000;

onmessage = async (e) => {
    const { module, memory } = e.data;
    let idx;
    try {
        await init({ module_or_path: module, memory, thread_stack_size: 4 * 1024 * 1024 });
        idx = jolt_webgpu_signal_addr() >> 2;
        postMessage({ type: 'gpu-ready' });
    } catch (err) {
        console.error('[gpu-worker] init failed', err);
        postMessage({ type: 'gpu-error', error: String(err) });
        return;
    }

    const wedge = new BroadcastChannel('jolt-webgpu-wedge');
    let drainTraps = 0;
    let rescuedWakeups = 0;
    const report = (kind, extra = {}) => {
        let engine;
        try {
            engine = JSON.parse(jolt_webgpu_introspect());
        } catch (err) {
            engine = { introspect_error: String(err) };
        }
        const sig = {
            kind,
            ...extra,
            drainTraps,
            rescuedWakeups,
            signal: Atomics.load(new Int32Array(memory.buffer), idx),
            engine,
            t: Date.now(),
        };
        console.error('[gpu-worker][wedge] ' + JSON.stringify(sig));
        try { wedge.postMessage(sig); } catch { /* page gone */ }
    };

    let prev = null;
    let ticks = 0;
    setInterval(() => {
        ticks++;
        let s;
        try {
            s = JSON.parse(jolt_webgpu_introspect());
        } catch (err) {
            report('introspect-failed', { error: String(err) });
            return;
        }
        // Quiet liveness beacon (every 30 s, channel only): its absence
        // under an overtime run means the GPU worker itself died.
        if (ticks % 6 === 0) {
            try { wedge.postMessage({ kind: 'heartbeat', engine: s, t: Date.now() }); } catch { /* page gone */ }
        }
        const running = s.jobs_started > s.jobs_finished;
        const jobAgeMs = s.current_job_start_ms > 0 ? s.now_ms - s.current_job_start_ms : 0;
        // started > finished with a cleared start stamp = the job's future
        // trapped mid-run (panic=abort stamps nothing on the way out).
        const jobDied = running && s.current_job_start_ms === 0;
        const stuckJob = jobAgeMs > WATCHDOG_MS;
        const pending = s.queue_len !== 0 || running;
        const noProgress = prev !== null
            && s.jobs_finished === prev.jobs_finished
            && s.last_drain_enter_ms === prev.last_drain_enter_ms;
        if (jobDied || stuckJob || (pending && noProgress)) {
            report('stall', { jobDied, stuckJob, jobAgeMs });
        } else if (s.device_lost || (prev !== null && s.device_errors > prev.device_errors)) {
            report('device', {});
        }
        prev = s;
    }, WATCHDOG_MS);

    // Re-create the view each iteration: shared wasm memory can grow,
    // and an old SharedArrayBuffer view does not cover new pages.
    let last = Atomics.load(new Int32Array(memory.buffer), idx);
    for (;;) {
        try {
            await jolt_webgpu_drain();
        } catch (err) {
            drainTraps++;
            report('drain-trap', { error: String(err) });
            postMessage({ type: 'gpu-error', error: String(err) });
            await new Promise((r) => setTimeout(r, Math.min(1000, 100 * drainTraps)));
        }
        const arr = new Int32Array(memory.buffer);
        const cur = Atomics.load(arr, idx);
        if (cur === last) {
            const wait = Atomics.waitAsync(arr, idx, cur, PARK_TIMEOUT_MS);
            if (wait.async) {
                const outcome = await wait.value;
                if (outcome === 'timed-out') {
                    const now = Atomics.load(new Int32Array(memory.buffer), idx);
                    if (now !== cur) {
                        // The signal moved while we were parked on it and no
                        // wakeup arrived — the wedge class this park's bound
                        // exists for. Rescued at the cost of ≤100 ms.
                        rescuedWakeups++;
                        report('lost-wakeup-rescued', { was: cur, now });
                    }
                }
            }
        }
        last = Atomics.load(new Int32Array(memory.buffer), idx);
    }
};
