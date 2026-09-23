import init, {
    initThreadPool,
    init_tracing,
    get_trace_json,
    clear_trace,
    WasmProver,
    WasmVerifier,
    gpu_mailbox_ptr,
    gpu_selftest,
    gpu_is_dead,
    set_gpu_enabled,
    set_gpu_disabled,
    set_gpu_commit_enabled,
    set_gpu_digit_range_enabled,
    set_gpu_op_timeout_ms,
    set_gpu_unavailable,
    set_digit_range_parity_rounds,
    set_gpu_stage2_enabled,
    set_relation_range_parity_rounds,
    gpu_trip_probe,
} from '/pkg/jolt_wasm_prover.js';

// Akita backend kernels recurse deeply on rayon workers (64 MiB stacks
// natively); wasm-bindgen sizes worker stacks from this value (multiple of
// 64 KiB).
const THREAD_STACK_SIZE = 32 * 1024 * 1024;
const SCHEDULES_URL = '/akita_schedules.bin';

let wasmExports = null;
let scheduleArtifacts = null;
let gpuProxy = null;
const provers = {};

async function fetchBytes(url) {
    const r = await fetch(url);
    if (!r.ok) throw new Error(`fetch ${url}: ${r.status}`);
    return new Uint8Array(await r.arrayBuffer());
}

// The prover blocks its threads, so the GPUDevice lives in gpu-proxy.js and
// talks to Rust through a mailbox in the shared wasm memory (src/gpu/).
// Resolves to {status: 'ok' | 'unavailable' | 'error: …', reason?, adapter?,
// selftestMs, roundtripUs}; the self-test runs once per session. Never throws.
async function initGpu() {
    if (gpu_is_dead()) return { status: 'unavailable', reason: DEAD_PROXY_REASON };
    if (gpuProxy) {
        set_gpu_enabled();
        return selftest();
    }
    const mailboxPtr = gpu_mailbox_ptr();
    if (mailboxPtr === 0) {
        set_gpu_unavailable();
        return { status: 'unavailable', reason: 'wasm built without the webgpu feature' };
    }
    if (typeof navigator.gpu === 'undefined') {
        set_gpu_unavailable();
        return { status: 'unavailable', reason: 'navigator.gpu missing (WebGPU unsupported or insecure context)' };
    }
    gpuProxy = new Worker('/gpu-proxy.js', { type: 'module' });
    const report = await new Promise((resolve) => {
        gpuProxy.onmessage = (e) => resolve(e.data);
        gpuProxy.onerror = (e) => resolve({ type: 'unavailable', reason: e.message || 'gpu-proxy failed to load' });
        gpuProxy.postMessage({ type: 'init', memory: wasmExports.memory, mailboxPtr });
    });
    if (report.type !== 'ready') {
        gpuProxy.terminate();
        gpuProxy = null;
        set_gpu_unavailable();
        return { status: 'unavailable', reason: report.reason };
    }
    set_gpu_enabled();
    return { ...selftest(), adapter: report.adapter, features: report.features, limits: report.limits };
}

function selftest() {
    const st = JSON.parse(gpu_selftest());
    const gpu = { status: st.status, selftestMs: st.selftest_ms, roundtripUs: st.roundtrip_us };
    if (st.status !== 'ok') {
        set_gpu_unavailable();
        gpu.reason = `GPU self-test failed: ${st.status}`;
    }
    return gpu;
}

// A mailbox op timed out: proving stays on the CPU for the rest of the session.
// The proxy is orphaned, not terminated: WebKit crashes the whole page when a
// worker that owns a GPUDevice is terminated (headless WebKit, Sep 2026).
const DEAD_PROXY_REASON = 'GPU proxy unresponsive — switched to CPU';
function dropDeadProxy() {
    if (!gpuProxy || !gpu_is_dead()) return null;
    gpuProxy = null;
    set_gpu_unavailable();
    return { status: 'unavailable', reason: DEAD_PROXY_REASON };
}

self.onmessage = async (e) => {
    const { type, data } = e.data;

    try {
        switch (type) {
            case 'init': {
                if (typeof SharedArrayBuffer === 'undefined') {
                    throw new Error('Your browser does not support SharedArrayBuffer (requires iOS 15.2+, Chrome 91+, Firefox 79+, Safari 15.2+).');
                }
                const [exports, schedules] = await Promise.all([
                    init({ module_or_path: '/pkg/jolt_wasm_prover_bg.wasm', thread_stack_size: THREAD_STACK_SIZE }),
                    fetchBytes(`${SCHEDULES_URL}${data.cacheBust ? `?${data.cacheBust}` : ''}`),
                ]);
                wasmExports = exports;
                scheduleArtifacts = schedules;
                await initThreadPool(data.numThreads);
                init_tracing();
                set_gpu_op_timeout_ms(data.gpuTimeoutMs ?? 30000);
                // gpu: true | false | 'w1' (GPU on, digit-range rounds on the CPU) | 'w2' (GPU on, trace commit on the CPU)
                //      | 's2off' (W1 + W2 on, stage-2 rounds on the CPU). Only `true` enables the stage-2 device.
                const gpu = data.gpu ? await initGpu() : { status: 'disabled' };
                set_gpu_commit_enabled(data.gpu !== 'w2');
                set_gpu_digit_range_enabled(data.gpu !== 'w1');
                set_gpu_stage2_enabled(data.gpu === true);
                set_digit_range_parity_rounds(data.parityRounds ?? 0);
                set_relation_range_parity_rounds(data.parityRounds ?? 0);
                self.postMessage({ type: 'init-done', gpu });
                break;
            }

            case 'set-gpu': {
                let gpu;
                if (data.enabled) {
                    gpu = await initGpu();
                } else {
                    set_gpu_disabled();
                    gpu = { status: 'disabled' };
                }
                self.postMessage({ type: 'gpu-status', gpu });
                break;
            }

            // Bench probe: `n` dependent one-dispatch RUN_SEQ trips with a 96 B readback each.
            case 'gpu-trip-probe': {
                self.postMessage({ type: 'gpu-trip-probe-done', msPerTrip: gpu_trip_probe(data.n) });
                break;
            }

            // Test hook (bench/test_ui_modes.py): the proxy stops answering the mailbox.
            case 'hang-gpu-proxy': {
                gpuProxy?.postMessage({ type: 'hang' });
                break;
            }

            case 'load-program': {
                const name = data.program;
                provers[name] = new WasmProver(
                    scheduleArtifacts,
                    new Uint8Array(data.programPreprocessing),
                    new Uint8Array(data.elfBytes)
                );
                self.postMessage({ type: 'program-loaded', program: name });
                break;
            }

            case 'prove': {
                const prover = provers[data.program];
                const start = performance.now();
                let result;

                switch (data.program) {
                    case 'sha2':
                        result = prover.prove_sha2(new Uint8Array(data.input));
                        break;
                    case 'ecdsa':
                        result = prover.prove_ecdsa(
                            BigUint64Array.from(data.z.map(BigInt)),
                            BigUint64Array.from(data.r.map(BigInt)),
                            BigUint64Array.from(data.s.map(BigInt)),
                            BigUint64Array.from(data.q.map(BigInt)),
                        );
                        break;
                    case 'keccak':
                        result = prover.prove_keccak_chain(
                            new Uint8Array(data.input),
                            data.numIters
                        );
                        break;
                    case 'sha2-chain':
                        result = prover.prove_sha2_chain(
                            new Uint8Array(data.input),
                            data.numIters
                        );
                        break;
                }

                const elapsed = performance.now() - start;
                const peakMemory = wasmExports.memory.buffer.byteLength;

                self.postMessage({
                    type: 'prove-done',
                    program: data.program,
                    proof: result.proof,
                    proofSize: result.proof_size,
                    programIo: result.program_io,
                    verifierPreprocessing: result.verifier_preprocessing,
                    numCycles: result.num_cycles,
                    paddedCycles: result.padded_cycles,
                    traceMs: result.trace_ms,
                    setupMs: result.setup_ms,
                    proveMs: result.prove_ms,
                    gpuStatus: result.gpu_status,
                    gpuSelftestMs: result.gpu_selftest_ms,
                    gpuSelftestMismatches: result.gpu_selftest_mismatches,
                    gpuRoundtripUs: result.gpu_roundtrip_us,
                    gpuCommit: result.gpu_commit,
                    gpuDigitRange: result.gpu_digit_range,
                    gpuStage2: result.gpu_stage2,
                    peakMemory,
                    elapsed,
                });
                break;
            }

            case 'verify': {
                // The Akita verifier setup is exact in the proof shape, so the
                // verifier is built from the preprocessing the prover emitted
                // for this proof (a real deployment pins it per program+shape).
                const start = performance.now();
                const verifier = new WasmVerifier(new Uint8Array(data.verifierPreprocessing));
                const valid = verifier.verify(data.proof, data.programIo);
                const elapsed = performance.now() - start;

                self.postMessage({
                    type: 'verify-done',
                    program: data.program,
                    valid,
                    elapsed,
                });
                break;
            }

            case 'get-trace': {
                const traceJson = get_trace_json();
                self.postMessage({
                    type: 'trace',
                    trace: traceJson,
                });
                break;
            }

            case 'clear-trace': {
                clear_trace();
                self.postMessage({ type: 'trace-cleared' });
                break;
            }
        }
    } catch (err) {
        const msg = err.message || String(err);
        console.error('[worker error]', msg);
        const gpu = dropDeadProxy();
        self.postMessage(gpu ? { type: 'error', error: msg, gpu } : { type: 'error', error: msg });
    }
};
