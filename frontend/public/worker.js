import init, {
    initThreadPool,
    init_tracing,
    get_trace_json,
    clear_trace,
    WasmProver,
    WasmVerifier,
    gpu_mailbox_ptr,
    set_gpu_enabled,
    set_gpu_unavailable,
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
// Resolves to the proxy's ready/unavailable report; never throws.
async function initGpu() {
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
    return { status: 'ready', adapter: report.adapter, features: report.features, limits: report.limits };
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
                const gpu = data.gpu === true ? await initGpu() : { status: 'disabled' };
                self.postMessage({ type: 'init-done', gpu });
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
        self.postMessage({ type: 'error', error: msg });
    }
};
