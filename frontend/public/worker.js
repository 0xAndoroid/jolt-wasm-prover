import init, {
    initThreadPool,
    init_inlines,
    init_tracing,
    get_trace_json,
    clear_trace,
    webgpu_configure,
    webgpu_warmup,
    webgpu_miller_served,
    bench_stream,
    WasmProver,
    WasmVerifier,
} from '/pkg/jolt_wasm_prover.js';

let wasmExports = null;
const provers = {};
const verifiers = {};

// Spawns the dedicated GPU worker (sharing our module + memory), waits for
// its pump to come up, then runs the blocking device warmup through the
// job queue. Any failure leaves the prover CPU-only — the Rust side is
// fail-closed and worker.js just reports what happened.
async function initWebGpu(module, config) {
    const gpuWorker = new Worker('/gpu-worker.js', { type: 'module' });
    const ready = new Promise((resolve, reject) => {
        gpuWorker.onmessage = (ev) => {
            if (ev.data.type === 'gpu-ready') resolve();
            if (ev.data.type === 'gpu-error') {
                // After ready this reject is a no-op — keep the record loud:
                // a post-ready gpu-error is a drain trap (engine-side panic).
                console.error('[worker] gpu-worker error:', ev.data.error);
                reject(new Error(ev.data.error));
            }
        };
        gpuWorker.onerror = (ev) => {
            console.error('[worker] gpu-worker crashed:', ev.message || ev);
            reject(new Error('gpu-worker crashed'));
        };
    });
    const t0 = performance.now();
    gpuWorker.postMessage({ module, memory: wasmExports.memory });
    await ready;
    const readyMs = performance.now() - t0;

    webgpu_configure(
        false,
        config.minTerms || 0,
        config.handoffLen || 0,
        config.millerCpuFraction ?? -1,
        config.commitPipeline !== false,
        // Default TRUE (W4-D): the Jacobian pair miscomputes on M5-class
        // GPU stacks at scale; explicit false selects it for A/B only.
        config.bucketXyzz !== false,
        config.minTermsCommit || 0,
        config.minTermsBytecode || 0,
        // 0 is meaningful (one shard per pass); absent keeps the default.
        config.millerCoalesce ?? -1,
        config.minTermsDoryFold || 0,
    );
    const t1 = performance.now();
    webgpu_warmup();
    const warmupMs = performance.now() - t1;
    console.log(
        `[worker] webgpu up: gpu-worker ${readyMs.toFixed(0)}ms, ` +
        `device+pipelines ${warmupMs.toFixed(0)}ms`
    );
    return { readyMs, warmupMs };
}

self.onmessage = async (e) => {
    const { type, data } = e.data;

    try {
        switch (type) {
            case 'init': {
                const response = await fetch('/pkg/jolt_wasm_prover_bg.wasm');
                const module = await WebAssembly.compileStreaming(response);
                wasmExports = await init({ module_or_path: module });
                await initThreadPool(data.numThreads);
                init_inlines();
                if (data.tracing !== false) init_tracing();
                let webgpu = null;
                if (data.webgpu) {
                    try {
                        webgpu = await initWebGpu(module, data.webgpu);
                    } catch (err) {
                        console.warn('[worker] webgpu unavailable, CPU-only:', err.message || err);
                    }
                }
                self.postMessage({ type: 'init-done', webgpu });
                break;
            }

            case 'bench-stream': {
                const out = bench_stream(data.kind, data.log2Len, data.passes);
                self.postMessage({ type: 'bench-done', result: JSON.parse(out) });
                break;
            }

            case 'load-program': {
                const name = data.program;
                provers[name] = new WasmProver(
                    new Uint8Array(data.proverPreprocessing),
                    new Uint8Array(data.verifierPreprocessing),
                    new Uint8Array(data.elfBytes)
                );
                verifiers[name] = new WasmVerifier(
                    new Uint8Array(data.verifierPreprocessing)
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
                    numCycles: result.num_cycles,
                    paddedCycles: result.padded_cycles,
                    peakMemory,
                    elapsed,
                    millerServed: webgpu_miller_served(),
                });
                break;
            }

            case 'verify': {
                const verifier = verifiers[data.program];
                const start = performance.now();
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
