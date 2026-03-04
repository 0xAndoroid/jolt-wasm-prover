import init, {
    initThreadPool,
    init_inlines,
    init_tracing,
    get_trace_json,
    clear_trace,
    WasmProver,
    WasmVerifier,
} from '/pkg/jolt_wasm_prover.js';

const WASM_URL = '/pkg/jolt_wasm_prover_bg.wasm';
const CACHE_DB = 'jolt-wasm-cache';
const CACHE_STORE = 'modules';

async function openCacheDB() {
    return new Promise((resolve, reject) => {
        const req = indexedDB.open(CACHE_DB, 1);
        req.onupgradeneeded = () => req.result.createObjectStore(CACHE_STORE);
        req.onsuccess = () => resolve(req.result);
        req.onerror = () => reject(req.error);
    });
}

async function getCachedModule(db, key) {
    return new Promise((resolve) => {
        const tx = db.transaction(CACHE_STORE, 'readonly');
        const req = tx.objectStore(CACHE_STORE).get(key);
        req.onsuccess = () => resolve(req.result || null);
        req.onerror = () => resolve(null);
    });
}

async function putCachedModule(db, key, module) {
    return new Promise((resolve) => {
        const tx = db.transaction(CACHE_STORE, 'readwrite');
        tx.objectStore(CACHE_STORE).put(module, key);
        tx.oncomplete = () => resolve();
        tx.onerror = () => resolve();
    });
}

async function loadWasmModule() {
    const resp = await fetch(WASM_URL);
    const etag = resp.headers.get('etag') || resp.headers.get('last-modified') || '';
    const cacheKey = WASM_URL + '|' + etag;

    let db;
    try { db = await openCacheDB(); } catch { /* indexedDB unavailable */ }

    if (db) {
        const cached = await getCachedModule(db, cacheKey);
        if (cached instanceof WebAssembly.Module) {
            return cached;
        }
    }

    const module = await WebAssembly.compileStreaming(resp);

    if (db) {
        // Clear old entries then store new one
        try {
            const tx = db.transaction(CACHE_STORE, 'readwrite');
            tx.objectStore(CACHE_STORE).clear();
            tx.oncomplete = () => putCachedModule(db, cacheKey, module);
        } catch { /* best-effort */ }
    }

    return module;
}

let wasmExports = null;
const provers = {};
const verifiers = {};

self.onmessage = async (e) => {
    const { type, data } = e.data;

    try {
        switch (type) {
            case 'init': {
                const module = await loadWasmModule();
                wasmExports = await init({ module });
                await initThreadPool(data.numThreads);
                init_inlines();
                init_tracing();
                self.postMessage({ type: 'init-done' });
                break;
            }

            case 'load-program': {
                const name = data.program;
                provers[name] = new WasmProver(
                    new Uint8Array(data.proverPreprocessing),
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
                }

                const elapsed = performance.now() - start;
                const peakMemory = wasmExports.memory.buffer.byteLength;

                self.postMessage({
                    type: 'prove-done',
                    program: data.program,
                    proof: result.proof,
                    proofSize: result.proof_size,
                    compressedProofSize: result.compressed_proof_size,
                    programIo: result.program_io,
                    numCycles: result.num_cycles,
                    peakMemory,
                    elapsed,
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
