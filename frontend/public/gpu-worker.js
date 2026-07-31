// Dedicated GPU worker for the webgpu prover arm: owns the WebGPU device
// and pumps the shared-memory job queue in jolt-kernels' engine.
// Initialized with the prove worker's compiled module + shared memory so
// both sides see one wasm heap. The loop awaits jolt_webgpu_drain (jobs run
// on this worker's event loop, which drives the WebGPU callbacks) and parks
// on Atomics.waitAsync between signals; this worker never blocks.

import init, {
    jolt_webgpu_signal_addr,
    jolt_webgpu_drain,
} from '/pkg/jolt_wasm_prover.js';

onmessage = async (e) => {
    const { module, memory } = e.data;
    try {
        await init({ module_or_path: module, memory, thread_stack_size: 4 * 1024 * 1024 });
        const addr = jolt_webgpu_signal_addr();
        const idx = addr >> 2;
        postMessage({ type: 'gpu-ready' });

        // Re-create the view each iteration: shared wasm memory can grow,
        // and an old SharedArrayBuffer view does not cover new pages.
        let last = Atomics.load(new Int32Array(memory.buffer), idx);
        for (;;) {
            await jolt_webgpu_drain();
            const arr = new Int32Array(memory.buffer);
            const cur = Atomics.load(arr, idx);
            if (cur === last) {
                const wait = Atomics.waitAsync(arr, idx, cur);
                if (wait.async) await wait.value;
            }
            last = Atomics.load(new Int32Array(memory.buffer), idx);
        }
    } catch (err) {
        console.error('[gpu-worker] failed', err);
        postMessage({ type: 'gpu-error', error: String(err) });
    }
};
