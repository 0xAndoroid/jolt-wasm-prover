// Dedicated GPU worker for the full-GPU Dory e2e prover: owns the WebGPU
// device and pumps the shared-memory job queue. Initialized with the prove
// worker's compiled module + shared memory so both sides see one heap.
// The loop awaits gpu_engine_drain (jobs run on this worker's event loop,
// which drives the WebGPU callbacks) and parks on Atomics.waitAsync between
// signals.

import init, {
  gpu_engine_signal_addr,
  gpu_engine_drain,
} from "/pkg/jolt_wasm_prover.js";

onmessage = async (e) => {
  const { module, memory } = e.data;
  try {
    await init({ module_or_path: module, memory, thread_stack_size: 4 * 1024 * 1024 });
    const addr = gpu_engine_signal_addr();
    const idx = addr >> 2;
    postMessage({ type: "gpu-ready" });
    console.log("[gpu-worker] ready, signal addr", addr);

    // Re-create the view each iteration: shared wasm memory can grow, and
    // the old SharedArrayBuffer view does not cover new pages.
    let last = Atomics.load(new Int32Array(memory.buffer), idx);
    let drains = 0;
    for (;;) {
      drains++;
      if (drains <= 5 || drains % 50 === 0) {
        console.log(`[gpu-worker] drain #${drains} (signal ${last})`);
      }
      await gpu_engine_drain();
      const arr = new Int32Array(memory.buffer);
      const cur = Atomics.load(arr, idx);
      if (cur === last) {
        const wait = Atomics.waitAsync(arr, idx, cur);
        if (wait.async) await wait.value;
      }
      last = Atomics.load(new Int32Array(memory.buffer), idx);
    }
  } catch (err) {
    console.error("[gpu-worker] failed", err);
    postMessage({ type: "gpu-error", error: String(err) });
  }
};
