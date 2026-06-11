// e2e prover bench worker: runs the full Jolt prove (CPU-WASM Dory or
// full-GPU Dory) and verifies every proof with the stock verifier. The GPU
// path spawns a dedicated GPU worker sharing this worker's wasm module and
// memory (one heap; the GPU engine job queue lives in it).

import init, {
  initThreadPool,
  init_inlines,
  WasmProver,
  WasmGpuProver,
  WasmVerifier,
} from "/pkg/jolt_wasm_prover.js";

const threads = Math.min(navigator.hardwareConcurrency || 4, 12);
let wasm = null;
let gpuReady = null;

async function setup() {
  const response = await fetch("/pkg/jolt_wasm_prover_bg.wasm");
  const module = await WebAssembly.compileStreaming(response);
  wasm = await init({ module_or_path: module });
  await initThreadPool(threads);
  init_inlines();

  const gpuWorker = new Worker("/e2e-gpu-worker.js", { type: "module" });
  gpuReady = new Promise((resolve, reject) => {
    gpuWorker.onmessage = (e) => {
      if (e.data.type === "gpu-ready") resolve();
      if (e.data.type === "gpu-error") reject(new Error(e.data.error));
    };
  });
  gpuWorker.postMessage({ module, memory: wasm.memory });
  postMessage({ type: "ready", threads });
}

const initialized = setup();
initialized.catch((e) => {
  console.error("[e2e-worker] init failed", e);
  postMessage({ type: "error", error: "init failed: " + e });
});

onmessage = async (e) => {
  const { id, pcs, iters, runs, preprocessing, elf, verifierPreprocessing } = e.data;
  try {
    await initialized;
    if (pcs === "gpu") await gpuReady;

    const prover =
      pcs === "gpu"
        ? new WasmGpuProver(new Uint8Array(preprocessing), new Uint8Array(elf))
        : new WasmProver(new Uint8Array(preprocessing), new Uint8Array(elf));
    const verifier = new WasmVerifier(new Uint8Array(verifierPreprocessing));
    const input = new Uint8Array(32).fill(7);

    const all = [];
    let cycles = 0;
    // runs + 1 with the first discarded (engine init, pipeline compilation,
    // allocator warmup).
    for (let run = 0; run <= runs; run++) {
      const t0 = performance.now();
      const result = prover.prove_keccak_chain(input, iters);
      const proveS = (performance.now() - t0) / 1000;
      cycles = result.num_cycles;

      const valid = verifier.verify(result.proof, result.program_io);
      if (!valid) throw new Error(`run ${run}: proof failed verification`);

      console.log(`[e2e-worker] ${pcs} run ${run}: ${proveS.toFixed(2)}s (verified)`);
      postMessage({
        type: "progress",
        id,
        run,
        warmup: run === 0,
        prove_s: proveS,
        peak_memory: wasm.memory.buffer.byteLength,
      });
      if (run > 0) all.push(proveS);
    }

    const sorted = [...all].sort((a, b) => a - b);
    postMessage({
      id,
      result: {
        pcs,
        iters,
        cycles,
        runs: all.length,
        prove_all: all,
        prove_median_s: sorted[Math.floor(sorted.length / 2)],
        threads,
        peak_memory: wasm.memory.buffer.byteLength,
        verified: true,
      },
    });
  } catch (err) {
    console.error("[e2e-worker] bench failed", err);
    postMessage({ id, error: String(err) });
  }
};
