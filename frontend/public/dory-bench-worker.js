console.log("[worker] script start");

import init, {
  initThreadPool,
  dory_bench_cpu,
  dory_bench_gpu,
} from "/pkg/jolt_wasm_prover.js";

const threads = Math.min(navigator.hardwareConcurrency || 4, 12);

let initialized = (async () => {
  console.log("[worker] wasm init…");
  await init();
  console.log("[worker] thread pool init…");
  await initThreadPool(threads);
  console.log("[worker] ready");
  postMessage({ type: "ready", threads });
})();
initialized.catch((e) => {
  console.error("[worker] init failed", e);
  postMessage({ type: "log", text: "worker init failed: " + e });
});

onmessage = async (e) => {
  const { id, mode, logN } = e.data;
  try {
    await initialized;
    console.log(`[worker] running ${mode} 2^${logN}`);
    const result = mode === "gpu" ? await dory_bench_gpu(logN) : dory_bench_cpu(logN);
    postMessage({ id, result });
  } catch (err) {
    console.error("[worker] bench failed", err);
    postMessage({ id, error: String(err) });
  }
};
