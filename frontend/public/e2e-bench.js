const log = (m) => {
  document.getElementById("log").textContent += m + "\n";
};

const worker = new Worker("/e2e-bench-worker.js", { type: "module" });
worker.onerror = (e) => log("worker error: " + e.message);
let ready = false;
const pending = new Map();
let nextId = 1;
window.__e2eResults = [];

let artifacts = null;
async function loadArtifacts() {
  if (artifacts) return artifacts;
  const fetchBuf = async (path) => (await fetch(path)).arrayBuffer();
  artifacts = {
    preprocessing: await fetchBuf("/keccak_prover.bin"),
    verifierPreprocessing: await fetchBuf("/keccak_verifier.bin"),
    elf: await fetchBuf("/keccak.elf"),
  };
  return artifacts;
}

worker.onmessage = (e) => {
  const msg = e.data;
  if (msg.type === "ready") {
    ready = true;
    log(`worker ready (${msg.threads} threads)`);
    return;
  }
  if (msg.type === "error") {
    log("worker init error: " + msg.error);
    return;
  }
  if (msg.type === "progress") {
    log(
      `  run ${msg.run}${msg.warmup ? " (warmup)" : ""}: ${msg.prove_s.toFixed(2)}s` +
        ` (heap ${(msg.peak_memory / 1e9).toFixed(2)} GB)`
    );
    return;
  }
  const cb = pending.get(msg.id);
  if (cb) {
    pending.delete(msg.id);
    cb(msg);
  }
};

async function runBench(pcs, iters, runs) {
  const a = await loadArtifacts();
  return new Promise((resolve) => {
    const id = nextId++;
    pending.set(id, resolve);
    worker.postMessage({ id, pcs, iters, runs, ...a });
  });
}

async function handle(pcs) {
  const iters = parseInt(document.getElementById("iters").value, 10);
  const runs = parseInt(document.getElementById("runs").value, 10);
  for (const b of document.querySelectorAll("button")) b.disabled = true;
  log(`running ${pcs} e2e, ${iters} keccak iters, ${runs} runs…`);
  const res = await runBench(pcs, iters, runs);
  if (res.error) {
    log(`${pcs} FAILED: ${res.error}`);
    window.__e2eResults.push({ pcs, iters, error: res.error });
  } else {
    const r = res.result;
    const row = document.createElement("tr");
    row.innerHTML = `<td>${r.pcs}</td><td>${r.cycles}</td><td>${r.prove_median_s.toFixed(2)}s</td><td>${r.prove_all.map((x) => x.toFixed(2)).join(", ")}</td><td>${r.verified}</td>`;
    document.getElementById("results").appendChild(row);
    window.__e2eResults.push(r);
    log(`${pcs}: median ${r.prove_median_s.toFixed(2)}s over ${r.runs} runs (all verified)`);
  }
  for (const b of document.querySelectorAll("button")) b.disabled = false;
}

document.getElementById("run-cpu").onclick = () => handle("cpu");
document.getElementById("run-gpu").onclick = () => handle("gpu");

const params = new URLSearchParams(location.search);
if (params.get("auto")) {
  const modes = params.get("auto").split(",");
  if (params.get("iters")) document.getElementById("iters").value = params.get("iters");
  if (params.get("runs")) document.getElementById("runs").value = params.get("runs");
  const waitReady = () =>
    new Promise((r) => {
      const iv = setInterval(() => {
        if (ready) {
          clearInterval(iv);
          r();
        }
      }, 100);
    });
  (async () => {
    await waitReady();
    for (const m of modes) {
      await handle(m);
    }
    window.__e2eDone = true;
  })();
}
