const log = (m) => { document.getElementById("log").textContent += m + "\n"; };
const worker = new Worker("/dory-bench-worker.js", { type: "module" });
worker.onerror = (e) => log("worker error: " + e.message);
worker.onmessageerror = (e) => log("worker messageerror");
let ready = false;
const pending = new Map();
let nextId = 1;
window.__doryResults = [];

worker.onmessage = (e) => {
  const msg = e.data;
  if (msg.type === "ready") { ready = true; log("worker ready (" + msg.threads + " threads)"); return; }
  if (msg.type === "log") { log(msg.text); return; }
  const cb = pending.get(msg.id);
  if (cb) { pending.delete(msg.id); cb(msg); }
};

function runBench(mode, logN) {
  return new Promise((resolve) => {
    const id = nextId++;
    pending.set(id, resolve);
    worker.postMessage({ id, mode, logN });
  });
}

async function handle(mode) {
  const logN = parseInt(document.getElementById("size").value, 10);
  for (const b of document.querySelectorAll("button")) b.disabled = true;
  log(`running ${mode} at 2^${logN}…`);
  const t0 = performance.now();
  const res = await runBench(mode, logN);
  const wall = ((performance.now() - t0) / 1000).toFixed(1);
  if (res.error) {
    log(`${mode} 2^${logN} FAILED: ${res.error}`);
    window.__doryResults.push({ mode, logN, error: res.error });
  } else {
    const r = JSON.parse(res.result);
    const row = document.createElement("tr");
    row.innerHTML = `<td>${mode} 2^${logN}</td><td>${(r.setup_ms / 1000).toFixed(2)}s</td><td>${(r.commit_ms / 1000).toFixed(2)}s</td><td>${(r.open_ms / 1000).toFixed(2)}s</td><td>${r.verified}</td>`;
    document.getElementById("results").appendChild(row);
    window.__doryResults.push({ mode, logN, ...r, wall_s: parseFloat(wall) });
    log(`${mode} 2^${logN}: commit ${(r.commit_ms / 1000).toFixed(2)}s, open ${(r.open_ms / 1000).toFixed(2)}s (wall ${wall}s)`);
  }
  for (const b of document.querySelectorAll("button")) b.disabled = false;
}

document.getElementById("run-gpu").onclick = () => handle("gpu");
document.getElementById("run-cpu").onclick = () => handle("cpu");

const params = new URLSearchParams(location.search);
if (params.get("auto")) {
  const modes = params.get("auto").split(",");
  const sizes = (params.get("size") || "16").split(",");
  const waitReady = () => new Promise((r) => {
    const iv = setInterval(() => { if (ready) { clearInterval(iv); r(); } }, 100);
  });
  (async () => {
    await waitReady();
    for (const m of modes) {
      for (const s of sizes) {
        document.getElementById("size").value = s;
        await handle(m);
      }
    }
    window.__doryDone = true;
  })();
}
