// Browser e2e prover benchmark driver: full Jolt prove with CPU-WASM Dory
// vs full-GPU (WebGPU) Dory in headless Chromium. Usage:
//   node server.mjs &        (after wasm-pack + frontend build, see CLAUDE.md)
//   node e2e-bench-browser.mjs [iters] [runs] [modes]
//   e.g. node e2e-bench-browser.mjs 300 3 cpu,gpu
import { chromium } from "playwright";

const iters = process.argv[2] || "300";
const runs = process.argv[3] || "3";
const modes = process.argv[4] || "cpu,gpu";
const url = `http://localhost:8080/e2e-bench.html?auto=${modes}&iters=${iters}&runs=${runs}`;

const browser = await chromium.launch({
  headless: true,
  args: [
    "--enable-unsafe-webgpu",
    "--enable-features=Vulkan",
    "--ignore-gpu-blocklist",
  ],
});
const page = await browser.newPage();
page.on("console", (msg) => console.error(`[page] ${msg.text()}`));
page.on("pageerror", (err) => console.error(`[pageerror] ${err}`));

console.error(`navigating: ${url}`);
await page.goto(url, { waitUntil: "domcontentloaded" });

const timeoutMin = parseInt(process.env.E2E_BENCH_TIMEOUT_MIN || "120", 10);
const deadline = Date.now() + timeoutMin * 60 * 1000;
let printed = 0;
let done = false;
while (!done && Date.now() < deadline) {
  await new Promise((r) => setTimeout(r, 5000));
  const state = await page.evaluate(() => ({
    results: window.__e2eResults,
    done: window.__e2eDone === true,
  }));
  while (printed < state.results.length) {
    console.log("RESULT " + JSON.stringify(state.results[printed]));
    printed++;
  }
  done = state.done;
}
console.log(done ? "DONE" : "TIMEOUT");
await browser.close();
