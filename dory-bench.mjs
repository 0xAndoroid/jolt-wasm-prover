// Browser Dory benchmark driver: runs CPU-WASM and WebGPU paths in headless
// Chromium against the local server. Usage:
//   node server.mjs &            (after wasm-pack + frontend build, see CLAUDE.md)
//   node dory-bench.mjs [size] [modes]   e.g. node dory-bench.mjs 16 gpu,cpu
import { chromium } from "playwright";

const size = process.argv[2] || "16";
const modes = process.argv[3] || "gpu,cpu";
const url = `http://localhost:8080/dory-bench.html?auto=${modes}&size=${size}`;

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

const timeoutMin = parseInt(process.env.DORY_BENCH_TIMEOUT_MIN || "60", 10);
const deadline = Date.now() + timeoutMin * 60 * 1000;
let printed = 0;
let done = false;
while (!done && Date.now() < deadline) {
  await new Promise((r) => setTimeout(r, 5000));
  const state = await page.evaluate(() => ({
    results: window.__doryResults,
    done: window.__doryDone === true,
  }));
  while (printed < state.results.length) {
    console.log("RESULT " + JSON.stringify(state.results[printed]));
    printed++;
  }
  done = state.done;
}
console.log(done ? "DONE" : "TIMEOUT");
await browser.close();
