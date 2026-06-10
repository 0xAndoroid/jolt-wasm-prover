// Diagnostic: load the bench page, report WebGPU adapter availability and
// the page log every 15 s.
import { chromium } from "playwright";

const browser = await chromium.launch({
  headless: true,
  args: ["--enable-unsafe-webgpu", "--ignore-gpu-blocklist"],
});
const page = await browser.newPage();
page.on("pageerror", (err) => console.error(`[pageerror] ${err}`));
page.on("console", (m) => console.error(`[console] ${m.text()}`));

await page.goto("http://localhost:8080/dory-bench.html?auto=gpu&size=16", {
  waitUntil: "domcontentloaded",
});

const gpuInfo = await page.evaluate(async () => {
  if (!navigator.gpu) return "navigator.gpu MISSING on main thread";
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) return "requestAdapter returned null";
  return `adapter ok: ${adapter.info ? JSON.stringify(adapter.info) : "no info"}`;
});
console.error(`main-thread webgpu: ${gpuInfo}`);

for (let i = 0; i < 20; i++) {
  await new Promise((r) => setTimeout(r, 15000));
  const state = await page.evaluate(() => ({
    log: document.getElementById("log").textContent,
    results: window.__doryResults,
    done: window.__doryDone === true,
  }));
  console.error(`--- t=${(i + 1) * 15}s log: ${JSON.stringify(state.log)}`);
  if (state.done || (state.results && state.results.length)) {
    console.log(JSON.stringify(state.results, null, 2));
    break;
  }
}
await browser.close();
