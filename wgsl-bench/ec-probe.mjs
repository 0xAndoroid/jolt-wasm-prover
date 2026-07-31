// One isolated composed-EC probe per process: node ec-probe.mjs <unroll> <guarded> <n> <k> <wg> [watchdogMs]
// Prints pipeline-ms and per-dispatch ms (2 dispatches: cold + warm).
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const dir = '/Users/andoroid/dev/jolt-wasm-prover/.worktrees/p2-wgsl-microbench/wgsl-bench';
const kind = process.argv[2];
const [unroll, guarded, n, k, wg, wd] = process.argv.slice(3).map(Number);
const K = await import('./ec-kernels.mjs');
const { create, globals } = await import('webgpu');
Object.assign(globalThis, globals);
const gpu = create([]);
const adapter = await gpu.requestAdapter();
const device = await adapter.requestDevice();
device.lost.then((l) => console.log('DEVICE LOST:', l.reason, l.message));
const tick = typeof device.tick === 'function' ? setInterval(() => device.tick(), 1) : null;
const tag = `${kind}-u${unroll}${guarded ? "g" : "n"} wg${wg} n${n} k${k}`;
setTimeout(() => { console.log(`${tag} WATCHDOG ${wd || 60000}ms`); process.exit(9); }, wd || 60000);

const mod = kind === 'dbl' ? K.ecDblModule(wg, unroll)
  : kind === 'xyzz' ? K.ecXyzzModule(wg, unroll)
  : K.ecMaddModule(wg, unroll, { guarded: !!guarded });
const t0 = Date.now();
const m = device.createShaderModule({ code: mod.src });
const pipe = await device.createComputePipelineAsync({ layout: 'auto', compute: { module: m, entryPoint: 'main_bench' } });
console.log(`${tag} pipeline ${Date.now() - t0}ms`);

const outbuf = device.createBuffer({ size: n * mod.wordsPerElem * 4, usage: GPUBufferUsage.STORAGE });
const params = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
device.queue.writeBuffer(params, 0, new Uint32Array([k, n, 0, 0]));
const bg = device.createBindGroup({
  layout: pipe.getBindGroupLayout(0),
  entries: [{ binding: 0, resource: { buffer: outbuf } }, { binding: 1, resource: { buffer: params } }],
});
for (let r = 0; r < 2; r++) {
  const enc = device.createCommandEncoder();
  const pass = enc.beginComputePass();
  pass.setPipeline(pipe);
  pass.setBindGroup(0, bg);
  pass.dispatchWorkgroups(Math.ceil(n / wg));
  pass.end();
  const t1 = Date.now();
  device.queue.submit([enc.finish()]);
  await device.queue.onSubmittedWorkDone();
  const ms = Date.now() - t1;
  const gmul = (n * k * (kind === "dbl" ? 7 : 11) * unroll) / ms / 1e6;
  console.log(`${tag} dispatch${r} ${ms}ms ${gmul.toFixed(4)} Gmul/s`);
}
process.exit(0);
