// Tint-parity vector check inside a REAL browser (the compiler stack the
// production wasm actually runs on), for cross-version Tint gating — the
// node-webgpu harness (jolt's wgsl-check/check.mjs) pins its OWN Dawn
// snapshot, which is neither Chrome 145's nor 150's Tint.
//
// Runs the exact exported WGSL modules + vectors byte-compared against the
// host expectations, one page.evaluate per case (cases carry MB-scale word
// arrays).
//
// Usage:
//   node wgsl-check-browser.mjs <vector-dir> [baseUrl]
//     vector-dir: jolt's target/wgsl-vectors (vectors.json + *.wgsl)
//     baseUrl:    a served secure context (default http://localhost:8125)
//   PW_CHANNEL selects the browser channel (unset = bundled chromium).
//
// Exit code = number of failing cases.

import { chromium } from 'playwright';
import { readFileSync } from 'node:fs';
import { resolve, join } from 'node:path';

const dir = resolve(
    process.argv[2] ?? '../../../jolt/.worktrees/w4-m/crates/jolt-kernels/target/wgsl-vectors',
);
const base = process.argv[3] ?? 'http://localhost:8080';
const cases = JSON.parse(readFileSync(join(dir, 'vectors.json'), 'utf8'));

const browser = await chromium.launch({
    headless: true,
    channel: process.env.PW_CHANNEL || undefined,
    executablePath: process.env.PW_EXECUTABLE || undefined,
});
const page = await browser.newPage();
page.on('console', (msg) => process.stderr.write('[page] ' + msg.text() + '\n'));
await page.goto(base, { waitUntil: 'domcontentloaded' });
process.stderr.write(`browser: ${browser.version()}\n`);

const adapterInfo = await page.evaluate(async () => {
    if (!navigator.gpu) return { error: 'no navigator.gpu' };
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) return { error: 'no adapter' };
    window.__device = await adapter.requestDevice();
    const i = adapter.info ?? {};
    return { vendor: i.vendor, architecture: i.architecture };
});
process.stderr.write(`adapter: ${JSON.stringify(adapterInfo)}\n`);
if (adapterInfo.error) {
    console.error(`FATAL: ${adapterInfo.error}`);
    process.exit(2);
}

let failures = 0;
for (const testCase of cases) {
    const modules = {};
    for (const step of testCase.steps) {
        modules[step.module] ??= readFileSync(join(dir, step.module), 'utf8');
    }
    const result = await page.evaluate(async ({ testCase, modules }) => {
        const device = window.__device;
        try {
            window.__pipelines ??= new Map();
            for (const [file, code] of Object.entries(modules)) {
                if (window.__pipelines.has(file)) continue;
                const module = device.createShaderModule({ code });
                const messages = (await module.getCompilationInfo()).messages.filter((m) => m.type === 'error');
                if (messages.length) {
                    return { error: `${file}: ${messages.map((m) => `${m.lineNum}: ${m.message}`).join(' | ')}` };
                }
                window.__pipelines.set(file, await device.createComputePipelineAsync({
                    layout: 'auto',
                    compute: { module, entryPoint: 'main' },
                }));
            }

            const buffers = new Map();
            for (const [name, spec] of Object.entries(testCase.buffers)) {
                const words = Array.isArray(spec) ? spec.length : spec;
                const buffer = device.createBuffer({
                    size: Math.max(4, words * 4),
                    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
                });
                if (Array.isArray(spec)) device.queue.writeBuffer(buffer, 0, new Uint32Array(spec));
                buffers.set(name, buffer);
            }

            const paramsBuffers = [];
            const encoder = device.createCommandEncoder();
            const pass = encoder.beginComputePass();
            for (const step of testCase.steps) {
                const pipeline = window.__pipelines.get(step.module);
                const params = device.createBuffer({
                    size: 64,
                    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
                });
                paramsBuffers.push(params);
                device.queue.writeBuffer(params, 0, new Uint32Array(step.params));
                const entries = [{ binding: 0, resource: { buffer: params } }];
                for (const [binding, name] of Object.entries(step.bindings)) {
                    entries.push({ binding: Number(binding), resource: { buffer: buffers.get(name) } });
                }
                pass.setPipeline(pipeline);
                pass.setBindGroup(0, device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries }));
                const wg = Array.isArray(step.workgroups) ? step.workgroups : [step.workgroups];
                pass.dispatchWorkgroups(wg[0], wg[1] ?? 1, wg[2] ?? 1);
            }
            pass.end();

            const expected = new Uint32Array(testCase.expect);
            const staging = device.createBuffer({
                size: expected.length * 4,
                usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
            });
            encoder.copyBufferToBuffer(buffers.get(testCase.readback), 0, staging, 0, expected.length * 4);
            device.queue.submit([encoder.finish()]);
            await staging.mapAsync(GPUMapMode.READ);
            const got = new Uint32Array(staging.getMappedRange().slice(0));
            staging.unmap();

            let firstDiff = -1;
            for (let i = 0; i < expected.length; i++) {
                if (got[i] !== expected[i]) { firstDiff = i; break; }
            }
            let diffCount = 0;
            if (firstDiff !== -1) {
                for (let i = 0; i < expected.length; i++) if (got[i] !== expected[i]) diffCount++;
            }
            for (const buffer of buffers.values()) buffer.destroy();
            for (const buffer of paramsBuffers) buffer.destroy();
            staging.destroy();
            return firstDiff === -1
                ? { pass: true, words: expected.length }
                : { pass: false, firstDiff, got: got[firstDiff], want: expected[firstDiff], diffCount, words: expected.length };
        } catch (error) {
            return { error: String(error && error.message || error) };
        }
    }, { testCase, modules });

    if (result.pass) {
        console.error(`PASS ${testCase.label} (${result.words} words)`);
    } else {
        failures++;
        console.error(result.error
            ? `FAIL ${testCase.label}: ${result.error}`
            : `FAIL ${testCase.label}: word ${result.firstDiff}: got ${result.got}, want ${result.want} (${result.diffCount}/${result.words} words differ)`);
    }
}

await browser.close();
console.error(failures === 0 ? `ALL ${cases.length} PASS` : `${failures}/${cases.length} FAIL`);
process.exit(failures);
