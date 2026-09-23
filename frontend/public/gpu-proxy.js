// Dedicated Worker that owns the WebGPU device for the wasm prover.
//
// The prover threads block synchronously, so they cannot await WebGPU
// promises themselves. Instead they fill a mailbox inside the shared wasm
// memory (see src/gpu/mailbox.rs — word offsets below mirror its layout),
// bump the doorbell and Atomics.wait on `status`; this worker never blocks:
// it Atomics.waitAsync's on the doorbell, runs the op, writes the result
// back into wasm memory and flips `status`.

const W = {
    DOORBELL: 0,
    STATUS: 1,
    OP: 2,
    ARGS: 4,
    REGIONS: 36,
    RET: 60,
    ERROR_LEN: 68,
    ERROR: 69,
    ERROR_BYTES: 256,
};
const MAX_REGIONS = 8;
const OP = { NOP: 1, CREATE_BUFFER: 2, UPLOAD: 3, DESTROY: 4, RUN: 5 };
const STATUS = { IDLE: 0, BUSY: 1, DONE: 2, ERROR: 3 };
const REGION = { UPLOAD: 1, READBACK: 2, HANDLE: 4 };
// Index = shader id in the RUN op (src/gpu/selftest.rs, src/gpu/trace_commit.rs);
// each entry lists the sources concatenated into one module (library first).
const SHADERS = [
    ['fp128', 'fp128_ops'],
    ['commit/common', 'commit/prep'],
    ['commit/common', 'commit/commit_accumulate'],
    ['commit/common', 'commit/reduce'],
];

let memory = null;
let base = 0;
let device = null;
const shaderSources = new Map();
const pipelines = new Map();
const handles = new Map();
let nextHandle = 1;
let uniformBuffer = null;
let uncapturedError = null;

const i32 = () => new Int32Array(memory.buffer);
const u32 = () => new Uint32Array(memory.buffer);
const align4 = (n) => (n + 3) & ~3;

async function fetchText(name) {
    const r = await fetch(`/wgsl/${name}.wgsl`);
    if (!r.ok) throw new Error(`fetch ${name}.wgsl: ${r.status}`);
    return r.text();
}

// `chunk` != 0 sets the kernel's `CHUNK` override constant (commit_accumulate.wgsl);
// pipelines are cached per (shader, chunk).
function pipelineFor(id, chunk) {
    const key = `${id}:${chunk}`;
    let p = pipelines.get(key);
    if (p) return p;
    const parts = SHADERS[id];
    if (parts === undefined) throw new Error(`unknown shader id ${id}`);
    const module = device.createShaderModule({ code: parts.map((n) => shaderSources.get(n)).join('\n') });
    const constants = chunk ? { CHUNK: chunk } : {};
    p = device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: 'main', constants } });
    pipelines.set(key, p);
    return p;
}

function readRegions() {
    const m = u32();
    const regions = [];
    for (let i = 0; i < MAX_REGIONS; i++) {
        const o = base + W.REGIONS + i * 3;
        regions.push({ ptr: m[o], len: m[o + 1], flags: m[o + 2] });
    }
    return regions;
}

function storageBuffer(size) {
    return device.createBuffer({
        size: Math.max(4, align4(size)),
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
}

function upload(buf, offset, ptr, len) {
    if (len > 0) device.queue.writeBuffer(buf, offset, memory.buffer, ptr, len);
}

async function readback(buf, ptr, len) {
    const staging = device.createBuffer({ size: align4(len), usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(buf, 0, staging, 0, align4(len));
    device.queue.submit([enc.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    new Uint8Array(memory.buffer, ptr, len).set(new Uint8Array(staging.getMappedRange(), 0, len));
    staging.unmap();
    staging.destroy();
}

function getHandle(h) {
    const buf = handles.get(h);
    if (!buf) throw new Error(`unknown buffer handle ${h}`);
    return buf;
}

async function runOp(op, args, regions, ret) {
    switch (op) {
        case OP.NOP:
            return;
        case OP.CREATE_BUFFER: {
            const h = nextHandle++;
            handles.set(h, storageBuffer(args[0]));
            ret[0] = h;
            return;
        }
        case OP.UPLOAD: {
            const r = regions[0];
            upload(getHandle(args[0]), args[1], r.ptr, r.len);
            await device.queue.onSubmittedWorkDone();
            return;
        }
        case OP.DESTROY: {
            getHandle(args[0]).destroy();
            handles.delete(args[0]);
            return;
        }
        case OP.RUN: {
            const [shader, wx, wy, wz, nparams] = args;
            const params = new Uint32Array(16);
            params.set(args.subarray(5, 5 + nparams));
            const nbind = args[21];
            const pipeline = pipelineFor(shader, args[22]);
            device.queue.writeBuffer(uniformBuffer, 0, params);
            const entries = [{ binding: 0, resource: { buffer: uniformBuffer } }];
            const temps = [];
            const readbacks = [];
            try {
                for (let i = 0; i < nbind; i++) {
                    const r = regions[i];
                    let buf;
                    if (r.flags & REGION.HANDLE) {
                        // ptr is a handle here, not a wasm address: a readback would land at address `handle`.
                        if (r.flags & (REGION.UPLOAD | REGION.READBACK)) throw new Error(`binding ${i}: handle regions cannot be uploaded or read back`);
                        buf = getHandle(r.ptr);
                    } else {
                        buf = storageBuffer(r.len);
                        temps.push(buf);
                        if (r.flags & REGION.UPLOAD) upload(buf, 0, r.ptr, r.len);
                    }
                    if (r.flags & REGION.READBACK) readbacks.push({ buf, ptr: r.ptr, len: r.len });
                    entries.push({ binding: i + 1, resource: { buffer: buf } });
                }
                const bindGroup = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries });
                const enc = device.createCommandEncoder();
                const pass = enc.beginComputePass();
                pass.setPipeline(pipeline);
                pass.setBindGroup(0, bindGroup);
                pass.dispatchWorkgroups(wx, wy, wz);
                pass.end();
                device.queue.submit([enc.finish()]);
                for (const rb of readbacks) await readback(rb.buf, rb.ptr, rb.len);
                if (readbacks.length === 0) await device.queue.onSubmittedWorkDone();
            } finally {
                for (const t of temps) t.destroy();
            }
            return;
        }
        default:
            throw new Error(`unknown op ${op}`);
    }
}

function writeError(msg) {
    const bytes = new TextEncoder().encode(msg).slice(0, W.ERROR_BYTES);
    new Uint8Array(memory.buffer, (base + W.ERROR) * 4, W.ERROR_BYTES).set(bytes);
    Atomics.store(u32(), base + W.ERROR_LEN, bytes.length);
}

async function serve() {
    const m = u32();
    const args = new Uint32Array(m.buffer, (base + W.ARGS) * 4, 32);
    const ret = new Uint32Array(m.buffer, (base + W.RET) * 4, 8);
    const op = Atomics.load(m, base + W.OP);
    const regions = readRegions();
    let status = STATUS.DONE;
    try {
        device.pushErrorScope('validation');
        device.pushErrorScope('out-of-memory');
        let thrown = null;
        try {
            await runOp(op, args, regions, ret);
        } catch (e) {
            thrown = e;
        }
        // Pop even when the op threw, or the scopes leak and swallow later errors.
        const oom = await device.popErrorScope();
        const validation = await device.popErrorScope();
        const err = validation || oom || uncapturedError;
        uncapturedError = null;
        if (thrown) throw thrown;
        if (err) throw new Error(err.message);
    } catch (e) {
        writeError(e && e.message ? e.message : String(e));
        status = STATUS.ERROR;
    }
    Atomics.store(i32(), base + W.STATUS, status);
    Atomics.notify(i32(), base + W.STATUS);
}

async function waitDoorbell(seen) {
    const view = i32();
    if (typeof Atomics.waitAsync === 'function') {
        const r = Atomics.waitAsync(view, base + W.DOORBELL, seen, 1000);
        if (r.async) await r.value;
    } else {
        await new Promise((resolve) => setTimeout(resolve, 0));
    }
}

async function loop() {
    let seen = Atomics.load(i32(), base + W.DOORBELL);
    for (;;) {
        const cur = Atomics.load(i32(), base + W.DOORBELL);
        if (cur === seen) {
            await waitDoorbell(seen);
            continue;
        }
        seen = cur;
        await serve();
    }
}

async function init(data) {
    memory = data.memory;
    base = data.mailboxPtr >>> 2;
    if (!navigator.gpu) return { type: 'unavailable', reason: 'navigator.gpu missing (WebGPU unsupported or insecure context)' };
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) return { type: 'unavailable', reason: 'requestAdapter returned null' };
    const limits = {};
    for (const k of ['maxBufferSize', 'maxStorageBufferBindingSize', 'maxComputeWorkgroupsPerDimension',
        'maxComputeWorkgroupSizeX', 'maxComputeInvocationsPerWorkgroup', 'maxStorageBuffersPerShaderStage',
        'maxComputeWorkgroupStorageSize']) {
        limits[k] = adapter.limits[k];
    }
    device = await adapter.requestDevice({
        requiredLimits: {
            maxBufferSize: adapter.limits.maxBufferSize,
            maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
            maxComputeWorkgroupStorageSize: adapter.limits.maxComputeWorkgroupStorageSize,
            maxComputeInvocationsPerWorkgroup: adapter.limits.maxComputeInvocationsPerWorkgroup,
            maxComputeWorkgroupSizeX: adapter.limits.maxComputeWorkgroupSizeX,
        },
    });
    device.addEventListener('uncapturederror', (e) => { uncapturedError = e.error; });
    device.lost.then((info) => console.error('[gpu-proxy] device lost:', info.message));
    uniformBuffer = device.createBuffer({ size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    for (const name of new Set(SHADERS.flat())) shaderSources.set(name, await fetchText(name));
    const info = adapter.info || {};
    loop();
    return {
        type: 'ready',
        adapter: { vendor: info.vendor, architecture: info.architecture, device: info.device, description: info.description },
        features: Array.from(adapter.features || []),
        limits,
    };
}

self.onmessage = async (e) => {
    if (e.data.type !== 'init') return;
    try {
        self.postMessage(await init(e.data));
    } catch (err) {
        self.postMessage({ type: 'unavailable', reason: err && err.message ? err.message : String(err) });
    }
};
