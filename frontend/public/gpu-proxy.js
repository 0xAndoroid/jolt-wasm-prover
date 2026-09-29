/// <reference lib="webworker" />
// Dedicated Worker that owns the WebGPU device for the wasm prover.
//
// The prover threads block synchronously, so they cannot await WebGPU
// promises themselves. Instead they fill a mailbox inside the shared wasm
// memory (see src/gpu/mailbox.rs — word offsets below mirror its layout),
// bump the doorbell and Atomics.wait on `status`; this worker never blocks:
// it Atomics.waitAsync's on the doorbell, runs the op, writes the result
// back into wasm memory and flips `status`.

/** @typedef {{ptr: number, len: number, flags: number}} Region */

const MAX_ARGS = 64;
const MAX_REGIONS = 8;
const W = {
    DOORBELL: 0,
    STATUS: 1,
    OP: 2,
    ARGS: 4,
    REGIONS: 4 + MAX_ARGS,
    RET: 4 + MAX_ARGS + 3 * MAX_REGIONS,
    ERROR_LEN: 4 + MAX_ARGS + 3 * MAX_REGIONS + 8,
    ERROR: 4 + MAX_ARGS + 3 * MAX_REGIONS + 9,
    ERROR_BYTES: 256,
};
const OP = { NOP: 1, CREATE_BUFFER: 2, UPLOAD: 3, DESTROY: 4, RUN: 5, RUN_SEQ: 6, DOWNLOAD: 7, ALLOC: 8, UPLOAD_MULTI: 9 };
// RUN_SEQ args[RUN_SEQ_COPY..] = [srcRegion, dstRegion, dstByteOffset, byteLen]: a copy after the passes (byteLen 0 = none).
const RUN_SEQ_COPY = MAX_ARGS - 4;
const STATUS = { IDLE: 0, BUSY: 1, DONE: 2, ERROR: 3 };
const REGION = { UPLOAD: 1, READBACK: 2, HANDLE: 4 };
// Index = shader id in the RUN op (src/gpu/selftest.rs, src/gpu/trace_commit.rs);
// each entry lists the sources concatenated into one module (library first).
const SHADERS = [
    ['fp128', 'fp128_ops'],
    ['commit/common', 'commit/prep'],
    ['commit/common', 'commit/commit_accumulate'],
    ['commit/common', 'commit/reduce'],
    ['fp128', 'digit_range/common', 'digit_range/round0'],
    ['fp128', 'digit_range/common', 'digit_range/lut'],
    ['fp128', 'digit_range/common', 'digit_range/round1'],
    ['fp128', 'digit_range/common', 'digit_range/field'],
    ['fp128', 'digit_range/common', 'digit_range/reduce'],
    ['fp128', 'stage2/common', 'stage2/round'],
    ['fp128', 'stage2/common', 'stage2/additional'],
    ['fp128', 'stage2/common', 'stage2/reduce'],
];

/** @type {WebAssembly.Memory} */
let memory;
let base = 0;
/** @type {GPUDevice} */
let device;
/** @type {Map<string, string>} */
const shaderSources = new Map();
/** @type {Map<string, GPUComputePipeline>} */
const pipelines = new Map();
/** @type {Map<number, GPUBuffer>} */
const handles = new Map();
let nextHandle = 1;
/** @type {GPUBuffer} */
let uniformBuffer;
// One 64 B uniform buffer per pass of a RUN_SEQ (writeBuffer between passes would race).
/** @type {GPUBuffer[]} */
const uniformPool = [];
/** @type {GPUError | null} */
let uncapturedError = null;
// Test hook: stop serving the mailbox so the prover's op deadline fires.
let hung = false;

const i32 = () => new Int32Array(memory.buffer);
const u32 = () => new Uint32Array(memory.buffer);
/** @param {number} n */
const align4 = (n) => (n + 3) & ~3;

/** @param {string} name */
async function fetchText(name) {
    const r = await fetch(`/wgsl/${name}.wgsl`);
    if (!r.ok) throw new Error(`fetch ${name}.wgsl: ${r.status}`);
    return r.text();
}

// `chunk` != 0 sets the kernel's `CHUNK` override constant (commit_accumulate.wgsl);
// pipelines are cached per (shader, chunk).
/** @param {number} id @param {number} chunk */
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
    const view = new DataView(memory.buffer);
    const regions = [];
    for (let i = 0; i < MAX_REGIONS; i++) {
        const o = base + W.REGIONS + i * 3;
        regions.push({ ptr: view.getUint32(o * 4, true), len: view.getUint32((o + 1) * 4, true), flags: view.getUint32((o + 2) * 4, true) });
    }
    return regions;
}

/** @param {number} size */
function storageBuffer(size) {
    return device.createBuffer({
        size: Math.max(4, align4(size)),
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
}

/** @param {GPUBuffer} buf @param {number} offset @param {number} ptr @param {number} len */
function upload(buf, offset, ptr, len) {
    if (len > 0) device.queue.writeBuffer(buf, offset, memory.buffer, ptr, len);
}

/** @param {GPUBuffer} buf @param {number} ptr @param {number} len */
async function readback(buf, ptr, len) {
    const staging = device.createBuffer({ size: align4(len), usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    try {
        const enc = device.createCommandEncoder();
        enc.copyBufferToBuffer(buf, 0, staging, 0, align4(len));
        device.queue.submit([enc.finish()]);
        await staging.mapAsync(GPUMapMode.READ);
        new Uint8Array(memory.buffer, ptr, len).set(new Uint8Array(staging.getMappedRange(), 0, len));
        staging.unmap();
    } finally {
        staging.destroy();
    }
}

/** @param {number} h */
function getHandle(h) {
    const buf = handles.get(h);
    if (!buf) throw new Error(`unknown buffer handle ${h}`);
    return buf;
}

/** @param {number} op @param {DataView} args @param {Region[]} regions @param {Uint32Array} ret */
async function runOp(op, args, regions, ret) {
    /** @param {number} index */
    const arg = (index) => args.getUint32(index * 4, true);
    switch (op) {
        case OP.NOP:
            return;
        case OP.CREATE_BUFFER: {
            const h = nextHandle++;
            handles.set(h, storageBuffer(arg(0)));
            ret[0] = h;
            return;
        }
        case OP.UPLOAD: {
            const r = regions[0];
            if (!r) throw new RangeError("unknown region 0");
            upload(getHandle(arg(0)), arg(1), r.ptr, r.len);
            await device.queue.onSubmittedWorkDone();
            return;
        }
        case OP.DESTROY: {
            getHandle(arg(0)).destroy();
            handles.delete(arg(0));
            return;
        }
        case OP.RUN: {
            const shader = arg(0);
            const wx = arg(1), wy = arg(2), wz = arg(3);
            const nparams = arg(4);
            const params = new Uint32Array(16);
            params.set(new Uint32Array(args.buffer, args.byteOffset + 5 * 4, nparams));
            const nbind = arg(21);
            const pipeline = pipelineFor(shader, arg(22));
            device.queue.writeBuffer(uniformBuffer, 0, params);
            const entries = [{ binding: 0, resource: { buffer: uniformBuffer } }];
            /** @type {GPUBuffer[]} */
            const temps = [];
            /** @type {{buf: GPUBuffer, ptr: number, len: number}[]} */
            const readbacks = [];
            try {
                for (let i = 0; i < nbind; i++) {
                    const r = regions[i];
                    if (!r) throw new RangeError(`unknown region ${i}`);
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
        case OP.RUN_SEQ: {
            // args = [npasses, {shader, wx, nbind, (binding, region)...}...];
            // region 0 holds npasses x 64 B params; other regions are shared by the passes.
            const npasses = arg(0);
            /** @type {(GPUBuffer | undefined)[]} */
            const bufs = Array.from({length: MAX_REGIONS});
            /** @type {GPUBuffer[]} */
            const temps = [];
            /** @type {{buf: GPUBuffer, ptr: number, len: number}[]} */
            const readbacks = [];
            try {
                /** @param {number} i */
                const resolve = (i) => {
                    const cached = bufs[i];
                    if (cached) return cached;
                    const r = regions[i];
                    if (!r) throw new RangeError(`unknown region ${i}`);
                    let buf;
                    if (r.flags & REGION.HANDLE) {
                        // ptr is a handle here, not a wasm address: a readback would land at address `handle`.
                        if (r.flags & (REGION.UPLOAD | REGION.READBACK)) throw new Error(`region ${i}: handle regions cannot be uploaded or read back`);
                        buf = getHandle(r.ptr);
                    } else {
                        buf = storageBuffer(r.len);
                        temps.push(buf);
                        if (r.flags & REGION.UPLOAD) upload(buf, 0, r.ptr, r.len);
                    }
                    if (r.flags & REGION.READBACK) readbacks.push({ buf, ptr: r.ptr, len: r.len });
                    bufs[i] = buf;
                    return buf;
                };
                const params = regions[0];
                if (!params) throw new RangeError("missing parameter region");
                const enc = device.createCommandEncoder();
                let a = 1;
                for (let p = 0; p < npasses; p++) {
                    const [shader, wx, nbind] = [arg(a), arg(a + 1), arg(a + 2)];
                    a += 3;
                    const uniform = uniformPool[p] ??= device.createBuffer({ size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
                    device.queue.writeBuffer(uniform, 0, memory.buffer, params.ptr + p * 64, 64);
                    const entries = [{ binding: 0, resource: { buffer: uniform } }];
                    for (let b = 0; b < nbind; b++, a += 2) {
                        entries.push({ binding: arg(a), resource: { buffer: resolve(arg(a + 1)) } });
                    }
                    const pipeline = pipelineFor(shader, 0);
                    const pass = enc.beginComputePass();
                    pass.setPipeline(pipeline);
                    pass.setBindGroup(0, device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries }));
                    pass.dispatchWorkgroups(wx);
                    pass.end();
                }
                const copyLen = arg(RUN_SEQ_COPY + 3);
                if (copyLen) enc.copyBufferToBuffer(resolve(arg(RUN_SEQ_COPY)), 0, resolve(arg(RUN_SEQ_COPY + 1)), arg(RUN_SEQ_COPY + 2), copyLen);
                device.queue.submit([enc.finish()]);
                for (const rb of readbacks) await readback(rb.buf, rb.ptr, rb.len);
                if (readbacks.length === 0) await device.queue.onSubmittedWorkDone();
            } finally {
                for (const t of temps) t.destroy();
            }
            return;
        }
        case OP.DOWNLOAD: {
            const r = regions[0];
            if (!r) throw new RangeError("unknown region 0");
            await readback(getHandle(arg(0)), r.ptr, r.len);
            return;
        }
        case OP.ALLOC: {
            // args = [nDestroy, handles..., nCreate, sizes...]; ret = the new handles.
            let a = 1;
            for (const end = a + arg(0); a < end; a++) {
                getHandle(arg(a)).destroy();
                handles.delete(arg(a));
            }
            const nc = arg(a++);
            if (nc > ret.length) throw new Error(`alloc: ${nc} handles exceed the RET slots`);
            for (let i = 0; i < nc; i++, a++) {
                const h = nextHandle++;
                handles.set(h, storageBuffer(arg(a)));
                ret[i] = h;
            }
            return;
        }
        case OP.UPLOAD_MULTI: {
            // args = [n, {handle, byteOffset, region}...]
            for (let a = 1, i = 0; i < arg(0); i++, a += 3) {
                const r = regions[arg(a + 2)];
                if (!r) throw new RangeError(`unknown region ${arg(a + 2)}`);
                upload(getHandle(arg(a)), arg(a + 1), r.ptr, r.len);
            }
            await device.queue.onSubmittedWorkDone();
            return;
        }
        default:
            throw new Error(`unknown op ${op}`);
    }
}

/** @param {string} msg */
function writeError(msg) {
    const bytes = new TextEncoder().encode(msg).slice(0, W.ERROR_BYTES);
    new Uint8Array(memory.buffer, (base + W.ERROR) * 4, W.ERROR_BYTES).set(bytes);
    Atomics.store(u32(), base + W.ERROR_LEN, bytes.length);
}

async function serve() {
    const m = u32();
    const args = new DataView(m.buffer, (base + W.ARGS) * 4, MAX_ARGS * 4);
    const ret = new Uint32Array(m.buffer, (base + W.RET) * 4, 8);
    const op = Atomics.load(m, base + W.OP);
    const regions = readRegions();
    let status = STATUS.DONE;
    try {
        device.pushErrorScope('validation');
        device.pushErrorScope('out-of-memory');
        /** @type {unknown} */
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
        if (thrown) throw thrown instanceof Error ? thrown : new Error("GPU operation failed", {cause: thrown});
        if (err) throw new Error(err.message);
    } catch (e) {
        writeError(e instanceof Error ? e.message : String(e));
        status = STATUS.ERROR;
    }
    Atomics.store(i32(), base + W.STATUS, status);
    Atomics.notify(i32(), base + W.STATUS);
}

/** @param {number} seen */
async function waitDoorbell(seen) {
    const view = i32();
    if (typeof Atomics.waitAsync === 'function') {
        const r = Atomics.waitAsync(view, base + W.DOORBELL, seen, 1000);
        if (r.async) await r.value;
    } else {
        await new Promise((resolve) => { setTimeout(resolve, 0); });
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
        if (hung) return;
        await serve();
    }
}

/** @param {Extract<import("../types/runtime").ProxyRequest, {type: "init"}>} data */
async function init(data) {
    memory = data.memory;
    base = data.mailboxPtr >>> 2;
    if (!navigator.gpu) return { type: 'unavailable', reason: 'navigator.gpu missing (WebGPU unsupported or insecure context)' };
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) return { type: 'unavailable', reason: 'requestAdapter returned null' };
    /** @type {Record<string, number>} */
    const limits = {};
    for (const k of /** @type {const} */ (['maxBufferSize', 'maxStorageBufferBindingSize', 'maxComputeWorkgroupsPerDimension',
        'maxComputeWorkgroupSizeX', 'maxComputeInvocationsPerWorkgroup', 'maxStorageBuffersPerShaderStage',
        'maxComputeWorkgroupStorageSize'])) {
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
    void device.lost.then((info) => {
        console.error('[gpu-proxy] device lost:', info.message);
    });
    uniformBuffer = device.createBuffer({ size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    for (const name of new Set(SHADERS.flat())) shaderSources.set(name, await fetchText(name));
    const info = adapter.info || {};
    void loop().catch((/** @type {unknown} */ err) => {
        console.error("[gpu-proxy] mailbox stopped:", err);
    });
    return {
        type: 'ready',
        adapter: { vendor: info.vendor, architecture: info.architecture, device: info.device, description: info.description },
        features: Array.from(adapter.features || []),
        limits,
    };
}

self.onmessage = async (/** @type {MessageEvent<import("../types/runtime").ProxyRequest>} */ e) => {
    if (e.data.type === 'hang') hung = true;
    if (e.data.type !== 'init') return;
    try {
        self.postMessage(await init(e.data));
    } catch (err) {
        self.postMessage({ type: 'unavailable', reason: err instanceof Error ? err.message : String(err) });
    }
};
