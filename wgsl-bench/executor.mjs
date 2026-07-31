// Harness-agnostic GPU job executor. Runs in Node (node-webgpu) and in
// Chrome (imported into the page as a data: URL), so it must stay
// self-contained: no imports, no BigInt literals in hot paths, results
// JSON-serializable.
//
// Timing model: each timed rep encodes D dispatches inside ONE compute pass
// of ONE submit. Wall time = submit -> onSubmittedWorkDone. GPU time = pass
// begin/end timestamp queries when the device has timestamp-query. Dispatch
// overhead is measured separately with k=0 dispatches and subtracted by the
// caller, never baked into reported rates.

export async function runJobs(gpu, jobs) {
  const adapter = await gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) return { error: 'no adapter' };
  const info = adapter.info || {};
  const adapterInfo = {
    vendor: info.vendor || '',
    architecture: info.architecture || '',
    device: info.device || '',
    description: info.description || '',
  };
  const hasTs = adapter.features.has('timestamp-query');
  const requiredFeatures = hasTs ? ['timestamp-query'] : [];
  const device = await adapter.requestDevice({ requiredFeatures });
  let lost = null;
  device.lost.then((l) => { lost = `${l.reason}: ${l.message}`; });

  // node-webgpu needs its event loop pumped for callbacks on some versions;
  // harmless no-op in the browser.
  const tick = typeof device.tick === 'function' ? () => device.tick() : null;
  const tickTimer = tick ? setInterval(tick, 1) : null;

  const results = [];
  for (const job of jobs) {
    try {
      results.push(await runJob(device, job, hasTs));
    } catch (e) {
      results.push({ label: job.label, error: String(e && e.message || e) });
    }
    if (lost) {
      results.push({ label: 'device-lost', error: lost });
      break;
    }
  }
  if (tickTimer) clearInterval(tickTimer);
  device.destroy();
  return { adapterInfo, hasTs, results };
}

function now() {
  return (typeof performance !== 'undefined' ? performance.now() : Date.now());
}

async function makePipeline(device, shader, entry) {
  const t0 = now();
  device.pushErrorScope('validation');
  const module = device.createShaderModule({ code: shader });
  const moduleMs = now() - t0;
  // Async creation: Chrome's sync path returns before compiling (0ms lie);
  // the async promise resolves after the real Tint->MSL->AIR compile.
  const t1 = now();
  const desc = { layout: 'auto', compute: { module, entryPoint: entry } };
  const pipeline = device.createComputePipelineAsync
    ? await device.createComputePipelineAsync(desc)
    : device.createComputePipeline(desc);
  const pipelineMs = now() - t1;
  const err = await device.popErrorScope();
  if (err) {
    let msgs = '';
    if (module.getCompilationInfo) {
      const ci = await module.getCompilationInfo();
      msgs = ci.messages.map((m) => `${m.lineNum}:${m.linePos} ${m.message}`).join(' | ');
    }
    throw new Error(`pipeline validation: ${err.message} ${msgs}`.slice(0, 800));
  }
  return { module, pipeline, moduleMs, pipelineMs };
}

function makeBuffers(device, job) {
  const bufs = {};
  bufs.outbuf = device.createBuffer({
    size: Math.max(4, job.outWords * 4),
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
  bufs.params = device.createBuffer({
    size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  if (job.inputWords) {
    bufs.inbuf = device.createBuffer({
      size: Math.max(4, job.inputWords.length * 4),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(bufs.inbuf, 0, new Uint32Array(job.inputWords));
  }
  if (job.streamWords) {
    for (const name of ['in_a', 'in_b']) {
      bufs[name] = device.createBuffer({
        size: job.streamWords * 4,
        usage: GPUBufferUsage.STORAGE,
      });
    }
  }
  return bufs;
}

// bindingMap: array of [bindingIndex, bufferName]
function bindGroupFor(device, pipeline, bufs, bindingMap) {
  return device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: bindingMap.map(([binding, name]) => ({ binding, resource: { buffer: bufs[name] } })),
  });
}

function writeParams(device, bufs, k, n, flags) {
  device.queue.writeBuffer(bufs.params, 0, new Uint32Array([k, n, flags, 0]));
}

async function readBack(device, srcBuf, words) {
  const staging = device.createBuffer({
    size: words * 4,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });
  const enc = device.createCommandEncoder();
  enc.copyBufferToBuffer(srcBuf, 0, staging, 0, words * 4);
  device.queue.submit([enc.finish()]);
  await staging.mapAsync(GPUMapMode.READ);
  const data = new Uint32Array(staging.getMappedRange().slice(0));
  staging.unmap();
  staging.destroy();
  return data;
}

function checksumOf(data) {
  let x = 0, s = 0;
  for (let i = 0; i < data.length; i++) {
    x ^= data[i];
    s = (s + data[i]) >>> 0;
  }
  return `${x >>> 0}:${s}`;
}

async function submitAndWait(device, passes) {
  const t0 = now();
  device.queue.submit(passes);
  await device.queue.onSubmittedWorkDone();
  return now() - t0;
}

function encodePass(device, pipeline, bindGroup, workgroups, dispatches, tsCtx) {
  const enc = device.createCommandEncoder();
  const desc = {};
  if (tsCtx) {
    desc.timestampWrites = {
      querySet: tsCtx.querySet,
      beginningOfPassWriteIndex: 0,
      endOfPassWriteIndex: 1,
    };
  }
  const pass = enc.beginComputePass(desc);
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  for (let d = 0; d < dispatches; d++) pass.dispatchWorkgroups(workgroups);
  pass.end();
  if (tsCtx) {
    enc.resolveQuerySet(tsCtx.querySet, 0, 2, tsCtx.resolve, 0);
    enc.copyBufferToBuffer(tsCtx.resolve, 0, tsCtx.staging, 0, 16);
  }
  return enc.finish();
}

function makeTsCtx(device) {
  return {
    querySet: device.createQuerySet({ type: 'timestamp', count: 2 }),
    resolve: device.createBuffer({ size: 16, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC }),
    staging: device.createBuffer({ size: 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST }),
  };
}

async function readTsMs(tsCtx) {
  await tsCtx.staging.mapAsync(GPUMapMode.READ);
  const v = new BigUint64Array(tsCtx.staging.getMappedRange().slice(0));
  tsCtx.staging.unmap();
  return Number(v[1] - v[0]) / 1e6;
}

async function runJob(device, job, hasTs) {
  if (job.type === 'kat') return runKat(device, job);
  if (job.type === 'chain_timed') return runChainTimed(device, job, hasTs);
  if (job.type === 'stream_timed') return runStreamTimed(device, job, hasTs);
  if (job.type === 'overhead') return runOverhead(device, job, hasTs);
  throw new Error(`unknown job type ${job.type}`);
}

async function runKat(device, job) {
  const { pipeline, moduleMs, pipelineMs } = await makePipeline(device, job.shader, 'main_kat');
  const bufs = makeBuffers(device, job);
  writeParams(device, bufs, job.k, job.nThreads, 0);
  const bg = bindGroupFor(device, pipeline, bufs, [[0, 'outbuf'], [1, 'params'], [2, 'inbuf']]);
  const workgroups = Math.ceil(job.nThreads / job.wgSize);
  const cmd = encodePass(device, pipeline, bg, workgroups, 1, null);
  await submitAndWait(device, [cmd]);
  const data = await readBack(device, bufs.outbuf, job.outWords);
  for (const b of Object.values(bufs)) b.destroy();
  return { label: job.label, moduleMs, pipelineMs, out: Array.from(data) };
}

// Calibrates k (chain loop count) and D (dispatches per pass) from a warmup
// dispatch, then runs `reps` timed passes.
async function runChainTimed(device, job, hasTs) {
  const { pipeline, moduleMs, pipelineMs } = await makePipeline(device, job.shader, 'main_bench');
  const bufs = makeBuffers(device, job);
  const bg = bindGroupFor(device, pipeline, bufs, [[0, 'outbuf'], [1, 'params']]);
  const workgroups = Math.ceil(job.nThreads / job.wgSize);

  const kCal = job.kCal || 64;
  writeParams(device, bufs, kCal, job.nThreads, 0);
  await submitAndWait(device, [encodePass(device, pipeline, bg, workgroups, 1, null)]);
  const calMs = await submitAndWait(device, [encodePass(device, pipeline, bg, workgroups, 1, null)]);

  const targetDispatchMs = job.targetDispatchMs || 30;
  let k = Math.round(kCal * targetDispatchMs / Math.max(calMs, 0.5));
  k = Math.max(64, Math.min(4096, k));
  const dispatchMsEst = calMs * k / kCal;
  const d = Math.max(2, Math.min(16, Math.round((job.targetPassMs || 120) / dispatchMsEst)));
  writeParams(device, bufs, k, job.nThreads, 0);

  const tsCtx = hasTs ? makeTsCtx(device) : null;
  const wallMs = [], gpuMs = [];
  for (let rep = 0; rep < (job.reps || 3); rep++) {
    const cmd = encodePass(device, pipeline, bg, workgroups, d, tsCtx);
    wallMs.push(await submitAndWait(device, [cmd]));
    if (tsCtx) gpuMs.push(await readTsMs(tsCtx));
  }

  const data = await readBack(device, bufs.outbuf, job.outWords);
  const samples = {};
  for (const tid of job.sampleTids || []) {
    samples[tid] = Array.from(data.subarray(tid * job.wordsPerElem, (tid + 1) * job.wordsPerElem));
  }
  const checksum = checksumOf(data);
  for (const b of Object.values(bufs)) b.destroy();
  return {
    label: job.label, moduleMs, pipelineMs, calMs, k, d,
    mulsPerDispatch: job.nThreads * 4 * k,
    wallMs, gpuMs: tsCtx ? gpuMs : null, checksum, samples,
  };
}

async function runStreamTimed(device, job, hasTs) {
  const fill = await makePipeline(device, job.shader, 'main_fill');
  const { pipeline, moduleMs, pipelineMs } = await makePipeline(device, job.shader, 'main_stream');
  const bufs = makeBuffers(device, job);
  const workgroups = Math.ceil(job.nElems / job.wgSize);
  writeParams(device, bufs, 0, job.nElems, 0);

  const fillBg = bindGroupFor(device, fill.pipeline, bufs, [[1, 'params'], [2, 'in_a'], [3, 'in_b']]);
  await submitAndWait(device, [encodePass(device, fill.pipeline, fillBg, workgroups, 1, null)]);

  const bg = bindGroupFor(device, pipeline, bufs, [[0, 'outbuf'], [1, 'params'], [2, 'in_a'], [3, 'in_b']]);
  const calMs = await submitAndWait(device, [encodePass(device, pipeline, bg, workgroups, 1, null)]);
  const d = Math.max(2, Math.min(32, Math.round((job.targetPassMs || 120) / Math.max(calMs, 0.5))));

  const tsCtx = hasTs ? makeTsCtx(device) : null;
  const wallMs = [], gpuMs = [];
  for (let rep = 0; rep < (job.reps || 3); rep++) {
    const cmd = encodePass(device, pipeline, bg, workgroups, d, tsCtx);
    wallMs.push(await submitAndWait(device, [cmd]));
    if (tsCtx) gpuMs.push(await readTsMs(tsCtx));
  }

  const data = await readBack(device, bufs.outbuf, job.outWords);
  const samples = {};
  for (const idx of job.sampleIdx || []) {
    samples[idx] = Array.from(data.subarray(idx * job.wordsPerElem, (idx + 1) * job.wordsPerElem));
  }
  const checksum = checksumOf(data);
  for (const b of Object.values(bufs)) b.destroy();
  return {
    label: job.label, moduleMs, pipelineMs, calMs, d,
    mulsPerDispatch: job.nElems,
    wallMs, gpuMs: tsCtx ? gpuMs : null, checksum, samples,
  };
}

// Dispatch-cost microbench with k=0 chains (derive + write only):
// (a) per-dispatch cost inside one pass, (b) full submit->done latency.
async function runOverhead(device, job, hasTs) {
  const { pipeline } = await makePipeline(device, job.shader, 'main_bench');
  const bufs = makeBuffers(device, job);
  writeParams(device, bufs, 0, job.nThreads, 0);
  const bg = bindGroupFor(device, pipeline, bufs, [[0, 'outbuf'], [1, 'params']]);
  const workgroups = Math.ceil(job.nThreads / job.wgSize);

  await submitAndWait(device, [encodePass(device, pipeline, bg, workgroups, 1, null)]);

  const tsCtx = hasTs ? makeTsCtx(device) : null;
  const D = 64;
  const inPass = [], inPassGpu = [];
  for (let rep = 0; rep < 5; rep++) {
    const cmd = encodePass(device, pipeline, bg, workgroups, D, tsCtx);
    inPass.push(await submitAndWait(device, [cmd]) / D);
    if (tsCtx) inPassGpu.push(await readTsMs(tsCtx) / D);
  }
  const single = [];
  for (let rep = 0; rep < 32; rep++) {
    single.push(await submitAndWait(device, [encodePass(device, pipeline, bg, workgroups, 1, null)]));
  }
  for (const b of Object.values(bufs)) b.destroy();
  return {
    label: job.label,
    perDispatchInPassMs: inPass,
    perDispatchInPassGpuMs: tsCtx ? inPassGpu : null,
    singleSubmitMs: single,
  };
}
