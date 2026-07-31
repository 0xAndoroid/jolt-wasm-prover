// Builds the benchmark job list and verifies GPU results against the BigInt
// reference. Verification is mandatory: a timed result whose samples fail is
// reported as INVALID and excluded from verdicts.

import { KERNELS, chainModule, streamModule } from './kernels.mjs';
import {
  P, makeCtx, katVectors, katInputWords, katExpected, chainRef, streamRef, checkOutput,
} from './ref.mjs';

export const CHAIN_N = 65536;
export const STREAM_N = 1 << 20;
const KAT_K = 4; // 16 muls per thread
const SAMPLE_TIDS = [0, 1, 2, 3, CHAIN_N / 2, CHAIN_N - 1];
const STREAM_SAMPLES = [0, 1, 12345, STREAM_N - 1];

export function buildJobs(profile = 'full') {
  const kernelNames = profile === 'quick' ? ['cios16_june', 'lazy13u'] : Object.keys(KERNELS);
  const wgSizes = profile === 'quick' ? [128] : [64, 128, 256];
  const jobs = [];

  for (const name of kernelNames) {
    const { src, ctx } = chainModule(name, 64);
    const pairs = katVectors(ctx);
    jobs.push({
      type: 'kat',
      label: `kat/${name}`,
      kernel: name,
      shader: src,
      nThreads: pairs.length,
      wgSize: 64,
      k: KAT_K,
      inputWords: katInputWords(ctx, pairs),
      outWords: pairs.length * ctx.L,
      wordsPerElem: ctx.L,
    });
  }

  for (const name of kernelNames) {
    for (const wg of wgSizes) {
      const { src, ctx } = chainModule(name, wg);
      jobs.push({
        type: 'chain_timed',
        label: `chain/${name}/wg${wg}`,
        kernel: name,
        shader: src,
        nThreads: CHAIN_N,
        wgSize: wg,
        outWords: CHAIN_N * ctx.L,
        wordsPerElem: ctx.L,
        sampleTids: SAMPLE_TIDS,
        reps: 3,
      });
    }
  }

  if (profile === 'full') {
    for (const name of ['lazy13u', 'cios16_june']) {
      for (const n of [4096, 16384, 262144]) {
        const { src, ctx } = chainModule(name, 128);
        jobs.push({
          type: 'chain_timed',
          label: `sat/${name}/n${n}`,
          kernel: name,
          shader: src,
          nThreads: n,
          wgSize: 128,
          outWords: n * ctx.L,
          wordsPerElem: ctx.L,
          sampleTids: [0, n - 1],
          reps: 3,
        });
      }
    }

    for (const kind of ['june8', 'raw13', 'packed13']) {
      const { src, ctx, wordsPerElem } = streamModule(kind, 256);
      jobs.push({
        type: 'stream_timed',
        label: `stream/${kind}`,
        kernel: kind,
        shader: src,
        nElems: STREAM_N,
        wgSize: 256,
        wordsPerElem,
        outWords: STREAM_N * wordsPerElem,
        streamWords: STREAM_N * wordsPerElem,
        sampleIdx: STREAM_SAMPLES,
        reps: 3,
      });
    }
  }

  {
    const { src, ctx } = chainModule('lazy13u', 64);
    jobs.push({
      type: 'overhead',
      label: 'overhead',
      kernel: 'lazy13u',
      shader: src,
      nThreads: 256,
      wgSize: 64,
      outWords: 256 * ctx.L,
      wordsPerElem: ctx.L,
    });
  }

  return jobs;
}

function median(arr) {
  const s = [...arr].sort((a, b) => a - b);
  return s[Math.floor(s.length / 2)];
}

const ctxFor = (job) => {
  if (job.type === 'stream_timed') {
    const W = job.kernel === 'june8' ? 16 : 13;
    return makeCtx(W, W === 16 ? 16 : 20);
  }
  const k = KERNELS[job.kernel];
  return makeCtx(k.W, k.L);
};

// Returns annotated rows: verification verdicts and Gmul/s rates.
export function verifyAndRate(jobs, results) {
  const rows = [];
  for (let i = 0; i < jobs.length; i++) {
    const job = jobs[i];
    const res = results[i];
    if (!res || res.error) {
      rows.push({ label: job.label, error: res ? res.error : 'missing result' });
      continue;
    }
    const ctx = ctxFor(job);
    const row = {
      label: job.label, type: job.type,
      moduleMs: res.moduleMs, pipelineMs: res.pipelineMs,
    };

    if (job.type === 'kat') {
      const pairs = katVectors(ctx);
      const expected = katExpected(ctx, pairs, 4 * KAT_K);
      const fails = [];
      for (let t = 0; t < pairs.length; t++) {
        const limbs = res.out.slice(t * ctx.L, (t + 1) * ctx.L);
        const chk = checkOutput(ctx, limbs, expected[t]);
        if (!chk.ok) fails.push(`pair${t}: ${chk.why}`);
      }
      row.ok = fails.length === 0;
      row.fails = fails;
    } else if (job.type === 'chain_timed') {
      const fails = [];
      for (const tid of job.sampleTids) {
        const limbs = res.samples[tid];
        const chk = checkOutput(ctx, limbs, chainRef(ctx, tid, 4 * res.k));
        if (!chk.ok) fails.push(`tid${tid}: ${chk.why}`);
      }
      row.ok = fails.length === 0;
      row.fails = fails;
      row.k = res.k;
      row.d = res.d;
      row.checksum = res.checksum;
      const muls = res.mulsPerDispatch * res.d;
      row.wallMedianMs = median(res.wallMs);
      row.gmulWall = muls / (row.wallMedianMs * 1e6);
      if (res.gpuMs) {
        row.gpuMedianMs = median(res.gpuMs);
        row.gmulGpu = muls / (row.gpuMedianMs * 1e6);
      }
    } else if (job.type === 'stream_timed') {
      const fails = [];
      for (const idx of job.sampleIdx) {
        const words = res.samples[idx];
        const expected = streamRef(ctx, idx);
        let chk;
        if (job.wordsPerElem === 8) {
          // Packed layouts store the value as 8 LE u32 words, not limbs.
          let v = 0n;
          for (let j = 7; j >= 0; j--) v = (v << 32n) | BigInt(words[j] >>> 0);
          chk = (v < 2n * P && v % P === expected)
            ? { ok: true }
            : { ok: false, why: `packed mismatch: got ${v.toString(16)}, want ${expected.toString(16)}` };
        } else {
          chk = checkOutput(ctx, words, expected);
        }
        if (!chk.ok) fails.push(`idx${idx}: ${chk.why}`);
      }
      row.ok = fails.length === 0;
      row.fails = fails;
      row.d = res.d;
      row.checksum = res.checksum;
      const muls = res.mulsPerDispatch * res.d;
      row.wallMedianMs = median(res.wallMs);
      row.gmulWall = muls / (row.wallMedianMs * 1e6);
      if (res.gpuMs) {
        row.gpuMedianMs = median(res.gpuMs);
        row.gmulGpu = muls / (row.gpuMedianMs * 1e6);
      }
      const bytes = 3 * job.wordsPerElem * 4 * res.mulsPerDispatch * res.d;
      row.gbps = bytes / ((row.gpuMedianMs || row.wallMedianMs) * 1e6);
    } else if (job.type === 'overhead') {
      row.ok = true;
      row.perDispatchInPassMs = median(res.perDispatchInPassMs);
      row.perDispatchInPassGpuMs = res.perDispatchInPassGpuMs ? median(res.perDispatchInPassGpuMs) : null;
      row.singleSubmitMedianMs = median(res.singleSubmitMs);
      row.singleSubmitMinMs = Math.min(...res.singleSubmitMs);
    }
    rows.push(row);
  }
  return rows;
}
