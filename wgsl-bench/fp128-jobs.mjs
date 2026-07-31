// fp128 suite: job list + verification. Same shapes and discipline as the
// BN254 suite (jobs.mjs): KAT gating before anything timed, dependent-chain
// throughput, wg-size sweep, saturation sweep up to 2^20 threads (the native
// MSL fp128 anchor used 2^20 threads).

import { FP128_KERNELS, chainModule128 } from './fp128-kernels.mjs';
import {
  makeCtx128, katVectors128, katInputWords128, katExpected128, chainRef128, checkOutput128,
} from './fp128-ref.mjs';

export const CHAIN_N = 65536;
const KAT_K = 4; // 16 muls per thread
const SAMPLE_TIDS = [0, 1, 2, 3, CHAIN_N / 2, CHAIN_N - 1];

export function buildJobs(profile = 'full') {
  const kernelNames = Object.keys(FP128_KERNELS);
  const wgSizes = profile === 'quick' ? [128] : [64, 128, 256];
  const jobs = [];

  for (const name of kernelNames) {
    const { src, ctx } = chainModule128(name, 64);
    const pairs = katVectors128(ctx);
    jobs.push({
      type: 'kat',
      label: `kat/${name}`,
      kernel: name,
      shader: src,
      nThreads: pairs.length,
      wgSize: 64,
      k: KAT_K,
      inputWords: katInputWords128(ctx, pairs),
      outWords: pairs.length * ctx.L,
      wordsPerElem: ctx.L,
    });
  }

  for (const name of kernelNames) {
    for (const wg of wgSizes) {
      const { src, ctx } = chainModule128(name, wg);
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
    for (const name of kernelNames) {
      for (const n of [262144, 1048576]) {
        const { src, ctx } = chainModule128(name, 64);
        jobs.push({
          type: 'chain_timed',
          label: `sat/${name}/n${n}`,
          kernel: name,
          shader: src,
          nThreads: n,
          wgSize: 64,
          outWords: n * ctx.L,
          wordsPerElem: ctx.L,
          sampleTids: [0, n - 1],
          reps: 3,
        });
      }
    }
  }

  return jobs;
}

function median(arr) {
  const s = [...arr].sort((a, b) => a - b);
  return s[Math.floor(s.length / 2)];
}

export function verifyAndRate(jobs, results) {
  const rows = [];
  for (let i = 0; i < jobs.length; i++) {
    const job = jobs[i];
    const res = results[i];
    if (!res || res.error) {
      rows.push({ label: job.label, error: res ? res.error : 'missing result' });
      continue;
    }
    const k = FP128_KERNELS[job.kernel];
    const ctx = makeCtx128(k.W, k.L);
    const row = {
      label: job.label, type: job.type,
      moduleMs: res.moduleMs, pipelineMs: res.pipelineMs,
    };

    if (job.type === 'kat') {
      const pairs = katVectors128(ctx);
      const expected = katExpected128(ctx, pairs, 4 * KAT_K);
      const fails = [];
      for (let t = 0; t < pairs.length; t++) {
        const limbs = res.out.slice(t * ctx.L, (t + 1) * ctx.L);
        const chk = checkOutput128(ctx, limbs, expected[t]);
        if (!chk.ok) fails.push(`pair${t}: ${chk.why}`);
      }
      row.ok = fails.length === 0;
      row.fails = fails;
    } else if (job.type === 'chain_timed') {
      const fails = [];
      for (const tid of job.sampleTids) {
        const limbs = res.samples[tid];
        const chk = checkOutput128(ctx, limbs, chainRef128(ctx, tid, 4 * res.k));
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
    }
    rows.push(row);
  }
  return rows;
}
