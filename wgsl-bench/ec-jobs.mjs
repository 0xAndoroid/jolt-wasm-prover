// Job list + verification for the composed-EC suite (W2a).
//
// Protocol: KATs (real BN254 G1 points incl. every guard path) are
// limb-exact against the BigInt reference; timed jobs re-verify sampled
// threads limb-exact against the same mirrors for the calibrated k. A row
// with ok=false is excluded from verdicts.
//
// Rates: gmul = fe_mont_mul calls / time (S=M, adds free — W1d convention,
// comparable to the 0.928 Gmul/s roof); mops = point-ops (or scan iters) /s.

import {
  ecSelfTest, enc, dec, mm, fsub, dv, ONE_M,
  jacIdentity, jacDbl, jacMadd, xyzzMadd, xyzzIdentity, juneLadder,
  genMultiples, feLimbs, jacLimbs, xyzzLimbs,
  dblChainRef, maddChainRef, binvChainRef, bucketRef, juneFoldRef,
} from './ec-ref.mjs';
import {
  ecDblModule, ecMaddModule, ecXyzzModule, ecBinvModule, ecBucketModule,
  ecJuneFoldModule, MULS, K1W, K2W, juneMulsForK,
} from './ec-kernels.mjs';

const CHAIN_N = 65536;
const BUCKET_K = 64;
const SAMPLE_TIDS = [0, 1, 2, 3, CHAIN_N / 2, CHAIN_N - 1];

// ---------------------------------------------------------------------------
// KAT vector builders (all coordinates Montgomery-domain BigInts)
// ---------------------------------------------------------------------------

function denormRep(aff, zPlain) {
  // Random-Z Jacobian representative: (x·z², y·z³, z).
  const z = enc(zPlain);
  const z2 = mm(z, z);
  return { x: mm(aff.x, z2), y: mm(mm(aff.y, z2), z), z };
}

function buildKats() {
  const M = genMultiples(6); // [G, 2G, ..., 6G] affine, Montgomery-form
  const G = M[0];
  const gJac = { x: G.x, y: G.y, z: ONE_M };
  const negG = { x: G.x, y: fsub(0n, G.y) };
  const pseudoJac = (s) => ({ x: dv(s), y: dv(s + 1), z: dv(s + 2) });

  const kats = [];

  {
    const vecs = [gJac, denormRep(M[2], 7n), jacIdentity(), pseudoJac(1001)];
    const k = 4;
    kats.push({
      name: 'ec_dbl', module: ecDblModule(64, 1), k,
      inputWords: vecs.flatMap(jacLimbs),
      expected: vecs.map((v) => {
        let a = v;
        for (let i = 0; i < k; i++) a = jacDbl(a);
        return jacLimbs(a);
      }),
    });
  }

  {
    // Guard coverage: z=0 promote, H=0&r=0 double, H=0 infinity.
    const rows = [
      [gJac, M[1]],
      [denormRep(M[1], 11n), G],
      [jacIdentity(), G],
      [gJac, G],
      [gJac, negG],
      [pseudoJac(1101), { x: dv(1104), y: dv(1105) }],
    ];
    const k = 2;
    kats.push({
      name: 'ec_madd', module: ecMaddModule(64, 1), k,
      inputWords: rows.flatMap(([a, q]) => [...jacLimbs(a), ...feLimbs(q.x), ...feLimbs(q.y)]),
      expected: rows.map(([a0, q]) => {
        let a = a0;
        for (let i = 0; i < k; i++) a = jacMadd(a, q.x, q.y);
        return jacLimbs(a);
      }),
    });
  }

  {
    // No-guard variant: vectors must avoid guard paths.
    const rows = [
      [gJac, M[1]],
      [denormRep(M[4], 13n), M[2]],
      [pseudoJac(1201), { x: dv(1204), y: dv(1205) }],
    ];
    const k = 3;
    kats.push({
      name: 'ec_madd_nogd', module: ecMaddModule(64, 1, { guarded: false }), k,
      inputWords: rows.flatMap(([a, q]) => [...jacLimbs(a), ...feLimbs(q.x), ...feLimbs(q.y)]),
      expected: rows.map(([a0, q]) => {
        let a = a0;
        for (let i = 0; i < k; i++) a = jacMadd(a, q.x, q.y);
        return jacLimbs(a);
      }),
    });
  }

  {
    const gX = { x: G.x, y: G.y, zz: ONE_M, zzz: ONE_M };
    const rows = [
      [gX, M[1]],
      [xyzzIdentity(), G],
      [gX, G],
      [gX, negG],
      [{ x: dv(1301), y: dv(1302), zz: dv(1303), zzz: dv(1304) }, { x: dv(1305), y: dv(1306) }],
    ];
    const k = 2;
    kats.push({
      name: 'ec_xyzz', module: ecXyzzModule(64, 1), k,
      inputWords: rows.flatMap(([a, q]) => [...xyzzLimbs(a), ...feLimbs(q.x), ...feLimbs(q.y)]),
      expected: rows.map(([a0, q]) => {
        let a = a0;
        for (let i = 0; i < k; i++) a = xyzzMadd(a, q.x, q.y);
        return xyzzLimbs(a);
      }),
    });
  }

  {
    const rows = [
      [ONE_M, ONE_M, ONE_M],
      [enc(2n), fsub(0n, enc(1n)), enc(3n)],
      [dv(1401), dv(1402), dv(1403)],
      [0n, dv(1404), dv(1405)],
    ];
    const k = 4;
    kats.push({
      name: 'ec_binv', module: ecBinvModule(64, 1), k,
      inputWords: rows.flatMap((r) => r.flatMap(feLimbs)),
      expected: rows.map(([e, acc0, t0]) => {
        let acc = acc0, t = t0;
        for (let i = 0; i < k; i++) {
          acc = mm(acc, e);
          t = mm(t, acc);
        }
        return [...feLimbs(acc), ...feLimbs(t)];
      }),
    });
  }

  {
    const rows = [
      [gJac, [M[1], M[2], M[3], M[4]]],
      [jacIdentity(), [G, M[1], M[2], M[3]]],
      [pseudoJac(1501), [0, 1, 2, 3].map((i) => ({ x: dv(1510 + 2 * i), y: dv(1511 + 2 * i) }))],
    ];
    const k = 6; // wraps the 4-point cycle
    kats.push({
      name: 'ec_bucket', module: ecBucketModule(64, BUCKET_K), k,
      inputWords: rows.flatMap(([a, pts]) =>
        [...jacLimbs(a), ...pts.flatMap((q) => [...feLimbs(q.x), ...feLimbs(q.y)])]),
      expected: rows.map(([a0, pts]) => {
        let a = a0;
        for (let i = 0; i < k; i++) a = jacMadd(a, pts[i & 3].x, pts[i & 3].y);
        return jacLimbs(a);
      }),
    });
  }

  {
    const rows = [
      { x: G.x, y: G.y, z: ONE_M },
      { x: dv(1601), y: dv(1602), z: dv(1603) },
      { x: dv(1604), y: dv(1605), z: 0n },
    ];
    const k = 24;
    kats.push({
      name: 'ec_june_fold', module: ecJuneFoldModule(64), k,
      inputWords: rows.flatMap(jacLimbs),
      expected: rows.map((p1) => jacLimbs(juneLadder(p1, k, K1W, K2W))),
    });
  }

  return kats;
}

// ---------------------------------------------------------------------------
// Jobs
// ---------------------------------------------------------------------------

export function buildJobs(profile = 'full') {
  ecSelfTest();
  const jobs = [];

  for (const kat of buildKats()) {
    const n = kat.expected.length;
    jobs.push({
      type: 'kat',
      label: `kat/${kat.name}`,
      shader: kat.module.src,
      nThreads: n,
      wgSize: 64,
      k: kat.k,
      inputWords: kat.inputWords,
      outWords: n * kat.module.wordsPerElem,
      wordsPerElem: kat.module.wordsPerElem,
      expected: kat.expected,
    });
  }

  const chainJob = (kind, moduleFn, wg, unroll, n = CHAIN_N, extra = {}) => {
    const m = moduleFn(wg, unroll);
    return {
      type: 'chain_timed',
      label: `${kind}/wg${wg}/u${unroll}${n !== CHAIN_N ? `/n${n}` : ''}`,
      kind, unroll,
      shader: m.src,
      nThreads: n,
      wgSize: wg,
      outWords: n * m.wordsPerElem,
      wordsPerElem: m.wordsPerElem,
      mulsPerK: m.mulsPerK,
      opsPerK: m.opsPerK,
      sampleTids: n === CHAIN_N ? SAMPLE_TIDS : [0, 1, n / 2, n - 1],
      reps: 3,
      // Small kCal/kMin keep calibration+timed dispatches bounded (~seconds)
      // even when a config lands on the collapsed tier (~0.02 Gmul/s).
      kCal: 8,
      kMin: 4,
      ...extra,
    };
  };

  const bucketJob = (wg, n, K) => {
    const m = ecBucketModule(wg, K);
    return {
      type: 'ec_timed',
      label: `bucket/wg${wg}/K${K}${n !== CHAIN_N ? `/n${n}` : ''}`,
      kind: 'bucket', K,
      shader: m.src,
      nThreads: n,
      wgSize: wg,
      outWords: n * m.wordsPerElem,
      wordsPerElem: m.wordsPerElem,
      mulsPerK: m.mulsPerK,
      nPoints: n * K,
      pointsWords: n * K * m.pointWords,
      sampleTids: n === CHAIN_N ? SAMPLE_TIDS : [0, 1, n / 2, n - 1],
      reps: 3,
      kCal: 8,
      kMin: 4,
    };
  };

  const juneJob = (wg, n = CHAIN_N) => {
    const m = ecJuneFoldModule(wg);
    return {
      type: 'ec_timed',
      label: `june_fold/wg${wg}${n !== CHAIN_N ? `/n${n}` : ''}`,
      kind: 'june',
      shader: m.src,
      nThreads: n,
      wgSize: wg,
      outWords: n * m.wordsPerElem,
      wordsPerElem: m.wordsPerElem,
      nPoints: n,
      pointsWords: n * m.pointWords,
      sampleTids: n === CHAIN_N ? SAMPLE_TIDS : [0, 1, n / 2, n - 1],
      reps: 3,
      kCal: 4,
      kMin: 2,
    };
  };

  if (profile === 'quick') {
    jobs.push(chainJob('dbl', ecDblModule, 64, 1));
    jobs.push(chainJob('madd', ecMaddModule, 64, 1));
    jobs.push(chainJob('binv', ecBinvModule, 64, 1));
    jobs.push(bucketJob(64, CHAIN_N, BUCKET_K));
    jobs.push(juneJob(64));
    return jobs;
  }

  // madd unroll 3/4 are EXCLUDED everywhere: they sit past the catastrophic
  // composition cliff (~10^4-10^5x below roof, ~100s+ per tiny dispatch —
  // measured via ec-probe.mjs, documented in the W2a report). The collapsed
  // tier (~0.02-0.1 Gmul/s: wg>=128 composed kernels, u2 straight-line) stays
  // in-suite — measurable with kMin=4. Profiles: 'safe' = wg 32/64 grid +
  // ladders (known-good envelope), 'risky' = wg 128/256 arm, 'full' = both.
  const safe = profile === 'safe' || profile === 'full';
  const risky = profile === 'risky' || profile === 'full';
  const wgs = [...(safe ? [32, 64] : []), ...(risky ? [128, 256] : [])];

  for (const wg of wgs) {
    for (const u of [1, 2]) {
      jobs.push(chainJob('dbl', ecDblModule, wg, u));
      jobs.push(chainJob('madd', ecMaddModule, wg, u));
    }
    jobs.push(chainJob('dbl', ecDblModule, wg, 4));
    jobs.push(chainJob('madd_nogd', (w, u) => ecMaddModule(w, u, { guarded: false }), wg, 1));
    jobs.push(chainJob('xyzz', ecXyzzModule, wg, 1));
    jobs.push(chainJob('binv', ecBinvModule, wg, 1));
    jobs.push(bucketJob(wg, CHAIN_N, BUCKET_K));
    jobs.push(juneJob(wg));
  }
  if (safe) {
    jobs.push(chainJob('madd_nogd', (w, u) => ecMaddModule(w, u, { guarded: false }), 32, 2));
    // Saturation ladders at the known-good shape (wg64).
    for (const n of [4096, 16384, 262144]) jobs.push(chainJob('madd', ecMaddModule, 64, 1, n));
    jobs.push(chainJob('dbl', ecDblModule, 64, 1, 262144));
    jobs.push(chainJob('binv', ecBinvModule, 64, 1, 262144));
    jobs.push(bucketJob(64, 16384, BUCKET_K));
    jobs.push(bucketJob(64, 262144, 16));
  }

  return jobs;
}

// ---------------------------------------------------------------------------
// Verification + rating
// ---------------------------------------------------------------------------

function median(arr) {
  const s = [...arr].sort((a, b) => a - b);
  return s[Math.floor(s.length / 2)];
}

function limbsEqual(got, want) {
  if (!got || got.length !== want.length) return false;
  for (let i = 0; i < want.length; i++) if ((got[i] >>> 0) !== want[i]) return false;
  return true;
}

function chainMirror(job, tid, iters) {
  switch (job.kind) {
    case 'dbl': return jacLimbs(dblChainRef(tid, iters));
    case 'madd': case 'madd_nogd': return jacLimbs(maddChainRef(tid, iters));
    case 'xyzz': return xyzzLimbs(maddChainRef(tid, iters, true));
    case 'binv': {
      const [acc, t] = binvChainRef(tid, iters);
      return [...feLimbs(acc), ...feLimbs(t)];
    }
    default: throw new Error(`unknown chain kind ${job.kind}`);
  }
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
    const row = { label: job.label, type: job.type, moduleMs: res.moduleMs, pipelineMs: res.pipelineMs };

    if (job.type === 'kat') {
      const fails = [];
      for (let t = 0; t < job.expected.length; t++) {
        const got = res.out.slice(t * job.wordsPerElem, (t + 1) * job.wordsPerElem);
        if (!limbsEqual(got, job.expected[t])) {
          fails.push(`vec${t}: got [${got.slice(0, 4)}...] want [${job.expected[t].slice(0, 4)}...]`);
        }
      }
      row.ok = fails.length === 0;
      row.fails = fails;
    } else {
      const fails = [];
      let mulsPerThread;
      if (job.type === 'chain_timed') {
        const iters = res.k * job.unroll;
        for (const tid of job.sampleTids) {
          if (!limbsEqual(res.samples[tid], chainMirror(job, tid, iters))) fails.push(`tid${tid}`);
        }
        mulsPerThread = res.k * job.mulsPerK;
        row.opsPerSec = null;
      } else if (job.kind === 'bucket') {
        for (const tid of job.sampleTids) {
          if (!limbsEqual(res.samples[tid], jacLimbs(bucketRef(tid, res.k, job.K)))) fails.push(`tid${tid}`);
        }
        mulsPerThread = res.k * job.mulsPerK;
      } else if (job.kind === 'june') {
        for (const tid of job.sampleTids) {
          if (!limbsEqual(res.samples[tid], jacLimbs(juneFoldRef(tid, res.k, K1W, K2W)))) fails.push(`tid${tid}`);
        }
        mulsPerThread = juneMulsForK(res.k) + 1;
      }
      row.ok = fails.length === 0;
      row.fails = fails;
      row.k = res.k;
      row.d = res.d;
      row.checksum = res.checksum;
      const muls = mulsPerThread * job.nThreads * res.d;
      row.wallMedianMs = median(res.wallMs);
      row.gmulWall = muls / (row.wallMedianMs * 1e6);
      if (res.gpuMs) {
        row.gpuMedianMs = median(res.gpuMs);
        row.gmulGpu = muls / (row.gpuMedianMs * 1e6);
      }
      const tMs = row.gpuMedianMs ?? row.wallMedianMs;
      const opsPerThread = job.type === 'chain_timed' ? res.k * job.opsPerK : res.k;
      row.mops = (opsPerThread * job.nThreads * res.d) / (tMs * 1e3);
    }
    rows.push(row);
  }
  return rows;
}
