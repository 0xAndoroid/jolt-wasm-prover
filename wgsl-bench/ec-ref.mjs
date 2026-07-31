// BigInt reference for the BN254 G1 composed-EC suite (W2a). All point
// coordinates are Montgomery-domain representatives (value·R mod p) stored as
// plain BigInts; the multiply is ctx.montMul, matching the 16-bit CIOS kernel
// exactly. Every GPU op canonicalizes to [0, p), so mirrors are plain mod-p
// arithmetic and outputs are compared limb-exact.
//
// Formula sources:
//   jacDbl  — dbl-2009-l (a=0), 2M+5S
//   jacMadd — madd-2007-bl, 7M+4S, with the guards a shipping MSM kernel
//             carries (Z1=0 -> promote, H=0&r=0 -> double, H=0 -> identity)
//   xyzzMadd/xyzzDbl — madd-2008-s / dbl-2008-s (a=0)
//   rcbDbl/rcbAdd — June's curve.wgsl (Renes-Costello-Batina complete,
//             Algorithms 9/7), mul_by_3b = ·9
//
// Mul-equivalent convention (matches W1d's Gmul/s): 1 per fe_mont_mul call,
// squares count as muls; adds/subs/·9/derives/loads are free.

import { P, makeCtx, deriveValue, modInv } from './ref.mjs';

export const ctx = makeCtx(16, 16);

export const mm = (a, b) => ctx.montMul(a, b);
export const fadd = (a, b) => (a + b) % P;
export const fsub = (a, b) => (a - b + P) % P;
const f2 = (a) => (2n * a) % P;

export const enc = (x) => (x * ctx.R) % P;
export const dec = (x) => mm(x, 1n);
export const ONE_M = enc(1n);

function powmod(b, e, m) {
  let r = 1n;
  b %= m;
  while (e > 0n) {
    if (e & 1n) r = (r * b) % m;
    b = (b * b) % m;
    e >>= 1n;
  }
  return r;
}

// Primitive cube root of unity mod p; any root works, kernel and mirror share
// this exact value (as a Montgomery-form literal).
export const BETA = (() => {
  for (let g = 2n; ; g++) {
    const b = powmod(g, (P - 1n) / 3n, P);
    if (b !== 1n) return b;
  }
})();
export const BETA_M = enc(BETA);

export const dv = (seed) => deriveValue(ctx, seed);

// ---------------------------------------------------------------------------
// Point ops (Montgomery-domain coordinates)
// ---------------------------------------------------------------------------

export const jacIdentity = () => ({ x: ONE_M, y: ONE_M, z: 0n });

// dbl-2009-l: A=X², B=Y², C=B², D=2((X+B)²−A−C), E=3A, F=E²,
// X3=F−2D, Y3=E(D−X3)−8C, Z3=2YZ
export function jacDbl(p) {
  const A = mm(p.x, p.x);
  const B = mm(p.y, p.y);
  const C = mm(B, B);
  const xb = fadd(p.x, B);
  const D = f2(fsub(fsub(mm(xb, xb), A), C));
  const E = fadd(f2(A), A);
  const F = mm(E, E);
  const X3 = fsub(F, f2(D));
  const Y3 = fsub(mm(E, fsub(D, X3)), f2(f2(f2(C))));
  const Z3 = f2(mm(p.y, p.z));
  return { x: X3, y: Y3, z: Z3 };
}

// madd-2007-bl with shipping-kernel guards.
export function jacMadd(p, qx, qy) {
  if (p.z === 0n) return { x: qx, y: qy, z: ONE_M };
  const z1z1 = mm(p.z, p.z);
  const u2 = mm(qx, z1z1);
  const s2 = mm(mm(qy, p.z), z1z1);
  const h = fsub(u2, p.x);
  const rr = fsub(s2, p.y);
  if (h === 0n) {
    if (rr === 0n) return jacDbl({ x: qx, y: qy, z: ONE_M });
    return jacIdentity();
  }
  const hh = mm(h, h);
  const i = f2(f2(hh));
  const j = mm(h, i);
  const r = f2(rr);
  const v = mm(p.x, i);
  const X3 = fsub(fsub(mm(r, r), j), f2(v));
  const Y3 = fsub(mm(r, fsub(v, X3)), f2(mm(p.y, j)));
  const zh = fadd(p.z, h);
  const Z3 = fsub(fsub(mm(zh, zh), z1z1), hh);
  return { x: X3, y: Y3, z: Z3 };
}

export const xyzzIdentity = () => ({ x: ONE_M, y: ONE_M, zz: 0n, zzz: 0n });

// dbl-2008-s (a=0): U=2Y, V=U², W=UV, S=XV, M=3X², X3=M²−2S,
// Y3=M(S−X3)−WY, ZZ3=V·ZZ, ZZZ3=W·ZZZ
export function xyzzDbl(p) {
  const u = f2(p.y);
  const v = mm(u, u);
  const w = mm(u, v);
  const s = mm(p.x, v);
  const x2 = mm(p.x, p.x);
  const m = fadd(f2(x2), x2);
  const X3 = fsub(mm(m, m), f2(s));
  const Y3 = fsub(mm(m, fsub(s, X3)), mm(w, p.y));
  return { x: X3, y: Y3, zz: mm(v, p.zz), zzz: mm(w, p.zzz) };
}

// madd-2008-s: U2=X2·ZZ, S2=Y2·ZZZ, P=U2−X1, R=S2−Y1, PP=P², PPP=P·PP,
// Q=X1·PP, X3=R²−PPP−2Q, Y3=R(Q−X3)−Y1·PPP, ZZ3=ZZ·PP, ZZZ3=ZZZ·PPP
export function xyzzMadd(p, qx, qy) {
  if (p.zz === 0n) return { x: qx, y: qy, zz: ONE_M, zzz: ONE_M };
  const u2 = mm(qx, p.zz);
  const s2 = mm(qy, p.zzz);
  const pp0 = fsub(u2, p.x);
  const r = fsub(s2, p.y);
  if (pp0 === 0n) {
    if (r === 0n) return xyzzDbl({ x: qx, y: qy, zz: ONE_M, zzz: ONE_M });
    return xyzzIdentity();
  }
  const pp = mm(pp0, pp0);
  const ppp = mm(pp0, pp);
  const q = mm(p.x, pp);
  const X3 = fsub(fsub(mm(r, r), ppp), f2(q));
  const Y3 = fsub(mm(r, fsub(q, X3)), mm(p.y, ppp));
  return { x: X3, y: Y3, zz: mm(p.zz, pp), zzz: mm(p.zzz, ppp) };
}

// June's RCB complete formulas, transcribed op-for-op from curve.wgsl.
const mul9 = (a) => (9n * a) % P;

export const projIdentity = () => ({ x: 0n, y: ONE_M, z: 0n });

export function rcbDbl(p) {
  let t0 = mm(p.y, p.y);
  let z3 = fadd(t0, t0);
  z3 = fadd(z3, z3);
  z3 = fadd(z3, z3);
  let t1 = mm(p.y, p.z);
  let t2 = mm(p.z, p.z);
  t2 = mul9(t2);
  let x3 = mm(t2, z3);
  let y3 = fadd(t0, t2);
  z3 = mm(t1, z3);
  t1 = fadd(t2, t2);
  t2 = fadd(t1, t2);
  t0 = fsub(t0, t2);
  y3 = mm(t0, y3);
  y3 = fadd(x3, y3);
  t1 = mm(p.x, p.y);
  x3 = mm(t0, t1);
  x3 = fadd(x3, x3);
  return { x: x3, y: y3, z: z3 };
}

export function rcbAdd(p, q) {
  let t0 = mm(p.x, q.x);
  let t1 = mm(p.y, q.y);
  let t2 = mm(p.z, q.z);
  let t3 = fadd(p.x, p.y);
  let t4 = fadd(q.x, q.y);
  t3 = mm(t3, t4);
  t4 = fadd(t0, t1);
  t3 = fsub(t3, t4);
  t4 = fadd(p.y, p.z);
  let x3 = fadd(q.y, q.z);
  t4 = mm(t4, x3);
  x3 = fadd(t1, t2);
  t4 = fsub(t4, x3);
  x3 = fadd(p.x, p.z);
  let y3 = fadd(q.x, q.z);
  x3 = mm(x3, y3);
  y3 = fadd(t0, t2);
  y3 = fsub(x3, y3);
  x3 = fadd(t0, t0);
  t0 = fadd(x3, t0);
  t2 = mul9(t2);
  let z3 = fadd(t1, t2);
  t1 = fsub(t1, t2);
  y3 = mul9(y3);
  x3 = mm(t4, y3);
  t2 = mm(t3, t1);
  x3 = fsub(t2, x3);
  y3 = mm(y3, t0);
  t1 = mm(t1, z3);
  y3 = fadd(t1, y3);
  t0 = mm(t0, t3);
  z3 = mm(z3, t4);
  z3 = fadd(z3, t0);
  return { x: x3, y: y3, z: z3 };
}

// ---------------------------------------------------------------------------
// Normalization + curve checks (standard domain), for self-tests only
// ---------------------------------------------------------------------------

export function jacToAffine(p) {
  if (p.z === 0n) return null;
  const z = dec(p.z);
  const zi = modInv(z, P);
  const zi2 = (zi * zi) % P;
  return { x: (dec(p.x) * zi2) % P, y: (dec(p.y) * zi2 % P) * zi % P };
}

export function projToAffine(p) {
  if (p.z === 0n) return null;
  const zi = modInv(dec(p.z), P);
  return { x: (dec(p.x) * zi) % P, y: (dec(p.y) * zi) % P };
}

export function xyzzToAffine(p) {
  if (p.zz === 0n) return null;
  return {
    x: (dec(p.x) * modInv(dec(p.zz), P)) % P,
    y: (dec(p.y) * modInv(dec(p.zzz), P)) % P,
  };
}

export const onCurve = (a) => a !== null && (a.y * a.y) % P === (a.x * a.x % P * a.x + 3n) % P;
const affEq = (a, b) => (a === null && b === null) || (a !== null && b !== null && a.x === b.x && a.y === b.y);

// Affine multiples of the generator G=(1,2), Montgomery-encoded: mults[k] = [k+1]G.
export function genMultiples(n) {
  const out = [{ x: enc(1n), y: enc(2n) }];
  let acc = { x: enc(1n), y: enc(2n), z: ONE_M };
  for (let k = 1; k < n; k++) {
    acc = jacMadd(acc, out[0].x, out[0].y);
    const a = jacToAffine(acc);
    if (!onCurve(a)) throw new Error(`genMultiples: [${k + 1}]G off-curve`);
    out.push({ x: enc(a.x), y: enc(a.y) });
  }
  return out;
}

// Validates the reference itself against curve facts before it is trusted as
// the KAT oracle. Throws on any failure.
export function ecSelfTest() {
  const g = { x: enc(1n), y: enc(2n) };
  const gJ = { ...g, z: ONE_M };

  // [2]G: dbl == madd doubling-guard == RCB dbl == XYZZ dbl, and on-curve.
  const d1 = jacToAffine(jacDbl(gJ));
  const d2 = jacToAffine(jacMadd(gJ, g.x, g.y));
  const d3 = projToAffine(rcbDbl({ ...g, z: ONE_M }));
  const d4 = xyzzToAffine(xyzzDbl({ ...g, zz: ONE_M, zzz: ONE_M }));
  const d5 = xyzzToAffine(xyzzMadd({ ...g, zz: ONE_M, zzz: ONE_M }, g.x, g.y));
  if (!onCurve(d1)) throw new Error('[2]G off-curve');
  for (const [i, d] of [d2, d3, d4, d5].entries()) {
    if (!affEq(d1, d)) throw new Error(`[2]G mismatch, variant ${i}`);
  }

  // Chain [k]G across all representations, on-curve each step.
  const mults = genMultiples(24);
  let j = { ...g, z: ONE_M };
  let x = { ...g, zz: ONE_M, zzz: ONE_M };
  let r = { ...g, z: ONE_M };
  for (let k = 1; k < 24; k++) {
    j = jacMadd(j, g.x, g.y);
    x = xyzzMadd(x, g.x, g.y);
    r = rcbAdd(r, { ...g, z: ONE_M });
    const want = { x: dec(mults[k].x), y: dec(mults[k].y) };
    if (!affEq(jacToAffine(j), want)) throw new Error(`jac chain diverged at ${k + 1}`);
    if (!affEq(xyzzToAffine(x), want)) throw new Error(`xyzz chain diverged at ${k + 1}`);
    if (!affEq(projToAffine(r), want)) throw new Error(`rcb chain diverged at ${k + 1}`);
  }

  // Identity paths: P + (−P) = identity; identity + Q = Q; dbl(identity) = identity.
  const negG = { x: g.x, y: fsub(0n, g.y) };
  if (jacMadd(gJ, negG.x, negG.y).z !== 0n) throw new Error('P + (−P) != identity');
  const idPlusQ = jacMadd(jacIdentity(), mults[4].x, mults[4].y);
  if (!affEq(jacToAffine(idPlusQ), { x: dec(mults[4].x), y: dec(mults[4].y) })) {
    throw new Error('identity + Q != Q');
  }
  if (jacDbl(jacIdentity()).z !== 0n) throw new Error('dbl(identity) != identity');
  if (rcbDbl(projIdentity()).z !== 0n) throw new Error('rcb dbl(identity) != identity');
  const rid = rcbAdd(projIdentity(), { ...g, z: ONE_M });
  if (!affEq(projToAffine(rid), { x: 1n, y: 2n })) throw new Error('rcb identity + G != G');

  // Random-Z Jacobian representative of G behaves identically.
  const z = enc(0x1234567n);
  const z2 = mm(z, z);
  const gz = { x: mm(g.x, z2), y: mm(mm(g.y, z2), z), z };
  if (!affEq(jacToAffine(jacMadd(gz, mults[2].x, mults[2].y)),
    jacToAffine(jacMadd(gJ, mults[2].x, mults[2].y)))) {
    throw new Error('random-Z representative diverged');
  }

  // GLV endomorphism: (β·x, y) of a curve point is on-curve.
  const phi = { x: (BETA * 1n) % P, y: 2n };
  if (!onCurve(phi)) throw new Error('beta endomorphism image off-curve');
  if (mm(g.x, BETA_M) !== enc(phi.x)) throw new Error('mont-domain beta mul mismatch');

  // Montgomery-domain sanity: enc/dec roundtrip and mul law.
  if (dec(mm(enc(123456789n), enc(987654321n))) !== (123456789n * 987654321n) % P) {
    throw new Error('montgomery mul law broken');
  }
  return true;
}

// ---------------------------------------------------------------------------
// Kernel mirrors: seed maps must match ec-kernels.mjs exactly
// ---------------------------------------------------------------------------

const PT_SEED = 0x80000000;
export const ptSeed = (i, c) => (PT_SEED + 2 * i + c) >>> 0;

export function deriveJac(tid) {
  const s = (8 * tid) >>> 0;
  return { x: dv(s), y: dv((s + 1) >>> 0), z: dv((s + 2) >>> 0) };
}

// ec_dbl chain: acc = derived pseudo-Jacobian, `iters` doublings.
export function dblChainRef(tid, iters) {
  let a = deriveJac(tid);
  for (let i = 0; i < iters; i++) a = jacDbl(a);
  return a;
}

// ec_madd chain: fixed derived pseudo-affine addend, `iters` mixed adds.
export function maddChainRef(tid, iters, xyzz = false) {
  const s = (8 * tid) >>> 0;
  const qx = dv((s + 3) >>> 0);
  const qy = dv((s + 4) >>> 0);
  if (xyzz) {
    let a = { x: dv(s), y: dv((s + 1) >>> 0), zz: dv((s + 2) >>> 0), zzz: dv((s + 5) >>> 0) };
    for (let i = 0; i < iters; i++) a = xyzzMadd(a, qx, qy);
    return a;
  }
  let a = deriveJac(tid);
  for (let i = 0; i < iters; i++) a = jacMadd(a, qx, qy);
  return a;
}

// ec_binv chain: acc = mm(acc, e); t = mm(t, acc) — Montgomery-trick
// product-scan shape, 2 dependent muls/iter, 3 Fq live.
export function binvChainRef(tid, iters) {
  const s = (8 * tid) >>> 0;
  const e = dv(s);
  let acc = dv((s + 1) >>> 0);
  let t = dv((s + 2) >>> 0);
  for (let i = 0; i < iters; i++) {
    acc = mm(acc, e);
    t = mm(t, acc);
  }
  return [acc, t];
}

// ec_bucket: acc = derived Jacobian; iters mixed adds streaming this thread's
// K-point segment cyclically (point j at seeds ptSeed(j, 0/1)).
export function bucketRef(tid, iters, K) {
  let a = deriveJac(tid);
  for (let i = 0; i < iters; i++) {
    const j = tid * K + (i & (K - 1));
    a = jacMadd(a, dv(ptSeed(j, 0)), dv(ptSeed(j, 1)));
  }
  return a;
}

// ec_june_fold: acc = identity; per iter: RCB dbl, then conditional RCB full
// adds of p1 (thread's pseudo-projective point) and p2 = (β·x, y, z), bits
// cycling through the 128-bit k1/k2 patterns.
export function juneLadder(p1, iters, k1words, k2words) {
  const bit = (w, b) => (w[b >> 5] >>> (b & 31)) & 1;
  const p2 = { x: mm(p1.x, BETA_M), y: p1.y, z: p1.z };
  let acc = projIdentity();
  for (let i = 0; i < iters; i++) {
    const b = i & 127;
    acc = rcbDbl(acc);
    if (bit(k1words, b)) acc = rcbAdd(acc, p1);
    if (bit(k2words, b)) acc = rcbAdd(acc, p2);
  }
  return acc;
}

export function juneFoldRef(tid, iters, k1words, k2words) {
  // Seed map matches the june fill entry: proj point i at seed base 0xC0000000 + 3i.
  const s = (0xC0000000 + 3 * tid) >>> 0;
  const p1 = { x: dv(s), y: dv((s + 1) >>> 0), z: dv((s + 2) >>> 0) };
  return juneLadder(p1, iters, k1words, k2words);
}

// Limb encoders for KAT inputs / output comparison.
export const feLimbs = (v) => ctx.toLimbs(v);
export const jacLimbs = (p) => [...feLimbs(p.x), ...feLimbs(p.y), ...feLimbs(p.z)];
export const xyzzLimbs = (p) => [...feLimbs(p.x), ...feLimbs(p.y), ...feLimbs(p.zz), ...feLimbs(p.zzz)];
