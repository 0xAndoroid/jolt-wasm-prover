// BigInt reference for Akita fp128, p = 2^128 - 275 (pseudo-Mersenne).
// Plain modular multiplication — NOT Montgomery. Kernels canonicalize their
// final output to [0, p), so verification is exact value equality.

import { deriveLimbs } from './ref.mjs';

export const P128 = (1n << 128n) - 275n;

// Limb context: W bits per limb, L limbs, capacity 2^(W*L). Kernel inputs and
// per-mul outputs are lazy residue representatives in [0, 2^(W*L)); topMask =
// mask so derive_fe produces full-range limbs (any limb pattern is a valid
// input under the pseudo-Mersenne lazy invariant).
export function makeCtx128(W, L) {
  if (W * L < 128) throw new Error(`capacity 2^${W * L} < 2^128: cannot represent fp128`);
  const mask = (1 << W) - 1;
  const toLimbs = (v) => {
    const out = new Array(L);
    for (let j = 0; j < L; j++) out[j] = Number((v >> BigInt(W * j)) & BigInt(mask));
    return out;
  };
  const fromLimbs = (limbs) => {
    let v = 0n;
    for (let j = limbs.length - 1; j >= 0; j--) v = (v << BigInt(W)) | BigInt(limbs[j] >>> 0);
    return v;
  };
  return { W, L, mask, topMask: mask, capacity: 1n << BigInt(W * L), toLimbs, fromLimbs };
}

export function deriveValue128(ctx, seed) {
  return ctx.fromLimbs(deriveLimbs(ctx, seed));
}

// Expected chain result: x <- (x * c) mod p, `muls` times, canonical output.
// Derived inputs may exceed p (redundant reps); congruence is preserved.
export function chainRef128(ctx, tid, muls) {
  let x = deriveValue128(ctx, (2 * tid) >>> 0) % P128;
  const c = deriveValue128(ctx, (2 * tid + 1) >>> 0) % P128;
  for (let k = 0; k < muls; k++) x = (x * c) % P128;
  return x;
}

// KAT vectors: canonical boundaries plus redundant representatives up to the
// scheme's full capacity (all-limbs-max maximizes column-accumulator growth —
// the bound the lazy kernels must survive). c = all-limbs-max keeps that
// stress applied on every one of the chained muls.
export function katVectors128(ctx) {
  const maxRep = ctx.capacity - 1n;
  const vals = [
    0n,
    1n,
    P128 - 1n,
    P128 >> 1n,
    275n,
    (1n << 64n) - 1n,
    (1n << 64n) + 1n,
    0x1234567890abcdef1234567890abcdefn % P128,
    P128,
    (1n << 128n) - 1n,
    maxRep,
  ];
  const pairs = [];
  for (const v of vals) pairs.push([v, maxRep]);
  pairs.push([0n, 0n], [1n, 1n], [P128 - 1n, P128 - 1n], [275n, P128 >> 1n]);
  return pairs;
}

// Same buffer layout as the BN254 suite: thread t reads x at [t*2L, t*2L+L)
// and c at [t*2L+L, t*2L+2L).
export function katInputWords128(ctx, pairs) {
  const words = [];
  for (const [x, c] of pairs) words.push(...ctx.toLimbs(x), ...ctx.toLimbs(c));
  return words;
}

export function katExpected128(ctx, pairs, muls) {
  return pairs.map(([x0, c0]) => {
    let x = x0 % P128;
    const c = c0 % P128;
    for (let k = 0; k < muls; k++) x = (x * c) % P128;
    return x;
  });
}

// Kernels canonicalize at chain end: require exact limbs of the canonical
// value (also guards limb-range: any bit above W would break fromLimbs round-trip).
export function checkOutput128(ctx, limbs, expected) {
  for (let j = 0; j < ctx.L; j++) {
    if ((limbs[j] & ~ctx.mask) !== 0) {
      return { ok: false, why: `limb${j} out of range: ${limbs[j].toString(16)}` };
    }
  }
  const v = ctx.fromLimbs(limbs);
  if (v !== expected) {
    return { ok: false, why: `mismatch: got ${v.toString(16)}, want ${expected.toString(16)}` };
  }
  return { ok: true };
}
