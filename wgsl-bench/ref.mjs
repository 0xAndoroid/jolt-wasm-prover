// BigInt reference for BN254 Fq Montgomery multiplication at arbitrary limb
// width, plus the exact JS mirror of the in-shader PCG input derivation.
// Kernel outputs are only congruence-checked (mod p) and bound-checked
// (< 2p): lazy kernels legally return redundant representatives.

export const P = 0x30644e72e131a029b85045b68181585d97816a916871ca8d3c208c16d87cfd47n;

export function egcd(a, b) {
  let [old_r, r] = [a, b];
  let [old_s, s] = [1n, 0n];
  while (r !== 0n) {
    const q = old_r / r;
    [old_r, r] = [r, old_r - q * r];
    [old_s, s] = [s, old_s - q * s];
  }
  return [old_r, old_s];
}

export function modInv(a, m) {
  const [g, x] = egcd(((a % m) + m) % m, m);
  if (g !== 1n) throw new Error('not invertible');
  return ((x % m) + m) % m;
}

// Field context for a limb scheme: W bits per limb, L limbs, R = 2^(W*L).
export function makeCtx(W, L) {
  const R = 1n << BigInt(W * L);
  // Every kernel ends with a conditional subtract to [0, p), so inputs are
  // always < p and out < p + p^2/R < 2p needs only R > p (one subtract).
  if (R <= P) throw new Error(`R=2^${W * L} <= p: Montgomery reduction unsound`);
  const Rinv = modInv(R % P, P);
  const twoW = 1n << BigInt(W);
  // n0 = -p^-1 mod 2^W
  const n0 = Number((twoW - modInv(P % twoW, twoW)) % twoW);
  const mask = (1 << W) - 1;
  // Derived inputs stay < 2^253 < p: top limb keeps 253 - W*(L-1) bits.
  const topBits = 253 - W * (L - 1);
  if (topBits < 1) throw new Error('limb scheme cannot bound derived values under p');
  const topMask = (1 << topBits) - 1;

  const toLimbs = (v) => {
    const out = new Array(L);
    for (let j = 0; j < L; j++) {
      out[j] = Number((v >> BigInt(W * j)) & BigInt(mask));
    }
    return out;
  };
  const fromLimbs = (limbs) => {
    let v = 0n;
    for (let j = limbs.length - 1; j >= 0; j--) v = (v << BigInt(W)) | BigInt(limbs[j] >>> 0);
    return v;
  };
  // mont(a, b) = a*b*R^-1 mod p — the canonical value every kernel output
  // must be congruent to.
  const montMul = (a, b) => (a * b * Rinv) % P;

  return { W, L, R, Rinv, n0, mask, topMask, toLimbs, fromLimbs, montMul };
}

// Exact mirror of the WGSL pcg() — every shift amount here is < 32 by
// construction ((state>>>28)+4 <= 19), so JS/WGSL masking rules never diverge.
export function pcg(v) {
  const state = (Math.imul(v | 0, 747796405) + 2891336453) >>> 0;
  const word = Math.imul((state >>> ((state >>> 28) + 4)) ^ state, 277803737) >>> 0;
  return ((word >>> 22) ^ word) >>> 0;
}

// Mirror of derive_fe(seed): limb j = pcg(seed*64 + j) masked; top limb
// clamped so the value is < 2^253 < p.
export function deriveLimbs(ctx, seed) {
  const limbs = new Array(ctx.L);
  for (let j = 0; j < ctx.L; j++) {
    limbs[j] = pcg(((Math.imul(seed, 64) >>> 0) + j) >>> 0) & ctx.mask;
  }
  limbs[ctx.L - 1] &= ctx.topMask;
  return limbs;
}

export function deriveValue(ctx, seed) {
  return ctx.fromLimbs(deriveLimbs(ctx, seed));
}

// Expected chain result for one thread: x <- mont(x, c), `muls` times.
export function chainRef(ctx, tid, muls) {
  let x = deriveValue(ctx, (2 * tid) >>> 0);
  const c = deriveValue(ctx, (2 * tid + 1) >>> 0);
  for (let k = 0; k < muls; k++) x = ctx.montMul(x, c);
  return x;
}

// KAT vectors: canonical field elements exercising zero, one, boundaries and
// the max-limb stress pattern that maximizes lazy-carry accumulator growth.
export function katVectors(ctx) {
  const maxStress = (() => {
    const limbs = new Array(ctx.L).fill(ctx.mask);
    const pTop = Number(P >> BigInt(ctx.W * (ctx.L - 1)));
    limbs[ctx.L - 1] = pTop - 1;
    return ctx.fromLimbs(limbs);
  })();
  const vals = [
    0n,
    1n,
    P - 1n,
    P >> 1n,
    maxStress,
    ctx.R % P,
    (ctx.R * ctx.R) % P,
    0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdefn % P,
  ];
  const pairs = [];
  for (const v of vals) pairs.push([v, maxStress]);
  pairs.push([0n, 0n], [1n, 1n], [P - 1n, P - 1n], [vals[7], vals[3]]);
  return pairs;
}

// Serializes KAT pairs into the kernel input buffer layout:
// thread t reads x at [t*2L, t*2L+L) and c at [t*2L+L, t*2L+2L).
export function katInputWords(ctx, pairs) {
  const words = [];
  for (const [x, c] of pairs) {
    words.push(...ctx.toLimbs(x), ...ctx.toLimbs(c));
  }
  return words;
}

export function katExpected(ctx, pairs, muls) {
  return pairs.map(([x0, c]) => {
    let x = x0;
    for (let k = 0; k < muls; k++) x = ctx.montMul(x, c);
    return x;
  });
}

// Streaming shape reference: element i is mont(derive(2i), derive(2i+1)).
export function streamRef(ctx, i) {
  return ctx.montMul(deriveValue(ctx, (2 * i) >>> 0), deriveValue(ctx, (2 * i + 1) >>> 0));
}

// Checks a kernel output (limb array) against an expected canonical value:
// congruent mod p and value < 2p.
export function checkOutput(ctx, limbs, expected) {
  const v = ctx.fromLimbs(limbs);
  if (v >= 2n * P) return { ok: false, why: `out of range: ${v.toString(16)} >= 2p` };
  if (v % P !== expected) {
    return { ok: false, why: `mod-p mismatch: got ${(v % P).toString(16)}, want ${expected.toString(16)}` };
  }
  return { ok: true };
}
