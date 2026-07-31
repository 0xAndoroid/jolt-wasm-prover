// WGSL codegen for Akita fp128 (p = 2^128 - 275, pseudo-Mersenne) mul kernels.
// Plain modular multiplication, NOT Montgomery. Both kernels are lazy between
// muls (redundant representatives) and canonicalize once at chain end
// (fe_canon), so the mul cores are compared under an identical policy.
//
//   fp128_school16 — 8×16-bit limbs, unrolled comba schoolbook (64 mul32,
//                    lo/hi split into 16 u32 columns) + Solinas fold:
//                    cols 8..15 ×275 into cols 0..7 (8 mul32), carry pass,
//                    overflow ×275 (1 mul32), select-fold. 73 mul32 total.
//                    Per-mul output: 16-bit limbs, value in [0, 2^128).
//   fp128_lazy13   — 10×13-bit limbs, unrolled schoolbook with NO product
//                    splits and NO inner carries (100 mul32 into 19 u32
//                    columns) + fold: normalize cols 10..18 to 13-bit limbs,
//                    ×1100 (2^130 ≡ 4·275) into cols 0..9 (11 mul32), carry
//                    pass, overflow ×1100 (1 mul32), select-fold. 112 mul32.
//                    Per-mul output: 13-bit limbs, value in [0, 2^130).
//
// Overflow bounds (worst case = both operands all-limbs-max, covered by KATs):
//   school16: column of 16-bit halves ≤ 15·(2^16-1) < 2^20; +fold ≤ 2^20·276
//   < 2^28.2; carry ≤ 2^12.2; fold2 term ≤ 2^12.2·275 < 2^20.4. All < 2^32.
//   lazy13: column ≤ 10·(2^13-1)^2 < 2^29.33; hi-normalize carry < 2^16.4 so
//   h10 = carry>>13 ≤ 10, folded as h10·1100^2 = h10·1210000 < 2^23.6;
//   lo column + h·1100 + h10·1210000 < 2^29.4; lo-normalize carry o < 2^16.4;
//   fold2 term o·1100 < 2^26.6 (spans limbs 0..2). All < 2^32.
//
// Intermediate widths rejected by mul-count math (M4 rate = mul32-roof /
// mul-count, per the BN254 sweep): any W<16 raises L and hence L^2 product
// muls while the fold stays ~L muls — 9×15 = 81+11 = 92, 10×14 = 100+split
// fold (1100·2^12 overflows u32), 11×12 = 121+. 8×16 dominates.

import { preamble, chainEntries, CHAIN_BINDINGS } from './kernels.mjs';
import { P128, makeCtx128 } from './fp128-ref.mjs';

const u = (x) => `${x >>> 0}u`;

function p128Limbs(W, L) {
  const mask = (1n << BigInt(W)) - 1n;
  const limbs = [];
  for (let j = 0; j < L; j++) limbs.push(Number((P128 >> BigInt(W * j)) & mask));
  return limbs;
}

// Branchless conditional subtract of p in W-bit limbs (same borrow idiom as
// the BN254 reduceOnceInline; sound because per-limb |diff| < 2^(W+1)).
function condSubP(W, L, src) {
  const pl = p128Limbs(W, L);
  const M = u((1 << W) - 1);
  let s = `    var red: Fe;\n    var d = 0u;\n    var brw = 0u;\n`;
  for (let i = 0; i < L; i++) {
    s += `    d = ${src}[${i}] - ${u(pl[i])} - brw;\n    red[${i}] = d & ${M};\n    brw = (d >> ${W}u) & 1u;\n`;
  }
  s += `    if (brw == 0u) { return red; }\n    return ${src};\n`;
  return s;
}

// ---------------------------------------------------------------------------
// 8×16-bit comba schoolbook + Solinas fold
// ---------------------------------------------------------------------------

function school16Body() {
  const L = 8;
  let s = `fn fe_mul(a: Fe, b: Fe) -> Fe {\n`;
  for (let k = 0; k < 2 * L; k++) s += `    var col${k} = 0u;\n`;
  s += `    var t = 0u;\n`;
  for (let i = 0; i < L; i++) {
    for (let j = 0; j < L; j++) {
      s += `    t = a[${i}] * b[${j}];\n    col${i + j} += t & 0xffffu;\n    col${i + j + 1} += t >> 16u;\n`;
    }
  }
  for (let k = 0; k < L; k++) s += `    col${k} += col${k + L} * 275u;\n`;
  s += `    var c = col0;\n    var v0 = c & 0xffffu;\n`;
  for (let k = 1; k < L; k++) {
    s += `    c = (c >> 16u) + col${k};\n    var v${k} = c & 0xffffu;\n`;
  }
  s += `    var f = (c >> 16u) * 275u;\n`;
  s += `    c = v0 + (f & 0xffffu);\n    v0 = c & 0xffffu;\n`;
  s += `    c = v1 + (f >> 16u) + (c >> 16u);\n    v1 = c & 0xffffu;\n`;
  for (let k = 2; k < L; k++) {
    s += `    c = v${k} + (c >> 16u);\n    v${k} = c & 0xffffu;\n`;
  }
  s += `    c = v0 + select(0u, 275u, (c >> 16u) != 0u);\n    v0 = c & 0xffffu;\n`;
  for (let k = 1; k < L; k++) {
    s += `    c = v${k} + (c >> 16u);\n    v${k} = c & 0xffffu;\n`;
  }
  // Final carry provably 0: a wrapped value is < 2^21 before the select-fold.
  s += `    var r: Fe;\n`;
  for (let k = 0; k < L; k++) s += `    r[${k}] = v${k};\n`;
  s += `    return r;\n}\n\n`;

  // Value < 2^128 = p + 275: one conditional subtract canonicalizes.
  s += `fn fe_canon(x: Fe) -> Fe {\n`;
  s += condSubP(16, L, 'x');
  s += `}\n\n`;
  return s;
}

// ---------------------------------------------------------------------------
// 10×13-bit lazy-carry schoolbook + Solinas fold
// ---------------------------------------------------------------------------

function lazy13Body() {
  const L = 10;
  const M = '0x1fffu';
  let s = `fn fe_mul(a: Fe, b: Fe) -> Fe {\n`;
  for (let k = 0; k < 2 * L - 1; k++) s += `    var col${k} = 0u;\n`;
  for (let i = 0; i < L; i++) {
    for (let j = 0; j < L; j++) {
      s += `    col${i + j} += a[${i}] * b[${j}];\n`;
    }
  }
  s += `    var c = col${L};\n    var h0 = c & ${M};\n`;
  for (let k = L + 1; k < 2 * L - 1; k++) {
    s += `    c = (c >> 13u) + col${k};\n    var h${k - L} = c & ${M};\n`;
  }
  s += `    c = c >> 13u;\n`;
  for (let k = 0; k < L - 1; k++) s += `    col${k} += h${k} * 1100u;\n`;
  s += `    col${L - 1} += (c & ${M}) * 1100u;\n`;
  s += `    col0 += (c >> 13u) * 1210000u;\n`;
  s += `    c = col0;\n    var r0 = c & ${M};\n`;
  for (let k = 1; k < L; k++) {
    s += `    c = (c >> 13u) + col${k};\n    var r${k} = c & ${M};\n`;
  }
  s += `    var f = (c >> 13u) * 1100u;\n`;
  s += `    c = r0 + (f & ${M});\n    r0 = c & ${M};\n`;
  s += `    c = r1 + ((f >> 13u) & ${M}) + (c >> 13u);\n    r1 = c & ${M};\n`;
  s += `    c = r2 + (f >> 26u) + (c >> 13u);\n    r2 = c & ${M};\n`;
  for (let k = 3; k < L; k++) {
    s += `    c = r${k} + (c >> 13u);\n    r${k} = c & ${M};\n`;
  }
  s += `    c = r0 + select(0u, 1100u, (c >> 13u) != 0u);\n    r0 = c & ${M};\n`;
  for (let k = 1; k < L; k++) {
    s += `    c = r${k} + (c >> 13u);\n    r${k} = c & ${M};\n`;
  }
  // Final carry provably 0: a wrapped value is < 2^27 before the select-fold.
  s += `    var r: Fe;\n`;
  for (let k = 0; k < L; k++) s += `    r[${k}] = r${k};\n`;
  s += `    return r;\n}\n\n`;

  // Canonicalize from [0, 2^130): fold bits 128..129 (r[9] bits 11..12) via
  // 2^128 ≡ 275 twice (after the first fold value < 2^128 + 825; a carry
  // ripple can set bit 128 once more, second fold clears it for good), then
  // one conditional subtract of p.
  s += `fn fe_canon(x: Fe) -> Fe {\n    var r = x;\n    var c = 0u;\n`;
  for (let pass = 0; pass < 2; pass++) {
    s += `    c = r[0] + (r[9] >> 11u) * 275u;\n    r[9] = r[9] & 0x7ffu;\n    r[0] = c & ${M};\n`;
    for (let k = 1; k < L; k++) {
      s += `    c = r[${k}] + (c >> 13u);\n    r[${k}] = c & ${M};\n`;
    }
  }
  s += condSubP(13, L, 'r');
  s += `}\n\n`;
  return s;
}

// ---------------------------------------------------------------------------
// Public registry + module builder (chain shape only — no stream kernels;
// the BN254 sweep already established fold-style kernels are DRAM-bound)
// ---------------------------------------------------------------------------

export const FP128_KERNELS = {
  fp128_school16: { W: 16, L: 8, body: school16Body, desc: 'fp128 8x16-bit comba schoolbook + Solinas x275 fold (73 mul32)' },
  fp128_lazy13: { W: 13, L: 10, body: lazy13Body, desc: 'fp128 10x13-bit lazy-carry schoolbook + Solinas x1100 fold (112 mul32)' },
};

export function chainModule128(kernelName, wgSize) {
  const k = FP128_KERNELS[kernelName];
  const ctx = makeCtx128(k.W, k.L);
  let src = `// fp128 chain module: ${kernelName} (${k.desc}), wg=${wgSize}\n`;
  src += preamble(ctx, { bindings: CHAIN_BINDINGS });
  src += k.body();
  src += chainEntries(ctx, wgSize, { mulFn: 'fe_mul', finalizeFn: 'fe_canon' });
  return { src, ctx, wordsPerElem: k.L };
}
