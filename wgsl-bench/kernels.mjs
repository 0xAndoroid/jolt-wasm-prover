// WGSL codegen for the mont-mul microbench kernels.
//
// Kernels (all BN254 Fq, Montgomery form; entry points main_bench / main_kat
// for chain modules, main_fill / main_stream for streaming modules):
//   cios16_june — June's production kernel verbatim: fully unrolled 16x16-bit
//                 CIOS with eager carry chains, modulus in a var<private>
//                 array (reproduces dory-gpu shader.rs field_ops_unrolled).
//   cios16_lit  — identical math, modulus limbs inlined as literals. Isolates
//                 the var<private>-array cost from the carry-chain cost.
//   lazy13u     — ZPrize'23 optimized 13-bit lazy-carry Montgomery (20 limbs,
//                 zero inner carries, one final carry pass), unrolled with
//                 scalar accumulators and literal modulus limbs.
//   lazy13r     — same algorithm in ZPrize's shipped shape: rolled loops,
//                 dynamically indexed local arrays, get_p()-style local p.
//   lazy14u / lazy15u — ZPrize "modified" variant: NSAFE-interleaved carries
//                 (W=14: n=19, NSAFE=8; W=15: n=17, NSAFE=2), unrolled.
//
// Overflow bounds (why the lazy variants are sound in u32):
//   W=13, n=20: a slot accumulates <= 2n products of (2^13-1)^2 plus shifted
//   carries: 40*(2^13-1)^2 + 20*2^19 ~= 2.69e9 < 2^32. No inner carries needed.
//   W=14/15: 2*NSAFE products + residual + injected carry < 2^32 with
//   NSAFE = 2^(32-2W)/2 (mitschabaude's bound), so carries every NSAFE steps.
//
// All kernels end with a branchless conditional subtract to [0, p) so the
// mul cores are compared under an identical reduction policy.

import { P, makeCtx } from './ref.mjs';

const u = (x) => `${x >>> 0}u`;

function pLimbs(W, L) {
  const mask = (1n << BigInt(W)) - 1n;
  const limbs = [];
  for (let j = 0; j < L; j++) limbs.push(Number((P >> BigInt(W * j)) & mask));
  return limbs;
}

function wgslArray(limbs) {
  return `array<u32, ${limbs.length}>(${limbs.map(u).join(', ')})`;
}

// ---------------------------------------------------------------------------
// Shared preamble: aliases, params, pcg, derive_fe
// ---------------------------------------------------------------------------

export function preamble(ctx, { bindings }) {
  let s = `alias Fe = array<u32, ${ctx.L}>;\n\n`;
  s += `struct Params { k: u32, n: u32, flags: u32, _pad: u32 }\n\n`;
  s += bindings;
  s += `
fn pcg(v: u32) -> u32 {
    let state = v * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

`;
  s += `fn derive_fe(seed: u32) -> Fe {\n    var r: Fe;\n`;
  for (let j = 0; j < ctx.L; j++) {
    s += `    r[${j}] = pcg(seed * 64u + ${u(j)}) & ${u(ctx.mask)};\n`;
  }
  s += `    r[${ctx.L - 1}] &= ${u(ctx.topMask)};\n    return r;\n}\n\n`;
  return s;
}

export const CHAIN_BINDINGS = `@group(0) @binding(0) var<storage, read_write> outbuf: array<u32>;
@group(0) @binding(1) var<uniform> params: Params;
@group(0) @binding(2) var<storage, read> inbuf: array<u32>;
`;

const STREAM_BINDINGS = `@group(0) @binding(0) var<storage, read_write> outbuf: array<u32>;
@group(0) @binding(1) var<uniform> params: Params;
@group(0) @binding(2) var<storage, read_write> in_a: array<u32>;
@group(0) @binding(3) var<storage, read_write> in_b: array<u32>;
`;

// ---------------------------------------------------------------------------
// 16-bit CIOS, faithful reproduction of June's field_ops_unrolled emission
// ---------------------------------------------------------------------------

function cios16Body(lit) {
  const L = 16;
  const limbs = pLimbs(16, L);
  const ctx16 = makeCtx(16, 16);
  const MOD = (i) => (lit ? u(limbs[i]) : `FE_MOD[${i}]`);
  let s = '';
  if (!lit) s += `var<private> FE_MOD: Fe = ${wgslArray(limbs)};\n`;
  s += `const FE_NP: u32 = ${u(ctx16.n0)};\n\n`;

  s += `fn fe_select(c: bool, a: Fe, b: Fe) -> Fe {\n    var r: Fe;\n`;
  for (let i = 0; i < L; i++) s += `    r[${i}] = select(b[${i}], a[${i}], c);\n`;
  s += `    return r;\n}\n\n`;

  s += `fn fe_reduce_once(t: Fe, hi: u32) -> Fe {\n    var s: Fe;\n    var d = 0u;\n    var brw = 0u;\n`;
  for (let i = 0; i < L; i++) {
    s += `    d = t[${i}] - ${MOD(i)} - brw;\n    s[${i}] = d & 0xffffu;\n    brw = (d >> 16u) & 1u;\n`;
  }
  s += `    return fe_select((hi != 0u) | (brw == 0u), s, t);\n}\n\n`;

  s += `fn fe_mont_mul(a: Fe, b: Fe) -> Fe {\n    var t: array<u32, 18>;\n    var x = 0u;\n    var c = 0u;\n    var m = 0u;\n    var bi = 0u;\n`;
  for (let i = 0; i < L; i++) {
    s += `    bi = b[${i}];\n    c = 0u;\n`;
    for (let j = 0; j < L; j++) {
      s += `    x = t[${j}] + a[${j}] * bi + c;\n    t[${j}] = x & 0xffffu;\n    c = x >> 16u;\n`;
    }
    s += `    x = t[16] + c;\n    t[16] = x & 0xffffu;\n    t[17] = x >> 16u;\n`;
    s += `    m = (t[0] * FE_NP) & 0xffffu;\n`;
    s += `    x = t[0] + m * ${MOD(0)};\n    c = x >> 16u;\n`;
    for (let j = 1; j < L; j++) {
      s += `    x = t[${j}] + m * ${MOD(j)} + c;\n    t[${j - 1}] = x & 0xffffu;\n    c = x >> 16u;\n`;
    }
    s += `    x = t[16] + c;\n    t[15] = x & 0xffffu;\n    t[16] = t[17] + (x >> 16u);\n`;
  }
  s += `    var lo: Fe;\n`;
  for (let i = 0; i < L; i++) s += `    lo[${i}] = t[${i}];\n`;
  s += `    return fe_reduce_once(lo, t[16]);\n}\n\n`;
  return s;
}

// June's packed Fe8 storage helpers (for the streaming shape).
function fe8Helpers() {
  let s = `alias Fe8 = array<u32, 8>;\n\nfn fe_unpack(w: Fe8) -> Fe {\n    var r: Fe;\n`;
  for (let k = 0; k < 8; k++) {
    s += `    r[${2 * k}] = w[${k}] & 0xffffu;\n    r[${2 * k + 1}] = w[${k}] >> 16u;\n`;
  }
  s += `    return r;\n}\n\nfn fe_pack(a: Fe) -> Fe8 {\n    var r: Fe8;\n`;
  for (let k = 0; k < 8; k++) {
    s += `    r[${k}] = a[${2 * k}] | (a[${2 * k + 1}] << 16u);\n`;
  }
  s += `    return r;\n}\n\n`;
  return s;
}

// ---------------------------------------------------------------------------
// 13-bit lazy-carry Montgomery (ZPrize optimized), unrolled scalar form
// ---------------------------------------------------------------------------

function lazy13UnrolledBody() {
  const W = 13, L = 20;
  const ctx = makeCtx(W, L);
  const pl = pLimbs(W, L);
  const M = u(ctx.mask);
  let s = `const N0_13: u32 = ${u(ctx.n0)};\n\n`;

  s += `fn fe_mont_mul(a: Fe, b: Fe) -> Fe {\n`;
  // Scalar accumulators s0..s18; s19 stays 0 between iterations (products
  // only reach position 19 pre-shift), so it is elided until the carry pass.
  for (let j = 0; j <= L - 2; j++) s += `    var s${j} = 0u;\n`;
  s += `    var t = 0u;\n    var qi = 0u;\n    var c = 0u;\n    var ai = 0u;\n`;
  for (let i = 0; i < L; i++) {
    s += `    ai = a[${i}];\n`;
    s += `    t = s0 + ai * b[0];\n`;
    s += `    qi = ((t & ${M}) * N0_13) & ${M};\n`;
    s += `    c = (t + qi * ${u(pl[0])}) >> ${W}u;\n`;
    s += `    s0 = s1 + ai * b[1] + qi * ${u(pl[1])} + c;\n`;
    for (let j = 2; j <= L - 2; j++) {
      s += `    s${j - 1} = s${j} + ai * b[${j}] + qi * ${u(pl[j])};\n`;
    }
    s += `    s${L - 2} = ai * b[${L - 1}] + qi * ${u(pl[L - 1])};\n`;
  }
  // Final carry pass to canonical 13-bit limbs, then branchless reduce.
  s += `    var r: Fe;\n    c = 0u;\n`;
  for (let j = 0; j <= L - 2; j++) {
    s += `    t = s${j} + c;\n    r[${j}] = t & ${M};\n    c = t >> ${W}u;\n`;
  }
  s += `    r[${L - 1}] = c;\n`;
  s += reduceOnceInline(W, L, 'r');
  s += `}\n\n`;
  return s;
}

// Branchless conditional subtract emitted at the tail of a mul: computes
// r - p with a borrow chain, selects the difference when r >= p.
function reduceOnceInline(W, L, src) {
  const pl = pLimbs(W, L);
  const M = u((1 << W) - 1);
  let s = `    var red: Fe;\n    var d = 0u;\n    var brw = 0u;\n`;
  for (let i = 0; i < L; i++) {
    s += `    d = ${src}[${i}] - ${u(pl[i])} - brw;\n    red[${i}] = d & ${M};\n    brw = (d >> ${W}u) & 1u;\n`;
  }
  s += `    if (brw == 0u) { return red; }\n    return ${src};\n`;
  return s;
}

// ---------------------------------------------------------------------------
// 13-bit lazy-carry, ZPrize shipped shape: rolled loops, local arrays
// ---------------------------------------------------------------------------

function lazy13RolledBody() {
  const W = 13, L = 20;
  const ctx = makeCtx(W, L);
  const pl = pLimbs(W, L);
  let s = `const N0_13: u32 = ${u(ctx.n0)};\nconst MASK13: u32 = ${u(ctx.mask)};\n\n`;
  s += `fn get_p() -> Fe {\n    var p: Fe;\n`;
  for (let j = 0; j < L; j++) s += `    p[${j}] = ${u(pl[j])};\n`;
  s += `    return p;\n}\n\n`;
  s += `fn fe_mont_mul(a: Fe, b: Fe) -> Fe {
    var s: Fe;
    var p = get_p();
    for (var i = 0u; i < ${L}u; i++) {
        let ai = a[i];
        let t = s[0] + ai * b[0];
        let qi = ((t & MASK13) * N0_13) & MASK13;
        let c = (t + qi * p[0]) >> ${W}u;
        s[0] = s[1] + ai * b[1] + qi * p[1] + c;
        for (var j = 2u; j < ${L - 1}u; j++) {
            s[j - 1u] = s[j] + ai * b[j] + qi * p[j];
        }
        s[${L - 2}] = ai * b[${L - 1}] + qi * p[${L - 1}];
    }
    var r: Fe;
    var c = 0u;
    for (var j = 0u; j < ${L}u; j++) {
        let v = s[j] + c;
        r[j] = v & MASK13;
        c = v >> ${W}u;
    }
    var red: Fe;
    var brw = 0u;
    for (var j = 0u; j < ${L}u; j++) {
        let d = r[j] - p[j] - brw;
        red[j] = d & MASK13;
        brw = (d >> ${W}u) & 1u;
    }
    if (brw == 0u) { return red; }
    return r;
}

`;
  return s;
}

// ---------------------------------------------------------------------------
// 14/15-bit "modified" variant: NSAFE-interleaved carries, unrolled
// ---------------------------------------------------------------------------

function lazyModifiedBody(W) {
  const L = W === 14 ? 19 : 17;
  const NSAFE = W === 14 ? 8 : 2;
  const ctx = makeCtx(W, L);
  const pl = pLimbs(W, L);
  const M = u(ctx.mask);
  let s = `const N0_${W}: u32 = ${u(ctx.n0)};\n\n`;
  s += `fn fe_mont_mul(a: Fe, b: Fe) -> Fe {\n`;
  for (let j = 0; j <= L - 2; j++) s += `    var s${j} = 0u;\n`;
  s += `    var t = 0u;\n    var qi = 0u;\n    var c = 0u;\n    var ai = 0u;\n`;
  for (let i = 0; i < L; i++) {
    s += `    ai = a[${i}];\n`;
    s += `    t = s0 + ai * b[0];\n`;
    s += `    qi = ((t & ${M}) * N0_${W}) & ${M};\n`;
    s += `    c = (t + qi * ${u(pl[0])}) >> ${W}u;\n`;
    for (let j = 1; j <= L - 2; j++) {
      const terms = [`s${j}`, `ai * b[${j}]`, `qi * ${u(pl[j])}`];
      if ((j - 1) % NSAFE === 0) terms.push('c');
      s += `    t = ${terms.join(' + ')};\n`;
      if (j % NSAFE === 0) {
        s += `    c = t >> ${W}u;\n    s${j - 1} = t & ${M};\n`;
      } else {
        s += `    s${j - 1} = t;\n`;
      }
    }
    s += `    s${L - 2} = ai * b[${L - 1}] + qi * ${u(pl[L - 1])};\n`;
  }
  s += `    var r: Fe;\n    c = 0u;\n`;
  for (let j = 0; j <= L - 2; j++) {
    s += `    t = s${j} + c;\n    r[${j}] = t & ${M};\n    c = t >> ${W}u;\n`;
  }
  s += `    r[${L - 1}] = c;\n`;
  s += reduceOnceInline(W, L, 'r');
  s += `}\n\n`;
  return s;
}

// ---------------------------------------------------------------------------
// Entry points
// ---------------------------------------------------------------------------

// Chain shape: per-thread dependent chain x <- mulFn(x, c), 4 muls per
// loop iteration (params.k iterations => 4k muls). main_kat reads inputs
// from inbuf instead of deriving them. finalizeFn (optional) canonicalizes
// the chain result before writeback — used by lazy fields whose per-mul
// outputs are redundant representatives.
export function chainEntries(ctx, wgSize, { mulFn = 'fe_mont_mul', finalizeFn = null } = {}) {
  const L = ctx.L;
  let s = '';
  for (const [entry, fromBuf] of [['main_bench', false], ['main_kat', true]]) {
    s += `@compute @workgroup_size(${wgSize})\nfn ${entry}(@builtin(global_invocation_id) gid: vec3<u32>) {\n`;
    s += `    let tid = gid.x;\n    if (tid >= params.n) { return; }\n`;
    if (fromBuf) {
      s += `    var x: Fe;\n    var cc: Fe;\n`;
      s += `    for (var j = 0u; j < ${L}u; j++) {\n`;
      s += `        x[j] = inbuf[tid * ${2 * L}u + j];\n`;
      s += `        cc[j] = inbuf[tid * ${2 * L}u + ${L}u + j];\n    }\n`;
    } else {
      s += `    var x = derive_fe(tid * 2u);\n    var cc = derive_fe(tid * 2u + 1u);\n`;
    }
    s += `    for (var k = 0u; k < params.k; k++) {\n`;
    for (let r = 0; r < 4; r++) s += `        x = ${mulFn}(x, cc);\n`;
    s += `    }\n`;
    if (finalizeFn) s += `    x = ${finalizeFn}(x);\n`;
    s += `    for (var j = 0u; j < ${L}u; j++) { outbuf[tid * ${L}u + j] = x[j]; }\n`;
    s += `}\n\n`;
  }
  return s;
}

// Streaming shape: out[i] = mont_mul(in_a[i], in_b[i]), one mul per element.
// main_fill populates in_a/in_b from the same derivation the reference
// mirrors, so no host->device upload is needed.
function streamEntries(ctx, wgSize, wordsPerElem, load, store, fillStore) {
  const L = ctx.L;
  let s = `@compute @workgroup_size(${wgSize})\nfn main_fill(@builtin(global_invocation_id) gid: vec3<u32>) {\n`;
  s += `    let i = gid.x;\n    if (i >= params.n) { return; }\n`;
  s += `    var x = derive_fe(i * 2u);\n    var cc = derive_fe(i * 2u + 1u);\n`;
  s += fillStore;
  s += `}\n\n`;
  s += `@compute @workgroup_size(${wgSize})\nfn main_stream(@builtin(global_invocation_id) gid: vec3<u32>) {\n`;
  s += `    let i = gid.x;\n    if (i >= params.n) { return; }\n`;
  s += load;
  s += `    let r = fe_mont_mul(x, cc);\n`;
  s += store;
  s += `}\n\n`;
  return s;
}

// 13-bit limbs <-> 256-bit LE words (values < p < 2^254 after reduction).
function pack13Fns() {
  const W = 13, L = 20;
  let pack = `fn pack13(a: Fe) -> array<u32, 8> {\n    var w: array<u32, 8>;\n`;
  for (let j = 0; j < L; j++) {
    const pos = W * j, wd = pos >> 5, off = pos & 31;
    pack += `    w[${wd}] |= a[${j}] << ${u(off)};\n`;
    if (off > 32 - W && wd + 1 < 8) pack += `    w[${wd + 1}] |= a[${j}] >> ${u(32 - off)};\n`;
  }
  pack += `    return w;\n}\n\n`;
  let unpack = `fn unpack13(w: array<u32, 8>) -> Fe {\n    var r: Fe;\n`;
  for (let j = 0; j < L; j++) {
    const pos = W * j, wd = pos >> 5, off = pos & 31;
    let expr = `(w[${wd}] >> ${u(off)})`;
    if (off > 32 - W && wd + 1 < 8) expr = `(${expr} | (w[${wd + 1}] << ${u(32 - off)}))`;
    unpack += `    r[${j}] = ${expr} & 0x1fffu;\n`;
  }
  unpack += `    return r;\n}\n\n`;
  return pack + unpack;
}

// ---------------------------------------------------------------------------
// Public: build a module for a kernel config
// ---------------------------------------------------------------------------

export const KERNELS = {
  cios16_june: { W: 16, L: 16, body: () => cios16Body(false), desc: "June unrolled 16-bit CIOS, var<private> modulus" },
  cios16_lit: { W: 16, L: 16, body: () => cios16Body(true), desc: '16-bit CIOS, literal modulus limbs' },
  lazy13u: { W: 13, L: 20, body: lazy13UnrolledBody, desc: '13-bit lazy-carry, unrolled scalar accumulators' },
  lazy13r: { W: 13, L: 20, body: lazy13RolledBody, desc: '13-bit lazy-carry, rolled loops (ZPrize shipped shape)' },
  lazy14u: { W: 14, L: 19, body: () => lazyModifiedBody(14), desc: '14-bit NSAFE=8 interleaved carries, unrolled' },
  lazy15u: { W: 15, L: 17, body: () => lazyModifiedBody(15), desc: '15-bit NSAFE=2 interleaved carries, unrolled' },
};

export function chainModule(kernelName, wgSize) {
  const k = KERNELS[kernelName];
  const ctx = makeCtx(k.W, k.L);
  let src = `// chain module: ${kernelName} (${k.desc}), wg=${wgSize}\n`;
  src += preamble(ctx, { bindings: CHAIN_BINDINGS });
  src += k.body();
  src += chainEntries(ctx, wgSize);
  return { src, ctx, wordsPerElem: k.L };
}

// stream kind: 'june8' (Fe8-packed 16-bit), 'raw13' (20 raw u32), 'packed13'
export function streamModule(kind, wgSize) {
  if (kind === 'june8') {
    const ctx = makeCtx(16, 16);
    let src = `// stream module: cios16_june Fe8-packed, wg=${wgSize}\n`;
    src += preamble(ctx, { bindings: STREAM_BINDINGS });
    src += fe8Helpers();
    src += cios16Body(false);
    src += streamEntries(
      ctx, wgSize, 8,
      `    var wa: Fe8;\n    var wb: Fe8;\n    for (var j = 0u; j < 8u; j++) { wa[j] = in_a[i * 8u + j]; wb[j] = in_b[i * 8u + j]; }\n    var x = fe_unpack(wa);\n    var cc = fe_unpack(wb);\n`,
      `    let w = fe_pack(r);\n    for (var j = 0u; j < 8u; j++) { outbuf[i * 8u + j] = w[j]; }\n`,
      `    let wa = fe_pack(x);\n    let wb = fe_pack(cc);\n    for (var j = 0u; j < 8u; j++) { in_a[i * 8u + j] = wa[j]; in_b[i * 8u + j] = wb[j]; }\n`,
    );
    return { src, ctx, wordsPerElem: 8 };
  }
  if (kind === 'raw13') {
    const ctx = makeCtx(13, 20);
    let src = `// stream module: lazy13 raw 20-word storage, wg=${wgSize}\n`;
    src += preamble(ctx, { bindings: STREAM_BINDINGS });
    src += lazy13UnrolledBody();
    src += streamEntries(
      ctx, wgSize, 20,
      `    var x: Fe;\n    var cc: Fe;\n    for (var j = 0u; j < 20u; j++) { x[j] = in_a[i * 20u + j]; cc[j] = in_b[i * 20u + j]; }\n`,
      `    for (var j = 0u; j < 20u; j++) { outbuf[i * 20u + j] = r[j]; }\n`,
      `    for (var j = 0u; j < 20u; j++) { in_a[i * 20u + j] = x[j]; in_b[i * 20u + j] = cc[j]; }\n`,
    );
    return { src, ctx, wordsPerElem: 20 };
  }
  if (kind === 'packed13') {
    const ctx = makeCtx(13, 20);
    let src = `// stream module: lazy13 value-packed 8-word storage, wg=${wgSize}\n`;
    src += preamble(ctx, { bindings: STREAM_BINDINGS });
    src += pack13Fns();
    src += lazy13UnrolledBody();
    src += streamEntries(
      ctx, wgSize, 8,
      `    var wa: array<u32, 8>;\n    var wb: array<u32, 8>;\n    for (var j = 0u; j < 8u; j++) { wa[j] = in_a[i * 8u + j]; wb[j] = in_b[i * 8u + j]; }\n    var x = unpack13(wa);\n    var cc = unpack13(wb);\n`,
      `    let w = pack13(r);\n    for (var j = 0u; j < 8u; j++) { outbuf[i * 8u + j] = w[j]; }\n`,
      `    let wa = pack13(x);\n    let wb = pack13(cc);\n    for (var j = 0u; j < 8u; j++) { in_a[i * 8u + j] = wa[j]; in_b[i * 8u + j] = wb[j]; }\n`,
    );
    return { src, ctx, wordsPerElem: 8 };
  }
  throw new Error(`unknown stream kind ${kind}`);
}
