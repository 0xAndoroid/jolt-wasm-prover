// WGSL codegen for the composed-EC suite (W2a). Field core = W1d's 16-bit
// CIOS with literal modulus (cios16_lit — measured within 1% of June's
// var<private> variant, both at the mul32 roof). Everything here composes
// that primitive into point operations; the question under test is whether
// composition (register pressure / occupancy) erodes the 0.93 Gmul/s roof.
//
// Kernels:
//   ec_dbl       — per-thread dependent chain of Jacobian doublings
//                  (dbl-2009-l, 7 mul-eq), 3 Fq persistent state.
//   ec_madd      — chain of Jacobian+affine mixed adds (madd-2007-bl,
//                  11 mul-eq) with shipping-kernel guards; 5 Fq persistent.
//   ec_madd_nogd — same without guards (isolates the guard tax).
//   ec_xyzz      — XYZZ mixed-add chain (madd-2008-s, 10 mul-eq), 6 Fq.
//   ec_binv      — Montgomery-trick product-scan control: 2 dependent
//                  muls/iter, 3 Fq live. Should sit at the W1d roof.
//   ec_bucket    — MSM bucket-accumulate: Jacobian accumulator, streams
//                  Fe8-packed affine points from storage (composition +
//                  memory pressure together).
//   ec_june_fold — June's glv_scale_add shape verbatim-in-spirit: RCB
//                  complete formulas, acc+p1+p2 (9 Fq) live across a rolled
//                  Shamir ladder with full 12-mul projective adds.
//
// Mul-eq accounting counts fe_mont_mul calls only (S=M); adds/subs/mul9/
// derives/loads are free — same convention as W1d, so rates are directly
// comparable to the 0.928 Gmul/s roof.

import { makeCtx, pcg, P } from './ref.mjs';
import { preamble, CHAIN_BINDINGS, KERNELS as FIELD_KERNELS } from './kernels.mjs';
import { ctx as ecCtx, ONE_M, BETA_M } from './ec-ref.mjs';

const ctx16 = makeCtx(16, 16);
const u = (x) => `${x >>> 0}u`;

// June-shape Shamir bit patterns: two fixed 128-bit pseudo-random scalars
// (~50% density, like real GLV sub-scalars). Shared with ec-jobs for exact
// mul counting and with ec-ref mirrors.
export const K1W = [pcg(0xA001), pcg(0xA002), pcg(0xA003), pcg(0xA004)];
export const K2W = [pcg(0xB001), pcg(0xB002), pcg(0xB003), pcg(0xB004)];

export const MULS = { dbl: 7, madd: 11, xyzz_madd: 10, binv_iter: 2, rcb_dbl: 8, rcb_add: 12 };

// ---------------------------------------------------------------------------
// Field layer: CIOS core from the W1d suite + add/sub/is_zero/constants
// ---------------------------------------------------------------------------

const feLitFn = (name, v) => {
  const limbs = ecCtx.toLimbs(v);
  let s = `fn ${name}() -> Fe {\n    var r: Fe;\n`;
  for (let i = 0; i < 16; i++) if (limbs[i]) s += `    r[${i}] = ${u(limbs[i])};\n`;
  return s + `    return r;\n}\n\n`;
};

const P_LIMBS = (() => {
  const out = [];
  for (let j = 0; j < 16; j++) out.push(Number((P >> BigInt(16 * j)) & 0xffffn));
  return out;
})();

function feAddSub() {
  let s = `fn fe_add(a: Fe, b: Fe) -> Fe {\n    var t: Fe;\n    var c = 0u;\n    var x = 0u;\n`;
  for (let i = 0; i < 16; i++) {
    s += `    x = a[${i}] + b[${i}] + c;\n    t[${i}] = x & 0xffffu;\n    c = x >> 16u;\n`;
  }
  s += `    return fe_reduce_once(t, 0u);\n}\n\n`;

  s += `fn fe_sub(a: Fe, b: Fe) -> Fe {\n    var d: Fe;\n    var s2: Fe;\n    var x = 0u;\n    var brw = 0u;\n    var c = 0u;\n`;
  for (let i = 0; i < 16; i++) {
    s += `    x = a[${i}] - b[${i}] - brw;\n    d[${i}] = x & 0xffffu;\n    brw = (x >> 16u) & 1u;\n`;
  }
  for (let i = 0; i < 16; i++) {
    s += `    x = d[${i}] + ${u(P_LIMBS[i])} + c;\n    s2[${i}] = x & 0xffffu;\n    c = x >> 16u;\n`;
  }
  s += `    return fe_select(brw == 1u, s2, d);\n}\n\n`;

  s += `fn fe_is_zero(a: Fe) -> bool {\n    var o = a[0];\n`;
  for (let i = 1; i < 16; i++) s += `    o = o | a[${i}];\n`;
  s += `    return o == 0u;\n}\n\n`;

  s += `fn fe_zero() -> Fe {\n    var r: Fe;\n    return r;\n}\n\n`;
  s += `fn fe_mul9(a: Fe) -> Fe {\n    let a2 = fe_add(a, a);\n    let a4 = fe_add(a2, a2);\n    let a8 = fe_add(a4, a4);\n    return fe_add(a8, a);\n}\n\n`;
  s += feLitFn('fe_mont_one', ONE_M);
  s += feLitFn('fe_beta', BETA_M);
  return s;
}

// 16-bit limbs <-> packed 8×u32 words (June's Fe8 storage layout).
function fe8Pack() {
  let s = `fn fe_unpack8(w: array<u32, 8>) -> Fe {\n    var r: Fe;\n`;
  for (let k = 0; k < 8; k++) {
    s += `    r[${2 * k}] = w[${k}] & 0xffffu;\n    r[${2 * k + 1}] = w[${k}] >> 16u;\n`;
  }
  s += `    return r;\n}\n\nfn fe_pack8(a: Fe) -> array<u32, 8> {\n    var r: array<u32, 8>;\n`;
  for (let k = 0; k < 8; k++) {
    s += `    r[${k}] = a[${2 * k}] | (a[${2 * k + 1}] << 16u);\n`;
  }
  s += `    return r;\n}\n\n`;
  return s;
}

const fieldBlock = () => FIELD_KERNELS.cios16_lit.body() + feAddSub();

// ---------------------------------------------------------------------------
// Point layer
// ---------------------------------------------------------------------------

const PT3 = `struct Pt3 { x: Fe, y: Fe, z: Fe }\n\n`;
const XYZZ = `struct Xyzz { x: Fe, y: Fe, zz: Fe, zzz: Fe }\n\n`;

// dbl-2009-l (a=0): 2M+5S = 7 mul-eq.
const jacDblFn = `fn jac_dbl(p: Pt3) -> Pt3 {
    let a = fe_mont_mul(p.x, p.x);
    let b = fe_mont_mul(p.y, p.y);
    let c = fe_mont_mul(b, b);
    let xb = fe_add(p.x, b);
    var d = fe_mont_mul(xb, xb);
    d = fe_sub(fe_sub(d, a), c);
    d = fe_add(d, d);
    let e = fe_add(fe_add(a, a), a);
    let f = fe_mont_mul(e, e);
    let x3 = fe_sub(f, fe_add(d, d));
    var c8 = fe_add(c, c);
    c8 = fe_add(c8, c8);
    c8 = fe_add(c8, c8);
    let y3 = fe_sub(fe_mont_mul(e, fe_sub(d, x3)), c8);
    var z3 = fe_mont_mul(p.y, p.z);
    z3 = fe_add(z3, z3);
    return Pt3(x3, y3, z3);
}

`;

// madd-2007-bl: 7M+4S = 11 mul-eq. Guards are what a shipping MSM bucket
// kernel carries; they never fire on bench data but cost registers/code.
function jacMaddFn(guarded) {
  const name = guarded ? 'jac_madd' : 'jac_madd_nogd';
  let s = `fn ${name}(p: Pt3, qx: Fe, qy: Fe) -> Pt3 {\n`;
  if (guarded) {
    s += `    if (fe_is_zero(p.z)) { return Pt3(qx, qy, fe_mont_one()); }\n`;
  }
  s += `    let z1z1 = fe_mont_mul(p.z, p.z);
    let u2 = fe_mont_mul(qx, z1z1);
    let s2 = fe_mont_mul(fe_mont_mul(qy, p.z), z1z1);
    let h = fe_sub(u2, p.x);
    let rr = fe_sub(s2, p.y);
`;
  if (guarded) {
    s += `    if (fe_is_zero(h)) {
        if (fe_is_zero(rr)) { return jac_dbl(Pt3(qx, qy, fe_mont_one())); }
        return Pt3(fe_mont_one(), fe_mont_one(), fe_zero());
    }
`;
  }
  s += `    let hh = fe_mont_mul(h, h);
    var i = fe_add(hh, hh);
    i = fe_add(i, i);
    let j = fe_mont_mul(h, i);
    let r = fe_add(rr, rr);
    let v = fe_mont_mul(p.x, i);
    let x3 = fe_sub(fe_sub(fe_mont_mul(r, r), j), fe_add(v, v));
    let yj = fe_mont_mul(p.y, j);
    let y3 = fe_sub(fe_mont_mul(r, fe_sub(v, x3)), fe_add(yj, yj));
    let zh = fe_add(p.z, h);
    let z3 = fe_sub(fe_sub(fe_mont_mul(zh, zh), z1z1), hh);
    return Pt3(x3, y3, z3);
}

`;
  return s;
}

// dbl-2008-s + madd-2008-s (XYZZ, a=0): 6M+3S and 8M+2S.
const xyzzFns = `fn xyzz_dbl(p: Xyzz) -> Xyzz {
    let uu = fe_add(p.y, p.y);
    let v = fe_mont_mul(uu, uu);
    let w = fe_mont_mul(uu, v);
    let s = fe_mont_mul(p.x, v);
    let x2 = fe_mont_mul(p.x, p.x);
    let m = fe_add(fe_add(x2, x2), x2);
    let x3 = fe_sub(fe_mont_mul(m, m), fe_add(s, s));
    let y3 = fe_sub(fe_mont_mul(m, fe_sub(s, x3)), fe_mont_mul(w, p.y));
    return Xyzz(x3, y3, fe_mont_mul(v, p.zz), fe_mont_mul(w, p.zzz));
}

fn xyzz_madd(p: Xyzz, qx: Fe, qy: Fe) -> Xyzz {
    if (fe_is_zero(p.zz)) { return Xyzz(qx, qy, fe_mont_one(), fe_mont_one()); }
    let u2 = fe_mont_mul(qx, p.zz);
    let s2 = fe_mont_mul(qy, p.zzz);
    let pp0 = fe_sub(u2, p.x);
    let r = fe_sub(s2, p.y);
    if (fe_is_zero(pp0)) {
        if (fe_is_zero(r)) { return xyzz_dbl(Xyzz(qx, qy, fe_mont_one(), fe_mont_one())); }
        return Xyzz(fe_mont_one(), fe_mont_one(), fe_zero(), fe_zero());
    }
    let pp = fe_mont_mul(pp0, pp0);
    let ppp = fe_mont_mul(pp0, pp);
    let q = fe_mont_mul(p.x, pp);
    let x3 = fe_sub(fe_sub(fe_mont_mul(r, r), ppp), fe_add(q, q));
    let y3 = fe_sub(fe_mont_mul(r, fe_sub(q, x3)), fe_mont_mul(p.y, ppp));
    return Xyzz(x3, y3, fe_mont_mul(p.zz, pp), fe_mont_mul(p.zzz, ppp));
}

`;

// June's RCB complete formulas (curve.wgsl Algorithms 9/7), op-for-op.
const rcbFns = `fn rcb_dbl(p: Pt3) -> Pt3 {
    var t0 = fe_mont_mul(p.y, p.y);
    var z3 = fe_add(t0, t0);
    z3 = fe_add(z3, z3);
    z3 = fe_add(z3, z3);
    var t1 = fe_mont_mul(p.y, p.z);
    var t2 = fe_mont_mul(p.z, p.z);
    t2 = fe_mul9(t2);
    var x3 = fe_mont_mul(t2, z3);
    var y3 = fe_add(t0, t2);
    z3 = fe_mont_mul(t1, z3);
    t1 = fe_add(t2, t2);
    t2 = fe_add(t1, t2);
    t0 = fe_sub(t0, t2);
    y3 = fe_mont_mul(t0, y3);
    y3 = fe_add(x3, y3);
    t1 = fe_mont_mul(p.x, p.y);
    x3 = fe_mont_mul(t0, t1);
    x3 = fe_add(x3, x3);
    return Pt3(x3, y3, z3);
}

fn rcb_add(p: Pt3, q: Pt3) -> Pt3 {
    var t0 = fe_mont_mul(p.x, q.x);
    var t1 = fe_mont_mul(p.y, q.y);
    var t2 = fe_mont_mul(p.z, q.z);
    var t3 = fe_add(p.x, p.y);
    var t4 = fe_add(q.x, q.y);
    t3 = fe_mont_mul(t3, t4);
    t4 = fe_add(t0, t1);
    t3 = fe_sub(t3, t4);
    t4 = fe_add(p.y, p.z);
    var x3 = fe_add(q.y, q.z);
    t4 = fe_mont_mul(t4, x3);
    x3 = fe_add(t1, t2);
    t4 = fe_sub(t4, x3);
    x3 = fe_add(p.x, p.z);
    var y3 = fe_add(q.x, q.z);
    x3 = fe_mont_mul(x3, y3);
    y3 = fe_add(t0, t2);
    y3 = fe_sub(x3, y3);
    x3 = fe_add(t0, t0);
    t0 = fe_add(x3, t0);
    t2 = fe_mul9(t2);
    var z3 = fe_add(t1, t2);
    t1 = fe_sub(t1, t2);
    y3 = fe_mul9(y3);
    x3 = fe_mont_mul(t4, y3);
    t2 = fe_mont_mul(t3, t1);
    x3 = fe_sub(t2, x3);
    y3 = fe_mont_mul(y3, t0);
    t1 = fe_mont_mul(t1, z3);
    y3 = fe_add(t1, y3);
    t0 = fe_mont_mul(t0, t3);
    z3 = fe_mont_mul(z3, t4);
    z3 = fe_add(z3, t0);
    return Pt3(x3, y3, z3);
}

`;

// ---------------------------------------------------------------------------
// Entry-point helpers
// ---------------------------------------------------------------------------

const TID_GUARD = `    let tid = gid.x;\n    if (tid >= params.n) { return; }\n`;
const ENTRY = (name, wg) => `@compute @workgroup_size(${wg})\nfn ${name}(@builtin(global_invocation_id) gid: vec3<u32>) {\n`;

const loadFeIn = `fn load_fe_in(off: u32) -> Fe {
    var r: Fe;
    for (var j = 0u; j < 16u; j++) { r[j] = inbuf[off + j]; }
    return r;
}

`;

function storePt3(accExpr) {
  return `    let ob = tid * 48u;
    for (var j = 0u; j < 16u; j++) {
        outbuf[ob + j] = ${accExpr}.x[j];
        outbuf[ob + 16u + j] = ${accExpr}.y[j];
        outbuf[ob + 32u + j] = ${accExpr}.z[j];
    }
`;
}

const BUCKET_BINDINGS = CHAIN_BINDINGS +
  `@group(0) @binding(3) var<storage, read_write> points: array<u32>;\n`;

// ---------------------------------------------------------------------------
// Kernel modules
// ---------------------------------------------------------------------------

export function ecDblModule(wg, unroll) {
  let src = `// ec_dbl: Jacobian doubling chain, wg=${wg}, unroll=${unroll}\n`;
  src += preamble(ctx16, { bindings: CHAIN_BINDINGS });
  src += fieldBlock() + PT3 + jacDblFn + loadFeIn;

  src += ENTRY('main_bench', wg) + TID_GUARD;
  src += `    let s = tid * 8u;\n    var acc = Pt3(derive_fe(s), derive_fe(s + 1u), derive_fe(s + 2u));\n`;
  src += `    for (var k = 0u; k < params.k; k++) {\n`;
  for (let r = 0; r < unroll; r++) src += `        acc = jac_dbl(acc);\n`;
  src += `    }\n` + storePt3('acc') + `}\n\n`;

  src += ENTRY('main_kat', wg) + TID_GUARD;
  src += `    var acc = Pt3(load_fe_in(tid * 48u), load_fe_in(tid * 48u + 16u), load_fe_in(tid * 48u + 32u));\n`;
  src += `    for (var k = 0u; k < params.k; k++) {\n`;
  for (let r = 0; r < unroll; r++) src += `        acc = jac_dbl(acc);\n`;
  src += `    }\n` + storePt3('acc') + `}\n`;

  return { src, wordsPerElem: 48, mulsPerK: MULS.dbl * unroll, opsPerK: unroll, katInWords: 48 };
}

export function ecMaddModule(wg, unroll, { guarded = true } = {}) {
  const fnName = guarded ? 'jac_madd' : 'jac_madd_nogd';
  let src = `// ec_madd${guarded ? '' : '_nogd'}: Jacobian+affine mixed-add chain, wg=${wg}, unroll=${unroll}\n`;
  src += preamble(ctx16, { bindings: CHAIN_BINDINGS });
  src += fieldBlock() + PT3 + jacDblFn + jacMaddFn(guarded) + loadFeIn;

  src += ENTRY('main_bench', wg) + TID_GUARD;
  src += `    let s = tid * 8u;\n    var acc = Pt3(derive_fe(s), derive_fe(s + 1u), derive_fe(s + 2u));\n`;
  src += `    let qx = derive_fe(s + 3u);\n    let qy = derive_fe(s + 4u);\n`;
  src += `    for (var k = 0u; k < params.k; k++) {\n`;
  for (let r = 0; r < unroll; r++) src += `        acc = ${fnName}(acc, qx, qy);\n`;
  src += `    }\n` + storePt3('acc') + `}\n\n`;

  src += ENTRY('main_kat', wg) + TID_GUARD;
  src += `    let base = tid * 80u;\n`;
  src += `    var acc = Pt3(load_fe_in(base), load_fe_in(base + 16u), load_fe_in(base + 32u));\n`;
  src += `    let qx = load_fe_in(base + 48u);\n    let qy = load_fe_in(base + 64u);\n`;
  src += `    for (var k = 0u; k < params.k; k++) {\n`;
  for (let r = 0; r < unroll; r++) src += `        acc = ${fnName}(acc, qx, qy);\n`;
  src += `    }\n` + storePt3('acc') + `}\n`;

  return { src, wordsPerElem: 48, mulsPerK: MULS.madd * unroll, opsPerK: unroll, katInWords: 80 };
}

export function ecXyzzModule(wg, unroll) {
  let src = `// ec_xyzz: XYZZ mixed-add chain, wg=${wg}, unroll=${unroll}\n`;
  src += preamble(ctx16, { bindings: CHAIN_BINDINGS });
  src += fieldBlock() + XYZZ + xyzzFns + loadFeIn;

  const store = `    let base = tid * 64u;
    for (var j = 0u; j < 16u; j++) {
        outbuf[base + j] = acc.x[j];
        outbuf[base + 16u + j] = acc.y[j];
        outbuf[base + 32u + j] = acc.zz[j];
        outbuf[base + 48u + j] = acc.zzz[j];
    }
`;

  src += ENTRY('main_bench', wg) + TID_GUARD;
  src += `    let s = tid * 8u;\n    var acc = Xyzz(derive_fe(s), derive_fe(s + 1u), derive_fe(s + 2u), derive_fe(s + 5u));\n`;
  src += `    let qx = derive_fe(s + 3u);\n    let qy = derive_fe(s + 4u);\n`;
  src += `    for (var k = 0u; k < params.k; k++) {\n`;
  for (let r = 0; r < unroll; r++) src += `        acc = xyzz_madd(acc, qx, qy);\n`;
  src += `    }\n` + store + `}\n\n`;

  src += ENTRY('main_kat', wg) + TID_GUARD;
  src += `    let ib = tid * 96u;\n`;
  src += `    var acc = Xyzz(load_fe_in(ib), load_fe_in(ib + 16u), load_fe_in(ib + 32u), load_fe_in(ib + 48u));\n`;
  src += `    let qx = load_fe_in(ib + 64u);\n    let qy = load_fe_in(ib + 80u);\n`;
  src += `    for (var k = 0u; k < params.k; k++) {\n`;
  for (let r = 0; r < unroll; r++) src += `        acc = xyzz_madd(acc, qx, qy);\n`;
  src += `    }\n` + store + `}\n`;

  return { src, wordsPerElem: 64, mulsPerK: MULS.xyzz_madd * unroll, opsPerK: unroll, katInWords: 96 };
}

export function ecBinvModule(wg, unroll) {
  let src = `// ec_binv: Montgomery-trick product-scan control, wg=${wg}, unroll=${unroll}\n`;
  src += preamble(ctx16, { bindings: CHAIN_BINDINGS });
  src += fieldBlock() + loadFeIn;

  const store = `    let base = tid * 32u;
    for (var j = 0u; j < 16u; j++) {
        outbuf[base + j] = acc[j];
        outbuf[base + 16u + j] = t[j];
    }
`;

  src += ENTRY('main_bench', wg) + TID_GUARD;
  src += `    let s = tid * 8u;\n    let e = derive_fe(s);\n    var acc = derive_fe(s + 1u);\n    var t = derive_fe(s + 2u);\n`;
  src += `    for (var k = 0u; k < params.k; k++) {\n`;
  for (let r = 0; r < unroll; r++) src += `        acc = fe_mont_mul(acc, e);\n        t = fe_mont_mul(t, acc);\n`;
  src += `    }\n` + store + `}\n\n`;

  src += ENTRY('main_kat', wg) + TID_GUARD;
  src += `    let ib = tid * 48u;\n    let e = load_fe_in(ib);\n    var acc = load_fe_in(ib + 16u);\n    var t = load_fe_in(ib + 32u);\n`;
  src += `    for (var k = 0u; k < params.k; k++) {\n`;
  for (let r = 0; r < unroll; r++) src += `        acc = fe_mont_mul(acc, e);\n        t = fe_mont_mul(t, acc);\n`;
  src += `    }\n` + store + `}\n`;

  return { src, wordsPerElem: 32, mulsPerK: MULS.binv_iter * unroll, opsPerK: unroll, katInWords: 48 };
}

// Bucket accumulate: points buffer holds nThreads*K affine points, Fe8-packed
// (16 words each). Thread t streams its segment cyclically.
export function ecBucketModule(wg, K) {
  if ((K & (K - 1)) !== 0) throw new Error('K must be a power of two');
  let src = `// ec_bucket: MSM bucket-accumulate, wg=${wg}, K=${K}\n`;
  src += preamble(ctx16, { bindings: BUCKET_BINDINGS });
  src += fieldBlock() + fe8Pack() + PT3 + jacDblFn + jacMaddFn(true) + loadFeIn;

  src += ENTRY('main_fill', 256);
  src += `    let i = gid.x;\n    if (i >= params.n) { return; }\n`;
  src += `    let x = fe_pack8(derive_fe(${u(0x80000000)} + 2u * i));\n`;
  src += `    let y = fe_pack8(derive_fe(${u(0x80000000)} + 2u * i + 1u));\n`;
  src += `    let base = i * 16u;\n`;
  src += `    for (var j = 0u; j < 8u; j++) {\n        points[base + j] = x[j];\n        points[base + 8u + j] = y[j];\n    }\n}\n\n`;

  src += ENTRY('main_bench', wg) + TID_GUARD;
  src += `    let s = tid * 8u;\n    var acc = Pt3(derive_fe(s), derive_fe(s + 1u), derive_fe(s + 2u));\n`;
  src += `    for (var i = 0u; i < params.k; i++) {\n`;
  src += `        let pb = (tid * ${u(K)} + (i & ${u(K - 1)})) * 16u;\n`;
  src += `        var wx: array<u32, 8>;\n        var wy: array<u32, 8>;\n`;
  src += `        for (var j = 0u; j < 8u; j++) {\n            wx[j] = points[pb + j];\n            wy[j] = points[pb + 8u + j];\n        }\n`;
  src += `        acc = jac_madd(acc, fe_unpack8(wx), fe_unpack8(wy));\n`;
  src += `    }\n` + storePt3('acc') + `}\n\n`;

  src += ENTRY('main_kat', wg) + TID_GUARD;
  src += `    let ib = tid * 176u;\n`;
  src += `    var acc = Pt3(load_fe_in(ib), load_fe_in(ib + 16u), load_fe_in(ib + 32u));\n`;
  src += `    for (var i = 0u; i < params.k; i++) {\n`;
  src += `        let qb = ib + 48u + (i & 3u) * 32u;\n`;
  src += `        acc = jac_madd(acc, load_fe_in(qb), load_fe_in(qb + 16u));\n`;
  src += `    }\n` + storePt3('acc') + `}\n`;

  return { src, wordsPerElem: 48, mulsPerK: MULS.madd, opsPerK: 1, katInWords: 176, pointWords: 16 };
}

// June's composed shape: RCB complete formulas, 9 Fq persistent (acc,p1,p2),
// rolled Shamir ladder with uniform-scalar conditional full adds.
export function ecJuneFoldModule(wg) {
  let src = `// ec_june_fold: June glv_scale_add shape (RCB complete, 9 Fq live), wg=${wg}\n`;
  src += preamble(ctx16, { bindings: BUCKET_BINDINGS });
  src += fieldBlock() + fe8Pack() + PT3 + rcbFns + loadFeIn;

  src += `const K1: vec4<u32> = vec4<u32>(${K1W.map(u).join(', ')});\n`;
  src += `const K2: vec4<u32> = vec4<u32>(${K2W.map(u).join(', ')});\n\n`;
  src += `fn kbit(k: vec4<u32>, b: u32) -> bool {\n    return ((k[b >> 5u] >> (b & 31u)) & 1u) == 1u;\n}\n\n`;

  src += ENTRY('main_fill', 256);
  src += `    let i = gid.x;\n    if (i >= params.n) { return; }\n`;
  src += `    let s = ${u(0xC0000000)} + 3u * i;\n`;
  src += `    let x = fe_pack8(derive_fe(s));\n    let y = fe_pack8(derive_fe(s + 1u));\n    let z = fe_pack8(derive_fe(s + 2u));\n`;
  src += `    let base = i * 24u;\n`;
  src += `    for (var j = 0u; j < 8u; j++) {\n        points[base + j] = x[j];\n        points[base + 8u + j] = y[j];\n        points[base + 16u + j] = z[j];\n    }\n}\n\n`;

  const ladder = `    var acc = Pt3(fe_zero(), fe_mont_one(), fe_zero());
    for (var i = 0u; i < params.k; i++) {
        let b = i & 127u;
        acc = rcb_dbl(acc);
        if (kbit(K1, b)) { acc = rcb_add(acc, p1); }
        if (kbit(K2, b)) { acc = rcb_add(acc, p2); }
    }
`;

  src += ENTRY('main_bench', wg) + TID_GUARD;
  src += `    let pb = tid * 24u;\n`;
  src += `    var wx: array<u32, 8>;\n    var wy: array<u32, 8>;\n    var wz: array<u32, 8>;\n`;
  src += `    for (var j = 0u; j < 8u; j++) {\n        wx[j] = points[pb + j];\n        wy[j] = points[pb + 8u + j];\n        wz[j] = points[pb + 16u + j];\n    }\n`;
  src += `    let p1 = Pt3(fe_unpack8(wx), fe_unpack8(wy), fe_unpack8(wz));\n`;
  src += `    let p2 = Pt3(fe_mont_mul(p1.x, fe_beta()), p1.y, p1.z);\n`;
  src += ladder + storePt3('acc') + `}\n\n`;

  src += ENTRY('main_kat', wg) + TID_GUARD;
  src += `    let ib = tid * 48u;\n`;
  src += `    let p1 = Pt3(load_fe_in(ib), load_fe_in(ib + 16u), load_fe_in(ib + 32u));\n`;
  src += `    let p2 = Pt3(fe_mont_mul(p1.x, fe_beta()), p1.y, p1.z);\n`;
  src += ladder + storePt3('acc') + `}\n`;

  return { src, wordsPerElem: 48, opsPerK: 1, katInWords: 48, pointWords: 24 };
}

// Exact fe_mont_mul count for a k-iteration june ladder window (per thread,
// excluding the fixed +1 beta mul which callers add).
export function juneMulsForK(k) {
  const bit = (w, b) => (w[b >> 5] >>> (b & 31)) & 1;
  let cycleMuls = 0;
  const prefix = [0];
  for (let b = 0; b < 128; b++) {
    cycleMuls += MULS.rcb_dbl + (bit(K1W, b) + bit(K2W, b)) * MULS.rcb_add;
    prefix.push(cycleMuls);
  }
  const full = Math.floor(k / 128);
  return full * cycleMuls + prefix[k % 128];
}
