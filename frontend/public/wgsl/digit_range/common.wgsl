// Digit-range sumcheck (akita stage 1, basis 8) shared declarations; appended after fp128.wgsl.
// Digits w in [-4,4); class idx(w) = w if w >= 0 else -w-1; range image V[idx] = idx*(idx+1) in {0,2,6,12}.
// Q(r) = r(r-2)(r-6)(r-12) = (r^2 - 2r)(r^2 - 18r + 72); every kernel accumulates the 5
// coefficients of sum_j weight(j) * Q(L_j + (R_j - L_j) X) into partials[wg*5 + c].
// weight(j) = eq[off_first + (j & (inner-1))] * eq[off_second + (j >> inner_bits)] (GruenSplitEq tables).

struct Params {
  n_units: u32,      // units (pairs; octets for round 0) in this dispatch
  inner_bits: u32,   // log2 |E_first| for this round
  off_first: u32,    // offset of E_first in eq[] (elements)
  off_second: u32,   // offset of E_second in eq[] (elements)
  ppt: u32,          // units per thread (power of two)
  bit_width: u32,    // PackedSignedDigits two's-complement width
  src_mode: u32,     // field kernel: 0 fold prev table, 1 materialize via LUT2f, 2 round 2; lut kernel: 0 LUT1, 1 LUT2f
  case_c: u32,       // 1: inner < ppt -> per-pair full weight
  r: vec4<u32>,      // field: fold challenge r_{k-1}; lut: r0
  r_aux: vec4<u32>,  // lut: r1
}

const WG: u32 = 256u;

// Reassemble 8 x 16-bit digit sums (each < 2^32) into a field element: sum_k d[k] * 2^(16k).
fn fp128_from_digits(d0: u32, d1: u32, d2: u32, d3: u32, d4: u32, d5: u32, d6: u32, d7: u32) -> vec4<u32> {
  let even = vec4<u32>(d0, d2, d4, d6);
  let lo = vec4<u32>(d1 << 16u, (d1 >> 16u) | (d3 << 16u), (d3 >> 16u) | (d5 << 16u), (d5 >> 16u) | (d7 << 16u));
  let tc = mul_wide(d7 >> 16u, FP128_C); // 2^128 * top -> top * C
  let e = fp128_canon(even);
  let l = fp128_add(fp128_canon(lo), vec4<u32>(tc.x, tc.y, 0u, 0u));
  return fp128_add(e, l);
}

// Bindings 0..3 are shared; kernels declare their own from 4 (layout 'auto' keeps only used ones).
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> digits: array<u32>;
@group(0) @binding(2) var<storage, read> eq: array<vec4<u32>>;
@group(0) @binding(3) var<storage, read_write> partials: array<vec4<u32>>;

// ---------- digit source: 8 classes (2 bits each) of octet o ----------
// PackedSignedDigits codec: digit i occupies bits [i*bw, (i+1)*bw) LSB-first, two's complement.
fn octet_classes(o: u32) -> u32 {
  let bw = params.bit_width;
  let mask = (1u << bw) - 1u;
  var out: u32 = 0u;
  for (var t: u32 = 0u; t < 8u; t++) {
    let bit = (8u * o + t) * bw;
    let word = bit >> 5u;
    let sh = bit & 31u;
    var v = digits[word] >> sh;
    if (sh + bw > 32u) { v |= digits[word + 1u] << (32u - sh); }
    let raw = v & mask;
    let neg = (raw >> (bw - 1u)) != 0u;
    out |= select(raw, mask - raw, neg) << (2u * t);
  }
  return out;
}

const RANGE_V = array<u32, 4>(0u, 2u, 6u, 12u);

// ---------- round polynomial coefficients of Q(L + D X) ----------
fn entry_coeffs(L: vec4<u32>, D: vec4<u32>) -> array<vec4<u32>, 5> {
  let twice = fp128_add(L, L);
  let four = fp128_add(twice, twice);
  let eight = fp128_add(four, four);
  let sixteen = fp128_add(eight, eight);
  let l2 = fp128_mul(L, L);
  let fq = fp128_sub(l2, twice);
  let sq = fp128_add(fp128_sub(l2, fp128_add(sixteen, twice)), fp_small(72u));
  let d2 = fp128_mul(D, D);
  let fl = fp128_mul(D, fp128_sub(twice, fp_small(2u)));
  let sl = fp128_mul(D, fp128_sub(twice, fp_small(18u)));
  var out: array<vec4<u32>, 5>;
  out[0] = fp128_mul(fq, sq);
  out[1] = fp128_add(fp128_mul(fq, sl), fp128_mul(fl, sq));
  out[2] = fp128_add(fp128_add(fp128_mul(fq, d2), fp128_mul(fl, sl)), fp128_mul(d2, sq));
  out[3] = fp128_mul(d2, fp128_add(fl, sl));
  out[4] = fp128_mul(d2, d2);
  return out;
}

// ---------- workgroup reduce of 5 coefficients -> partials[wg*5 + c] ----------
var<workgroup> red: array<vec4<u32>, 1280>; // 256 x 5

fn wg_reduce_store(acc: array<vec4<u32>, 5>, lid: u32, wg: u32) {
  for (var c: u32 = 0u; c < 5u; c++) { red[c * WG + lid] = acc[c]; }
  workgroupBarrier();
  for (var s: u32 = WG / 2u; s > 0u; s >>= 1u) {
    if (lid < s) {
      for (var c: u32 = 0u; c < 5u; c++) { red[c * WG + lid] = fp128_add(red[c * WG + lid], red[c * WG + lid + s]); }
    }
    workgroupBarrier();
  }
  if (lid < 5u) { partials[wg * 5u + lid] = red[lid * WG]; }
}

// ---------- thread -> units mapping ----------
// Threads of one workgroup cover WG*ppt consecutive units split into blocks of blk = min(inner, WG*ppt)
// units that share one E_second entry, so each thread multiplies by e_out once (case_c == 0).
struct Map { unit0: u32, stride: u32, count: u32 }
fn map_units(lid: u32, wg: u32) -> Map {
  let wg_units = WG * params.ppt;
  let inner = 1u << params.inner_bits;
  var m: Map;
  if (params.case_c == 1u) {
    m.unit0 = wg * wg_units + lid; m.stride = WG; m.count = params.ppt;
  } else {
    let blk = min(inner, wg_units);
    let nblk = wg_units / blk;
    let tpb = WG / nblk;
    let b = lid / tpb; let lane = lid % tpb;
    m.unit0 = wg * wg_units + b * blk + lane; m.stride = tpb; m.count = params.ppt;
  }
  return m;
}
fn e_in_of(unit: u32) -> vec4<u32> { return eq[params.off_first + (unit & ((1u << params.inner_bits) - 1u))]; }
fn e_out_of(unit: u32) -> vec4<u32> { return eq[params.off_second + (unit >> params.inner_bits)]; }
fn weight_of(unit: u32) -> vec4<u32> {
  let e = e_in_of(unit);
  return select(e, fp128_mul(e, e_out_of(unit)), params.case_c == 1u);
}
fn finish_thread(acc: ptr<function, array<vec4<u32>, 5>>, m: Map) {
  if (params.case_c == 0u) {
    let eo = e_out_of(m.unit0);
    for (var c: u32 = 0u; c < 5u; c++) { (*acc)[c] = fp128_mul((*acc)[c], eo); }
  }
}
fn accumulate(acc: ptr<function, array<vec4<u32>, 5>>, L: vec4<u32>, Rt: vec4<u32>, w: vec4<u32>) {
  let co = entry_coeffs(L, fp128_sub(Rt, L));
  for (var c: u32 = 0u; c < 5u; c++) { (*acc)[c] = fp128_add((*acc)[c], fp128_mul(co[c], w)); }
}
