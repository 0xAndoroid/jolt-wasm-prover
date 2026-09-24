// Stage-2 relation-range sumcheck (akita RelationRangeImageProver) shared declarations; appended after fp128.wgsl.
// Flat witness index i = lane * coeff_count + coefficient; a round binds the lowest variable, so unit j is the
// pair (2j, 2j+1) of the current (folded) tables. Per pair with L = w(2j), R = w(2j+1), dw = R - L,
// P0/P1 the relation+linear weights and dp = P1 - P0, the kernels accumulate
//   norm  q = [L(L+1), dw(2L+1), dw^2] weighted by eq(j) = E_first[j & mask] * E_second[j >> inner_bits]
//   rel   c = [L*P0, L*dp + dw*P0, dw*dp]                                         (unweighted)
// into partials[wg*6 + t]; the host multiplies the norm part by the Gruen linear factor.

struct Params {
  n_units: u32,      // pairs in this dispatch
  inner_bits: u32,   // log2 |E_first|
  off_first: u32,    // offset of E_first in aux[] (elements)
  off_second: u32,   // offset of E_second in aux[] (elements)
  ppt: u32,          // units per thread (power of two)
  bit_width: u32,    // PackedSignedDigits two's-complement width
  src_mode: u32,     // witness: 0 compact eval (round 0), 1 compact fold + materialize (round 1), 2 field fold
  _pad: u32,
  r: vec4<u32>,      // fold challenge r_{k-1} (src_mode 1, 2; dense P fold)
  live_len: u32,     // source witness length (digits for src_mode 0/1, W entries for 2); reads past it are 0
  src_w_off: u32,    // W offset in src[] (src_mode 2)
  dst_w_off: u32,    // W offset in dst[] (src_mode 1, 2)
  wflags: u32,       // bit0 factored weights (alpha x lane_w + segments) else dense P; bit1 write P to dst; bit2 fold dense P (else read pairs)
}

// aux header (u32 slots of aux[0..7)): [cw_bits, alpha_off, seg_off, n_segments, src_p_off, dst_p_off, pairs_off, n_pairs,
//   lane_map_off (elements of lane_w), n_sources, live_lanes, 0] then 16 source value offsets (elements of aux).
const WG: u32 = 256u;
const NT: u32 = 6u; // terms per round-pass partial

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> digits: array<u32>;
@group(0) @binding(2) var<storage, read> aux: array<vec4<u32>>;
@group(0) @binding(3) var<storage, read_write> partials: array<vec4<u32>>;

fn hdr(i: u32) -> u32 { return aux[i >> 2u][i & 3u]; }

// Signed digit i of the packed witness as a two's complement i32 (0 past live_len).
fn digit_i32(i: u32) -> i32 {
  if (i >= params.live_len) { return 0; }
  let bw = params.bit_width;
  let bit = i * bw;
  let word = bit >> 5u;
  let sh = bit & 31u;
  var v = digits[word] >> sh;
  if (sh + bw > 32u) { v |= digits[word + 1u] << (32u - sh); }
  let raw = v & ((1u << bw) - 1u);
  return select(i32(raw), i32(raw) - (1 << bw), (raw >> (bw - 1u)) != 0u);
}

fn fp_signed(k: i32) -> vec4<u32> {
  return select(fp_small(u32(k)), fp128_sub(ZERO4, fp_small(u32(-k))), k < 0);
}

// a + r (b - a)
fn fold(a: vec4<u32>, b: vec4<u32>) -> vec4<u32> { return fp128_add(a, fp128_mul(params.r, fp128_sub(b, a))); }
// w0 + r (w1 - w0) for small signed digits
fn fold_digits(w0: i32, w1: i32) -> vec4<u32> {
  return fp128_add(fp_signed(w0), fp128_mul_signed(params.r, u32(w1 - w0)));
}

// ---------- thread -> units mapping (blocks of units sharing one E_second entry) ----------
// The host clamps ppt <= |E_first|, so a thread's ppt units always sit in one block.
struct Map { unit0: u32, stride: u32, count: u32 }
fn map_units(lid: u32, wg: u32) -> Map {
  let wg_units = WG * params.ppt;
  let blk = min(1u << params.inner_bits, wg_units);
  let nblk = wg_units / blk;
  let tpb = WG / nblk;
  let b = lid / tpb; let lane = lid % tpb;
  var m: Map;
  m.unit0 = wg * wg_units + b * blk + lane; m.stride = tpb; m.count = params.ppt;
  return m;
}
fn e_in_of(unit: u32) -> vec4<u32> { return aux[params.off_first + (unit & ((1u << params.inner_bits) - 1u))]; }
fn e_out_of(unit: u32) -> vec4<u32> { return aux[params.off_second + (unit >> params.inner_bits)]; }

// ---------- workgroup reduce of NT terms -> partials[wg*NT + t] ----------
var<workgroup> red: array<vec4<u32>, 1536>; // 256 x 6

fn wg_reduce_store(acc: array<vec4<u32>, 6>, lid: u32, wg: u32) { wg_reduce_store_at(acc, lid, wg, 0u); }
fn wg_reduce_store_at(acc: array<vec4<u32>, 6>, lid: u32, wg: u32, off: u32) {
  for (var t: u32 = 0u; t < NT; t++) { red[t * WG + lid] = acc[t]; }
  workgroupBarrier();
  for (var s: u32 = WG / 2u; s > 0u; s >>= 1u) {
    if (lid < s) {
      for (var t: u32 = 0u; t < NT; t++) { red[t * WG + lid] = fp128_add(red[t * WG + lid], red[t * WG + lid + s]); }
    }
    workgroupBarrier();
  }
  if (lid < NT) { partials[off + wg * NT + lid] = red[lid * WG]; }
}
