// Fused fold + evaluate for one stage-2 round. Unit = one pair of the tables after this round's fold.
//  src_mode 0: L, R = digits 2j, 2j+1 (round 0, nothing written)
//  src_mode 1: L = fold(digits 4j, 4j+1), R = fold(4j+2, 4j+3) -> dst[dst_w_off + 2j, +1]
//  src_mode 2: same from src[src_w_off + 4j..4j+3] -> dst
// Weights: dense (wflags bit0 = 0): P0, P1 = src[src_p_off + 2j, +1], or with bit2 the fold of src[src_p_off + 4j..4j+3]
//          written to dst[dst_p_off + 2j, +1] when bit1; factored (bit0): P(i) = alpha[i & cmask] * lane_w[i >> cw]
//          + sum over the lane's segments (lane_map word start << 8 | count, stored after the lane weights) of
//          factor * S[source_off[source_index] + source_lane * cc + (i & cmask)], written to dst when bit1 (the
//          lane-round handoff). Segment = 2 vec4 in aux at seg_off: [source_lane, source_index, 0, 0], factor.
@group(0) @binding(4) var<storage, read> src: array<vec4<u32>>;
@group(0) @binding(5) var<storage, read_write> dst: array<vec4<u32>>;
@group(0) @binding(6) var<storage, read> lane_w: array<vec4<u32>>;

fn w_src(i: u32) -> vec4<u32> {
  if (i >= params.live_len) { return ZERO4; }
  return src[params.src_w_off + i];
}

fn factored_weight(i: u32) -> vec4<u32> {
  let cw = hdr(0u);
  let alpha_off = hdr(1u);
  let seg_off = hdr(2u);
  let nseg = hdr(3u);
  let cc = 1u << cw;
  let c = i & (cc - 1u);
  let lane = i >> cw;
  var p = fp128_mul(aux[alpha_off + c], lane_w[lane]);
  // Lanes past the live count carry no linear term (and no lane-map word).
  if (lane >= hdr(10u)) { return p; }
  let lane_map_off = hdr(8u);
  let m = lane_w[lane_map_off + (lane >> 2u)][lane & 3u];
  let start = m >> 8u;
  let count = m & 0xFFu;
  for (var t: u32 = 0u; t < count; t++) {
    let s = start + t;
    if (s >= nseg) { break; }
    let seg = aux[seg_off + 2u * s];
    let value_off = hdr(12u + seg.y);
    p = fp128_add(p, fp128_mul(aux[seg_off + 2u * s + 1u], aux[value_off + seg.x * cc + c]));
  }
  return p;
}

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) l: vec3<u32>, @builtin(workgroup_id) w: vec3<u32>) {
  let lid = l.x; let wg = w.x;
  let m = map_units(lid, wg);
  let factored = (params.wflags & 1u) != 0u;
  let write_p = (params.wflags & 2u) != 0u;
  let fold_p = (params.wflags & 4u) != 0u;
  let src_p_off = hdr(4u);
  let dst_p_off = hdr(5u);
  var acc: array<vec4<u32>, 6>;
  for (var t: u32 = 0u; t < NT; t++) { acc[t] = ZERO4; }
  for (var it: u32 = 0u; it < m.count; it++) {
    let j = m.unit0 + m.stride * it;
    if (j >= params.n_units) { break; }
    var L: vec4<u32>; var R: vec4<u32>;
    var q0: vec4<u32>; var q1: vec4<u32>; var q2: vec4<u32>;
    if (params.src_mode == 0u) {
      let w0 = digit_i32(2u * j); let w1 = digit_i32(2u * j + 1u);
      let dw = w1 - w0;
      L = fp_signed(w0); R = fp_signed(w1);
      q0 = fp_signed(w0 * (w0 + 1)); q1 = fp_signed(dw * (2 * w0 + 1)); q2 = fp_signed(dw * dw);
    } else {
      if (params.src_mode == 1u) {
        L = fold_digits(digit_i32(4u * j), digit_i32(4u * j + 1u));
        R = fold_digits(digit_i32(4u * j + 2u), digit_i32(4u * j + 3u));
      } else {
        L = fold(w_src(4u * j), w_src(4u * j + 1u));
        R = fold(w_src(4u * j + 2u), w_src(4u * j + 3u));
      }
      // The folded witness has ceil(live_len / 2) entries; nothing is written past it.
      let dst_len = (params.live_len + 1u) / 2u;
      if (2u * j < dst_len) { dst[params.dst_w_off + 2u * j] = L; }
      if (2u * j + 1u < dst_len) { dst[params.dst_w_off + 2u * j + 1u] = R; }
      let dw = fp128_sub(R, L);
      let l1 = fp128_add(L, fp_small(1u));
      q0 = fp128_mul(L, l1); q1 = fp128_mul(dw, fp128_add(L, l1)); q2 = fp128_mul(dw, dw);
    }
    var P0: vec4<u32>; var P1: vec4<u32>;
    if (factored) {
      P0 = factored_weight(2u * j); P1 = factored_weight(2u * j + 1u);
    } else if (fold_p) {
      P0 = fold(src[src_p_off + 4u * j], src[src_p_off + 4u * j + 1u]);
      P1 = fold(src[src_p_off + 4u * j + 2u], src[src_p_off + 4u * j + 3u]);
    } else {
      P0 = src[src_p_off + 2u * j]; P1 = src[src_p_off + 2u * j + 1u];
    }
    if (write_p) { dst[dst_p_off + 2u * j] = P0; dst[dst_p_off + 2u * j + 1u] = P1; }
    let dw = fp128_sub(R, L);
    let dp = fp128_sub(P1, P0);
    let e = weight_of(j);
    acc[0] = fp128_add(acc[0], fp128_mul(q0, e));
    acc[1] = fp128_add(acc[1], fp128_mul(q1, e));
    acc[2] = fp128_add(acc[2], fp128_mul(q2, e));
    acc[3] = fp128_add(acc[3], fp128_mul(L, P0));
    acc[4] = fp128_add(acc[4], fp128_add(fp128_mul(L, dp), fp128_mul(dw, P0)));
    acc[5] = fp128_add(acc[5], fp128_mul(dw, dp));
  }
  if (params.case_c == 0u && m.unit0 < params.n_units) {
    let eo = e_out_of(m.unit0);
    for (var t: u32 = 0u; t < 3u; t++) { acc[t] = fp128_mul(acc[t], eo); }
  }
  wg_reduce_store(acc, lid, wg);
}
