// Field rounds. Unit = one pair evaluated this round.
//  src_mode 0: fold prev table src[4j..4j+3] by r into dst[2j], dst[2j+1], evaluate pair (fused fold + eval)
//  src_mode 1: materialize from octets 2j, 2j+1 via LUT2f (value after r0, r1) folded by r2 -> dst[2j], dst[2j+1], evaluate
//  src_mode 2: round 2: pair = octet j, L = LUT2f[low byte], R = LUT2f[high byte], no table write
fn fold(a: vec4<u32>, b: vec4<u32>) -> vec4<u32> { return fp128_add(a, fp128_mul(params.r, fp128_sub(b, a))); }
fn octet_value(o: u32) -> vec4<u32> {
  let cls = octet_classes(o);
  return fold(lut[cls & 0xFFu], lut[cls >> 8u]);
}
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) l: vec3<u32>, @builtin(workgroup_id) w: vec3<u32>) {
  let lid = l.x; let wg = w.x;
  let m = map_units(lid, wg);
  var acc: array<vec4<u32>, 5>;
  for (var c: u32 = 0u; c < 5u; c++) { acc[c] = ZERO4; }
  for (var i: u32 = 0u; i < params.ppt; i++) {
    let j = m.unit0 + m.stride * i;
    if (j >= params.n_units) { break; }
    var L: vec4<u32>; var Rt: vec4<u32>;
    if (params.src_mode == 0u) {
      L = fold(src[4u * j], src[4u * j + 1u]); Rt = fold(src[4u * j + 2u], src[4u * j + 3u]);
      dst[2u * j] = L; dst[2u * j + 1u] = Rt;
    } else if (params.src_mode == 1u) {
      L = octet_value(2u * j); Rt = octet_value(2u * j + 1u);
      dst[2u * j] = L; dst[2u * j + 1u] = Rt;
    } else {
      let cls = octet_classes(j);
      L = lut[cls & 0xFFu]; Rt = lut[cls >> 8u];
    }
    accumulate(&acc, L, Rt, e_in_of(j));
  }
  if (m.unit0 < params.n_units) { finish_thread(&acc, m); }
  wg_reduce_store(acc, lid, wg);
}
