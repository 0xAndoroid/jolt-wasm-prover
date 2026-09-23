// Round 0: pairs of raw digits. Unit = octet (4 pairs). Private 16-class eq histogram per thread,
// then LUT0 (signed i32 q-coefficients per class pair) x histogram, then x e_out.
// inner is in PAIRS and >= 4; ppt <= inner/4 so every thread's octets share one E_second entry.
@group(0) @binding(4) var<storage, read> lut0: array<vec4<u32>>; // 20 x vec4 = 80 i32

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) l: vec3<u32>, @builtin(workgroup_id) w: vec3<u32>) {
  let lid = l.x; let wg = w.x;
  let wg_units = WG * params.ppt;
  let inner_o = 1u << (params.inner_bits - 2u);
  let blk = min(inner_o, wg_units);
  let nblk = wg_units / blk;
  let tpb = WG / nblk;
  let b = lid / tpb; let lane = lid % tpb;
  let unit0 = wg * wg_units + b * blk + lane;
  var hist: array<vec4<u32>, 16>;
  for (var k: u32 = 0u; k < 16u; k++) { hist[k] = ZERO4; }
  for (var i: u32 = 0u; i < params.ppt; i++) {
    let o = unit0 + tpb * i;
    if (o >= params.n_units) { break; }
    let cls = octet_classes(o);
    for (var q: u32 = 0u; q < 4u; q++) {
      let pair = 4u * o + q;
      let cp = (cls >> (4u * q)) & 0xFu;
      hist[cp] = fp128_add(hist[cp], eq[params.off_first + (pair & ((1u << params.inner_bits) - 1u))]);
    }
  }
  var acc: array<vec4<u32>, 5>;
  for (var c: u32 = 0u; c < 5u; c++) {
    var s = ZERO4;
    for (var cp: u32 = 0u; cp < 16u; cp++) {
      let k = lut0[(c * 16u + cp) / 4u][(c * 16u + cp) % 4u];
      s = fp128_add(s, fp128_mul_signed(hist[cp], k));
    }
    acc[c] = s;
  }
  if (unit0 < params.n_units) {
    let eo = eq[params.off_second + ((4u * unit0) >> params.inner_bits)];
    for (var c: u32 = 0u; c < 5u; c++) { acc[c] = fp128_mul(acc[c], eo); }
  } else {
    for (var c: u32 = 0u; c < 5u; c++) { acc[c] = ZERO4; }
  }
  wg_reduce_store(acc, lid, wg);
}
