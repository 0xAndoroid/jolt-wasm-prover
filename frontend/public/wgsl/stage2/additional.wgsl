// Sparse additional terms: unit = one pair entry [m | l0 l1 b0 b1] (5 vec4 at hdr pairs_off) with m the pair
// index in this round's tables, l the compression linear weights and b the batched negative-binary weights.
// Witness values come from the digits (src_mode 0) or the table the round pass just wrote (dst, otherwise);
// live_len is that table's length (reads past it are 0).
// Cubic c(X) = w(X) l(X) + b(X) w(X)(w(X)+1) accumulated into partials[off_first + wg*6 + 0..4]; n_units = pairs.
@group(0) @binding(5) var<storage, read> dst: array<vec4<u32>>;

fn w_at(i: u32) -> vec4<u32> {
  if (params.src_mode == 0u) { return fp_signed(digit_i32(i)); }
  if (i >= params.live_len) { return ZERO4; }
  return dst[params.dst_w_off + i];
}

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) l: vec3<u32>, @builtin(workgroup_id) w: vec3<u32>) {
  let lid = l.x; let wg = w.x;
  let pairs_off = hdr(6u);
  let n_pairs = params.n_units;
  var acc: array<vec4<u32>, 6>;
  for (var t: u32 = 0u; t < NT; t++) { acc[t] = ZERO4; }
  for (var it: u32 = 0u; it < params.ppt; it++) {
    let t = (wg * params.ppt + it) * WG + lid;
    if (t >= n_pairs) { break; }
    let base = pairs_off + 5u * t;
    let m = aux[base].x;
    let l0 = aux[base + 1u]; let l1 = aux[base + 2u];
    let b0 = aux[base + 3u]; let b1 = aux[base + 4u];
    let L = w_at(2u * m); let R = w_at(2u * m + 1u);
    let dw = fp128_sub(R, L);
    let dl = fp128_sub(l1, l0);
    let db = fp128_sub(b1, b0);
    let l1p = fp128_add(L, fp_small(1u));
    let q0 = fp128_mul(L, l1p); let q1 = fp128_mul(dw, fp128_add(L, l1p)); let q2 = fp128_mul(dw, dw);
    acc[0] = fp128_add(acc[0], fp128_add(fp128_mul(L, l0), fp128_mul(b0, q0)));
    acc[1] = fp128_add(acc[1], fp128_add(fp128_add(fp128_mul(L, dl), fp128_mul(dw, l0)), fp128_add(fp128_mul(b0, q1), fp128_mul(db, q0))));
    acc[2] = fp128_add(acc[2], fp128_add(fp128_mul(dw, dl), fp128_add(fp128_mul(b0, q2), fp128_mul(db, q1))));
    acc[3] = fp128_add(acc[3], fp128_mul(db, q2));
  }
  wg_reduce_store_at(acc, lid, wg, params.off_first);
}
