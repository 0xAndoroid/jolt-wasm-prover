// LUT kernel, 256 threads = one per 4-digit class byte t = a | b<<2 | c<<4 | d<<6.
// left = V[a] + r0 (V[b]-V[a]), right = V[c] + r0 (V[d]-V[c]).
// src_mode 0: lut_out[5t+c] = coefficients of Q(left + (right-left) X)   (round 1 LUT)
// src_mode 1: lut_out[t] = left + r1 (right - left)                       (LUT2f, value after two folds)
@group(0) @binding(4) var<storage, read_write> lut_out: array<vec4<u32>>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) l: vec3<u32>) {
  let t = l.x;
  let va = fp_small(RANGE_V[t & 3u]); let vb = fp_small(RANGE_V[(t >> 2u) & 3u]);
  let vc = fp_small(RANGE_V[(t >> 4u) & 3u]); let vd = fp_small(RANGE_V[(t >> 6u) & 3u]);
  let left = fp128_add(va, fp128_mul(params.r, fp128_sub(vb, va)));
  let right = fp128_add(vc, fp128_mul(params.r, fp128_sub(vd, vc)));
  if (params.src_mode == 0u) {
    let co = entry_coeffs(left, fp128_sub(right, left));
    for (var c: u32 = 0u; c < 5u; c++) { lut_out[5u * t + c] = co[c]; }
  } else {
    lut_out[t] = fp128_add(left, fp128_mul(params.r_aux, fp128_sub(right, left)));
  }
}
