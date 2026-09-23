// reduce: sum per-chunk digit partials, reassemble to 160 bits, reduce mod p, write canonical fp128.
// PART layout: index = ((chunk * colcap + c) * blocks + b) * 512 + i, 2 vec4 each:
//   vec4 #0 = digits 0,2,4,6 (low halves of limbs 0..3), vec4 #1 = digits 1,3,5,7 (high halves).
// RES layout: index = (c * blocks + b) * 512 + i, one vec4 (canonical limbs).
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> PART: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read_write> RES: array<vec4<u32>>;

fn mul32(a: u32, b: u32) -> vec2<u32> {   // (lo, hi) of a*b
  let a0 = a & 0xFFFFu; let a1 = a >> 16u;
  let b0 = b & 0xFFFFu; let b1 = b >> 16u;
  let p00 = a0 * b0; let p01 = a0 * b1; let p10 = a1 * b0; let p11 = a1 * b1;
  let mid = p01 + p10;
  let midc = u32(mid < p01);
  let lo = p00 + (mid << 16u);
  let c1 = u32(lo < p00);
  let hi = p11 + (mid >> 16u) + (midc << 16u) + c1;
  return vec2<u32>(lo, hi);
}

// v = v[0..4] (5 limbs). Fold limb 4 down: v <- v mod 2^128 + v[4] * C. Result limb 4 is the carry (0/1).
fn fold(v: ptr<function, array<u32, 5>>) {
  let m = mul32((*v)[4], C_LO);
  var s = (*v)[0] + m.x;
  var c = u32(s < m.x);
  (*v)[0] = s;
  s = (*v)[1] + m.y;
  var c2 = u32(s < m.y);
  s = s + c;
  c = c2 + u32(s < c);
  (*v)[1] = s;
  s = (*v)[2] + c;
  c = u32(s < c);
  (*v)[2] = s;
  s = (*v)[3] + c;
  c = u32(s < c);
  (*v)[3] = s;
  (*v)[4] = c;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let stride = P.colcap * P.blocks * 512u;
  var L: array<u32, 8>;
  for (var k = 0u; k < 8u; k++) { L[k] = 0u; }
  for (var ch = 0u; ch < P.num_chunks; ch++) {
    let base = (ch * stride + idx) * 2u;
    let lo = PART[base];
    let hi = PART[base + 1u];
    L[0] += lo.x; L[1] += hi.x; L[2] += lo.y; L[3] += hi.y;
    L[4] += lo.z; L[5] += hi.z; L[6] += lo.w; L[7] += hi.w;
  }
  // digits -> 160-bit value (L[k] + carry never overflows: L[k] <= 2^32 - 2^16, carry < 2^16)
  var d: array<u32, 8>;
  var t = L[0];
  for (var k = 0u; k < 8u; k++) {
    d[k] = t & 0xFFFFu;
    let carry = t >> 16u;
    if (k < 7u) { t = L[k + 1u] + carry; } else { t = carry; }
  }
  var v: array<u32, 5>;
  v[0] = d[0] | (d[1] << 16u);
  v[1] = d[2] | (d[3] << 16u);
  v[2] = d[4] | (d[5] << 16u);
  v[3] = d[6] | (d[7] << 16u);
  v[4] = t;
  fold(&v);
  fold(&v);
  // now v < 2^128 + 2^32 (limb 4 in {0,1}); one conditional subtract of p
  let ge = (v[4] != 0u) || (v[3] == FULL && v[2] == FULL && v[1] == FULL && v[0] >= P0);
  if (ge) {
    // v - p = v - 2^128 + C  (limbs 1..3 of p are all ones)
    let s0 = v[0] + C_LO;
    let c0 = u32(s0 < v[0]);
    var s1 = v[1] + c0;
    let c1 = u32(s1 < c0);
    var s2 = v[2] + c1;
    let c2 = u32(s2 < c1);
    let s3 = v[3] + c2;
    v[0] = s0; v[1] = s1; v[2] = s2; v[3] = s3;
  }
  RES[idx] = vec4<u32>(v[0], v[1], v[2], v[3]);
}
