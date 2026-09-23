// prep: A2[q][j] = A[q][j] for j < 512, p - A[q][j-512] for j >= 512.
// rot(A[q], s)[i] == A2[q][(i - s) mod 1024] for 0 <= s < 512 (negacyclic wrap folded into the index).
@group(0) @binding(0) var<storage, read> A: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read_write> A2: array<vec4<u32>>;

fn neg_p(x: vec4<u32>) -> vec4<u32> {
  // p - x with borrow; x canonical (< p). x == 0 yields p (still 0 mod p, digits still < 2^16).
  let p = vec4<u32>(P0, FULL, FULL, FULL);
  var r: vec4<u32>;
  var b: u32 = 0u;
  for (var k = 0u; k < 4u; k++) {
    let t = p[k] - x[k];
    let b1 = u32(p[k] < x[k]);
    r[k] = t - b;
    let b2 = u32(t < b);
    b = b1 + b2;
  }
  return r;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;                 // q * 1024 + j
  let q = idx >> 10u;
  let j = idx & 1023u;
  if (j < 512u) {
    A2[idx] = A[q * 512u + j];
  } else {
    A2[idx] = neg_p(A[q * 512u + j - 512u]);
  }
}
