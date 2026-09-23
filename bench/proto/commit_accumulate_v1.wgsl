// V1: one thread per output coefficient (c, b, i); loops over all positions and rows,
// gathering A2 straight from global memory. Baseline only.
// dispatch: (512/64, colcap, blocks); writes chunk-0 partials (num_chunks must be 1).
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> A2: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read> HOT: array<u32>;      // code bytes, row stride = colcap bytes
@group(0) @binding(3) var<storage, read_write> PART: array<vec4<u32>>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  let c = gid.y;
  let b = gid.z;
  var lo = vec4<u32>(0u);
  var hi = vec4<u32>(0u);
  let row_u32 = P.colcap / 4u;
  let cword = c / 4u;
  let cshift = (c % 4u) * 8u;
  for (var q = 0u; q < P.positions; q++) {
    let row0 = (b * P.positions + q) * 32u;
    let abase = q * 1024u;
    for (var r = 0u; r < 32u; r++) {
      let code = (HOT[(row0 + r) * row_u32 + cword] >> cshift) & 0xFFu;
      if (code < 16u) {
        let s = r * 16u + code;
        let x = A2[abase + ((i - s) & 1023u)];
        lo += x & vec4<u32>(0xFFFFu);
        hi += x >> vec4<u32>(16u);
      }
    }
  }
  let out = ((c * P.blocks + b) * 512u + i) * 2u;
  PART[out] = lo;
  PART[out + 1u] = hi;
}
