// Sum n_units workgroup partials (5 coefficients each) into out[0..5]. One workgroup.
@group(0) @binding(4) var<storage, read_write> out: array<vec4<u32>>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) l: vec3<u32>) {
  let lid = l.x;
  var acc: array<vec4<u32>, 5>;
  for (var c: u32 = 0u; c < 5u; c++) { acc[c] = ZERO4; }
  for (var i: u32 = lid; i < params.n_units; i += WG) {
    for (var c: u32 = 0u; c < 5u; c++) { acc[c] = fp128_add(acc[c], partials[5u * i + c]); }
  }
  for (var c: u32 = 0u; c < 5u; c++) { red[c * WG + lid] = acc[c]; }
  workgroupBarrier();
  for (var s: u32 = WG / 2u; s > 0u; s >>= 1u) {
    if (lid < s) { for (var c: u32 = 0u; c < 5u; c++) { red[c * WG + lid] = fp128_add(red[c * WG + lid], red[c * WG + lid + s]); } }
    workgroupBarrier();
  }
  if (lid < 5u) { out[lid] = red[lid * WG]; }
}
