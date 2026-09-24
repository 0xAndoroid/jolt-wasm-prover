// Sum n_units round-pass partials (6 terms at partials[wg*6]) and inner_bits additional-pass partials (4 terms,
// stored 6-wide at partials[off_first + wg*6]) into out[0..6] and out[6..10]. One workgroup; the totals are read
// from the workgroup-memory reduction tree (`red`), which workgroupBarrier orders — a storage round trip would not be.
@group(0) @binding(4) var<storage, read_write> out: array<vec4<u32>>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) l: vec3<u32>) {
  let lid = l.x;
  var acc: array<vec4<u32>, 6>;
  for (var t: u32 = 0u; t < NT; t++) { acc[t] = ZERO4; }
  for (var i: u32 = lid; i < params.n_units; i += WG) {
    for (var t: u32 = 0u; t < NT; t++) { acc[t] = fp128_add(acc[t], partials[NT * i + t]); }
  }
  wg_reduce_store(acc, lid, 0u);
  workgroupBarrier();
  if (lid < NT) { out[lid] = red[lid * WG]; }
  workgroupBarrier();
  for (var t: u32 = 0u; t < NT; t++) { acc[t] = ZERO4; }
  for (var i: u32 = lid; i < params.inner_bits; i += WG) {
    for (var t: u32 = 0u; t < 4u; t++) { acc[t] = fp128_add(acc[t], partials[params.off_first + NT * i + t]); }
  }
  wg_reduce_store(acc, lid, 0u);
  workgroupBarrier();
  if (lid < 4u) { out[NT + lid] = red[lid * WG]; }
}
