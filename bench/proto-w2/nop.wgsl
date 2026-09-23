// Dependent-round-trip floor: trivial dispatch writing 5 words, then 80 B readback.
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) l: vec3<u32>) {
  if (l.x < 5u) { out[l.x] = vec4<u32>(params.r.x + l.x, params.n_units, 0u, 0u); }
}
