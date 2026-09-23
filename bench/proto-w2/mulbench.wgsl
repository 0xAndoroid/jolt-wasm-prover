// fp128 mul throughput: each thread runs ppt dependent muls on a pseudo-random pair, writes the result.
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let i = g.x;
  if (i >= params.n_units) { return; }
  var x = vec4<u32>(i * 2654435761u + 1u, i ^ 0x9E3779B9u, i * 40503u + 7u, (i >> 3u) + 0x12345u);
  let b = vec4<u32>(i + 3u, i * 7u + 1u, 0xDEADBEEFu ^ i, (i * 31u) & 0x7FFFFFFFu);
  for (var k: u32 = 0u; k < params.ppt; k++) { x = fp128_mul(x, b); }
  out[i] = x;
}
