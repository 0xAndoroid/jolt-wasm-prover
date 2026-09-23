// V2: staged 16-bit digits (two 16 KB workgroup arrays), pure u32 adds in the inner loop.
// Workgroup = (chunk of CHUNK positions, group of 8 columns, block); 512 threads, thread = coefficient i.
// A2[q] is staged in workgroup memory per position; each thread keeps 8 columns x 8 digit accumulators.
// dispatch: (num_chunks, colcap/8, blocks). Partials per chunk; reduce.wgsl sums chunks and canonicalizes.
override CHUNK: u32 = 128u;
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> A2: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read> HOT: array<vec2<u32>>;   // 8 code bytes per (row, column group)
@group(0) @binding(3) var<storage, read_write> PART: array<vec4<u32>>;
var<workgroup> SLO: array<vec4<u32>, 1024>;
var<workgroup> SHI: array<vec4<u32>, 1024>;
@compute @workgroup_size(512)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
  let i = lid.x;
  let chunk = wid.x;
  let g = wid.y;
  let b = wid.z;
  var lo0 = vec4<u32>(0u);
  var hi0 = vec4<u32>(0u);
  var lo1 = vec4<u32>(0u);
  var hi1 = vec4<u32>(0u);
  var lo2 = vec4<u32>(0u);
  var hi2 = vec4<u32>(0u);
  var lo3 = vec4<u32>(0u);
  var hi3 = vec4<u32>(0u);
  var lo4 = vec4<u32>(0u);
  var hi4 = vec4<u32>(0u);
  var lo5 = vec4<u32>(0u);
  var hi5 = vec4<u32>(0u);
  var lo6 = vec4<u32>(0u);
  var hi6 = vec4<u32>(0u);
  var lo7 = vec4<u32>(0u);
  var hi7 = vec4<u32>(0u);
  let q0 = chunk * CHUNK;
  for (var qq = 0u; qq < CHUNK; qq++) {
    let q = q0 + qq;
    workgroupBarrier();
    for (var k = i; k < 1024u; k += 512u) {
      let x = A2[q * 1024u + k];
      SLO[k] = x & vec4<u32>(0xFFFFu);
      SHI[k] = x >> vec4<u32>(16u);
    }
    workgroupBarrier();
    let row0 = (b * P.positions + q) * 32u;
    for (var r = 0u; r < 32u; r++) {
      let w = HOT[(row0 + r) * P.colcap_vec2 + g];
      let sbase = i - r * 16u;
      {
        let code = (w.x >> 0u) & 0xFFu;
        if (code < 16u) {
          let j = (sbase - code) & 1023u;
          lo0 += SLO[j];
          hi0 += SHI[j];
        }
      }
      {
        let code = (w.x >> 8u) & 0xFFu;
        if (code < 16u) {
          let j = (sbase - code) & 1023u;
          lo1 += SLO[j];
          hi1 += SHI[j];
        }
      }
      {
        let code = (w.x >> 16u) & 0xFFu;
        if (code < 16u) {
          let j = (sbase - code) & 1023u;
          lo2 += SLO[j];
          hi2 += SHI[j];
        }
      }
      {
        let code = (w.x >> 24u) & 0xFFu;
        if (code < 16u) {
          let j = (sbase - code) & 1023u;
          lo3 += SLO[j];
          hi3 += SHI[j];
        }
      }
      {
        let code = (w.y >> 0u) & 0xFFu;
        if (code < 16u) {
          let j = (sbase - code) & 1023u;
          lo4 += SLO[j];
          hi4 += SHI[j];
        }
      }
      {
        let code = (w.y >> 8u) & 0xFFu;
        if (code < 16u) {
          let j = (sbase - code) & 1023u;
          lo5 += SLO[j];
          hi5 += SHI[j];
        }
      }
      {
        let code = (w.y >> 16u) & 0xFFu;
        if (code < 16u) {
          let j = (sbase - code) & 1023u;
          lo6 += SLO[j];
          hi6 += SHI[j];
        }
      }
      {
        let code = (w.y >> 24u) & 0xFFu;
        if (code < 16u) {
          let j = (sbase - code) & 1023u;
          lo7 += SLO[j];
          hi7 += SHI[j];
        }
      }
    }
  }
  let cbase = g * 8u;
  { let o = ((chunk * P.colcap + cbase + 0u) * P.blocks + b) * 512u + i; PART[o * 2u] = lo0; PART[o * 2u + 1u] = hi0; }
  { let o = ((chunk * P.colcap + cbase + 1u) * P.blocks + b) * 512u + i; PART[o * 2u] = lo1; PART[o * 2u + 1u] = hi1; }
  { let o = ((chunk * P.colcap + cbase + 2u) * P.blocks + b) * 512u + i; PART[o * 2u] = lo2; PART[o * 2u + 1u] = hi2; }
  { let o = ((chunk * P.colcap + cbase + 3u) * P.blocks + b) * 512u + i; PART[o * 2u] = lo3; PART[o * 2u + 1u] = hi3; }
  { let o = ((chunk * P.colcap + cbase + 4u) * P.blocks + b) * 512u + i; PART[o * 2u] = lo4; PART[o * 2u + 1u] = hi4; }
  { let o = ((chunk * P.colcap + cbase + 5u) * P.blocks + b) * 512u + i; PART[o * 2u] = lo5; PART[o * 2u + 1u] = hi5; }
  { let o = ((chunk * P.colcap + cbase + 6u) * P.blocks + b) * 512u + i; PART[o * 2u] = lo6; PART[o * 2u + 1u] = hi6; }
  { let o = ((chunk * P.colcap + cbase + 7u) * P.blocks + b) * 512u + i; PART[o * 2u] = lo7; PART[o * 2u + 1u] = hi7; }
}
