// Round 1: unit = 4-digit group (one pair of once-folded values), class byte from the digits.
// Workgroup = one block segment of blk = min(inner, 2048) groups sharing e_out; shared 256-class
// histogram of e_in as 8 x 16-bit digit sums (atomics, exact for <= 65536 terms), then thread t
// (= class) reassembles and multiplies by LUT1[t] (5 muls), tree-reduce, x e_out.
var<workgroup> hist: array<atomic<u32>, 2048>; // 256 classes x 8 digits
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) l: vec3<u32>, @builtin(workgroup_id) w: vec3<u32>) {
  let lid = l.x; let wg = w.x;
  for (var k: u32 = lid; k < 2048u; k += WG) { atomicStore(&hist[k], 0u); }
  workgroupBarrier();
  let inner = 1u << params.inner_bits;
  let blk = min(inner, 2048u);
  let base = wg * blk;
  for (var g: u32 = base + lid; g < base + blk; g += WG) {
    let cls = octet_classes(g >> 1u);
    let byte = (cls >> (8u * (g & 1u))) & 0xFFu;
    let e = eq[params.off_first + (g & (inner - 1u))];
    let h = byte * 8u;
    atomicAdd(&hist[h + 0u], e.x & 0xFFFFu); atomicAdd(&hist[h + 1u], e.x >> 16u);
    atomicAdd(&hist[h + 2u], e.y & 0xFFFFu); atomicAdd(&hist[h + 3u], e.y >> 16u);
    atomicAdd(&hist[h + 4u], e.z & 0xFFFFu); atomicAdd(&hist[h + 5u], e.z >> 16u);
    atomicAdd(&hist[h + 6u], e.w & 0xFFFFu); atomicAdd(&hist[h + 7u], e.w >> 16u);
  }
  workgroupBarrier();
  let h = lid * 8u;
  let hv = fp128_from_digits(atomicLoad(&hist[h]), atomicLoad(&hist[h + 1u]), atomicLoad(&hist[h + 2u]), atomicLoad(&hist[h + 3u]),
                             atomicLoad(&hist[h + 4u]), atomicLoad(&hist[h + 5u]), atomicLoad(&hist[h + 6u]), atomicLoad(&hist[h + 7u]));
  let eo = eq[params.off_second + (base >> params.inner_bits)];
  var acc: array<vec4<u32>, 5>;
  for (var c: u32 = 0u; c < 5u; c++) { acc[c] = fp128_mul(fp128_mul(hv, lut[5u * lid + c]), eo); }
  wg_reduce_store(acc, lid, wg);
}
