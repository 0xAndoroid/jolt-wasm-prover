// Shared MSM declarations: params, bindings 0-5, digit extraction, affine
// base loading, partial/result IO, small scalar mul, and the window-combine
// kernel. Included by both the fused (msm.wgsl, G1) and split
// (msm_split.wgsl, G2) accumulation variants.
//
// Injected consts: MSM_C, MSM_NB = 2^(MSM_C-1), MSM_NW, MSM_CHUNK.
// Template prefix PF_ -> g1_/g2_.

struct MsmParams {
    rows: u32,
    n: u32,
    n_chunks: u32,
    base_offset: u32,
    scalar_offset: u32,
    scalar_stride: u32,
    out_offset: u32,
    num_windows: u32,
}

const PF_PW: u32 = 3u * PF_FEW;

@group(0) @binding(0) var<uniform> PF_msm_params: MsmParams;
@group(0) @binding(1) var<storage, read> PF_msm_bases: array<u32>;
@group(0) @binding(2) var<storage, read> PF_msm_canon: array<u32>;
@group(0) @binding(3) var<storage, read> PF_msm_masks: array<u32>;
@group(0) @binding(4) var<storage, read_write> PF_msm_partials: array<u32>;
@group(0) @binding(5) var<storage, read_write> PF_msm_results: array<u32>;

var<workgroup> PF_sh_digits: array<u32, MSM_CHUNK>;
var<workgroup> PF_sh_points: array<u32, MSM_NW * PF_PW>;

fn PF_msm_digit(scalar_idx: u32, w: u32) -> u32 {
    let bit = w * MSM_C;
    let word_idx = bit >> 5u;
    let shift = bit & 31u;
    var raw = PF_msm_canon[scalar_idx * 8u + word_idx] >> shift;
    if (shift + MSM_C > 32u && word_idx < 7u) {
        raw |= PF_msm_canon[scalar_idx * 8u + word_idx + 1u] << (32u - shift);
    }
    raw &= (1u << MSM_C) - 1u;
    let mask_word = PF_msm_masks[scalar_idx * 2u + (w >> 5u)];
    let carry = (mask_word >> (w & 31u)) & 1u;
    let v = raw + carry;
    // Per-scalar sign (signed small-scalar path) lives in bit 31 of the
    // second mask word — unreachable as a carry bit (windows <= 37) — and
    // negating a scalar negates every signed digit.
    let scalar_neg = (PF_msm_masks[scalar_idx * 2u + 1u] >> 31u) & 1u;
    let neg = u32(v > MSM_NB) ^ scalar_neg;
    let mag = select(v, (MSM_NB << 1u) - v, v > MSM_NB);
    return (mag << 1u) | neg;
}

fn PF_bases_fe(base: u32) -> PF_Fe {
    var w: array<u32, 16>;
    for (var k = 0u; k < PF_FEW; k++) {
        w[k] = PF_msm_bases[base + k];
    }
    return PF_unpack_words(w);
}

fn PF_partials_fe(base: u32) -> PF_Fe {
    var w: array<u32, 16>;
    for (var k = 0u; k < PF_FEW; k++) {
        w[k] = PF_msm_partials[base + k];
    }
    return PF_unpack_words(w);
}

fn PF_load_base_affine(idx: u32, negate: bool) -> PF_Affine {
    let base = idx * 2u * PF_FEW;
    let x = PF_bases_fe(base);
    var y = PF_bases_fe(base + PF_FEW);
    y = PF_select(negate, PF_neg(y), y);
    return PF_Affine(x, y);
}

fn PF_load_partial(point_idx: u32) -> PF_Point {
    let base = point_idx * 3u * PF_FEW;
    let x = PF_partials_fe(base);
    let y = PF_partials_fe(base + PF_FEW);
    let z = PF_partials_fe(base + 2u * PF_FEW);
    return PF_Point(x, y, z);
}

fn PF_shpoint_load(slot: u32) -> PF_Point {
    var w: array<u32, 16>;
    let base = slot * PF_PW;
    var p: PF_Point;
    for (var k = 0u; k < PF_FEW; k++) {
        w[k] = PF_sh_points[base + k];
    }
    p.x = PF_unpack_words(w);
    for (var k = 0u; k < PF_FEW; k++) {
        w[k] = PF_sh_points[base + PF_FEW + k];
    }
    p.y = PF_unpack_words(w);
    for (var k = 0u; k < PF_FEW; k++) {
        w[k] = PF_sh_points[base + 2u * PF_FEW + k];
    }
    p.z = PF_unpack_words(w);
    return p;
}

fn PF_shpoint_store(slot: u32, p: PF_Point) {
    let base = slot * PF_PW;
    var w = PF_pack_words(p.x);
    for (var k = 0u; k < PF_FEW; k++) {
        PF_sh_points[base + k] = w[k];
    }
    w = PF_pack_words(p.y);
    for (var k = 0u; k < PF_FEW; k++) {
        PF_sh_points[base + PF_FEW + k] = w[k];
    }
    w = PF_pack_words(p.z);
    for (var k = 0u; k < PF_FEW; k++) {
        PF_sh_points[base + 2u * PF_FEW + k] = w[k];
    }
}

fn PF_partial_store(point_idx: u32, p: PF_Point) {
    let base = point_idx * 3u * PF_FEW;
    var w = PF_pack_words(p.x);
    for (var k = 0u; k < PF_FEW; k++) {
        PF_msm_partials[base + k] = w[k];
    }
    w = PF_pack_words(p.y);
    for (var k = 0u; k < PF_FEW; k++) {
        PF_msm_partials[base + PF_FEW + k] = w[k];
    }
    w = PF_pack_words(p.z);
    for (var k = 0u; k < PF_FEW; k++) {
        PF_msm_partials[base + 2u * PF_FEW + k] = w[k];
    }
}

fn PF_result_store(point_idx: u32, p: PF_Point) {
    let base = point_idx * 3u * PF_FEW;
    var w = PF_pack_words(p.x);
    for (var k = 0u; k < PF_FEW; k++) {
        PF_msm_results[base + k] = w[k];
    }
    w = PF_pack_words(p.y);
    for (var k = 0u; k < PF_FEW; k++) {
        PF_msm_results[base + PF_FEW + k] = w[k];
    }
    w = PF_pack_words(p.z);
    for (var k = 0u; k < PF_FEW; k++) {
        PF_msm_results[base + 2u * PF_FEW + k] = w[k];
    }
}

// k in [1, 2^(MSM_C-1)] (up to 128 for c=8): 8-bit double-and-add, branchless.
fn PF_point_mul_small(p: PF_Point, k: u32) -> PF_Point {
    var acc = PF_point_identity();
    for (var bit = 0u; bit < 8u; bit++) {
        acc = PF_point_double(acc);
        let is_set = ((k >> (7u - bit)) & 1u) == 1u;
        acc = PF_point_select(is_set, PF_point_add(acc, p), acc);
    }
    return acc;
}

@compute @workgroup_size(64)
fn msm_combine(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let row = wid.x;
    let tid = lid.x;
    let p = PF_msm_params;

    for (var w = tid; w < p.num_windows; w += 64u) {
        var acc = PF_point_identity();
        for (var ch = 0u; ch < p.n_chunks; ch++) {
            let pidx = (row * p.num_windows + w) * p.n_chunks + ch;
            acc = PF_point_add(acc, PF_load_partial(pidx));
        }
        for (var k = 0u; k < w * MSM_C; k++) {
            acc = PF_point_double(acc);
        }
        PF_shpoint_store(w, acc);
    }
    workgroupBarrier();

    for (var off = 32u; off > 0u; off >>= 1u) {
        if (tid < off && tid + off < p.num_windows) {
            PF_shpoint_store(tid, PF_point_add(PF_shpoint_load(tid), PF_shpoint_load(tid + off)));
        }
        workgroupBarrier();
    }

    if (tid == 0u) {
        PF_result_store(p.out_offset + row, PF_shpoint_load(0u));
    }
}
