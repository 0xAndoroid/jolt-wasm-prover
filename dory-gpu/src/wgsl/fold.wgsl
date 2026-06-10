// Dory vector fold kernels (template PF_ -> g1_/g2_). One thread per vector
// element; the scalar is shared across the whole vector (a transcript
// challenge), so the double-and-add bit pattern is uniform across threads.
//
// Layouts: projective points are 3*PF_FEW packed words, affine 2*PF_FEW.
// The shared scalar arrives canonical (non-Montgomery) as 8 u32 words in a
// small storage buffer.

struct FoldParams {
    n: u32,
    v_offset: u32,
    a_offset: u32,
    out_offset: u32,
}

@group(0) @binding(0) var<uniform> PF_fold_params: FoldParams;
@group(0) @binding(1) var<storage, read> PF_fold_scalar: array<u32>;
@group(0) @binding(2) var<storage, read_write> PF_fold_v: array<u32>;
@group(0) @binding(3) var<storage, read> PF_fold_a: array<u32>;

fn PF_fv_load_fe(base: u32) -> PF_Fe {
    var w: array<u32, 16>;
    for (var k = 0u; k < PF_FEW; k++) {
        w[k] = PF_fold_v[base + k];
    }
    return PF_unpack_words(w);
}

fn PF_fv_store_fe(base: u32, v: PF_Fe) {
    let w = PF_pack_words(v);
    for (var k = 0u; k < PF_FEW; k++) {
        PF_fold_v[base + k] = w[k];
    }
}

fn PF_fa_load_fe(base: u32) -> PF_Fe {
    var w: array<u32, 16>;
    for (var k = 0u; k < PF_FEW; k++) {
        w[k] = PF_fold_a[base + k];
    }
    return PF_unpack_words(w);
}

fn PF_fv_load_point(idx: u32) -> PF_Point {
    let base = idx * 3u * PF_FEW;
    return PF_Point(
        PF_fv_load_fe(base),
        PF_fv_load_fe(base + PF_FEW),
        PF_fv_load_fe(base + 2u * PF_FEW),
    );
}

fn PF_fv_store_point(idx: u32, p: PF_Point) {
    let base = idx * 3u * PF_FEW;
    PF_fv_store_fe(base, p.x);
    PF_fv_store_fe(base + PF_FEW, p.y);
    PF_fv_store_fe(base + 2u * PF_FEW, p.z);
}

fn PF_fa_load_affine(idx: u32) -> PF_Affine {
    let base = idx * 2u * PF_FEW;
    return PF_Affine(PF_fa_load_fe(base), PF_fa_load_fe(base + PF_FEW));
}

fn PF_fold_scalar_bit(b: u32) -> bool {
    let word = PF_fold_scalar[b >> 5u];
    return ((word >> (b & 31u)) & 1u) == 1u;
}

// v[out_offset + i] = k * v[v_offset + i] + v[a_offset + i]  (all projective,
// one buffer: WebGPU forbids aliasing a writable binding with another one)
@compute @workgroup_size(64)
fn fold_scale_add(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let p = PF_fold_params;
    if (i >= p.n) {
        return;
    }
    let base = PF_fv_load_point(p.v_offset + i);
    var acc = PF_point_identity();
    for (var bit = 0u; bit < 256u; bit++) {
        acc = PF_point_double(acc);
        let added = PF_point_add(acc, base);
        acc = PF_point_select(PF_fold_scalar_bit(255u - bit), added, acc);
    }
    acc = PF_point_add(acc, PF_fv_load_point(p.a_offset + i));
    PF_fv_store_point(p.out_offset + i, acc);
}

// v[out_offset + i] = v[v_offset + i] + k * g[a_offset + i]   (g affine, never identity)
@compute @workgroup_size(64)
fn fold_add_scaled_base(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let p = PF_fold_params;
    if (i >= p.n) {
        return;
    }
    let g = PF_fa_load_affine(p.a_offset + i);
    var acc = PF_point_identity();
    for (var bit = 0u; bit < 256u; bit++) {
        acc = PF_point_double(acc);
        let added = PF_point_madd(acc, g);
        acc = PF_point_select(PF_fold_scalar_bit(255u - bit), added, acc);
    }
    acc = PF_point_add(acc, PF_fv_load_point(p.v_offset + i));
    PF_fv_store_point(p.out_offset + i, acc);
}

// v[out_offset + i] = table-based fixed-base scalar mul by prepared scalar
// (a_offset + i). Table: MSM_NW windows x MSM_NB entries of affine points
// (d * 2^(MSM_C*w) * G, d in [1, MSM_NB]); digits use the MSM_C prep buffers.
@group(0) @binding(4) var<storage, read> PF_fb_table: array<u32>;
@group(0) @binding(5) var<storage, read> PF_fb_canon: array<u32>;
@group(0) @binding(6) var<storage, read> PF_fb_masks: array<u32>;

fn PF_fb_digit(scalar_idx: u32, w: u32) -> u32 {
    let bit = w * MSM_C;
    let word_idx = bit >> 5u;
    let shift = bit & 31u;
    var raw = PF_fb_canon[scalar_idx * 8u + word_idx] >> shift;
    if (shift + MSM_C > 32u && word_idx < 7u) {
        raw |= PF_fb_canon[scalar_idx * 8u + word_idx + 1u] << (32u - shift);
    }
    raw &= (1u << MSM_C) - 1u;
    let mask_word = PF_fb_masks[scalar_idx * 2u + (w >> 5u)];
    let carry = (mask_word >> (w & 31u)) & 1u;
    let v = raw + carry;
    let neg = v > MSM_NB;
    let mag = select(v, (MSM_NB << 1u) - v, neg);
    return (mag << 1u) | u32(neg);
}

fn PF_fb_table_affine(w: u32, mag: u32, negate: bool) -> PF_Affine {
    let base = (w * MSM_NB + (mag - 1u)) * 2u * PF_FEW;
    var wx: array<u32, 16>;
    var wy: array<u32, 16>;
    for (var k = 0u; k < PF_FEW; k++) {
        wx[k] = PF_fb_table[base + k];
        wy[k] = PF_fb_table[base + PF_FEW + k];
    }
    let x = PF_unpack_words(wx);
    var y = PF_unpack_words(wy);
    y = PF_select(negate, PF_neg(y), y);
    return PF_Affine(x, y);
}

@compute @workgroup_size(64)
fn fixed_base_mul(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let p = PF_fold_params;
    if (i >= p.n) {
        return;
    }
    var acc = PF_point_identity();
    for (var w = 0u; w < MSM_NW; w++) {
        let d = PF_fb_digit(p.a_offset + i, w);
        let mag = d >> 1u;
        if (mag != 0u) {
            let entry = PF_fb_table_affine(w, mag, (d & 1u) == 1u);
            acc = PF_point_madd(acc, entry);
        }
    }
    PF_fv_store_point(p.out_offset + i, acc);
}
