// Projective -> affine normalization via per-thread Fermat inversion.
// The identity (Z = 0) maps to (0, 0) automatically since 0^(p-2) = 0; the
// MSM accumulation kernels skip (0, 0) bases via select.
//
// Template prefix PF_ -> g1_/g2_; PF_inv comes from the instantiation glue.

struct NormalizeParams {
    count: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> PF_norm_params: NormalizeParams;
@group(0) @binding(1) var<storage, read> PF_norm_in: array<u32>;
@group(0) @binding(2) var<storage, read_write> PF_norm_out: array<u32>;

fn PF_norm_load_fe(base: u32) -> PF_Fe {
    var w: array<u32, 16>;
    for (var k = 0u; k < PF_FEW; k++) {
        w[k] = PF_norm_in[base + k];
    }
    return PF_unpack_words(w);
}

fn PF_norm_store_fe(base: u32, v: PF_Fe) {
    let w = PF_pack_words(v);
    for (var k = 0u; k < PF_FEW; k++) {
        PF_norm_out[base + k] = w[k];
    }
}

@compute @workgroup_size(64)
fn normalize_points(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= PF_norm_params.count) {
        return;
    }
    let x = PF_norm_load_fe(i * 3u * PF_FEW);
    let y = PF_norm_load_fe(i * 3u * PF_FEW + PF_FEW);
    let z = PF_norm_load_fe(i * 3u * PF_FEW + 2u * PF_FEW);
    let zinv = PF_inv(z);
    PF_norm_store_fe(i * 2u * PF_FEW, PF_mul(x, zinv));
    PF_norm_store_fe(i * 2u * PF_FEW + PF_FEW, PF_mul(y, zinv));
}
