// Scalar preparation for MSM: converts Montgomery-form Fr scalars to
// canonical form and precomputes the signed-digit carry chain for window
// size MSM_C, so digit extraction in the bucket kernel is O(1) per window.
//
// Module requirements: field.wgsl instantiated for Fr, plus consts MSM_C and
// MSM_NW injected. Carry mask convention: bit w = carry INTO window w.

struct PrepParams {
    count: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> prep_params: PrepParams;
@group(0) @binding(1) var<storage, read> mont_scalars: array<u32>;
@group(0) @binding(2) var<storage, read_write> canon_out: array<u32>;
@group(0) @binding(3) var<storage, read_write> masks_out: array<u32>;
@group(0) @binding(4) var<storage, read> prep_signs: array<u32>;

fn raw_window(c: ptr<function, Fe8>, w: u32) -> u32 {
    let bit = w * MSM_C;
    let word_idx = bit >> 5u;
    let shift = bit & 31u;
    var raw = (*c)[word_idx] >> shift;
    if (shift + MSM_C > 32u && word_idx < 7u) {
        raw |= (*c)[word_idx + 1u] << (32u - shift);
    }
    return raw & ((1u << MSM_C) - 1u);
}

@compute @workgroup_size(64)
fn msm_prep(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= prep_params.count) {
        return;
    }
    var w8: Fe8;
    for (var k = 0u; k < 8u; k++) {
        w8[k] = mont_scalars[i * 8u + k];
    }
    let canon = fe_pack(fe_from_mont(fe_unpack(w8)));
    var c: Fe8 = canon;
    for (var k = 0u; k < 8u; k++) {
        canon_out[i * 8u + k] = c[k];
    }

    let half = 1u << (MSM_C - 1u);
    var mask0 = 0u;
    var mask1 = 0u;
    var carry = 0u;
    for (var w = 0u; w < MSM_NW; w++) {
        if (carry == 1u) {
            if (w < 32u) {
                mask0 |= (1u << w);
            } else {
                mask1 |= (1u << (w - 32u));
            }
        }
        let v = raw_window(&c, w) + carry;
        carry = u32(v > half);
    }
    masks_out[i * 2u] = mask0;
    masks_out[i * 2u + 1u] = mask1;
}

// Signed small-scalar prep: the input is already canonical magnitudes (the
// CPU recoded sign and magnitude), so only the carry chain is computed; the
// per-scalar sign lands in bit 31 of the second mask word, where the digit
// decoder XORs it into every digit sign.
@compute @workgroup_size(64)
fn msm_prep_small(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= prep_params.count) {
        return;
    }
    var c: Fe8;
    for (var k = 0u; k < 8u; k++) {
        c[k] = mont_scalars[i * 8u + k];
    }

    let half = 1u << (MSM_C - 1u);
    var mask0 = 0u;
    var mask1 = 0u;
    var carry = 0u;
    for (var w = 0u; w < MSM_NW; w++) {
        if (carry == 1u) {
            if (w < 32u) {
                mask0 |= (1u << w);
            } else {
                mask1 |= (1u << (w - 32u));
            }
        }
        let v = raw_window(&c, w) + carry;
        carry = u32(v > half);
    }
    let sign = (prep_signs[i >> 5u] >> (i & 31u)) & 1u;
    masks_out[i * 2u] = mask0;
    masks_out[i * 2u + 1u] = mask1 | (sign << 31u);
}
