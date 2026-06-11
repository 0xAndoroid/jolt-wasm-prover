// One-hot tier-1 row commitments: each output row (k, c) is the sum of the
// setup bases at columns where chunk c selects bucket k. One thread per
// output row; a single mixed-add call site keeps the kernel within the
// Apple Metal size envelope. Bases are setup generators (never identity).
//
// Index layout: indices[c * cols + col] = bucket in [0, K) or ONEHOT_NONE.
// Output row layout matches jolt-core's scatter: row = k * rows_per_k + c.

struct OnehotParams {
    cols: u32,
    rows_per_k: u32,
    num_rows: u32,
    out_offset: u32,
}

@group(0) @binding(0) var<uniform> PF_oh_params: OnehotParams;
@group(0) @binding(1) var<storage, read> PF_oh_indices: array<u32>;
@group(0) @binding(2) var<storage, read> PF_oh_bases: array<u32>;
@group(0) @binding(3) var<storage, read_write> PF_oh_out: array<u32>;

fn PF_oh_load_fe(base: u32) -> PF_Fe {
    var w: array<u32, 16>;
    for (var k = 0u; k < PF_FEW; k++) {
        w[k] = PF_oh_bases[base + k];
    }
    return PF_unpack_words(w);
}

fn PF_oh_load_affine(idx: u32) -> PF_Affine {
    let base = idx * 2u * PF_FEW;
    return PF_Affine(PF_oh_load_fe(base), PF_oh_load_fe(base + PF_FEW));
}

fn PF_oh_store_fe(base: u32, v: PF_Fe) {
    let w = PF_pack_words(v);
    for (var k = 0u; k < PF_FEW; k++) {
        PF_oh_out[base + k] = w[k];
    }
}

@compute @workgroup_size(64)
fn onehot_rows(@builtin(global_invocation_id) gid: vec3<u32>) {
    let r = gid.x;
    let p = PF_oh_params;
    if (r >= p.num_rows) {
        return;
    }
    let k = r / p.rows_per_k;
    let c = r % p.rows_per_k;
    var acc = PF_point_identity();
    for (var col = 0u; col < p.cols; col++) {
        if (PF_oh_indices[c * p.cols + col] == k) {
            acc = PF_point_madd(acc, PF_oh_load_affine(col));
        }
    }
    let base = (p.out_offset + r) * 3u * PF_FEW;
    PF_oh_store_fe(base, acc.x);
    PF_oh_store_fe(base + PF_FEW, acc.y);
    PF_oh_store_fe(base + 2u * PF_FEW, acc.z);
}
